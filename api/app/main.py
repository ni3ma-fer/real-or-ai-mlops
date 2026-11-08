"""API « vraie photo ou image IA ? ».

GET  /health   état de l'API et version du modèle
POST /predict  envoie une image (multipart, champ « file »), reçoit la probabilité qu'elle soit générée par IA
GET  /metrics  compteurs pour Prometheus
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from prometheus_client import make_asgi_app
from PIL import UnidentifiedImageError

from shared.preprocessing import preprocess

from . import metrics
from .model import ModelHolder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MAX_BYTES = 10 * 1024 * 1024  # 10 Mo
THRESHOLD = 0.5
holder = ModelHolder()


@asynccontextmanager
async def lifespan(app: FastAPI):
    holder.load_in_background()
    yield


app = FastAPI(title="Real or AI", version="0.1.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.get("/health")
def health():
    model = holder.current
    if model is None:
        raise HTTPException(503, "Modèle en cours de chargement")
    return {"status": "ok", "model_version": model.version}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    start = time.perf_counter()
    status = "200"
    try:
        model = holder.current
        if model is None:
            raise HTTPException(503, "Modèle en cours de chargement")
        data = file.file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "Image trop lourde (10 Mo max)")
        try:
            pixels = preprocess(data)
        except (UnidentifiedImageError, OSError):
            raise HTTPException(400, "Fichier illisible : envoyez une image (JPEG, PNG, WebP...)")

        t0 = time.perf_counter()
        prob_ai, _embedding = model.predict(pixels)
        metrics.INFERENCE_LATENCY.observe(time.perf_counter() - t0)

        label = "ai" if prob_ai >= THRESHOLD else "real"
        metrics.PREDICTIONS.labels(label, model.version).inc()
        metrics.PROB_AI.observe(prob_ai)
        return {
            "label": label,
            "prob_ai": round(prob_ai, 4),
            "model_version": model.version,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        }
    except HTTPException as exc:
        status = str(exc.status_code)
        raise
    finally:
        metrics.REQUESTS.labels("/predict", status).inc()
        metrics.REQUEST_LATENCY.labels("/predict").observe(time.perf_counter() - start)
