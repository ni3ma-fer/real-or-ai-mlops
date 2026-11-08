"""API « vraie photo ou image IA ? ».

GET  /health        état de l'API et version du modèle (public)
POST /predict       image (multipart, champ « file ») -> probabilité IA       [clé API]
GET  /usage/me      ma consommation                                           [clé API]
GET  /admin/usage   consommation de chaque utilisateur                        [clé admin]
GET  /metrics       compteurs pour Prometheus
"""

import logging
import time
from contextlib import asynccontextmanager
from datetime import timedelta

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from PIL import UnidentifiedImageError
from prometheus_client import make_asgi_app
from sqlalchemy import func, select

from shared.preprocessing import preprocess

from . import metrics
from .auth import Caller, require_admin, require_key
from .db import SessionLocal, Usage, User, init_db, now
from .model import ModelHolder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

MAX_BYTES = 10 * 1024 * 1024  # 10 Mo
THRESHOLD = 0.5
holder = ModelHolder()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    holder.load_in_background()
    yield


app = FastAPI(title="Real or AI", version="0.2.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def count_requests(request: Request, call_next):
    """Compte toutes les requêtes (y compris les 401) et mesure leur durée."""
    path = request.url.path
    if path.startswith("/metrics"):
        return await call_next(request)
    start = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    endpoint = route.path if route else "other"
    metrics.REQUESTS.labels(endpoint, str(response.status_code)).inc()
    metrics.REQUEST_LATENCY.labels(endpoint).observe(time.perf_counter() - start)
    return response


def log_usage(caller: Caller, status: int, latency_ms: float, result: dict | None) -> None:
    try:
        with SessionLocal.begin() as s:
            s.add(Usage(user_id=caller.user_id, endpoint="/predict", status=status, latency_ms=latency_ms,
                        label=result and result["label"], prob_ai=result and result["prob_ai"],
                        model_version=result and result["model_version"]))
    except Exception:  # le suivi d'usage ne doit jamais faire échouer une prédiction
        log.exception("Échec de l'enregistrement de l'usage")


@app.get("/health")
def health():
    model = holder.current
    if model is None:
        raise HTTPException(503, "Modèle en cours de chargement")
    return {"status": "ok", "model_version": model.version}


@app.post("/predict")
def predict(file: UploadFile = File(...), caller: Caller = Depends(require_key)):
    start = time.perf_counter()
    status, result = 200, None
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
        result = {"label": label, "prob_ai": round(prob_ai, 4), "model_version": model.version,
                  "latency_ms": round((time.perf_counter() - start) * 1000, 1)}
        return result
    except HTTPException as exc:
        status = exc.status_code
        raise
    finally:
        metrics.USER_REQUESTS.labels(caller.name).inc()
        log_usage(caller, status, (time.perf_counter() - start) * 1000, result)


@app.get("/usage/me")
def my_usage(caller: Caller = Depends(require_key)):
    with SessionLocal() as s:
        total, ok, avg_ms, last = s.execute(
            select(func.count(), func.count().filter(Usage.status == 200), func.avg(Usage.latency_ms), func.max(Usage.ts))
            .where(Usage.user_id == caller.user_id)
        ).one()
        last_24h = s.scalar(select(func.count()).where(Usage.user_id == caller.user_id,
                                                        Usage.ts >= now() - timedelta(hours=24)))
    return {"user": caller.name, "requests_total": total, "requests_ok": ok, "requests_last_24h": last_24h,
            "avg_latency_ms": round(avg_ms or 0, 1), "last_request": last}


@app.get("/admin/usage")
def all_usage(_: Caller = Depends(require_admin)):
    with SessionLocal() as s:
        rows = s.execute(
            select(User.name, func.count(Usage.id), func.count(Usage.id).filter(Usage.label == "ai"),
                   func.avg(Usage.latency_ms), func.max(Usage.ts))
            .outerjoin(Usage, Usage.user_id == User.id).group_by(User.name).order_by(func.count(Usage.id).desc())
        ).all()
    return [{"user": n, "requests_total": t, "predicted_ai": ai, "avg_latency_ms": round(a or 0, 1), "last_request": last}
            for n, t, ai, a, last in rows]
