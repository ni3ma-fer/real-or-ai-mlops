"""Chargement du modèle « champion » depuis MLflow et prédiction avec ONNX Runtime."""

import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts

from .metrics import MODEL_INFO

log = logging.getLogger("api.model")

MODEL_NAME = os.getenv("MODEL_NAME", "real-or-ai")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
RETRY_SECONDS = 10


@dataclass
class LoadedModel:
    version: str
    run_id: str
    session: ort.InferenceSession

    def predict(self, pixels: np.ndarray) -> tuple[float, np.ndarray]:
        """pixels (3, 224, 224) -> (probabilité IA, embedding de 512 nombres)."""
        prob, emb = self.session.run(None, {"pixel_values": pixels[None].astype(np.float32)})
        return float(prob[0]), emb[0]


class ModelHolder:
    """Garde le modèle en mémoire. Le chargement tourne en arrière-plan et réessaie
    tant que MLflow n'est pas prêt ou qu'aucun modèle n'a l'alias champion."""

    def __init__(self) -> None:
        self.current: LoadedModel | None = None
        self._lock = threading.Lock()

    def load(self) -> LoadedModel:
        client = MlflowClient()
        mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        log.info("Téléchargement de %s version %s (%s)...", MODEL_NAME, mv.version, MODEL_ALIAS)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(download_artifacts(artifact_uri=f"models:/{MODEL_NAME}/{mv.version}", dst_path=tmp))
            onnx_file = next(path.rglob("*.onnx"))
            session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
        model = LoadedModel(version=str(mv.version), run_id=mv.run_id, session=session)
        with self._lock:
            self.current = model
        MODEL_INFO.clear()
        MODEL_INFO.labels(model.version).set(1)
        log.info("Modèle version %s prêt.", model.version)
        return model

    def load_in_background(self) -> None:
        def loop() -> None:
            while self.current is None:
                try:
                    self.load()
                except Exception as exc:  # MLflow pas encore démarré, pas de champion...
                    log.warning("Modèle pas encore disponible (%s), nouvel essai dans %ss", exc, RETRY_SECONDS)
                    time.sleep(RETRY_SECONDS)

        threading.Thread(target=loop, daemon=True).start()
