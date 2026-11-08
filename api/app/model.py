"""Chargement des modèles depuis MLflow et prédiction avec ONNX Runtime.

Deux emplacements, désignés par des alias MLflow :
  - champion   : le modèle de production, reçoit le trafic par défaut
  - challenger : (optionnel) le nouveau modèle en canary, reçoit CANARY_WEIGHT du trafic
Un fil d'arrière-plan interroge MLflow toutes les RELOAD_SECONDS : si un alias pointe vers
une nouvelle version, elle est chargée puis échangée à chaud, sans redémarrer l'API.
"""

import logging
import os
import random
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException

from .metrics import MODEL_INFO, MODEL_RELOADS

log = logging.getLogger("api.model")

MODEL_NAME = os.getenv("MODEL_NAME", "real-or-ai")
CHAMPION = os.getenv("MODEL_ALIAS", "champion")
CHALLENGER = os.getenv("CANARY_ALIAS", "challenger")
CANARY_WEIGHT = float(os.getenv("CANARY_WEIGHT", "0"))  # 0.1 = 10 % du trafic vers le challenger
RELOAD_SECONDS = int(os.getenv("RELOAD_SECONDS", "30"))
ORT_THREADS = int(os.getenv("ORT_THREADS", "0"))  # 0 = tous les cœurs


@dataclass
class LoadedModel:
    version: str
    run_id: str
    alias: str
    quantized: bool
    session: ort.InferenceSession

    def predict(self, pixels: np.ndarray) -> tuple[float, np.ndarray]:
        """pixels (3, 224, 224) -> (probabilité IA, embedding de 512 nombres)."""
        prob, emb = self.session.run(None, {"pixel_values": pixels[None].astype(np.float32)})
        return float(prob[0]), emb[0]


def _load(client: MlflowClient, version: str, alias: str, run_id: str, tags: dict) -> LoadedModel:
    log.info("Téléchargement de %s version %s (%s)...", MODEL_NAME, version, alias)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = ORT_THREADS
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(download_artifacts(artifact_uri=f"models:/{MODEL_NAME}/{version}", dst_path=tmp))
        session = ort.InferenceSession(str(next(path.rglob("*.onnx"))), opts, providers=["CPUExecutionProvider"])
    return LoadedModel(version=version, run_id=run_id, alias=alias,
                       quantized=tags.get("quantized") == "int8", session=session)


class ModelHolder:
    def __init__(self) -> None:
        self.models: dict[str, LoadedModel | None] = {CHAMPION: None, CHALLENGER: None}

    @property
    def current(self) -> LoadedModel | None:
        return self.models[CHAMPION]

    def pick(self) -> LoadedModel | None:
        """Canary : une petite part du trafic va au challenger s'il existe."""
        challenger = self.models[CHALLENGER]
        if challenger is not None and CANARY_WEIGHT > 0 and random.random() < CANARY_WEIGHT:
            return challenger
        return self.models[CHAMPION]

    def refresh(self) -> None:
        client = MlflowClient()
        aliases = [CHAMPION] + ([CHALLENGER] if CANARY_WEIGHT > 0 else [])
        for alias in aliases:
            try:
                mv = client.get_model_version_by_alias(MODEL_NAME, alias)
            except MlflowException:
                if self.models[alias] is not None and alias == CHALLENGER:
                    log.info("Alias %s supprimé : fin du canary.", alias)
                    self.models[alias] = None
                continue
            loaded = self.models[alias]
            if loaded is None or loaded.version != str(mv.version):
                old = loaded.version if loaded else None
                self.models[alias] = _load(client, str(mv.version), alias, mv.run_id, mv.tags or {})
                MODEL_RELOADS.labels(alias).inc()
                log.info("%s : version %s -> %s (chargée à chaud)", alias, old, mv.version)
        MODEL_INFO.clear()
        for alias, m in self.models.items():
            if m is not None:
                MODEL_INFO.labels(m.version, alias).set(CANARY_WEIGHT if alias == CHALLENGER else 1 - (
                    CANARY_WEIGHT if self.models[CHALLENGER] else 0))

    def start(self) -> None:
        def loop() -> None:
            while True:
                try:
                    self.refresh()
                except Exception as exc:  # MLflow pas encore prêt, réseau...
                    log.warning("Rafraîchissement des modèles impossible (%s)", exc)
                time.sleep(RELOAD_SECONDS if self.current else 10)

        threading.Thread(target=loop, daemon=True).start()
