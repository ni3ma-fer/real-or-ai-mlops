"""Détection de data drift sur les embeddings CLIP des images reçues par l'API.

Méthode « domain classifier » (celle d'Evidently pour les embeddings) : on entraîne un petit
classifieur à distinguer les images de référence (entraînement) des images récentes (production).
  - AUC ~ 0,5 : il n'y arrive pas -> les données ressemblent à l'entraînement, pas de drift
  - AUC proche de 1 : il y arrive -> les données ont changé, drift
Seuil 0,70 calibré sur Tiny-GenImage (fenêtres sans doublons) : trafic normal 0,55-0,64 ;
50 % de générateurs inconnus 0,72-0,76 ; 70 % d'inconnus 0,81-0,83.
Les images identiques sont dédoublonnées : une même image reçue 10 fois n'est pas un changement
de distribution, et le classifieur la « reconnaîtrait » (score gonflé : 0,69 au lieu de 0,60).

Toutes les INTERVAL secondes : lit les WINDOW derniers embeddings en base, calcule le score,
l'expose sur :8001/metrics (Prometheus) et l'enregistre dans la table drift_reports.
"""

import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from prometheus_client import Gauge, start_http_server
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s drift: %(message)s")
log = logging.getLogger("drift")

DATABASE_URL = os.environ["DATABASE_URL"]
MODEL_NAME = os.getenv("MODEL_NAME", "real-or-ai")
WINDOW = int(os.getenv("DRIFT_WINDOW", "200"))
MIN_SAMPLES = int(os.getenv("DRIFT_MIN_SAMPLES", "100"))
THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.70"))
INTERVAL = int(os.getenv("DRIFT_INTERVAL", "60"))

SCORE = Gauge("drift_score", "AUC référence vs récent (0,5 = identique, 1 = totalement différent)")
UNIQUE = Gauge("drift_window_unique", "Images distinctes dans la fenêtre (après dédoublonnage)")
DETECTED = Gauge("drift_detected", "1 si le score dépasse le seuil")
THRESHOLD_G = Gauge("drift_threshold", "Seuil d'alerte")
WINDOW_SIZE = Gauge("drift_window_size", "Nombre d'images récentes analysées")
SHARE_AI = Gauge("drift_window_share_ai", "Part des images récentes prédites « IA »")
LAST_RUN = Gauge("drift_last_run_timestamp_seconds", "Heure du dernier calcul")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)


class Reference:
    """Embeddings de référence du modèle champion (artefact MLflow), rechargés si le champion change."""

    def __init__(self) -> None:
        self.version: str | None = None
        self.X: np.ndarray | None = None

    def refresh(self) -> None:
        mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
        if mv.version == self.version:
            return
        with tempfile.TemporaryDirectory() as tmp:
            path = download_artifacts(run_id=mv.run_id, artifact_path="reference_embeddings.npy", dst_path=tmp)
            self.X = np.load(Path(path))
        self.version = mv.version
        log.info("Référence : %d embeddings du modèle version %s", len(self.X), self.version)


def recent_window() -> tuple[np.ndarray, np.ndarray]:
    with engine.connect() as c:
        rows = c.execute(text("SELECT vector, prob_ai FROM embeddings ORDER BY id DESC LIMIT :n"), {"n": WINDOW}).all()
    WINDOW_SIZE.set(len(rows))
    if not rows:
        return np.empty((0, 512), np.float32), np.empty(0)
    X = np.stack([np.frombuffer(r[0], dtype=np.float32) for r in rows])
    X, idx = np.unique(X, axis=0, return_index=True)  # dédoublonnage des images identiques
    return X, np.array([r[1] for r in rows])[idx]


def drift_score(ref: np.ndarray, cur: np.ndarray) -> float:
    X = np.vstack([ref, cur])
    y = np.r_[np.zeros(len(ref)), np.ones(len(cur))]
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    return float(cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean())


def run_once(ref: Reference) -> None:
    ref.refresh()
    cur, probs = recent_window()
    UNIQUE.set(len(cur))
    LAST_RUN.set(time.time())
    if len(cur) < MIN_SAMPLES:
        log.info("Seulement %d images récentes (< %d) : pas de calcul", len(cur), MIN_SAMPLES)
        return
    score = drift_score(ref.X, cur)
    drift = score >= THRESHOLD
    share_ai = float((probs >= 0.5).mean())
    SCORE.set(score)
    DETECTED.set(int(drift))
    SHARE_AI.set(share_ai)
    with engine.begin() as c:
        c.execute(text("INSERT INTO drift_reports (ts, window_size, score, threshold, drift, share_ai, reference_version) "
                       "VALUES (now(), :n, :s, :t, :d, :a, :v)"),
                  {"n": len(cur), "s": score, "t": THRESHOLD, "d": drift, "a": share_ai, "v": ref.version})
    log.info("%d images, score %.3f (seuil %.2f) -> %s", len(cur), score, THRESHOLD, "DRIFT" if drift else "ok")


def main() -> None:
    start_http_server(8001)
    THRESHOLD_G.set(THRESHOLD)
    ref = Reference()
    while True:
        try:
            run_once(ref)
        except Exception as exc:  # base ou MLflow pas prêts, table pas encore créée...
            log.warning("Calcul impossible : %s", exc)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
