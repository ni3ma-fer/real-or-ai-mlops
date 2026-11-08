"""Entraîne le classifieur réel/IA sur les embeddings CLIP, l'exporte en ONNX et l'enregistre dans MLflow.

Usage :
    python training/train.py                 # v1 : Midjourney et Wukong mis de côté
    python training/train.py --holdout ""    # v2 : tous les générateurs (après le drift)

Prérequis : python training/extract_embeddings.py
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import onnx
import onnxruntime as ort
import torch
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import CLIPVisionModelWithProjection

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from shared.preprocessing import IMAGE_SIZE  # noqa: E402

CLIP_NAME = "openai/clip-vit-base-patch32"
EMB = ROOT / "data" / "embeddings"
MODEL_NAME = "real-or-ai"
GENERATORS = {0: "Real", 1: "ADM", 2: "BigGAN", 3: "GLIDE", 4: "Midjourney",
              5: "SD14", 6: "SD15", 7: "VQDM", 8: "Wukong"}
N_REFERENCE = 2000  # embeddings de référence gardés pour la détection de drift


def load(split: str):
    d = np.load(EMB / f"{split}.npz")
    return d["X"], d["label"], d["generator"]


class RealOrAI(torch.nn.Module):
    """CLIP + classifieur dans un seul graphe : image -> (proba IA, embedding)."""

    def __init__(self, clip, coef: np.ndarray, intercept: float):
        super().__init__()
        self.clip = clip
        self.head = torch.nn.Linear(coef.shape[0], 1)
        self.head.weight.data = torch.tensor(coef[None, :], dtype=torch.float32)
        self.head.bias.data = torch.tensor([intercept], dtype=torch.float32)

    def forward(self, pixel_values):
        emb = torch.nn.functional.normalize(self.clip(pixel_values=pixel_values).image_embeds, dim=-1)
        prob_ai = torch.sigmoid(self.head(emb)).squeeze(-1)
        return prob_ai, emb


def evaluate(clf, X, y, g) -> dict:
    """Scores globaux + taux de bonnes réponses pour chaque générateur."""
    p = clf.predict_proba(X)[:, 1]
    pred = (p >= 0.5).astype(int)
    m = {"accuracy": accuracy_score(y, pred), "auc": roc_auc_score(y, p)}
    for gid in np.unique(g):
        mask = g == gid
        m[f"acc_{GENERATORS[int(gid)]}"] = accuracy_score(y[mask], pred[mask])
    return m


def export_onnx(clf, path: Path) -> float:
    """Exporte en ONNX, vérifie qu'il donne les mêmes résultats que PyTorch, mesure la latence."""
    clip = CLIPVisionModelWithProjection.from_pretrained(CLIP_NAME).eval()
    model = RealOrAI(clip, clf.coef_[0], float(clf.intercept_[0])).eval()
    dummy = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(
        model, (dummy,), str(path), input_names=["pixel_values"], output_names=["prob_ai", "embedding"],
        dynamic_axes={"pixel_values": {0: "batch"}, "prob_ai": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=18, dynamo=False,
    )
    onnx.checker.check_model(str(path))

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.inference_mode():
        ref = model(x)[0].numpy()
    out = sess.run(None, {"pixel_values": x.numpy()})[0]
    diff = float(np.abs(ref - out).max())
    assert diff < 1e-4, f"ONNX diffère de PyTorch ({diff})"

    one = x[:1].numpy()
    for _ in range(5):
        sess.run(None, {"pixel_values": one})
    t0 = time.perf_counter()
    for _ in range(50):
        sess.run(None, {"pixel_values": one})
    return (time.perf_counter() - t0) / 50 * 1000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--holdout", default="4,8", help="ids des générateurs exclus de l'entraînement (vide = aucun)")
    p.add_argument("--C", type=float, default=1.0, help="régularisation de la régression logistique")
    p.add_argument("--tracking-uri", default=f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    p.add_argument("--no-alias", action="store_true", help="ne pas promouvoir en champion")
    args = p.parse_args()
    holdout = [int(h) for h in args.holdout.split(",") if h.strip()]

    Xtr, ytr, gtr = load("train")
    Xva, yva, gva = load("validation")
    keep = ~np.isin(gtr, holdout)
    Xtr, ytr, gtr = Xtr[keep], ytr[keep], gtr[keep]
    seen = ~np.isin(gva, holdout)
    print(f"Entraînement : {len(Xtr)} images ({int((ytr == 0).sum())} réelles, {int((ytr == 1).sum())} IA)")
    print(f"Générateurs mis de côté : {[GENERATORS[h] for h in holdout] or 'aucun'}")

    t0 = time.time()
    clf = LogisticRegression(C=args.C, max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    train_s = time.time() - t0

    m_seen = evaluate(clf, Xva[seen], yva[seen], gva[seen])
    m_all = evaluate(clf, Xva, yva, gva)
    print(f"Entraîné en {train_s:.1f} s")
    print(f"Validation, générateurs connus : accuracy {m_seen['accuracy']:.3f}, AUC {m_seen['auc']:.3f}")
    for gid, name in GENERATORS.items():
        if f"acc_{name}" in m_all:
            tag = " (jamais vu)" if gid in holdout else ""
            print(f"  {name:<11} {m_all[f'acc_{name}']:.3f}{tag}")

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(MODEL_NAME)
    with mlflow.start_run(run_name="v-holdout" if holdout else "v-all") as run, tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mlflow.log_params({"clip": CLIP_NAME, "head": "logistic_regression", "C": args.C,
                           "holdout": ",".join(GENERATORS[h] for h in holdout) or "none",
                           "n_train": len(Xtr), "jpeg_reencode_quality": 90})
        mlflow.log_metrics({f"val_seen_{k}": v for k, v in m_seen.items()} | {f"val_all_{k}": v for k, v in m_all.items()}
                           | {"train_seconds": train_s})

        onnx_path = tmp / "model.onnx"
        latency_ms = export_onnx(clf, onnx_path)
        size_mb = onnx_path.stat().st_size / 1e6
        mlflow.log_metrics({"onnx_latency_ms_batch1": latency_ms, "onnx_size_mb": size_mb})
        print(f"ONNX : {size_mb:.0f} Mo, {latency_ms:.0f} ms par image (CPU)")

        # Référence pour la détection de drift : à quoi ressemblent les images « normales ».
        rng = np.random.default_rng(0)
        ref = Xtr[rng.choice(len(Xtr), size=min(N_REFERENCE, len(Xtr)), replace=False)]
        np.save(tmp / "reference_embeddings.npy", ref)
        mlflow.log_artifact(str(tmp / "reference_embeddings.npy"))
        (tmp / "metrics.json").write_text(json.dumps({"seen": m_seen, "all": m_all}, indent=2))
        mlflow.log_artifact(str(tmp / "metrics.json"))

        info = mlflow.onnx.log_model(onnx.load(str(onnx_path)), name="model", registered_model_name=MODEL_NAME)
        version = info.registered_model_version
        if not args.no_alias:
            MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", version)
        print(f"MLflow : run {run.info.run_id}, {MODEL_NAME} version {version}"
              + ("" if args.no_alias else " -> alias 'champion'"))


if __name__ == "__main__":
    main()
