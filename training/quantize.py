"""Quantization INT8 d'une version du modèle, recalibration, évaluation, puis enregistrement MLflow.

1. Quantization dynamique (ONNX Runtime) des MatMul : poids float32 -> int8 (l'essentiel d'un ViT).
   La convolution d'entrée reste en float32 (ConvInteger non implémenté sur CPU) ; la
   quantization « par canal » a été testée et donne de mauvais résultats sur ce modèle.
2. Recalibration : l'INT8 décale les logits (≈ +1,4 vers « IA » : vraies photos 92 % -> 63 %).
   On mesure ce décalage sur 1 000 images d'ENTRAÎNEMENT et on ajoute avant la sigmoïde finale
   une correction affine  logit' = a * logit + b  (calibration de Platt, 2 paramètres).
3. Évaluation float32 vs int8 sur data/samples (images de validation, jamais utilisées ici).
4. Enregistrement de la version INT8 dans MLflow, tag quantized=int8, sans alias.

    python training/quantize.py --version 2
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import onnx
import onnxruntime as ort
import pyarrow.parquet as pq
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from onnx import helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from shared.preprocessing import preprocess  # noqa: E402

NAME = "real-or-ai"
SAMPLES = ROOT / "data" / "samples"
CALIB_FILE = ROOT / "data" / "raw" / "tiny-genimage" / "data" / "train-00000-of-00014.parquet"


def final_sigmoid(model: onnx.ModelProto) -> onnx.NodeProto:
    # CLIP contient aussi des sigmoïdes internes (QuickGELU) : la sortie est la DERNIÈRE.
    return [n for n in model.graph.node if n.op_type == "Sigmoid"][-1]


def session_with_logit(model: onnx.ModelProto) -> ort.InferenceSession:
    m = onnx.ModelProto()
    m.CopyFrom(model)
    m.graph.output.append(helper.make_tensor_value_info(final_sigmoid(m).input[0], onnx.TensorProto.FLOAT, None))
    return ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])


def add_calibration(model: onnx.ModelProto, a: float, b: float) -> None:
    g = model.graph
    sig = final_sigmoid(model)
    i = list(g.node).index(sig)
    g.initializer.extend([numpy_helper.from_array(np.array(a, np.float32), "calib_a"),
                          numpy_helper.from_array(np.array(b, np.float32), "calib_b")])
    logit = sig.input[0]
    sig.input[0] = "calib_logit"
    g.node.insert(i, helper.make_node("Add", ["calib_mul", "calib_b"], ["calib_logit"], name="calib/Add"))
    g.node.insert(i, helper.make_node("Mul", [logit, "calib_a"], ["calib_mul"], name="calib/Mul"))


def latency_ms(sess: ort.InferenceSession, n: int = 50) -> float:
    x = np.random.default_rng(0).standard_normal((1, 3, 224, 224)).astype(np.float32)
    for _ in range(5):
        sess.run(None, {"pixel_values": x})
    t0 = time.perf_counter()
    for _ in range(n):
        sess.run(None, {"pixel_values": x})
    return (time.perf_counter() - t0) / n * 1000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--version", help="version à quantifier (défaut : champion)")
    p.add_argument("--per-generator", type=int, default=100, help="images d'évaluation par catégorie")
    p.add_argument("--calib", type=int, default=1000, help="images d'entraînement pour la recalibration")
    p.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    args = p.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_registry_uri(args.tracking_uri)
    client = MlflowClient()
    mv = client.get_model_version(NAME, args.version) if args.version else client.get_model_version_by_alias(NAME, "champion")
    src_run = client.get_run(mv.run_id)
    print(f"Source : version {mv.version}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src_dir = Path(download_artifacts(artifact_uri=f"models:/{NAME}/{mv.version}", dst_path=str(tmp / "src")))
        fp32_path = next(src_dir.rglob("*.onnx"))
        size32 = sum(f.stat().st_size for f in fp32_path.parent.iterdir()) / 1e6  # poids en fichier externe
        ref = Path(download_artifacts(run_id=mv.run_id, artifact_path="reference_embeddings.npy", dst_path=str(tmp)))

        # 1. Quantization
        int8_path = tmp / "int8.onnx"
        t0 = time.time()
        quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8,
                         op_types_to_quantize=["MatMul", "Gemm"])
        print(f"Quantization en {time.time() - t0:.0f} s")
        int8 = onnx.load(str(int8_path))

        # 2. Recalibration sur des images d'entraînement
        t = pq.read_table(CALIB_FILE, columns=["image", "label"]).slice(0, args.calib)
        X = np.stack([preprocess(r["bytes"]) for r in t.column("image").to_pylist()])
        y = np.array(t.column("label").to_pylist())
        s_logit = session_with_logit(int8)
        logits = np.concatenate([s_logit.run(None, {"pixel_values": X[i:i + 32]})[2] for i in range(0, len(X), 32)]).ravel()
        lr = LogisticRegression(C=1e6).fit(logits.reshape(-1, 1), y)
        a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        print(f"Recalibration sur {len(y)} images d'entraînement : logit' = {a:.3f} x logit {b:+.3f} "
              f"(accuracy entraînement {((logits >= 0) == y).mean():.3f} -> {(((a * logits + b) >= 0) == y).mean():.3f})")
        add_calibration(int8, a, b)
        onnx.save(int8, str(int8_path))
        size8 = int8_path.stat().st_size / 1e6

        # 3. Évaluation float32 vs int8
        s32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
        s8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
        lat32, lat8 = latency_ms(s32), latency_ms(s8)
        res: dict = {"fp32": {}, "int8": {}}
        agree = n_total = 0
        cos = []
        for gen_dir in sorted(SAMPLES.iterdir()):
            truth = 0 if gen_dir.name == "Real" else 1
            c32 = c8 = 0
            files = sorted(gen_dir.iterdir())[: args.per_generator]
            for f in files:
                px = preprocess(f.read_bytes())[None]
                p32, e32 = s32.run(None, {"pixel_values": px})
                p8, e8 = s8.run(None, {"pixel_values": px})
                y32, y8 = int(p32[0] >= 0.5), int(p8[0] >= 0.5)
                c32 += y32 == truth
                c8 += y8 == truth
                agree += y32 == y8
                cos.append(float(e32[0] @ e8[0]))
            res["fp32"][gen_dir.name] = c32 / len(files)
            res["int8"][gen_dir.name] = c8 / len(files)
            n_total += len(files)
        acc32 = float(np.mean(list(res["fp32"].values())))
        acc8 = float(np.mean(list(res["int8"].values())))

        print(f"Taille   : {size32:.0f} Mo -> {size8:.0f} Mo ({size32 / size8:.1f}x plus petit)")
        print(f"Latence  : {lat32:.0f} ms -> {lat8:.0f} ms par image ({lat32 / lat8:.2f}x plus rapide)")
        print(f"Accuracy moyenne ({n_total} images de validation) : {acc32:.3f} -> {acc8:.3f} ({(acc8 - acc32) * 100:+.1f} pts)")
        print(f"Accord fp32/int8 : {agree / n_total:.1%} ; similarité des embeddings : {np.mean(cos):.4f}")
        for g in res["fp32"]:
            print(f"  {g:<11} {res['fp32'][g]:.2f} -> {res['int8'][g]:.2f}")

        # 4. MLflow
        metrics = {"size_mb_fp32": size32, "size_mb_int8": size8, "latency_ms_fp32": lat32, "latency_ms_int8": lat8,
                   "sample_acc_fp32": acc32, "sample_acc_int8": acc8, "agreement": agree / n_total,
                   "embedding_cosine": float(np.mean(cos)), "calib_a": a, "calib_b": b}
        (tmp / "quantization.json").write_text(json.dumps({"metrics": metrics, "per_generator": res}, indent=2))
        mlflow.set_experiment(NAME)
        with mlflow.start_run(run_name=f"v{mv.version}-int8"):
            mlflow.log_params(src_run.data.params | {"quantization": "dynamic_int8_matmul+platt_calibration",
                                                     "source_version": mv.version, "calib_images": len(y)})
            mlflow.log_metrics(metrics | {"onnx_size_mb": size8, "onnx_latency_ms_batch1": lat8})
            mlflow.log_artifact(str(ref))
            mlflow.log_artifact(str(tmp / "quantization.json"))
            info = mlflow.onnx.log_model(onnx.load(str(int8_path)), name="model", registered_model_name=NAME)
        client.set_model_version_tag(NAME, info.registered_model_version, "quantized", "int8")
        client.set_model_version_tag(NAME, info.registered_model_version, "source_version", mv.version)
        print(f"MLflow : version {info.registered_model_version} (INT8 de la v{mv.version}), tag quantized=int8")


if __name__ == "__main__":
    main()
