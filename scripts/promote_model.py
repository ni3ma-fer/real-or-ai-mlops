"""Copie une version de modèle d'un serveur MLflow à un autre (dev -> production).

Le modèle n'est pas réentraîné : on recopie exactement le fichier ONNX validé, avec ses
paramètres, ses métriques et les embeddings de référence (pour le drift), puis on lui
donne l'alias « champion » en production. L'API de production le charge d'elle-même.

    python scripts/promote_model.py --target http://localhost:5001
    python scripts/promote_model.py --target http://localhost:5001 --version 2 --alias challenger

En pratique, utiliser scripts/promote_model.sh qui ouvre le tunnel SSH vers la VM.
"""

import argparse
import tempfile
from pathlib import Path

import mlflow
import onnx
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts

NAME = "real-or-ai"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="http://localhost:5000", help="MLflow de dev")
    p.add_argument("--target", required=True, help="MLflow de production (via tunnel SSH)")
    p.add_argument("--version", help="version à promouvoir (défaut : le champion de dev)")
    p.add_argument("--alias", default="champion", help="alias à donner en production")
    args = p.parse_args()

    src = MlflowClient(args.source)
    mv = src.get_model_version(NAME, args.version) if args.version else src.get_model_version_by_alias(NAME, "champion")
    run = src.get_run(mv.run_id)
    print(f"Source : {NAME} version {mv.version} (run {mv.run_id}) sur {args.source}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mlflow.set_registry_uri(args.source)  # sinon MLflow cherche le registre dans une base locale par défaut
        model_dir = Path(download_artifacts(artifact_uri=f"models:/{NAME}/{mv.version}",
                                            dst_path=str(tmp / "model"), tracking_uri=args.source))
        onnx_file = next(model_dir.rglob("*.onnx"))
        available = {a.path for a in MlflowClient(args.source).list_artifacts(mv.run_id)}
        extras = [Path(download_artifacts(run_id=mv.run_id, artifact_path=a, dst_path=str(tmp), tracking_uri=args.source))
                  for a in ("reference_embeddings.npy", "metrics.json", "quantization.json") if a in available]

        mlflow.set_tracking_uri(args.target)
        mlflow.set_registry_uri(args.target)
        mlflow.set_experiment(NAME)
        with mlflow.start_run(run_name=f"promoted-from-dev-v{mv.version}"):
            mlflow.log_params(run.data.params)
            mlflow.log_metrics(run.data.metrics)
            mlflow.set_tags({"promoted_from": args.source, "source_version": mv.version, "source_run_id": mv.run_id})
            for f in extras:
                mlflow.log_artifact(str(f))
            info = mlflow.onnx.log_model(onnx.load(str(onnx_file)), name="model", registered_model_name=NAME)
        target = MlflowClient(args.target)
        for k, v in (mv.tags or {}).items():  # ex. quantized=int8
            target.set_model_version_tag(NAME, info.registered_model_version, k, v)
        target.set_registered_model_alias(NAME, args.alias, info.registered_model_version)
    print(f"Production : {NAME} version {info.registered_model_version} -> alias « {args.alias} » sur {args.target}")


if __name__ == "__main__":
    main()
