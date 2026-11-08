# real-or-ai-mlops

Détecteur d'images générées par IA (« vraie photo ou image IA ? »), mis en production avec une chaîne MLOps complète : API sécurisée par token, suivi d'usage par utilisateur, versioning MLflow, détection de data drift, monitoring, test de charge, puis Kubernetes, canary et quantization.

- **Modèle** : encodeur d'images CLIP + tête de classification (réel / IA), exporté en ONNX.
- **Données** : [Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage) (35 000 images, 8 générateurs), puis [Synthbuster](https://zenodo.org/records/10066460) (DALL·E 3, Firefly, Midjourney v5, SDXL).
- **Drift** : chaque nouveau générateur produit des images que le modèle n'a jamais vues. Le système doit le détecter, puis on réentraîne et on déploie la nouvelle version en canary.

## Architecture

```
  Utilisateur ──► UI Streamlit ──┐
  Client (curl + token) ─────────┤  header X-API-Key
                                 ▼
                    ┌──────────────────────┐      ┌────────────┐
                    │ API FastAPI          │─────►│ PostgreSQL │ clés, logs d'usage
                    │ + ONNX Runtime       │      └────────────┘
                    └──┬────────────────┬──┘
     charge le modèle  │                │ /metrics
                       ▼                ▼
             ┌─────────────────┐   ┌────────────┐    ┌─────────┐
             │ MLflow          │   │ Prometheus │───►│ Grafana │
             │ (registry)      │   └─────▲──────┘    └─────────┘
             └─────────────────┘         │
                               ┌─────────┴────────┐
                               │ Job de drift     │
                               │ (Evidently)      │
                               └──────────────────┘
```

## Structure du dépôt

| Dossier | Contenu |
|---|---|
| `data/` | Jeux de données en local (non versionnés) |
| `notebooks/` | Exploration des données, entraînement Kaggle |
| `training/` | Entraînement, export ONNX, quantization, enregistrement MLflow |
| `api/` | API FastAPI (prédiction, auth, suivi d'usage) |
| `ui/` | Interface Streamlit |
| `monitoring/` | Prometheus, dashboards Grafana, job de détection de drift |
| `loadtest/` | Scénarios Locust |
| `infra/` | Manifests Kubernetes, Terraform |
| `scripts/` | Utilitaires (téléchargement des données, création de clés API…) |
| `.github/workflows/` | CI/CD |

## Avancement

- [x] Phase 0 — Mise en place
- [x] Phase 1 — Modèle
- [x] Phase 2 — Niveau 0 : tout le système en local avec une commande
- [ ] Phase 3 — Niveau 1 : déploiement automatique sur VM, token, suivi d'usage, test de charge
- [ ] Phase 4 — Niveau 2 : UI, data drift, MLflow
- [ ] Phase 5 — Niveau 3 : quantization, Kubernetes, scaling, canary
- [ ] Phase 6 — Rendu

## Lancer le projet

Prérequis : Docker Desktop, Python 3.13.

```bash
docker compose up -d --build        # lance MLflow, l'API, Prometheus et Grafana
```

| Service | Adresse |
|---|---|
| API (doc interactive) | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana (admin / admin) | http://localhost:3000 |

**Première fois seulement** : le serveur MLflow est vide, il faut entraîner et enregistrer le modèle (l'API le charge automatiquement dès qu'il apparaît) :

```bash
python -m venv .venv && .venv/Scripts/activate      # Linux/Mac : source .venv/bin/activate
pip install -r training/requirements.txt huggingface_hub
python scripts/download_data.py                     # ~8,4 Go, ~10 min
python training/extract_embeddings.py               # ~30 min sur CPU
python training/train.py                            # enregistre la version 1 avec l'alias champion
```

Tester :

```bash
python scripts/export_samples.py                    # images d'exemple dans data/samples/
curl -F "file=@data/samples/Real/real_0000.jpg" http://localhost:8000/predict
# {"label":"real","prob_ai":0.0825,"model_version":"1","latency_ms":83.7}
```
