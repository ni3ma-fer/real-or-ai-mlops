# real-or-ai-mlops

Détecteur d'images générées par IA (« vraie photo ou image IA ? »), mis en production avec une chaîne MLOps complète : API sécurisée par token, suivi d'usage par utilisateur, versioning MLflow, détection de data drift, monitoring, test de charge, puis Kubernetes, canary et quantization.

- **Modèle** : encodeur d'images CLIP + tête de classification (réel / IA), exporté en ONNX.
- **Données** : [Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage) (35 000 images, 8 générateurs), puis [Synthbuster](https://zenodo.org/records/10066460) (DALL·E 3, Firefly, Midjourney v5, SDXL).
- **Drift** : chaque nouveau générateur produit des images que le modèle n'a jamais vues. Le système doit le détecter, puis on réentraîne et on déploie la nouvelle version en canary.

## Architecture

```
 Utilisateur ─► Interface Streamlit ─┐
 Client (curl + X-API-Key) ──────────┤
                                     ▼
                         ┌───────────────────────┐
                         │ Passerelle nginx      │  répartition + rejeu si panne
                         └───┬───────┬───────┬───┘
                             ▼       ▼       ▼
                     ┌─────────────────────────────┐      ┌────────────┐
                     │ API FastAPI × N copies      │─────►│ PostgreSQL │ clés, usage,
                     │ ONNX Runtime (INT8)         │      └─────▲──────┘ embeddings
                     │ champion + challenger 10 %  │            │
                     └──┬───────────────────┬──────┘            │
   alias MLflow (30 s)  │                   │ /metrics   ┌──────┴───────┐
                        ▼                   ▼            │ Job de drift │
                 ┌────────────┐      ┌────────────┐◄─────│ (domain      │
                 │ MLflow     │◄─────│ Prometheus │      │  classifier) │
                 │ registry   │      └─────┬──────┘      └──────────────┘
                 └────────────┘            ▼  alertes
                                     ┌─────────┐
                                     │ Grafana │
                                     └─────────┘
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
| `scripts/` | Utilitaires (téléchargement des données, images d'exemple) |
| `shared/` | Code commun entraînement / API (préparation des images) |
| `docs/` | Chiffres pour la présentation |
| `.github/workflows/` | CI/CD |

## Avancement

- [x] Phase 0 — Mise en place
- [x] Phase 1 — Modèle
- [x] Phase 2 — Niveau 0 : tout le système en local avec une commande
- [x] Phase 3 — Niveau 1 : déploiement automatique sur VM, token, suivi d'usage, test de charge
- [x] Phase 4 — Niveau 2 : UI, data drift, MLflow (versioning + mise à jour à chaud)
- [x] Phase 5 — Niveau 3 : quantization INT8, scaling horizontal, canary
- [ ] Phase 6 — Rendu

## Lancer le projet

Prérequis : Docker Desktop, Python 3.13.

```bash
docker compose up -d --build        # lance MLflow, l'API, Prometheus et Grafana
```

| Service | Adresse |
|---|---|
| Interface web | http://localhost:8501 |
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
docker compose exec api python -m app.cli create-key alice       # affiche la clé une seule fois
curl -H "X-API-Key: roa_..." -F "file=@data/samples/Real/real_0000.jpg" http://localhost:8000/predict
# {"label":"real","prob_ai":0.0825,"model_version":"1","latency_ms":83.7}
curl -H "X-API-Key: roa_..." http://localhost:8000/usage/me              # sa consommation
```

Gestion des clés : `python -m app.cli create-key <nom> [--admin]`, `list`, `revoke <début de clé>` (dans le conteneur `api`). Une clé admin donne accès à `GET /admin/usage` (usage de chaque utilisateur).

Test de charge (rapport HTML dans `loadtest/reports/`) : créer `loadtest/.keys.env` avec `ALICE_KEY=...` et `BOB_KEY=...`, puis

```bash
docker compose --profile loadtest run --rm locust            # 40 utilisateurs, 3 min
USERS=10 RUN_TIME=1m docker compose --profile loadtest run --rm locust
```

## Production (VM)

```bash
multipass launch 24.04 --name realorai-prod --cpus 2 --memory 4G --disk 25G --cloud-init infra/vm/cloud-init.yaml  # une fois, à la main
scripts/deploy.sh ubuntu@<IP> --ship-images     # déploiement en une commande (idempotent)
scripts/promote_model.sh ubuntu@<IP>            # copie le champion de dev vers la prod
scripts/promote_model.sh ubuntu@<IP> --version 4 --alias challenger   # canary
scripts/set_alias.sh ubuntu@<IP> promote        # challenger -> champion (rechargé à chaud)
scripts/set_alias.sh ubuntu@<IP> rollback       # arrêt du canary
```

Réglages de prod dans `~/real-or-ai.env` sur la VM : `API_REPLICAS=2`, `ORT_THREADS=1`, `CANARY_WEIGHT=0.1`.
CI/CD : `.github/workflows/ci-cd.yml` (tests + images, puis déploiement par un runner installé dans la VM : `scripts/setup_runner.sh`).

## Démo du drift

```bash
python scripts/simulate_traffic.py --scenario normal --n 200   # score ~0,6 : ok
python scripts/simulate_traffic.py --scenario drift  --n 200   # score ~0,8 : DRIFT + alerte Prometheus
python training/train.py --holdout "" --no-alias               # v2 avec tous les générateurs
python training/quantize.py --version 2                        # v2 en INT8 (recalibrée)
```

Tous les chiffres mesurés : [docs/chiffres.md](docs/chiffres.md).
