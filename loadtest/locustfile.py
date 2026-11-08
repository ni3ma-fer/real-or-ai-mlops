"""Test de charge de l'API avec Locust.

Chaque utilisateur virtuel envoie une image d'exemple au hasard (data/samples/), attend
~1 s, recommence ; de temps en temps il consulte sa consommation.

Variables d'environnement : API_KEYS (clés séparées par des virgules, réparties entre les utilisateurs).
Lancement : voir le service « locust » du docker-compose (profil loadtest) ou le README.
"""

import itertools
import os
import random
from pathlib import Path

from locust import HttpUser, between, task

SAMPLES = sorted(p for p in Path(os.getenv("SAMPLES_DIR", "data/samples")).rglob("*") if p.is_file())
IMAGES = [(p.name, p.read_bytes()) for p in SAMPLES]
KEYS = itertools.cycle([k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()])

if not IMAGES:
    raise SystemExit("Aucune image : lancez d'abord python scripts/export_samples.py")


class ApiUser(HttpUser):
    wait_time = between(0.5, 1.5)

    def on_start(self):
        self.client.headers["X-API-Key"] = next(KEYS)

    @task(10)
    def predict(self):
        name, data = random.choice(IMAGES)
        self.client.post("/predict", files={"file": (name, data)}, name="/predict")

    @task(1)
    def usage(self):
        self.client.get("/usage/me", name="/usage/me")
