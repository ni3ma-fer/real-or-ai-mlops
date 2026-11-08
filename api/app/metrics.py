"""Compteurs exposés sur /metrics et relevés par Prometheus."""

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("api_requests_total", "Requêtes reçues", ["endpoint", "status"])
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds", "Durée totale d'une requête", ["endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 2, 5),
)
INFERENCE_LATENCY = Histogram(
    "model_inference_seconds", "Durée du passage dans le modèle ONNX",
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1),
)
PREDICTIONS = Counter("predictions_total", "Prédictions par classe", ["label", "model_version"])
PROB_AI = Histogram(
    "prediction_prob_ai", "Probabilité « IA » renvoyée",
    buckets=tuple(i / 10 for i in range(1, 11)),
)
MODEL_INFO = Gauge("model_info", "Version du modèle chargé (1 = actif)", ["model_version"])
