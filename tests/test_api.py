"""Tests de l'API sans base de données ni modèle (le lifespan n'est pas lancé)."""

from fastapi.testclient import TestClient

from app.auth import KEY_PREFIX, generate_key, hash_key
from app.main import app

client = TestClient(app)


def test_keys_are_random_and_prefixed():
    a, b = generate_key(), generate_key()
    assert a != b
    assert a.startswith(KEY_PREFIX) and len(a) > 40


def test_hash_is_sha256_and_stable():
    k = generate_key()
    assert hash_key(k) == hash_key(k)
    assert len(hash_key(k)) == 64
    assert k not in hash_key(k)


def test_predict_without_key_is_refused():
    r = client.post("/predict", files={"file": ("x.jpg", b"...")})
    assert r.status_code == 401


def test_admin_without_key_is_refused():
    assert client.get("/admin/usage").status_code == 401


def test_health_is_503_while_no_model():
    assert client.get("/health").status_code == 503


def test_metrics_are_public():
    r = client.get("/metrics/")
    assert r.status_code == 200
    assert "api_requests_total" in r.text
