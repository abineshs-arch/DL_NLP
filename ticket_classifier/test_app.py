"""Tests for the ticket classifier API (run with pytest)."""
import pytest
from fastapi.testclient import TestClient

from app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:      # "with" runs the startup code, so models are loaded
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "models_loaded": True}


def test_predict_returns_category_priority_confidence(client):
    r = client.post("/predict", json={"text": "I was charged twice for my subscription, please refund"})
    assert r.status_code == 200
    data = r.json()
    assert set(data) == {"category", "priority", "confidence", "priority_confidence"}
    assert data["priority"] in {"High", "Medium", "Low"}
    assert 0 <= data["confidence"] <= 1


def test_empty_text_returns_422(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_only_spaces_returns_422(client):
    assert client.post("/predict", json={"text": "    "}).status_code == 422


def test_missing_text_field_returns_422(client):
    assert client.post("/predict", json={}).status_code == 422


def test_wrong_type_returns_422(client):
    assert client.post("/predict", json={"text": 123}).status_code == 422