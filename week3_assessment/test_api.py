"""Tests for the sentiment API."""
import pytest
from fastapi.testclient import TestClient

from api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_positive(client):
    r = client.post("/predict", json={"text": "A wonderful, moving film with great acting"})
    assert r.status_code == 200
    assert r.json()["label"] == "positive"
    assert 0 <= r.json()["confidence"] <= 1


def test_empty_text_returns_422(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_spaces_only_returns_422(client):
    assert client.post("/predict", json={"text": "   "}).status_code == 422


def test_missing_text_returns_422(client):
    assert client.post("/predict", json={}).status_code == 422