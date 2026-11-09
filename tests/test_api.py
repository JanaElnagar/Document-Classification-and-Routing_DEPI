"""API test with a tiny model trained on toy data, so it runs in CI without real data."""
import importlib

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


@pytest.fixture()
def client(tmp_path, monkeypatch):
    texts = ["please pay this invoice amount due total tax"] * 5 + ["this agreement between the parties terms"] * 5
    labels = ["Invoice"] * 5 + ["Contract"] * 5
    model = make_pipeline(TfidfVectorizer(), LogisticRegression()).fit(texts, labels)
    joblib.dump(model, tmp_path / "m.joblib")
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "m.joblib"))
    monkeypatch.setenv("PREDICTION_LOG", str(tmp_path / "log.db"))
    import docrouter.api as api
    importlib.reload(api)
    return TestClient(api.app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_classify(client):
    r = client.post("/classify", json={"text": "Invoice total amount due including tax, please pay"})
    assert r.status_code == 200
    body = r.json()
    assert body["category"] in {"Invoice", "Contract"} and "department" in body


def test_rejects_short_text(client):
    assert client.post("/classify", json={"text": "hi"}).status_code == 422


def test_batch_continues_after_bad_item(client):
    docs = [{"text": "this agreement between the parties sets the terms"}, {"text": "XXXX XXXX XXXX XXXX XXXX XXXX"}]
    results = client.post("/classify/batch", json={"documents": docs}).json()["results"]
    assert "category" in results[0] and "error" in results[1]
