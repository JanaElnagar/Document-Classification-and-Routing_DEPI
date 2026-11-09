"""Inference API: text in, category + confidence + department out (FR1-FR7, FR19).

Run:  uvicorn docrouter.api:app --reload --app-dir src
Docs: http://localhost:8000/docs
"""
import os
import sqlite3
from datetime import datetime, timezone

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .preprocess import clean_text
from .routing import Router

MODEL_PATH = os.environ.get("MODEL_PATH", "models/baseline.joblib")
ROUTING_PATH = os.environ.get("ROUTING_PATH", "configs/routing.yaml")
LOG_DB = os.environ.get("PREDICTION_LOG", "predictions.db")
MODEL_VERSION = os.environ.get("MODEL_VERSION", os.path.basename(MODEL_PATH))

app = FastAPI(title="Document Classification & Routing API", version="0.1.0")
_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(503, f"model file not found: {MODEL_PATH}. Train one first.")
        _model = joblib.load(MODEL_PATH)
    return _model


def log_prediction(result: dict):
    """One row per prediction, without the document text (FR6, NFR5)."""
    with sqlite3.connect(LOG_DB) as con:
        con.execute("CREATE TABLE IF NOT EXISTS predictions (ts TEXT, model_version TEXT, "
                    "category TEXT, confidence REAL, department TEXT, needs_review INTEGER)")
        con.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)",
                    (datetime.now(timezone.utc).isoformat(), MODEL_VERSION, result["category"],
                     result["confidence"], result["department"], int(result["needs_review"])))


class Document(BaseModel):
    text: str = Field(..., min_length=20, max_length=50_000, description="Document text")


class Batch(BaseModel):
    documents: list[Document] = Field(..., min_length=1, max_length=100)


def classify_text(text: str) -> dict:
    cleaned = clean_text(text)
    if not cleaned:
        raise HTTPException(422, "document is empty after cleaning")
    model = get_model()
    probs = dict(zip(model.classes_, model.predict_proba([cleaned])[0]))
    result = Router.from_yaml(ROUTING_PATH).route(probs)  # re-read: routing edits apply at once
    result["model_version"] = MODEL_VERSION
    log_prediction(result)
    return result


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH), "model_version": MODEL_VERSION}


@app.post("/classify")
def classify(doc: Document):
    return classify_text(doc.text)


@app.post("/classify/batch")
def classify_batch(batch: Batch):
    results = []
    for i, doc in enumerate(batch.documents):
        try:
            results.append({"index": i, **classify_text(doc.text)})
        except HTTPException as err:  # one bad document does not stop the batch (NFR7)
            results.append({"index": i, "error": err.detail})
    return {"results": results}
