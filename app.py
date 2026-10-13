"""FastAPI serving app for the Dark Pattern Detector.

The notebook writes model_artifacts/metadata.json with the validation-selected
model and thresholds. This API lazy-loads models: merely having both artifacts
on disk does NOT mean both occupy RAM at startup.
"""

from __future__ import annotations

import html
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

import pickle
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ARTIFACT_DIR = Path(__file__).parent / "model_artifacts"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

if not METADATA_PATH.exists():
    raise RuntimeError("model_artifacts/metadata.json is missing. Run the notebook first.")

with METADATA_PATH.open(encoding="utf-8") as f:
    METADATA = json.load(f)

LABEL_NAMES = METADATA.get("label_names", ["Not Dark Pattern", "Potential Dark Pattern"])
MAX_TOKEN_LEN = int(METADATA.get("max_token_len", 64))
DEFAULT_MODEL = METADATA.get("recommended_model", "classical")
THRESHOLDS = METADATA.get("thresholds", {"classical": 0.5, "bert": 0.5})


def normalize_text(text: str) -> str:
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def load_classical():
    path = ARTIFACT_DIR / "classical_pipeline.pkl"
    if not path.exists():
        raise RuntimeError("classical_pipeline.pkl not found")
    with path.open("rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def load_bert():
    path = ARTIFACT_DIR / "distilbert_dark_pattern"
    if not path.exists():
        raise RuntimeError("distilbert_dark_pattern/ not found")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model


def available_models() -> list[str]:
    models = []
    if (ARTIFACT_DIR / "classical_pipeline.pkl").exists():
        models.append("classical")
    if (ARTIFACT_DIR / "distilbert_dark_pattern").exists():
        models.append("bert")
    return models


def predict_probability(text: str, model_choice: str) -> float:
    """Single-text convenience wrapper around predict_probabilities_batch."""
    return predict_probabilities_batch([text], model_choice)[0]


def predict_probabilities_batch(
    texts: list[str],
    model_choice: str
) -> list[float]:
    """Predict dark-pattern probabilities for a list of text snippets."""

    if not texts:
        return []

    if model_choice == "classical":
        model = load_classical()
        cleaned = [normalize_text(t) for t in texts]
        return model.predict_proba(cleaned)[:, 1].tolist()

    if model_choice == "bert":
        tokenizer, model = load_bert()

        inputs = tokenizer(
            [str(t) for t in texts],
            truncation=True,
            padding=True,
            max_length=MAX_TOKEN_LEN,
            return_tensors="pt",
            return_token_type_ids=False
        )

        # DistilBERT does not accept token_type_ids.
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            logits = model(**inputs).logits

        return torch.softmax(logits, dim=-1)[:, 1].tolist()

    raise ValueError(f"Unknown model: {model_choice}")

def classify(text: str, model_choice: str) -> dict:
    return classify_batch([text], model_choice)[0]


def classify_batch(texts: list[str], model_choice: str) -> list[dict]:
    probabilities = predict_probabilities_batch(texts, model_choice)
    threshold = float(THRESHOLDS.get(model_choice, 0.5))
    results = []
    for dark_probability in probabilities:
        is_dark = dark_probability >= threshold
        label = LABEL_NAMES[1] if is_dark else LABEL_NAMES[0]
        confidence = dark_probability if is_dark else 1.0 - dark_probability
        results.append({
            "label": label,
            "confidence": confidence,
            "dark_pattern_probability": dark_probability,
            "threshold": threshold,
        })
    return results


app = FastAPI(
    title="Dark Pattern Detector API",
    description=(
        "Screens e-commerce UI text for language associated with potential dark patterns. "
        "Text-only predictions do not prove that a claim is false or intentionally manipulative."
    ),
    version="2.1.0",
)

# CORS is wide-open here for local/classroom demos (e.g. a browser extension or the
# Streamlit app calling this API from a different origin). Before any real deployment,
# replace allow_origins=["*"] with the actual origin(s) that should be allowed to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    model: Literal["classical", "bert"] | None = None


class PredictResponse(BaseModel):
    text: str
    label: str
    confidence: float
    dark_pattern_probability: float
    threshold: float
    model_used: str


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)
    model: Literal["classical", "bert"] | None = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "default_model": DEFAULT_MODEL,
        "available_models": available_models(),
        "task": METADATA.get("task", "binary_text_risk_screening"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model_choice = req.model or DEFAULT_MODEL
    if model_choice not in available_models():
        raise HTTPException(status_code=400, detail=f"Model '{model_choice}' is not available in model_artifacts/")
    try:
        result = classify(req.text, model_choice)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return PredictResponse(text=req.text, model_used=model_choice, **result)


@app.post("/predict-batch")
def predict_batch(req: BatchPredictRequest):
    model_choice = req.model or DEFAULT_MODEL
    if model_choice not in available_models():
        raise HTTPException(status_code=400, detail=f"Model '{model_choice}' is not available in model_artifacts/")

    cleaned = [t.strip() for t in req.texts if t and t.strip()]
    if not cleaned:
        raise HTTPException(status_code=400, detail="No non-empty texts supplied")

    classifications = classify_batch(cleaned, model_choice)
    results = [
        {"text": text, "model_used": model_choice, **result}
        for text, result in zip(cleaned, classifications)
    ]

    risky = [r for r in results if r["dark_pattern_probability"] >= r["threshold"]]
    risk_score = round(100 * sum(r["dark_pattern_probability"] for r in results) / len(results), 1)
    return {
        "results": results,
        "flagged_count": len(risky),
        "total_count": len(results),
        "manipulation_risk_score": risk_score,
        "note": "Risk score is a screening summary of model probabilities, not proof of deceptive intent."
    }
