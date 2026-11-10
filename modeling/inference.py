"""Production inference for the phishing classifier."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from modeling.features import FEATURE_COLUMNS, feature_vector, normalize_url


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def artifacts_available() -> bool:
    root = _project_root()
    return (
        (root / "modeling_output" / "winner.joblib").is_file()
        and (root / "modeling_output" / "model_metadata.json").is_file()
    )


@lru_cache(maxsize=1)
def _load_model():
    root = _project_root()
    metadata_path = root / "modeling_output" / "model_metadata.json"
    winner_path = root / "modeling_output" / "winner.joblib"

    if not metadata_path.is_file() or not winner_path.is_file():
        raise FileNotFoundError(
            "Model artifacts not found. Run `python -m modeling.train` first."
        )

    with metadata_path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    if metadata.get("feature_columns") != FEATURE_COLUMNS:
        raise RuntimeError("Stored feature columns do not match the current extractor.")

    model = joblib.load(winner_path)
    return model, metadata


def predict_url(raw_url: str) -> dict:
    """Classify a raw URL and return a template-ready result dictionary."""
    model, metadata = _load_model()

    normalized = normalize_url(raw_url)
    if not normalized:
        raise ValueError("The URL does not contain a valid host.")

    sample = pd.DataFrame([feature_vector(normalized)], columns=FEATURE_COLUMNS)

    prediction = int(model.predict(sample)[0])

    phishing_probability = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(sample)[0]
        classes = list(model.classes_)
        if 1 in classes:
            phishing_probability = float(probabilities[classes.index(1)]) * 100

    metrics = metadata["metrics"]

    return {
        "normalized_url": normalized,
        "is_phishing": prediction,
        "label": "Possible phishing" if prediction == 1 else "Likely legitimate",
        "level": "danger" if prediction == 1 else "success",
        "model_name": str(metadata["model"]).replace("_", " ").title(),
        "phishing_probability": phishing_probability,
        "threshold": float(metadata.get("threshold", 0.5)),
        "test_f1": float(metrics.get("f1", 0.0)) * 100,
        "test_precision": float(metrics.get("precision", 0.0)) * 100,
        "test_recall": float(metrics.get("recall", 0.0)) * 100,
        "test_accuracy": float(metrics.get("accuracy", 0.0)) * 100,
        "message": (
            "The model identified lexical and syntactic patterns associated with phishing URLs."
            if prediction == 1
            else "The model did not identify enough patterns to classify this URL as phishing."
        ),
    }
