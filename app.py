"""Flask application for the Phishing Risk Detection project."""

from functools import lru_cache
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import pandas as pd
from flask import Flask, abort, render_template, request

from auth import auth_bp, login_required, validate_csrf
from db import close_db, init_db
from templates.methodology.phase3.features import (
    FEATURE_COLUMNS,
    extract_features,
    normalize_url,
)

app = Flask(__name__)

# Secret used to sign session cookies and password-reset tokens.
# In production (Render) it is injected through render.yaml; the fallback is for local development only.
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-insecure-secret-change-me")
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    # Enable SESSION_COOKIE_SECURE=True behind HTTPS (e.g. Render) in production.
)

app.register_blueprint(auth_bp)
app.teardown_appcontext(close_db)
init_db()

BASE_DIR = Path(__file__).resolve().parent
MODEL_OUTPUT_DIR = BASE_DIR / "phase4_output"
MODELS_DIR = MODEL_OUTPUT_DIR / "models"
COMPARISON_PATH = MODEL_OUTPUT_DIR / "model_comparison.csv"
METADATA_PATH = MODEL_OUTPUT_DIR / "modeling_metadata.json"
MAX_URL_LENGTH = 2048

METHODOLOGY_PHASES = (
    {"slug": "business-understanding", "name": "Business Understanding",
     "template": "methodology/business_understanding.html"},
    {"slug": "data-understanding", "name": "Data Understanding",
     "template": "methodology/data_understanding.html"},
    {"slug": "data-preparation", "name": "Data Preparation",
     "template": "methodology/phase3/data_preparation.html"},
    {"slug": "modeling", "name": "Modeling",
     "template": "methodology/modeling.html"},
    {"slug": "evaluation", "name": "Evaluation",
     "template": "methodology/evaluation.html"},
    {"slug": "deployment", "name": "Deployment",
     "template": "methodology/deployment.html"},
)


@app.context_processor
def inject_methodology_phases():
    return {"methodology_phases": METHODOLOGY_PHASES}


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/pgc/", strict_slashes=False)
def pgc():
    return render_template("PGC/pgc.html")


@app.route("/application/context/")
def application_context():
    return render_template("application/context.html")


@app.route("/application/platform/", methods=["GET", "POST"])
@login_required
def application_platform():
    result = None
    error = None
    submitted_url = ""

    if request.method == "POST":
        if not validate_csrf():
            abort(400)
        submitted_url = request.form.get("url", "").strip()
        if not submitted_url:
            error = "Escribe una URL para analizar."
        elif len(submitted_url) > MAX_URL_LENGTH:
            error = f"La URL no puede superar los {MAX_URL_LENGTH} caracteres."
        else:
            try:
                normalized_url = normalize_url(submitted_url)
                parsed = urlsplit(normalized_url)
                if not normalized_url or not parsed.hostname:
                    raise ValueError("URL sin dominio válido.")
                result = predict_url(normalized_url)
            except ValueError as exc:
                error = str(exc) or "La URL no es válida. Revísala e inténtalo de nuevo."
            except (FileNotFoundError, RuntimeError, KeyError, OSError) as exc:
                app.logger.exception("No fue posible ejecutar la predicción: %s", exc)
                error = (
                    "El modelo todavía no está disponible. Ejecuta primero la fase 4 "
                    "y verifica que exista la carpeta phase4_output."
                )

    return render_template(
        "application/platform.html",
        result=result,
        error=error,
        submitted_url=submitted_url,
        model_available=model_artifacts_available(),
    )


def model_artifacts_available():
    """Return whether the files required for inference exist."""
    return (
        COMPARISON_PATH.is_file()
        and METADATA_PATH.is_file()
        and MODELS_DIR.is_dir()
    )


@lru_cache(maxsize=1)
def load_selected_model():
    """Load once the model with the best preliminary cross-validation F1."""
    if not model_artifacts_available():
        raise FileNotFoundError("No se encontraron los artefactos de la fase 4.")

    comparison = pd.read_csv(COMPARISON_PATH)
    if comparison.empty or "cv_f1_mean" not in comparison.columns:
        raise RuntimeError("El archivo de comparación de modelos está vacío o incompleto.")

    best = comparison.sort_values("cv_f1_mean", ascending=False).iloc[0]
    model_name = str(best["model"])
    model_path = MODELS_DIR / f"{model_name}.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"No se encontró el modelo {model_path.name}.")

    with METADATA_PATH.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    feature_columns = metadata.get("features", [])
    if not feature_columns or feature_columns != FEATURE_COLUMNS:
        raise RuntimeError(
            "Las características del modelo no coinciden con el extractor de URLs."
        )

    model = joblib.load(model_path)
    return model, model_name, best.to_dict(), feature_columns


def predict_url(normalized_url):
    """Extract URL features and classify it without retraining the model."""
    model, model_name, metrics, feature_columns = load_selected_model()
    features = extract_features(normalized_url)
    missing = [column for column in feature_columns if column not in features]
    if missing:
        raise RuntimeError(
            "El extractor no generó todas las características requeridas: "
            + ", ".join(missing)
        )

    sample = pd.DataFrame(
        [[features[column] for column in feature_columns]],
        columns=feature_columns,
    )
    prediction = int(model.predict(sample)[0])

    phishing_probability = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(sample)[0]
        classes = list(model.classes_)
        if 1 in classes:
            phishing_probability = float(probabilities[classes.index(1)]) * 100

    error_rate = metrics.get("cv_error_rate_mean")
    if error_rate is not None and pd.notna(error_rate):
        error_rate = max(0.0, min(100.0, float(error_rate) * 100))
    else:
        error_rate = None

    return {
        "normalized_url": normalized_url,
        "is_phishing": prediction,
        "label": "Posible phishing" if prediction == 1 else "Probablemente legítima",
        "level": "danger" if prediction == 1 else "success",
        "model_name": model_name.replace("_", " ").title(),
        "phishing_probability": phishing_probability,
        "cv_error_rate": error_rate,
        "cv_f1": float(metrics["cv_f1_mean"]) * 100,
        "message": (
            "El modelo identificó patrones asociados con URLs de phishing."
            if prediction == 1
            else "El modelo no identificó suficientes patrones para clasificar esta URL como phishing."
        ),
    }


@app.route("/methodology/<phase_slug>/")
def methodology_phase(phase_slug):
    phase_index = next(
        (
            index
            for index, phase in enumerate(METHODOLOGY_PHASES)
            if phase["slug"] == phase_slug
        ),
        None,
    )
    if phase_index is None:
        abort(404)

    return render_template(
        METHODOLOGY_PHASES[phase_index]["template"],
        phase=METHODOLOGY_PHASES[phase_index],
        phase_number=phase_index + 1,
        previous_phase=METHODOLOGY_PHASES[phase_index - 1] if phase_index else None,
        next_phase=(
            METHODOLOGY_PHASES[phase_index + 1]
            if phase_index + 1 < len(METHODOLOGY_PHASES)
            else None
        ),
        total_phases=len(METHODOLOGY_PHASES),
    )


if __name__ == "__main__":
    app.run(debug=True)
