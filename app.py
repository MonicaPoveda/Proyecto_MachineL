"""Flask application for the Phishing Risk Detection project."""

from functools import lru_cache
import json
from pathlib import Path
from urllib.parse import urlsplit

import joblib
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from flask import Flask, abort, render_template, request

from templates.methodology.phase3.features import (
    FEATURE_COLUMNS,
    extract_features,
    normalize_url,
)

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_OUTPUT_DIR = BASE_DIR / "phase4_output"
MODELS_DIR = MODEL_OUTPUT_DIR / "models"
COMPARISON_PATH = MODEL_OUTPUT_DIR / "model_comparison.csv"
METADATA_PATH = MODEL_OUTPUT_DIR / "modeling_metadata.json"
MAX_URL_LENGTH = 2048

FEATURE_REPORT = {
    "url_length": ("Longitud total de la URL", "measure"),
    "host_length": ("Longitud del dominio", "measure"),
    "path_length": ("Longitud de la ruta", "measure"),
    "query_length": ("Longitud de los parámetros", "measure"),
    "n_dots_url": ("Puntos en la URL", "measure"),
    "n_hyphens_url": ("Guiones en la URL", "measure"),
    "n_underscores_url": ("Guiones bajos en la URL", "measure"),
    "n_slashes_url": ("Barras en la URL", "measure"),
    "n_digits_url": ("Dígitos en la URL", "measure"),
    "digit_ratio_url": ("Proporción de dígitos en la URL", "measure"),
    "n_special_chars_url": ("Caracteres especiales en la URL", "measure"),
    "n_subdomains": ("Subdominios", "measure"),
    "host_n_hyphens": ("Guiones en el dominio", "measure"),
    "host_n_digits": ("Dígitos en el dominio", "measure"),
    "host_digit_ratio": ("Proporción de dígitos en el dominio", "measure"),
    "has_ip_host": ("El dominio es una dirección IP", "alert"),
    "has_port": ("Puerto explícito", "measure"),
    "port_is_nonstandard": ("Puerto no estándar", "alert"),
    "path_depth": ("Niveles de profundidad de la ruta", "measure"),
    "n_query_params": ("Parámetros de consulta", "measure"),
    "has_at_symbol": ("Símbolo @ en el dominio", "alert"),
    "has_double_slash_in_path": ("Barras dobles en la ruta", "alert"),
    "has_punycode": ("Codificación Punycode en el dominio", "alert"),
    "is_shortener": ("Dominio de acortador conocido", "alert"),
    "n_suspicious_words": ("Palabras de alerta detectadas", "alert_count"),
    "has_executable_ext": ("Extensión de archivo ejecutable", "alert"),
    "risky_tld": ("Dominio con terminación de riesgo", "alert"),
    "host_entropy": ("Entropía del dominio", "measure"),
    "url_entropy": ("Entropía de la URL", "measure"),
    "longest_host_token": ("Longitud del fragmento más largo del dominio", "measure"),
    "tld_length": ("Longitud de la terminación del dominio", "measure"),
}

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
def application_platform():
    result = None
    error = None
    submitted_url = ""

    if request.method == "POST":
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


def explain_random_forest_paths(model, sample, feature_columns, feature_values):
    """Summarize split conditions followed by trees for each predicted class."""
    classifier = getattr(model, "named_steps", {}).get("model")
    if not isinstance(classifier, RandomForestClassifier):
        return None

    sample_values = sample.to_numpy(dtype=float)
    class_labels = {0: "Legítima", 1: "Phishing"}
    vote_counts = {label: 0 for label in class_labels}
    rule_counts = {label: {} for label in class_labels}

    for tree in classifier.estimators_:
        leaf = int(tree.apply(sample_values)[0])
        leaf_values = tree.tree_.value[leaf, 0]
        predicted_class = int(tree.classes_[leaf_values.argmax()])
        vote_counts[predicted_class] += 1

        visited_nodes = tree.decision_path(sample_values).indices
        path_rules = set()
        for node in visited_nodes:
            feature_index = tree.tree_.feature[node]
            if feature_index < 0:
                continue
            threshold = float(tree.tree_.threshold[node])
            value = float(sample_values[0, feature_index])
            operator = "<=" if value <= threshold else ">"
            path_rules.add((feature_index, operator, threshold))

        for feature_index, operator, threshold in path_rules:
            key = (feature_index, operator, threshold)
            rule_counts[predicted_class][key] = (
                rule_counts[predicted_class].get(key, 0) + 1
            )

    class_summaries = []
    for label, class_name in class_labels.items():
        rules = sorted(
            rule_counts[label].items(),
            key=lambda item: (-item[1], item[0][0]),
        )[:5]
        class_summaries.append(
            {
                "label": class_name,
                "tree_votes": vote_counts[label],
                "rules": [
                    {
                        "name": FEATURE_REPORT[feature_columns[index]][0],
                        "condition": (
                            f"{operator} {threshold:.3f}".rstrip("0").rstrip(".")
                        ),
                        "value": feature_values[feature_columns[index]],
                        "tree_count": count,
                    }
                    for (index, operator, threshold), count in rules
                ],
            }
        )

    return {
        "total_trees": len(classifier.estimators_),
        "classes": class_summaries,
    }


def predict_url(normalized_url):
    """Extract URL features and classify it without retraining the model."""
    model, _, _, feature_columns = load_selected_model()
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

    feature_display_values = {}
    for column in feature_columns:
        _, kind = FEATURE_REPORT[column]
        value = features[column]
        if kind == "measure":
            if column.endswith("_ratio_url") or column.endswith("_ratio"):
                display_value = f"{value:.3f} ({value:.1%})"
            elif isinstance(value, float) and not value.is_integer():
                display_value = f"{value:.3f}"
            else:
                display_value = str(int(value))
        elif kind == "alert_count":
            display_value = str(int(value))
        else:
            display_value = "Sí" if value else "No"
        feature_display_values[column] = display_value
    classification_criteria = explain_random_forest_paths(
        model,
        sample,
        feature_columns,
        feature_display_values,
    )

    return {
        "normalized_url": normalized_url,
        "is_phishing": prediction,
        "label": "Posible phishing" if prediction == 1 else "Probablemente legítima",
        "level": "danger" if prediction == 1 else "success",
        "phishing_probability": phishing_probability,
        "classification_criteria": classification_criteria,
        "message": (
            "Se encontraron patrones asociados con URLs de phishing."
            if prediction == 1
            else "No se encontraron suficientes patrones para clasificar esta URL como phishing."
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
