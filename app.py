"""Flask application for the Phishing Risk Detection project."""

from functools import lru_cache
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import joblib
from sklearn.ensemble import RandomForestClassifier
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
MODEL_OUTPUT_DIR = BASE_DIR / "outputs" / "phase4_output"
MODELS_DIR = MODEL_OUTPUT_DIR / "models"
COMPARISON_PATH = MODEL_OUTPUT_DIR / "model_comparison.csv"
METADATA_PATH = MODEL_OUTPUT_DIR / "modeling_metadata.json"
MAX_URL_LENGTH = 2048

FEATURE_REPORT = {
    "url_length": ("Total URL length", "measure"),
    "host_length": ("Hostname length", "measure"),
    "path_length": ("Path length", "measure"),
    "query_length": ("Query length", "measure"),
    "n_dots_url": ("Dots in URL", "measure"),
    "n_hyphens_url": ("Hyphens in URL", "measure"),
    "n_underscores_url": ("Underscores in URL", "measure"),
    "n_slashes_url": ("Slashes in URL", "measure"),
    "n_digits_url": ("Digits in URL", "measure"),
    "digit_ratio_url": ("Digit ratio in URL", "measure"),
    "n_special_chars_url": ("Special characters in URL", "measure"),
    "n_subdomains": ("Subdomain count", "measure"),
    "host_n_hyphens": ("Hyphens in hostname", "measure"),
    "host_n_digits": ("Digits in hostname", "measure"),
    "host_digit_ratio": ("Digit ratio in hostname", "measure"),
    "has_ip_host": ("Hostname is an IP address", "alert"),
    "has_port": ("Explicit port", "measure"),
    "port_is_nonstandard": ("Non-standard port", "alert"),
    "path_depth": ("Path depth", "measure"),
    "n_query_params": ("Query parameter count", "measure"),
    "has_at_symbol": ("At sign in hostname", "alert"),
    "has_double_slash_in_path": ("Double slash in path", "alert"),
    "has_punycode": ("Punycode hostname", "alert"),
    "is_shortener": ("Known URL shortener", "alert"),
    "n_suspicious_words": ("Suspicious words detected", "alert_count"),
    "has_executable_ext": ("Executable file extension", "alert"),
    "risky_tld": ("Risk-associated top-level domain", "alert"),
    "host_entropy": ("Hostname entropy", "measure"),
    "url_entropy": ("URL entropy", "measure"),
    "longest_host_token": ("Longest hostname token", "measure"),
    "tld_length": ("Top-level domain length", "measure"),
}

METHODOLOGY_PHASES = (
    {"slug": "business-understanding", "name": "Business Understanding",
     "template": "methodology/phase 1/business_understanding.html"},
    {"slug": "data-understanding", "name": "Data Understanding",
     "template": "methodology/phase 2/data_understanding.html"},
    {"slug": "data-preparation", "name": "Data Preparation",
     "template": "methodology/phase3/data_preparation.html"},
    {"slug": "modeling", "name": "Modeling",
     "template": "methodology/phase4/modeling.html"},
    {"slug": "evaluation", "name": "Evaluation",
     "template": "methodology/phase 5/evaluation.html"},
    {"slug": "deployment", "name": "Deployment",
     "template": "methodology/phase 6/deployment.html"},
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
            error = "Enter a URL to analyze."
        elif len(submitted_url) > MAX_URL_LENGTH:
            error = f"The URL cannot exceed {MAX_URL_LENGTH} characters."
        else:
            try:
                normalized_url = normalize_url(submitted_url)
                parsed = urlsplit(normalized_url)
                if not normalized_url or not parsed.hostname:
                    raise ValueError("The URL does not contain a valid hostname.")
                result = predict_url(normalized_url)
            except ValueError as exc:
                error = str(exc) or "The URL is invalid. Please check it and try again."
            except (FileNotFoundError, RuntimeError, KeyError, OSError) as exc:
                app.logger.exception("Prediction failed: %s", exc)
                error = (
                    "The model is not available yet. Run the Phase 4 training script "
                    "and verify that outputs/phase4_output contains its artifacts."
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
        raise FileNotFoundError("Phase 4 model artifacts were not found.")

    comparison = pd.read_csv(COMPARISON_PATH)
    if comparison.empty or "cv_f1_mean" not in comparison.columns:
        raise RuntimeError("The model comparison file is empty or incomplete.")

    best = comparison.sort_values("cv_f1_mean", ascending=False).iloc[0]
    model_name = str(best["model"])
    model_path = MODELS_DIR / f"{model_name}.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file {model_path.name} was not found.")

    with METADATA_PATH.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    feature_columns = metadata.get("features", [])
    if not feature_columns or feature_columns != FEATURE_COLUMNS:
        raise RuntimeError(
            "The model features do not match the URL feature extractor."
        )

    model = joblib.load(model_path)
    return model, model_name, best.to_dict(), feature_columns


def explain_random_forest_paths(model, sample, feature_columns, feature_values):
    """Summarize split conditions followed by trees for each predicted class."""
    classifier = getattr(model, "named_steps", {}).get("model")
    if not isinstance(classifier, RandomForestClassifier):
        return None

    sample_values = sample.to_numpy(dtype=float)
    class_labels = {0: "Legitimate", 1: "Phishing"}
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
            "The URL feature extractor did not generate all required features: "
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
            display_value = "Yes" if value else "No"
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
        "label": "Possible phishing" if prediction == 1 else "Likely legitimate",
        "level": "danger" if prediction == 1 else "success",
        "phishing_probability": phishing_probability,
        "classification_criteria": classification_criteria,
        "message": (
            "The URL has patterns commonly associated with phishing."
            if prediction == 1
            else "The URL does not have enough patterns to classify it as phishing."
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
