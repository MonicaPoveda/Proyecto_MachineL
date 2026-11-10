"""Flask application for the Phishing Risk Detection project."""

import os

from flask import Flask, abort, render_template, request

from auth import auth_bp, login_required, validate_csrf
from db import close_db, init_db
from modeling.inference import artifacts_available, predict_url

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
            error = "Enter a URL to analyze."
        elif len(submitted_url) > MAX_URL_LENGTH:
            error = f"The URL cannot exceed {MAX_URL_LENGTH} characters."
        else:
            try:
                result = predict_url(submitted_url)
            except ValueError as exc:
                error = str(exc) or "The URL is not valid. Please review it and try again."
            except (FileNotFoundError, RuntimeError, KeyError, OSError) as exc:
                app.logger.exception("Prediction failed: %s", exc)
                error = (
                    "The model is not available yet. Run `python -m modeling.train` "
                    "first and confirm that modeling_output/ contains the artifacts."
                )

    return render_template(
        "application/platform.html",
        result=result,
        error=error,
        submitted_url=submitted_url,
        model_available=artifacts_available(),
    )


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
