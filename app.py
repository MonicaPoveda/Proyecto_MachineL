from flask import Flask, abort, render_template

app = Flask(__name__)

METHODOLOGY_PHASES = (
    {
        "slug": "business-understanding",
        "name": "Business Understanding",
        "template": "methodology/business_understanding.html",
    },
    {
        "slug": "data-understanding",
        "name": "Data Understanding",
        "template": "methodology/data_understanding.html",
    },
    {
        "slug": "data-preparation",
        "name": "Data Preparation",
        "template": "methodology/phase3/data_preparation.html",
    },
    {
        "slug": "modeling",
        "name": "Modeling",
        "template": "methodology/modeling.html",
    },
    {
        "slug": "evaluation",
        "name": "Evaluation",
        "template": "methodology/evaluation.html",
    },
    {
        "slug": "deployment",
        "name": "Deployment",
        "template": "methodology/deployment.html",
    },
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


@app.route("/application/platform/")
def application_platform():
    return render_template("application/platform.html")


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
    app.run()
