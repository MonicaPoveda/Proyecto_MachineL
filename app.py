from flask import Flask, render_template, request
import os  
import pandas as pd 
from reinforcement import ACTION_NAMES, GOAL, GRID, START, train
# Logistic Regression Imports
from logistic_Regression import ( 
    classify_message,
    accuracy as log_accuracy,
    precision as log_precision,
    recall as log_recall,
    f1 as log_f1,
    confusion_matrix_result as log_matriz
)

# SVM Imports
from SVM import (
    msv_model,
    scaler,
    accuracy as svm_accuracy,
    precision as svm_precision, 
    recall as svm_recall, 
    f1 as svm_f1,
    conf_matrix, 
    data_plot_url,
    confusion_plot_url,
    num_records as svm_num_records,
    num_train, 
    num_test, 
    interpretation
)

# Linear Regression Gradients Concept Module
import LinearRegressionGraddes

# Linear Regression Application Module
from LinearRegressionApplication import (
    data, 
    data_preview, 
    model as model_lr, 
    num_records as lr_num_records,
    coefficient, 
    intercept, 
    plot_url, 
    predict_download_time
)

# SVM Model Classifier Module (Additional helper if needed)
from ModeloClasif import generateSvmPlotBase64, model as model_svm

app = Flask(__name__)

               
# 1. GENERAL & HOME ROUTES
@app.route("/")
def home():
    return render_template("home.html")


 
# 2. MACHINE LEARNING & USE CASES ROUTES

@app.route("/what-is-ml/")
def what_is_ml():
    return render_template("machine_learning/what_is_ml.html")

@app.route("/types/")
def types():
    return render_template("machine_learning/types.html")

@app.route("/unsupervised/concepts")
def unsupervised_concepts():
    return render_template("unsupervised_learning/concepts.html")

@app.route("/unsupervised/manual-exercise")
def unsupervised_manual():
    base = os.path.join(app.root_path, "static", "datasets")

    df = pd.read_csv(os.path.join(base, "manual_customers.csv"))
    df = df.rename(columns={
        "Annual_Income": "annual_income",
        "Spending_Score": "spending_score",
    })

    iterations = []
    for n in [1, 2, 3]:
        it_df = pd.read_csv(os.path.join(base, f"iteration_{n}_records.csv"))
        iterations.append({
            "n": n,
            "data_preview": it_df.head(10).to_dict(orient="records"),
            "data_full": it_df.to_dict(orient="records"),
        })

    return render_template(
        "unsupervised_learning/manual.html",
        manual_num_records=len(df),
        manual_data_preview=df.head(10).to_dict(orient="records"),
        manual_data_full=df.to_dict(orient="records"),
        iterations=iterations
    )



from kmeans_model import generate_clusters
@app.route("/unsupervised/application")
def unsupervised_application():
    (
        df, centroids, score, plot_url, summary,
        num_records, data_preview, data_full,
        income_min, income_max, score_min, score_max,
        interpretations
    ) = generate_clusters()

    if score >= 0.50:
        silhouette_interpretation = (
            "The clustering structure is well separated: records are generally "
            "close to their assigned cluster and far from other clusters."
        )
    elif score >= 0.25:
        silhouette_interpretation = (
            "The clustering structure is reasonable, although some records "
            "may overlap with neighboring clusters."
        )
    else:
        silhouette_interpretation = (
            "The clustering structure has substantial overlap between groups; "
            "the cluster assignments should therefore be interpreted with caution."
        )

    return render_template(
        "unsupervised_learning/application.html",
        centroids=centroids,
        score=score,
        plot_url=plot_url,
        summary=summary,
        num_records=num_records,
        data_preview=data_preview,
        data_full=data_full,
        income_min=income_min,
        income_max=income_max,
        score_min=score_min,
        score_max=score_max,
        interpretations=interpretations,
        silhouette_interpretation=silhouette_interpretation
    )


@app.route("/unsupervised/evaluation-metrics")
def unsupervised_evalMetrics():
    (
        df, centroids, score, plot_url, summary,
        num_records, data_preview, data_full,
        income_min, income_max, score_min, score_max,
        interpretations
    ) = generate_clusters()

    if score >= 0.50:
        silhouette_interpretation = (
            "The clustering structure is well separated: records are generally "
            "close to their assigned cluster and far from other clusters."
        )
    elif score >= 0.25:
        silhouette_interpretation = (
            "The clustering structure is reasonable, although some records "
            "may overlap with neighboring clusters."
        )
    else:
        silhouette_interpretation = (
            "The clustering structure has substantial overlap between groups; "
            "the cluster assignments should therefore be interpreted with caution."
        )

    return render_template(
        "unsupervised_learning/evalMetrics.html",
        centroids=centroids,
        score=score,
        plot_url=plot_url,
        summary=summary,
        num_records=num_records,
        data_preview=data_preview,
        data_full=data_full,
        income_min=income_min,
        income_max=income_max,
        score_min=score_min,
        score_max=score_max,
        interpretations=interpretations,
        silhouette_interpretation=silhouette_interpretation
    )

@app.route("/use_cases")
def use_cases():
    return render_template("machine_learning/use_cases/use_cases.html")

@app.route('/case1/')
def case1():
    return render_template('machine_learning/use_cases/case1.html')

@app.route('/case2/')
def case2():
    return render_template('machine_learning/use_cases/case2.html')

@app.route('/case3/')
def case3():
    return render_template('machine_learning/use_cases/case3.html')

@app.route('/case4/')
def case4():
    return render_template('machine_learning/use_cases/case4.html')


 
# 3. LINEAR REGRESSION ROUTES
 

@app.route("/linear_regression")
def linear_regression_menu():
    return render_template("linear_regression/linear_regression.html")

@app.route('/linear_regression/conceptsRL/', methods=['GET', 'POST'])
def linear_regression_concepts():
    result = None
    hours = None

    if request.method == 'POST':
        try:
            hours = float(request.form.get('hours'))
            result = LinearRegressionGraddes.calculateGrade(hours)
        except (ValueError, TypeError):
            result = None

    plot_url = LinearRegressionGraddes.generate_plot(
        predicted_hours=hours, predicted_grade=result
    )

    return render_template(
        'linear_regression/conceptsLR.html',
        result=result,
        hours=hours,
        plot_url=plot_url
    )

@app.route("/linear_regression/application", methods=["GET", "POST"])
def linear_regression_application():
    prediction = None
    error = None
    file_size = None

    if request.method == "POST":
        value = request.form.get("file_size", "").strip()

        if not value:
            error = "Please enter a file size."
        else:
            try:
                file_size = float(value)

                if file_size <= 0:
                    error = "File size must be greater than 0."
                else:
                    prediction = predict_download_time(file_size)
                    prediction = round(prediction, 2)

            except ValueError:
                error = "Please enter a valid numeric value."

    # Convert the complete dataset to dictionaries
    data_full = data[["file_size_mb", "download_time_sec"]].to_dict("records")

    return render_template(
        "linear_regression/applicationLR.html",
        num_records=lr_num_records,
        data_preview=data_preview,
        data_full=data_full,
        plot_url=plot_url,
        prediction=prediction,
        file_size=file_size,
        error=error,
        coefficient=coefficient,
        intercept=intercept
    )


 
# 4. LOGISTIC REGRESSION ROUTES

@app.route("/logistic")
@app.route("/logistic_regression")
def logistic_regression_menu():
    return render_template("logistic_regression/logistic_regression.html")

@app.route("/logistic_concepts")
def logistic_concepts():
    return render_template("logistic_regression/conceptsLogR.html")

@app.route("/logistic_regression/application", methods=["GET", "POST"])
def logistic_regression_application():
    result = None
    probability = None

    if request.method == "POST":
        try:
            suspicious_words = int(request.form.get("suspicious_words"))
            number_of_links = int(request.form.get("number_of_links"))
            result, probabilities = classify_message(suspicious_words,number_of_links)
            probability = round(probabilities[1] * 100, 2)

        except (ValueError, TypeError):
            result = "Invalid input values"
            probability = 0.0

    return render_template(
        "logistic_regression/applicationLogR.html",
        result=result,
        probability=probability)


@app.route("/logistic_regression/evaluation-metrics")
def logistic_regression_metrics():
    return render_template(
        "logistic_regression/metricsLogR.html",
        accuracy=log_accuracy,
        precision=log_precision,
        recall=log_recall,
        f1=log_f1,
        confusion_matrix=log_matriz
    )

 
# 5. SUPPORT VECTOR MACHINE (SVM) ROUTES
 
@app.route("/SVM")
def svm_menu():
    return render_template("svm/svm.html")

@app.route("/SVM/concepts")
def svm_concepts():
    return render_template("svm/conceptsSVM.html")

@app.route("/SVM/application", methods=["GET", "POST"])
def svm_application():
    prediction = None
    error = None
    glucose = None
    blood_pressure = None
    bmi = None
    age = None

    if request.method == "POST":
        try:
            # Extract numerical values from form inputs
            glucose = float(request.form.get("glucose", "").strip())
            blood_pressure = float(request.form.get("blood_pressure", "").strip())
            bmi = float(request.form.get("bmi", "").strip())
            age = float(request.form.get("age", "").strip())

            # Validate input boundaries
            if any(value <= 0 for value in [glucose, blood_pressure, bmi, age]):
                error = "All values must be greater than 0."
            else:
                # Format raw input data for prediction
                input_data = [[glucose, blood_pressure, bmi, age]]

                # Standardize input values using the pre-trained scaler from SVM module
                input_data_scaled = scaler.transform(input_data)

                # Execute classification prediction using the trained model
                prediction = msv_model.predict(input_data_scaled)[0]

        except (ValueError, TypeError):
            error = "Please enter valid numerical values."

    return render_template(
        "svm/svmApplication.html",
        prediction=prediction,
        glucose=glucose,
        blood_pressure=blood_pressure,
        bmi=bmi,
        age=age,
        error=error,
        num_records=svm_num_records,
        num_train=num_train,
        num_test=num_test,
        data_plot_url=data_plot_url
    )

@app.route("/SVM/EvaluationMetrics")
def svm_evalMetrics():
    return render_template(
        "svm/evalMetrics.html",
        accuracy=svm_accuracy,
        precision=svm_precision,
        recall=svm_recall,
        f1=svm_f1,
        conf_matrix=conf_matrix,
        confusion_plot_url=confusion_plot_url,
        interpretation=interpretation
    )

from reinforcement import (
    GRID,
    ROWS,
    COLUMNS,
    START,
    GOAL,
    ACTION_NAMES,
    REWARDS,
    MAX_STEPS,
    environment_counts,
    train as train_agent,
)

from reinforcement_teacher import train as train_teacher, GRID as TEACHER_GRID, START as TEACHER_START, GOAL as TEACHER_GOAL, ACTION_NAMES as TEACHER_ACTIONS

@app.route("/reinforcement/teacher-example", methods=["GET", "POST"])
def reinforcement_teacher_example():
    result = None
    if request.method == "POST":
        result = train_teacher(episodes=300)
    return render_template(
        "reinforcement/teacher_example.html",
        result=result,
        grid=TEACHER_GRID,
        start=TEACHER_START,
        goal=TEACHER_GOAL,
        actions=TEACHER_ACTIONS,
    )
 
@app.route("/reinforcement/concepts")
def reinforcement_concepts():
    return render_template("reinforcement/concepts.html")

 
@app.route("/reinforcement/application", methods=["GET", "POST"])
def reinforcement_application():
    result = None
 
    if request.method == "POST":
        result = train_agent(episodes=300)
 
    return render_template(
        "reinforcement/application.html",
        grid=GRID,
        rows=ROWS,
        columns=COLUMNS,
        start=START,
        goal=GOAL,
        actions=ACTION_NAMES,
        rewards=REWARDS,
        env_counts=environment_counts(),
        max_steps=MAX_STEPS,
        result=result,
    )


# APPLICATION ENTRY POINT
if __name__ == '__main__':
    app.run(debug=True)