# Import libraries
import io
import base64

import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import tools for splitting and standardizing the data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import the Support Vector Machine classifier
from sklearn.svm import SVC

# Import evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score
)

# 1. Load the dataset from the CSV file
df = pd.read_csv("static/datasets/diabetes.csv")

# 2. Define the predictor variables (X) and target variable (y)
x = df[["Glucose", "BloodPressure", "BMI", "Age"]].values
y = df["Outcome"].values

# 3. Split the data into training and testing sets (80-20)
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# 4. Standardize the data
scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.transform(x_test)

# 5 Create the SVM model with a linear kernel
msv_model = SVC(kernel="linear", C=1.0)

# 6 Train the SVM model
msv_model.fit(x_train_scaler, y_train)

#7 Make predictions using the test dataset
y_pred = msv_model.predict(x_test_scaler)

# 8 Create the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# 9 Calculate the classification metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
matrix = confusion_matrix(y_test, y_pred)

#console output 
# print(classification_report(y_test, y_pred))
# print(f"Model accuracy: {accuracy * 100:.2f}%")
# print(f"Model precision: {precision * 100:.2f}%")
# print(f"Model recall: {recall * 100:.2f}%")
# print(f"Model F1-score: {f1 * 100:.2f}%")
# print(f"Confusion matrix:\n{matrix}") 

# 10. Convert a figure to Base64
def figure_to_base64(fig):
    buffer = io.BytesIO()

    fig.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        dpi=100
    )

    buffer.seek(0)

    image_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    plt.close(fig)

    return image_base64

# 11. Create the dataset visualization
# Glucose vs BMI, differentiating classes 0 and 1
fig_data, ax = plt.subplots(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Glucose",
    y="BMI",
    hue="Outcome",
    style="Outcome",
    s=70,
    ax=ax
)

ax.set_title("Patient distribution according to Glucose and BMI")
ax.set_xlabel("Glucose (mg/dL)")
ax.set_ylabel("BMI (kg/m²)")
ax.legend(title="Outcome")

fig_data.tight_layout()

data_plot_url = figure_to_base64(fig_data)

# 12. Create the confusion matrix visualization
fig_matrix, ax = plt.subplots(figsize=(6, 5))

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    ax=ax
)

ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted class")
ax.set_ylabel("Actual class")
ax.set_xticklabels(["Healthy (0)", "Diabetic (1)"])
ax.set_yticklabels(["Healthy (0)", "Diabetic (1)"])

fig_matrix.tight_layout()

confusion_plot_url = figure_to_base64(fig_matrix)

# 13. Store information about the dataset
num_records = len(df)
num_train = len(x_train)
num_test = len(x_test)


# 14. Generate the interpretation of the results

# The metric values are between 0 and 1
# * 100 converts the value to a percentage
# :.2f displays the percentage with two decimal places
# Example: 0.7727 = 77.27%. 

interpretation = (
    f"The SVM model obtained an accuracy of "
    f"{accuracy * 100:.2f}%, which means that it correctly classified "
    f"approximately {accuracy * 100:.2f}% of the patients in the test "
    f"dataset. The precision was {precision * 100:.2f}%, while the "
    f"recall reached {recall * 100:.2f}%. The F1-score was "
    f"{f1 * 100:.2f}%, showing a balance between precision and recall."
)