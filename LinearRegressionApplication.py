
import os
import base64
from io import BytesIO

import pandas as pd
import matplotlib

# Uses a non-interactive backend so Matplotlib can generate images
# without opening a graphical window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Configuration
# Defines the base directory and the location of the CSV dataset.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(
    BASE_DIR,
    "static",
    "datasets",
    "dataset_linear_regression_download_time.csv"
)


# Load dataset
# Reads the CSV file and stores its contents in a Pandas DataFrame.
data = pd.read_csv(DATASET_PATH)


# Dataset validation
# Verifies that the dataset contains the columns required by the model.
required_columns = [ "file_size_mb","download_time_sec"]

for column in required_columns:
    if column not in data.columns:
        raise ValueError(
            f"Required column not found in dataset: {column}"
        )


# Remove incomplete values
# Removes rows where any required variable has a missing value.

data = data.dropna(subset=required_columns).copy()


# Convert values to numeric
# Converts the required columns into numeric values.
# Invalid values are converted to missing values.

data["file_size_mb"] = pd.to_numeric(data["file_size_mb"],errors="coerce")
data["download_time_sec"] = pd.to_numeric(data["download_time_sec"],errors="coerce")


# Remove invalid rows
# Removes rows that became invalid after numeric conversion.
data = data.dropna(subset=required_columns).copy()


# Check minimum number of records
# Ensures that the dataset contains at least 500 valid observations.
num_records = len(data)
if num_records < 500:
    raise ValueError(
        f"The dataset must contain at least 500 records. "
        f"Current records: {num_records}")


# Define variables
# X contains the independent variable used as the model input.
# y contains the dependent variable that the model predicts.

X = data[["file_size_mb"]]
y = data["download_time_sec"]


# Create linear regression model
# Creates the Simple Linear Regression model that will learn
# the relationship between file size and download time.
model = LinearRegression()


# Train model
# Trains the model using the dataset variables X and y.
model.fit(X, y)


# Model parameters
# Extracts the coefficient and intercept learned during training.
# These values form the regression equation.

coefficient = model.coef_[0]
intercept = model.intercept_


# Model predictions for original data
# Generates predicted download times for all original observations
# and stores the results in a new DataFrame column.

data["predicted_download_time"] = model.predict(X)


# Create regression plot
# Generates a scatter plot with the original observations
# and the regression line learned by the model.

def generate_regression_plot():
    figure, axis = plt.subplots(figsize=(10, 6))

    # Original observations
    # Displays each dataset observation as a point.

    axis.scatter(
        data["file_size_mb"],
        data["download_time_sec"],
        alpha=0.65,
        label="Original Data"
    )

    # Regression line
    # Sorts the data by file size so the predicted values
    # can be displayed as a continuous regression line.

    sorted_data = data.sort_values(by="file_size_mb")

    axis.plot(
        sorted_data["file_size_mb"],
        sorted_data["predicted_download_time"],
        linewidth=2,
        label="Regression Line"
    )

    # Plot title
    # Defines the title displayed above the graph.

    axis.set_title(
        "File Size vs. Download Time",
        fontsize=16,
        fontweight="bold"
    )

    # Axis labels
    # Identifies the independent variable on the X axis
    # and the dependent variable on the Y axis.

    axis.set_xlabel(
        "File Size (MB)",
        fontsize=12
    )

    axis.set_ylabel(
        "Download Time (seconds)",
        fontsize=12
    )

    # Grid and legend
    # Adds a visual grid and identifies the elements of the graph.

    axis.grid(alpha=0.2)
    axis.legend()

    # Improve layout
    # Adjusts the spacing so that all graph elements fit correctly.

    figure.tight_layout()

    # Convert plot to Base64
    # Saves the generated graph in memory as a PNG image.
    # Base64 encoding allows the image to be sent to the HTML template
    # without requiring a separate image file.

    image_buffer = BytesIO()

    figure.savefig(
        image_buffer,
        format="png",
        dpi=120,
        bbox_inches="tight"
    )

    plt.close(figure)
    image_buffer.seek(0)

    encoded_image = base64.b64encode(
        image_buffer.getvalue()
    ).decode("utf-8")

    return encoded_image


# Generate graph once when application starts
# Creates the regression graph when this module is loaded.

plot_url = generate_regression_plot()


# Prediction function
# Receives a new file size and uses the trained model
# to calculate the estimated download time.

def predict_download_time(file_size):
    """
    Receives a file size in MB and returns
    the download time predicted by the
    trained Linear Regression model.
    """

    prediction = model.predict(
        [[file_size]]
    )

    return float(prediction[0])


# Dataset preview
# Selects the two model variables and prepares
# the first 10 records for display in the HTML template.

data_preview = data[[ "file_size_mb","download_time_sec"]].head(10).to_dict("records")


# Complete dataset
# Selects the two model variables and converts
# all valid records into a format that can be displayed by the template.

data_full = data[[ "file_size_mb","download_time_sec"]].to_dict("records")


