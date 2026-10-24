import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# Load the dataset
data = pd.read_csv("static/datasets/spam_Detection.csv")

# Select the input features
X = data[["Suspicious_Words", "Number_of_Links"]]


# Select the target variable
y = data["Spam"]


# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=42, stratify=y)


# Create the Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions with the test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


# Generate the confusion matrix
confusion_matrix_result = confusion_matrix(y_test, y_pred)

# Function to classify a new message
def classify_message(suspicious_words, number_of_links):
    # Create a DataFrame with the new message
    new_message = pd.DataFrame({
        "Suspicious_Words": [suspicious_words],
        "Number_of_Links": [number_of_links]
    })

    # Predict the class
    prediction = model.predict(new_message)

    # Get the probability for each class
    probability = model.predict_proba(new_message)

    return prediction[0], probability[0]