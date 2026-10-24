import io
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from sklearn.svm import SVC  # Import the Support Vector Classifier


# 1. Training Data (Study Hours, Attendance %)
data = {
    "Study_Hours": [10, 15, 3, 12, 8, 14, 5, 18, 7, 11, 2, 16, 9, 4, 13, 6, 17, 1, 20, 5],
    "Attendance":  [85, 90, 40, 80, 70, 95, 50, 98, 60, 75, 30, 92, 65, 45, 88, 55, 89, 20, 95, 60],
    "Status":      [1,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  0,  1,  0,  1,  0,  1,  0] 
    # 1 = Approved (Blue), 0 = Failed (Red)
}
df = pd.DataFrame(data)
# Predictor variables (X) and target variable (y)
X = df[["Study_Hours", "Attendance"]].values
y = df["Status"].values


# 2. Train the SVM Model with a Linear Kernel
# Using a soft margin (C=1.0) to allow optimal classifications
model = SVC(kernel='linear', C=1.0)
model.fit(X, y)



# 3. Form Function to Classify a New Student
def classifyStudent(hours, attendance):
    """Predicts whether the student passes (1) or fails (0)."""
    prediction = model.predict([[hours, attendance]])[0]
    # Return friendly text and the numerical code
    result_text = "APPROVED" if prediction == 1 else "FAILED"
    return result_text, int(prediction)


# 4. Function to Generate the Plot with Hyperplane and Return in Base64
def generateSvmPlotBase64(user_hours=None, user_attendance=None, user_pred=None):
    """Generates an SVM decision boundary plot and returns it as a Base64 encoded string."""
    plt.figure(figsize=(9, 6))
    
    # Plot existing training data
    approved = df[df["Status"] == 1]
    failed = df[df["Status"] == 0]
    
    plt.scatter(approved["Study_Hours"], approved["Attendance"], color='blue', label='Approved (Data)', s=50)
    plt.scatter(failed["Study_Hours"], failed["Attendance"], color='red', label='Failed (Data)', s=50)
    
    # Create a mesh grid to draw the decision boundaries (Hyperplane and Margins)
    ax = plt.gca()
    xlim = (0, 22)
    ylim = (10, 105)
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 30), np.linspace(ylim[0], ylim[1], 30))
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    Z = model.decision_function(xy).reshape(xx.shape)
    
    # Draw the Hyperplane (0) and Margins (-1, 1)
    ax.contour(xx, yy, Z, colors='black', levels=[-1, 0, 1], alpha=0.6, linestyles=['--', '-', '--'])
    
    # Highlight Support Vectors (key points defining the boundary margin)
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=120,
               facecolors='none', edgecolors='black', linewidths=1.5, label='Support Vectors')
    
    # If the user submitted data from the form, plot "Your Point"
    if user_hours is not None and user_attendance is not None:
        point_color = 'darkblue' if user_pred == 1 else 'darkred'
        plt.scatter(user_hours, user_attendance, color=point_color, marker='X', s=250, 
                    edgecolors='black', label=f'Your Query ({user_hours}h, {user_attendance}%)')
    
    # Plot configuration labels and titles
    plt.title('Student Classification with Support Vector Machine (SVM)')
    plt.xlabel('Study Hours')
    plt.ylabel('Attendance %')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Convert plot to a Base64 string for HTML embedding
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    plt.close()
    
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"