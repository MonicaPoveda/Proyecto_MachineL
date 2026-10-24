import io
import base64
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



# Dataset and Initial Model Training

data = {
    "Study Hours": [10, 15, 12, 8, 14, 5, 16, 7, 11, 13, 9, 4, 18, 3, 17, 6, 14, 2, 20, 1],
    "Final Grade": [3.8, 4.2, 3.6, 3, 4.5, 2.5, 4.8, 2.8, 3.7, 4, 3.2, 2.2, 5, 1.8, 4.9, 2.7, 4.4, 1.5, 5, 1]
}
df = pd.DataFrame(data)

x = df[["Study Hours"]]
y = df[["Final Grade"]]

# Train the Linear Regression model
model = LinearRegression()
model.fit(x, y)


def calculateGrade(hours):
    """Predict the final grade based on input study hours."""
    result = model.predict([[hours]])[0]
    return result


def generate_plot(predicted_hours=None, predicted_grade=None):
    """Generates a styled scatter plot with the regression line and returns it as a Base64 string."""
    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=100)
    
    # Set background colors
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    
    # Plot historical data points
    ax.scatter(
        df["Study Hours"], 
        df["Final Grade"], 
        color='#1478ff', 
        alpha=0.8, 
        edgecolors='white', 
        linewidth=0.5, 
        label='Historical Data', 
        s=45
    )
    
    # Generate predictions and plot the regression line
    y_pred = model.predict(x)
    ax.plot(
        df["Study Hours"], 
        y_pred, 
        color='#00d2ff', 
        linewidth=2, 
        label='Regression Line'
    )
    
    # Highlight the new prediction on the plot if provided
    if predicted_hours is not None and predicted_grade is not None:
        # Extract scalar value if prediction comes as a NumPy array
        val_grade = float(predicted_grade[0]) if hasattr(predicted_grade, '__len__') else float(predicted_grade)
        
        ax.scatter(
            [predicted_hours], 
            [val_grade], 
            color="#ff0019", 
            s=120, 
            zorder=5, 
            label=f'Prediction ({predicted_hours}h)', 
            edgecolors='white', 
            linewidth=1.5
        )
        ax.annotate(
            f'{val_grade:.2f}', 
            (predicted_hours, val_grade), 
            textcoords="offset points", 
            xytext=(0, 10), 
            ha='center', 
            color='#ffffff', 
            fontweight='bold'
        )

    # Configure plot labels, title, and styling
    ax.set_title('Study Hours vs Final Grade', color="#000000FF", fontsize=11, pad=10, fontweight='bold')
    ax.set_xlabel('Study Hours', color="#000000FF", fontsize=9)
    ax.set_ylabel('Final Grade', color="#000000FF", fontsize=9)
    
    ax.tick_params(colors="#000000FF", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#000000FF")
        
    ax.grid(True, linestyle='--', alpha=0.15, color='#1478ff')
    ax.legend(facecolor="#A4E0F6FF", edgecolor='#1478ff59', labelcolor="#000000", fontsize=8)
    
    plt.tight_layout()

    # Convert the generated plot to a Base64 encoded string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return image_base64