# kmeans_model.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import io
import base64

def generate_clusters():
    # 1. Load data
    df = pd.read_csv('Data/customer_segmentation.csv')
    
    # 2. Remove missing values
    df = df.dropna()
    
    # 3. Select variables for clustering
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    
    # 4. Apply K-Means with 5 clusters
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)
    
    # 5. Get centroids
    centroids = kmeans.cluster_centers_
    
    # 6. Calculate Silhouette Score
    score = silhouette_score(X, df['Cluster'])
    
    # 7. Create scatter plot
    plt.figure(figsize=(10, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for i in range(5):
        cluster_data = df[df['Cluster'] == i]
        plt.scatter(cluster_data['Annual Income (k$)'], 
                   cluster_data['Spending Score (1-100)'], 
                   c=colors[i], label=f'Cluster {i}', alpha=0.6, s=50)
    
    plt.scatter(centroids[:, 0], centroids[:, 1], 
                s=300, c='red', marker='X', label='Centroids', edgecolors='black')
    
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title('Customer Segmentation with K-Means')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Convert plot to base64
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    # 9. Cluster summary
    summary = df.groupby('Cluster').agg({
        'Annual Income (k$)': 'mean',
        'Spending Score (1-100)': 'mean',
        'Cluster': 'count'
    }).rename(columns={'Cluster': 'Count'}).round(2)
    
    # 10. Cluster interpretation
    interpretations = {}

    income_mean = df['Annual Income (k$)'].mean()
    score_mean = df['Spending Score (1-100)'].mean()

    for cluster_id, row in summary.iterrows():
        income = row['Annual Income (k$)']
        spending = row['Spending Score (1-100)']

        if income >= income_mean and spending >= score_mean:
            title = "High Income / High Spending"
            description = "Customers with above-average income and above-average spending."
        elif income >= income_mean and spending < score_mean:
            title = "High Income / Low Spending"
            description = "Customers with above-average income and below-average spending."
        elif income < income_mean and spending >= score_mean:
            title = "Low Income / High Spending"
            description = "Customers with below-average income and above-average spending."
        else:
            title = "Low Income / Low Spending"
            description = "Customers with below-average income and below-average spending."

        interpretations[int(cluster_id)] = {
            "title": title,
            "description": description
        }

    # 11. Dataset statistics
    num_records = len(df)

    # 12. Complete data and preview with cluster assignment
    data_columns = [
        'CustomerID',
        'Annual Income (k$)',
        'Spending Score (1-100)',
        'Cluster'
    ]

    data_full = df[data_columns].to_dict(orient='records')
    data_preview = data_full[:20]

    # 13. Data ranges
    income_min = df['Annual Income (k$)'].min()
    income_max = df['Annual Income (k$)'].max()
    score_min = df['Spending Score (1-100)'].min()
    score_max = df['Spending Score (1-100)'].max()

    return (
        df,
        centroids,
        score,
        plot_url,
        summary,
        num_records,
        data_preview,
        data_full,
        income_min,
        income_max,
        score_min,
        score_max,
        interpretations
    )


if __name__ == "__main__":
    result = generate_clusters()
    print(f"Clusters generated: {result[5]} records")
    print(f"Silhouette Score: {result[2]:.4f}")
    print(result[4])
    print(f"\nData Preview (first 5 records):")
    for row in result[6][:5]:
        print(row)