import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Step 1: Load the Dataset
df = pd.read_csv("customer_data.csv")
print("Initial Data:\n", df.head())
print("\nData Summary:\n", df.describe())
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())

# Step 2: Data Preprocessing
features = df[['Age', 'Annual Income', 'Spending Score']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 3: Elbow Method to find optimal clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.savefig('elbow_plot.png')
plt.show()

# Optional: Silhouette Score
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    labels = kmeans.fit_predict(scaled_features)
    score = silhouette_score(scaled_features, labels)
    print(f'Silhouette Score for {i} clusters: {score:.4f}')

# Step 4: Apply KMeans (Choose optimal cluster, e.g., 4)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Step 5: Visualization using PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_features)
df['PCA1'] = pca_data[:, 0]
df['PCA2'] = pca_data[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set1')
plt.title('Customer Segments by PCA')
plt.savefig('pca_clusters.png')
plt.show()

# Pair Plot
sns.pairplot(df, hue='Cluster', vars=['Age', 'Annual Income', 'Spending Score'])
plt.savefig('pairplot_clusters.png')
plt.show()

# Save the clustered data
df.to_csv("clustered_customer_data.csv", index=False)
print("Clustered data saved to clustered_customer_data.csv")
