
# Customer Segmentation using KMeans Clustering

## Objective
Segment customers based on purchasing behavior using clustering techniques for targeted marketing strategies.

## Dataset
- **File:** customer_data.csv
- **Columns:** Customer ID, Age, Annual Income, Spending Score

## Steps Performed
1. **Data Preprocessing**: Standardized features for clustering.
2. **Clustering**: Used KMeans with optimal cluster count determined via Elbow Method.
3. **Visualization**: Created Elbow Plot and PCA Scatter Plot.

## Output Files
- `clustered_customer_data.csv`
- `elbow_plot.png`
- `pca_clusters.png`

## Recommendations
- Cluster 0: High income, low spending → Target with luxury promotions.
- Cluster 1: Young, high spending → Engage with loyalty rewards.
- Cluster 2: Average spenders → Upsell products.
- Cluster 3: Low income, low spending → Use budget-friendly offers.
