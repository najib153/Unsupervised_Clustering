from data_loader import load_data
from preprocessing import train_kmeans, evaluate_kmeans
from visualization import plot_clusters, plot_elbow, plot_silhouette

df = load_data(data/"mall_customers.csv")
features = ["Annual_Income", "Spending_Score"]

model = train_kmeans(df, features, n_clusters=5)
df["Cluster"] = model.labels_

plot_clusters(df, "Annual_Income", "Spending_Score")
eval_df = evaluate_kmeans(df, features, range(3, 9))
plot_elbow(eval_df)
plot_silhouette(eval_df)
