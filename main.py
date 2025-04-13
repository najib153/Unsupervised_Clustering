import streamlit as st
from data_loader import load_data
from preprocessing import train_kmeans, evaluate_kmeans
from visualization import plot_clusters, plot_elbow, plot_silhouette

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title(" Customer Segmentation with K-Means Clustering")

# Load data
try:
    df = load_data("data/mall_customers.csv")
    st.success("Data loaded successfully.")
except FileNotFoundError:
    st.error(" mall_customers.csv not found in the /data folder.")
    st.stop()

# Show data preview
st.subheader(" Dataset Preview")
st.dataframe(df.head())

# Feature selection
features = ["Annual_Income", "Spending_Score"]
n_clusters = st.slider("Select number of clusters:", 2, 10, 5)

# Train KMeans model
model = train_kmeans(df, features, n_clusters)
df["Cluster"] = model.labels_

# Show clusters
st.subheader("🗺 Cluster Visualization")
plot_clusters(df, "Annual_Income", "Spending_Score")

# Evaluate across different cluster values
st.subheader(" K-Means Evaluation")
eval_df = evaluate_kmeans(df, features, range(3, 9))

st.markdown("**Elbow Plot (WSS Score)**")
plot_elbow(eval_df)

st.markdown("**Silhouette Plot**")
plot_silhouette(eval_df)
