from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

def train_kmeans(df: pd.DataFrame, features: list, n_clusters: int) -> KMeans:
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(df[features])
    return model

def evaluate_kmeans(df: pd.DataFrame, features: list, k_range: range) -> pd.DataFrame:
    results = {"cluster": [], "WSS_Score": [], "Silhouette_Score": []}
    for k in k_range:
        model = train_kmeans(df, features, k)
        labels = model.labels_
        wss = model.inertia_
        silhouette = silhouette_score(df[features], labels)
        results["cluster"].append(k)
        results["WSS_Score"].append(wss)
        results["Silhouette_Score"].append(silhouette)
    return pd.DataFrame(results)
