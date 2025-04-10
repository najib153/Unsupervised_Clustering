import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_clusters(df: pd.DataFrame, x: str, y: str, hue: str = "Cluster"):
    sns.scatterplot(x=x, y=y, data=df, hue=hue, palette="colorblind")
    plt.show()

def plot_elbow(df: pd.DataFrame):
    df.plot(x="cluster", y="WSS_Score")
    plt.xlabel("No. of clusters")
    plt.ylabel("WSS Score")
    plt.title("Elbow Plot")
    plt.show()

def plot_silhouette(df: pd.DataFrame):
    df.plot(x="cluster", y="Silhouette_Score")
    plt.xlabel("No. of clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Plot")
    plt.show()
