import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

def plot_clusters(df: pd.DataFrame, x: str, y: str, hue: str = "Cluster"):
    fig, ax = plt.subplots()
    sns.scatterplot(x=x, y=y, data=df, hue=hue, palette="colorblind", ax=ax)
    ax.set_title("K-Means Clusters")
    st.pyplot(fig)

def plot_elbow(df: pd.DataFrame):
    fig, ax = plt.subplots()
    df.plot(x="cluster", y="WSS_Score", marker='o', ax=ax)
    ax.set_xlabel("No. of clusters")
    ax.set_ylabel("WSS Score")
    ax.set_title("Elbow Plot")
    st.pyplot(fig)

def plot_silhouette(df: pd.DataFrame):
    fig, ax = plt.subplots()
    df.plot(x="cluster", y="Silhouette_Score", marker='o', color='green', ax=ax)
    ax.set_xlabel("No. of clusters")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Plot")
    st.pyplot(fig)
