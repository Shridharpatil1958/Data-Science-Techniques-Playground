import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def kmeans_clustering(
    data: pd.DataFrame,
    features: list,
    n_clusters: int = 3
):
    """
    Perform K-Means clustering.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    features : list
        Numerical features used for clustering.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    pd.DataFrame
        Dataset with cluster labels.
    """

    X = data[features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    clusters = model.fit_predict(X_scaled)

    result = data.copy()
    result["Cluster"] = clusters

    return result, model


def plot_clusters(data, x_feature, y_feature):
    plt.figure(figsize=(8, 6))

    plt.scatter(
        data[x_feature],
        data[y_feature],
        c=data["Cluster"]
    )

    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.title("K-Means Clustering")

    plt.show()
