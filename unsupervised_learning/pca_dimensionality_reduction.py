import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def perform_pca(data: pd.DataFrame, features: list, n_components=2):

    X = data[features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)

    components = pca.fit_transform(X_scaled)

    columns = [
        f"PC{i + 1}"
        for i in range(n_components)
    ]

    result = pd.DataFrame(
        components,
        columns=columns,
        index=data.index
    )

    return result, pca


def explained_variance(pca):

    return pd.DataFrame({
        "Component": range(
            1,
            len(pca.explained_variance_ratio_) + 1
        ),
        "Explained_Variance": pca.explained_variance_ratio_
    })
