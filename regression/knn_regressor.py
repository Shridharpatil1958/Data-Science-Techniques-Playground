"""
K-Nearest Neighbors Regressor
================================
Usage:
    python regression/knn_regressor.py
"""

from sklearn.neighbors import KNeighborsRegressor
from _common import load_and_prepare, evaluate_regressor


def main(n_neighbors: int = 5):
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = load_and_prepare()

    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train_scaled, y_train)

    evaluate_regressor(model, X_test_scaled, y_test, model_name=f"KNN Regressor (k={n_neighbors})", n_features=X_train.shape[1])
    return model


if __name__ == "__main__":
    main()
