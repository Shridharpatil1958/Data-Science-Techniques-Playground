"""
K-Nearest Neighbors Classifier
================================
Usage:
    python classification/knn_classifier.py
"""

from sklearn.neighbors import KNeighborsClassifier
from _common import load_and_prepare, evaluate_classifier


def main(n_neighbors: int = 5):
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = load_and_prepare()

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train_scaled, y_train)

    evaluate_classifier(model, X_test_scaled, y_test, model_name=f"KNN (k={n_neighbors})")
    return model


if __name__ == "__main__":
    main()
