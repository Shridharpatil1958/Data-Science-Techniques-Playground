"""
Logistic Regression Classifier
================================
Usage:
    python classification/logistic_regression.py
"""

from sklearn.linear_model import LogisticRegression
from _common import load_and_prepare, evaluate_classifier


def main():
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = load_and_prepare()

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    evaluate_classifier(model, X_test_scaled, y_test, model_name="Logistic Regression")
    return model


if __name__ == "__main__":
    main()
