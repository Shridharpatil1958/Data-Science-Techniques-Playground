"""
Gradient Boosting Classifier
===============================
Usage:
    python classification/gradient_boosting.py
"""

from sklearn.ensemble import GradientBoostingClassifier
from _common import load_and_prepare, evaluate_classifier


def main(n_estimators: int = 150, learning_rate: float = 0.1):
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = load_and_prepare()

    model = GradientBoostingClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate, random_state=42
    )
    model.fit(X_train, y_train)

    evaluate_classifier(model, X_test, y_test, model_name="Gradient Boosting")
    return model


if __name__ == "__main__":
    main()
