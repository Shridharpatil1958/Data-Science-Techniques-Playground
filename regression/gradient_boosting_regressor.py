"""
Gradient Boosting Regressor
==============================
Usage:
    python regression/gradient_boosting_regressor.py
"""

from sklearn.ensemble import GradientBoostingRegressor
from _common import load_and_prepare, evaluate_regressor


def main(n_estimators: int = 150, learning_rate: float = 0.1):
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = load_and_prepare()

    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
    model.fit(X_train, y_train)

    evaluate_regressor(model, X_test, y_test, model_name="Gradient Boosting Regressor", n_features=X_train.shape[1])
    return model


if __name__ == "__main__":
    main()
