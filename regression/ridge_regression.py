"""
Ridge Regression (L2 Regularization)
=======================================
Usage:
    python regression/ridge_regression.py
"""

from sklearn.linear_model import Ridge
from _common import load_and_prepare, evaluate_regressor


def main(alpha: float = 1.0):
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = load_and_prepare()

    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train_scaled, y_train)

    evaluate_regressor(model, X_test_scaled, y_test, model_name=f"Ridge Regression (alpha={alpha})", n_features=X_train.shape[1])
    return model


if __name__ == "__main__":
    main()
