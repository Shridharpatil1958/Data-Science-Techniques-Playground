"""
Linear Regression
====================
Usage:
    python regression/linear_regression.py
"""

from sklearn.linear_model import LinearRegression
from _common import load_and_prepare, evaluate_regressor


def main():
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = load_and_prepare()

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    evaluate_regressor(model, X_test_scaled, y_test, model_name="Linear Regression", n_features=X_train.shape[1])
    return model


if __name__ == "__main__":
    main()
