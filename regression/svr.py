"""
Support Vector Regressor (SVR)
=================================
Usage:
    python regression/svr.py
"""

from sklearn.svm import SVR
from _common import load_and_prepare, evaluate_regressor


def main(kernel: str = "rbf", C: float = 1.0, epsilon: float = 0.1):
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = load_and_prepare()

    model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    model.fit(X_train_scaled, y_train)

    evaluate_regressor(model, X_test_scaled, y_test, model_name=f"SVR ({kernel} kernel)", n_features=X_train.shape[1])
    return model


if __name__ == "__main__":
    main()
