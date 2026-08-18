"""
Lasso Regression (L1 Regularization)
=======================================
Usage:
    python regression/lasso_regression.py
"""

from sklearn.linear_model import Lasso
from _common import load_and_prepare, evaluate_regressor


def main(alpha: float = 0.1):
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = load_and_prepare()

    model = Lasso(alpha=alpha, random_state=42)
    model.fit(X_train_scaled, y_train)

    evaluate_regressor(model, X_test_scaled, y_test, model_name=f"Lasso Regression (alpha={alpha})", n_features=X_train.shape[1])

    print("\nCoefficients (0 = feature dropped by L1):")
    for name, coef in zip(X_train.columns, model.coef_):
        print(f"  {name}: {coef:.4f}")

    return model


if __name__ == "__main__":
    main()
