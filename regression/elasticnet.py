"""
ElasticNet Regression (L1 + L2 Regularization)
=================================================
Usage:
    python regression/elasticnet.py
"""

from sklearn.linear_model import ElasticNet
from _common import load_and_prepare, evaluate_regressor


def main(alpha: float = 0.1, l1_ratio: float = 0.5):
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = load_and_prepare()

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    model.fit(X_train_scaled, y_train)

    evaluate_regressor(
        model, X_test_scaled, y_test,
        model_name=f"ElasticNet (alpha={alpha}, l1_ratio={l1_ratio})",
        n_features=X_train.shape[1],
    )
    return model


if __name__ == "__main__":
    main()
