"""
Decision Tree Regressor
==========================
Usage:
    python regression/decision_tree_regressor.py
"""

from sklearn.tree import DecisionTreeRegressor
from _common import load_and_prepare, evaluate_regressor


def main(max_depth: int = 5):
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = load_and_prepare()

    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    evaluate_regressor(model, X_test, y_test, model_name=f"Decision Tree Regressor (max_depth={max_depth})", n_features=X_train.shape[1])
    return model


if __name__ == "__main__":
    main()
