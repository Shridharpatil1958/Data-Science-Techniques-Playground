"""
Decision Tree Classifier
==========================
Usage:
    python classification/decision_tree.py
"""

from sklearn.tree import DecisionTreeClassifier
from _common import load_and_prepare, evaluate_classifier


def main(max_depth: int = 5):
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = load_and_prepare()

    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    evaluate_classifier(model, X_test, y_test, model_name=f"Decision Tree (max_depth={max_depth})")

    print("\nFeature Importances:")
    for name, importance in zip(X_train.columns, model.feature_importances_):
        print(f"  {name}: {importance:.4f}")

    return model


if __name__ == "__main__":
    main()
