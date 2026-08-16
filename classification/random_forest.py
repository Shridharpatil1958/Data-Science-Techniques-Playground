"""
Random Forest Classifier
==========================
Usage:
    python classification/random_forest.py
"""

from sklearn.ensemble import RandomForestClassifier
from _common import load_and_prepare, evaluate_classifier


def main(n_estimators: int = 200, max_depth: int = 6):
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = load_and_prepare()

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)

    evaluate_classifier(model, X_test, y_test, model_name="Random Forest")

    print("\nFeature Importances:")
    for name, importance in sorted(
        zip(X_train.columns, model.feature_importances_), key=lambda x: -x[1]
    ):
        print(f"  {name}: {importance:.4f}")

    return model


if __name__ == "__main__":
    main()
