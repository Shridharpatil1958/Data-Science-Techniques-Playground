"""
End-to-End Model Pipeline
============================
Combines preprocessing and model training into a single
Pipeline object, then evaluates and saves the fitted pipeline.

Usage:
    python pipelines/model_pipeline.py
"""

import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from preprocessing_pipeline import build_preprocessor, load_split_data, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def build_full_pipeline(model=None):
    if model is None:
        model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)

    preprocessor = build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    return pipeline


def main(save_path: str = "saved_models/model_pipeline.joblib"):
    X_train, X_test, y_train, y_test = load_split_data()

    pipeline = build_full_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Pipeline Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(pipeline, save_path)
    print(f"\nSaved fitted pipeline to {save_path}")

    return pipeline


if __name__ == "__main__":
    main()
