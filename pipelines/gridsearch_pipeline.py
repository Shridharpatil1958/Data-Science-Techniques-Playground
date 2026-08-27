"""
GridSearchCV Pipeline
========================
Wraps the full preprocessing + model pipeline in GridSearchCV
for hyperparameter tuning, using the pipeline-prefixed param
naming convention (e.g. "model__n_estimators").

Usage:
    python pipelines/gridsearch_pipeline.py
"""

import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from preprocessing_pipeline import load_split_data
from model_pipeline import build_full_pipeline

PARAM_GRID = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [4, 6, 8, None],
    "model__min_samples_split": [2, 5, 10],
}


def run_gridsearch(param_grid: dict = PARAM_GRID, cv: int = 5, scoring: str = "accuracy"):
    X_train, X_test, y_train, y_test = load_split_data()

    pipeline = build_full_pipeline(model=RandomForestClassifier(random_state=42))

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print(f"\nBest params: {search.best_params_}")
    print(f"Best CV {scoring}: {search.best_score_:.4f}")

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return search


if __name__ == "__main__":
    search = run_gridsearch()
    joblib.dump(search.best_estimator_, "saved_models/best_pipeline.joblib")
    print("\nSaved best pipeline to saved_models/best_pipeline.joblib")
