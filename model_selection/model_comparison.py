import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def compare_classification_models(
    models,
    X_train,
    X_test,
    y_train,
    y_test
):

    results = []

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(
                y_test,
                predictions
            ),
            "Precision": precision_score(
                y_test,
                predictions,
                zero_division=0
            ),
            "Recall": recall_score(
                y_test,
                predictions,
                zero_division=0
            ),
            "F1 Score": f1_score(
                y_test,
                predictions,
                zero_division=0
            )
        })

    return pd.DataFrame(results).sort_values(
        by="F1 Score",
        ascending=False
  )
