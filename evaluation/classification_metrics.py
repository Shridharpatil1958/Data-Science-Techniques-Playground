import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def classification_report_dataframe(
    y_true,
    y_pred,
    y_probability=None
):

    metrics = {
        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),
        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_true,
            y_pred,
            zero_division=0
        )
    }

    if y_probability is not None:

        metrics["ROC-AUC"] = roc_auc_score(
            y_true,
            y_probability
        )

    return pd.DataFrame(
        metrics.items(),
        columns=["Metric", "Score"]
    )
