from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def create_smote_pipeline():

    pipeline = Pipeline([
        ("scaler", StandardScaler()),

        ("smote", SMOTE(
            random_state=42
        )),

        ("classifier", LogisticRegression(
            max_iter=1000
        ))
    ])

    return pipeline
