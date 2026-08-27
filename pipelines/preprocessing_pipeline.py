"""
Preprocessing Pipeline
=========================
Builds a reusable ColumnTransformer-based preprocessing pipeline
that handles numeric scaling and categorical encoding, preventing
data leakage between train/test splits.

Usage:
    python pipelines/preprocessing_pipeline.py
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

DATA_PATH = "datasets/titanic.csv"
TARGET = "Survived"
NUMERIC_FEATURES = ["Age", "Fare", "SibSp", "Parch"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked"]


def build_preprocessor(numeric_features=NUMERIC_FEATURES, categorical_features=CATEGORICAL_FEATURES):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    return preprocessor


def load_split_data(path: str = DATA_PATH, target: str = TARGET):
    df = pd.read_csv(path)
    features = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    df = df.dropna(subset=[target])

    X = df[features]
    y = df[target]

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_split_data()

    numeric = [c for c in NUMERIC_FEATURES if c in X_train.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    preprocessor = build_preprocessor(numeric, categorical)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    print(f"Train shape after preprocessing: {X_train_transformed.shape}")
    print(f"Test shape after preprocessing:  {X_test_transformed.shape}")
