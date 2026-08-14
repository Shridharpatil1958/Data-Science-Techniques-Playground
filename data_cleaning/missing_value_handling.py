"""
Missing Value Handling
=======================
Demonstrates common strategies for handling missing data:
mean, median, mode, and KNN imputation.

Usage:
    python data_cleaning/missing_value_handling.py
"""

import pandas as pd
from sklearn.impute import KNNImputer

DATA_PATH = "datasets/titanic.csv"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def report_missing(df: pd.DataFrame) -> pd.Series:
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print("\nMissing values per column:")
    print(missing)
    return missing


def impute_mean(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    df[column] = df[column].fillna(df[column].mean())
    return df


def impute_median(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    df[column] = df[column].fillna(df[column].median())
    return df


def impute_mode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    df[column] = df[column].fillna(df[column].mode()[0])
    return df


def impute_knn(df: pd.DataFrame, numeric_columns: list, n_neighbors: int = 5) -> pd.DataFrame:
    df = df.copy()
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    return df


if __name__ == "__main__":
    df = load_data()
    report_missing(df)

    if "Age" in df.columns:
        df = impute_median(df, "Age")
    if "Embarked" in df.columns:
        df = impute_mode(df, "Embarked")
    if "Fare" in df.columns:
        df = impute_mean(df, "Fare")

    print("\nAfter imputation:")
    report_missing(df)
