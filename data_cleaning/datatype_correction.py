"""
Data Type Correction & Category Standardization
=================================================
Fixes incorrect column dtypes and standardizes categorical values
(casing, whitespace, inconsistent labels).

Usage:
    python data_cleaning/datatype_correction.py
"""

import pandas as pd

DATA_PATH = "datasets/titanic.csv"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def show_dtypes(df: pd.DataFrame):
    print("\nCurrent dtypes:")
    print(df.dtypes)


def correct_dtypes(df: pd.DataFrame, dtype_map: dict) -> pd.DataFrame:
    """
    dtype_map example:
        {"Survived": "category", "Pclass": "category", "Age": "float64"}
    """
    df = df.copy()
    for column, dtype in dtype_map.items():
        if column in df.columns:
            df[column] = df[column].astype(dtype)
    return df


def standardize_categories(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )
    return df


if __name__ == "__main__":
    df = load_data()
    show_dtypes(df)

    dtype_map = {}
    if "Survived" in df.columns:
        dtype_map["Survived"] = "category"
    if "Pclass" in df.columns:
        dtype_map["Pclass"] = "category"

    df = correct_dtypes(df, dtype_map)

    if "Sex" in df.columns:
        df = standardize_categories(df, "Sex")

    show_dtypes(df)
