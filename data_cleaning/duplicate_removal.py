"""
Duplicate Removal
==================
Identify and remove duplicate rows / records from a dataset.

Usage:
    python data_cleaning/duplicate_removal.py
"""

import pandas as pd

DATA_PATH = "datasets/titanic.csv"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def report_duplicates(df: pd.DataFrame, subset: list = None) -> int:
    n_dupes = df.duplicated(subset=subset).sum()
    print(f"Found {n_dupes} duplicate rows" + (f" (subset={subset})" if subset else ""))
    return n_dupes


def drop_duplicates(df: pd.DataFrame, subset: list = None, keep: str = "first") -> pd.DataFrame:
    before = df.shape[0]
    df_clean = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    after = df_clean.shape[0]
    print(f"Removed {before - after} duplicate rows ({before} -> {after})")
    return df_clean


if __name__ == "__main__":
    df = load_data()
    report_duplicates(df)
    df_clean = drop_duplicates(df)

    id_col = "PassengerId" if "PassengerId" in df.columns else None
    if id_col:
        report_duplicates(df, subset=[id_col])
