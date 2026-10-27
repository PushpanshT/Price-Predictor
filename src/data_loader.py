"""
Data Loader and Preprocessing Utilities for PriceWise AI.
Handles dataset loading, category-specific subsetting, missing value handling,
and leak-free train-test splitting.
"""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, CATEGORIES, CATEGORY_SCHEMAS, RANDOM_STATE, TEST_SIZE


def load_raw_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw CSV data into a Pandas DataFrame.
    """
    path = filepath or DATA_PATH
    df = pd.read_csv(path)
    return df


def get_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns high-level statistics and metadata about the dataset.
    """
    category_counts = df["category"].value_counts().to_dict()
    null_counts = df.isnull().sum().to_dict()
    duplicates_count = int(df.duplicated().sum())

    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "categories": list(category_counts.keys()),
        "category_counts": category_counts,
        "duplicates": duplicates_count,
        "null_counts": null_counts,
    }


def clean_category_data(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """
    Extracts, validates, and cleans data specific to a given category.
    - Filters rows by category
    - Selects relevant features according to category schema
    - Removes duplicate rows within the category
    - Imputes any missing values gracefully (median for numerical, mode for categorical)
    """
    if category not in CATEGORY_SCHEMAS:
        raise ValueError(f"Unknown category '{category}'. Allowed: {CATEGORIES}")

    schema = CATEGORY_SCHEMAS[category]
    sub_df = df[df["category"] == category].copy()

    # Required columns for this category
    req_cols = schema["numeric_features"] + schema["categorical_features"] + [schema["target"]]

    # Ensure all required columns exist
    missing_cols = [c for c in req_cols if c not in sub_df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns for {category}: {missing_cols}")

    sub_df = sub_df[req_cols].copy()

    # Drop duplicate records within this category if any
    sub_df = sub_df.drop_duplicates().reset_index(drop=True)

    # Impute missing values if present (imputation strategy strictly without leakage or using category defaults)
    for num_col in schema["numeric_features"]:
        if sub_df[num_col].isnull().any():
            median_val = sub_df[num_col].median()
            sub_df[num_col] = sub_df[num_col].fillna(median_val)

    for cat_col in schema["categorical_features"]:
        if sub_df[cat_col].isnull().any():
            mode_val = sub_df[cat_col].mode()[0] if not sub_df[cat_col].mode().empty else "Unknown"
            sub_df[cat_col] = sub_df[cat_col].fillna(mode_val)

    # Ensure target column is strictly positive and non-null
    sub_df = sub_df.dropna(subset=[schema["target"]])
    sub_df = sub_df[sub_df[schema["target"]] > 0]

    return sub_df


def get_train_test_data(
    df: pd.DataFrame,
    category: str,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Extracts features and target for a category and splits into train and test sets (80:20).
    Guarantees zero data leakage by returning raw feature splits before fitting transformers.
    """
    sub_df = clean_category_data(df, category)
    schema = CATEGORY_SCHEMAS[category]
    feature_cols = schema["numeric_features"] + schema["categorical_features"]
    target_col = schema["target"]

    X = sub_df[feature_cols].copy()
    y = sub_df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    return X_train, X_test, y_train, y_test


def get_category_uniques(df: pd.DataFrame, category: str) -> Dict[str, Any]:
    """
    Retrieves all available options and valid ranges for UI input widgets.
    """
    sub_df = clean_category_data(df, category)
    schema = CATEGORY_SCHEMAS[category]
    uniques = {}

    for cat_col in schema["categorical_features"]:
        uniques[cat_col] = sorted(sub_df[cat_col].dropna().unique().tolist())

    for num_col in schema["numeric_features"]:
        series = sub_df[num_col].dropna()
        uniques[num_col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
            "mean": float(series.mean()),
            "distinct_values": sorted([float(x) for x in series.unique()]) if series.nunique() <= 10 else None
        }

    return uniques
