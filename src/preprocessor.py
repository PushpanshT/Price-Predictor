"""
Preprocessor pipelines and transformer builders for PriceWise AI.
Ensures zero data leakage by encapsulating all scaling and encoding steps
inside Scikit-Learn ColumnTransformer and Pipeline objects.
"""

from typing import List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator

from src.config import CATEGORY_SCHEMAS


def create_preprocessor(category: str) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer for a specific category.
    Handles numeric scaling with StandardScaler and categorical encoding with OneHotEncoder.
    Imputers are embedded inside each branch to prevent data leakage during inference or cross-validation.
    """
    if category not in CATEGORY_SCHEMAS:
        raise ValueError(f"Unknown category: {category}")

    schema = CATEGORY_SCHEMAS[category]
    numeric_features = schema["numeric_features"]
    categorical_features = schema["categorical_features"]

    transformers = []

    if numeric_features:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipeline, numeric_features))

    if categorical_features:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipeline, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor


def build_pipeline(category: str, model: BaseEstimator) -> Pipeline:
    """
    Constructs an end-to-end Scikit-Learn Pipeline combining preprocessing and model estimation.
    """
    preprocessor = create_preprocessor(category)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])
    return pipeline


def get_feature_names_from_preprocessor(pipeline: Pipeline, category: str) -> List[str]:
    """
    Extracts human-readable feature names generated after one-hot encoding.
    """
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        feature_names = []

        for name, transformer, cols in preprocessor.transformers_:
            if name == "num":
                feature_names.extend(cols)
            elif name == "cat":
                ohe = transformer.named_steps["ohe"]
                cat_names = ohe.get_feature_names_out(cols).tolist()
                feature_names.extend(cat_names)

        return feature_names
    except Exception:
        # Fallback if names cannot be retrieved
        schema = CATEGORY_SCHEMAS.get(category, {})
        return schema.get("numeric_features", []) + schema.get("categorical_features", [])
