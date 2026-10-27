"""
Inference engine and input validation for PriceWise AI.
Loads saved Scikit-Learn pipelines, validates user inputs against schemas,
and formats predicted prices in Indian Rupees (INR) with confidence ranges.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import joblib
import pandas as pd
import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    MODELS_DIR,
    CATEGORIES,
    CATEGORY_SCHEMAS,
    DATASET_BASE_YEAR,
)

# Global in-memory cache for loaded pipelines
_PIPELINE_CACHE: Dict[str, Any] = {}


def load_model(category: str, model_type: str = "best") -> Any:
    """
    Loads a saved pipeline from disk using in-memory caching.
    `model_type` can be 'best' or an algorithm slug (e.g. 'linear_regression', 'random_forest').
    """
    if category not in CATEGORIES:
        raise ValueError(f"Invalid category '{category}'. Must be one of {CATEGORIES}")

    cache_key = f"{category.lower()}_{model_type.lower().replace(' ', '_')}"
    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key]

    if model_type.lower() == "best":
        filename = f"{category.lower()}_best_model.joblib"
    else:
        slug = model_type.lower().replace(" ", "_")
        filename = f"{category.lower()}_{slug}.joblib"

    filepath = MODELS_DIR / filename
    if not filepath.exists():
        # Fallback to best model if specific model file not found
        fallback_path = MODELS_DIR / f"{category.lower()}_best_model.joblib"
        if not fallback_path.exists():
            raise FileNotFoundError(f"Model file not found at {filepath} or {fallback_path}")
        filepath = fallback_path

    pipeline = joblib.load(filepath)
    _PIPELINE_CACHE[cache_key] = pipeline
    return pipeline


def validate_input(category: str, user_input: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validates user input against expected category schema and acceptable physical ranges.
    Returns (is_valid, error_message).
    """
    if category not in CATEGORY_SCHEMAS:
        return False, f"Unknown category: {category}"

    schema = CATEGORY_SCHEMAS[category]

    # Check for presence of required numeric features (excluding product_age_years which can be auto-computed)
    for num_col in schema["numeric_features"]:
        if num_col == "product_age_years":
            continue
        if num_col not in user_input or user_input[num_col] is None:
            return False, f"Missing required numeric feature: '{num_col}'"
        try:
            val = float(user_input[num_col])
        except (ValueError, TypeError):
            return False, f"Feature '{num_col}' must be a valid number, got: {user_input[num_col]}"

    # Check for presence of required categorical features
    for cat_col in schema["categorical_features"]:
        if cat_col not in user_input or not str(user_input[cat_col]).strip():
            return False, f"Missing required option for: '{cat_col}'"

    # Category-specific logical validations
    year = float(user_input.get("year", 2023))
    if year < 1970 or year > (DATASET_BASE_YEAR + 5):
        return False, f"Year must be between 1970 and {DATASET_BASE_YEAR + 5}."

    if category == "Laptop":
        ram = float(user_input.get("ram_gb", 0))
        storage = float(user_input.get("storage_gb", 0))
        screen = float(user_input.get("screen_inches", 0))
        if ram <= 0 or ram > 256:
            return False, "RAM must be between 1 GB and 256 GB."
        if storage <= 0 or storage > 16384:
            return False, "Storage must be between 64 GB and 16,384 GB."
        if screen <= 8 or screen > 24:
            return False, "Screen size must be between 8 and 24 inches."

    elif category == "Car":
        mileage = float(user_input.get("mileage_km", 0))
        if mileage < 0 or mileage > 1_000_000:
            return False, "Mileage must be between 0 and 1,000,000 km."

    elif category == "House":
        area = float(user_input.get("area_sqft", 0))
        bedrooms = float(user_input.get("bedrooms", 0))
        bathrooms = float(user_input.get("bathrooms", 0))
        if area < 100 or area > 50_000:
            return False, "Area must be between 100 and 50,000 sq.ft."
        if bedrooms < 1 or bedrooms > 20:
            return False, "Bedrooms must be between 1 and 20."
        if bathrooms < 1 or bathrooms > 20:
            return False, "Bathrooms must be between 1 and 20."

    return True, None


def format_inr(amount: float) -> str:
    """
    Formats a numeric price into a stylish Indian Rupee string.
    E.g.:
      18,500,000 -> ₹ 1.85 Cr
      750,000 -> ₹ 7.50 Lakh
      42,500 -> ₹ 42,500
    """
    if amount < 0:
        amount = 0.0

    if amount >= 10_000_000:
        cr = amount / 10_000_000
        return f"₹ {cr:.2f} Cr"
    elif amount >= 100_000:
        lakh = amount / 100_000
        return f"₹ {lakh:.2f} Lakh"
    else:
        return f"₹ {amount:,.2f}"


def predict_price(
    category: str,
    raw_input: Dict[str, Any],
    model_type: str = "best"
) -> Dict[str, Any]:
    """
    Full inference workflow:
    1. Validates raw input
    2. Constructs feature row with auto-derived features
    3. Executes pipeline inference
    4. Formats price and computes estimated confidence range
    """
    is_valid, error_msg = validate_input(category, raw_input)
    if not is_valid:
        return {
            "success": False,
            "error": error_msg,
            "predicted_price": None,
            "formatted_price": None,
        }

    schema = CATEGORY_SCHEMAS[category]
    feature_row = {}

    # Copy input features
    for col in schema["numeric_features"]:
        if col == "product_age_years":
            # Auto-calculate from year if not provided or ensure consistency
            year = float(raw_input.get("year", DATASET_BASE_YEAR))
            age = max(0.0, float(DATASET_BASE_YEAR - year))
            feature_row[col] = age
        else:
            feature_row[col] = float(raw_input[col])

    for col in schema["categorical_features"]:
        feature_row[col] = str(raw_input[col])

    # Convert to single-row DataFrame
    ordered_cols = schema["numeric_features"] + schema["categorical_features"]
    input_df = pd.DataFrame([feature_row])[ordered_cols]

    try:
        pipeline = load_model(category, model_type)
        raw_pred = float(pipeline.predict(input_df)[0])

        # Floor prediction at non-negative value
        predicted_price = max(0.0, raw_pred)

        # Retrieve category MAE for confidence range estimation
        mae = 5000.0
        try:
            from src.config import EVAL_METRICS_PATH
            import json
            if EVAL_METRICS_PATH.exists():
                with open(EVAL_METRICS_PATH, "r", encoding="utf-8") as f:
                    metrics_data = json.load(f)
                    cat_metrics = metrics_data.get(category, {})
                    active_model = model_type if model_type != "best" else list(cat_metrics.keys())[0]
                    if active_model in cat_metrics:
                        mae = float(cat_metrics[active_model].get("MAE", 5000.0))
        except Exception:
            pass

        lower_bound = max(0.0, predicted_price - mae)
        upper_bound = predicted_price + mae

        return {
            "success": True,
            "category": category,
            "model_type": model_type,
            "predicted_price": round(predicted_price, 2),
            "formatted_price": format_inr(predicted_price),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "formatted_range": f"{format_inr(lower_bound)} - {format_inr(upper_bound)}",
            "features_used": feature_row,
            "mae": mae,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Prediction failed: {str(e)}",
            "predicted_price": None,
            "formatted_price": None,
        }
