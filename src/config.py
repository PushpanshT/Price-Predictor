"""
Configuration and constants for PriceWise AI.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "pricewise_combined_dataset.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

EVAL_METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"
TEST_PREDICTIONS_PATH = MODELS_DIR / "test_predictions.json"
METADATA_PATH = MODELS_DIR / "metadata.json"

CATEGORIES = ["Laptop", "Car", "House", "Furniture"]

# Feature schemas for each category
CATEGORY_SCHEMAS = {
    "Laptop": {
        "numeric_features": ["year", "ram_gb", "storage_gb", "screen_inches", "product_age_years"],
        "categorical_features": ["brand", "model", "condition", "processor", "gpu"],
        "target": "price_inr",
        "icon": "💻",
        "description": "Laptops & Notebooks with specs such as RAM, Storage, CPU, GPU, and Display"
    },
    "Car": {
        "numeric_features": ["year", "mileage_km", "product_age_years"],
        "categorical_features": ["brand", "model", "condition", "fuel_type"],
        "target": "price_inr",
        "icon": "🚗",
        "description": "Pre-owned & New Automobiles with mileage, fuel type, brand, and condition"
    },
    "House": {
        "numeric_features": ["year", "area_sqft", "bedrooms", "bathrooms", "product_age_years"],
        "categorical_features": [],
        "target": "price_inr",
        "icon": "🏡",
        "description": "Residential Properties evaluated by square footage, bed/bath count, and age"
    },
    "Furniture": {
        "numeric_features": ["year", "product_age_years"],
        "categorical_features": ["brand", "model", "condition", "material"],
        "target": "price_inr",
        "icon": "🛋️",
        "description": "Home & Office Furniture pieces classified by material, item type, and brand"
    }
}

# Reference year used in the dataset
DATASET_BASE_YEAR = 2026
RANDOM_STATE = 42
TEST_SIZE = 0.20
