"""
Unit and integration tests for PriceWise AI.
Tests dataset integrity, preprocessing, model evaluation, and inference validation.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    DATA_PATH,
    MODELS_DIR,
    CATEGORIES,
    CATEGORY_SCHEMAS,
    EVAL_METRICS_PATH,
    METADATA_PATH,
)
from src.data_loader import (
    load_raw_data,
    clean_category_data,
    get_train_test_data,
    get_dataset_overview,
)
from src.preprocessor import create_preprocessor, build_pipeline
from src.predictor import (
    predict_price,
    validate_input,
    format_inr,
    load_model,
)


class TestDataLoader:
    """Tests for dataset loading, cleaning, and train-test splitting."""

    def test_raw_data_loads(self):
        df = load_raw_data(DATA_PATH)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4000
        assert set(df["category"].unique()) == set(CATEGORIES)

    def test_dataset_overview(self):
        df = load_raw_data(DATA_PATH)
        overview = get_dataset_overview(df)
        assert overview["total_rows"] == 4000
        assert overview["duplicates"] == 0
        for cat in CATEGORIES:
            assert overview["category_counts"][cat] == 1000

    @pytest.mark.parametrize("category", CATEGORIES)
    def test_clean_category_data(self, category):
        df = load_raw_data(DATA_PATH)
        cleaned = clean_category_data(df, category)
        schema = CATEGORY_SCHEMAS[category]
        expected_cols = schema["numeric_features"] + schema["categorical_features"] + [schema["target"]]

        assert len(cleaned) == 1000
        assert list(cleaned.columns) == expected_cols
        assert cleaned.isnull().sum().sum() == 0
        assert (cleaned[schema["target"]] > 0).all()

    @pytest.mark.parametrize("category", CATEGORIES)
    def test_train_test_split_shapes(self, category):
        df = load_raw_data(DATA_PATH)
        X_train, X_test, y_train, y_test = get_train_test_data(df, category, test_size=0.2)

        assert len(X_train) == 800
        assert len(X_test) == 200
        assert len(y_train) == 800
        assert len(y_test) == 200
        assert not X_train.isnull().any().any()
        assert not X_test.isnull().any().any()


class TestPreprocessing:
    """Tests for column transformers and preprocessors."""

    @pytest.mark.parametrize("category", CATEGORIES)
    def test_preprocessor_fitting(self, category):
        df = load_raw_data(DATA_PATH)
        X_train, X_test, y_train, y_test = get_train_test_data(df, category)
        preprocessor = create_preprocessor(category)

        # Fit preprocessor on training data only (no data leakage)
        transformed_train = preprocessor.fit_transform(X_train)
        transformed_test = preprocessor.transform(X_test)

        assert transformed_train.shape[0] == 800
        assert transformed_test.shape[0] == 200
        assert transformed_train.shape[1] == transformed_test.shape[1]
        assert not np.isnan(transformed_train).any()
        assert not np.isnan(transformed_test).any()


class TestModelArtifacts:
    """Tests for trained model persistence and evaluation metrics integrity."""

    def test_saved_artifacts_exist(self):
        assert EVAL_METRICS_PATH.exists()
        assert METADATA_PATH.exists()
        for cat in CATEGORIES:
            best_model_file = MODELS_DIR / f"{cat.lower()}_best_model.joblib"
            assert best_model_file.exists()

    @pytest.mark.parametrize("category", CATEGORIES)
    def test_model_loading(self, category):
        model = load_model(category, "best")
        assert model is not None
        assert hasattr(model, "predict")

    def test_metrics_sanity(self):
        import json
        with open(EVAL_METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        for cat in CATEGORIES:
            assert cat in metrics
            for model_name in ["Linear Regression", "Decision Tree", "Random Forest", "Extra Trees"]:
                assert model_name in metrics[cat]
                m = metrics[cat][model_name]
                assert m["MAE"] > 0
                assert m["RMSE"] > 0
                assert m["R2"] <= 1.0


class TestPredictorAndValidation:
    """Tests for input validation and inference pipeline."""

    def test_laptop_prediction(self):
        payload = {
            "brand": "Dell",
            "model": "Model_57",
            "year": 2023,
            "condition": "Like New",
            "ram_gb": 16,
            "storage_gb": 512,
            "processor": "Intel i7",
            "gpu": "NVIDIA RTX",
            "screen_inches": 15.6,
        }
        res = predict_price("Laptop", payload)
        assert res["success"] is True
        assert res["predicted_price"] > 0
        assert "₹" in res["formatted_price"]

    def test_car_prediction(self):
        payload = {
            "brand": "Toyota",
            "model": "Model_10",
            "year": 2021,
            "condition": "Good",
            "mileage_km": 35000,
            "fuel_type": "Petrol",
        }
        res = predict_price("Car", payload)
        assert res["success"] is True
        assert res["predicted_price"] > 0

    def test_house_prediction(self):
        payload = {
            "year": 2015,
            "area_sqft": 2500,
            "bedrooms": 4,
            "bathrooms": 3,
        }
        res = predict_price("House", payload)
        assert res["success"] is True
        assert res["predicted_price"] > 0

    def test_furniture_prediction(self):
        payload = {
            "brand": "Urban Ladder",
            "model": "Sofa",
            "year": 2022,
            "condition": "Good",
            "material": "Wood",
        }
        res = predict_price("Furniture", payload)
        assert res["success"] is True
        assert res["predicted_price"] > 0

    def test_invalid_year_validation(self):
        payload = {"year": 1850, "area_sqft": 2000, "bedrooms": 3, "bathrooms": 2}
        is_valid, err = validate_input("House", payload)
        assert is_valid is False
        assert "Year must be between" in err

    def test_missing_field_validation(self):
        payload = {"brand": "Apple"}
        is_valid, err = validate_input("Laptop", payload)
        assert is_valid is False
        assert "Missing required" in err

    def test_format_inr(self):
        assert format_inr(15000000) == "₹ 1.50 Cr"
        assert format_inr(650000) == "₹ 6.50 Lakh"
        assert format_inr(45000) == "₹ 45,000.00"
        assert format_inr(-100) == "₹ 0.00"
