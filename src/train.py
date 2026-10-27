"""
Training and Evaluation Pipeline for PriceWise AI.
Trains and compares 4 regression algorithms separately for each category:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Extra Trees Regressor

Calculates MAE, RMSE, and R2 on the holdout test set (80:20 split, zero leakage).
Saves all models, evaluation metrics, and test predictions for UI rendering.
"""

import json
import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    DATA_PATH,
    MODELS_DIR,
    CATEGORIES,
    EVAL_METRICS_PATH,
    TEST_PREDICTIONS_PATH,
    METADATA_PATH,
    RANDOM_STATE,
)
from src.data_loader import load_raw_data, get_train_test_data
from src.preprocessor import build_pipeline, get_feature_names_from_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PriceWiseTrainer")


def get_model_instances() -> Dict[str, Any]:
    """
    Returns standard instances of the 4 required regression algorithms.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes regression evaluation metrics: MAE, RMSE, and R2.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
    }


def train_and_evaluate_all() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Executes full training lifecycle across all categories and all 4 models.
    Persists trained pipelines, metrics, and test predictions.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw_data(DATA_PATH)

    all_metrics: Dict[str, Any] = {}
    all_predictions: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {
        "trained_at": datetime.now().isoformat(),
        "categories": CATEGORIES,
        "best_models": {},
        "model_rankings": {},
        "is_synthetic_dataset": True,
        "disclosure": "PriceWise AI models are trained on synthetic benchmark data for academic and evaluation purposes. Real-world commercial transactions require verified appraiser inspection.",
    }

    for category in CATEGORIES:
        logger.info(f"--- Training models for category: {category} ---")
        X_train, X_test, y_train, y_test = get_train_test_data(df, category)

        category_metrics = {}
        category_predictions = {
            "y_test_actual": [round(float(v), 2) for v in y_test.tolist()]
        }

        models = get_model_instances()
        category_pipelines = {}

        best_model_name = None
        best_r2 = -float("inf")

        for model_name, model_instance in models.items():
            slug = model_name.lower().replace(" ", "_")
            logger.info(f"Fitting {model_name} for {category}...")

            pipeline = build_pipeline(category, model_instance)
            pipeline.fit(X_train, y_train)

            # Predict on unseen test set
            preds = pipeline.predict(X_test)
            metrics = evaluate_predictions(y_test.values, preds)

            logger.info(
                f"[{category}] {model_name} -> MAE: ₹{metrics['MAE']:,.2f} | RMSE: ₹{metrics['RMSE']:,.2f} | R2: {metrics['R2']:.4f}"
            )

            # Extract feature importance / coefficients if available
            feature_importance_dict = {}
            feature_names = get_feature_names_from_preprocessor(pipeline, category)

            regressor = pipeline.named_steps["regressor"]
            if hasattr(regressor, "feature_importances_"):
                importances = regressor.feature_importances_
                top_idx = np.argsort(importances)[::-1][:15]
                feature_importance_dict = {
                    feature_names[i]: round(float(importances[i]), 4)
                    for i in top_idx if i < len(feature_names)
                }
            elif hasattr(regressor, "coef_"):
                coefs = np.abs(regressor.coef_)
                top_idx = np.argsort(coefs)[::-1][:15]
                feature_importance_dict = {
                    feature_names[i]: round(float(coefs[i]), 4)
                    for i in top_idx if i < len(feature_names)
                }

            metrics["feature_importances"] = feature_importance_dict
            category_metrics[model_name] = metrics
            category_predictions[model_name] = [round(float(v), 2) for v in preds.tolist()]
            category_pipelines[model_name] = pipeline

            # Save individual pipeline
            pipe_filename = f"{category.lower()}_{slug}.joblib"
            joblib.dump(pipeline, MODELS_DIR / pipe_filename)

            # Check if this model is the best performer
            if metrics["R2"] > best_r2:
                best_r2 = metrics["R2"]
                best_model_name = model_name

        # Rank models by R2 score descending
        ranked_models = sorted(
            category_metrics.keys(),
            key=lambda m: category_metrics[m]["R2"],
            reverse=True
        )

        metadata["model_rankings"][category] = ranked_models
        metadata["best_models"][category] = {
            "model_name": best_model_name,
            "R2": category_metrics[best_model_name]["R2"],
            "MAE": category_metrics[best_model_name]["MAE"],
            "RMSE": category_metrics[best_model_name]["RMSE"],
            "model_file": f"{category.lower()}_best_model.joblib",
        }

        # Save designated best model for quick inference
        best_pipeline = category_pipelines[best_model_name]
        joblib.dump(best_pipeline, MODELS_DIR / f"{category.lower()}_best_model.joblib")
        logger.info(f"🏆 Best model for {category}: {best_model_name} (R2={best_r2:.4f}) saved.")

        all_metrics[category] = category_metrics
        all_predictions[category] = category_predictions

    # Save metrics and predictions to JSON
    with open(EVAL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    with open(TEST_PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, indent=2)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("All models trained, evaluated, and artifacts saved successfully!")
    return all_metrics, all_predictions, metadata


if __name__ == "__main__":
    train_and_evaluate_all()
