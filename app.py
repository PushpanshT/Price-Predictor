"""
PriceWise AI – Multi-Category Price Prediction System
A production-grade Machine Learning Streamlit web application.
Predicts prices in INR for Laptops, Cars, Houses, and Furniture using Scikit-Learn pipelines.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import streamlit as st

# Setup sys.path to find local modules
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    DATA_PATH,
    CATEGORIES,
    CATEGORY_SCHEMAS,
    EVAL_METRICS_PATH,
    TEST_PREDICTIONS_PATH,
    METADATA_PATH,
    DATASET_BASE_YEAR,
)
from src.data_loader import load_raw_data, clean_category_data, get_dataset_overview, get_category_uniques
from src.predictor import predict_price, format_inr
from src.visualizer import (
    plot_model_comparison,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_feature_importance,
    plot_price_distribution,
    plot_correlation_heatmap,
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="PriceWise AI – Multi-Category Price Prediction",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Styling (CSS)
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main container spacing */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 28px 32px;
        color: #F8FAFC;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 6px;
        background: linear-gradient(90deg, #60A5FA, #A78BFA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 1.02rem;
        color: #94A3B8;
        font-weight: 400;
        max-width: 850px;
        line-height: 1.5;
    }

    /* Card styling */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }

    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #0F172A;
    }

    .metric-badge {
        display: inline-block;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-top: 6px;
    }

    .badge-success {
        background-color: #DCFCE7;
        color: #166534;
    }

    .badge-primary {
        background-color: #DBEAFE;
        color: #1E40AF;
    }

    /* Result Prediction Box */
    .prediction-box {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 2px solid #86EFAC;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-top: 15px;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.12);
    }

    .pred-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: #166534;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .pred-price {
        font-size: 2.5rem;
        font-weight: 800;
        color: #14532D;
        margin: 8px 0;
    }

    .pred-range {
        font-size: 0.95rem;
        color: #15803D;
        font-weight: 500;
    }

    /* Disclosure Banner */
    .disclosure-banner {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        color: #92400E;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 0.88rem;
        line-height: 1.45;
        margin-bottom: 20px;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Helper Functions with Streamlit Cache
# ---------------------------------------------------------
@st.cache_data
def get_cached_raw_data():
    return load_raw_data(DATA_PATH)


@st.cache_data
def get_cached_metrics():
    if EVAL_METRICS_PATH.exists():
        with open(EVAL_METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def get_cached_predictions():
    if TEST_PREDICTIONS_PATH.exists():
        with open(TEST_PREDICTIONS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def get_cached_metadata():
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# Load data assets
raw_df = get_cached_raw_data()
eval_metrics = get_cached_metrics()
test_predictions = get_cached_predictions()
metadata = get_cached_metadata()

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=400&q=80",
        caption="PriceWise AI Valuator",
        width="stretch",
    )
    st.markdown("### 🏷️ PriceWise AI Settings")

    selected_category = st.selectbox(
        "Select Valuation Category",
        options=CATEGORIES,
        index=0,
        format_func=lambda c: f"{CATEGORY_SCHEMAS[c]['icon']} {c}",
        help="Choose the asset category to predict or analyze."
    )

    st.markdown("---")
    st.markdown("#### ⚙️ Inference Model")

    # Available models for selected category
    available_models = ["🏆 Best Model (Recommended)", "Linear Regression", "Decision Tree", "Random Forest", "Extra Trees"]
    selected_model_choice = st.selectbox(
        "Choose Algorithm",
        options=available_models,
        index=0,
        help="Select which algorithm to use for price estimation."
    )

    model_type_key = "best"
    if selected_model_choice != "🏆 Best Model (Recommended)":
        model_type_key = selected_model_choice.lower().replace(" ", "_")

    st.markdown("---")
    st.markdown("#### 📌 Category Description")
    schema_info = CATEGORY_SCHEMAS[selected_category]
    st.info(f"{schema_info['icon']} **{selected_category}**: {schema_info['description']}")

    st.markdown("---")
    st.caption("PriceWise AI v1.0.0 • Scikit-Learn • Streamlit")


# ---------------------------------------------------------
# Header & Hero Banner
# ---------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-container">
        <div class="hero-title">🏷️ PriceWise AI – Multi-Category Price Prediction</div>
        <div class="hero-subtitle">
            Enterprise-grade machine learning valuation engine for <b>Laptops, Cars, Houses, and Furniture</b>.
            Trained and benchmarked across 4 Scikit-Learn regression algorithms with zero data leakage.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Mandatory Synthetic Dataset Disclosure
st.markdown(
    """
    <div class="disclosure-banner">
        ⚠️ <b>Dataset Disclosure & Academic Disclaimer:</b>
        The dataset utilized by PriceWise AI (<code>pricewise_combined_dataset.csv</code>) is <b>synthetic benchmark data</b>.
        All displayed evaluation metrics (MAE, RMSE, R²) represent <b>real, holdout test-set results</b> calculated on an 80:20 split.
        Results are intended for academic practicals, portfolio demonstrations, and methodology exploration.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tab_predict, tab_analytics, tab_explorer, tab_report = st.tabs([
    "🔮 Price Predictor",
    "📊 Analytics & Model Comparison",
    "🔍 Dataset Explorer & EDA",
    "📑 Project Report & Methodology",
])


# =========================================================
# TAB 1: PRICE PREDICTOR
# =========================================================
with tab_predict:
    st.markdown(f"### {schema_info['icon']} Predict {selected_category} Valuation")
    st.caption(f"Adjust feature inputs below to compute instant estimated market price in INR.")

    col_form, col_result = st.columns([1.1, 0.9], gap="large")

    uniques = get_category_uniques(raw_df, selected_category)

    user_input = {}

    with col_form:
        st.markdown("#### 📝 Specifications Form")

        # Category-specific dynamic input widgets
        if selected_category == "Laptop":
            c1, c2 = st.columns(2)
            with c1:
                user_input["brand"] = st.selectbox("Brand", options=uniques.get("brand", ["Dell", "HP", "Apple", "Lenovo", "Acer", "Asus", "MSI"]))
                user_input["processor"] = st.selectbox("Processor", options=uniques.get("processor", ["Intel i7", "Intel i5", "AMD Ryzen 7", "Apple M2"]))
                user_input["ram_gb"] = st.select_slider("RAM (GB)", options=[4, 8, 16, 32, 64], value=16)
                user_input["year"] = st.slider("Manufacture Year", min_value=2018, max_value=2026, value=2023)

            with c2:
                user_input["model"] = st.selectbox("Model Series", options=uniques.get("model", ["Model_1", "Model_50", "Model_57"])[:20])
                user_input["gpu"] = st.selectbox("Graphics Card (GPU)", options=uniques.get("gpu", ["Integrated", "NVIDIA GTX", "NVIDIA RTX", "AMD Radeon"]))
                user_input["storage_gb"] = st.select_slider("Storage (GB)", options=[256, 512, 1024, 2048], value=512)
                user_input["screen_inches"] = st.selectbox("Display Size (Inches)", options=[13.3, 14.0, 15.6, 16.0, 17.3], index=2)

            user_input["condition"] = st.segmented_control("Condition", options=["Fair", "Good", "Like New", "New"], default="Good")
            if not user_input["condition"]:
                user_input["condition"] = "Good"

        elif selected_category == "Car":
            c1, c2 = st.columns(2)
            with c1:
                user_input["brand"] = st.selectbox("Car Brand", options=uniques.get("brand", ["Maruti Suzuki", "Hyundai", "Tata", "Mahindra", "Toyota", "Honda", "Kia"]))
                user_input["year"] = st.slider("Registration Year", min_value=2008, max_value=2026, value=2019)
                user_input["fuel_type"] = st.selectbox("Fuel Type", options=uniques.get("fuel_type", ["Petrol", "Diesel", "CNG", "Electric"]))

            with c2:
                user_input["model"] = st.selectbox("Model Code", options=uniques.get("model", ["Model_1", "Model_10", "Model_50"])[:25])
                user_input["mileage_km"] = st.number_input("Odometer Mileage (km)", min_value=1000, max_value=300000, value=45000, step=2000)
                user_input["condition"] = st.selectbox("Overall Vehicle Condition", options=["Fair", "Good", "Like New", "New"], index=1)

        elif selected_category == "House":
            c1, c2 = st.columns(2)
            with c1:
                user_input["area_sqft"] = st.number_input("Super Built-up Area (sq. ft.)", min_value=350, max_value=10000, value=1850, step=50)
                user_input["year"] = st.slider("Construction Year", min_value=1986, max_value=2026, value=2018)

            with c2:
                user_input["bedrooms"] = st.slider("Bedrooms (BHK)", min_value=1, max_value=6, value=3)
                user_input["bathrooms"] = st.slider("Bathrooms", min_value=1, max_value=5, value=2)

        elif selected_category == "Furniture":
            c1, c2 = st.columns(2)
            with c1:
                user_input["brand"] = st.selectbox("Brand / Manufacturer", options=uniques.get("brand", ["Urban Ladder", "Godrej", "IKEA", "Nilkamal", "Local"]))
                user_input["model"] = st.selectbox("Furniture Type", options=uniques.get("model", ["Sofa", "Bed", "Dining Table", "Desk", "Chair", "Wardrobe"]))

            with c2:
                user_input["material"] = st.selectbox("Primary Material", options=uniques.get("material", ["Wood", "Engineered Wood", "Metal", "Fabric", "Plastic"]))
                user_input["year"] = st.slider("Purchase Year", min_value=2011, max_value=2026, value=2022)

            user_input["condition"] = st.segmented_control("Condition", options=["Fair", "Good", "Like New", "New"], default="Good")
            if not user_input["condition"]:
                user_input["condition"] = "Good"

        predict_button = st.button("🚀 Calculate Estimated Price", type="primary", width="stretch")

    with col_result:
        st.markdown("#### 💎 Valuation Result")

        # Automatically execute prediction on form submit or initial view
        prediction_result = predict_price(selected_category, user_input, model_type=model_type_key)

        if prediction_result["success"]:
            price_val = prediction_result["predicted_price"]
            formatted_p = prediction_result["formatted_price"]
            formatted_r = prediction_result["formatted_range"]
            used_model = selected_model_choice

            # Get benchmark median price for comparison
            cat_median = float(raw_df[raw_df["category"] == selected_category]["price_inr"].median())
            diff_pct = ((price_val - cat_median) / cat_median) * 100
            diff_label = f"{abs(diff_pct):.1f}% {'above' if diff_pct >= 0 else 'below'} category median"

            st.markdown(
                f"""
                <div class="prediction-box">
                    <div class="pred-label">Estimated Valuation ({selected_category})</div>
                    <div class="pred-price">{formatted_p}</div>
                    <div class="pred-range">Estimated Range: <b>{formatted_r}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # Context metric badges
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Algorithm</div>
                        <div style="font-weight:700; font-size:1.05rem; color:#1E3A8A;">{used_model}</div>
                        <span class="metric-badge badge-primary">Zero Leakage</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m2:
                badge_class = "badge-success" if diff_pct >= 0 else "badge-primary"
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Market Benchmark</div>
                        <div style="font-weight:700; font-size:1.05rem; color:#0F172A;">₹ {cat_median:,.0f}</div>
                        <span class="metric-badge {badge_class}">{diff_label}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("🔍 View Normalized Input Payload", expanded=False):
                st.json(prediction_result["features_used"])

        else:
            st.error(f"❌ Valuation Error: {prediction_result.get('error', 'Unknown issue')}")


# =========================================================
# TAB 2: ANALYTICS & MODEL COMPARISON
# =========================================================
with tab_analytics:
    st.markdown(f"### 📊 Machine Learning Performance – {selected_category}")
    st.caption("Empirical evaluation metrics computed on the holdout test set (80:20 split, random_state=42).")

    cat_metrics = eval_metrics.get(selected_category, {})
    cat_preds = test_predictions.get(selected_category, {})

    if cat_metrics:
        # Best model summary KPIs
        best_name = metadata.get("best_models", {}).get(selected_category, {}).get("model_name", "Linear Regression")
        best_m = cat_metrics.get(best_name, {})

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">🏆 Top Performer</div>
                    <div style="font-size:1.2rem; font-weight:800; color:#1E40AF;">{best_name}</div>
                    <span class="metric-badge badge-success">Selected for Inference</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">R² Score (Holdout)</div>
                    <div class="metric-value">{best_m.get('R2', 0):.4f}</div>
                    <span class="metric-badge badge-primary">Variance Explained</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k3:
            mae_disp = format_inr(best_m.get('MAE', 0))
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Mean Absolute Error</div>
                    <div class="metric-value" style="font-size:1.35rem;">{mae_disp}</div>
                    <span class="metric-badge badge-primary">Avg Deviation</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k4:
            rmse_disp = format_inr(best_m.get('RMSE', 0))
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Root Mean Sq. Error</div>
                    <div class="metric-value" style="font-size:1.35rem;">{rmse_disp}</div>
                    <span class="metric-badge badge-primary">Penalizes Outliers</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Full Comparison Table
        st.markdown("#### 📋 4-Model Comparison Scorecard")
        scorecard_rows = []
        for m_name, m_vals in cat_metrics.items():
            is_best = "🏆 Best" if m_name == best_name else "Standard"
            scorecard_rows.append({
                "Model Algorithm": m_name,
                "Status": is_best,
                "R² Score": m_vals["R2"],
                "MAE (INR)": f"₹ {m_vals['MAE']:,.2f}",
                "RMSE (INR)": f"₹ {m_vals['RMSE']:,.2f}",
            })
        score_df = pd.DataFrame(scorecard_rows).sort_values(by="R² Score", ascending=False)
        st.dataframe(score_df, hide_index=True, width="stretch")

        st.markdown("---")

        # Comparison Bar Charts
        st.markdown("#### 📈 Benchmark Visualizations Across Algorithms")
        fig_comp = plot_model_comparison(cat_metrics, selected_category)
        st.pyplot(fig_comp)

        st.markdown("---")

        # Actual vs Predicted and Residuals Plots
        st.markdown(f"#### 🎯 Actual vs Predicted Prices on Holdout Test Set")
        st.caption("Comparing real test target values against model predictions. Points closer to the dashed red line indicate higher accuracy.")

        chart_model_choice = st.selectbox(
            "Select Model for Detailed Diagnostic Plots",
            options=list(cat_metrics.keys()),
            index=list(cat_metrics.keys()).index(best_name) if best_name in cat_metrics else 0
        )

        y_test_actual = cat_preds.get("y_test_actual", [])
        y_test_pred = cat_preds.get(chart_model_choice, [])

        if y_test_actual and y_test_pred:
            c_diag1, c_diag2 = st.columns([1, 1])
            with c_diag1:
                fig_scatter = plot_actual_vs_predicted(y_test_actual, y_test_pred, chart_model_choice, selected_category)
                st.pyplot(fig_scatter)

            with c_diag2:
                # Top Feature Drivers
                feat_importances = cat_metrics.get(chart_model_choice, {}).get("feature_importances", {})
                fig_feat = plot_feature_importance(feat_importances, chart_model_choice, selected_category)
                st.pyplot(fig_feat)

            st.markdown("#### 📉 Residuals and Error Dispersion")
            fig_res = plot_residuals(y_test_actual, y_test_pred, chart_model_choice, selected_category)
            st.pyplot(fig_res)

    else:
        st.warning("No evaluation metrics found. Please execute `python src/train.py`.")


# =========================================================
# TAB 3: DATASET EXPLORER & EDA
# =========================================================
with tab_explorer:
    st.markdown(f"### 🔍 Dataset Explorer – {selected_category}")
    cat_df = clean_category_data(raw_df, selected_category)

    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.metric("Total Dataset Records", f"{len(raw_df):,}")
    with e2:
        st.metric(f"{selected_category} Records", f"{len(cat_df):,}")
    with e3:
        st.metric("Missing Values", f"{cat_df.isnull().sum().sum()}")
    with e4:
        st.metric("Duplicates in Category", f"{cat_df.duplicated().sum()}")

    st.markdown("---")

    # Interactive Table with Search & Filter
    st.markdown("#### 📄 Category Data Sample")
    c_f1, c_f2 = st.columns([1, 2])
    with c_f1:
        rows_to_show = st.slider("Rows to Preview", min_value=5, max_value=100, value=15, step=5)
    with c_f2:
        search_query = st.text_input("Filter Data by Keyword (e.g. Brand, Material, Model)", "")

    display_df = cat_df.copy()
    if search_query:
        # Search across all columns
        mask = display_df.astype(str).apply(lambda row: row.str.contains(search_query, case=False).any(), axis=1)
        display_df = display_df[mask]

    st.dataframe(display_df.head(rows_to_show), width="stretch")

    # Download Button
    csv_bytes = cat_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download Cleaned {selected_category} Dataset (CSV)",
        data=csv_bytes,
        file_name=f"pricewise_{selected_category.lower()}_cleaned.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # Exploratory Data Analysis Charts
    st.markdown(f"#### 📊 Exploratory Price Distribution ({selected_category})")
    fig_dist = plot_price_distribution(cat_df["price_inr"], selected_category)
    st.pyplot(fig_dist)

    st.markdown("---")
    st.markdown(f"#### 🔗 Feature Correlation Heatmap")
    fig_corr = plot_correlation_heatmap(cat_df, selected_category)
    st.pyplot(fig_corr)

    with st.expander("📊 Numerical Descriptive Statistics (`describe()`)"):
        st.dataframe(cat_df.describe().T, width="stretch")


# =========================================================
# TAB 4: PROJECT REPORT & METHODOLOGY
# =========================================================
with tab_report:
    st.markdown("### 📑 PriceWise AI Technical Documentation & Project Report")

    st.markdown(
        """
        #### 1. Executive Summary & Objective
        **PriceWise AI** is an end-to-end Machine Learning valuation system capable of predicting accurate market prices in Indian Rupees (INR) across four diverse consumer and commercial asset domains:
        - 💻 **Laptops**: Technology specs including RAM, solid-state storage, processor tier, GPU, and display dimensions.
        - 🚗 **Pre-Owned Cars**: Automotive features including odometer mileage, fuel type, vehicle age, and physical condition.
        - 🏡 **Residential Real Estate**: Built-up square footage, bedroom count, bathroom count, and construction year.
        - 🛋️ **Furniture**: Material composition, furniture form-factor, brand origin, and physical wear.

        ---

        #### 2. Machine Learning Architecture & Anti-Leakage Pipeline
        To guarantee enterprise-grade machine learning rigor and prevent data leakage:
        1. **Category Isolation**: The combined multi-category dataset is sliced per category into specialized feature spaces.
        2. **Strict Train-Test Split**: An 80:20 split (`random_state=42`) is applied to raw features *before* computing normalization parameters or encoding frequencies.
        3. **Scikit-Learn ColumnTransformer**:
           - **Numeric Features**: Handled via `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
           - **Categorical Features**: Handled via `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
        4. **Pipeline Encapsulation**: Estimators are directly bound to transformers within Scikit-Learn `Pipeline` objects, ensuring test and production inference inputs undergo identical mathematical transformations.

        ---

        #### 3. Evaluated Algorithms & Mathematical Metrics
        Each category is independently trained and benchmarked across four fundamental regression algorithms:
        - **Linear Regression**: Ordinary Least Squares baseline with feature coefficient interpretability.
        - **Decision Tree Regressor**: Non-linear recursive partitioning tree model.
        - **Random Forest Regressor**: Ensemble bagging technique over decision trees reducing variance.
        - **Extra Trees Regressor**: Extremely Randomized Trees introducing random threshold cutoffs for enhanced regularization.

        **Evaluation Metrics**:
        - **Mean Absolute Error (MAE)**:
          $$\\text{MAE} = \\frac{1}{n} \\sum_{i=1}^{n} |y_i - \\hat{y}_i|$$
        - **Root Mean Squared Error (RMSE)**:
          $$\\text{RMSE} = \\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2}$$
        - **Coefficient of Determination ($R^2$)**:
          $$R^2 = 1 - \\frac{\\sum (y_i - \\hat{y}_i)^2}{\\sum (y_i - \\bar{y})^2}$$

        ---

        #### 4. Empirical Evaluation Findings
        - **Laptops**: Linear Regression attained $R^2 = 0.9019$ ($MAE = ₹4,456.41$), demonstrating strong additive pricing relationships across RAM and storage capacity.
        - **Cars**: Linear Regression led with $R^2 = 0.8203$ ($MAE = ₹92,670.30$), with Extra Trees closely following at $R^2 = 0.7873$.
        - **Furniture**: Ensemble trees dominated, with Random Forest achieving $R^2 = 0.8997$ and Extra Trees at $R^2 = 0.8806$, capturing combinatorial interaction effects between materials and item types.
        - **Houses**: Linear Regression achieved $R^2 = 0.4639$, with square footage being the primary driver. Decision trees exhibited overfitting when unrestricted, highlighting the importance of tree depth regularization.

        ---

        #### 5. Deployment Guide (Streamlit Community Cloud)
        Deploying PriceWise AI to Streamlit Community Cloud takes less than 3 minutes:
        1. Push the project repository to GitHub:
           ```bash
           git init
           git add .
           git commit -m "Initial commit: PriceWise AI"
           git remote add origin https://github.com/your-username/pricewise-ai.git
           git push -u origin main
           ```
        2. Visit [share.streamlit.io](https://share.streamlit.io) and authenticate with GitHub.
        3. Click **"New App"**, select the repository, choose `main` branch, and set `app.py` as the main file path.
        4. Streamlit Community Cloud will automatically detect `requirements.txt`, install dependencies, and launch the web app with a public HTTPS URL.
        """
    )
