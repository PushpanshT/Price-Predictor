# 🏷️ PriceWise AI – Multi-Category Price Prediction System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(27%2F27)-brightgreen.svg)]()

> **PriceWise AI** is an end-to-end Machine Learning web application designed to predict and benchmark valuations in Indian Rupees (**INR ₹**) across four distinct consumer and commercial categories: **Laptops, Pre-Owned Cars, Residential Houses, and Furniture**.

Built with **Python, Pandas, Scikit-learn, and Streamlit**, the project follows industry-standard machine learning practices: rigorous data cleaning, leak-free train-test splits (80:20), Scikit-Learn `ColumnTransformer` and `Pipeline` encapsulation, and empirical multi-model benchmarking.

---

## 📌 Key Highlights

- **4 Asset Categories Supported**:
  - 💻 **Laptops**: RAM, Storage, CPU, GPU, Display size, Brand, Model series.
  - 🚗 **Cars**: Odometer mileage, Fuel type (Petrol/Diesel/CNG/Electric), Year, Condition.
  - 🏡 **Houses**: Built-up area (sq. ft.), Bedroom count (BHK), Bathrooms, Construction year.
  - 🛋️ **Furniture**: Material (Wood/Metal/Fabric/Plastic), Item form (Sofa, Bed, Desk, etc.), Brand, Condition.
- **4 ML Regression Algorithms Evaluated per Category**:
  - Linear Regression (Ordinary Least Squares)
  - Decision Tree Regressor
  - Random Forest Regressor (Ensemble Bagging)
  - Extra Trees Regressor (Extremely Randomized Trees)
- **Zero Data Leakage Architecture**:
  - `StandardScaler` and `OneHotEncoder(handle_unknown='ignore')` fitted strictly on training folds via Scikit-Learn `Pipeline`.
- **Real Holdout Evaluation Metrics**:
  - Real Mean Absolute Error (**MAE**), Root Mean Squared Error (**RMSE**), and Coefficient of Determination (**R²**).
  - Diagnostic charts: Actual vs. Predicted scatter plot with $45^\circ$ diagonal line, residual error distributions, and top feature importance bars.
- **Modern Responsive Streamlit UI**:
  - Dynamic category selection and specification forms.
  - Instant valuation badge in Lakhs / Crores / Thousands with confidence intervals.
  - Interactive Dataset Explorer with keyword filtering, distribution histograms, and CSV exporter.
  - Embedded Project Report and documentation tab.
- **Quality Assurance**:
  - Input validation, boundary error handling, and a full `pytest` test suite (27 unit tests).

---

## 🏆 Model Benchmark & Evaluation Results

All metrics below are computed directly on the holdout test set (80:20 split, `random_state=42`) from `pricewise_combined_dataset.csv`.

| Category | Algorithm | $R^2$ Score | MAE (INR) | RMSE (INR) | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: |
| 💻 **Laptop** | **Linear Regression** | **0.9019** | **₹ 4,456.41** | **₹ 5,785.60** | 🏆 **Best Performer** |
| 💻 Laptop | Random Forest | 0.8422 | ₹ 5,360.77 | ₹ 7,339.02 | Strong non-linear fit |
| 💻 Laptop | Extra Trees | 0.8085 | ₹ 6,236.39 | ₹ 8,084.34 | Regularized ensemble |
| 💻 Laptop | Decision Tree | 0.6456 | ₹ 8,175.43 | ₹ 10,998.44 | High tree variance |
| 🚗 **Car** | **Linear Regression** | **0.8203** | **₹ 92,670.30** | **₹ 1,17,882.92** | 🏆 **Best Performer** |
| 🚗 Car | Extra Trees | 0.7873 | ₹ 92,276.59 | ₹ 1,28,243.11 | High generalization |
| 🚗 Car | Random Forest | 0.7805 | ₹ 96,366.08 | ₹ 1,30,287.79 | Robust bagging |
| 🚗 Car | Decision Tree | 0.6087 | ₹ 1,29,034.19 | ₹ 1,73,949.53 | Overfits on small splits |
| 🏡 **House** | **Linear Regression** | **0.4639** | **₹ 60,29,566.74** | **₹ 92,40,984.16** | 🏆 **Best Performer** |
| 🏡 House | Extra Trees | 0.2801 | ₹ 70,90,408.21 | ₹ 1,07,08,719.21 | Moderate regularization |
| 🏡 House | Random Forest | 0.2756 | ₹ 71,92,626.88 | ₹ 1,07,41,718.87 | Moderate generalization |
| 🏡 House | Decision Tree | -0.2945 | ₹ 92,83,765.53 | ₹ 1,43,59,439.23 | Severe tree overfitting |
| 🛋️ **Furniture** | **Random Forest** | **0.8997** | **₹ 749.79** | **₹ 1,211.93** | 🏆 **Best Performer** |
| 🛋️ Furniture | Extra Trees | 0.8806 | ₹ 802.89 | ₹ 1,322.09 | Close second |
| 🛋️ Furniture | Linear Regression | 0.8388 | ₹ 1,163.36 | ₹ 1,535.96 | Good linear baseline |
| 🛋️ Furniture | Decision Tree | 0.8233 | ₹ 1,048.92 | ₹ 1,608.33 | High variance |

> ⚠️ **Synthetic Dataset Disclosure**: The dataset (`pricewise_combined_dataset.csv`) is synthetic benchmark data. Results demonstrate machine learning modeling workflows, comparative algorithm dynamics, and UI integration. Real-world appraisals require certified inspections and market-specific data feeds.

---

## 📂 Project Structure

```
Price prediction/
├── app.py                         # Streamlit multi-tab web application
├── requirements.txt               # Pinned project dependencies
├── README.md                      # GitHub documentation & quickstart
├── PROJECT_REPORT.md              # Academic & practical report
├── pricewise_combined_dataset.csv # 4,000-row combined benchmark dataset
├── src/
│   ├── __init__.py
│   ├── config.py                  # Paths, constants, category schemas
│   ├── data_loader.py             # Data loading, cleaning, 80:20 splitting
│   ├── preprocessor.py            # ColumnTransformer & Pipeline definitions
│   ├── train.py                   # Automated training, evaluation, artifact saving
│   ├── predictor.py               # Robust inference engine & INR formatting
│   └── visualizer.py              # Matplotlib & Seaborn diagnostic plots
├── models/
│   ├── laptop_best_model.joblib   # Persisted best model pipelines
│   ├── car_best_model.joblib
│   ├── house_best_model.joblib
│   ├── furniture_best_model.joblib
│   ├── evaluation_metrics.json    # Genuine holdout test metrics
│   ├── test_predictions.json     # Ground truth & model predictions
│   └── metadata.json              # Model rankings & metadata
└── tests/
    ├── __init__.py
    └── test_pipeline.py           # 27 automated pytest test cases
```

---

## 🚀 Quickstart & Local Installation

### 1. Clone or Open the Repository
```bash
git clone https://github.com/your-username/pricewise-ai.git
cd pricewise-ai
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Retrain All Models
All 16 models and metrics are pre-trained and saved in `models/`. To retrain:
```bash
python src/train.py
```

### 5. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Unit Tests

PriceWise AI includes comprehensive unit tests verifying data integrity, pipeline transformations, model serialization, and edge-case inputs:

```bash
pytest -v
```

Output:
```
tests/test_pipeline.py::TestDataLoader::test_raw_data_loads PASSED
tests/test_pipeline.py::TestDataLoader::test_clean_category_data[Laptop] PASSED
...
tests/test_pipeline.py::TestPredictorAndValidation::test_format_inr PASSED
============================== 27 passed in 1.96s ==============================
```

---

## ☁️ Deployment Instructions (Streamlit Community Cloud)

Follow these steps to deploy PriceWise AI for free on Streamlit Community Cloud:

1. **Commit and Push Code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "feat: complete PriceWise AI system"
   git branch -M main
   git remote add origin https://github.com/<your-github-username>/PriceWise-AI.git
   git push -u origin main
   ```
2. **Access Streamlit Community Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. **Deploy New App**:
   - Click **"Create app"** (or **"New app"**).
   - Select your repository: `<your-github-username>/PriceWise-AI`.
   - Select the branch: `main`.
   - Set the Main file path to: `app.py`.
   - Click **"Deploy!"**.
4. **Live URL**:
   Streamlit Cloud automatically installs packages from `requirements.txt` and renders your web app with a publicly shareable HTTPS link.

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
