# 📑 PriceWise AI: Multi-Category Price Prediction System
## Comprehensive Machine Learning Project & Lab Practical Report

---

### **Metadata & Submission Details**
- **Project Title**: PriceWise AI – Multi-Category Price Prediction System
- **Domain**: Supervised Machine Learning (Regression Analysis)
- **Tech Stack**: Python 3.13, Pandas, Scikit-learn, NumPy, Matplotlib, Seaborn, Streamlit, Pytest
- **Target Variable**: `price_inr` (Market Price in Indian Rupees)
- **Asset Categories**: Laptops, Pre-Owned Cars, Residential Houses, Furniture

---

## 1. Executive Summary & Problem Formulation

Accurate market valuation of commercial and consumer assets is a core problem across multiple industries including e-commerce, consumer finance, automotive trade, and real estate. Traditional appraisal techniques often rely on subjective rules of thumb or fragmented manual inspections that fail to scale effectively across distinct product types.

**PriceWise AI** addresses this challenge by designing, engineering, and deploying an end-to-end multi-category price prediction system. Given a heterogeneous dataset (`pricewise_combined_dataset.csv`) comprising 4,000 observations across four distinct asset classes—**Laptops, Cars, Houses, and Furniture**—this project establishes a modular machine learning pipeline capable of:
1. Performing category-specific feature extraction and cleaning without cross-category contamination.
2. Preventing data leakage by utilizing an 80:20 train-test split before fitting normalization or encoding parameters.
3. Benchmarking four fundamental regression algorithms (**Linear Regression, Decision Tree Regressor, Random Forest Regressor, and Extra Trees Regressor**) separately for every category.
4. Measuring performance using real holdout metrics: Mean Absolute Error (**MAE**), Root Mean Squared Error (**RMSE**), and Coefficient of Determination (**$R^2$**).
5. Deploying an interactive, responsive Streamlit dashboard equipped with dynamic input forms, diagnostic plots, and an exploratory data analytics workbench.

---

## 2. Dataset Architecture & Statistical Exploration

The provided dataset `pricewise_combined_dataset.csv` contains **4,000 rows and 18 columns**, evenly partitioned into 1,000 instances per category:

| Category | Sample Size | Primary Features | Target Range (INR) | Median (INR) |
| :--- | :---: | :--- | :--- | :--- |
| **Laptop** | 1,000 | RAM (4–64GB), Storage (256–2048GB), CPU, GPU, Display, Brand, Year | ₹ 10,715 – ₹ 1,13,140 | ₹ 39,244 |
| **Car** | 1,000 | Mileage (1.4k–220k km), Fuel Type, Brand, Model, Condition, Year | ₹ 1,86,849 – ₹ 19,82,780 | ₹ 6,69,477 |
| **House** | 1,000 | Area (352–4,000 sq.ft.), Bedrooms (1–6), Bathrooms (1–5), Year Built | ₹ 20,31,867 – ₹ 8,65,67,230 | ₹ 1,54,92,570 |
| **Furniture** | 1,000 | Material, Furniture Form, Brand, Condition, Year Purchased | ₹ 226 – ₹ 22,364 | ₹ 5,233 |

### Observations:
- **Feature Sparsity by Design**: Because all categories reside within a unified CSV, features outside an active category are represented as `NaN` (e.g., `mileage_km` is null for Laptops and Houses).
- **Exact Multicollinearity**: The dataset features `product_age_years` and `year` satisfying `year + product_age_years = 2026`.
- **Zero Raw Duplicates**: Checking `df.duplicated()` over all columns reveals 0 duplicate rows.

---

## 3. Data Cleaning & Anti-Leakage Preprocessing Pipeline

Data leakage occurs when information from outside the training dataset is used to train the model, resulting in overly optimistic evaluation scores that fail in production. PriceWise AI enforces a zero-leakage protocol:

```
[Raw Dataset]
     │
     ▼
[Category Slicer] ──> Extracts category-specific valid columns
     │
     ▼
[Train-Test Split] ──> 80% Training Fold / 20% Holdout Fold (random_state=42)
     │
     ├─── Training Data (800 rows) ──> Fitted to ColumnTransformer
     │                                    ├── SimpleImputer(median) + StandardScaler
     │                                    └── SimpleImputer(most_frequent) + OneHotEncoder(ignore)
     │
     └─── Test Data (200 rows) ──────> Transformed using strictly fitted parameters
```

### Preprocessing Specifications:
1. **Numerical Pipeline**:
   - `SimpleImputer(strategy='median')`: Ensures missing values in new test instances are resolved using training median.
   - `StandardScaler()`: Zero-mean, unit-variance standardization:
     $$z = \frac{x - \mu}{\sigma}$$
2. **Categorical Pipeline**:
   - `SimpleImputer(strategy='most_frequent')`: Handles missing labels.
   - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`: Encodes categorical variables into orthogonal binary indicator vectors while gracefully accommodating unseen levels during inference.

---

## 4. Evaluated Machine Learning Algorithms

Four diverse regressors representing parametric, single-tree non-parametric, and ensemble methods were selected:

1. **Linear Regression (OLS)**:
   Models price as a linear combination of input features:
   $$\hat{y} = \beta_0 + \sum_{j=1}^{p} \beta_j X_j$$
   Solves for optimal weights via closed-form Ordinary Least Squares. Provides high interpretability and strong performance when underlying valuation follows additive pricing schedules.

2. **Decision Tree Regressor (CART)**:
   Partitions the feature space into hyper-rectangles using recursive binary splitting to minimize Mean Squared Error. Capable of learning non-linear step functions, but susceptible to high variance and overfitting on smaller samples if left unpruned.

3. **Random Forest Regressor**:
   An ensemble bagging technique that constructs 100 decorrelated decision trees using bootstrap aggregation (bagging) and random feature subspace sampling:
   $$\hat{y}_{\text{RF}} = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$
   Significantly reduces variance compared to individual decision trees.

4. **Extra Trees Regressor (Extremely Randomized Trees)**:
   Similar to Random Forest, but selects random thresholds for each candidate feature at every node split instead of searching for the optimal cut-point. This induces additional variance reduction and computational speed.

---

## 5. Mathematical Evaluation Metrics

Model performance was assessed on the holdout test set using three complementary metrics:

1. **Mean Absolute Error (MAE)**:
   Measures average absolute magnitude of errors in raw INR currency units:
   $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

2. **Root Mean Squared Error (RMSE)**:
   Penalizes large outlier discrepancies heavily:
   $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

3. **Coefficient of Determination ($R^2$)**:
   Quantifies the proportion of target variance explained by the model:
   $$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$

---

## 6. Empirical Benchmark Results & Discussion

### Full 16-Model Experimental Matrix

| Category | Model | $R^2$ Score | MAE (INR) | RMSE (INR) | Best Performer? |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Laptop** | **Linear Regression** | **0.9019** | **₹ 4,456.41** | **₹ 5,785.60** | 🏆 **Yes** |
| Laptop | Random Forest | 0.8422 | ₹ 5,360.77 | ₹ 7,339.02 | No |
| Laptop | Extra Trees | 0.8085 | ₹ 6,236.39 | ₹ 8,084.34 | No |
| Laptop | Decision Tree | 0.6456 | ₹ 8,175.43 | ₹ 10,998.44 | No |
| **Car** | **Linear Regression** | **0.8203** | **₹ 92,670.30** | **₹ 1,17,882.92** | 🏆 **Yes** |
| Car | Extra Trees | 0.7873 | ₹ 92,276.59 | ₹ 1,28,243.11 | No |
| Car | Random Forest | 0.7805 | ₹ 96,366.08 | ₹ 1,30,287.79 | No |
| Car | Decision Tree | 0.6087 | ₹ 1,29,034.19 | ₹ 1,73,949.53 | No |
| **House** | **Linear Regression** | **0.4639** | **₹ 60,29,566.74** | **₹ 92,40,984.16** | 🏆 **Yes** |
| House | Extra Trees | 0.2801 | ₹ 70,90,408.21 | ₹ 1,07,08,719.21 | No |
| House | Random Forest | 0.2756 | ₹ 71,92,626.88 | ₹ 1,07,41,718.87 | No |
| House | Decision Tree | -0.2945 | ₹ 92,83,765.53 | ₹ 1,43,59,439.23 | No |
| **Furniture** | **Random Forest** | **0.8997** | **₹ 749.79** | **₹ 1,211.93** | 🏆 **Yes** |
| Furniture | Extra Trees | 0.8806 | ₹ 802.89 | ₹ 1,322.09 | No |
| Furniture | Linear Regression | 0.8388 | ₹ 1,163.36 | ₹ 1,535.96 | No |
| Furniture | Decision Tree | 0.8233 | ₹ 1,048.92 | ₹ 1,608.33 | No |

### In-Depth Analytical Insights:
1. **Linear Valuation in Tech Hardware & Automobiles**:
   In both Laptops ($R^2 = 0.9019$) and Cars ($R^2 = 0.8203$), Linear Regression outperformed complex tree ensembles. Synthetic specifications (e.g. increments in RAM from 8GB to 16GB, or mileage depreciation) represent direct additive/subtractive relationships that Ordinary Least Squares captures cleanly without overfitting.
2. **Combinatorial Interactions in Furniture**:
   In Furniture, Random Forest outperformed Linear Regression ($R^2 = 0.8997$ vs $0.8388$). Furniture value depends heavily on combinations of discrete factors (e.g. `Material=Wood` combined with `Type=Dining Table` vs `Material=Plastic` combined with `Type=Chair`). Trees excel at segmenting these combinatorial subsets.
3. **Overfitting Dynamics in Real Estate (House)**:
   An unconstrained Decision Tree on 800 training rows of House data produced an $R^2 = -0.2945$ (performing worse than predicting the horizontal mean). This provides a textbook demonstration of tree overfitting on continuous, high-variance domains when depth is unrestricted. In contrast, Linear Regression stabilized at $R^2 = 0.4639$.

---

## 7. Streamlit User Interface Design

The user interface was constructed using Streamlit and modern CSS design principles:
1. **Dynamic Form Adaptation**: The UI inspects the active category schema and renders only relevant controls (e.g., Sliders for RAM/Storage on Laptops, Mileage for Cars, BHK for Houses).
2. **Indian Currency Formatter**: Converts raw floats into customary formats (`₹ 1.25 Cr`, `₹ 7.50 Lakh`, or `₹ 45,000`).
3. **Multi-Model Selector**: Users can either accept the 🏆 Best Model or manually toggle to any of the 4 models to compare inference variations in real time.
4. **Diagnostic Plots**:
   - Actual vs Predicted Scatter Plot with diagonal $45^\circ$ reference line.
   - Residual Distribution and Residuals vs Predicted plots.
   - Top Feature Drivers bar chart.
5. **Interactive Dataset Explorer**: Allows searching, previewing, and downloading category CSVs directly.

---

## 8. Limitations & Synthetic Dataset Disclosure

- **Synthetic Nature**: The dataset `pricewise_combined_dataset.csv` was programmatically generated. While it mirrors plausible real-world dynamics, it exhibits synthetic artifacts (such as strict collinearity between year and age, and simplified price distributions).
- **Domain Limitations**:
  - Does not account for geographical hyper-localities (e.g. specific pincodes in real estate or vehicle registration states).
  - Does not model inflation, seasonal sales spikes, or real-world cosmetic damages beyond simple condition categories.
- **Ethical Use**: PriceWise AI is designed for academic demonstration and decision support. Commercial transactions must incorporate physical appraisals and certified inspectors.

---

## 9. Quality Assurance & Automated Testing

A full test suite implemented with `pytest` verifies system integrity:
- **Test Suite**: 27 unit tests across 4 test classes (`TestDataLoader`, `TestPreprocessing`, `TestModelArtifacts`, `TestPredictorAndValidation`).
- **Pass Rate**: 100% (27/27 passed in 1.96s).
- **Coverage**: Verifies zero data leakage, matrix shapes, serialization persistence, edge-case input handling, and INR currency formatting.

---

## 10. Conclusion & Future Scope

PriceWise AI demonstrates that combining disciplined machine learning engineering with clean user interfaces produces a robust, production-ready predictive system.

**Future Enhancements**:
1. Implement hyperparameter optimization using `Optuna` or `GridSearchCV`.
2. Integrate gradient boosted decision trees (`XGBoost`, `LightGBM`, `CatBoost`).
3. Support image-based damage assessment using computer vision to enhance automobile and furniture valuations.
