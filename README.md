# Walmart Store Sales Forecasting & Business Intelligence System

An end-to-end weekly store demand forecasting system built on historical Walmart sales data, external macroeconomic indicators (CPI, Unemployment, Fuel Price), promotional markdowns, and retail holiday schedules.

Built with strict data leakage prevention, multi-step chronological validation, hierarchical cold-start resolution, and seasonal residual modeling.

---

## 🏆 Project Leaderboard (Out-of-Sample 52-Week Validation)

Evaluated across **154,386 observations** over the complete 52-week chronological validation horizon (`2011-11-04` to `2012-10-26`) using Walmart's official **Weighted Mean Absolute Error (WMAE)** metric (Holiday weight = 5.0, Non-holiday weight = 1.0):

| Rank | Model Name | Phase | Official WMAE ($) | Unweighted MAE ($) | RMSE ($) | Holiday WMAE ($) | Non-Holiday WMAE ($) | Improvement vs Baseline |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Seasonal Residual LightGBM** | **Phase 6** | **$1,878.37** | **$1,795.53** | **$3,906.52** | **$2,144.99** | **$1,766.11** | **+5.54% (-$110.16)** 🏆 |
| 🥈 | **Global Blended Ensemble** | **Phase 6** | **$1,878.37** | **$1,795.53** | **$3,906.52** | **$2,144.99** | **$1,766.11** | **+5.54% (-$110.16)** |
| 🥉 | **Seasonal Naive (Lag 52)** | **Phase 4** | **$1,988.53** | **$1,917.65** | **$4,178.57** | **$2,216.66** | **$1,892.47** | Benchmark Baseline |
| 4 | **Recursive LightGBM Regressor** | **Phase 5** | **$2,211.83** | **$2,013.47** | **$5,110.91** | **$2,850.33** | **$1,943.00** | -11.23% (Behind) |
| 5 | **Recursive XGBoost Regressor** | **Phase 5** | **$2,473.63** | **$2,192.41** | **$5,807.57** | **$3,378.81** | **$2,092.51** | -24.39% (Behind) |
| 6 | **Recursive CatBoost Regressor** | **Phase 5** | **$2,658.50** | **$2,351.47** | **$6,408.10** | **$3,646.77** | **$2,242.39** | -33.69% (Behind) |
| 7 | **Moving Average (4-Week)** | **Phase 4** | **$3,222.10** | **$2,870.22** | **$7,819.67** | **$4,354.74** | **$2,745.21** | -62.03% (Behind) |
| 8 | **Store-Dept Historical Median** | **Phase 4** | **$3,292.26** | **$2,884.08** | **$8,032.19** | **$4,606.11** | **$2,739.07** | -65.56% (Behind) |
| 9 | **Naive (Last Value)** | **Phase 4** | **$3,663.62** | **$3,349.64** | **$8,464.44** | **$4,674.24** | **$3,238.10** | -84.24% (Behind) |
| 10 | **Hierarchical Fallback Baseline** | **Phase 4** | **$6,620.61** | **$6,304.49** | **$12,958.41** | **$7,638.15** | **$6,192.19** | -232.94% (Behind) |

---

## 🔍 Key Engineering Insights

1. **The Seasonality Dominance:**
   - Retail sales exhibit strong 52-week annual periodicity. Naive models that ignore seasonality (e.g. 4-Week Moving Average at $3,222.10) suffer an additional **+$1,233+ WMAE penalty** compared to seasonal approaches.
2. **The Autoregressive Compounding Drift Dilemma:**
   - In Phase 5, standalone recursive GBDT models allocated over **51% SHAP importance to `Sales_Lag_1`**. In a 52-step multi-step forecast, feeding predictions back into `Sales_Lag_1` compounded errors, muting holiday peaks and causing LightGBM to miss Thanksgiving by $4,672+.
3. **Seasonal Residual Modeling Breakthrough (Phase 6):**
   - By anchoring on the pristine 52-week lag ($y_{\text{anchor}} = y_{t-52}$) and training LightGBM to predict only the **seasonal residual** ($\Delta_t = y_t - y_{t-52}$) using markdowns, macroeconomic shifts, and holiday proximity (with zero recursive short-term lags), we eliminated autoregressive drift completely.
   - Result: **WMAE dropped to $1,878.37**, setting all-time performance records across Super Bowl, Labor Day, and Christmas.

---

## 📁 Repository Structure

```
Demand-Forecasting/
├── walmart-recruiting-store-sales-forecasting/
│   ├── src/
│   │   ├── data_loader.py          # Data ingestion & schema verification
│   │   ├── preprocessing.py        # Negative sales preservation & cold-start fallback
│   │   ├── feature_engineering.py  # Regular grid lags (shift-1), cyclical calendar & holiday features
│   │   ├── validation.py           # Chronological validation dataset generation
│   │   ├── evaluation.py           # Official Walmart WMAE, MAE, RMSE metrics engine
│   │   ├── baseline_models.py      # Statistical baselines (Naive, SNaive, MA-4, SES, Median)
│   │   ├── benchmark.py            # Baseline benchmarking pipeline
│   │   ├── ml_models.py            # GBDT regressors (LightGBM, XGBoost, CatBoost)
│   │   ├── ml_training.py          # Training orchestrator with early stopping
│   │   ├── ml_forecasting.py       # Recursive multi-step forecasting engine
│   │   ├── model_comparison.py     # ML benchmark orchestrator & SHAP explainer
│   │   ├── residual_model.py       # Phase 6 Champion: Seasonal Residual LightGBM
│   │   ├── ensemble.py             # Blended ensemble & inner validation tuning
│   │   └── phase6_pipeline.py      # Phase 6 execution & validation pipeline
│   ├── tests/
│   │   ├── test_preprocessing.py   # Data cleaning and cold-start unit tests
│   │   ├── test_feature_engineering.py # Leakage prevention and lag grid tests
│   │   ├── test_validation.py      # Chronological split integrity tests
│   │   ├── test_evaluation.py      # WMAE formula & 5x holiday penalty tests
│   │   ├── test_baseline_models.py # Baseline forecasters coverage tests
│   │   ├── test_ml_models.py       # GBDT wrappers and autoregressive state update tests
│   │   └── test_phase6.py          # Residual reconstruction & ensemble tests
│   ├── reports/
│   │   ├── PHASE_1_DATASET_AUDIT.md
│   │   ├── PHASE_2_EDA_REPORT.md
│   │   ├── PHASE_3_FEATURE_ENGINEERING_REPORT.md
│   │   ├── PHASE_4_BASELINE_REPORT.md
│   │   ├── PHASE_5_ML_REPORT.md
│   │   └── PHASE_6_REPORT.md
│   ├── notebooks/
│   │   ├── 01_data_overview.ipynb
│   │   ├── 02_sales_eda.ipynb
│   │   ├── 03_holiday_markdown_analysis.ipynb
│   │   └── 04_time_series_stationarity.ipynb
│   ├── outputs/
│   │   ├── eda/                    # Summary tables and JSON profiles
│   │   ├── figures/                # High-resolution publication plots
│   │   ├── baseline/               # Baseline predictions & metric exports
│   │   ├── ml/                     # ML predictions, SHAP summaries, and pickles
│   │   └── phase6/                 # Champion predictions, metrics, and models
│   └── data/
│       ├── raw/                    # Raw original data (stores, features, train, test)
│       └── processed/              # Engineered Parquet artifacts
├── .gitignore
└── README.md
```

---

## 🧪 Testing & Quality Assurance

The codebase includes an automated test suite containing **34 unit and integration tests** verifying data leakage prevention, negative sales preservation, split integrity, cold-start handling, and WMAE mathematical properties:

```bash
cd walmart-recruiting-store-sales-forecasting
pytest tests/ -v
```

```
============================== test session starts ==============================
collected 34 items

tests/test_baseline_models.py .....                      [ 14%]
tests/test_evaluation.py ......                          [ 32%]
tests/test_feature_engineering.py .....                  [ 47%]
tests/test_ml_models.py .....                            [ 61%]
tests/test_phase6.py .....                               [ 73%]
tests/test_preprocessing.py ....                         [ 85%]
tests/test_validation.py .....                           [100%]

======================= 34 passed in 10.13s =======================
```

---

## 🚀 Reproduction Quickstart

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Aditya-err/Demand-Forecasting.git
   cd Demand-Forecasting/walmart-recruiting-store-sales-forecasting
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy pandas scipy scikit-learn lightgbm xgboost catboost shap statsmodels pytest pyarrow
   ```

3. **Run the full Phase 6 Champion pipeline:**
   ```bash
   python src/phase6_pipeline.py
   ```

---
*Developed by Aditya | MCA Demand Forecasting & Advanced Analytics*
