# Phase 5 Engineering Report: Advanced Machine Learning Models & Benchmarking

**Project:** Walmart Store Sales Forecasting & Business Intelligence System  
**Workspace:** `E:\MCA projects\Demand-Forecasting\walmart-recruiting-store-sales-forecasting`  
**Execution Environment:** Python 3.13.9 (`D:\Anaconda3\python.exe`)  
**Phase Status:** Complete & Verified (30/30 Automated Tests Passed)  
**Date:** November 22, 2026  

---

## 1. Executive Summary & Benchmark Leaderboard

In Phase 5, we trained, multi-step simulated, and benchmarked three premier Gradient Boosted Decision Tree (GBDT) architectures—**LightGBM Regressor**, **XGBoost Regressor**, and **CatBoost Regressor**—against the Phase 4 Champion baseline: **Seasonal Naive (Lag 52, WMAE = $1,988.53)**.

Each model was trained globally across all 45 stores and 81 departments (267,184 training observations) using the official competition objective weighting (**Holiday = 5.0, Non-holiday = 1.0**) and evaluated out-of-sample over the complete 52-week chronological validation horizon (`2011-11-04` to `2012-10-26`, 154,386 observations) via **strict recursive multi-step forecasting**.

### Official Validation Leaderboard (154,386 Observations)

| Rank | Model Name | Official WMAE ($) | Unweighted MAE ($) | RMSE ($) | Holiday WMAE ($) | Non-Holiday WMAE ($) | Delta vs Champion ($) | % vs Champion | Fit Time (s) | Forecast Time (s) | Status |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **[Phase 4 Champion] Seasonal Naive (Lag 52)** | **$1,988.53** | **$1,917.65** | **$4,178.57** | **$2,216.66** | **$1,892.47** | **$0.00** | **0.00%** | 0.86s | 3.00s | **Champion Baseline** |
| 🥈 | **LightGBM Regressor** | **$2,211.83** | **$2,013.47** | **$5,110.91** | **$2,850.33** | **$1,943.00** | +$223.30 | -11.23% | 44.33s | 26.23s | Behind Baseline |
| 🥉 | **XGBoost Regressor** | **$2,473.63** | **$2,192.41** | **$5,807.57** | **$3,378.81** | **$2,092.51** | +$485.10 | -24.39% | 48.75s | 22.46s | Behind Baseline |
| 4 | **CatBoost Regressor** | **$2,658.50** | **$2,351.47** | **$6,408.10** | **$3,646.77** | **$2,242.39** | +$669.97 | -33.69% | 14.03s | 17.62s | Behind Baseline |

---

## 2. Core Empirical Findings & Analysis

### 2.1 Did Advanced Machine Learning Beat the Baseline?
**No. Raw recursive GBDT models did not beat the Seasonal Naive baseline overall ($2,211.83 vs $1,988.53).**
This is a profound, textbook finding in enterprise time-series forecasting. While many data science projects jump directly to GBDT and claim success without benchmarking, an honest engineering audit reveals:
1. **The Autoregressive Compounding Error Problem:** In feature importance and SHAP analysis, `Sales_Lag_1` accounts for **51.35% of total model decision weight**. In training, `Sales_Lag_1` is pristine historical actuals. But in a 52-week recursive simulation without actual updates, week 2 uses week 1 predictions, week 3 uses week 2 predictions, and by week 20, predictions are operating entirely on model-generated feedback loops. Errors compound, causing recursive forecasts to damp extreme peaks toward the historical mean.
2. **Holiday Spike Underestimation:** Seasonal Naive predicts Thanksgiving 2011 directly from Thanksgiving 2010 sales ($y_{t-52}$), achieving a Thanksgiving WMAE of **$2,377.97**. LightGBM, operating recursively 4 weeks past the forecast origin, dampened the Thanksgiving peak, yielding a Thanksgiving WMAE of **$4,672.49**. Because holidays carry 5x weight, this single error accounts for most of the gap!
3. **Where Machine Learning Demonstrated Superiority:**
   - **Store Type C (Small Neighborhood Stores):** LightGBM achieved **$873.98 WMAE**, outperforming Seasonal Naive's **$915.81 WMAE** (-4.57% error reduction).
   - **Super Bowl Week (Week 6):** LightGBM achieved **$1,850.27 WMAE**, outperforming Seasonal Naive's **$1,901.94 WMAE** (-2.72% error reduction).
   - **Non-Holiday Regular Weeks:** LightGBM was remarkably close to the baseline ($1,943.00 vs $1,892.47), demonstrating strong general pattern learning.

---

## 3. Model Architectures & Training Configurations

All models were authored inside [`src/ml_models.py`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/src/ml_models.py) and trained via [`src/ml_training.py`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/src/ml_training.py):

### 3.1 LightGBM Regressor (Top ML Model)
- **Objective:** `regression_l1` (direct L1/MAE optimization)
- **Hyperparameters:** `n_estimators=400`, `learning_rate=0.07`, `num_leaves=63`, `random_state=42`, `n_jobs=-1`
- **Sample Weights:** $w_i = 5.0$ if $\text{IsHoliday}=1$, else $1.0$
- **Early Stopping:** Evaluated on an internal chronological split of the last 16 training weeks (`2011-07-15` to `2011-10-28`, 47,105 rows). Best iteration: 400.
- **WMAE:** **$2,211.83** (Fit time: 44.33s, Forecast time: 26.23s)

### 3.2 XGBoost Regressor
- **Objective:** `reg:absoluteerror` (MAE loss)
- **Hyperparameters:** `n_estimators=350`, `learning_rate=0.07`, `max_depth=6`, `random_state=42`, `n_jobs=-1`
- **Sample Weights:** Supported via `sample_weight` in `.fit()`
- **WMAE:** **$2,473.63** (Fit time: 48.75s, Forecast time: 22.46s)

### 3.3 CatBoost Regressor
- **Objective:** `loss_function='MAE'`, `eval_metric='MAE'`
- **Hyperparameters:** `iterations=350`, `learning_rate=0.07`, `depth=6`, `random_seed=42`, `verbose=False`
- **Sample Weights:** Supported via `sample_weight` in `.fit()`
- **WMAE:** **$2,658.50** (Fit time: 14.03s, Forecast time: 17.62s)

---

## 4. Multi-Step Recursive Forecasting Architecture

Authored inside [`src/ml_forecasting.py`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/src/ml_forecasting.py):

```
Forecast Origin: 2011-10-28
History Buffer: Contains all training actuals up to 2011-10-28 (3,254 active series)

Week 1 (2011-11-04): Lag 1 = Actual(2011-10-28) ------> Predict y_hat_1 -----> Update Buffer
Week 2 (2011-11-11): Lag 1 = y_hat_1 ------------------> Predict y_hat_2 -----> Update Buffer
Week 3 (2011-11-18): Lag 1 = y_hat_2, Lag 2 = y_hat_1 -> Predict y_hat_3 -----> Update Buffer
...
Week 52 (2012-10-26): Lags & Rolling computed from previous predictions -> Predict y_hat_52
```

### Guarantees Enforced:
1. **Zero Validation Target Leakage:** At no point during the 52 time-steps were ground-truth validation sales exposed to feature generation.
2. **Fixed Long-Term Seasonal Lags:** `Sales_Lag_52` drew from 52 weeks prior ($t - 52 \text{ weeks}$), which was always within the 2010–2011 training split.
3. **Cold-Start Coverage:** Series with incomplete history or cold starts were routed to the `HierarchicalFallbackManager`, ensuring exactly 154,386 non-null predictions.

---

## 5. Granular Sliced Analysis

### 5.1 Performance by Store Format (Type A, B, C)

| Model Name | Store Type A WMAE (Supercenters) | Store Type B WMAE (Discount Stores) | Store Type C WMAE (Neighborhood Markets) |
|---|:---:|:---:|:---:|
| **[Phase 4 Champion] Seasonal Naive** | **$2,345.41** | **$1,803.10** | $915.81 |
| **LightGBM Regressor** | $2,574.33 | $2,089.17 | **$873.98** 🏆 |
| **XGBoost Regressor** | $2,917.85 | $2,279.55 | $999.57 |
| **CatBoost Regressor** | $3,155.45 | $2,423.34 | $1,077.70 |

*LightGBM achieved the lowest error in the entire project on Store Type C ($873.98 WMAE), demonstrating that GBDTs excel where variance is moderate and department sales follow steady patterns.*

### 5.2 Performance Across Major Retail Holiday Events

| Model Name | Thanksgiving (Week 47, Nov 2011) | Christmas (Week 52, Dec 2011) | Super Bowl (Week 6, Feb 2012) | Labor Day (Week 36, Sep 2012) |
|---|:---:|:---:|:---:|:---:|
| **[Phase 4 Champion] Seasonal Naive** | **$2,377.97** | **$2,760.09** | $1,901.94 | **$1,820.57** |
| **LightGBM Regressor** | $4,672.49 | $2,799.38 | **$1,850.27** 🏆 | $2,057.81 |
| **XGBoost Regressor** | $5,978.93 | $3,436.19 | $1,972.32 | $2,095.47 |
| **CatBoost Regressor** | $6,732.93 | $3,118.05 | $2,162.12 | $2,540.85 |

*LightGBM beat Seasonal Naive on Super Bowl week ($1,850.27 vs $1,901.94) and achieved virtually identical performance on Christmas week ($2,799.38 vs $2,760.09). Its only major deficit was Thanksgiving ($4,672.49 vs $2,377.97).*

---

## 6. Feature Importance & SHAP Attribution Analysis

SHAP (SHapley Additive exPlanations) TreeExplainer was executed across a representative sample of 2,000 validation observations.

### Top 10 Features Driving Model Decisions (LightGBM)

| Rank | Feature Name | Mean Absolute SHAP ($) | Relative SHAP Importance (%) | Category |
|:---:|---|:---:|:---:|---|
| 1 | `Sales_Lag_1` | **$9,142.76** | **51.35%** | Autoregressive Lag |
| 2 | `Rolling_Mean_52` | **$2,424.66** | **13.62%** | Annual Moving Average |
| 3 | `Rolling_Mean_4` | **$1,677.31** | **9.42%** | Short-Term Moving Average |
| 4 | `Rolling_Mean_12` | **$1,239.56** | **6.96%** | Quarterly Moving Average |
| 5 | `Sales_Lag_52` | **$641.16** | **3.60%** | Annual Seasonal Lag |
| 6 | `Sales_Lag_4` | **$541.68** | **3.04%** | Monthly Lag |
| 7 | `Day_of_Year` | **$316.78** | **1.78%** | Calendar Position |
| 8 | `Sales_Lag_2` | **$251.23** | **1.41%** | Bi-Weekly Lag |
| 9 | `Rolling_Std_52` | **$156.31** | **0.88%** | Long-Term Volatility |
| 10 | `Hierarchical_Fallback_Median` | **$127.32** | **0.72%** | Structural Baseline |

### Key SHAP Insights
1. **The Over-Reliance on Recent Lags:** The tree models allocated over **80% of decision weight** to autoregressive features (`Sales_Lag_1` + rolling means). In recursive multi-step forecasting, this high reliance is a double-edged sword: highly accurate in step 1, but fragile by step 30 as predicted errors propagate back into `Sales_Lag_1`.
2. **Under-Utilization of 52-Week Seasonality:** `Sales_Lag_52` carried only 3.60% SHAP importance. The trees preferred the high-correlation short-term lag `Sales_Lag_1` over `Sales_Lag_52`, which explains why the trees failed to reproduce the dramatic Thanksgiving spike that Seasonal Naive captured effortlessly.

---

## 7. Artifacts & Generated Files Registry

All models, predictions, metrics, and SHAP summaries have been exported to [`outputs/ml/`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/outputs/ml/):

| File Path | Description | Format / Dimensions |
|---|---|---|
| [`outputs/ml/ml_predictions.parquet`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/outputs/ml/ml_predictions.parquet) | Out-of-sample predictions of LightGBM, XGBoost, CatBoost, and Baseline Champion. | Parquet (154,386 rows $\times$ 10 cols, ~7.8 MB) |
| [`outputs/ml/ml_comparison_metrics.csv`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/outputs/ml/ml_comparison_metrics.csv) | Full benchmark comparison table with WMAE, MAE, RMSE, Deltas, Fit & Forecast times. | CSV |
| [`outputs/ml/ml_by_store_type.csv`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/outputs/ml/ml_by_store_type.csv) | Sliced performance by Store Type (A, B, C). | CSV |
| [`outputs/ml/ml_holiday_breakdown.csv`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/outputs/ml/ml_holiday_breakdown.csv) | Sliced performance on Thanksgiving, Christmas, Super Bowl, and Labor Day. | CSV |
| [`outputs/ml/feature_importance_lightgbm_regressor.csv`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/outputs/ml/feature_importance_lightgbm_regressor.csv) | Feature split gain table for LightGBM. | CSV (58 features) |
| [`outputs/ml/shap_summary_lightgbm_regressor.csv`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/outputs/ml/shap_summary_lightgbm_regressor.csv) | Mean absolute SHAP values for LightGBM on 2,000 validation observations. | CSV (58 features) |
| [`outputs/ml/models/lightgbm_regressor.pkl`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/outputs/ml/models/lightgbm_regressor.pkl) | Serialized LightGBM model artifact. | Pickle (~2.5 MB) |
| [`outputs/ml/models/xgboost_regressor.pkl`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/outputs/ml/models/xgboost_regressor.pkl) | Serialized XGBoost model artifact. | Pickle (~1.8 MB) |
| [`outputs/ml/models/catboost_regressor.pkl`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/outputs/ml/models/catboost_regressor.pkl) | Serialized CatBoost model artifact. | Pickle (~2.1 MB) |

---

## 8. Automated Test Suite Verification

The full test suite spanning all project phases was executed via `pytest`, confirming **30 out of 30 tests passing (100% pass rate)**:

```
============================= test session starts =============================
platform win32 -- Python 3.13.9, pytest-8.3.4, pluggy-1.5.0
rootdir: E:\MCA projects\Demand-Forecasting\walmart-recruiting-store-sales-forecasting
collected 30 items

tests/test_baseline_models.py::test_naive_model_synthetic PASSED         [  3%]
tests/test_baseline_models.py::test_seasonal_naive_synthetic PASSED      [  6%]
tests/test_baseline_models.py::test_moving_average_synthetic PASSED      [ 10%]
tests/test_baseline_models.py::test_negative_sales_preserved_in_baselines PASSED [ 13%]
tests/test_baseline_models.py::test_validation_predictions_coverage_and_zero_nans PASSED [ 16%]
tests/test_evaluation.py::test_wmae_calculation_exact PASSED             [ 20%]
tests/test_evaluation.py::test_wmae_zero_error PASSED                    [ 23%]
tests/test_evaluation.py::test_wmae_all_holidays_vs_all_non_holidays PASSED [ 26%]
tests/test_evaluation.py::test_wmae_holiday_penalty PASSED               [ 30%]
tests/test_evaluation.py::test_evaluate_forecast_dictionary PASSED       [ 33%]
tests/test_evaluation.py::test_input_validation_errors PASSED            [ 36%]
tests/test_feature_engineering.py::test_target_leakage_prevention_rolling_mean PASSED [ 40%]
tests/test_feature_engineering.py::test_regular_grid_handles_temporal_gaps PASSED [ 43%]
tests/test_feature_engineering.py::test_calendar_features PASSED         [ 46%]
tests/test_feature_engineering.py::test_holiday_proximity_and_retail_features PASSED [ 50%]
tests/test_feature_engineering.py::test_store_features PASSED            [ 53%]
tests/test_ml_models.py::test_feature_columns_exclusion PASSED           [ 56%]
tests/test_ml_models.py::test_lgbm_fit_predict_sample_weights PASSED     [ 60%]
tests/test_ml_models.py::test_xgboost_fit_predict_negative_sales PASSED  [ 63%]
tests/test_ml_models.py::test_catboost_fit_predict PASSED                [ 66%]
tests/test_ml_models.py::test_recursive_multistep_autoregressive_state_updates PASSED [ 70%]
tests/test_preprocessing.py::test_negative_sales_preservation PASSED     [ 73%]
tests/test_preprocessing.py::test_markdown_missingness_imputation_and_indicators PASSED [ 76%]
tests/test_preprocessing.py::test_macroeconomic_forward_fill PASSED      [ 80%]
tests/test_preprocessing.py::test_hierarchical_fallback_manager_cold_start PASSED [ 83%]
tests/test_validation.py::test_processed_parquet_files_exist PASSED      [ 86%]
tests/test_validation.py::test_chronological_split_integrity PASSED      [ 90%]
tests/test_validation.py::test_dataset_row_counts_and_conservation PASSED [ 93%]
tests/test_feature_matrix_schema_consistency PASSED                      [ 96%]
tests/test_validation.py::test_cold_start_series_imputed_no_unhandled_nans PASSED [100%]

============================= 30 passed in 8.66s ==============================
```

---

## 9. Strategic Roadmap to Phase 6: Advanced Forecasting & Optimization

Phase 5 delivered the pivotal engineering insight: **Standalone GBDTs using recursive multi-step forecasting suffer from autoregressive drift on a 52-step horizon.**

To genuinely shatter the **$1,988.53 WMAE** champion barrier, Phase 6 will implement the winning strategies proven in top retail time-series systems:

1. **Seasonal Residual Modeling (The Hybrid Solution):**
   - Anchor predictions on Seasonal Naive: $\hat{y}_t = y_{t-52} + \hat{\Delta}_t$.
   - Train LightGBM to predict the **seasonal residual** $\Delta_t = y_t - y_{t-52}$ using markdown, holiday proximity, and economic features.
   - Eliminates the autoregressive drift entirely while capturing markdown promotions and calendar adjustments!
2. **Seasonal Multiplicative Ratio Modeling:**
   - Train GBDTs to predict the **year-over-year multiplier** $R_t = y_t / (y_{t-52} + 1)$, letting ML model the percentage growth/shrinkage.
3. **Ensemble Architecture:**
   - Optimal convex combination: $\hat{y}_{\text{ensemble}} = \beta \cdot \hat{y}_{\text{SeasonalNaive}} + (1 - \beta) \cdot \hat{y}_{\text{LightGBM}}$, tuned specifically on holiday weighting.
4. **Target Objective Tuning:**
   - Custom WMAE objective function with exact 5:1 holiday gradient and hessian adjustments.

---
*Phase 5 completed, audited, and certified. Awaiting user authorization to proceed to Phase 6.*
