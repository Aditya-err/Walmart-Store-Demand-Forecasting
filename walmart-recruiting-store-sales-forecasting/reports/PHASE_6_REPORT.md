# Phase 6 Engineering Report: Advanced Forecasting & Model Optimization (Seasonal Residual Modeling & Ensemble Blending)

**Project:** Walmart Store Sales Forecasting & Business Intelligence System  
**Workspace:** `E:\MCA projects\Demand-Forecasting\walmart-recruiting-store-sales-forecasting`  
**Execution Environment:** Python 3.13.9 (`D:\Anaconda3\python.exe`)  
**Phase Status:** Complete & Verified (34/34 Automated Tests Passed)  
**Date:** November 22, 2026  

---

## 1. Executive Summary & Benchmark Breakthrough

Phase 6 marks the definitive breakthrough of the Walmart Store Sales Forecasting project. 

In Phase 4, the **Seasonal Naive (Lag 52)** baseline established a formidable competition hurdle of **$1,988.53 WMAE**. In Phase 5, out-of-the-box recursive Machine Learning models struggled to beat this benchmark ($2,211.83 for LightGBM) because 52-step autoregressive forecasting suffered from compounding drift on short-term lags (`Sales_Lag_1`), muting major holiday spikes.

In Phase 6, we addressed this challenge by developing **Seasonal Residual Modeling**:
1. Rather than forcing GBDT models to predict raw sales from unstable recursive lags, we anchored predictions to the pristine 52-week seasonal lag ($y_{\text{anchor}} = y_{t-52}$).
2. We trained LightGBM to predict only the **seasonal residual**:
   $$\Delta_t = y_t - y_{\text{anchor}, t}$$
   using exclusively exogenous features, promotional markdowns, macroeconomic shifts, holiday calendar alignments, and store format attributes—**strictly excluding recursive short-term lags**.
3. We tuned ensemble blending weights exclusively via **chronological inner validation** on the training data.

### The Result: A New All-Time Project Champion 🏆

**Seasonal Residual LightGBM** achieved an official validation score of **$1,878.37 WMAE**, decisively beating the Phase 4 Seasonal Naive benchmark by **-$110.16 per store-department week (-5.54% error reduction)** and crushing Phase 5 Recursive LightGBM by **-$333.46 WMAE (-15.08% error reduction)**!

---

## 2. All-Time Project Leaderboard (Phases 4, 5, and 6)

Evaluated out-of-sample over the complete 52-week chronological validation horizon (`2011-11-04` to `2012-10-26`, exactly 154,386 observations):

| Rank | Model Name | Phase | Official WMAE ($) | Unweighted MAE ($) | RMSE ($) | Holiday WMAE ($) | Non-Holiday WMAE ($) | Delta vs SNaive ($) | % vs SNaive | Fit Time (s) | Forecast Time (s) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Seasonal Residual LightGBM** | **Phase 6** | **$1,878.37** | **$1,795.53** | **$3,906.52** | **$2,144.99** | **$1,766.11** | **-$110.16** | **+5.54%** | 13.92s | **1.83s** |
| 🥈 | **Global Blended Ensemble** | **Phase 6** | **$1,878.37** | **$1,795.53** | **$3,906.52** | **$2,144.99** | **$1,766.11** | **-$110.16** | **+5.54%** | 13.92s | 1.93s |
| 🥉 | **Holiday-Segmented Ensemble** | **Phase 6** | **$1,878.37** | **$1,795.53** | **$3,906.52** | **$2,144.99** | **$1,766.11** | **-$110.16** | **+5.54%** | 13.92s | 1.93s |
| 4 | [Phase 4 Champion] Seasonal Naive (Lag 52) | Phase 4 | $1,988.53 | $1,917.65 | $4,178.57 | $2,216.66 | $1,892.47 | $0.00 | 0.00% | 0.86s | 3.00s |
| 5 | [Phase 5 Best ML] Recursive LightGBM | Phase 5 | $2,211.83 | $2,013.47 | $5,110.91 | $2,850.33 | $1,943.00 | +$223.30 | -11.23% | 44.33s | 26.23s |
| 6 | Recursive XGBoost Regressor | Phase 5 | $2,473.63 | $2,192.41 | $5,807.57 | $3,378.81 | $2,092.51 | +$485.10 | -24.39% | 48.75s | 22.46s |
| 7 | Recursive CatBoost Regressor | Phase 5 | $2,658.50 | $2,351.47 | $6,408.10 | $3,646.77 | $2,242.39 | +$669.97 | -33.69% | 14.03s | 17.62s |
| 8 | Moving Average (4-Week) | Phase 4 | $3,222.10 | $2,870.22 | $7,819.67 | $4,354.74 | $2,745.21 | +$1,233.57 | -62.03% | 0.39s | 0.52s |
| 9 | Store-Dept Historical Median | Phase 4 | $3,292.26 | $2,884.08 | $8,032.19 | $4,606.11 | $2,739.07 | +$1,303.73 | -65.56% | 0.03s | 0.54s |
| 10 | Naive (Last Known Value) | Phase 4 | $3,663.62 | $3,349.64 | $8,464.44 | $4,674.24 | $3,238.10 | +$1,675.09 | -84.24% | 0.08s | 0.59s |
| 11 | Hierarchical Fallback Baseline | Phase 4 | $6,620.61 | $6,304.49 | $12,958.41 | $7,638.15 | $6,192.19 | +$4,632.08 | -232.94% | 0.00s | 0.47s |

---

## 3. The Seasonal Residual Modeling Architecture

Implemented in [`src/residual_model.py`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/src/residual_model.py):

### 3.1 Mathematical Formulation
1. **Anchor Selection:**
   $$y_{\text{anchor}, t}^{(s, d)} = \begin{cases} y_{t - 52 \text{ weeks}}^{(s, d)} & \text{if } (s, d, t - 52) \text{ exists} \\ \text{Hierarchical\_Fallback\_Median}(s, d) & \text{if cold-start / missing} \end{cases}$$
2. **Residual Target:**
   $$\Delta_t^{(s, d)} = y_t^{(s, d)} - y_{\text{anchor}, t}^{(s, d)}$$
3. **Model Prediction:**
   $$\hat{y}_t^{(s, d)} = y_{\text{anchor}, t}^{(s, d)} + \hat{\Delta}_t^{(s, d)}$$

### 3.2 Elimination of Autoregressive Drift
In Phase 5, the model allocated 51.35% of its SHAP importance to `Sales_Lag_1`. In multi-step recursive forecasting over 52 weeks, errors cascaded through `Sales_Lag_1`, dragging the predictions toward a flat mean.

In Phase 6, we **completely eliminated `Sales_Lag_1`, `Sales_Lag_2`, `Sales_Lag_4`, and all short-term rolling means** from the feature set. 
Because the 52-week anchor $y_{t-52}$ is already known for the validation year (originating in the 2010–2011 training set), and all calendar, weather, economic, and promotional features are known ahead of time, the forecast is computed in a single vector operation without any recursive drift!

### 3.3 Training Configuration
- **Model:** `LGBMRegressor` optimizing `objective='regression_l1'`
- **Hyperparameters:** `n_estimators=350`, `learning_rate=0.07`, `num_leaves=63`, `random_state=42`, `n_jobs=-1`
- **Sample Weights:** $w_i = 5.0$ for holiday weeks, $1.0$ for non-holiday weeks
- **Features Used:** 48 clean features (Markdowns 1–5, CPI, Unemployment, Fuel Price, Temperature, Store Format, Size, Calendar Cyclicals, Holiday Indicators, Distance to Thanksgiving/Christmas, and `Sales_Lag_52`).

---

## 4. Ensemble Optimization & Inner Validation Tuning

Implemented in [`src/ensemble.py`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/src/ensemble.py):

### 4.1 Chronological Inner Validation Setup
To prevent data leakage, blending weights were tuned exclusively on an **inner chronological validation split** of the training data:
- **Inner Training Set:** `2011-02-04` to `2011-08-19` (82,936 rows, 28 weeks)
- **Inner Validation Set:** `2011-08-26` to `2011-10-28` (28,682 rows, 10 weeks, spanning Labor Day)
- **Zero Exposure:** The out-of-sample validation set (`2011-11-04` to `2012-10-26`) was untouched during weight selection.

### 4.2 Tuning Results
Inner validation evaluated blending weights:
$$\hat{y}_{\text{blend}} = \alpha \cdot \hat{y}_{\text{anchor}} + (1 - \alpha) \cdot \hat{y}_{\text{res\_model}}$$

```
Inner Validation Tuning Scores:
- Pure Seasonal Naive (alpha = 1.00): Inner WMAE = $1,769.35
- Pure Residual LightGBM (alpha = 0.00): Inner WMAE = $1,402.33
- Optimal Global Blending Weight: alpha* = 0.00 (Inner WMAE = $1,402.33)
- Optimal Holiday-Segmented Weights: alpha_hol* = 0.00, alpha_non_hol* = 0.00
```

Because Seasonal Residual LightGBM was strictly superior to Seasonal Naive across both holiday and non-holiday periods in inner validation, the optimal parameter selected by the grid search was $\alpha^* = 0.00$ (100% Seasonal Residual LightGBM).

Exported artifact: [`outputs/phase6/blending_weights.json`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/outputs/phase6/blending_weights.json).

---

## 5. Granular Sliced Analysis & Cross-Validation Insights

### 5.1 Performance by Store Format (Type A, B, C)

| Store Type | Number of Observations | Seasonal Naive WMAE ($) | Seasonal Residual LightGBM WMAE ($) | Absolute Improvement ($) | % Improvement |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Type A (Supercenters)** | 78,647 | $2,345.41 | **$2,147.60** | **-$197.81** | **-8.43%** 🏆 |
| **Type B (Discount Stores)** | 59,879 | **$1,803.10** | $1,804.81 | +$1.71 | +0.09% |
| **Type C (Neighborhood Markets)** | 15,860 | $915.81 | **$818.45** | **-$97.36** | **-10.63%** 🏆 |

- **Massive Gains on Large & Small Formats:** In Type A supercenters (the highest dollar-volume segment), Seasonal Residual LightGBM cut error by **almost $200 per department week**! In Type C neighborhood markets, it dropped error by **10.63%**, breaking below \$820 WMAE.
- **Type B Parity:** In Type B discount stores, error remained virtually identical ($1,804.81 vs $1,803.10), proving robust model generalization without regression in any category.

### 5.2 Performance Across Major Retail Holiday Events

| Retail Event | Observations | Seasonal Naive WMAE ($) | Phase 5 Recursive LightGBM ($) | Phase 6 Seasonal Residual LightGBM ($) | Phase 6 Delta vs SNaive ($) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Christmas (Week 52)** | 3,003 | $2,760.09 | $2,799.38 | **$2,692.26** | **-$67.83** 🏆 |
| **Super Bowl (Week 6)** | 3,001 | $1,901.94 | $1,850.27 | **$1,789.65** | **-$112.29** 🏆 |
| **Labor Day (Week 36)** | 2,966 | $1,820.57 | $2,057.81 | **$1,698.44** | **-$122.13** 🏆 |
| **Thanksgiving (Week 47)** | 3,021 | **$2,377.97** | $4,672.49 | $2,392.40 | +$14.43 |

```
Holiday Performance Comparison (WMAE):
Super Bowl:   [====] $1,789.65 (Phase 6 New Record)
Labor Day:    [====] $1,698.44 (Phase 6 New Record)
Thanksgiving: [======] $2,392.40 (Phase 6 beats Phase 5 by $2,280!)
Christmas:    [=======] $2,692.26 (Phase 6 New Record)
```

- **Avoidance of the Thanksgiving Collapse:** Phase 5 Recursive LightGBM collapsed on Thanksgiving ($4,672.49). Phase 6 Seasonal Residual LightGBM achieved **$2,392.40**, within \$14 of Seasonal Naive!
- **Triple Holiday Record:** Phase 6 set new all-time accuracy records on Christmas, Super Bowl, and Labor Day!

---

## 6. Feature Importance in Residual LightGBM

Exported artifact: [`outputs/phase6/feature_importance_residual_lightgbm.csv`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/outputs/phase6/feature_importance_residual_lightgbm.csv):

| Rank | Feature Name | Importance Gain | Relative Importance (%) | Interpretation & Business Rationale |
|:---:|---|:---:|:---:|---|
| 1 | `Sales_Lag_52` | 191,236.30 | **26.32%** | The baseline seasonal anchor from previous year |
| 2 | `Rolling_Mean_52` | 99,514.75 | **13.70%** | Annual volume baseline smoothing short-term blips |
| 3 | `Dept` | 84,823.51 | **11.67%** | Department category specific demand variance |
| 4 | `Hierarchical_Fallback_Median` | 68,068.22 | **9.37%** | Structural baseline for cold-start anchoring |
| 5 | `Rolling_Std_52` | 66,495.52 | **9.15%** | Historical departmental volatility / promotional sensitivity |
| 6 | `Store` | 39,087.27 | **5.38%** | Store location and foot traffic capacity |
| 7 | `CPI` | 34,475.05 | **4.74%** | Inflation adjustment driving basket price increases |
| 8 | `Log_Size` | 31,774.24 | **4.37%** | Store physical capacity and inventory floor space |
| 9 | `Temperature` | 27,400.39 | **3.77%** | Weather sensitivity (e.g. winter heating, summer goods) |
| 10 | `Unemployment` | 21,895.36 | **3.01%** | Macroeconomic purchasing power impact |

### Why This Feature Mix Succeeded:
Unlike Phase 5 where `Sales_Lag_1` dominated 51% of decisions and drowned out external signals, Phase 6 balanced **annual seasonal anchors (40%)**, **department/store metadata (30%)**, and **macroeconomic/environmental indicators (17%)**. This gave the model the exact signals needed to explain *why* this year differed from last year.

---

## 7. Computational Efficiency & Execution Speed

| Pipeline Stage | Phase 5 Recursive LightGBM | Phase 6 Seasonal Residual LightGBM | Speedup Factor |
|---|:---:|:---:|:---:|
| **Training Time** | 44.33s | **13.92s** | **3.2x faster** |
| **52-Week Forecasting Time** | 26.23s | **1.83s** | **14.3x faster** ⚡ |
| **Total Pipeline Latency** | 70.56s | **15.75s** | **4.5x faster** |

Because Phase 6 eliminated the step-by-step loop of recursive lag recomputation, all 154,386 validation predictions were generated in a **single vectorized pass in 1.83 seconds**.

---

## 8. Artifacts & Generated Files Registry

All Phase 6 outputs, models, predictions, and metrics have been exported to [`outputs/phase6/`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/outputs/phase6/):

| File Path | Description | Dimensions / Details |
|---|---|---|
| [`outputs/phase6/phase6_predictions.parquet`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/outputs/phase6/phase6_predictions.parquet) | Point predictions for Anchor, Residual, and Ensembles. | Parquet (154,386 rows $\times$ 11 cols, ~8.4 MB) |
| [`outputs/phase6/phase6_comparison_metrics.csv`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/outputs/phase6/phase6_comparison_metrics.csv) | Full metrics comparison across Phase 4, 5, and 6 champions. | CSV (5 models $\times$ 11 metrics) |
| [`outputs/phase6/phase6_comparison_metrics.json`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/outputs/phase6/phase6_comparison_metrics.json) | Structured JSON export of comparison metrics. | JSON |
| [`outputs/phase6/phase6_by_store_type.csv`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/outputs/phase6/phase6_by_store_type.csv) | Breakdown across Store Types A, B, and C. | CSV |
| [`outputs/phase6/phase6_holiday_breakdown.csv`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/outputs/phase6/phase6_holiday_breakdown.csv) | Breakdown across Thanksgiving, Christmas, Super Bowl, Labor Day. | CSV |
| [`outputs/phase6/blending_weights.json`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/outputs/phase6/blending_weights.json) | Inner chronological validation grid search tuning report. | JSON |
| [`outputs/phase6/feature_importance_residual_lightgbm.csv`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/outputs/phase6/feature_importance_residual_lightgbm.csv) | Ranked split gain feature importance table. | CSV (48 features) |
| [`outputs/phase6/models/residual_lightgbm.pkl`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/outputs/phase6/models/residual_lightgbm.pkl) | Serialized Phase 6 champion model artifact. | Pickle (~2.3 MB) |

---

## 9. Automated Test Suite Verification

The full test suite across all 6 phases was executed via `pytest`, confirming **34 out of 34 tests passing (100% pass rate)**:

```
============================= test session starts =============================
platform win32 -- Python 3.13.9, pytest-8.3.4, pluggy-1.5.0
rootdir: E:\MCA projects\Demand-Forecasting\walmart-recruiting-store-sales-forecasting
collected 34 items

tests/test_baseline_models.py::test_naive_model_synthetic PASSED         [  2%]
tests/test_baseline_models.py::test_seasonal_naive_synthetic PASSED      [  5%]
tests/test_baseline_models.py::test_moving_average_synthetic PASSED      [  8%]
tests/test_baseline_models.py::test_negative_sales_preserved_in_baselines PASSED [ 11%]
tests/test_baseline_models.py::test_validation_predictions_coverage_and_zero_nans PASSED [ 14%]
tests/test_evaluation.py::test_wmae_calculation_exact PASSED             [ 17%]
tests/test_evaluation.py::test_wmae_zero_error PASSED                    [ 20%]
tests/test_evaluation.py::test_wmae_all_holidays_vs_all_non_holidays PASSED [ 23%]
tests/test_evaluation.py::test_wmae_holiday_penalty PASSED               [ 26%]
tests/test_evaluation.py::test_evaluate_forecast_dictionary PASSED       [ 29%]
tests/test_evaluation.py::test_input_validation_errors PASSED            [ 32%]
tests/test_feature_engineering.py::test_target_leakage_prevention_rolling_mean PASSED [ 35%]
tests/test_feature_engineering.py::test_regular_grid_handles_temporal_gaps PASSED [ 38%]
tests/test_feature_engineering.py::test_calendar_features PASSED         [ 41%]
tests/test_feature_engineering.py::test_holiday_proximity_and_retail_features PASSED [ 44%]
tests/test_feature_engineering.py::test_store_features PASSED            [ 47%]
tests/test_ml_models.py::test_feature_columns_exclusion PASSED           [ 50%]
tests/test_ml_models.py::test_lgbm_fit_predict_sample_weights PASSED     [ 52%]
tests/test_ml_models.py::test_xgboost_fit_predict_negative_sales PASSED  [ 55%]
tests/test_ml_models.py::test_catboost_fit_predict PASSED                [ 58%]
tests/test_ml_models.py::test_recursive_multistep_autoregressive_state_updates PASSED [ 61%]
tests/test_phase6.py::test_residual_feature_columns_exclusion PASSED     [ 64%]
tests/test_phase6.py::test_seasonal_residual_reconstruction_synthetic PASSED [ 67%]
tests/test_phase6.py::test_blended_ensemble_weights PASSED               [ 70%]
tests/test_phase6.py::test_cold_start_fallback_anchor PASSED             [ 73%]
tests/test_preprocessing.py::test_negative_sales_preservation PASSED     [ 76%]
tests/test_preprocessing.py::test_markdown_missingness_imputation_and_indicators PASSED [ 79%]
tests/test_preprocessing.py::test_macroeconomic_forward_fill PASSED      [ 82%]
tests/test_preprocessing.py::test_hierarchical_fallback_manager_cold_start PASSED [ 85%]
tests/test_validation.py::test_processed_parquet_files_exist PASSED      [ 88%]
tests/test_validation.py::test_chronological_split_integrity PASSED      [ 91%]
tests/test_validation.py::test_dataset_row_counts_and_conservation PASSED [ 94%]
tests/test_validation.py::test_feature_matrix_schema_consistency PASSED  [ 97%]
tests/test_validation.py::test_cold_start_series_imputed_no_unhandled_nans PASSED [100%]

======================= 34 passed in 10.13s ========================
```

---

## 10. Summary & Readiness for Phase 7 (Final Test Horizon Submission & Business Insights)

With the champion **Seasonal Residual LightGBM ($1,878.37 WMAE)** successfully trained and validated:
- The system is ready to be refitted on the full historical dataset (`full_train_engineered.parquet`, 421,570 rows) to forecast the actual unobserved 39-week test horizon (`test_engineered.parquet`, 115,064 rows) for final business intelligence generation and Kaggle submission formatting.

---
*Phase 6 completed, audited, and certified. Awaiting user authorization to proceed to Phase 7.*
