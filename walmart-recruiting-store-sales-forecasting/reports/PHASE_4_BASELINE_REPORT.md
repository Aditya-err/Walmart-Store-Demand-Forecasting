# Phase 4 Engineering Report: Baseline Forecasting Models & Validation Benchmarking

**Project:** Walmart Store Sales Forecasting & Business Intelligence System  
**Workspace:** `E:\MCA projects\Demand-Forecasting\walmart-recruiting-store-sales-forecasting`  
**Execution Environment:** Python 3.13.9 (`D:\Anaconda3\python.exe`)  
**Phase Status:** Complete & Verified (25/25 Automated Tests Passed)  
**Date:** November 22, 2026  

---

## 1. Executive Summary & Benchmark Scoreboard

Phase 4 establishes the official quantitative benchmarks for the Walmart Store Sales Forecasting system. In time-series forecasting, advanced machine learning models (e.g., LightGBM, XGBoost, deep learning) should only be deployed if they demonstrably beat rigorous, leakage-free statistical baselines.

We implemented, multi-step simulated, and benchmarked **six distinct forecasting architectures** on the complete 52-week chronological validation horizon (`2011-11-04` to `2012-10-26`, exactly 154,386 observations), fitted strictly on the historical training split (`2010-02-05` to `2011-10-28`, 267,184 observations).

### Benchmark Leaderboard (Sorted by Official Walmart WMAE)

| Rank | Model Name | Official WMAE ($) | Unweighted MAE ($) | RMSE ($) | Holiday WMAE ($) | Non-Holiday WMAE ($) | WMAE / MAE Ratio | Fit Time (s) | Pred Time (s) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Seasonal Naive (Lag 52)** | **$1,988.53** | **$1,917.65** | **$4,178.57** | **$2,216.66** | **$1,892.47** | **1.0370** | 0.86s | 3.00s |
| 🥈 | **Moving Average (4-Week)** | **$3,222.10** | **$2,870.22** | **$7,819.67** | **$4,354.74** | **$2,745.21** | 1.1226 | 0.39s | 0.52s |
| 🥉 | **Store-Dept Historical Median** | **$3,292.26** | **$2,884.08** | **$8,032.19** | **$4,606.11** | **$2,739.07** | 1.1415 | 0.03s | 0.54s |
| 4 | **Exponential Smoothing (SES)** | **$3,403.63** | **$3,055.78** | **$8,184.31** | **$4,523.28** | **$2,932.20** | 1.1138 | 22.02s | 0.58s |
| 5 | **Naive (Last Value)** | **$3,663.62** | **$3,349.64** | **$8,464.44** | **$4,674.24** | **$3,238.10** | 1.0937 | 0.08s | 0.59s |
| 6 | **Hierarchical Fallback Baseline** | **$6,620.61** | **$6,304.49** | **$12,958.41** | **$7,638.15** | **$6,192.19** | 1.0501 | 0.00s | 0.47s |

### Core Engineering Takeaways
1. **The Seasonality Dominance:** **Seasonal Naive (Lag 52)** is the indisputable champion baseline model, achieving a WMAE of **$1,988.53**—outperforming the best non-seasonal baseline (4-Week Moving Average at $3,222.10) by **$1,233.57 per store-department week (-38.28% error reduction)**.
2. **The Holiday Multiplier Effect:** On holiday weeks, non-seasonal models fail dramatically. During Thanksgiving week (Week 47), non-seasonal baselines produced an error exceeding **$8,300**, while Seasonal Naive achieved **$2,377.97** (a **71.5% error reduction**).
3. **Official Threshold for Phase 5:** Any Machine Learning model (LightGBM, XGBoost, CatBoost) proposed in Phase 5 **must beat $1,988.53 WMAE** on this validation set to justify deployment.

---

## 2. Baseline Model Architecture & Implementation Details

All models are authored inside [`src/baseline_models.py`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/src/baseline_models.py) adhering to an object-oriented interface (`fit(train_df)` and `predict(target_df)`):

```
BaseBaselineModel (Abstract Interface)
├── NaiveModel
├── SeasonalNaiveModel
├── MovingAverageModel
├── ExponentialSmoothingModel
├── HistoricalMedianModel
└── HierarchicalFallbackModel
```

### 2.1 Model 1: Naive (Last Known Value)
- **Mathematical Formulation:**
  $$\hat{y}_{t}^{(s, d)} = y_{T_0}^{(s, d)}$$
  where $T_0 = \text{2011-10-28}$ is the forecast origin date, and $y_{T_0}^{(s, d)}$ is the last observed sales value in the training history for store $s$ and department $d$.
- **Behavior:** Persists a constant flat line into the future. Because the cutoff occurs in late October, pre-holiday trends are locked in, failing to anticipate November/December spikes and overshooting normal post-holiday January sales.

### 2.2 Model 2: Seasonal Naive (Lag 52)
- **Mathematical Formulation:**
  $$\hat{y}_{t}^{(s, d)} = \begin{cases} y_{t - 52 \text{ weeks}}^{(s, d)} & \text{if } (s, d, t - 52) \in \mathcal{D}_{\text{train}} \\ \text{Fallback}(s, d) & \text{otherwise} \end{cases}$$
- **Calendar Alignment:** Because our validation set is exactly 52 calendar weeks (`2011-11-04` to `2012-10-26`), subtracting 52 weeks (364 days) maps precisely to `2010-11-05` through `2011-10-28`, which lies 100% within the historical training split.
- **Coverage & Cold-Starts:** 149,923 out of 154,386 validation predictions (97.11%) had direct 52-week historical counterparts. For the remaining 4,463 records (2.89%, representing departments founded mid-2011), the model automatically routed to the hierarchical fallback manager without leaking validation data.

### 2.3 Model 3: Moving Average (4-Week)
- **Mathematical Formulation:**
  $$\hat{y}_{t}^{(s, d)} = \frac{1}{\min(W, N^{(s,d)})} \sum_{k=0}^{\min(W, N^{(s,d)})-1} y_{T_0 - k}^{(s, d)}, \quad W=4$$
- **Behavior:** Smooths out short-term idiosyncratic weekly noise right before the cutoff date, improving upon the raw Naive model by \$441.52 WMAE ($3,222.10 vs $3,663.62). However, like Naive, it projects a flat forecast into the horizon.

### 2.4 Model 4: Exponential Smoothing (SES)
- **Mathematical Formulation:**
  $$\hat{y}_{T_0 + h \mid T_0} = \ell_{T_0}, \quad \text{where } \ell_t = \alpha y_t + (1 - \alpha) \ell_{t-1}$$
- **Optimization:** For each series with $N \ge 4$ observations, estimated $\alpha$ via statsmodels `SimpleExpSmoothing`.
- **Finding on Holt-Winters Seasonality:** Standard Holt-Winters additive seasonality ($m=52$) requires at least two full seasonal cycles ($2 \times 52 = 104$ weeks). Because the training set spans 91 weeks, full seasonal Holt-Winters cannot be initialized by heuristic methods on the training split, affirming Simple Exponential Smoothing as the correct baseline specification.

### 2.5 Model 5: Store-Department Historical Median
- **Mathematical Formulation:**
  $$\hat{y}_{t}^{(s, d)} = \text{Median}\left( \{ y_{\tau}^{(s, d)} \mid \tau \in \mathcal{D}_{\text{train}} \} \right)$$
- **Behavior:** Immune to extreme outlier spikes (e.g., Black Friday anomalies). Achieved $3,292.26 WMAE, competitive with the 4-week moving average.

### 2.6 Model 6: Hierarchical Fallback Baseline
- **Mathematical Formulation:**
  $$\hat{y}_{t}^{(s, d)} = \text{Median}(y \mid \text{Type}_s, \text{Dept}_d) \rightarrow \text{Median}(y \mid \text{Dept}_d) \rightarrow \text{Median}(y \mid \text{Store}_s) \rightarrow \text{Global Median}$$
- **Behavior:** Serves as the structural macro lower-bound ($6,620.61 WMAE). It captures inter-departmental scale differences (e.g., Grocery vs. Jewelry) but lacks individual store calibration.

---

## 3. Multi-Step Forecasting Simulation & Zero Data Leakage Guarantees

In genuine real-world store sales planning, forecasts must be generated for multiple weeks or quarters into the future before any future sales are observed.

### Leakage Prevention Protocol
1. **Strict Temporal Boundary:** All model states, lookups, parameters, and hierarchical medians were fitted **exclusively on data observed $\le \text{2011-10-28}$**.
2. **No Autoregressive Peeking:** At no point during the 52 validation weeks did any model observe the actual validation sales $y_t$ to predict $y_{t+1}$.
3. **Negative Sales Preservation:** Predictions strictly allowed negative values where historical data warranted (verified by unit test `test_negative_sales_preserved_in_baselines`).
4. **Complete Matrix Output:** Every model generated **exactly 154,386 predictions** with **zero NaN values**.

---

## 4. Sliced Performance Analysis

### 4.1 Performance by Store Format (Type A, B, C)

| Model Name | Store Type A WMAE (Supercenters) | Store Type B WMAE (Discount Stores) | Store Type C WMAE (Small Neighborhood) |
|---|:---:|:---:|:---:|
| **Seasonal Naive (Lag 52)** | **$2,345.41** | **$1,803.10** | **$915.81** |
| Moving Average (4-Week) | $3,857.21 | $2,962.20 | $1,048.21 |
| Store-Dept Historical Median | $3,961.56 | $2,969.88 | $1,184.61 |
| Exponential Smoothing (SES) | $4,081.68 | $3,122.34 | $1,097.20 |
| Naive (Last Value) | $4,422.84 | $3,298.39 | $1,271.03 |
| Hierarchical Fallback Baseline | $7,968.43 | $5,977.75 | $2,352.31 |

- **Format Scale Effect:** Type A stores have the largest sales volume (mean weekly sales ~\$20k), resulting in higher absolute dollar errors across all models.
- **Type C Accuracy:** Small Type C stores have much lower sales variance, where Seasonal Naive breaks below \$1,000 WMAE (**$915.81**).

### 4.2 Performance Across Major Retail Holiday Events

Holiday weeks carry a **5x penalty** in WMAE. The table below isolates model error across the four official Walmart holiday weeks in the validation period:

| Model Name | Thanksgiving (Week 47, Nov 2011) | Christmas (Week 52, Dec 2011) | Super Bowl (Week 6, Feb 2012) | Labor Day (Week 36, Sep 2012) |
|---|:---:|:---:|:---:|:---:|
| **Seasonal Naive (Lag 52)** | **$2,377.97** | **$2,760.09** | **$1,901.94** | **$1,820.57** |
| Moving Average (4-Week) | $8,337.50 | $3,593.38 | $2,614.11 | $2,830.13 |
| Exponential Smoothing (SES) | $8,335.47 | $3,536.96 | $2,940.37 | $3,240.62 |
| Naive (Last Value) | $8,323.52 | $3,361.64 | $3,245.29 | $3,732.07 |
| Store-Dept Historical Median | $8,856.24 | $3,791.73 | $3,028.46 | $2,697.96 |
| Hierarchical Fallback Baseline | $11,684.15 | $6,212.92 | $6,484.33 | $6,127.56 |

```
Thanksgiving Week Error Comparison:
Seasonal Naive:  [==] $2,377.97
Moving Average:  [========] $8,337.50
Historical Med:  [=========] $8,856.24
Hierarchical FB: [============] $11,684.15
```

- **The Thanksgiving Catastrophe for Non-Seasonal Models:** Non-seasonal models missed Thanksgiving by an average of **\$8,300+ per department**. Because Thanksgiving occurs 4 weeks after the forecast origin, a flat forecast cannot predict the sudden 3x demand surge.
- **Why Seasonal Naive Excels:** Seasonal Naive predicts Thanksgiving 2011 directly from Thanksgiving 2010, matching the exact holiday pattern with high fidelity.

---

## 5. Artifacts & Generated Files Registry

All benchmark outputs, comparison tables, and prediction files have been exported to [`outputs/baseline/`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/outputs/baseline/):

| File Path | Description | Format / Dimensions |
|---|---|---|
| [`outputs/baseline/baseline_comparison_metrics.csv`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/outputs/baseline/baseline_comparison_metrics.csv) | Full benchmark comparison table with WMAE, MAE, RMSE, fit and prediction times. | CSV (6 models $\times$ 9 metrics) |
| [`outputs/baseline/baseline_comparison_metrics.json`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/outputs/baseline/baseline_comparison_metrics.json) | Structured JSON export of all baseline performance metrics. | JSON |
| [`outputs/baseline/baseline_by_store_type.csv`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/outputs/baseline/baseline_by_store_type.csv) | Granular error metrics sliced by Store Type (A, B, C). | CSV (18 rows) |
| [`outputs/baseline/baseline_holiday_breakdown.csv`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/outputs/baseline/baseline_holiday_breakdown.csv) | Performance isolated across Thanksgiving, Christmas, Super Bowl, and Labor Day. | CSV (24 rows) |
| [`outputs/baseline/baseline_predictions.parquet`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/outputs/baseline/baseline_predictions.parquet) | Point-by-point predictions of all 6 models alongside ground truth. | Parquet (154,386 rows $\times$ 12 cols, ~7.8 MB) |

---

## 6. Automated Test Suite Execution

The complete test suite in [`tests/`](file:///e:/MCA%20projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/tests/) was executed via `pytest`, confirming 100% pass rate:

```
============================= test session starts =============================
platform win32 -- Python 3.13.9, pytest-8.3.4, pluggy-1.5.0
rootdir: E:\MCA projects\Demand-Forecasting\walmart-recruiting-store-sales-forecasting
collected 25 items

tests/test_baseline_models.py::test_naive_model_synthetic PASSED         [  4%]
tests/test_baseline_models.py::test_seasonal_naive_synthetic PASSED      [  8%]
tests/test_baseline_models.py::test_moving_average_synthetic PASSED      [ 12%]
tests/test_baseline_models.py::test_negative_sales_preserved_in_baselines PASSED [ 16%]
tests/test_baseline_models.py::test_validation_predictions_coverage_and_zero_nans PASSED [ 20%]
tests/test_evaluation.py::test_wmae_calculation_exact PASSED             [ 24%]
tests/test_evaluation.py::test_wmae_zero_error PASSED                    [ 28%]
tests/test_evaluation.py::test_wmae_all_holidays_vs_all_non_holidays PASSED [ 32%]
tests/test_evaluation.py::test_wmae_holiday_penalty PASSED               [ 36%]
tests/test_evaluation.py::test_evaluate_forecast_dictionary PASSED       [ 40%]
tests/test_evaluation.py::test_input_validation_errors PASSED            [ 44%]
tests/test_feature_engineering.py::test_target_leakage_prevention_rolling_mean PASSED [ 48%]
tests/test_feature_engineering.py::test_regular_grid_handles_temporal_gaps PASSED [ 52%]
tests/test_feature_engineering.py::test_calendar_features PASSED         [ 56%]
tests/test_feature_engineering.py::test_holiday_proximity_and_retail_features PASSED [ 60%]
tests/test_feature_engineering.py::test_store_features PASSED            [ 64%]
tests/test_preprocessing.py::test_negative_sales_preservation PASSED     [ 68%]
tests/test_preprocessing.py::test_markdown_missingness_imputation_and_indicators PASSED [ 72%]
tests/test_preprocessing.py::test_macroeconomic_forward_fill PASSED      [ 76%]
tests/test_preprocessing.py::test_hierarchical_fallback_manager_cold_start PASSED [ 80%]
tests/test_validation.py::test_processed_parquet_files_exist PASSED      [ 84%]
tests/test_validation.py::test_chronological_split_integrity PASSED      [ 88%]
tests/test_validation.py::test_dataset_row_counts_and_conservation PASSED [ 92%]
tests/test_feature_matrix_schema_consistency PASSED                      [ 96%]
tests/test_validation.py::test_cold_start_series_imputed_no_unhandled_nans PASSED [100%]

============================= 25 passed in 7.49s ==============================
```

---

## 7. Strategic Transition to Phase 5: Advanced Machine Learning Models

### What the Champion Baseline ($1,988.53 WMAE) Teaches Us
While Seasonal Naive achieves an impressive \$1,988.53 WMAE, it possesses inherent structural limitations that Machine Learning can overcome:
1. **Promotional Markdown Inelasticity:** Seasonal Naive cannot respond to MarkDown1 through MarkDown5 discounts. Markdowns were introduced in November 2011, meaning 2010 historical sales lacked promotions entirely! ML models with markdown interaction features can capture markdown-driven demand lift.
2. **Economic & Environmental Covariates:** Seasonal Naive ignores changes in Fuel Price, Unemployment, and CPI.
3. **Calendar Shifts:** When retail holidays shift by calendar days, pure 52-week lag alignment suffers slight phase mismatches. Feature engineered proximity features (`Days_To_Thanksgiving`, `Days_To_Christmas`, `Is_PreChristmas`) allow GBDTs to align peaks perfectly.
4. **Cold-Start Sensitivity:** For incomplete series, Seasonal Naive falls back to static medians. GBDT models can generalize across store types and department clusters.

### Target for Phase 5 (LightGBM / XGBoost / CatBoost)
- **Target:** Outperform Seasonal Naive ($1,988.53 WMAE). A realistic target for a well-tuned LightGBM model utilizing the 60 engineered features is **WMAE < $1,400 – $1,600**.

---
*Phase 4 completed, audited, and certified. Awaiting user authorization to proceed to Phase 5.*
