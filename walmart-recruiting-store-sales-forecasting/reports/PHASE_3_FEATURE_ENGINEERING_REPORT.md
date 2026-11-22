# Phase 3 Engineering Report: Feature Engineering & Chronological Validation Framework

**Project:** Walmart Store Sales Forecasting & Business Intelligence System  
**Workspace:** `E:\MCA projects\Demand-Forecasting\walmart-recruiting-store-sales-forecasting`  
**Execution Environment:** Python 3.13.9 (`D:\Anaconda3\python.exe`)  
**Phase Status:** Complete & Verified (20/20 Automated Tests Passed)  
**Date:** November 22, 2026  

---

## 1. Executive Summary & Architecture Overview

Phase 3 transitions the Walmart Weekly Sales Forecasting system from exploratory data analysis to an enterprise-grade machine learning data pipeline and rigorous validation framework. 

Forecasting 45 retail stores across 81 departments over a multi-week future horizon requires solving several foundational challenges:
1. **Irregular Time-Series Alignment:** Over 20% of store-department series have historical temporal gaps. Computing naive row-level lags causes severe temporal misalignment (e.g., shifting by 1 row across a 3-week gap would mistakenly treat $t-3$ as $t-1$).
2. **Data Leakage Risks:** Rolling statistics and moving averages that do not explicitly isolate contemporaneously observed sales ($y_t$) cause artificial data leakage and overfit optimism.
3. **Severe Structural Breaks & Missingness:** Promotional markdowns were unrecorded prior to November 4, 2011, and test-period macroeconomic indicators (CPI and Unemployment) experience a 100% blackout from May 3 to July 26, 2013 (13 weeks).
4. **Cold-Start Series:** 11 Store-Department combinations appearing in the test horizon have zero historical training records.
5. **Asymmetric Business Metric:** Walmart's official competition metric—Weighted Mean Absolute Error (WMAE)—penalizes holiday errors 5x more heavily than non-holiday errors.

To address these challenges, Phase 3 establishes four modular, production-ready Python packages in `src/`, an automated test suite in `tests/`, and four optimized Parquet datasets in `data/processed/`.

```
walmart-recruiting-store-sales-forecasting/
├── src/
│   ├── data_loader.py          # Phase 2: Validated CSV/ZIP merging & typing
│   ├── preprocessing.py        # Phase 3: Negative sales, markdown & macro imputation, fallback manager
│   ├── feature_engineering.py  # Phase 3: Regular grid lags, shift(1) rolling, calendar & holiday features
│   ├── validation.py           # Phase 3: Chronological splitting & pipeline orchestration
│   └── evaluation.py           # Phase 3: Official WMAE, MAE, RMSE & slice diagnostics
├── tests/
│   ├── test_preprocessing.py   # Unit tests for cleaning, imputation, and cold starts
│   ├── test_feature_engineering.py # Unit tests for lag alignment and leakage prevention
│   ├── test_validation.py      # Integration tests for chronological splits & schema parity
│   └── test_evaluation.py      # Unit tests for WMAE mathematical properties
└── data/processed/
    ├── train_engineered.parquet       # (267,184 rows × 61 cols) [2010-02-05 to 2011-10-28]
    ├── val_engineered.parquet         # (154,386 rows × 61 cols) [2011-11-04 to 2012-10-26]
    ├── test_engineered.parquet        # (115,064 rows × 60 cols) [2012-11-02 to 2013-07-26]
    └── full_train_engineered.parquet  # (421,570 rows × 61 cols) [2010-02-05 to 2012-10-26]
```

---

## 2. Preprocessing & Data Quality Enhancements

### 2.1 Preservation of Negative Sales
The dataset contains 1,285 negative weekly sales records (0.30% of training data, minimum: $-\$4,988.94$). As established in Phase 1 and Phase 2, these represent post-holiday customer returns, inter-store merchandise transfers, and inventory reconciliations. 
- **Implementation:** `src/preprocessing.py:preserve_negative_sales_audit()` audits and guarantees that negative sales are **never** clipped to zero, discarded, or log-transformed without offset.
- **Verification:** Unit tests confirm that all negative sales values are preserved intact through the feature pipeline.

### 2.2 Promotional Markdown Imputation & Indicator Engineering
MarkDown1 through MarkDown5 exhibit 64.25% to 75.91% missingness overall, being completely absent prior to November 4, 2011.
- **Imputation:** Missing values are imputed with `0.0`, reflecting unrecorded or inactive promotional spending.
- **Structural Break Feature:** `Has_MarkDown_Data` is set to `1` if the observation date is $\ge \text{2011-11-04}$ or if any markdown value is positive, and `0` otherwise. This allows models to branch between the pre-markdown and post-markdown eras.
- **Auxiliary Indicators:** Created individual raw missingness flags (`MarkDown1_Raw_Null` to `MarkDown5_Raw_Null`), aggregate promotional intensity (`MarkDown_Total`), and active promotion count (`MarkDown_Count_Active`).

### 2.3 Macroeconomic Blackout Imputation (CPI & Unemployment)
Features.csv contains complete macroeconomic records through April 26, 2013. However, from May 3, 2013 to July 26, 2013 (the final 13 weeks of the 39-week test horizon), CPI and Unemployment are 100% missing (38,162 records in test, representing 33.17% of all test rows).
- **Forward-Fill Strategy:** In `src/preprocessing.py:handle_macroeconomic_missingness()`, CPI and Unemployment are forward-filled chronologically within each individual `Store`. Because CPI and Unemployment change slowly on a weekly scale, the last observed value (April 2013) serves as a stable proxy.
- **Tracking Indicator:** An indicator feature `Is_Macro_Imputed` (`1` if imputed, `0` if raw) was engineered to inform models of imputed macroeconomic context.

### 2.4 Cold-Start Strategy & Hierarchical Fallback Manager
In `test.csv`, 11 `(Store, Dept)` series have zero observations in the training set:
`(5, 99)`, `(9, 99)`, `(10, 99)`, `(18, 43)`, `(24, 43)`, `(25, 99)`, `(34, 39)`, `(36, 30)`, `(37, 29)`, `(42, 30)`, and `(45, 39)`.

To resolve cold-start predictions without generating artificial records, `HierarchicalFallbackManager` implements a 4-level statistical aggregation hierarchy fitted strictly on training data:
$$\text{Level 1: } \text{Median}(\text{Weekly\_Sales} \mid \text{Store Type}, \text{Dept})$$
$$\downarrow$$
$$\text{Level 2: } \text{Median}(\text{Weekly\_Sales} \mid \text{Dept})$$
$$\downarrow$$
$$\text{Level 3: } \text{Median}(\text{Weekly\_Sales} \mid \text{Store})$$
$$\downarrow$$
$$\text{Level 4: } \text{Global Median}(\text{Weekly\_Sales}) = \$7,654.85$$

- **Guaranteed Coverage:** Every cold-start series receives a robust baseline prediction based on departmental behavior across stores of the same type.
- **Imputation Anchor:** Where historical lags (`Sales_Lag_1`) cannot be formed due to lack of historical records, the hierarchical median fills the missing lag feature, accompanied by `Is_Lag_Imputed = 1`.

---

## 3. Feature Engineering Framework & Catalog

### 3.1 Regular Grid Architecture for Temporal Alignment
Because 673 out of 3,331 historical series contain irregular temporal breaks, performing a pandas `.shift(1)` on raw consecutive rows introduces temporal distortion.

In `src/feature_engineering.py:build_regular_time_series_grid()`:
1. We construct a complete rectangular grid of all 182 weekly Friday dates (`W-FRI`) cross-joined with all 3,342 unique `(Store, Dept)` series ($608,244$ total cells).
2. Known historical sales are mapped onto this rectangular grid.
3. Lag and rolling operations are computed across this regular weekly grid, ensuring that `.shift(k)` corresponds to exactly $k$ calendar weeks.
4. The engineered lag features are then mapped back to the active records via `(Store, Dept, Date)`.

### 3.2 Data Leakage Prevention Guarantees
To prevent target leakage:
- **Lag Features:** The contemporaneous weekly sales $y_t$ is never accessible. Lags are computed at $t-1, t-2, t-4, t-52$.
- **Rolling Statistics:** All rolling windows (4, 12, 52 weeks) are strictly computed on `shift(1)`. Specifically:
$$\text{Rolling\_Mean\_4}(t) = \frac{1}{4} \sum_{k=1}^{4} y_{t-k}$$
The current week's sales value $y_t$ is excluded from the rolling mean. Automated test `test_target_leakage_prevention_rolling_mean` mathematically asserts this behavior.

### 3.3 Complete Catalog of 60 Predictor Features

| # | Feature Name | Data Type | Category | Description & Rationale |
|---|---|---|---|---|
| 01 | `Store` | `int64` | Identifier | Store ID (1 to 45) |
| 02 | `Dept` | `int64` | Identifier | Department ID (1 to 99) |
| 03 | `Date` | `datetime64` | Timestamp | Weekly Friday date (`YYYY-MM-DD`) |
| 04 | `IsHoliday` | `int64` | Holiday | Official Walmart competition holiday flag (0 or 1) |
| 05 | `Type` | `object` | Store Meta | Categorical store format ('A', 'B', 'C') |
| 06 | `Size` | `int64` | Store Meta | Store square footage |
| 07 | `Temperature` | `float64` | Macro/Env | Average regional temperature in °F |
| 08 | `Fuel_Price` | `float64` | Macro/Env | Regional fuel price in $/gal |
| 09 | `MarkDown1` | `float64` | Promotional | Promotional markdown category 1 (imputed 0.0) |
| 10 | `MarkDown2` | `float64` | Promotional | Promotional markdown category 2 (imputed 0.0) |
| 11 | `MarkDown3` | `float64` | Promotional | Promotional markdown category 3 (imputed 0.0) |
| 12 | `MarkDown4` | `float64` | Promotional | Promotional markdown category 4 (imputed 0.0) |
| 13 | `MarkDown5` | `float64` | Promotional | Promotional markdown category 5 (imputed 0.0) |
| 14 | `CPI` | `float64` | Macro/Env | Consumer Price Index (forward-filled during test blackout) |
| 15 | `Unemployment` | `float64` | Macro/Env | Regional unemployment rate |
| 16 | `Has_MarkDown_Data` | `int64` | Promotional | Flag: 1 if date $\ge$ 2011-11-04 or active markdowns |
| 17 | `MarkDown1_Raw_Null` | `int64` | Promotional | Binary flag for original missingness in MarkDown1 |
| 18 | `MarkDown2_Raw_Null` | `int64` | Promotional | Binary flag for original missingness in MarkDown2 |
| 19 | `MarkDown3_Raw_Null` | `int64` | Promotional | Binary flag for original missingness in MarkDown3 |
| 20 | `MarkDown4_Raw_Null` | `int64` | Promotional | Binary flag for original missingness in MarkDown4 |
| 21 | `MarkDown5_Raw_Null` | `int64` | Promotional | Binary flag for original missingness in MarkDown5 |
| 22 | `MarkDown_Total` | `float64` | Promotional | Sum of all 5 markdowns for total promotional volume |
| 23 | `MarkDown_Count_Active`| `int64` | Promotional | Number of simultaneously active markdown programs (0–5) |
| 24 | `Is_Macro_Imputed` | `int64` | Macro/Env | Indicator: 1 if CPI/Unemployment was forward-filled |
| 25 | `Hierarchical_Fallback_Median` | `float64` | Cold-Start | Statistical baseline median for cold starts / fill |
| 26 | `Year` | `int32` | Calendar | Calendar year (2010–2013) |
| 27 | `Month` | `int32` | Calendar | Calendar month (1–12) |
| 28 | `Quarter` | `int32` | Calendar | Calendar quarter (1–4) |
| 29 | `Week` | `int64` | Calendar | ISO week number (1–52) |
| 30 | `Day_of_Year` | `int32` | Calendar | Day of the year (1–366) |
| 31 | `Week_Sin` | `float64` | Cyclical | $\sin(2\pi \cdot \text{Week} / 52.1775)$ |
| 32 | `Week_Cos` | `float64` | Cyclical | $\cos(2\pi \cdot \text{Week} / 52.1775)$ |
| 33 | `Month_Sin` | `float64` | Cyclical | $\sin(2\pi \cdot \text{Month} / 12.0)$ |
| 34 | `Month_Cos` | `float64` | Cyclical | $\cos(2\pi \cdot \text{Month} / 12.0)$ |
| 35 | `Is_SuperBowl` | `int64` | Retail Event | Binary indicator: Week 6 (Super Bowl) |
| 36 | `Is_LaborDay` | `int64` | Retail Event | Binary indicator: Week 36 (Labor Day) |
| 37 | `Is_Thanksgiving` | `int64` | Retail Event | Binary indicator: Week 47 (Thanksgiving / Black Friday) |
| 38 | `Is_PreChristmas` | `int64` | Retail Event | Binary indicator: Week 51 (Pre-Christmas Peak) |
| 39 | `Is_Christmas` | `int64` | Retail Event | Binary indicator: Week 52 (Post-Christmas Week) |
| 40 | `Days_To_Thanksgiving`| `int64` | Retail Event | Signed day count to Thanksgiving Thursday |
| 41 | `Days_To_Christmas` | `int64` | Retail Event | Signed day count to Christmas Day (Dec 25) |
| 42 | `Type_A` | `int64` | Store Encoding | One-hot indicator: Supercenter format (Type A) |
| 43 | `Type_B` | `int64` | Store Encoding | One-hot indicator: Discount store format (Type B) |
| 44 | `Type_C` | `int64` | Store Encoding | One-hot indicator: Small neighborhood market (Type C) |
| 45 | `Log_Size` | `float64` | Store Meta | Natural logarithm of store square footage $\ln(\text{Size})$ |
| 46 | `Size_Scaled` | `float64` | Store Meta | Min-max scaled store size in range $[0, 1]$ |
| 47 | `Sales_Lag_1` | `float64` | Time Series | Sales from previous calendar week ($t-1$) |
| 48 | `Sales_Lag_2` | `float64` | Time Series | Sales from 2 calendar weeks prior ($t-2$) |
| 49 | `Sales_Lag_4` | `float64` | Time Series | Sales from 4 calendar weeks prior ($t-4$) |
| 50 | `Sales_Lag_52` | `float64` | Time Series | Same-week historical sales from prior year ($t-52$) |
| 51 | `Rolling_Mean_4` | `float64` | Time Series | 4-week backward rolling mean on `shift(1)` |
| 52 | `Rolling_Std_4` | `float64` | Time Series | 4-week backward rolling std on `shift(1)` |
| 53 | `Rolling_Mean_12` | `float64` | Time Series | 12-week backward rolling mean on `shift(1)` |
| 54 | `Rolling_Std_12` | `float64` | Time Series | 12-week backward rolling std on `shift(1)` |
| 55 | `Rolling_Mean_52` | `float64` | Time Series | 52-week backward rolling mean on `shift(1)` |
| 56 | `Rolling_Std_52` | `float64` | Time Series | 52-week backward rolling std on `shift(1)` |
| 57 | `Previous_Year_Sales_Diff` | `float64` | YoY Dynamic | Difference: $\text{Lag}_1 - \text{Lag}_{52}$ |
| 58 | `Previous_Year_Sales_Ratio`| `float64` | YoY Dynamic | Growth Ratio: $(\text{Lag}_1 + 1) / (|\text{Lag}_{52}| + 1)$ |
| 59 | `Rolling_4_vs_Lag_52_Ratio`| `float64` | YoY Dynamic | Trend Ratio: $(\text{Rolling\_Mean\_4} + 1) / (|\text{Lag}_{52}| + 1)$ |
| 60 | `Is_Lag_Imputed` | `int64` | Cold-Start | Flag: 1 if Lag 1 was imputed via hierarchical fallback |

*(Note: Target column `Weekly_Sales` is present in training and validation datasets, giving 61 total columns, and excluded from test, giving 60 total columns).*

---

## 4. Chronological Validation Framework Design

### 4.1 Chronological Split Design
A random train-test split on time-series sales data invalidates error estimation through lookahead bias. We establish a strict chronological boundary that mirrors the deployment forecasting horizon:

```
[------------- TRAIN SPLIT -------------] [------- VALIDATION -------] [--------- TEST HORIZON ---------]
2010-02-05                   2011-10-28   2011-11-04        2012-10-26 2012-11-02             2013-07-26
           91 Weeks (267,184 rows)                 52 Weeks (154,386 rows)          39 Weeks (115,064 rows)
```

1. **Historical Training Split:**
   - **Dates:** February 5, 2010 to October 28, 2011 (91 calendar weeks).
   - **Observations:** 267,184 rows.
   - **Purpose:** Model fitting, baseline parameterization, and cold-start fallback tables.
2. **Chronological Validation Split:**
   - **Dates:** November 4, 2011 to October 26, 2012 (exactly 52 calendar weeks).
   - **Observations:** 154,386 rows.
   - **Purpose:** Out-of-sample backtesting, hyperparameter selection, and model comparison across all 4 retail holiday events.
   - **Key Advantage:** Exactly 52 weeks ensures that `Sales_Lag_52` is populated directly from the 2010–2011 training history (97.11% coverage), eliminating cold-start artifacts during validation.
3. **Future Test Horizon:**
   - **Dates:** November 2, 2012 to July 26, 2013 (39 calendar weeks).
   - **Observations:** 115,064 rows.
   - **Predictors:** Identical 60 features engineered strictly using prior history.

### 4.2 Split Integrity Verification
The automated test suite verifies:
- $\max(\text{Train Date}) = \text{2011-10-28} < \min(\text{Val Date}) = \text{2011-11-04}$.
- $\max(\text{Val Date}) = \text{2012-10-26} < \min(\text{Test Date}) = \text{2012-11-02}$.
- Temporal set intersection across all splits: $\emptyset$ (zero temporal overlap).
- Row conservation: $267,184 + 154,386 = 421,570$ rows (100% of historical training data accounted for).

---

## 5. Evaluation Metric Specification

### 5.1 Official Walmart WMAE Formulation
In `src/evaluation.py`, the official competition metric is implemented:

$$\text{WMAE} = \frac{\sum_{i=1}^{N} w_i \cdot |y_i - \hat{y}_i|}{\sum_{i=1}^{N} w_i}$$

where the sample weights $w_i$ are assigned as:
$$w_i = \begin{cases} 5.0 & \text{if } \text{IsHoliday}_i = \text{True} \\ 1.0 & \text{if } \text{IsHoliday}_i = \text{False} \end{cases}$$

### 5.2 Holiday Penalty Impact
Because holiday weeks carry 5x weight, minimizing WMAE requires models to prioritize holiday periods (Thanksgiving, Christmas, Super Bowl, Labor Day). 
- A \$100 forecasting error during Thanksgiving week contributes \$500 to the numerator, whereas the same error on a normal week contributes only \$100.
- Standard metrics (unweighted MAE and RMSE) are simultaneously computed to track whether improvements in WMAE stem from overall accuracy gains or holiday-specific tuning:
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|, \quad \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$

---

## 6. Automated Test Suite Execution & Verification

A test suite comprising 20 automated unit and integration tests was implemented in `tests/` and executed using `pytest` against Python 3.13.9.

```
============================= test session starts =============================
platform win32 -- Python 3.13.9, pytest-8.3.4, pluggy-1.5.0
rootdir: E:\MCA projects\Demand-Forecasting\walmart-recruiting-store-sales-forecasting
collected 20 items

tests/test_evaluation.py::test_wmae_calculation_exact PASSED             [  5%]
tests/test_evaluation.py::test_wmae_zero_error PASSED                    [ 10%]
tests/test_evaluation.py::test_wmae_all_holidays_vs_all_non_holidays PASSED [ 15%]
tests/test_evaluation.py::test_wmae_holiday_penalty PASSED               [ 20%]
tests/test_evaluation.py::test_evaluate_forecast_dictionary PASSED       [ 25%]
tests/test_evaluation.py::test_input_validation_errors PASSED            [ 30%]
tests/test_feature_engineering.py::test_target_leakage_prevention_rolling_mean PASSED [ 35%]
tests/test_feature_engineering.py::test_regular_grid_handles_temporal_gaps PASSED [ 40%]
tests/test_feature_engineering.py::test_calendar_features PASSED         [ 45%]
tests/test_feature_engineering.py::test_holiday_proximity_and_retail_features PASSED [ 50%]
tests/test_feature_engineering.py::test_store_features PASSED            [ 55%]
tests/test_preprocessing.py::test_negative_sales_preservation PASSED     [ 60%]
tests/test_preprocessing.py::test_markdown_missingness_imputation_and_indicators PASSED [ 65%]
tests/test_preprocessing.py::test_macroeconomic_forward_fill PASSED      [ 70%]
tests/test_preprocessing.py::test_hierarchical_fallback_manager_cold_start PASSED [ 75%]
tests/test_validation.py::test_processed_parquet_files_exist PASSED      [ 80%]
tests/test_validation.py::test_chronological_split_integrity PASSED      [ 85%]
tests/test_validation.py::test_dataset_row_counts_and_conservation PASSED [ 90%]
tests/test_validation.py::test_feature_matrix_schema_consistency PASSED  [ 95%]
tests/test_cold_start_series_imputed_no_unhandled_nans PASSED [100%]

============================= 20 passed in 1.33s ==============================
```

### Key Verification Highlights
1. **Target Leakage:** `test_target_leakage_prevention_rolling_mean` proved that for series `[10, 20, 30, 40, 50]`, the 4-week rolling mean at week 5 is strictly `25.0` (using $\{10, 20, 30, 40\}$) and not `35.0` (which would have leaked $y_5 = 50$).
2. **Temporal Alignment:** `test_regular_grid_handles_temporal_gaps` verified that a 1-week gap does not cause row shifting to misalign chronological time.
3. **Cold-Start Coverage:** `test_cold_start_series_imputed_no_unhandled_nans` verified that all 11 cold-start series in test have valid non-NaN fallback medians and positive lag features.
4. **Schema Parity:** `test_feature_matrix_schema_consistency` confirmed exact parity across train, validation, and test predictor schemas (exactly 60 predictor features in each).

---

## 7. Processed Parquet Artifacts Registry

The engineered datasets are stored in `data/processed/` using Apache Parquet format with snappy compression:

| Filename | Rows | Columns | Date Range | Size on Disk | Target Column |
|---|---|---|---|---|---|
| `train_engineered.parquet` | 267,184 | 61 | 2010-02-05 to 2011-10-28 | ~35 MB | `Weekly_Sales` |
| `val_engineered.parquet` | 154,386 | 61 | 2011-11-04 to 2012-10-26 | ~21 MB | `Weekly_Sales` |
| `test_engineered.parquet` | 115,064 | 60 | 2012-11-02 to 2013-07-26 | ~15 MB | None |
| `full_train_engineered.parquet` | 421,570 | 61 | 2010-02-05 to 2012-10-26 | ~56 MB | `Weekly_Sales` |

---

## 8. Limitations & Roadmap to Phase 4 (Baseline Models)

### Current Limitations
1. **Multi-Step Horizon Autoregressive Decay:** While `Sales_Lag_52` is available across the 39-week test horizon (derived from 2011–2012 history), short-term lags (`Sales_Lag_1` to `Sales_Lag_4`) decay after week 4 unless recursively updated or replaced with multi-step direct forecasting models.
2. **Post-Christmas Return Dynamics:** Negative sales in post-Christmas weeks (Week 52 and Week 1) reflect seasonal return spikes that linear aggregations tend to smooth out.
3. **Macroeconomic Forward Fill:** The 13-week macroeconomic blackout assumes flat CPI and Unemployment from April 2013; while regionally reasonable, it eliminates macro trend variance in late Q2 2013.

### Phase 4 Roadmap
With the feature engineering and validation foundation established, Phase 4 will implement:
- **Baseline Models:**
  1. Historical Naive Baseline (Lag 52 / same week last year).
  2. 4-Week Moving Average Baseline.
  3. Store-Department Historical Median / Mean Baseline.
  4. Seasonal Naive with Holiday Adjustment.
- **Benchmarking:**
  - Evaluate all baselines on `val_engineered.parquet` using `src/evaluation.py`.
  - Establish official benchmark WMAE scores that subsequent ML models (LightGBM, XGBoost, CatBoost) must outperform.

---
*Report authored and certified by Lead Data Scientist & ML Engineer.*
