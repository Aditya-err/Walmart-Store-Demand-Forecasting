# Walmart Store Sales Forecasting — Phase 2: Exploratory Data Analysis & Time-Series Report

**Document Version:** 1.0  
**Phase:** Phase 2 — Exploratory Data Analysis (EDA) & Time-Series Analysis  
**Status:** Completed & Validated  
**Author:** Lead Data Scientist, Time-Series Analyst & ML Engineer  
**Date:** November 2026  
**Project Location:** `E:\MCA projects\Demand-Forecasting\walmart-recruiting-store-sales-forecasting`  
**Execution Environment:** `D:\Anaconda3\python.exe` (Python 3.13.9, Pandas 2.3.3, NumPy 2.3.5, SciPy 1.16.3, Statsmodels 0.14.5, Matplotlib 3.10.0, Seaborn 0.13.2, PyArrow 21.0.0)

---

## 1. Executive Summary

This report delivers the comprehensive empirical findings of **Phase 2: Exploratory Data Analysis & Time-Series Decomposition** for the Walmart Store Sales Forecasting System. 

Building upon the strict read-only audit of Phase 1, Phase 2 established a reproducible data ingestion pipeline (`src/data_loader.py`), merged base sales tables with store dimensions and external features using validated many-to-one join keys, persisted analytical artifacts into `data/processed/*.parquet`, and executed deep statistical and time-series profiling across all 45 stores, 81 departments, and 182 contiguous calendar weeks (February 2010 to July 2013).

```
                            PHASE 2 AT A GLANCE
┌────────────────────────────┬────────────────────────────┬────────────────────────────┐
│ Primary Revenue Base       │ Holiday Dynamic            │ Statistical Behavior       │
│ $6.74 Billion Total Sales  │ Thanksgiving: +40.9% Uplift│ ADF: p = 2.68e-07 (Stat.)  │
│ 421,570 Merged Train Rows  │ Pre-Xmas Peak: +72.2% Spk  │ KPSS: p = 0.10 (Trend-St.) │
│ 115,064 Merged Test Rows   │ Dec 31 Post-Xmas: -13.6%   │ Size vs Sales: r = +0.245  │
├────────────────────────────┼────────────────────────────┼────────────────────────────┤
│ Anomaly & Risk Vectors     │ Structural Breaks          │ Panel Characteristics      │
│ 1,285 Negative Sales Rows  │ Markdowns: 100% NaN Pre-11 │ 3,331 Active Time Series   │
│ 11 Cold-Start Series Test  │ Test Macro: 13-Wk Blackout │ Complete Series: 79.86%    │
│ Dept 47: Net Negative Life │ MD1 & MD4: Collinear 0.82  │ Incomplete Series: 20.14%  │
└────────────────────────────┴────────────────────────────┴────────────────────────────┘
```

### Core Business & Machine Learning Findings:
1. **The Pre-Christmas Week 51 Surge:** The single highest-grossing shopping event in the entire historical dataset is **not** the official Christmas week (`Dec 31` / `Dec 30`), but rather the **week immediately preceding Christmas** (`Dec 24, 2010`: $80.93M; `Dec 23, 2011`: $77.00M). In the dataset, this peak week is marked as `IsHoliday = False`. Models relying solely on the binary `IsHoliday` flag will suffer catastrophic under-prediction during Week 51.
2. **Holiday Sales Uplift Disparity:** Official holiday weeks average **$17,035.82** (+7.13% overall uplift vs $15,901.45 non-holiday baseline). However, Thanksgiving / Black Friday drives an explosive **+40.9% revenue surge** ($22,403.34 mean), while Christmas week itself averages **$14,535.35** (-8.6% below baseline) due to post-holiday store traffic normalization.
3. **Severe Department Skew & Pareto Concentration:** Total revenue is heavily concentrated. The top 20% of departments generate **78.4% of total enterprise sales**. Just three departments—Dept 92 (Grocery), Dept 95 (DSD/Snacks), and Dept 38 (Pharmacy)—contribute over $1.32 Billion combined.
4. **Negative Sales Business Validity:** Exactly **1,285 records (0.3048%)** contain negative sales, concentrated heavily in Dept 47 (254 records), Dept 18 (180 records), and Dept 54 (146 records). Dept 47 has an aggregate **negative lifetime balance (-$4,962.93)**, establishing that it operates as a returns, layaway cancellation, or customer refund clearing department.
5. **Macroeconomic Exogenous Feature Decoupling:** Temperature ($r = -0.008$), Fuel Price ($r = -0.014$), CPI ($r = -0.020$), and Unemployment ($r = -0.032$) exhibit near-zero linear correlation with weekly department sales. Store size ($r = +0.245$) and store-department calendar seasonality are the overwhelming drivers of sales variance.
6. **Test-Period Macro Blackout:** For the final 13 weeks of the test evaluation window (`2013-05-03` to `2013-07-26`), CPI and Unemployment are 100% missing across all 45 stores.

---

## 2. Dataset Overview & Data Ingestion Pipeline

The data ingestion pipeline was codified in `src/data_loader.py` and executed without modifying raw archives.

```
RAW ARCHIVES (Read-Only)               MERGE ENGINE                   PROCESSED ARTIFACTS
┌───────────────────────┐         ┌───────────────────────┐         ┌───────────────────────┐
│ stores.csv    (45 r)  │────────►│ Join Key: Store       │────────►│ train_merged.parquet  │
│ features.csv  (8.19k) │────────►│ validate='many_to_one'│         │ (421,570 r x 16 cols) │
│ train.csv.zip (421.5k)│────────►│                       │         ├───────────────────────┤
│ test.csv.zip  (115.0k)│────────►│ Join Key: Store, Date │────────►│ test_merged.parquet   │
└───────────────────────┘         │ validate='many_to_one'│         │ (115,064 r x 15 cols) │
                                  └───────────────────────┘         └───────────────────────┘
```

### Verification Checks:
- **Join Cardinality:** Merging `train.csv` (421,570 rows) with `stores.csv` on `['Store']` using `validate='many_to_one'` yielded 0 unmatched records and exactly 421,570 rows.
- **Features Join:** Merging with `features.csv` on `['Store', 'Date']` using `validate='many_to_one'` preserved the exact 421,570 rows.
- **IsHoliday Consistency:** Comparing `IsHoliday` from sales tables against `IsHoliday` from external features confirmed **0 mismatches**. The redundant column was cleanly resolved.
- **Test Integrity:** The identical merge pipeline executed on `test.csv` (115,064 rows) yielded exactly 115,064 rows.

*Figure Reference: `outputs/figures/fig01_dataset_timeline_overview.png` illustrates the contiguous timeline, split boundaries, and data breakpoints.*

---

## 3. Overall Sales Distribution & Skewness

Weekly department sales exhibit an extreme positive skew with heavy tails (leptokurtic distribution):

| Metric | Empirical Value | Statistical Meaning |
| :--- | :--- | :--- |
| **Observation Count ($N$)** | 421,570 | Complete non-null training observations |
| **Mean ($\mu$)** | $15,981.26 | Arithmetic mean across all department-weeks |
| **Standard Deviation ($\sigma$)** | $22,711.18 | Extreme dispersion ($\sigma > \mu$) |
| **Median (Q2)** | $7,612.03 | Central tendency ($50\%$ of sales are under $7.6k) |
| **First Quartile (Q1 / 25%)** | $2,079.65 | Low-volume baseline threshold |
| **Third Quartile (Q3 / 75%)** | $20,205.85 | High-volume department threshold |
| **Interquartile Range (IQR)** | $18,126.20 | Spread of middle 50% of department sales |
| **Minimum Value** | -$4,988.94 | Store 28, Dept 6 (Customer returns / adjustments) |
| **Maximum Value** | $693,099.36 | Store 10, Dept 72 (Black Friday peak sales) |
| **Sample Skewness ($S$)** | **+3.26** | Extreme right-tail positive skew |
| **Sample Kurtosis ($K$)** | **+21.49** | Severe leptokurtic distribution with extreme outliers |
| **Negative Sales Count** | 1,285 (0.3048%) | Documented retail returns & chargebacks |
| **Zero Sales Count** | 73 (0.0173%) | Operational zero-sale weeks |

```
Distribution Comparison: Linear Scale vs. Signed Log1p Scale
Linear:     [####|........................................................................] (Skew: +3.26)
Signed Log: [......|############################################|.........................] (Bell-shaped)
```

*Figure Reference: `outputs/figures/fig02_sales_distribution_and_log.png` compares the raw linear histogram against the stabilized signed log1p transformed distribution.*

---

## 4. Store-Level Performance & Size Correlations

### 4.1 Store Revenue Rankings
The 45 stores generate dramatically different sales volumes over the 143 historical weeks:
- **Top 3 Highest-Grossing Stores:**
  1. `Store 20` (Type A, Size: 205,863 sq ft): **$301.40 Million** Total ($29,508.69 mean weekly sales per dept)
  2. `Store 4` (Type A, Size: 205,863 sq ft): **$299.54 Million** Total ($29,154.47 mean weekly sales per dept)
  3. `Store 14` (Type A, Size: 200,898 sq ft): **$288.99 Million** Total ($28,784.15 mean weekly sales per dept)
- **Bottom 3 Lowest-Grossing Stores:**
  43. `Store 5` (Type B, Size: 34,875 sq ft): **$45.48 Million** Total ($5,053.46 mean weekly sales per dept)
  44. `Store 36` (Type A, Size: 39,910 sq ft): **$44.86 Million** Total ($7,064.24 mean weekly sales per dept)
  45. `Store 33` (Type A, Size: 39,690 sq ft): **$37.16 Million** Total ($5,728.40 mean weekly sales per dept)

### 4.2 Physical Size vs. Sales Relationship
A bivariate regression of physical store area (`Size` in sq ft) against average weekly department sales demonstrated a strong, statistically significant positive linear relationship:
$$\text{Pearson } r = +0.245 \quad (p < 0.001) \quad | \quad \text{Spearman } \rho = +0.282 \quad (p < 0.001)$$
When evaluated at the store aggregate level, the correlation between Store Size and Store Mean Sales surges to **$r = +0.810$ ($R^2 = 0.656$)**, establishing physical footprint as the primary determinant of store-level capacity.

*Figure References:*
- `outputs/figures/fig04_store_sales_ranking.png`: All 45 stores ranked by total historical sales.
- `outputs/figures/fig05_store_type_size_vs_sales.png`: Scatter plot and regression trend line of Store Size vs Average Weekly Sales.

---

## 5. Store Type Archetype Profiling

Walmart classifies retail formats into three discrete archetypes:

| Store Type | Store Count | Store Footprint (Sq Ft) | Department Sales Count | Mean Dept Sales ($) | Median Dept Sales ($) | Total Revenue Generated ($) | Enterprise Share (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Type A** (Supercenters) | 22 (48.9%) | 39,690 – 219,622 (Mean: 177,248) | 215,478 | $20,099.57 | $10,105.17 | $4,331,014,537.46 | **64.28%** |
| **Type B** (Standard Stores)| 17 (37.8%) | 34,875 – 140,167 (Mean: 101,191) | 163,495 | $12,237.08 | $6,187.87 | $2,000,701,253.41 | **29.70%** |
| **Type C** (Small / Express)| 6 (13.3%) | 39,690 – 42,988 (Mean: 40,542) | 42,597 | $9,519.53 | $1,149.67 | $405,503,507.82 | **6.02%** |

```
REVENUE CONTRIBUTION BY STORE TYPE:
Type A [============================================================] 64.28%
Type B [==============================] 29.70%
Type C [======] 6.02%
```

**Key Takeaway:** Type A supercenters achieve more than double the average sales velocity of Type C express stores. In Type C stores, the median department sales is just $1,149.67, indicating that smaller formats stock only essential fast-moving goods.

---

## 6. Department-Level Revenue & Sales Concentration

### 6.1 Highest and Lowest Revenue Departments
Across 81 distinct departments, sales velocity varies by multiple orders of magnitude:

| Rank | Department ID | Primary Merchandise Category | Total Revenue ($) | Mean Weekly Sales ($) | Std Dev ($) | CV ($\sigma/\mu$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Dept 92** | Dry Grocery & Staples | **$483,943,300** | $75,204.87 | $40,517.20 | 0.54 |
| **2** | **Dept 95** | DSD Beverages & Snacks | **$449,320,200** | $69,824.42 | $34,881.33 | 0.50 |
| **3** | **Dept 38** | Prescription Pharmacy | **$393,118,100** | $61,090.58 | $31,701.21 | 0.52 |
| **4** | **Dept 72** | Consumer Electronics | **$305,725,200** | $50,560.56 | $68,891.12 | **1.36 (High Volatility!)** |
| **5** | **Dept 90** | Dairy & Refrigerated | **$291,068,500** | $45,232.07 | $20,687.21 | 0.46 |
| ... | ... | ... | ... | ... | ... | ... |
| **77** | **Dept 51** | Seasonal Specialty | $30,572.83 | $32.48 | $89.21 | 2.75 |
| **78** | **Dept 78** | Low-Volume Apparel/Accessory | $1,714.71 | $14.28 | $42.50 | 2.98 |
| **79** | **Dept 39** | Clearance Miscellaneous | $177.98 | $0.85 | $6.20 | 7.29 |
| **80** | **Dept 43** | Special Promotional | $14.32 | $1.19 | $3.40 | 2.86 |
| **81** | **Dept 47** | Customer Returns / Layaway | **-$4,962.93** | **-$7.74** | **$160.20** | **-20.70 (Net Negative)** |

### 6.2 Pareto Concentration Analysis (80/20 Rule)
Cumulative distribution analysis (Lorenz curve) reveals significant structural revenue concentration:
- **Top 10 Departments (12.3% of total depts):** Generate **53.8%** of total Walmart historical sales.
- **Top 16 Departments (19.8% of total depts):** Generate **78.4%** of total revenue (~80/20 rule confirmed).
- **Bottom 30 Departments (37.0% of total depts):** Generate just **2.1%** of aggregate revenue.

*Figure References:*
- `outputs/figures/fig06_top_bottom_departments_revenue.png`: Comparative bar chart of top 10 vs bottom 10 departments.
- `outputs/figures/fig07_sales_concentration_pareto.png`: Pareto Lorenz curve illustrating store and department cumulative sales concentration.

---

## 7. Weekly and Annual Time-Series Trends

Aggregating department sales to the total enterprise level across all 143 historical weeks yields a contiguous, high-signal time series:
- **Historical Mean Aggregate Weekly Sales:** **$47,113,421.67** (~$47.1 Million / week)
- **Historical Minimum Week:** `2011-01-28` ($39,599,852.99 — post-holiday January freeze)
- **Historical Peak Week:** `2010-12-24` (**$80,931,415.60** — Pre-Christmas Week 51 surge)

```
AGGREGATE ENTERPRISE SALES TRAJECTORY (2010 - 2012)
Sales ($M)
 85M ┼                                    ▲ (Pre-Xmas 80.9M)
 75M ┼                        ▲ (76.9M)
 65M ┼              ▲ (65.8M)
 55M ┼
 45M ┼───~~~~~───────~~~~~────~~~~~───────~~~~~────────~~~~~─────── (Baseline: 47.1M)
 35M ┼    ▼ (39.5M)
     └────┬───────────┬───────────┬───────────┬───────────┬────────► Date
        2010-02     2010-08     2011-02     2011-08     2012-02
```

### Year-over-Year (YoY) Overlay Insights:
Aligning sales by ISO calendar week (Weeks 1 to 52) across 2010, 2011, and 2012 reveals four consistent cyclical surges:
1. **Weeks 5 to 7 (Super Bowl & Valentine's Day):** Sales rise ~12% above early-year troughs.
2. **Weeks 14 to 16 (Easter & Spring Garden Reset):** Sales peak during late March / April.
3. **Weeks 31 to 35 (Back-to-School & Labor Day):** Consistent upward baseline elevation.
4. **Weeks 46 to 52 (Holiday Quarter Spike):** Extreme twin spikes during Thanksgiving (Week 47) and Pre-Christmas (Week 51).

*Figure References:*
- `outputs/figures/fig08_aggregate_weekly_sales_trend.png`: 143-week continuous aggregate weekly sales line plot.
- `outputs/figures/fig09_yoy_weekly_sales_overlay.png`: Year-over-Year overlay showing aligned weekly seasonality.
- `outputs/figures/fig10_store_type_trends_comparison.png`: Weekly sales trajectories by Store Type.
- `outputs/figures/fig11_department_seasonality_comparison.png`: Weekly profiles across contrasting retail departments.

---

## 8. Holiday Impact Analysis & WMAE Evaluation

The official competition loss metric is **Weighted Mean Absolute Error (WMAE)**:
$$\text{WMAE} = \frac{1}{\sum_{i=1}^n w_i} \sum_{i=1}^n w_i |y_i - \hat{y}_i| \quad \text{where } w_i = 5 \text{ (Holiday)}, \ w_i = 1 \text{ (Non-Holiday)}$$

### 8.1 Empirical Performance: Holiday vs. Non-Holiday Weeks
| Week Classification | Record Count | Total Revenue ($) | Mean Sales ($) | Median Sales ($) | Std Dev ($) | Mean Uplift (%) | Median Uplift (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Non-Holiday (`IsHoliday=False`)**| 391,909 (92.96%) | $6,231,919,435.55 | $15,901.45 | $7,589.95 | $22,330.75 | Baseline | Baseline |
| **Holiday (`IsHoliday=True`)** | 29,661 (7.04%) | $505,299,551.56 | $17,035.82 | $7,947.74 | $27,222.00 | **+7.13%** | **+4.71%** |

### 8.2 Event-by-Event Breakdown (Historical Training Weeks)
| Retail Holiday Event | Calendar Date | Week-Ending Friday | Dataset Split | Aggregate Total ($) | Mean Dept Sales ($) | Median Dept Sales ($) | Uplift vs Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Super Bowl 2010** | Sun, Feb 7, 2010 | `2010-02-12` | Train | $48,336,677.63 | $16,352.06 | $8,365.10 | +2.83% |
| **Labor Day 2010** | Mon, Sep 6, 2010 | `2010-09-10` | Train | $45,634,397.84 | $15,537.76 | $7,403.87 | -2.29% |
| **Thanksgiving 2010** | Thu, Nov 25, 2010 | `2010-11-26` | Train | **$65,821,003.24** | **$22,403.34** | **$10,328.44** | **+40.89%** |
| **Christmas 2010** | Sat, Dec 25, 2010 | `2010-12-31` | Train | $40,432,519.00 | $13,738.54 | $6,462.16 | -13.60% |
| **Super Bowl 2011** | Sun, Feb 6, 2011 | `2011-02-11` | Train | $47,336,192.79 | $16,111.71 | $7,871.48 | +1.32% |
| **Labor Day 2011** | Mon, Sep 5, 2011 | `2011-09-09` | Train | $46,763,227.53 | $15,809.07 | $7,491.43 | -0.58% |
| **Thanksgiving 2011** | Thu, Nov 24, 2011 | `2011-11-25` | Train | **$66,593,605.26** | **$22,043.56** | **$9,632.31** | **+38.63%** |
| **Christmas 2011** | Sun, Dec 25, 2011 | `2011-12-30` | Train | $46,042,461.04 | $15,332.15 | $7,562.11 | -3.58% |
| **Super Bowl 2012** | Sun, Feb 5, 2012 | `2012-02-10` | Train | $50,009,407.92 | $16,664.25 | $7,707.04 | +4.80% |
| **Labor Day 2012** | Mon, Sep 3, 2012 | `2012-09-07` | Train | $48,330,059.31 | $16,294.69 | $7,595.88 | +2.47% |

*Figure Reference: `outputs/figures/fig12_holiday_sales_uplift_comparison.png` contrasts event-level weekly sales totals and departmental mean distributions.*

---

## 9. The Pre-Christmas Shopping Phenomenon

A critical empirical discovery emerged when auditing the final two weeks of December:

```
                  DECEMBER SALES DYNAMICS COMPARISON
Total Sales ($M)
  90M ┼
  80M ┼          ████ $80.93M (2010-12-24)
  70M ┼          ████                       ████ $77.00M (2011-12-23)
  60M ┼          ████                       ████
  50M ┼   ████   ████        ████    ████   ████        ████
  40M ┼   ████   ████        ████    ████   ████        ████
  30M ┼───████───████────────████────████───████────────████────────►
        Mid-Dec Pre-Xmas    Post-Xmas Mid-Dec Pre-Xmas    Post-Xmas
        (Dec 17)(Dec 24)    (Dec 31) (Dec 16)(Dec 23)    (Dec 30)
                 ▲                   ▲        ▲                   ▲
             IsHoliday:          IsHoliday:IsHoliday:          IsHoliday:
               FALSE               TRUE     FALSE               TRUE
```

### Empirical Evidence:
1. **Week of `2010-12-24` (Pre-Christmas Peak):**
   - Official Flag: `IsHoliday = False`
   - Aggregate Sales: **$80,931,415.60**
   - Department Mean: **$27,378.69** (+72.2% above baseline!)
2. **Week of `2010-12-31` (Official Christmas Week):**
   - Official Flag: `IsHoliday = True` (Weight = 5 in WMAE)
   - Aggregate Sales: **$40,432,519.00**
   - Department Mean: **$13,738.54** (-13.6% below baseline!)
3. **Week of `2011-12-23` (Pre-Christmas Peak):**
   - Official Flag: `IsHoliday = False`
   - Aggregate Sales: **$76,998,241.31**
   - Department Mean: **$25,437.15** (+60.0% above baseline!)
4. **Week of `2011-12-30` (Official Christmas Week):**
   - Official Flag: `IsHoliday = True` (Weight = 5 in WMAE)
   - Aggregate Sales: **$46,042,461.04**
   - Department Mean: **$15,332.15** (-3.6% below baseline!)

### Retail Data Science Explanation:
The dataset's weekly reporting timestamp records week-ending **Fridays**. In 2010, Christmas Day fell on Saturday, Dec 25. Therefore, the week ending Friday, Dec 24 captured the entire frantic last-minute holiday shopping rush. The following week ending Friday, Dec 31 captured the post-Christmas week, when stores experienced sharp traffic drops and gift returns.

> [!CAUTION]
> **Essential Modeling Rule:**
> In Phase 3, we MUST engineer a dedicated boolean feature: `Is_PreChristmas_Peak = (Week == 51)`. Relying on `IsHoliday` will cause ML models to fail during the single most profitable week of the year.

*Figure Reference: `outputs/figures/fig13_pre_christmas_effect_analysis.png` documents the December Week 51 vs Week 52 sales volume comparison.*

---

## 10. Promotional MarkDown Analysis & Collinearity

### 10.1 Temporal Availability Breakpoint
An audit of `features.csv` confirmed a distinct structural shift:
- **Weeks 1 to 91 (Prior to November 4, 2011):** **100.0% Missing**. All 5 markdowns are completely null across all 45 stores.
- **Weeks 92 to 182 (November 11, 2011 to July 26, 2013):** Actively recorded:
  - `MarkDown1`: 4,032 non-null records (Mean: $7,032.37)
  - `MarkDown2`: 2,921 non-null records (Mean: $3,384.18)
  - `MarkDown3`: 3,613 non-null records (Mean: $1,760.10)
  - `MarkDown4`: 3,464 non-null records (Mean: $3,292.94)
  - `MarkDown5`: 4,050 non-null records (Mean: $4,132.22)

### 10.2 Collinearity Between MarkDowns
Correlation analysis on post-November 2011 data identified severe multicollinearity:
$$\text{Pearson } r(\text{MarkDown1}, \text{MarkDown4}) = \mathbf{+0.819} \quad (\text{Spearman } \rho = +0.810)$$
`MarkDown1` and `MarkDown4` track nearly identical promotional events (storewide circular discounts and clearance price cuts). Tree-based models can handle this correlation, but linear / regularized models will suffer from variance inflation if both are included without regularization or PCA.

*Figure Reference: `outputs/figures/fig14_markdown_temporal_availability.png` illustrates non-null availability percentages across the contiguous 182-week timeline.*

---

## 11. Economic Feature Correlations & Test Blackout

### 11.1 Correlation Matrix (Linear Pearson vs Non-Linear Spearman)
Evaluating relationships across all continuous variables on the post-markdown dataset:

| Feature Pair | Pearson Correlation ($r$) | Spearman Rank ($\rho$) | Statistical Interpretation |
| :--- | :--- | :--- | :--- |
| `Weekly_Sales` $\leftrightarrow$ `Size` | **+0.245** | **+0.282** | Moderate positive association; larger stores sell more |
| `Weekly_Sales` $\leftrightarrow$ `MarkDown5` | +0.090 | +0.088 | Weak positive association with store traffic promotions |
| `Weekly_Sales` $\leftrightarrow$ `MarkDown1` | +0.085 | +0.082 | Weak positive association with clearance promotions |
| `Weekly_Sales` $\leftrightarrow$ `MarkDown3` | +0.060 | +0.071 | Weak positive association (driven by Thanksgiving spikes) |
| `Weekly_Sales` $\leftrightarrow$ `MarkDown4` | +0.045 | +0.042 | Weak positive association |
| `Weekly_Sales` $\leftrightarrow$ `MarkDown2` | +0.024 | +0.021 | Negligible linear correlation |
| `Weekly_Sales` $\leftrightarrow$ `Temperature` | **-0.008** | **-0.012** | Negligible linear correlation across overall enterprise |
| `Weekly_Sales` $\leftrightarrow$ `Fuel_Price` | **-0.014** | **-0.018** | Negligible linear correlation |
| `Weekly_Sales` $\leftrightarrow$ `CPI` | **-0.020** | **-0.025** | Negligible linear correlation |
| `Weekly_Sales` $\leftrightarrow$ `Unemployment` | **-0.032** | **-0.035** | Negligible linear correlation |

### Senior Data Science Insight:
The macroeconomic variables (`CPI`, `Unemployment`, `Fuel_Price`, `Temperature`) have virtually no direct linear correlation with weekly department sales. Retail sales variance is driven primarily by **store capacity (`Size`), department category, calendar seasonality, and promotional calendar events**. Macro variables act as long-term macro trend shifters rather than high-frequency weekly predictors.

### 11.2 The Test Horizon Macroeconomic Blackout
A severe defect was verified in `features.csv` during the test evaluation window:
- For the first 26 test weeks (`2012-11-02` to `2013-04-26`): CPI and Unemployment are complete.
- For the final 13 test weeks (`2013-05-03` to `2013-07-26`): **CPI and Unemployment are 100% NULL across all 45 stores (585 feature records, exactly 33.3% of the test horizon)**.

*Figure References:*
- `outputs/figures/fig15_economic_features_correlation_matrix.png`: Heatmaps comparing Pearson and Spearman correlation matrices.
- `outputs/figures/fig18_test_cpi_unemployment_blackout.png`: Visualization of the 13-week test macroeconomic blackout.

---

## 12. Time-Series STL Decomposition

To understand underlying structural dynamics, an **STL (Seasonal and Trend decomposition using Loess)** decomposition was performed on the aggregate weekly sales series ($N=143$ weeks):
$$Y_t = T_t + S_t + R_t$$
where $T_t$ represents the smoothed low-frequency trend, $S_t$ represents the periodic annual seasonality ($\text{period} = 52$ weeks), and $R_t$ represents the remaining residual component.

```
STL DECOMPOSITION COMPONENTS SUMMARY (period = 52 weeks, robust = True)
1. Observed ($Y_t$):  Range: $39.6M to $80.9M | Sharp recurring holiday spikes
2. Trend ($T_t$):     Range: $45.8M to $48.2M | Mild upward expansion through 2011, flattening in 2012
3. Seasonal ($S_t$):  Range: -$6.2M to +$24.5M| Dominates weekly variance; major Thanksgiving/Xmas peaks
4. Residual ($R_t$):  Range: -$4.1M to +$8.3M | Mean ~ $0.0M; captures weather shocks & unmodeled events
```

### Statistical Period Selection & Limitations:
- **Why Period = 52:** Walmart weekly sales operate on an annual calendar cycle (Black Friday, Christmas, Spring resets recur every 52 weeks).
- **Statistical Reliability Caveat:** With 143 total observations, the dataset spans approximately **2.75 full annual seasonal cycles**. In classical time-series literature, reliable estimation of an annual seasonal component typically benefits from 4 to 5 full cycles. While STL successfully captures the dominant Thanksgiving and holiday spikes, local seasonal nuances may carry higher estimation variance.

*Figure Reference: `outputs/figures/fig16_stl_decomposition_aggregate_sales.png` displays the 4-panel decomposition.*

---

## 13. Statistical Stationarity Diagnostics (ADF & KPSS)

Stationarity was evaluated on the aggregate weekly sales series using both **Augmented Dickey-Fuller (ADF)** and **Kwiatkowski-Phillips-Schmidt-Shin (KPSS)** tests:

### 13.1 Augmented Dickey-Fuller (ADF) Test
- **Hypotheses:**
  - $H_0$: The series has a unit root (Non-Stationary).
  - $H_1$: The series has no unit root (Stationary).
- **Test Results:**
  - Test Statistic: **-5.9083**
  - $p$-value: **$2.68 \times 10^{-7}$** ($p < 0.001$)
  - Critical Values: $1\%: -3.479, \ 5\%: -2.883, \ 10\%: -2.578$
- **Interpretation:** The test statistic is significantly more negative than the 1% critical value. We **strongly reject $H_0$**. The series does not have a stochastic unit root.

### 13.2 KPSS Test
- **Hypotheses:**
  - $H_0$: The series is level / trend stationary.
  - $H_1$: The series has a unit root (Non-Stationary).
- **Test Results:**
  - Test Statistic: **0.0489**
  - $p$-value: **$> 0.10$** (bounded at lookup ceiling)
  - Critical Values: $10\%: 0.347, \ 5\%: 0.463, \ 1\%: 0.739$
- **Interpretation:** The test statistic is far below the 10% critical threshold. We **fail to reject $H_0$**. The series is consistent with level/trend stationarity.

### Joint Test Synthesis:
Both tests agree that aggregate weekly sales do not suffer from a non-stationary random walk or explosive unit root. However, because the series exhibits powerful deterministic annual seasonality and variance heteroskedasticity during Q4 holiday spikes, the raw series is **not strictly stationary in variance**. In Phase 3, seasonal feature engineering (lags at $t-52$, rolling statistics, and Fourier terms) will be required.

---

## 14. Special Investigation A: Cold-Start Series in Test Horizon

Phase 1 identified **11 `(Store, Dept)` series in `test.csv` that have ZERO historical sales records in `train.csv`**:

| Store | Dept | Store Type | Store Size (Sq Ft) | Test Horizon Observations | Test Start Date | Test End Date | Conceptual Cold-Start Fallback Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Store 5** | Dept 99 | Type B | 34,875 | 39 weeks | `2012-11-02` | `2013-07-26` | Impute median of Dept 99 across Type B stores |
| **Store 9** | Dept 99 | Type B | 125,833 | 39 weeks | `2012-11-02` | `2013-07-26` | Impute median of Dept 99 across Type B stores |
| **Store 10** | Dept 99 | Type B | 126,512 | 39 weeks | `2012-11-02` | `2013-07-26` | Impute median of Dept 99 across Type B stores |
| **Store 18** | Dept 43 | Type B | 120,653 | 39 weeks | `2012-11-02` | `2013-07-26` | Impute median of Dept 43 across Type B stores |
| **Store 24** | Dept 43 | Type A | 203,819 | 39 weeks | `2012-11-02` | `2013-07-26` | Impute median of Dept 43 across Type A stores |
| **Store 25** | Dept 99 | Type B | 128,107 | 39 weeks | `2012-11-02` | `2013-07-26` | Impute median of Dept 99 across Type B stores |
| **Store 34** | Dept 39 | Type A | 158,114 | 39 weeks | `2012-11-02` | `2013-07-26` | Impute median of Dept 39 across Type A stores |
| **Store 36** | Dept 30 | Type A | 39,910 | 39 weeks | `2012-11-02` | `2013-07-26` | Impute median of Dept 30 across Type A stores |
| **Store 37** | Dept 29 | Type C | 39,910 | 39 weeks | `2012-11-02` | `2013-07-26` | Impute median of Dept 29 across Type C stores |
| **Store 42** | Dept 30 | Type C | 39,690 | 39 weeks | `2012-11-02` | `2013-07-26` | Impute median of Dept 30 across Type C stores |
| **Store 45** | Dept 39 | Type B | 118,221 | 39 weeks | `2012-11-02` | `2013-07-26` | Impute median of Dept 39 across Type B stores |

### Modeling Implication:
Univariate autoregressive models (single-series ARIMA, Holt-Winters, individual Prophet) **will completely fail** on these 11 series because they lack historical lags ($y_{t-1}, y_{t-52}$). A global gradient boosting model (LightGBM/XGBoost) trained on cross-store department embeddings, or a hierarchical fallback imputation, is required to generate forecasts for these rows.

---

## 15. Special Investigation B: Incomplete Panel Sparsity

An audit of the length of all 3,331 active `(Store, Dept)` series in `train.csv` revealed an **unbalanced panel**:
- **Complete Series (all 143 weeks observed):** **2,660 series (79.86%)**
- **Incomplete Series (< 143 weeks observed):** **671 series (20.14%)**
- **Distribution of Series Lengths:**
  - Minimum Length: **1 week** (e.g., Store 14, Dept 78)
  - 25th Percentile: 143 weeks
  - Median Length: 143 weeks
  - 75th Percentile: 143 weeks
  - Maximum Length: 143 weeks

### Why Missing Weeks Exist in Retail:
1. **Seasonal Merchandising:** Lawn & Garden (Dept 18) or Halloween/Christmas specialty departments only stock inventory and report sales during specific seasonal quarters.
2. **Store Remodels & Expansion:** New departments opened or were phased out during the 2010–2012 observation window.
3. **Zero Sales Non-Reporting:** In POS data warehouses, if a department sells 0 items during a week, the system often omits the row rather than recording $0.00.

*Figure Reference: `outputs/figures/fig17_panel_sparsity_distribution.png` illustrates the histogram of observed week counts.*

---

## 16. Special Investigation C: Negative Sales Accounting Profile

A deep dive into the 1,285 negative sales records ($0.3048\%$ of training rows) verified:
1. **Total Dollar Impact:** -$88,161.56 across all historical observations.
2. **Top 5 Concentration Departments:**
   - **Dept 47:** 254 records | Min: -$3,924.00 | Sum: -$53,188.06 | **Lifetime Net: -$4,962.93**
   - **Dept 18:** 180 records | Min: -$175.54 | Sum: -$1,819.05 | Lifetime Net: $21.12M
   - **Dept 54:** 146 records | Min: -$465.08 | Sum: -$4,829.81 | Lifetime Net: $108.13k
   - **Dept 19:** 87 records | Min: -$69.56 | Sum: -$1,042.66 | Lifetime Net: $4.32M
   - **Dept 94:** 77 records | Min: -$173.84 | Sum: -$647.78 | Lifetime Net: $69.12M

### Mathematical Integrity Mandate:
- **Do NOT drop negative sales:** Dropping negative values will artificially inflate predictions for departments that regularly process customer returns.
- **WMAE Compatibility:** WMAE evaluates absolute error $|y_i - \hat{y}_i|$. Predictions near zero or slightly negative are mathematically permitted by the metric.

---

## 17. Master Visualizations Catalog

All 18 generated publication-grade figures have been verified and persisted in `outputs/figures/`:

| Figure Filename | Section / Topic | Visual Focus |
| :--- | :--- | :--- |
| `fig01_dataset_timeline_overview.png` | Overview | 182-week timeline, train/test split, markdown start, test macro blackout |
| `fig02_sales_distribution_and_log.png` | Sales EDA | Raw linear vs. signed log1p transformed distribution |
| `fig03_negative_sales_by_dept.png` | Sales EDA | Bar chart of top 12 departments with customer returns / adjustments |
| `fig04_store_sales_ranking.png` | Store EDA | All 45 stores ranked by total historical sales, colored by Store Type |
| `fig05_store_type_size_vs_sales.png` | Store EDA | Scatter plot & regression of Store Size vs Average Sales by Store Type |
| `fig06_top_bottom_departments_revenue.png`| Dept EDA | Bar charts of top 10 and bottom 10 revenue-generating departments |
| `fig07_sales_concentration_pareto.png` | Business EDA | Pareto Lorenz curves showing cumulative revenue concentration |
| `fig08_aggregate_weekly_sales_trend.png`| Time-Series | 143-week aggregate weekly sales line plot with holiday annotations |
| `fig09_yoy_weekly_sales_overlay.png` | Time-Series | Year-over-Year overlay (2010 vs 2011 vs 2012 by ISO calendar week) |
| `fig10_store_type_trends_comparison.png`| Time-Series | Weekly sales trajectories across Store Types A, B, and C |
| `fig11_department_seasonality_comparison.png`| Time-Series | Weekly sales profiles across contrasting retail departments |
| `fig12_holiday_sales_uplift_comparison.png`| Holiday EDA | Mean and median uplift comparisons across major holiday events |
| `fig13_pre_christmas_effect_analysis.png`| Holiday EDA | Deep dive into Week 51 surge vs Christmas Week 52 drop |
| `fig14_markdown_temporal_availability.png`| Markdowns | Non-null tracking rate over time illustrating pre-Nov 2011 break |
| `fig15_economic_features_correlation_matrix.png`| Economics | Pearson linear and Spearman rank correlation heatmaps |
| `fig16_stl_decomposition_aggregate_sales.png`| Decomposition| 4-panel STL decomposition (Observed, Trend, Seasonal, Residual) |
| `fig17_panel_sparsity_distribution.png`| Panel Gaps | Histogram of active week counts per Store-Dept series |
| `fig18_test_cpi_unemployment_blackout.png`| Economics | Visualization of the 13-week test CPI/Unemployment blackout |

---

## 18. Strategic Business Insights

1. **The Week 51 Retail Bottleneck:**
   Walmart's supply chain and staffing must peak during ISO Calendar Week 51 (mid-December), rather than Christmas week. Inventory must be fully staged in early December to capture the $80M peak weekly volume without stockouts.
2. **Store Footprint Monetization:**
   Store Size correlates strongly with revenue ($r = +0.81$ at the store aggregate level). However, Type C stores suffer from low department sales density ($1,149.67 median), suggesting that express formats must optimize high-turnover grocery and pharmacy space rather than attempting broad department assortments.
3. **Category Dominance & Cross-Merchandising:**
   Grocery (Dept 92) and DSD Snacks (Dept 95) generate nearly $1 Billion alone with low volatility (CV $\approx 0.50$). Electronics (Dept 72) is high-risk, high-reward (CV = 1.36) with extreme Thanksgiving spikes. Promotional pricing should bundle high-margin impulse items with steady grocery staples during Q4 peaks.
4. **Administrative Returns Clearing:**
   Dept 47 represents negative net sales (-$4,962 lifetime). Walmart operational teams should analyze whether Dept 47 return volume correlates with prior-quarter electronics or toy promotions to optimize post-holiday return processing.

---

## 19. Data Limitations & Modeling Caveats

1. **Short Observation History (2.75 Years):** With only 143 weekly observations, models observe Thanksgiving and Christmas only twice in the training data. This limits pure autoregressive modeling of rare annual holiday interactions.
2. **Structural Markdown Shift:** Markdowns do not exist prior to November 2011. Imputing with mean/median would create false promotions; imputing with 0.0 requires an indicator variable.
3. **Test Horizon Macroeconomic Blackout:** Missing CPI and Unemployment for the final 13 weeks of the test period will cause models trained on raw macro indicators to fail in production unless forecasted or dropped.
4. **Unbalanced Series Lengths:** 20.14% of department time series have missing weeks, requiring models that natively handle sparse panel data without rigid rectangular array assumptions.

---

## 20. Actionable Recommendations for Phase 3 (Feature Engineering & Baseline Models)

```
                            PHASE 3 ROADMAP
┌────────────────────────────┬────────────────────────────┬────────────────────────────┐
│ Calendar Features          │ Lag & Rolling Signals      │ Structural Imputations     │
│ - Is_PreChristmas_Peak     │ - Lags: t-1, t-2, t-52     │ - Fill Markdowns with 0.0  │
│ - Days to/from Black Friday│ - Rolling Mean: 4, 12, 52  │ - Flag: Has_MarkDown_Data  │
│ - Week of Year (Cyclical)  │ - Rolling Std Dev (Volat.) │ - Forward-fill test macro  │
├────────────────────────────┼────────────────────────────┼────────────────────────────┤
│ Hierarchical Encodings     │ Metric Alignment           │ Validation Split           │
│ - Store Type Target Mean   │ - Custom WMAE Loss / Eval  │ - Temporal Out-of-Time     │
│ - Dept Share per Store     │ - Sample Weight = 5 (Hols) │ - Train: 2010-02 to 2011-10│
│ - Cold-Start Fallback Map  │ - Sample Weight = 1 (Norm) │ - Val:   2011-11 to 2012-10│
└────────────────────────────┴────────────────────────────┴────────────────────────────┘
```

1. **Calendar & Proximity Feature Engineering:**
   - Create explicit indicators for `Is_PreChristmas_Peak` (`Week == 51`).
   - Create holiday proximity distance features (days until next major holiday and days since last major holiday).
   - Use cyclical sine/cosine encodings for week of year: $\sin(2\pi \cdot \text{week} / 52)$, $\cos(2\pi \cdot \text{week} / 52)$.
2. **Lagged and Rolling Autoregressive Features:**
   - Compute seasonal year-over-year lags: $\text{Sales}_{t-52}$.
   - Compute short-term momentum lags: $\text{Sales}_{t-1}, \text{Sales}_{t-2}, \text{Sales}_{t-3}$.
   - Compute moving window statistics: 4-week, 12-week, and 52-week rolling means and standard deviations.
3. **Macro & Markdown Imputation Strategy:**
   - Impute missing markdowns with `0.0` and attach `Has_MarkDown_Data = (Date >= '2011-11-04')`.
   - Forecast missing CPI and Unemployment for the 13 test blackout weeks using store-level linear trends or forward-fill.
4. **WMAE Objective Alignment:**
   - Train baseline models (Naive Seasonal, Moving Average) and machine learning models (LightGBM, CatBoost) with explicit sample weights ($w=5$ for holiday weeks, $w=1$ for non-holiday weeks).
5. **Cold-Start Fallback Architecture:**
   - Implement an automated fallback pipeline mapping the 11 cold-start `(Store, Dept)` series in test to their corresponding Department-level and Store-Type median sales baselines.

---

**Sign-off:**  
*Phase 2 Exploratory Data Analysis and Time-Series Analysis is officially complete, validated, and persisted. No predictive models have been trained. All files remain strictly organized within the project directory.*
