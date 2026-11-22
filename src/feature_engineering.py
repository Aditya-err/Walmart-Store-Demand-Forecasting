"""
Walmart Sales Forecasting — Feature Engineering Engine
Module: src/feature_engineering.py
Author: Lead Data Scientist & ML Engineer

Responsibilities:
- Build leak-free historical sales lags (1, 2, 4, 52 weeks) grouped by (Store, Dept).
- Apply shift(1) strictly before calculating rolling statistics (mean, std over 4, 12, 52 weeks).
- Compute year-over-year previous-year comparison features (sales differences and growth ratios).
- Extract calendar dynamics (Year, Month, Quarter, ISO Week, Day of Year, Cyclical Sin/Cos encodings).
- Generate holiday and retail event indicators (Super Bowl, Labor Day, Thanksgiving, Pre-Christmas Peak, Christmas).
- Engineer holiday proximity features (days to Thanksgiving and Christmas).
- Encode Store attributes (Type one-hot, Log Size, Scaled Size).
- Integrate hierarchical fallback values and imputation indicators for cold-start and missing lags.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("WalmartFeatureEngineering")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract calendar date features, cyclical wave encodings, and day-of-year indicators.
    """
    df = df.copy()
    dates = pd.to_datetime(df["Date"])
    
    df["Year"] = dates.dt.year
    df["Month"] = dates.dt.month
    df["Quarter"] = dates.dt.quarter
    # ISO calendar week (1 to 52/53)
    df["Week"] = dates.dt.isocalendar().week.astype(int)
    df["Day_of_Year"] = dates.dt.dayofyear

    # Cyclical trigonometric transforms for week-of-year (annual cycle: ~52.1775 weeks)
    cycle = 52.1775
    df["Week_Sin"] = np.sin(2 * np.pi * df["Week"] / cycle)
    df["Week_Cos"] = np.cos(2 * np.pi * df["Week"] / cycle)

    # Cyclical transforms for month (12 months)
    df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12.0)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12.0)

    return df


def add_retail_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer specialized retail event flags and holiday proximity distances.
    Includes the critical empirical discovery: Pre-Christmas Week 51 surge.
    """
    df = df.copy()
    dates = pd.to_datetime(df["Date"])
    weeks = dates.dt.isocalendar().week.astype(int)
    
    # Official holiday flag
    df["IsHoliday"] = df["IsHoliday"].astype(int)

    # 1. Major retail holiday week flags
    # Super Bowl: Week 6 (early February)
    df["Is_SuperBowl"] = (weeks == 6).astype(int)

    # Labor Day: Week 36 (early September)
    df["Is_LaborDay"] = (weeks == 36).astype(int)

    # Thanksgiving / Black Friday: Week 47 (late November)
    df["Is_Thanksgiving"] = (weeks == 47).astype(int)

    # Pre-Christmas Shopping Surge: Week 51 (the highest volume week of the year)
    df["Is_PreChristmas"] = (weeks == 51).astype(int)

    # Christmas Week: Week 52 (post-Christmas week)
    df["Is_Christmas"] = (weeks == 52).astype(int)

    # 2. Continuous Proximity to Thanksgiving and Christmas (in days)
    # Thanksgiving occurs on the 4th Thursday of November
    thanksgiving_dates = {
        2010: pd.Timestamp("2010-11-25"),
        2011: pd.Timestamp("2011-11-24"),
        2012: pd.Timestamp("2012-11-22"),
        2013: pd.Timestamp("2013-11-28"),
    }
    
    # Christmas Day is December 25
    christmas_dates = {
        2010: pd.Timestamp("2010-12-25"),
        2011: pd.Timestamp("2011-12-25"),
        2012: pd.Timestamp("2012-12-25"),
        2013: pd.Timestamp("2013-12-25"),
    }

    days_to_tg = []
    days_to_xmas = []
    
    for d in dates:
        yr = d.year
        tg_target = thanksgiving_dates.get(yr, pd.Timestamp(f"{yr}-11-25"))
        xmas_target = christmas_dates.get(yr, pd.Timestamp(f"{yr}-12-25"))
        
        days_to_tg.append((tg_target - d).days)
        days_to_xmas.append((xmas_target - d).days)

    df["Days_To_Thanksgiving"] = days_to_tg
    df["Days_To_Christmas"] = days_to_xmas

    return df


def add_store_department_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode store format, size transformations, and departmental attributes.
    """
    df = df.copy()

    # One-hot encode Store Type (A, B, C)
    for stype in ["A", "B", "C"]:
        df[f"Type_{stype}"] = (df["Type"] == stype).astype(int)

    # Non-linear size transformations
    df["Log_Size"] = np.log(df["Size"].astype(float))
    min_sz, max_sz = 34875.0, 219622.0
    df["Size_Scaled"] = (df["Size"] - min_sz) / (max_sz - min_sz)

    return df


def build_regular_time_series_grid(historical_sales_df: pd.DataFrame, 
                                   target_horizon_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a complete, regular rectangular weekly grid (W-FRI) for every active
    (Store, Dept) series from the earliest historical date through the end of the horizon.
    This guarantees that shifting by k rows is mathematically guaranteed to represent exactly k weeks.
    """
    logger.info("Constructing regular rectangular time-series grid for lag computation...")
    
    # Collect all unique (Store, Dept) pairs
    all_sd = set(zip(historical_sales_df["Store"], historical_sales_df["Dept"]))
    if target_horizon_df is not None:
        all_sd = all_sd.union(set(zip(target_horizon_df["Store"], target_horizon_df["Dept"])))
        
    sd_df = pd.DataFrame(list(all_sd), columns=["Store", "Dept"])
    
    # Determine global date bounds
    min_date = historical_sales_df["Date"].min()
    max_date = historical_sales_df["Date"].max()
    if target_horizon_df is not None:
        max_date = max(max_date, target_horizon_df["Date"].max())
        
    all_dates = pd.date_range(min_date, max_date, freq="W-FRI")
    dates_df = pd.DataFrame({"Date": all_dates})
    
    # Cartesian cross-join of all dates and all (Store, Dept) series
    grid = dates_df.merge(sd_df, how="cross")
    
    # Merge known historical sales into the grid
    sales_sub = historical_sales_df[["Store", "Dept", "Date", "Weekly_Sales"]].drop_duplicates(subset=["Store", "Dept", "Date"])
    grid = grid.merge(sales_sub, on=["Store", "Dept", "Date"], how="left")
    
    # Sort strictly by Store, Dept, and Date
    grid = grid.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)
    logger.info(f"Grid built successfully: {grid.shape[0]:,} cells across {len(all_sd):,} series and {len(all_dates)} weeks.")
    return grid


def compute_historical_lags_and_rolling(grid_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lag and rolling features on the regular rectangular grid.
    STRICT DATA LEAKAGE PREVENTION:
    - shift(1) is applied strictly before calculating any rolling statistics.
    - Rolling calculations never observe the contemporaneous week's Weekly_Sales.
    """
    logger.info("Computing historical lags and rolling features on regular grid...")
    grid = grid_df.copy()
    grp = grid.groupby(["Store", "Dept"])["Weekly_Sales"]
    
    # Lags (1, 2, 4, 52 weeks)
    grid["Sales_Lag_1"] = grp.shift(1)
    grid["Sales_Lag_2"] = grp.shift(2)
    grid["Sales_Lag_4"] = grp.shift(4)
    grid["Sales_Lag_52"] = grp.shift(52)

    # Rolling window statistics (Strictly shifted by 1!)
    shifted_sales = grp.shift(1)
    
    # 4-Week Rolling
    grid["Rolling_Mean_4"] = grp.apply(lambda s: s.shift(1).rolling(4, min_periods=1).mean()).reset_index(drop=True)
    grid["Rolling_Std_4"] = grp.apply(lambda s: s.shift(1).rolling(4, min_periods=2).std()).reset_index(drop=True)

    # 12-Week Rolling
    grid["Rolling_Mean_12"] = grp.apply(lambda s: s.shift(1).rolling(12, min_periods=1).mean()).reset_index(drop=True)
    grid["Rolling_Std_12"] = grp.apply(lambda s: s.shift(1).rolling(12, min_periods=2).std()).reset_index(drop=True)

    # 52-Week Rolling
    grid["Rolling_Mean_52"] = grp.apply(lambda s: s.shift(1).rolling(52, min_periods=1).mean()).reset_index(drop=True)
    grid["Rolling_Std_52"] = grp.apply(lambda s: s.shift(1).rolling(52, min_periods=2).std()).reset_index(drop=True)

    # Previous-Year Comparisons (YoY Differences and Ratios)
    grid["Previous_Year_Sales_Diff"] = grid["Sales_Lag_1"] - grid["Sales_Lag_52"]
    grid["Previous_Year_Sales_Ratio"] = (grid["Sales_Lag_1"] + 1.0) / (np.abs(grid["Sales_Lag_52"]) + 1.0)
    grid["Rolling_4_vs_Lag_52_Ratio"] = (grid["Rolling_Mean_4"] + 1.0) / (np.abs(grid["Sales_Lag_52"]) + 1.0)

    logger.info("Lags and rolling features computed.")
    return grid


def map_lag_features_to_dataset(target_df: pd.DataFrame, 
                                grid_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map computed lag and rolling features from the regular rectangular grid back
    to the target observation dataframe (train, val, or test).
    Handles cold-start series by falling back to Hierarchical_Fallback_Median and
    setting Is_Lag_Imputed indicator.
    """
    logger.info(f"Mapping lag features to target dataset ({target_df.shape[0]:,} rows)...")
    feature_cols = [
        "Store", "Dept", "Date",
        "Sales_Lag_1", "Sales_Lag_2", "Sales_Lag_4", "Sales_Lag_52",
        "Rolling_Mean_4", "Rolling_Std_4",
        "Rolling_Mean_12", "Rolling_Std_12",
        "Rolling_Mean_52", "Rolling_Std_52",
        "Previous_Year_Sales_Diff", "Previous_Year_Sales_Ratio", "Rolling_4_vs_Lag_52_Ratio"
    ]
    grid_sub = grid_features_df[feature_cols].copy()
    
    merged = target_df.merge(grid_sub, on=["Store", "Dept", "Date"], how="left")
    
    if len(merged) != len(target_df):
        raise AssertionError("Row count altered during lag feature mapping!")

    # Impute missing lag_1 (cold-start series or initial weeks) using hierarchical fallback
    lag1_null = merged["Sales_Lag_1"].isnull()
    merged["Is_Lag_Imputed"] = lag1_null.astype(int)
    
    if lag1_null.any():
        logger.info(f"Imputing {lag1_null.sum():,} missing Sales_Lag_1 entries with Hierarchical_Fallback_Median...")
        merged["Sales_Lag_1"] = merged["Sales_Lag_1"].fillna(merged["Hierarchical_Fallback_Median"])
        merged["Rolling_Mean_4"] = merged["Rolling_Mean_4"].fillna(merged["Hierarchical_Fallback_Median"])
        merged["Rolling_Mean_12"] = merged["Rolling_Mean_12"].fillna(merged["Hierarchical_Fallback_Median"])
        merged["Rolling_Mean_52"] = merged["Rolling_Mean_52"].fillna(merged["Hierarchical_Fallback_Median"])

    # Fill rolling standard deviations with 0.0 when insufficient history exists (min_periods < 2)
    merged["Rolling_Std_4"] = merged["Rolling_Std_4"].fillna(0.0)
    merged["Rolling_Std_12"] = merged["Rolling_Std_12"].fillna(0.0)
    merged["Rolling_Std_52"] = merged["Rolling_Std_52"].fillna(0.0)

    return merged


def generate_full_feature_set(base_df: pd.DataFrame, 
                              grid_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Master pipeline: applies calendar, retail holiday, store attributes, and mapped lag features.
    """
    logger.info(f"Generating full feature set for dataset of shape {base_df.shape}...")
    df = base_df.copy()
    
    # 1. Calendar Features
    df = add_calendar_features(df)
    
    # 2. Retail Holiday Indicators & Proximity
    df = add_retail_holiday_features(df)
    
    # 3. Store Attributes & Non-Linear Encodings
    df = add_store_department_features(df)
    
    # 4. Map Historical Lags & Rolling Statistics
    df = map_lag_features_to_dataset(df, grid_features_df)
    
    logger.info(f"Full feature engineering completed. Output shape: {df.shape}")
    return df
