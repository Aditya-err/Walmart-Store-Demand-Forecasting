"""
Unit tests for feature engineering, lag computation, and data leakage prevention.
"""
import numpy as np
import pandas as pd
import pytest
from src.feature_engineering import (
    build_regular_time_series_grid,
    compute_historical_lags_and_rolling,
    add_calendar_features,
    add_retail_holiday_features,
    add_store_department_features
)


def test_target_leakage_prevention_rolling_mean():
    """
    CRITICAL TEST: Ensure rolling statistics strictly use shift(1)
    and NEVER include the current week's sales value (y_t).
    
    Given series with sales: [10.0, 20.0, 30.0, 40.0, 50.0]
    At week 5 (sales=50.0), rolling mean 4 MUST be:
      (10.0 + 20.0 + 30.0 + 40.0) / 4 = 25.0
    If leakage occurred (no shift), it would be:
      (20.0 + 30.0 + 40.0 + 50.0) / 4 = 35.0
    """
    dates = pd.date_range("2010-01-01", periods=5, freq="W-FRI")
    df = pd.DataFrame({
        "Store": [1]*5,
        "Dept": [1]*5,
        "Date": dates,
        "Weekly_Sales": [10.0, 20.0, 30.0, 40.0, 50.0]
    })
    
    grid = build_regular_time_series_grid(df, target_horizon_df=None)
    grid_with_lags = compute_historical_lags_and_rolling(grid)
    
    w5_row = grid_with_lags[grid_with_lags["Date"] == dates[4]].iloc[0]
    
    # 1. Lag 1 must be sales from week 4 (40.0), not week 5 (50.0)
    assert w5_row["Sales_Lag_1"] == 40.0
    
    # 2. Lag 2 must be sales from week 3 (30.0)
    assert w5_row["Sales_Lag_2"] == 30.0
    
    # 3. Rolling mean 4 must strictly be 25.0, NOT 35.0 (proof of zero leakage)
    assert w5_row["Rolling_Mean_4"] == 25.0
    
    # 4. At week 1, lag 1 must be NaN (no prior information)
    w1_row = grid_with_lags[grid_with_lags["Date"] == dates[0]].iloc[0]
    assert np.isnan(w1_row["Sales_Lag_1"])


def test_regular_grid_handles_temporal_gaps():
    """
    Verify that temporal gaps in historical series do NOT cause
    a raw row-shift to misalign time.
    
    Series with a 1-week gap:
    Week 1 (2010-01-01): 100.0
    Week 2 (2010-01-08): [MISSING from historical_sales_df]
    Week 3 (2010-01-15): 300.0
    
    At Week 3:
    Sales_Lag_1 must be NaN (since Week 2 had no sales recorded),
    NOT 100.0 (which was 2 weeks ago!).
    """
    df = pd.DataFrame({
        "Store": [1, 1],
        "Dept": [1, 1],
        "Date": pd.to_datetime(["2010-01-01", "2010-01-15"]),
        "Weekly_Sales": [100.0, 300.0]
    })
    
    grid = build_regular_time_series_grid(df, target_horizon_df=None)
    grid_with_lags = compute_historical_lags_and_rolling(grid)
    
    w3_row = grid_with_lags[grid_with_lags["Date"] == "2010-01-15"].iloc[0]
    
    # Sales_Lag_1 (1 week prior, 2010-01-08) was missing -> must be NaN on grid
    assert np.isnan(w3_row["Sales_Lag_1"])
    # Sales_Lag_2 (2 weeks prior, 2010-01-01) had sales -> must be 100.0
    assert w3_row["Sales_Lag_2"] == 100.0


def test_calendar_features():
    """Verify calendar feature extraction and cyclical transformations."""
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2010-02-05", "2010-12-31"])
    })
    
    cal_df = add_calendar_features(df)
    
    assert cal_df.loc[0, "Year"] == 2010
    assert cal_df.loc[0, "Month"] == 2
    assert cal_df.loc[0, "Quarter"] == 1
    assert cal_df.loc[0, "Week"] == 5
    assert cal_df.loc[0, "Day_of_Year"] == 36
    
    assert cal_df.loc[1, "Month"] == 12
    assert cal_df.loc[1, "Week"] == 52
    assert cal_df.loc[1, "Day_of_Year"] == 365
    
    # Check cyclical bounds
    assert -1.0 <= cal_df["Week_Sin"].min() <= cal_df["Week_Sin"].max() <= 1.0
    assert -1.0 <= cal_df["Week_Cos"].min() <= cal_df["Week_Cos"].max() <= 1.0


def test_holiday_proximity_and_retail_features():
    """Verify retail holiday flags and Pre-Christmas peak detection."""
    df = pd.DataFrame({
        "Date": pd.to_datetime([
            "2010-02-12",  # Super Bowl (Week 6)
            "2010-09-10",  # Labor Day (Week 36)
            "2010-11-26",  # Thanksgiving / Black Friday (Week 47)
            "2010-12-24",  # Pre-Christmas (Week 51)
            "2010-12-31",  # Christmas week (Week 52)
        ]),
        "IsHoliday": [True, True, True, False, True]
    })
    
    res = add_retail_holiday_features(df)
    
    # Week 51 (2010-12-24) is Pre-Christmas peak!
    assert res.loc[3, "Is_PreChristmas"] == 1
    assert res.loc[0, "Is_PreChristmas"] == 0
    
    # Thanksgiving (Week 47)
    assert res.loc[2, "Is_Thanksgiving"] == 1
    assert res.loc[0, "Is_Thanksgiving"] == 0
    
    # Super Bowl (Week 6)
    assert res.loc[0, "Is_SuperBowl"] == 1
    assert res.loc[1, "Is_SuperBowl"] == 0
    
    # Distance to Thanksgiving / Christmas
    assert "Days_To_Thanksgiving" in res.columns
    assert "Days_To_Christmas" in res.columns


def test_store_features():
    """Verify store categorical encoding and size transformations."""
    df = pd.DataFrame({
        "Type": ["A", "B", "C"],
        "Size": [150000, 100000, 40000]
    })
    
    res = add_store_department_features(df)
    
    assert res.loc[0, "Type_A"] == 1
    assert res.loc[0, "Type_B"] == 0
    assert res.loc[1, "Type_B"] == 1
    assert res.loc[2, "Type_C"] == 1
    
    assert "Log_Size" in res.columns
    assert "Size_Scaled" in res.columns
