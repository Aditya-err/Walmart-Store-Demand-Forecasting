"""
Unit and integration tests for baseline forecasting models in Walmart Sales Forecasting.
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.baseline_models import (
    NaiveModel,
    SeasonalNaiveModel,
    MovingAverageModel,
    ExponentialSmoothingModel,
    HistoricalMedianModel,
    HierarchicalFallbackModel
)
from src.preprocessing import HierarchicalFallbackManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
if not DATA_DIR.exists():
    DATA_DIR = PROJECT_ROOT / "walmart-recruiting-store-sales-forecasting" / "data" / "processed"


def test_naive_model_synthetic():
    """Verify NaiveModel persists the last historical value and handles cold-starts."""
    train_df = pd.DataFrame({
        "Store": [1, 1, 1],
        "Dept": [1, 1, 1],
        "Date": pd.to_datetime(["2010-01-01", "2010-01-08", "2010-01-15"]),
        "Weekly_Sales": [100.0, 150.0, 200.0],
        "Type": ["A", "A", "A"]
    })
    
    val_df = pd.DataFrame({
        "Store": [1, 2],  # Store 1 is known, Store 2 is cold-start
        "Dept": [1, 1],
        "Date": pd.to_datetime(["2010-01-22", "2010-01-22"]),
        "Type": ["A", "A"]
    })
    
    model = NaiveModel().fit(train_df)
    preds = model.predict(val_df)
    
    # Store 1, Dept 1 must persist last value: 200.0
    assert preds[0] == 200.0
    # Store 2, Dept 1 (cold start) falls back to (Type A, Dept 1) median = 150.0
    assert preds[1] == 150.0


def test_seasonal_naive_synthetic():
    """Verify SeasonalNaiveModel retrieves exact 52-week lag value."""
    # 53 weeks of history
    dates_train = pd.date_range("2010-01-01", periods=53, freq="W-FRI")
    sales_train = [float(i * 10) for i in range(53)]
    
    train_df = pd.DataFrame({
        "Store": [1] * 53,
        "Dept": [1] * 53,
        "Date": dates_train,
        "Weekly_Sales": sales_train,
        "Type": ["A"] * 53
    })
    
    # Validation point: 52 weeks after week 0 (2010-01-01) is week 52 (2010-12-31)
    # Validation point at week 53: 52 weeks prior was week 1 (2010-01-08) -> sales = 10.0
    val_date = dates_train[1] + pd.Timedelta(weeks=52)
    val_df = pd.DataFrame({
        "Store": [1, 99],  # known and cold-start
        "Dept": [1, 99],
        "Date": [val_date, val_date],
        "Type": ["A", "A"]
    })
    
    model = SeasonalNaiveModel().fit(train_df)
    preds = model.predict(val_df)
    
    # 52 weeks prior to val_date is dates_train[1] (sales = 10.0)
    assert preds[0] == 10.0
    # Cold start falls back to global median
    assert not np.isnan(preds[1])


def test_moving_average_synthetic():
    """Verify MovingAverageModel computes exact mean of the last 4 observations."""
    train_df = pd.DataFrame({
        "Store": [1, 1, 1, 1, 1],
        "Dept": [1, 1, 1, 1, 1],
        "Date": pd.date_range("2010-01-01", periods=5, freq="W-FRI"),
        "Weekly_Sales": [10.0, 20.0, 30.0, 40.0, 50.0],
        "Type": ["A"] * 5
    })
    
    val_df = pd.DataFrame({
        "Store": [1],
        "Dept": [1],
        "Date": pd.to_datetime(["2010-02-12"]),
        "Type": ["A"]
    })
    
    model = MovingAverageModel(window=4).fit(train_df)
    preds = model.predict(val_df)
    
    # Last 4 observations: 20, 30, 40, 50 -> Mean = 35.0
    assert preds[0] == 35.0


def test_negative_sales_preserved_in_baselines():
    """Verify that negative sales (returns) are preserved and not arbitrarily clipped to zero."""
    train_df = pd.DataFrame({
        "Store": [1],
        "Dept": [1],
        "Date": pd.to_datetime(["2010-01-01"]),
        "Weekly_Sales": [-1500.50],
        "Type": ["A"]
    })
    
    val_df = pd.DataFrame({
        "Store": [1],
        "Dept": [1],
        "Date": pd.to_datetime(["2010-01-08"]),
        "Type": ["A"]
    })
    
    naive = NaiveModel().fit(train_df)
    pred_naive = naive.predict(val_df)
    assert pred_naive[0] == -1500.50, "Naive model truncated negative sales!"


def test_validation_predictions_coverage_and_zero_nans():
    """
    Integration test:
    Verify baseline models generate exactly 154,386 predictions with zero NaNs on validation.
    """
    train_df = pd.read_parquet(DATA_DIR / "train_engineered.parquet")
    val_df = pd.read_parquet(DATA_DIR / "val_engineered.parquet")
    
    fallback_mgr = HierarchicalFallbackManager().fit(train_df)
    
    # Test Naive and Seasonal Naive
    naive = NaiveModel().fit(train_df, fallback_manager=fallback_mgr)
    preds_naive = naive.predict(val_df)
    assert len(preds_naive) == 154386
    assert np.isnan(preds_naive).sum() == 0
    
    snaive = SeasonalNaiveModel().fit(train_df, fallback_manager=fallback_mgr)
    preds_snaive = snaive.predict(val_df)
    assert len(preds_snaive) == 154386
    assert np.isnan(preds_snaive).sum() == 0
