"""
Unit and integration tests for chronological split integrity and processed parquet artifacts.
"""
from pathlib import Path
import pandas as pd
import pytest

DATA_DIR = Path("E:/MCA projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting/data/processed")


def test_processed_parquet_files_exist():
    """Verify all engineered Parquet artifacts exist on disk."""
    expected_files = [
        "train_engineered.parquet",
        "val_engineered.parquet",
        "test_engineered.parquet",
        "full_train_engineered.parquet"
    ]
    for filename in expected_files:
        file_path = DATA_DIR / filename
        assert file_path.exists(), f"Missing required artifact: {file_path}"
        assert file_path.stat().st_size > 0, f"Artifact is empty: {file_path}"


def test_chronological_split_integrity():
    """
    Verify strict chronological separation:
    Train max date < Val min date < Test min date.
    Zero temporal leakage or overlap.
    """
    train_df = pd.read_parquet(DATA_DIR / "train_engineered.parquet", columns=["Date"])
    val_df = pd.read_parquet(DATA_DIR / "val_engineered.parquet", columns=["Date"])
    test_df = pd.read_parquet(DATA_DIR / "test_engineered.parquet", columns=["Date"])
    
    train_max = train_df["Date"].max()
    val_min = val_df["Date"].min()
    val_max = val_df["Date"].max()
    test_min = test_df["Date"].min()
    
    # Check boundaries
    assert train_max == pd.Timestamp("2011-10-28"), f"Unexpected train max date: {train_max}"
    assert val_min == pd.Timestamp("2011-11-04"), f"Unexpected val min date: {val_min}"
    assert val_max == pd.Timestamp("2012-10-26"), f"Unexpected val max date: {val_max}"
    assert test_min == pd.Timestamp("2012-11-02"), f"Unexpected test min date: {test_min}"
    
    # Strict temporal ordering
    assert train_max < val_min, f"Overlap between Train ({train_max}) and Val ({val_min})"
    assert val_max < test_min, f"Overlap between Val ({val_max}) and Test ({test_min})"
    
    # Zero set intersection of dates
    train_dates = set(train_df["Date"].unique())
    val_dates = set(val_df["Date"].unique())
    test_dates = set(test_df["Date"].unique())
    
    assert len(train_dates.intersection(val_dates)) == 0, "Temporal overlap between Train and Val!"
    assert len(val_dates.intersection(test_dates)) == 0, "Temporal overlap between Val and Test!"
    assert len(train_dates.intersection(test_dates)) == 0, "Temporal overlap between Train and Test!"


def test_dataset_row_counts_and_conservation():
    """
    Verify row conservation:
    Train split rows + Val split rows == Full Train rows (421,570).
    Test rows == 115,064.
    """
    train_df = pd.read_parquet(DATA_DIR / "train_engineered.parquet", columns=["Store"])
    val_df = pd.read_parquet(DATA_DIR / "val_engineered.parquet", columns=["Store"])
    full_train_df = pd.read_parquet(DATA_DIR / "full_train_engineered.parquet", columns=["Store"])
    test_df = pd.read_parquet(DATA_DIR / "test_engineered.parquet", columns=["Store"])
    
    assert len(train_df) == 267184, f"Unexpected train rows: {len(train_df)}"
    assert len(val_df) == 154386, f"Unexpected val rows: {len(val_df)}"
    assert len(train_df) + len(val_df) == len(full_train_df), "Row counts do not add up to full train!"
    assert len(full_train_df) == 421570, f"Unexpected full train rows: {len(full_train_df)}"
    assert len(test_df) == 115064, f"Unexpected test rows: {len(test_df)}"


def test_feature_matrix_schema_consistency():
    """
    Verify that train, val, and test feature matrices have identical predictor schemas.
    """
    train_df = pd.read_parquet(DATA_DIR / "train_engineered.parquet")
    val_df = pd.read_parquet(DATA_DIR / "val_engineered.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test_engineered.parquet")
    full_train_df = pd.read_parquet(DATA_DIR / "full_train_engineered.parquet")
    
    train_cols = set(train_df.columns)
    val_cols = set(val_df.columns)
    test_cols = set(test_df.columns)
    full_train_cols = set(full_train_df.columns)
    
    # Train, Val, and Full Train should have exactly the same columns including Weekly_Sales
    assert train_cols == val_cols == full_train_cols
    assert "Weekly_Sales" in train_cols
    
    # Predictors in train must exactly match predictors in test
    train_predictors = train_cols - {"Weekly_Sales"}
    assert train_predictors == test_cols, f"Schema mismatch! Diff: {train_predictors.symmetric_difference(test_cols)}"
    assert len(train_predictors) == 60, f"Expected 60 predictor features, got {len(train_predictors)}"


def test_cold_start_series_imputed_no_unhandled_nans():
    """
    Verify that cold-start series in test have valid fallback values
    and no missing values in critical lag features.
    """
    test_df = pd.read_parquet(DATA_DIR / "test_engineered.parquet")
    
    cold_start_pairs = [
        (5, 99), (9, 99), (10, 99), (18, 43), (24, 43),
        (25, 99), (34, 39), (36, 30), (37, 29), (42, 30), (45, 39)
    ]
    
    for store, dept in cold_start_pairs:
        subset = test_df[(test_df["Store"] == store) & (test_df["Dept"] == dept)]
        assert len(subset) > 0, f"Cold-start pair ({store}, {dept}) not found in test!"
        # Fallback median must not be NaN
        assert not subset["Hierarchical_Fallback_Median"].isna().any(), f"NaN fallback for ({store}, {dept})"
        # Lag 1 must not be NaN (fallback imputed)
        assert not subset["Sales_Lag_1"].isna().any(), f"NaN Sales_Lag_1 for ({store}, {dept})"
        # Values must be positive reasonable sales figures
        assert (subset["Sales_Lag_1"] > 0).all(), f"Non-positive Sales_Lag_1 for ({store}, {dept})"
