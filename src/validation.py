"""
Walmart Sales Forecasting — Chronological Validation Framework
Module: src/validation.py
Author: Lead Data Scientist & ML Engineer

Responsibilities:
- Enforce strict chronological out-of-time train/validation splitting:
    * Training Split:   2010-02-05 to 2011-10-28 (91 weeks, 267,184 rows)
    * Validation Split: 2011-11-04 to 2012-10-26 (52 weeks, 154,386 rows)
    * Test Horizon:     2012-11-02 to 2013-07-26 (39 weeks, 115,064 rows)
- Prevent all forms of data leakage across validation boundaries.
- Fit preprocessors and hierarchical fallback managers strictly on training history.
- Compute leak-free lags, rolling statistics, and calendar features.
- Persist engineered datasets into data/processed/ for reproducible modeling.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")

# Add project root to path
sys.path.append(PROJECT_DIR)
from src.data_loader import load_processed_data
from src.preprocessing import clean_and_prepare_dataset, HierarchicalFallbackManager
from src.feature_engineering import (
    build_regular_time_series_grid,
    compute_historical_lags_and_rolling,
    generate_full_feature_set
)

logger = logging.getLogger("WalmartValidation")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Explicit date cutoffs
TRAIN_END_DATE = pd.Timestamp("2011-10-28")
VAL_START_DATE = pd.Timestamp("2011-11-04")
VAL_END_DATE = pd.Timestamp("2012-10-26")
TEST_START_DATE = pd.Timestamp("2012-11-02")
TEST_END_DATE = pd.Timestamp("2013-07-26")


def verify_chronological_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame = None):
    """
    Validate that train, validation, and test splits strictly honor temporal ordering
    without any overlapping timestamps or date leakage.
    """
    logger.info("Verifying chronological split integrity...")
    max_train_date = train_df["Date"].max()
    min_val_date = val_df["Date"].min()
    max_val_date = val_df["Date"].max()

    if max_train_date >= min_val_date:
        raise ValueError(f"Temporal overlap detected: Max train date ({max_train_date}) >= Min val date ({min_val_date})")

    if max_train_date != TRAIN_END_DATE:
        raise ValueError(f"Train split end date mismatch: expected {TRAIN_END_DATE}, got {max_train_date}")

    if min_val_date != VAL_START_DATE:
        raise ValueError(f"Validation start date mismatch: expected {VAL_START_DATE}, got {min_val_date}")

    if max_val_date != VAL_END_DATE:
        raise ValueError(f"Validation end date mismatch: expected {VAL_END_DATE}, got {max_val_date}")

    if test_df is not None:
        min_test_date = test_df["Date"].min()
        max_test_date = test_df["Date"].max()
        if max_val_date >= min_test_date:
            raise ValueError(f"Temporal overlap: Max val date ({max_val_date}) >= Min test date ({min_test_date})")
        if min_test_date != TEST_START_DATE:
            raise ValueError(f"Test start date mismatch: expected {TEST_START_DATE}, got {min_test_date}")

    logger.info("Chronological split integrity confirmed. Zero temporal overlap.")


def build_validation_datasets():
    """
    Execute end-to-end chronological data preparation and feature engineering pipeline:
    1. Ingest clean merged datasets.
    2. Split training data into train_split (91 weeks) and val_split (52 weeks).
    3. Fit fallback manager strictly on train_split.
    4. Preprocess train_split and val_split with fallback baselines and missing value indicators.
    5. Construct regular grid and compute leak-free lags/rolling features.
    6. Generate engineered feature tables for train_split, val_split, and test.
    7. Save engineered parquet files to data/processed/.
    """
    logger.info("=== STARTING PHASE 3 CHRONOLOGICAL VALIDATION PIPELINE ===")
    
    # 1. Ingest merged datasets
    raw_train, raw_test = load_processed_data()
    raw_train["Date"] = pd.to_datetime(raw_train["Date"])
    raw_test["Date"] = pd.to_datetime(raw_test["Date"])

    # 2. Chronological Split of Historical Data
    train_split_mask = raw_train["Date"] <= TRAIN_END_DATE
    val_split_mask = (raw_train["Date"] >= VAL_START_DATE) & (raw_train["Date"] <= VAL_END_DATE)

    train_base = raw_train[train_split_mask].copy()
    val_base = raw_train[val_split_mask].copy()

    logger.info(f"Historical Train Split: {train_base.shape[0]:,} rows ({train_base['Date'].nunique()} weeks)")
    logger.info(f"Historical Val Split:   {val_base.shape[0]:,} rows ({val_base['Date'].nunique()} weeks)")
    
    verify_chronological_splits(train_base, val_base, raw_test)

    # 3. Fit Fallback Manager STRICTLY on train_split (No leakage from validation!)
    fallback_mgr = HierarchicalFallbackManager()
    fallback_mgr.fit(train_base)

    # 4. Preprocess train_split and val_split
    train_prep, _ = clean_and_prepare_dataset(train_base, fallback_manager=fallback_mgr, is_train=True)
    val_prep, _ = clean_and_prepare_dataset(val_base, fallback_manager=fallback_mgr, is_train=False)

    # 5. Build Time-Series Grid for Validation Evaluation
    # Grid covers train_split through val_split (2010-02-05 to 2012-10-26)
    # Using sales from train_base + val_base (with shift(1) strictly applied)
    grid_val = build_regular_time_series_grid(raw_train, None)
    grid_val_features = compute_historical_lags_and_rolling(grid_val)

    # 6. Generate Full Feature Sets
    logger.info("Generating feature tables for train_split and val_split...")
    train_engineered = generate_full_feature_set(train_prep, grid_val_features)
    val_engineered = generate_full_feature_set(val_prep, grid_val_features)

    # 7. Preprocess and Engineer Test Set (Using Full Train for Fallbacks & Lags)
    logger.info("Preprocessing and engineering test horizon features...")
    full_fallback_mgr = HierarchicalFallbackManager()
    full_fallback_mgr.fit(raw_train)
    
    test_prep, _ = clean_and_prepare_dataset(raw_test, fallback_manager=full_fallback_mgr, is_train=False)
    
    # Grid for test covers full train through test (2010-02-05 to 2013-07-26)
    grid_test = build_regular_time_series_grid(raw_train, raw_test)
    grid_test_features = compute_historical_lags_and_rolling(grid_test)
    test_engineered = generate_full_feature_set(test_prep, grid_test_features)

    # Full train engineered (for final retraining before test submission)
    full_train_prep, _ = clean_and_prepare_dataset(raw_train, fallback_manager=full_fallback_mgr, is_train=True)
    full_train_engineered = generate_full_feature_set(full_train_prep, grid_test_features)

    # 8. Assert Final Row Counts
    assert len(train_engineered) == 267184, f"Train engineered row count mismatch: {len(train_engineered)}"
    assert len(val_engineered) == 154386, f"Val engineered row count mismatch: {len(val_engineered)}"
    assert len(test_engineered) == 115064, f"Test engineered row count mismatch: {len(test_engineered)}"
    assert len(full_train_engineered) == 421570, f"Full train engineered row count mismatch: {len(full_train_engineered)}"

    # 9. Schema Consistency Check
    feature_cols_train = set(train_engineered.columns) - {"Weekly_Sales"}
    feature_cols_val = set(val_engineered.columns) - {"Weekly_Sales"}
    feature_cols_test = set(test_engineered.columns)

    if feature_cols_train != feature_cols_val or feature_cols_train != feature_cols_test:
        diff_val = feature_cols_train ^ feature_cols_val
        diff_test = feature_cols_train ^ feature_cols_test
        raise ValueError(f"Schema mismatch across datasets! Diff Val: {diff_val}, Diff Test: {diff_test}")
    logger.info(f"Schema consistency verified: Exactly {len(feature_cols_test)} predictor features across all datasets.")

    # 10. Persist Engineered Datasets
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    p_train = os.path.join(PROCESSED_DATA_DIR, "train_engineered.parquet")
    p_val = os.path.join(PROCESSED_DATA_DIR, "val_engineered.parquet")
    p_test = os.path.join(PROCESSED_DATA_DIR, "test_engineered.parquet")
    p_full_train = os.path.join(PROCESSED_DATA_DIR, "full_train_engineered.parquet")

    logger.info(f"Saving train engineered dataset: {p_train}")
    train_engineered.to_parquet(p_train, engine="pyarrow", index=False)
    
    logger.info(f"Saving val engineered dataset:   {p_val}")
    val_engineered.to_parquet(p_val, engine="pyarrow", index=False)
    
    logger.info(f"Saving test engineered dataset:  {p_test}")
    test_engineered.to_parquet(p_test, engine="pyarrow", index=False)

    logger.info(f"Saving full train engineered:    {p_full_train}")
    full_train_engineered.to_parquet(p_full_train, engine="pyarrow", index=False)

    logger.info("=== PHASE 3 VALIDATION DATASETS SUCCESSFULLY GENERATED & SAVED ===")
    return train_engineered, val_engineered, test_engineered, full_train_engineered


if __name__ == "__main__":
    build_validation_datasets()
