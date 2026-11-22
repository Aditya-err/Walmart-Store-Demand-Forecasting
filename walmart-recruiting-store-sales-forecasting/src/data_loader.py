"""
Walmart Sales Forecasting — Data Loading and Validation Pipeline
Module: src/data_loader.py
Author: Lead Data Scientist & ML Engineer

Responsibilities:
- Ingest raw stores.csv, train.csv.zip, test.csv.zip, and features.csv.zip.
- Validate schemas, data types, and identifiers (Store, Dept, Date).
- Verify candidate keys and join cardinalities using pandas merge validation ('many_to_one').
- Detect missing join matches and verify IsHoliday consistency across tables.
- Guarantee that row counts before and after merging remain completely unchanged.
- Persist analytical merged datasets into data/processed/ (Parquet format with pyarrow).
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("WalmartDataLoader")

# Default directories relative to project root
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")

# Schema definitions
EXPECTED_COLUMNS = {
    "stores": ["Store", "Type", "Size"],
    "features": [
        "Store", "Date", "Temperature", "Fuel_Price",
        "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
        "CPI", "Unemployment", "IsHoliday"
    ],
    "train": ["Store", "Dept", "Date", "Weekly_Sales", "IsHoliday"],
    "test": ["Store", "Dept", "Date", "IsHoliday"]
}


def load_raw_data(data_dir=None):
    """
    Ingest raw CSV / ZIP archives without extracting or altering original files.

    Parameters:
    -----------
    data_dir : str, optional
        Path to the folder containing raw dataset archives. Defaults to PROJECT_DIR.

    Returns:
    --------
    tuple: (stores_df, features_df, train_df, test_df)
    """
    if data_dir is None:
        data_dir = PROJECT_DIR

    logger.info(f"Ingesting raw datasets from: {data_dir}")

    stores_path = os.path.join(data_dir, "stores.csv")
    features_path = os.path.join(data_dir, "features.csv.zip")
    train_path = os.path.join(data_dir, "train.csv.zip")
    test_path = os.path.join(data_dir, "test.csv.zip")

    for path in [stores_path, features_path, train_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required dataset file not found: {path}")

    stores = pd.read_csv(stores_path)
    features = pd.read_csv(features_path)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    logger.info(f"Loaded stores.csv: {stores.shape[0]} rows x {stores.shape[1]} cols")
    logger.info(f"Loaded features.csv.zip: {features.shape[0]} rows x {features.shape[1]} cols")
    logger.info(f"Loaded train.csv.zip: {train.shape[0]} rows x {train.shape[1]} cols")
    logger.info(f"Loaded test.csv.zip: {test.shape[0]} rows x {test.shape[1]} cols")

    # Validate Schema
    for name, df in [("stores", stores), ("features", features), ("train", train), ("test", test)]:
        missing_cols = set(EXPECTED_COLUMNS[name]) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Schema violation in {name}: missing columns {missing_cols}")

    # Parse Dates
    for df in [features, train, test]:
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")

    # Validate Store IDs (Must be integers between 1 and 45)
    for name, df in [("stores", stores), ("features", features), ("train", train), ("test", test)]:
        stores_in_df = set(df["Store"].unique())
        if not stores_in_df.issubset(set(range(1, 46))):
            raise ValueError(f"Invalid store identifiers found in {name}: {stores_in_df - set(range(1, 46))}")

    # Validate Dept IDs (Must be positive integers)
    for name, df in [("train", train), ("test", test)]:
        if (df["Dept"] <= 0).any():
            raise ValueError(f"Non-positive department identifiers found in {name}")

    logger.info("Schema, data types, dates, and ID ranges validated successfully.")
    return stores, features, train, test


def validate_and_merge(base_df, features_df, stores_df, dataset_name="train"):
    """
    Safely merge base sales data (train or test) with store dimensions and external features.
    Enforces strict join validation rules:
    - Many-to-one validation with stores on ['Store']
    - Many-to-one validation with features on ['Store', 'Date']
    - Separate IsHoliday comparison before duplicate column resolution
    - Verification that no rows are multiplied or lost

    Parameters:
    -----------
    base_df : pd.DataFrame
        train or test dataframe
    features_df : pd.DataFrame
        features dataframe
    stores_df : pd.DataFrame
        stores dataframe
    dataset_name : str
        'train' or 'test' for logging and row-count verification

    Returns:
    --------
    pd.DataFrame: Merged, validated analytical dataset
    """
    initial_rows = len(base_df)
    logger.info(f"Beginning merge and validation pipeline for '{dataset_name}' (Initial rows: {initial_rows:,})")

    # 1. Validate primary/candidate keys of dimension tables
    if not stores_df["Store"].is_unique:
        raise ValueError("stores.csv candidate key 'Store' contains duplicates!")
    
    if features_df.duplicated(subset=["Store", "Date"]).any():
        raise ValueError("features.csv candidate key ('Store', 'Date') contains duplicates!")

    # 2. Merge with stores (Join key: Store)
    merged = base_df.merge(
        stores_df,
        on="Store",
        how="left",
        validate="many_to_one"
    )

    if len(merged) != initial_rows:
        raise AssertionError(f"Row count mismatch after store merge: expected {initial_rows}, got {len(merged)}")

    if merged["Type"].isnull().any() or merged["Size"].isnull().any():
        raise ValueError(f"Unmatched stores found in {dataset_name} after store merge!")

    logger.info(f"Store dimension merged successfully (validate='many_to_one'). Rows: {len(merged):,}")

    # 3. Merge with features (Join key: Store, Date)
    # Do NOT include IsHoliday in join key as instructed; validate separately
    merged = merged.merge(
        features_df,
        on=["Store", "Date"],
        how="left",
        validate="many_to_one",
        suffixes=("_base", "_features")
    )

    if len(merged) != initial_rows:
        raise AssertionError(f"Row count mismatch after features merge: expected {initial_rows}, got {len(merged)}")

    # 4. Validate IsHoliday consistency
    holiday_mismatch = (merged["IsHoliday_base"] != merged["IsHoliday_features"]).sum()
    logger.info(f"IsHoliday consistency check between {dataset_name} and features: {holiday_mismatch} mismatches found.")
    if holiday_mismatch != 0:
        raise ValueError(f"Discrepancy detected in IsHoliday between {dataset_name} and features: {holiday_mismatch} mismatched rows!")

    # Resolve duplicate IsHoliday column safely
    merged.drop(columns=["IsHoliday_features"], inplace=True)
    merged.rename(columns={"IsHoliday_base": "IsHoliday"}, inplace=True)

    # 5. Final assertion on expected benchmark row counts
    if dataset_name == "train" and len(merged) != 421570:
        raise AssertionError(f"Train merged rows ({len(merged)}) != expected 421,570")
    elif dataset_name == "test" and len(merged) != 115064:
        raise AssertionError(f"Test merged rows ({len(merged)}) != expected 115,064")

    logger.info(f"Merge pipeline for '{dataset_name}' complete. Final shape: {merged.shape}")
    return merged


def save_processed_data(train_merged, test_merged, output_dir=None):
    """
    Save analytical merged datasets to disk.
    Prefers Parquet (using pyarrow engine) with graceful fallback to CSV.
    """
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR

    os.makedirs(output_dir, exist_ok=True)

    train_parquet = os.path.join(output_dir, "train_merged.parquet")
    test_parquet = os.path.join(output_dir, "test_merged.parquet")

    try:
        logger.info(f"Saving merged training data to Parquet: {train_parquet}")
        train_merged.to_parquet(train_parquet, engine="pyarrow", index=False)
        logger.info(f"Saving merged test data to Parquet: {test_parquet}")
        test_merged.to_parquet(test_parquet, engine="pyarrow", index=False)
        logger.info("Successfully persisted Parquet analytical datasets.")
    except Exception as e:
        logger.warning(f"Parquet engine failed ({e}). Falling back to compressed CSV...")
        train_csv = os.path.join(output_dir, "train_merged.csv.gz")
        test_csv = os.path.join(output_dir, "test_merged.csv.gz")
        train_merged.to_csv(train_csv, index=False, compression="gzip")
        test_merged.to_csv(test_csv, index=False, compression="gzip")
        logger.info(f"Saved CSV fallbacks: {train_csv}, {test_csv}")


def load_processed_data(output_dir=None):
    """
    Load persisted analytical merged datasets from data/processed/.

    Returns:
    --------
    tuple: (train_merged, test_merged)
    """
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR

    train_parquet = os.path.join(output_dir, "train_merged.parquet")
    test_parquet = os.path.join(output_dir, "test_merged.parquet")

    if os.path.exists(train_parquet) and os.path.exists(test_parquet):
        logger.info(f"Loading processed datasets from Parquet: {output_dir}")
        train_df = pd.read_parquet(train_parquet, engine="pyarrow")
        test_df = pd.read_parquet(test_parquet, engine="pyarrow")
    else:
        train_csv = os.path.join(output_dir, "train_merged.csv.gz")
        test_csv = os.path.join(output_dir, "test_merged.csv.gz")
        logger.info(f"Loading processed datasets from CSV fallback: {output_dir}")
        train_df = pd.read_csv(train_csv, parse_dates=["Date"])
        test_df = pd.read_csv(test_csv, parse_dates=["Date"])

    return train_df, test_df


if __name__ == "__main__":
    logger.info("=== WALMART SALES FORECASTING DATA LOADER RUNTIME TEST ===")
    stores, features, train, test = load_raw_data()
    train_merged = validate_and_merge(train, features, stores, dataset_name="train")
    test_merged = validate_and_merge(test, features, stores, dataset_name="test")
    save_processed_data(train_merged, test_merged)
    logger.info("Data loader execution and persistence complete.")
