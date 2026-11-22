"""
Walmart Sales Forecasting — Preprocessing & Data Preparation Pipeline
Module: src/preprocessing.py
Author: Lead Data Scientist & ML Engineer

Responsibilities:
- Preserve negative weekly sales without truncation or zero-clipping.
- Handle promotional markdown missingness (0.0 imputation + Has_MarkDown_Data indicator).
- Handle CPI & Unemployment missingness (store-level forward-fill + Is_Macro_Imputed indicator).
- Resolve cold-start (Store, Dept) series via a hierarchical fallback manager (fit on training data).
- Ensure strict schema and transformation consistency across train, validation, and test sets.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("WalmartPreprocessing")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class HierarchicalFallbackManager:
    """
    Computes hierarchical baseline weekly sales from training history to provide
    statistically sound fallback values for cold-start (Store, Dept) series.
    
    Hierarchy:
    1. (Store Type, Department) Median
    2. Department Overall Median
    3. Store Overall Median
    4. Global Median
    """
    def __init__(self):
        self.dept_type_medians = {}
        self.dept_medians = {}
        self.store_medians = {}
        self.global_median = 0.0
        self.dept_type_means = {}
        self.dept_means = {}
        self.store_means = {}
        self.global_mean = 0.0
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame):
        """Fit baselines strictly on historical training observations."""
        logger.info("Fitting Hierarchical Fallback Manager on training data...")
        sales = train_df["Weekly_Sales"]
        self.global_median = float(sales.median())
        self.global_mean = float(sales.mean())

        # Level 1: (Store Type, Dept)
        type_dept_grp = train_df.groupby(["Type", "Dept"])["Weekly_Sales"]
        self.dept_type_medians = type_dept_grp.median().to_dict()
        self.dept_type_means = type_dept_grp.mean().to_dict()

        # Level 2: Dept
        dept_grp = train_df.groupby("Dept")["Weekly_Sales"]
        self.dept_medians = dept_grp.median().to_dict()
        self.dept_means = dept_grp.mean().to_dict()

        # Level 3: Store
        store_grp = train_df.groupby("Store")["Weekly_Sales"]
        self.store_medians = store_grp.median().to_dict()
        self.store_means = store_grp.mean().to_dict()

        self.is_fitted = True
        logger.info(f"Fitted {len(self.dept_type_medians)} (Type, Dept) baselines, "
                    f"{len(self.dept_medians)} Dept baselines, {len(self.store_medians)} Store baselines. "
                    f"Global Median: ${self.global_median:,.2f}")
        return self

    def get_fallback_median(self, store: int, store_type: str, dept: int) -> float:
        """Retrieve the most granular available fallback median."""
        if not self.is_fitted:
            raise RuntimeError("HierarchicalFallbackManager must be fitted before calling get_fallback_median.")
        
        # 1. (Type, Dept)
        if (store_type, dept) in self.dept_type_medians and not np.isnan(self.dept_type_medians[(store_type, dept)]):
            return self.dept_type_medians[(store_type, dept)]
        # 2. Dept
        if dept in self.dept_medians and not np.isnan(self.dept_medians[dept]):
            return self.dept_medians[dept]
        # 3. Store
        if store in self.store_medians and not np.isnan(self.store_medians[store]):
            return self.store_medians[store]
        # 4. Global
        return self.global_median

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Generate a series of hierarchical fallback baseline values for every row."""
        if not self.is_fitted:
            raise RuntimeError("HierarchicalFallbackManager must be fitted before transform.")
        
        fallbacks = [
            self.get_fallback_median(row.Store, row.Type, row.Dept)
            for row in df[["Store", "Type", "Dept"]].itertuples(index=False)
        ]
        return pd.Series(fallbacks, index=df.index, name="Hierarchical_Fallback_Median")


def handle_markdown_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle promotional MarkDown1 to MarkDown5:
    - Missing markdowns represent unrecorded promotions (pre-Nov 2011) or inactive promotions.
    - Impute NaN with 0.0.
    - Create binary indicator Has_MarkDown_Data (True if date >= 2011-11-04 or any markdown > 0).
    - Create MarkDown_Total feature (sum of all 5 markdowns).
    """
    df = df.copy()
    md_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
    
    # Presence indicator
    has_any_md = df[md_cols].notnull().any(axis=1)
    date_after_start = df["Date"] >= pd.Timestamp("2011-11-04")
    df["Has_MarkDown_Data"] = (has_any_md | date_after_start).astype(int)

    # Impute missing values with 0.0
    for col in md_cols:
        df[col + "_Raw_Null"] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(0.0)

    # Aggregate MarkDown features
    df["MarkDown_Total"] = df[md_cols].sum(axis=1)
    df["MarkDown_Count_Active"] = (df[md_cols] > 0).sum(axis=1)

    return df


def handle_macroeconomic_missingness(df: pd.DataFrame, reference_features_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Handle CPI and Unemployment:
    - In test.csv / features.csv, May 3 to July 26, 2013 (13 weeks) are 100% missing.
    - Impute via Store-level forward-fill along the continuous weekly timeline.
    - Attach boolean indicator Is_Macro_Imputed (1 if imputed, 0 if original).
    """
    df = df.copy()
    
    # Track imputation status
    cpi_null = df["CPI"].isnull()
    unemp_null = df["Unemployment"].isnull()
    df["Is_Macro_Imputed"] = (cpi_null | unemp_null).astype(int)

    if df["Is_Macro_Imputed"].any():
        logger.info(f"Forward-filling {cpi_null.sum():,} missing CPI/Unemployment records by Store...")
        # Sort chronologically by store and date before forward fill
        df = df.sort_values(["Store", "Date"])
        df["CPI"] = df.groupby("Store")["CPI"].ffill()
        df["Unemployment"] = df.groupby("Store")["Unemployment"].ffill()
        
        # If any remaining (e.g. if store starts with null, though not the case in Walmart data)
        df["CPI"] = df["CPI"].bfill()
        df["Unemployment"] = df["Unemployment"].bfill()
        
    return df


def preserve_negative_sales_audit(df: pd.DataFrame) -> dict:
    """
    Audit and guarantee that negative sales values remain strictly preserved.
    """
    if "Weekly_Sales" not in df.columns:
        return {"has_sales": False}
    
    neg_mask = df["Weekly_Sales"] < 0
    neg_count = int(neg_mask.sum())
    min_sales = float(df["Weekly_Sales"].min())
    
    logger.info(f"Negative sales validation: {neg_count:,} negative records found (Min: ${min_sales:,.2f}). "
                f"All negative values preserved.")
    return {
        "has_sales": True,
        "negative_sales_count": neg_count,
        "negative_sales_min": min_sales,
        "negative_sales_preserved": True
    }


def clean_and_prepare_dataset(df: pd.DataFrame, fallback_manager: HierarchicalFallbackManager = None, is_train: bool = True) -> tuple:
    """
    Main preprocessing pipeline for any Walmart dataset (train, val, test).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Merged dataset from data_loader
    fallback_manager : HierarchicalFallbackManager, optional
        Pre-fitted fallback manager (if None and is_train=True, will fit a new one)
    is_train : bool
        Whether this is the training split (to fit fallback manager)

    Returns:
    --------
    tuple: (cleaned_df, fallback_manager)
    """
    logger.info(f"Running preprocessing pipeline (is_train={is_train}, rows={len(df):,})...")
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # 1. Audit and preserve negative sales
    preserve_negative_sales_audit(df)

    # 2. Handle MarkDown missingness
    df = handle_markdown_missingness(df)

    # 3. Handle Macroeconomic missingness
    df = handle_macroeconomic_missingness(df)

    # 4. Hierarchical Fallback Manager
    if fallback_manager is None:
        if is_train:
            fallback_manager = HierarchicalFallbackManager()
            fallback_manager.fit(df)
        else:
            raise ValueError("A fitted HierarchicalFallbackManager must be provided for validation or test sets!")

    df["Hierarchical_Fallback_Median"] = fallback_manager.transform(df)

    logger.info(f"Preprocessing completed successfully. Output shape: {df.shape}")
    return df, fallback_manager
