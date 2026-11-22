"""
Walmart Sales Forecasting — Evaluation Metrics Engine
Module: src/evaluation.py
Author: Lead Data Scientist & ML Engineer

Responsibilities:
- Implement official Walmart Weighted Mean Absolute Error (WMAE) metric.
- Calculate Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- Support sliced evaluation (Holiday vs. Non-Holiday, Store-level, Dept-level).
- Provide automated diagnostics for forecast evaluation and benchmarking.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("WalmartEvaluation")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def compute_wmae(y_true: np.ndarray, y_pred: np.ndarray, is_holiday: np.ndarray) -> float:
    """
    Compute official Walmart Weighted Mean Absolute Error (WMAE):
    
        WMAE = sum(w_i * |y_i - y_hat_i|) / sum(w_i)
        
    where w_i = 5.0 if is_holiday is True/1, and w_i = 1.0 if False/0.

    Parameters:
    -----------
    y_true : array-like
        Ground truth actual sales values
    y_pred : array-like
        Predicted sales values
    is_holiday : array-like
        Boolean or binary indicators of holiday week (weight = 5)

    Returns:
    --------
    float: WMAE score in dollar units ($)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    is_holiday = np.asarray(is_holiday, dtype=bool)

    if len(y_true) != len(y_pred) or len(y_true) != len(is_holiday):
        raise ValueError(f"Array length mismatch: y_true ({len(y_true)}), "
                         f"y_pred ({len(y_pred)}), is_holiday ({len(is_holiday)})")

    if len(y_true) == 0:
        return 0.0

    # Weights: 5 for holiday, 1 for non-holiday
    weights = np.where(is_holiday, 5.0, 1.0)
    abs_errors = np.abs(y_true - y_pred)
    
    wmae = float(np.sum(weights * abs_errors) / np.sum(weights))
    return wmae


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute standard unweighted Mean Absolute Error (MAE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute standard Root Mean Squared Error (RMSE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return 0.0
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate_forecast(y_true: np.ndarray, 
                      y_pred: np.ndarray, 
                      is_holiday: np.ndarray, 
                      metadata_df: pd.DataFrame = None) -> dict:
    """
    Comprehensive forecast evaluation returning overall and sliced error metrics.

    Parameters:
    -----------
    y_true : array-like
        Actual sales values
    y_pred : array-like
        Predicted sales values
    is_holiday : array-like
        Holiday boolean indicators
    metadata_df : pd.DataFrame, optional
        Dataframe containing Store and Dept columns for granular breakdown

    Returns:
    --------
    dict: Detailed evaluation metrics report
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    is_holiday = np.asarray(is_holiday, dtype=bool)

    overall_wmae = compute_wmae(y_true, y_pred, is_holiday)
    overall_mae = compute_mae(y_true, y_pred)
    overall_rmse = compute_rmse(y_true, y_pred)

    # Holiday vs Non-Holiday Subsets
    hol_mask = is_holiday == True
    non_hol_mask = ~hol_mask

    hol_count = int(np.sum(hol_mask))
    non_hol_count = int(np.sum(non_hol_mask))

    hol_wmae = compute_wmae(y_true[hol_mask], y_pred[hol_mask], is_holiday[hol_mask]) if hol_count > 0 else 0.0
    non_hol_wmae = compute_wmae(y_true[non_hol_mask], y_pred[non_hol_mask], is_holiday[non_hol_mask]) if non_hol_count > 0 else 0.0

    metrics = {
        "WMAE": overall_wmae,
        "MAE": overall_mae,
        "RMSE": overall_rmse,
        "WMAE_Holiday": hol_wmae,
        "WMAE_NonHoliday": non_hol_wmae,
        "Total_Observations": len(y_true),
        "Holiday_Observations": hol_count,
        "NonHoliday_Observations": non_hol_count,
        "WMAE_to_MAE_Ratio": float(overall_wmae / overall_mae) if overall_mae > 0 else 1.0
    }

    # Department and Store breakdowns if metadata is provided
    if metadata_df is not None and "Dept" in metadata_df.columns:
        eval_df = metadata_df.copy()
        eval_df["y_true"] = y_true
        eval_df["y_pred"] = y_pred
        eval_df["is_holiday"] = is_holiday
        
        dept_wmae = {}
        for dept, grp in eval_df.groupby("Dept"):
            dept_wmae[int(dept)] = compute_wmae(grp["y_true"].values, grp["y_pred"].values, grp["is_holiday"].values)
        metrics["Dept_WMAE"] = dept_wmae

    return metrics
