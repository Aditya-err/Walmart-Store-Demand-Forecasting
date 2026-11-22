"""
Walmart Sales Forecasting — Ensemble Blending & Weight Optimization Engine
Module: src/ensemble.py
Author: Lead ML Engineer & Time-Series Specialist

Implements optimal ensemble blending between Seasonal Naive and Seasonal Residual LightGBM:
- Global Blending: y_blend = alpha * y_anchor + (1 - alpha) * y_res_model
- Holiday-Segmented Blending: Separate weights for holiday vs. non-holiday weeks.
- Strict Chronological Inner Validation: Tunes blending weights exclusively on an inner
  training split (zero lookahead or validation leakage).
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.evaluation import compute_wmae
from src.preprocessing import HierarchicalFallbackManager
from src.residual_model import get_residual_feature_columns

logger = logging.getLogger("WalmartEnsemble")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def tune_blending_weights_chronologically(
    train_df: pd.DataFrame,
    fallback_mgr: HierarchicalFallbackManager,
    inner_val_weeks: int = 10,
    grid_steps: int = 21
) -> Dict[str, Any]:
    """
    Tune ensemble blending weights strictly using a chronological inner validation split
    on the historical training data.
    
    Guarantees 100% zero leakage: Never touches the out-of-sample validation set.
    """
    logger.info("=== TUNING ENSEMBLE BLENDING WEIGHTS VIA INNER CHRONOLOGICAL VALIDATION ===")
    
    train_has_l52 = train_df["Sales_Lag_52"].notnull()
    sub_train = train_df[train_has_l52].sort_values("Date").reset_index(drop=True)
    dates = sorted(sub_train["Date"].unique())
    
    split_date = dates[-inner_val_weeks]
    logger.info(f"Inner training dates: {dates[0].strftime('%Y-%m-%d')} to {dates[-inner_val_weeks-1].strftime('%Y-%m-%d')} "
                f"| Inner validation: {split_date.strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')} ({inner_val_weeks} weeks)")

    inner_tr_mask = sub_train["Date"] < split_date
    inner_val_mask = sub_train["Date"] >= split_date

    feature_cols = get_residual_feature_columns(sub_train)
    
    anchor_tr = sub_train.loc[inner_tr_mask, "Sales_Lag_52"].values
    anchor_val = sub_train.loc[inner_val_mask, "Sales_Lag_52"].values
    
    y_tr_actual = sub_train.loc[inner_tr_mask, "Weekly_Sales"].values
    y_val_actual = sub_train.loc[inner_val_mask, "Weekly_Sales"].values
    is_hol_val = sub_train.loc[inner_val_mask, "IsHoliday"].astype(bool).values
    
    y_tr_res = y_tr_actual - anchor_tr
    w_tr = np.where(sub_train.loc[inner_tr_mask, "IsHoliday"] == 1, 5.0, 1.0)

    # Train inner model on inner training split
    logger.info(f"Fitting inner residual model on {inner_tr_mask.sum():,} inner training rows...")
    inner_model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=300,
        learning_rate=0.08,
        num_leaves=63,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    inner_model.fit(sub_train.loc[inner_tr_mask, feature_cols], y_tr_res, sample_weight=w_tr)

    # Predict residuals on inner validation
    pred_res_val = inner_model.predict(sub_train.loc[inner_val_mask, feature_cols])
    pred_full_res_val = anchor_val + pred_res_val

    # Benchmark individual models on inner validation
    score_anchor = compute_wmae(y_val_actual, anchor_val, is_hol_val)
    score_res = compute_wmae(y_val_actual, pred_full_res_val, is_hol_val)
    logger.info(f"Inner Val Score - Pure Seasonal Naive Anchor: ${score_anchor:,.2f}")
    logger.info(f"Inner Val Score - Pure Residual LightGBM:     ${score_res:,.2f}")

    # 1. Global Grid Search: alpha * anchor + (1 - alpha) * res
    alpha_grid = np.linspace(0.0, 1.0, grid_steps)
    best_global_alpha = 0.0
    best_global_wmae = float("inf")

    grid_results = []
    for a in alpha_grid:
        blend = a * anchor_val + (1.0 - a) * pred_full_res_val
        wmae = compute_wmae(y_val_actual, blend, is_hol_val)
        grid_results.append({"alpha": round(float(a), 4), "inner_wmae": round(float(wmae), 2)})
        if wmae < best_global_wmae:
            best_global_wmae = wmae
            best_global_alpha = float(a)

    logger.info(f"Optimal Global Weight: alpha = {best_global_alpha:.2f} "
                f"({best_global_alpha:.0%} SNaive + {1-best_global_alpha:.0%} ResLGBM) | Inner WMAE: ${best_global_wmae:,.2f}")

    # 2. Segmented Grid Search: Holiday vs. Non-Holiday weights
    # Test if separate holiday blending is justified
    best_hol_alpha = best_global_alpha
    best_non_hol_alpha = best_global_alpha
    best_segmented_wmae = float("inf")

    coarse_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for a_hol in coarse_grid:
        for a_nh in coarse_grid:
            blend_seg = np.where(
                is_hol_val,
                a_hol * anchor_val + (1.0 - a_hol) * pred_full_res_val,
                a_nh * anchor_val + (1.0 - a_nh) * pred_full_res_val
            )
            wmae_seg = compute_wmae(y_val_actual, blend_seg, is_hol_val)
            if wmae_seg < best_segmented_wmae:
                best_segmented_wmae = wmae_seg
                best_hol_alpha = float(a_hol)
                best_non_hol_alpha = float(a_nh)

    logger.info(f"Optimal Segmented Weights: Holiday alpha = {best_hol_alpha:.2f}, "
                f"Non-Holiday alpha = {best_non_hol_alpha:.2f} | Inner WMAE: ${best_segmented_wmae:,.2f}")

    tuning_report = {
        "inner_split_date": split_date.strftime("%Y-%m-%d"),
        "inner_val_weeks": inner_val_weeks,
        "optimal_global_alpha": round(best_global_alpha, 4),
        "best_global_inner_wmae": round(best_global_wmae, 2),
        "optimal_holiday_alpha": round(best_hol_alpha, 4),
        "optimal_non_holiday_alpha": round(best_non_hol_alpha, 4),
        "best_segmented_inner_wmae": round(best_segmented_wmae, 2),
        "global_grid_search": grid_results
    }
    return tuning_report


class BlendedEnsembleForecaster:
    """
    Ensemble model that blends Seasonal Naive and Seasonal Residual LightGBM.
    """
    def __init__(self,
                 global_alpha: float = 0.20,
                 holiday_alpha: Optional[float] = None,
                 non_holiday_alpha: Optional[float] = None,
                 name: str = "Blended Ensemble"):
        self.name = name
        self.global_alpha = global_alpha
        self.holiday_alpha = holiday_alpha
        self.non_holiday_alpha = non_holiday_alpha

    def blend(self,
              y_anchor: np.ndarray,
              y_res_model: np.ndarray,
              is_holiday: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Combine anchor and residual predictions using tuned weights.
        """
        if self.holiday_alpha is not None and self.non_holiday_alpha is not None and is_holiday is not None:
            # Segmented blending
            is_hol = np.asarray(is_holiday, dtype=bool)
            return np.where(
                is_hol,
                self.holiday_alpha * y_anchor + (1.0 - self.holiday_alpha) * y_res_model,
                self.non_holiday_alpha * y_anchor + (1.0 - self.non_holiday_alpha) * y_res_model
            )
        else:
            # Global blending
            return self.global_alpha * y_anchor + (1.0 - self.global_alpha) * y_res_model
