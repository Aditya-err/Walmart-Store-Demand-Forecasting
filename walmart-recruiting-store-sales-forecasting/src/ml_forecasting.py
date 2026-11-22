"""
Walmart Sales Forecasting — Recursive Multi-Step ML Forecasting Engine
Module: src/ml_forecasting.py
Author: Senior ML Engineer & Data Scientist

Implements strict recursive multi-step forecasting across the 52-week validation horizon:
- Iterates week-by-week chronologically from 2011-11-04 to 2012-10-26.
- In week w, short-term historical lags (Lag 1, Lag 2, Lag 4) and rolling features (Rolling Mean/Std 4, 12, 52)
  are populated dynamically using previous-step PREDICTIONS (for validation dates)
  and training actuals (for pre-validation dates).
- Never accesses actual validation sales, guaranteeing absolute zero data leakage.
- Long-term seasonal lag (Sales_Lag_52) draws from the 52-week prior historical training observations.
- Uses HierarchicalFallbackManager for cold-start series.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.ml_models import BaseMLModel, get_feature_columns
from src.preprocessing import HierarchicalFallbackManager

logger = logging.getLogger("WalmartMLForecasting")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class RecursiveMultiStepForecaster:
    """
    Executes chronological week-by-week multi-step forecasting with autoregressive state updates.
    """
    def __init__(self, model: BaseMLModel, fallback_manager: HierarchicalFallbackManager):
        self.model = model
        self.fallback_manager = fallback_manager
        self.history_buffer: Dict[Tuple[int, int], List[float]] = {}

    def initialize_history_from_train(self, train_df: pd.DataFrame):
        """
        Populate the historical sales buffer for each (Store, Dept) up to the forecast origin (2011-10-28).
        """
        logger.info("Initializing multi-step history buffer from historical training data...")
        sorted_train = train_df.sort_values(["Store", "Dept", "Date"])
        
        self.history_buffer = {}
        for (store, dept), grp in sorted_train.groupby(["Store", "Dept"]):
            self.history_buffer[(store, dept)] = list(grp["Weekly_Sales"].values)
            
        logger.info(f"Initialized history buffer for {len(self.history_buffer):,} series.")

    def forecast_validation_horizon(self, val_df: pd.DataFrame) -> np.ndarray:
        """
        Generate recursive multi-step predictions for all validation rows.
        
        Parameters:
        -----------
        val_df : pd.DataFrame
            Validation dataset (154,386 rows, 2011-11-04 to 2012-10-26)
            
        Returns:
        --------
        np.ndarray: Predicted weekly sales for every row in val_df
        """
        logger.info(f"Starting recursive multi-step forecasting with {self.model.name} across {len(val_df):,} validation observations...")
        t0 = time.time()
        
        # Working copy to preserve original dataframe
        val_work = val_df.copy().reset_index(drop=True)
        feature_cols = self.model.feature_names
        
        val_dates = sorted(val_work["Date"].unique())
        total_weeks = len(val_dates)
        logger.info(f"Validation horizon encompasses {total_weeks} chronological weekly time-steps.")

        all_predictions = np.empty(len(val_work), dtype=float)
        indices_seen = 0

        for step, current_date in enumerate(val_dates, 1):
            week_mask = val_work["Date"] == current_date
            week_indices = val_work.index[week_mask]
            week_rows = val_work.loc[week_indices].copy()
            
            # Dynamically update autoregressive lag and rolling features from recursive history buffer
            stores = week_rows["Store"].values
            depts = week_rows["Dept"].values
            lag_52_vals = week_rows["Sales_Lag_52"].values
            fallbacks = week_rows["Hierarchical_Fallback_Median"].values
            
            lag1_list = []
            lag2_list = []
            lag4_list = []
            roll4_mean_list = []
            roll4_std_list = []
            roll12_mean_list = []
            roll12_std_list = []
            roll52_mean_list = []
            roll52_std_list = []
            prev_diff_list = []
            prev_ratio_list = []
            roll4_ratio_list = []
            
            for s, d, l52, fb in zip(stores, depts, lag_52_vals, fallbacks):
                hist = self.history_buffer.get((s, d), [])
                
                # If cold start or empty buffer, use hierarchical fallback
                if len(hist) == 0:
                    l1 = fb
                    l2 = fb
                    l4 = fb
                    r4_mean = fb
                    r4_std = 0.0
                    r12_mean = fb
                    r12_std = 0.0
                    r52_mean = fb
                    r52_std = 0.0
                else:
                    l1 = hist[-1]
                    l2 = hist[-2] if len(hist) >= 2 else hist[-1]
                    l4 = hist[-4] if len(hist) >= 4 else hist[-1]
                    
                    # Rolling 4
                    sub4 = hist[-4:]
                    r4_mean = float(np.mean(sub4))
                    r4_std = float(np.std(sub4)) if len(sub4) >= 2 else 0.0
                    
                    # Rolling 12
                    sub12 = hist[-12:]
                    r12_mean = float(np.mean(sub12))
                    r12_std = float(np.std(sub12)) if len(sub12) >= 2 else 0.0
                    
                    # Rolling 52
                    sub52 = hist[-52:]
                    r52_mean = float(np.mean(sub52))
                    r52_std = float(np.std(sub52)) if len(sub52) >= 2 else 0.0

                l52_clean = l52 if not np.isnan(l52) else fb
                prev_diff = l1 - l52_clean
                prev_ratio = (l1 + 1.0) / (abs(l52_clean) + 1.0)
                r4_ratio = (r4_mean + 1.0) / (abs(l52_clean) + 1.0)
                
                lag1_list.append(l1)
                lag2_list.append(l2)
                lag4_list.append(l4)
                roll4_mean_list.append(r4_mean)
                roll4_std_list.append(r4_std)
                roll12_mean_list.append(r12_mean)
                roll12_std_list.append(r12_std)
                roll52_mean_list.append(r52_mean)
                roll52_std_list.append(r52_std)
                prev_diff_list.append(prev_diff)
                prev_ratio_list.append(prev_ratio)
                roll4_ratio_list.append(r4_ratio)

            # Assign recursively computed feature values
            week_rows["Sales_Lag_1"] = lag1_list
            week_rows["Sales_Lag_2"] = lag2_list
            week_rows["Sales_Lag_4"] = lag4_list
            week_rows["Rolling_Mean_4"] = roll4_mean_list
            week_rows["Rolling_Std_4"] = roll4_std_list
            week_rows["Rolling_Mean_12"] = roll12_mean_list
            week_rows["Rolling_Std_12"] = roll12_std_list
            week_rows["Rolling_Mean_52"] = roll52_mean_list
            week_rows["Rolling_Std_52"] = roll52_std_list
            week_rows["Previous_Year_Sales_Diff"] = prev_diff_list
            week_rows["Previous_Year_Sales_Ratio"] = prev_ratio_list
            week_rows["Rolling_4_vs_Lag_52_Ratio"] = roll4_ratio_list

            # Predict week w
            X_week = week_rows[feature_cols]
            week_preds = self.model.predict(X_week)
            
            # Store predictions
            all_predictions[week_indices] = week_preds
            indices_seen += len(week_indices)

            # Update history buffer with current predictions for autoregression
            for s, d, pred_val in zip(stores, depts, week_preds):
                if (s, d) not in self.history_buffer:
                    self.history_buffer[(s, d)] = []
                self.history_buffer[(s, d)].append(float(pred_val))

            if step % 13 == 0 or step == total_weeks:
                elapsed = time.time() - t0
                logger.info(f"Completed Step {step:02d}/{total_weeks} ({current_date.strftime('%Y-%m-%d')}) | "
                            f"{indices_seen:,} rows predicted ({elapsed:.1f}s)")

        logger.info(f"Recursive forecasting finished in {time.time() - t0:.2f} seconds.")
        return all_predictions
