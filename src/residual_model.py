"""
Walmart Sales Forecasting — Seasonal Residual Modeling Engine
Module: src/residual_model.py
Author: Lead ML Engineer & Time-Series Specialist

Implements Seasonal Residual Forecasting:
- Uses Seasonal Naive (Sales_Lag_52 / Hierarchical Fallback) as the robust baseline anchor.
- Trains LightGBM to predict the seasonal residual: Residual = Weekly_Sales - Anchor.
- Eliminates autoregressive drift by excluding recursive short-term lags (Lag 1, Lag 2, Lag 4).
- Focuses learning on promotional markdowns, macroeconomic shifts, holiday calendar shifts,
  and department/store-level interaction dynamics.
- Final prediction: Forecast = Anchor + Predicted_Residual.
"""

import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.preprocessing import HierarchicalFallbackManager

logger = logging.getLogger("WalmartResidualModel")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Exclude target, timestamps, raw string identifiers, and recursive short-term lags to prevent autoregressive drift
RESIDUAL_EXCLUDE_COLUMNS = {
    "Weekly_Sales", "Date", "Type",
    "Sales_Lag_1", "Sales_Lag_2", "Sales_Lag_4",
    "Rolling_Mean_4", "Rolling_Std_4",
    "Rolling_Mean_12", "Rolling_Std_12",
    "Previous_Year_Sales_Diff", "Previous_Year_Sales_Ratio", "Rolling_4_vs_Lag_52_Ratio"
}


def get_residual_feature_columns(df: pd.DataFrame) -> List[str]:
    """Extract strictly exogenous, calendar, markdown, macro, and anchor features for residual modeling."""
    cols = [c for c in df.columns if c not in RESIDUAL_EXCLUDE_COLUMNS]
    return sorted(cols)


class SeasonalResidualForecaster:
    """
    Forecaster that models seasonal residuals:
        y_hat = y_anchor + Delta_hat
    """
    def __init__(self,
                 n_estimators: int = 350,
                 learning_rate: float = 0.07,
                 num_leaves: int = 63,
                 random_state: int = 42,
                 name: str = "Seasonal Residual LightGBM"):
        self.name = name
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_state = random_state
        
        self.fallback_manager: Optional[HierarchicalFallbackManager] = None
        self.model: Optional[lgb.LGBMRegressor] = None
        self.feature_names: List[str] = []
        self.is_fitted = False

    def _compute_anchor(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute baseline anchor vector for every row:
        - If Sales_Lag_52 is present: Anchor = Sales_Lag_52
        - If Sales_Lag_52 is missing (early 2010 or cold-start): Anchor = Hierarchical_Fallback_Median
        Returns: (anchor_values, has_lag52_mask)
        """
        has_l52 = df["Sales_Lag_52"].notnull().values
        fallbacks = self.fallback_manager.transform(df).values
        l52_vals = df["Sales_Lag_52"].values
        
        anchors = np.where(has_l52, l52_vals, fallbacks)
        return anchors, has_l52

    def fit(self,
            train_df: pd.DataFrame,
            fallback_manager: Optional[HierarchicalFallbackManager] = None,
            eval_weeks: int = 12,
            refit_full: bool = False):
        """
        Fit the residual LightGBM model on the historical training split.
        Uses sample_weight (Holiday = 5.0, Non-holiday = 1.0) to optimize WMAE directly.
        If refit_full is True, evaluates early-stopping trajectory on holdout, then retrains
        on all eligible training observations using best_iteration (or n_estimators).
        """
        logger.info(f"Fitting {self.name} on {len(train_df):,} training rows...")
        if fallback_manager is None:
            self.fallback_manager = HierarchicalFallbackManager().fit(train_df)
        else:
            self.fallback_manager = fallback_manager

        # Extract features and compute training residual
        self.feature_names = get_residual_feature_columns(train_df)
        logger.info(f"Selected {len(self.feature_names)} features for residual learning (zero short-term lags).")

        anchors, has_l52 = self._compute_anchor(train_df)
        y_actual = train_df["Weekly_Sales"].values
        residuals = y_actual - anchors

        weights = np.where(train_df["IsHoliday"] == 1, 5.0, 1.0)

        # Train on the subset of data where 52-week lag is observed, or on all data
        # We train on rows where Lag 52 is available to learn pristine year-over-year residual dynamics
        train_mask = has_l52
        X_train_full = train_df.loc[train_mask, self.feature_names]
        y_res_full = residuals[train_mask]
        w_full = weights[train_mask]
        dates_sub = train_df.loc[train_mask, "Date"]

        logger.info(f"Training residual regressor on {len(X_train_full):,} rows with historical 52-week lag...")

        # Chronological internal validation split for early stopping
        unique_dates = sorted(dates_sub.unique())
        split_date = unique_dates[-eval_weeks]
        fit_idx = dates_sub < split_date
        eval_idx = dates_sub >= split_date

        X_fit = X_train_full[fit_idx]
        y_fit = y_res_full[fit_idx]
        w_fit = w_full[fit_idx]

        X_eval = X_train_full[eval_idx]
        y_eval = y_res_full[eval_idx]
        w_eval = w_full[eval_idx]

        logger.info(f"Internal split: Fit ({len(X_fit):,} rows up to {unique_dates[-eval_weeks-1].strftime('%Y-%m-%d')}) | "
                    f"Early-Stop Eval ({len(X_eval):,} rows from {split_date.strftime('%Y-%m-%d')})")

        eval_model = lgb.LGBMRegressor(
            objective="regression_l1",  # Directly minimizes MAE/WMAE!
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
            importance_type="gain"
        )

        eval_model.fit(
            X_fit, y_fit,
            sample_weight=w_fit,
            eval_set=[(X_eval, y_eval)],
            eval_sample_weight=[w_eval],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )

        best_iter = getattr(eval_model, "best_iteration_", self.n_estimators)
        if best_iter is None or best_iter <= 0:
            best_iter = self.n_estimators
        
        self.early_stopped = bool(best_iter < self.n_estimators)
        self.best_iteration = int(best_iter)
        logger.info(f"Internal holdout evaluation complete. Best iteration: {best_iter} "
                    f"(early_stopped={self.early_stopped})")

        if refit_full:
            logger.info(f"Refitting final model on all {len(X_train_full):,} eligible observations with {best_iter} trees...")
            self.model = lgb.LGBMRegressor(
                objective="regression_l1",
                n_estimators=best_iter,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
                importance_type="gain"
            )
            self.model.fit(X_train_full, y_res_full, sample_weight=w_full)
            self.fitted_rows_count = len(X_train_full)
            self.final_tree_count = self.model.booster_.num_trees()
            logger.info(f"Final full retraining complete: {self.fitted_rows_count:,} rows, {self.final_tree_count} trees.")
        else:
            self.model = eval_model
            self.fitted_rows_count = len(X_fit)
            self.final_tree_count = self.model.booster_.num_trees()

        self.is_fitted = True
        logger.info(f"{self.name} training complete. Trees: {self.final_tree_count}, Fitted rows: {self.fitted_rows_count:,}")
        return self

    def predict(self, target_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions on target dataset:
        Returns:
            final_predictions: Anchor + Predicted_Residual
            anchors: Baseline Seasonal Naive anchor values
            predicted_residuals: Predicted residual adjustments
        """
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before predict.")

        anchors, has_l52 = self._compute_anchor(target_df)
        X_target = target_df[self.feature_names]

        predicted_residuals = self.model.predict(X_target)
        final_predictions = anchors + predicted_residuals

        return final_predictions, anchors, predicted_residuals

    def get_feature_importances(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before computing feature importances.")
        importances = self.model.feature_importances_
        df_imp = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance_Gain": importances
        }).sort_values("Importance_Gain", ascending=False).reset_index(drop=True)
        total_gain = df_imp["Importance_Gain"].sum()
        df_imp["Relative_Importance_Pct"] = (df_imp["Importance_Gain"] / total_gain) * 100.0 if total_gain > 0 else 0.0
        return df_imp
