"""
Walmart Sales Forecasting — Advanced Machine Learning Models Engine
Module: src/ml_models.py
Author: Senior ML Engineer & Data Scientist

Defines unified wrappers for Gradient Boosted Decision Tree (GBDT) architectures:
1. LGBMModel: LightGBM Regressor optimizing L1/MAE loss with official Walmart sample weights.
2. XGBoostModel: XGBoost Regressor optimizing absolute error with sample weights.
3. CatBoostModel: CatBoost Regressor optimizing MAE (available if installed & compatible).

All models provide consistent fit, predict, feature importance, and SHAP interfaces.
"""

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

# 1. LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# 2. XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# 3. CatBoost
try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# 4. SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

logger = logging.getLogger("WalmartMLModels")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# Standard predictor feature column filter
EXCLUDE_COLUMNS = {"Date", "Weekly_Sales", "Type"}


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return standard sorted list of predictor feature columns from an engineered dataset."""
    cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
    return sorted(cols)


class BaseMLModel:
    """Abstract base class for machine learning regressors."""
    def __init__(self, name: str):
        self.name = name
        self.model: Any = None
        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: Optional[np.ndarray] = None,
            eval_set: Optional[Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray]]] = None):
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def get_feature_importances(self) -> pd.DataFrame:
        raise NotImplementedError


class LGBMModel(BaseMLModel):
    """
    LightGBM Regressor with L1 / MAE Objective:
    Natively supports sample weighting (Holiday = 5, Non-holiday = 1) to align directly
    with the official Walmart Weighted Mean Absolute Error (WMAE) metric.
    """
    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.08, num_leaves: int = 63,
                 random_state: int = 42, n_jobs: int = -1, name: str = "LightGBM Regressor"):
        super().__init__(name)
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm is not installed.")
        self.params = {
            "objective": "regression_l1",  # Optimizes MAE directly!
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "verbose": -1,
            "importance_type": "gain"
        }
        self.model = lgb.LGBMRegressor(**self.params)

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: Optional[np.ndarray] = None,
            eval_set: Optional[Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray]]] = None):
        self.feature_names = list(X.columns)
        logger.info(f"Training {self.name} on {X.shape[0]:,} rows × {X.shape[1]} features...")
        
        fit_kwargs: Dict[str, Any] = {"sample_weight": sample_weight}
        if eval_set is not None:
            X_val, y_val, w_val = eval_set
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["eval_sample_weight"] = [w_val] if w_val is not None else None
            fit_kwargs["callbacks"] = [lgb.early_stopping(stopping_rounds=30, verbose=False)]
            
        self.model.fit(X, y, **fit_kwargs)
        self.is_fitted = True
        logger.info(f"{self.name} training complete. Best iteration: {getattr(self.model, 'best_iteration_', self.params['n_estimators'])}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before predict.")
        # Ensure column order matches training
        X_aligned = X[self.feature_names]
        return self.model.predict(X_aligned)

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


class XGBoostModel(BaseMLModel):
    """
    XGBoost Regressor with Absolute Error Objective:
    Optimizes reg:absoluteerror with sample_weight (Holiday = 5, Non-holiday = 1).
    """
    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.08, max_depth: int = 6,
                 random_state: int = 42, n_jobs: int = -1, name: str = "XGBoost Regressor"):
        super().__init__(name)
        if not HAS_XGBOOST:
            raise ImportError("xgboost is not installed.")
        self.params = {
            "objective": "reg:absoluteerror",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "verbosity": 0
        }
        self.model = xgb.XGBRegressor(**self.params)

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: Optional[np.ndarray] = None,
            eval_set: Optional[Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray]]] = None):
        self.feature_names = list(X.columns)
        logger.info(f"Training {self.name} on {X.shape[0]:,} rows × {X.shape[1]} features...")
        
        fit_kwargs: Dict[str, Any] = {"sample_weight": sample_weight}
        if eval_set is not None:
            X_val, y_val, w_val = eval_set
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["sample_weight_eval_set"] = [w_val] if w_val is not None else None
            fit_kwargs["verbose"] = False
            
        self.model.fit(X, y, **fit_kwargs)
        self.is_fitted = True
        logger.info(f"{self.name} training complete.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before predict.")
        X_aligned = X[self.feature_names]
        return self.model.predict(X_aligned)

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


class CatBoostModel(BaseMLModel):
    """
    CatBoost Regressor with MAE Objective:
    Handles non-linear feature interactions and respects sample weights.
    """
    def __init__(self, iterations: int = 300, learning_rate: float = 0.08, depth: int = 6,
                 random_seed: int = 42, name: str = "CatBoost Regressor"):
        super().__init__(name)
        if not HAS_CATBOOST:
            raise ImportError("catboost is not installed.")
        self.params = {
            "loss_function": "MAE",
            "eval_metric": "MAE",
            "iterations": iterations,
            "learning_rate": learning_rate,
            "depth": depth,
            "random_seed": random_seed,
            "verbose": False
        }
        self.model = cb.CatBoostRegressor(**self.params)

    @classmethod
    def is_available(cls) -> bool:
        return HAS_CATBOOST

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: Optional[np.ndarray] = None,
            eval_set: Optional[Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray]]] = None):
        self.feature_names = list(X.columns)
        logger.info(f"Training {self.name} on {X.shape[0]:,} rows × {X.shape[1]} features...")
        
        fit_kwargs: Dict[str, Any] = {"sample_weight": sample_weight}
        if eval_set is not None:
            X_val, y_val, w_val = eval_set
            eval_pool = cb.Pool(X_val, y_val, weight=w_val)
            fit_kwargs["eval_set"] = eval_pool
            fit_kwargs["early_stopping_rounds"] = 30
            
        self.model.fit(X, y, **fit_kwargs)
        self.is_fitted = True
        logger.info(f"{self.name} training complete.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before predict.")
        X_aligned = X[self.feature_names]
        return self.model.predict(X_aligned)

    def get_feature_importances(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before computing feature importances.")
        importances = self.model.get_feature_importance()
        df_imp = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance_Gain": importances
        }).sort_values("Importance_Gain", ascending=False).reset_index(drop=True)
        total_gain = df_imp["Importance_Gain"].sum()
        df_imp["Relative_Importance_Pct"] = (df_imp["Importance_Gain"] / total_gain) * 100.0 if total_gain > 0 else 0.0
        return df_imp
