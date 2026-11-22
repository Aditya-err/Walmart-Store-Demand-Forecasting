"""
Walmart Sales Forecasting — Baseline Forecasting Models Engine
Module: src/baseline_models.py
Author: Senior Data Scientist & Time-Series Forecasting Engineer

Provides reusable baseline forecasting models:
1. NaiveModel: Persists the last available historical sales value before forecast origin.
2. SeasonalNaiveModel: Predicts sales from exactly 52 weeks prior (same seasonal week last year).
3. MovingAverageModel: Predicts the mean of the last W historical weeks (default W=4).
4. ExponentialSmoothingModel: Fits Simple Exponential Smoothing (SES) level on historical series.
5. HierarchicalFallbackModel: Static hierarchical baseline for structural/cold-start benchmarking.

All models incorporate the HierarchicalFallbackManager to guarantee 100% coverage
for cold-start series or series with missing historical intervals without data leakage.
"""

import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from src.preprocessing import HierarchicalFallbackManager

logger = logging.getLogger("WalmartBaselines")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class BaseBaselineModel:
    """Abstract base class for all baseline forecasting models."""
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.fallback_manager: Optional[HierarchicalFallbackManager] = None

    def fit(self, train_df: pd.DataFrame, fallback_manager: Optional[HierarchicalFallbackManager] = None):
        raise NotImplementedError

    def predict(self, target_df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class NaiveModel(BaseBaselineModel):
    """
    Naive Forecast:
    Persists the last available historical weekly sales observation
    for each (Store, Dept) prior to the forecast origin.
    
    For a multi-step horizon without intermediate actuals, the last observed
    value is projected flatly across the entire forecast window.
    """
    def __init__(self, name: str = "Naive (Last Value)"):
        super().__init__(name)
        self.last_sales_map: Dict[Tuple[int, int], float] = {}

    def fit(self, train_df: pd.DataFrame, fallback_manager: Optional[HierarchicalFallbackManager] = None):
        logger.info(f"Fitting {self.name} on {len(train_df):,} training rows...")
        if fallback_manager is None:
            self.fallback_manager = HierarchicalFallbackManager().fit(train_df)
        else:
            self.fallback_manager = fallback_manager

        # Extract last observation per (Store, Dept)
        sorted_train = train_df.sort_values(["Store", "Dept", "Date"])
        last_obs = sorted_train.groupby(["Store", "Dept"])["Weekly_Sales"].last()
        self.last_sales_map = last_obs.to_dict()
        
        self.is_fitted = True
        logger.info(f"{self.name} fitted successfully across {len(self.last_sales_map):,} historical series.")
        return self

    def predict(self, target_df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before predict.")
        
        fallbacks = self.fallback_manager.transform(target_df).values
        stores = target_df["Store"].values
        depts = target_df["Dept"].values
        
        preds = np.empty(len(target_df), dtype=float)
        for i in range(len(target_df)):
            key = (stores[i], depts[i])
            if key in self.last_sales_map:
                preds[i] = self.last_sales_map[key]
            else:
                preds[i] = fallbacks[i]
                
        return preds


class SeasonalNaiveModel(BaseBaselineModel):
    """
    Seasonal Naive Forecast (SNaive):
    Predicts the sales from the exact same calendar week 52 weeks prior:
        y_hat_{t} = y_{t - 52 weeks}
        
    For our 52-week validation set (2011-11-04 to 2012-10-26), the 52-week prior
    period is 2010-11-05 to 2011-10-28, which lies strictly within the historical training split.
    
    If an individual series was unobserved 52 weeks prior (incomplete series),
    the model seamlessly falls back to the HierarchicalFallbackManager.
    """
    def __init__(self, name: str = "Seasonal Naive (Lag 52)"):
        super().__init__(name)
        self.historical_lookup: Dict[Tuple[int, int, pd.Timestamp], float] = {}

    def fit(self, train_df: pd.DataFrame, fallback_manager: Optional[HierarchicalFallbackManager] = None):
        logger.info(f"Fitting {self.name} on {len(train_df):,} training rows...")
        if fallback_manager is None:
            self.fallback_manager = HierarchicalFallbackManager().fit(train_df)
        else:
            self.fallback_manager = fallback_manager

        # Map (Store, Dept, Date) -> Weekly_Sales
        train_dates = pd.to_datetime(train_df["Date"])
        subset = train_df[["Store", "Dept"]].copy()
        subset["Date"] = train_dates
        subset["Weekly_Sales"] = train_df["Weekly_Sales"].values
        
        self.historical_lookup = {
            (row.Store, row.Dept, row.Date): row.Weekly_Sales
            for row in subset.itertuples(index=False)
        }
        
        self.is_fitted = True
        logger.info(f"{self.name} fitted successfully with {len(self.historical_lookup):,} historical date keys.")
        return self

    def predict(self, target_df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before predict.")
            
        fallbacks = self.fallback_manager.transform(target_df).values
        stores = target_df["Store"].values
        depts = target_df["Dept"].values
        dates = pd.to_datetime(target_df["Date"]).values
        
        preds = np.empty(len(target_df), dtype=float)
        fallback_count = 0
        
        for i in range(len(target_df)):
            # 52 weeks = 364 days
            lag_date = pd.Timestamp(dates[i]) - pd.Timedelta(weeks=52)
            key = (stores[i], depts[i], lag_date)
            
            if key in self.historical_lookup:
                preds[i] = self.historical_lookup[key]
            else:
                preds[i] = fallbacks[i]
                fallback_count += 1
                
        if fallback_count > 0:
            logger.info(f"{self.name}: {fallback_count:,} / {len(target_df):,} predictions "
                        f"({fallback_count/len(target_df):.2%}) used hierarchical fallback due to missing 52-week lag.")
        return preds


class MovingAverageModel(BaseBaselineModel):
    """
    Moving Average Forecast (MA-W):
    Computes the average sales of the last W historical observations
    (default W=4 weeks) prior to the forecast origin for each (Store, Dept).
    
    In multi-step horizon forecasting without actual updates, this rolling
    historical average projects flatly across the horizon.
    """
    def __init__(self, window: int = 4, name: Optional[str] = None):
        name = name or f"Moving Average ({window}-Week)"
        super().__init__(name)
        self.window = window
        self.ma_map: Dict[Tuple[int, int], float] = {}

    def fit(self, train_df: pd.DataFrame, fallback_manager: Optional[HierarchicalFallbackManager] = None):
        logger.info(f"Fitting {self.name} on {len(train_df):,} training rows...")
        if fallback_manager is None:
            self.fallback_manager = HierarchicalFallbackManager().fit(train_df)
        else:
            self.fallback_manager = fallback_manager

        sorted_train = train_df.sort_values(["Store", "Dept", "Date"])
        
        # Take the tail W observations per (Store, Dept)
        ma_series = (
            sorted_train.groupby(["Store", "Dept"])["Weekly_Sales"]
            .apply(lambda s: s.iloc[-self.window:].mean())
        )
        self.ma_map = ma_series.to_dict()
        
        self.is_fitted = True
        logger.info(f"{self.name} fitted successfully across {len(self.ma_map):,} series.")
        return self

    def predict(self, target_df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before predict.")
            
        fallbacks = self.fallback_manager.transform(target_df).values
        stores = target_df["Store"].values
        depts = target_df["Dept"].values
        
        preds = np.empty(len(target_df), dtype=float)
        for i in range(len(target_df)):
            key = (stores[i], depts[i])
            if key in self.ma_map:
                preds[i] = self.ma_map[key]
            else:
                preds[i] = fallbacks[i]
                
        return preds


class ExponentialSmoothingModel(BaseBaselineModel):
    """
    Simple Exponential Smoothing Model (SES):
    Fits an exponential smoothing level model on each historical (Store, Dept) series.
    
    For series with >= 4 historical observations, estimates the optimal smoothing
    parameter alpha. For shorter or degenerate series, smoothly falls back to the
    last observation or hierarchical fallback.
    """
    def __init__(self, name: str = "Exponential Smoothing (SES)"):
        super().__init__(name)
        self.forecast_map: Dict[Tuple[int, int], float] = {}

    def fit(self, train_df: pd.DataFrame, fallback_manager: Optional[HierarchicalFallbackManager] = None):
        logger.info(f"Fitting {self.name} on {len(train_df):,} training rows...")
        if fallback_manager is None:
            self.fallback_manager = HierarchicalFallbackManager().fit(train_df)
        else:
            self.fallback_manager = fallback_manager

        sorted_train = train_df.sort_values(["Store", "Dept", "Date"])
        
        # Fit SES for each series
        for (store, dept), grp in sorted_train.groupby(["Store", "Dept"]):
            series = grp["Weekly_Sales"].values
            if len(series) >= 4:
                try:
                    # Exponential smoothing level forecast
                    ses_fit = SimpleExpSmoothing(series, initialization_method="estimated").fit()
                    self.forecast_map[(store, dept)] = float(ses_fit.forecast(1)[0])
                except Exception:
                    self.forecast_map[(store, dept)] = float(series[-1])
            elif len(series) > 0:
                self.forecast_map[(store, dept)] = float(series[-1])

        self.is_fitted = True
        logger.info(f"{self.name} fitted successfully across {len(self.forecast_map):,} series.")
        return self

    def predict(self, target_df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before predict.")
            
        fallbacks = self.fallback_manager.transform(target_df).values
        stores = target_df["Store"].values
        depts = target_df["Dept"].values
        
        preds = np.empty(len(target_df), dtype=float)
        for i in range(len(target_df)):
            key = (stores[i], depts[i])
            if key in self.forecast_map:
                preds[i] = self.forecast_map[key]
            else:
                preds[i] = fallbacks[i]
                
        return preds


class HistoricalMedianModel(BaseBaselineModel):
    """
    Store-Department Historical Median Model:
    Predicts the historical median sales for each (Store, Dept) across all training weeks.
    Provides a highly robust non-seasonal benchmark resistant to holiday skewness.
    """
    def __init__(self, name: str = "Store-Dept Historical Median"):
        super().__init__(name)
        self.median_map: Dict[Tuple[int, int], float] = {}

    def fit(self, train_df: pd.DataFrame, fallback_manager: Optional[HierarchicalFallbackManager] = None):
        logger.info(f"Fitting {self.name} on {len(train_df):,} training rows...")
        if fallback_manager is None:
            self.fallback_manager = HierarchicalFallbackManager().fit(train_df)
        else:
            self.fallback_manager = fallback_manager

        medians = train_df.groupby(["Store", "Dept"])["Weekly_Sales"].median()
        self.median_map = medians.to_dict()
        
        self.is_fitted = True
        logger.info(f"{self.name} fitted successfully across {len(self.median_map):,} series.")
        return self

    def predict(self, target_df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before predict.")
            
        fallbacks = self.fallback_manager.transform(target_df).values
        stores = target_df["Store"].values
        depts = target_df["Dept"].values
        
        preds = np.empty(len(target_df), dtype=float)
        for i in range(len(target_df)):
            key = (stores[i], depts[i])
            if key in self.median_map:
                preds[i] = self.median_map[key]
            else:
                preds[i] = fallbacks[i]
                
        return preds


class HierarchicalFallbackModel(BaseBaselineModel):
    """
    Pure Hierarchical Fallback Model:
    Generates forecasts strictly from the structural hierarchy:
    (Store Type, Dept) Median -> Dept Median -> Store Median -> Global Median.
    Serves as the foundational lower-bound benchmark for cold-start estimation.
    """
    def __init__(self, name: str = "Hierarchical Fallback Baseline"):
        super().__init__(name)

    def fit(self, train_df: pd.DataFrame, fallback_manager: Optional[HierarchicalFallbackManager] = None):
        logger.info(f"Fitting {self.name} on {len(train_df):,} training rows...")
        if fallback_manager is None:
            self.fallback_manager = HierarchicalFallbackManager().fit(train_df)
        else:
            self.fallback_manager = fallback_manager
            
        self.is_fitted = True
        return self

    def predict(self, target_df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before predict.")
        return self.fallback_manager.transform(target_df).values
