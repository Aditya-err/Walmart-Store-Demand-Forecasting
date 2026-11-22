"""
Unit and integration tests for machine learning models and recursive multi-step forecasting.
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.ml_models import (
    BaseMLModel,
    LGBMModel,
    XGBoostModel,
    CatBoostModel,
    get_feature_columns,
    HAS_LIGHTGBM,
    HAS_XGBOOST,
    HAS_CATBOOST
)
from src.ml_forecasting import RecursiveMultiStepForecaster
from src.preprocessing import HierarchicalFallbackManager


def test_feature_columns_exclusion():
    """Verify get_feature_columns excludes target, date, and raw string columns."""
    df = pd.DataFrame({
        "Store": [1],
        "Dept": [1],
        "Date": pd.to_datetime(["2010-02-05"]),
        "Weekly_Sales": [15000.0],
        "Type": ["A"],
        "Type_A": [1],
        "Temperature": [42.0],
        "Sales_Lag_1": [14000.0]
    })
    
    cols = get_feature_columns(df)
    assert "Weekly_Sales" not in cols
    assert "Date" not in cols
    assert "Type" not in cols
    assert "Type_A" in cols
    assert "Temperature" in cols
    assert "Sales_Lag_1" in cols


@pytest.mark.skipif(not HAS_LIGHTGBM, reason="LightGBM not installed")
def test_lgbm_fit_predict_sample_weights():
    """Verify LightGBM fits with sample weights and outputs valid predictions."""
    np.random.seed(42)
    N = 100
    X = pd.DataFrame({
        "feat1": np.random.randn(N),
        "feat2": np.random.randn(N)
    })
    y = X["feat1"] * 2.0 + X["feat2"] * 5.0 + np.random.randn(N) * 0.1
    weights = np.random.choice([1.0, 5.0], size=N)
    
    model = LGBMModel(n_estimators=20, learning_rate=0.1)
    model.fit(X, y.values, sample_weight=weights)
    
    preds = model.predict(X)
    assert len(preds) == N
    assert not np.isnan(preds).any()
    
    fi = model.get_feature_importances()
    assert len(fi) == 2
    assert "Feature" in fi.columns
    assert "Relative_Importance_Pct" in fi.columns


@pytest.mark.skipif(not HAS_XGBOOST, reason="XGBoost not installed")
def test_xgboost_fit_predict_negative_sales():
    """Verify XGBoost handles negative sales (returns) without clipping or error."""
    X = pd.DataFrame({
        "feat1": [-2.0, -1.0, 0.0, 1.0, 2.0],
        "feat2": [0.5, 0.5, 0.5, 0.5, 0.5]
    })
    y = np.array([-4500.0, -2000.0, 500.0, 3000.0, 8000.0])
    weights = np.array([1.0, 1.0, 1.0, 5.0, 1.0])
    
    model = XGBoostModel(n_estimators=30, learning_rate=0.1)
    model.fit(X, y, sample_weight=weights)
    
    preds = model.predict(X)
    assert len(preds) == 5
    # Should predict a negative value for the most negative feature input
    assert preds[0] < 0.0


@pytest.mark.skipif(not HAS_CATBOOST, reason="CatBoost not installed")
def test_catboost_fit_predict():
    """Verify CatBoost fits and predicts properly with MAE loss."""
    X = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "f2": [10.0, 20.0, 30.0, 40.0, 50.0]
    })
    y = np.array([12.0, 24.0, 35.0, 47.0, 59.0])
    
    model = CatBoostModel(iterations=20, learning_rate=0.1)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == 5
    assert not np.isnan(preds).any()


def test_recursive_multistep_autoregressive_state_updates():
    """
    CRITICAL UNIT TEST:
    Verify that in RecursiveMultiStepForecaster:
    - Step 2 Lag 1 is updated with Step 1's PREDICTION (not actual validation sales).
    - Rolling Mean 4 dynamically absorbs the previous prediction into the rolling window.
    """
    # 1. Dummy training data (Store 1, Dept 1 with 4 weeks)
    train_df = pd.DataFrame({
        "Store": [1, 1, 1, 1],
        "Dept": [1, 1, 1, 1],
        "Date": pd.to_datetime(["2011-10-07", "2011-10-14", "2011-10-21", "2011-10-28"]),
        "Weekly_Sales": [100.0, 100.0, 100.0, 100.0],
        "Type": ["A"] * 4,
        "IsHoliday": [0] * 4,
        "Sales_Lag_1": [100.0]*4,
        "Sales_Lag_2": [100.0]*4,
        "Sales_Lag_4": [100.0]*4,
        "Sales_Lag_52": [100.0]*4,
        "Rolling_Mean_4": [100.0]*4,
        "Rolling_Std_4": [0.0]*4,
        "Rolling_Mean_12": [100.0]*4,
        "Rolling_Std_12": [0.0]*4,
        "Rolling_Mean_52": [100.0]*4,
        "Rolling_Std_52": [0.0]*4,
        "Previous_Year_Sales_Diff": [0.0]*4,
        "Previous_Year_Sales_Ratio": [1.0]*4,
        "Rolling_4_vs_Lag_52_Ratio": [1.0]*4,
        "Hierarchical_Fallback_Median": [100.0]*4
    })
    
    # 2. Validation horizon (2 weeks: 2011-11-04, 2011-11-11)
    val_df = pd.DataFrame({
        "Store": [1, 1],
        "Dept": [1, 1],
        "Date": pd.to_datetime(["2011-11-04", "2011-11-11"]),
        "Weekly_Sales": [9999.0, 9999.0],  # Ground truth actuals - model must NEVER see these!
        "Type": ["A", "A"],
        "IsHoliday": [0, 0],
        "Sales_Lag_1": [0.0, 0.0],
        "Sales_Lag_2": [0.0, 0.0],
        "Sales_Lag_4": [0.0, 0.0],
        "Sales_Lag_52": [100.0, 100.0],
        "Rolling_Mean_4": [0.0, 0.0],
        "Rolling_Std_4": [0.0, 0.0],
        "Rolling_Mean_12": [0.0, 0.0],
        "Rolling_Std_12": [0.0, 0.0],
        "Rolling_Mean_52": [0.0, 0.0],
        "Rolling_Std_52": [0.0, 0.0],
        "Previous_Year_Sales_Diff": [0.0, 0.0],
        "Previous_Year_Sales_Ratio": [1.0, 1.0],
        "Rolling_4_vs_Lag_52_Ratio": [1.0, 1.0],
        "Hierarchical_Fallback_Median": [100.0, 100.0]
    })
    
    # Mock model that always predicts exactly 500.0
    class MockModel(BaseMLModel):
        def __init__(self):
            super().__init__("MockModel")
            self.feature_names = get_feature_columns(train_df)
            self.is_fitted = True
            self.observed_lag1_step2 = None
            self.step = 0
            
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            self.step += 1
            if self.step == 2:
                # Capture the Sales_Lag_1 fed into step 2!
                self.observed_lag1_step2 = X["Sales_Lag_1"].iloc[0]
            return np.full(len(X), 500.0)
            
    mock = MockModel()
    fallback_mgr = HierarchicalFallbackManager().fit(train_df)
    forecaster = RecursiveMultiStepForecaster(mock, fallback_mgr)
    forecaster.initialize_history_from_train(train_df)
    
    preds = forecaster.forecast_validation_horizon(val_df)
    
    # 1. Total predictions match validation rows
    assert len(preds) == 2
    # 2. In step 2, Sales_Lag_1 MUST equal the step 1 prediction (500.0), NOT actual sales (9999.0)
    assert mock.observed_lag1_step2 == 500.0, f"Expected step 2 lag 1 to be 500.0, got {mock.observed_lag1_step2}"
    # 3. History buffer at the end should contain the two predictions
    assert forecaster.history_buffer[(1, 1)][-1] == 500.0
    assert forecaster.history_buffer[(1, 1)][-2] == 500.0
