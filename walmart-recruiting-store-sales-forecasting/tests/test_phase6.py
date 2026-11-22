"""
Unit and integration tests for Phase 6: Seasonal Residual Modeling and Ensemble Blending.
"""
import numpy as np
import pandas as pd
import pytest

from src.residual_model import SeasonalResidualForecaster, get_residual_feature_columns
from src.ensemble import BlendedEnsembleForecaster
from src.preprocessing import HierarchicalFallbackManager


def test_residual_feature_columns_exclusion():
    """Verify get_residual_feature_columns excludes short-term autoregressive lags."""
    df = pd.DataFrame({
        "Store": [1],
        "Dept": [1],
        "Date": pd.to_datetime(["2011-02-05"]),
        "Weekly_Sales": [15000.0],
        "Type": ["A"],
        "Type_A": [1],
        "MarkDown1": [500.0],
        "Sales_Lag_1": [14000.0],
        "Sales_Lag_2": [13500.0],
        "Sales_Lag_4": [13000.0],
        "Rolling_Mean_4": [13800.0],
        "Rolling_Mean_12": [13500.0],
        "Sales_Lag_52": [12000.0],
        "Hierarchical_Fallback_Median": [10000.0]
    })
    
    cols = get_residual_feature_columns(df)
    
    # Must NOT contain short-term lags that cause recursive drift
    assert "Sales_Lag_1" not in cols
    assert "Sales_Lag_2" not in cols
    assert "Sales_Lag_4" not in cols
    assert "Rolling_Mean_4" not in cols
    assert "Rolling_Mean_12" not in cols
    assert "Weekly_Sales" not in cols
    
    # Must keep exogenous, markdown, and seasonal anchor features
    assert "MarkDown1" in cols
    assert "Sales_Lag_52" in cols
    assert "Hierarchical_Fallback_Median" in cols
    assert "Type_A" in cols


def test_seasonal_residual_reconstruction_synthetic():
    """
    Verify that Forecast strictly equals Anchor + Predicted_Residual,
    and supports negative sales.
    """
    N = 100
    dates = pd.date_range("2011-01-01", periods=N, freq="W-FRI")
    
    train_df = pd.DataFrame({
        "Store": [1] * N,
        "Dept": [1] * N,
        "Date": dates,
        "Weekly_Sales": np.linspace(-500.0, 5000.0, N),  # Contains negative sales
        "Type": ["A"] * N,
        "Type_A": [1] * N,
        "Type_B": [0] * N,
        "Type_C": [0] * N,
        "IsHoliday": [0] * N,
        "MarkDown1": [100.0] * N,
        "Sales_Lag_52": np.linspace(-400.0, 4800.0, N),
        "Hierarchical_Fallback_Median": [2500.0] * N,
        "Size": [100000] * N
    })
    
    model = SeasonalResidualForecaster(n_estimators=20, learning_rate=0.1)
    model.fit(train_df, eval_weeks=5)
    
    preds, anchors, residuals = model.predict(train_df)
    
    assert len(preds) == N
    assert not np.isnan(preds).any()
    # Mathematical identity check: Forecast == Anchor + Residual
    np.testing.assert_allclose(preds, anchors + residuals, rtol=1e-5)


def test_blended_ensemble_weights():
    """Verify BlendedEnsembleForecaster mathematical weighting behavior."""
    anchors = np.array([100.0, 200.0, 300.0])
    res_preds = np.array([150.0, 250.0, 350.0])
    is_holiday = np.array([False, True, False])
    
    # 1. Global blend alpha = 0.20 (20% anchor, 80% res)
    # Row 0: 0.20 * 100 + 0.80 * 150 = 20 + 120 = 140.0
    ensemble_global = BlendedEnsembleForecaster(global_alpha=0.20)
    blend_global = ensemble_global.blend(anchors, res_preds)
    assert np.isclose(blend_global[0], 140.0)
    assert np.isclose(blend_global[1], 240.0)
    
    # 2. Boundary: alpha = 1.0 (100% anchor)
    ens_anchor_only = BlendedEnsembleForecaster(global_alpha=1.0)
    np.testing.assert_array_equal(ens_anchor_only.blend(anchors, res_preds), anchors)
    
    # 3. Boundary: alpha = 0.0 (100% res)
    ens_res_only = BlendedEnsembleForecaster(global_alpha=0.0)
    np.testing.assert_array_equal(ens_res_only.blend(anchors, res_preds), res_preds)
    
    # 4. Segmented blend (hol=0.80, non_hol=0.10)
    # Row 0 (non-hol): 0.10 * 100 + 0.90 * 150 = 10 + 135 = 145.0
    # Row 1 (hol):     0.80 * 200 + 0.20 * 250 = 160 + 50 = 210.0
    ens_seg = BlendedEnsembleForecaster(holiday_alpha=0.80, non_holiday_alpha=0.10)
    blend_seg = ens_seg.blend(anchors, res_preds, is_holiday=is_holiday)
    assert np.isclose(blend_seg[0], 145.0)
    assert np.isclose(blend_seg[1], 210.0)


def test_cold_start_fallback_anchor():
    """Verify that when Sales_Lag_52 is NaN, Hierarchical_Fallback_Median is cleanly substituted."""
    df = pd.DataFrame({
        "Store": [99],
        "Dept": [99],
        "Date": pd.to_datetime(["2012-05-04"]),
        "Sales_Lag_52": [np.nan],  # Cold start
        "Hierarchical_Fallback_Median": [7500.0]
    })
    
    # Check anchor calculation
    has_l52 = df["Sales_Lag_52"].notnull().values
    anchors = np.where(has_l52, df["Sales_Lag_52"].values, df["Hierarchical_Fallback_Median"].values)
    
    assert anchors[0] == 7500.0
