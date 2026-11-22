"""
Unit tests for evaluation metrics (WMAE, MAE, RMSE) in Walmart Sales Forecasting.
"""
import numpy as np
import pytest
from src.evaluation import compute_wmae, compute_mae, compute_rmse, evaluate_forecast


def test_wmae_calculation_exact():
    """
    Test WMAE on hand-calculated synthetic cases.
    Weights: 5 for holiday, 1 for non-holiday.
    """
    # Case 1: 2 rows: 1 non-holiday (err=10, w=1), 1 holiday (err=20, w=5)
    # WMAE = (10*1 + 20*5) / (1 + 5) = 110 / 6 = 18.333333333333332
    y_true = np.array([100.0, 100.0])
    y_pred = np.array([90.0, 80.0])
    is_holiday = np.array([False, True])
    
    wmae = compute_wmae(y_true, y_pred, is_holiday)
    assert np.isclose(wmae, 110.0 / 6.0), f"Expected {110.0/6.0}, got {wmae}"


def test_wmae_zero_error():
    """Perfect predictions should yield 0.0 error."""
    y_true = np.array([500.0, 1000.0, 1500.0])
    y_pred = np.array([500.0, 1000.0, 1500.0])
    is_holiday = np.array([False, True, False])
    
    assert compute_wmae(y_true, y_pred, is_holiday) == 0.0
    assert compute_mae(y_true, y_pred) == 0.0
    assert compute_rmse(y_true, y_pred) == 0.0


def test_wmae_all_holidays_vs_all_non_holidays():
    """When all are holidays or non-holidays, WMAE should equal MAE."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 320.0])
    
    # All non-holiday
    is_holiday_false = np.array([False, False, False])
    wmae_false = compute_wmae(y_true, y_pred, is_holiday_false)
    mae = compute_mae(y_true, y_pred)
    assert np.isclose(wmae_false, mae)

    # All holiday
    is_holiday_true = np.array([True, True, True])
    wmae_true = compute_wmae(y_true, y_pred, is_holiday_true)
    assert np.isclose(wmae_true, mae)


def test_wmae_holiday_penalty():
    """
    An error on a holiday must be penalized 5x more heavily than
    an equivalent error on a non-holiday.
    """
    y_true = np.array([100.0])
    y_pred = np.array([150.0])  # error = 50
    
    wmae_non_holiday = compute_wmae(y_true, y_pred, np.array([False]))
    wmae_holiday = compute_wmae(y_true, y_pred, np.array([True]))
    
    assert wmae_non_holiday == 50.0
    assert wmae_holiday == 50.0
    
    # In mixed dataset:
    # Row 1: error = 100 on non-holiday (w=1) -> contrib 100
    # Row 2: error = 0 on holiday (w=5)       -> contrib 0
    # total = 100 / 6 = 16.67
    wmae_1 = compute_wmae(np.array([100.0, 100.0]), np.array([0.0, 100.0]), np.array([False, True]))
    
    # Row 1: error = 0 on non-holiday (w=1)   -> contrib 0
    # Row 2: error = 100 on holiday (w=5)     -> contrib 500
    # total = 500 / 6 = 83.33
    wmae_2 = compute_wmae(np.array([100.0, 100.0]), np.array([100.0, 0.0]), np.array([False, True]))
    
    assert np.isclose(wmae_2 / wmae_1, 5.0)


def test_evaluate_forecast_dictionary():
    """Test full dictionary returned by evaluate_forecast."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 35.0, 38.0])
    is_holiday = np.array([False, True, False, True])
    
    metrics = evaluate_forecast(y_true, y_pred, is_holiday)
    assert "WMAE" in metrics
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "WMAE_Holiday" in metrics
    assert "WMAE_NonHoliday" in metrics
    assert "Total_Observations" in metrics
    assert "Holiday_Observations" in metrics
    assert "NonHoliday_Observations" in metrics
    assert metrics["Total_Observations"] == 4
    assert metrics["Holiday_Observations"] == 2
    assert metrics["NonHoliday_Observations"] == 2


def test_input_validation_errors():
    """Ensure dimension mismatch raises ValueError."""
    with pytest.raises(ValueError):
        compute_wmae(np.array([1.0, 2.0]), np.array([1.0]), np.array([False, False]))
        
    with pytest.raises(ValueError):
        compute_wmae(np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([False]))
