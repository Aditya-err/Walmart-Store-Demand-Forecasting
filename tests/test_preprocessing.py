"""
Unit tests for data preprocessing and cleaning modules in Walmart Sales Forecasting.
"""
import numpy as np
import pandas as pd
import pytest
from src.preprocessing import (
    HierarchicalFallbackManager,
    handle_markdown_missingness,
    handle_macroeconomic_missingness,
    clean_and_prepare_dataset
)


def test_negative_sales_preservation():
    """Verify negative sales records (returns/refunds) are strictly preserved."""
    df = pd.DataFrame({
        "Store": [1, 1, 1],
        "Dept": [1, 1, 1],
        "Date": pd.to_datetime(["2010-02-05", "2010-02-12", "2010-02-19"]),
        "Weekly_Sales": [-4988.94, 0.0, 15000.50],
        "IsHoliday": [False, True, False],
        "Type": ["A", "A", "A"],
        "Size": [151315, 151315, 151315],
        "Temperature": [42.31, 38.51, 39.93],
        "Fuel_Price": [2.572, 2.548, 2.514],
        "MarkDown1": [np.nan, np.nan, np.nan],
        "MarkDown2": [np.nan, np.nan, np.nan],
        "MarkDown3": [np.nan, np.nan, np.nan],
        "MarkDown4": [np.nan, np.nan, np.nan],
        "MarkDown5": [np.nan, np.nan, np.nan],
        "CPI": [211.09, 211.24, 211.28],
        "Unemployment": [8.106, 8.106, 8.106],
    })
    
    cleaned_df, fallback_mgr = clean_and_prepare_dataset(df, is_train=True)
    
    assert len(cleaned_df) == 3
    assert -4988.94 in cleaned_df["Weekly_Sales"].values
    assert (cleaned_df["Weekly_Sales"] < 0).sum() == 1


def test_markdown_missingness_imputation_and_indicators():
    """Verify Markdown NaN values are imputed to 0.0 and indicator flags are generated."""
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2010-02-05", "2011-11-11", "2012-05-04"]),
        "MarkDown1": [np.nan, 150.0, 0.0],
        "MarkDown2": [50.0, np.nan, np.nan],
        "MarkDown3": [np.nan, np.nan, np.nan],
        "MarkDown4": [np.nan, 20.0, np.nan],
        "MarkDown5": [10.0, np.nan, np.nan],
    })
    
    result = handle_markdown_missingness(df)
    
    # 1. No NaNs remaining in Markdown columns
    assert result[["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]].isna().sum().sum() == 0
    
    # 2. Values properly set
    assert result.loc[0, "MarkDown1"] == 0.0
    assert result.loc[1, "MarkDown1"] == 150.0
    
    # 3. Indicators created
    assert result.loc[0, "MarkDown1_Raw_Null"] == 1
    assert result.loc[1, "MarkDown1_Raw_Null"] == 0
    assert result.loc[0, "Has_MarkDown_Data"] == 1  # has active markdown values
    assert result.loc[1, "Has_MarkDown_Data"] == 1  # date >= 2011-11-04
    
    # 4. Total and Active MarkDown count
    assert result.loc[0, "MarkDown_Total"] == 60.0  # 50 + 10
    assert result.loc[0, "MarkDown_Count_Active"] == 2
    assert result.loc[2, "MarkDown_Total"] == 0.0
    assert result.loc[2, "MarkDown_Count_Active"] == 0


def test_macroeconomic_forward_fill():
    """Verify CPI and Unemployment are forward-filled per Store, and flagged."""
    df = pd.DataFrame({
        "Store": [1, 1, 1, 2, 2],
        "Date": pd.to_datetime(["2013-04-19", "2013-04-26", "2013-05-03", "2013-04-26", "2013-05-03"]),
        "CPI": [220.0, 221.0, np.nan, 135.0, np.nan],
        "Unemployment": [7.5, 7.4, np.nan, 6.2, np.nan],
    })
    
    result = handle_macroeconomic_missingness(df)
    
    assert result["CPI"].isna().sum() == 0
    assert result["Unemployment"].isna().sum() == 0
    
    # Store 1 on 2013-05-03 should forward fill from 2013-04-26 (221.0)
    store1_last = result[(result["Store"] == 1) & (result["Date"] == "2013-05-03")].iloc[0]
    assert store1_last["CPI"] == 221.0
    assert store1_last["Unemployment"] == 7.4
    assert store1_last["Is_Macro_Imputed"] == 1
    
    # Store 2 on 2013-05-03 should forward fill from 135.0
    store2_last = result[(result["Store"] == 2) & (result["Date"] == "2013-05-03")].iloc[0]
    assert store2_last["CPI"] == 135.0
    assert store2_last["Unemployment"] == 6.2
    assert store2_last["Is_Macro_Imputed"] == 1


def test_hierarchical_fallback_manager_cold_start():
    """
    Test hierarchical fallback resolution for cold-start Store-Department combinations:
    Hierarchy: (Type, Dept) -> Dept -> Store -> Global.
    """
    train_df = pd.DataFrame({
        "Store": [1, 1, 2, 2],
        "Dept": [1, 2, 1, 2],
        "Type": ["A", "A", "B", "B"],
        "Weekly_Sales": [1000.0, 2000.0, 3000.0, 4000.0]
    })
    
    fallback_mgr = HierarchicalFallbackManager()
    fallback_mgr.fit(train_df)
    
    # Unseen Store 3, Dept 1, Type A -> Matches (Type A, Dept 1) median = 1000.0
    fb_val_1 = fallback_mgr.get_fallback_median(store=3, store_type="A", dept=1)
    assert fb_val_1 == 1000.0
    
    # Completely unseen Store 99, Dept 999, Type C -> Matched at level 4 (Global) = median([1000, 2000, 3000, 4000]) = 2500.0
    fb_val_2 = fallback_mgr.get_fallback_median(store=99, store_type="C", dept=999)
    assert fb_val_2 == 2500.0
    
    # Store 1, unseen Dept 999, Type A -> Matched at level 3 (Store 1) = median([1000, 2000]) = 1500.0
    fb_val_3 = fallback_mgr.get_fallback_median(store=1, store_type="A", dept=999)
    assert fb_val_3 == 1500.0
    
    # Transform batch
    val_unseen = pd.DataFrame({
        "Store": [3, 99, 1],
        "Dept": [1, 999, 999],
        "Type": ["A", "C", "A"]
    })
    fallbacks = fallback_mgr.transform(val_unseen)
    assert list(fallbacks) == [1000.0, 2500.0, 1500.0]
