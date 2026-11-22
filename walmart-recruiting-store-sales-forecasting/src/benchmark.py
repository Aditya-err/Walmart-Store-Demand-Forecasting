"""
Walmart Sales Forecasting — Baseline Benchmarking Orchestrator
Module: src/benchmark.py
Author: Senior Data Scientist & Time-Series Forecasting Engineer

Orchestrates the evaluation of all baseline forecasting models on the 52-week validation horizon:
- Fits models strictly on the historical training split (2010-02-05 to 2011-10-28; 267,184 rows).
- Evaluates out-of-sample predictions on the validation horizon (2011-11-04 to 2012-10-26; 154,386 rows).
- Computes official Walmart WMAE, MAE, RMSE, Holiday WMAE, and Non-Holiday WMAE.
- Analyzes sliced performance across Store Types (A, B, C) and retail holiday events.
- Exports comprehensive artifacts to outputs/baseline/.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.baseline_models import (
    NaiveModel,
    SeasonalNaiveModel,
    MovingAverageModel,
    ExponentialSmoothingModel,
    HistoricalMedianModel,
    HierarchicalFallbackModel
)
from src.evaluation import compute_wmae, compute_mae, compute_rmse, evaluate_forecast
from src.preprocessing import HierarchicalFallbackManager

logger = logging.getLogger("WalmartBenchmark")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path("E:/MCA projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting")
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs" / "baseline"


def run_benchmark():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=== STARTING PHASE 4 BASELINE BENCHMARKING PIPELINE ===")

    # 1. Load data
    train_path = DATA_DIR / "train_engineered.parquet"
    val_path = DATA_DIR / "val_engineered.parquet"
    
    logger.info(f"Loading training data: {train_path}")
    train_df = pd.read_parquet(train_path)
    logger.info(f"Loading validation data: {val_path}")
    val_df = pd.read_parquet(val_path)

    y_true = val_df["Weekly_Sales"].values
    is_holiday = val_df["IsHoliday"].astype(bool).values
    total_val_rows = len(val_df)
    assert total_val_rows == 154386, f"Expected 154,386 validation rows, got {total_val_rows:,}"

    # 2. Fit shared HierarchicalFallbackManager
    logger.info("Fitting shared HierarchicalFallbackManager on training data...")
    fallback_mgr = HierarchicalFallbackManager().fit(train_df)

    # 3. Instantiate baseline models
    models = [
        HierarchicalFallbackModel("Hierarchical Fallback Baseline"),
        NaiveModel("Naive (Last Value)"),
        MovingAverageModel(window=4, name="Moving Average (4-Week)"),
        ExponentialSmoothingModel("Exponential Smoothing (SES)"),
        HistoricalMedianModel("Store-Dept Historical Median"),
        SeasonalNaiveModel("Seasonal Naive (Lag 52)")
    ]

    predictions_dict = {
        "Store": val_df["Store"].values,
        "Dept": val_df["Dept"].values,
        "Date": val_df["Date"].values,
        "Type": val_df["Type"].values,
        "IsHoliday": val_df["IsHoliday"].values,
        "Weekly_Sales": y_true
    }

    metrics_list = []
    store_type_records = []
    holiday_records = []

    logger.info(f"Evaluating {len(models)} baseline models across {total_val_rows:,} validation observations...")

    for model in models:
        logger.info(f"\n--- Running: {model.name} ---")
        
        # Fit timing
        t0_fit = time.time()
        model.fit(train_df, fallback_manager=fallback_mgr)
        fit_time = time.time() - t0_fit

        # Predict timing
        t0_pred = time.time()
        preds = model.predict(val_df)
        pred_time = time.time() - t0_pred

        # Assert output validity
        assert len(preds) == total_val_rows, f"Prediction length mismatch for {model.name}: {len(preds)}"
        nan_count = np.isnan(preds).sum()
        assert nan_count == 0, f"{nan_count} NaNs detected in predictions of {model.name}"

        # Store predictions
        col_name = "Pred_" + model.name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        predictions_dict[col_name] = preds

        # Overall Metrics
        m = evaluate_forecast(y_true, preds, is_holiday)
        metrics_record = {
            "Model": model.name,
            "WMAE": round(m["WMAE"], 2),
            "MAE": round(m["MAE"], 2),
            "RMSE": round(m["RMSE"], 2),
            "WMAE_Holiday": round(m["WMAE_Holiday"], 2),
            "WMAE_NonHoliday": round(m["WMAE_NonHoliday"], 2),
            "WMAE_to_MAE_Ratio": round(m["WMAE_to_MAE_Ratio"], 4),
            "Fit_Time_Sec": round(fit_time, 2),
            "Pred_Time_Sec": round(pred_time, 2)
        }
        metrics_list.append(metrics_record)

        # Store Type Breakdown
        for stype in ["A", "B", "C"]:
            type_mask = (val_df["Type"] == stype).values
            type_wmae = compute_wmae(y_true[type_mask], preds[type_mask], is_holiday[type_mask])
            type_mae = compute_mae(y_true[type_mask], preds[type_mask])
            store_type_records.append({
                "Model": model.name,
                "Store_Type": stype,
                "Observations": int(np.sum(type_mask)),
                "WMAE": round(type_wmae, 2),
                "MAE": round(type_mae, 2)
            })

        # Holiday Event Slices
        holiday_events = {
            "Thanksgiving (Week 47)": val_df["Date"] == "2011-11-25",
            "Christmas (Week 52)": val_df["Date"] == "2011-12-30",
            "Super Bowl (Week 6)": val_df["Date"] == "2012-02-10",
            "Labor Day (Week 36)": val_df["Date"] == "2012-09-07"
        }
        for event_name, event_mask in holiday_events.items():
            if event_mask.any():
                e_wmae = compute_wmae(y_true[event_mask.values], preds[event_mask.values], is_holiday[event_mask.values])
                e_mae = compute_mae(y_true[event_mask.values], preds[event_mask.values])
                holiday_records.append({
                    "Model": model.name,
                    "Holiday_Event": event_name,
                    "Observations": int(event_mask.sum()),
                    "WMAE": round(e_wmae, 2),
                    "MAE": round(e_mae, 2)
                })

        logger.info(f"--> {model.name} Result: WMAE = ${m['WMAE']:,.2f} | MAE = ${m['MAE']:,.2f} | Hol_WMAE = ${m['WMAE_Holiday']:,.2f}")

    # 4. Save Artifacts
    logger.info("\n=== SAVING BENCHMARK ARTIFACTS ===")
    
    # A. Predictions Parquet
    preds_df = pd.DataFrame(predictions_dict)
    preds_parquet_path = OUTPUT_DIR / "baseline_predictions.parquet"
    preds_df.to_parquet(preds_parquet_path, index=False)
    logger.info(f"Saved validation predictions to: {preds_parquet_path} ({preds_df.shape})")

    # B. Metrics Comparison Table
    metrics_df = pd.DataFrame(metrics_list).sort_values("WMAE").reset_index(drop=True)
    metrics_csv_path = OUTPUT_DIR / "baseline_comparison_metrics.csv"
    metrics_json_path = OUTPUT_DIR / "baseline_comparison_metrics.json"
    metrics_df.to_csv(metrics_csv_path, index=False)
    metrics_df.to_json(metrics_json_path, orient="records", indent=2)
    logger.info(f"Saved overall metrics to: {metrics_csv_path}")

    # C. Store Type Breakdown
    type_df = pd.DataFrame(store_type_records)
    type_csv_path = OUTPUT_DIR / "baseline_by_store_type.csv"
    type_df.to_csv(type_csv_path, index=False)
    logger.info(f"Saved store type breakdown to: {type_csv_path}")

    # D. Holiday Event Breakdown
    hol_df = pd.DataFrame(holiday_records)
    hol_csv_path = OUTPUT_DIR / "baseline_holiday_breakdown.csv"
    hol_df.to_csv(hol_csv_path, index=False)
    logger.info(f"Saved holiday breakdown to: {hol_csv_path}")

    # 5. Display Final Summary Table
    print("\n" + "="*95)
    print("WALMART SALES FORECASTING — PHASE 4 BASELINE BENCHMARK SUMMARY")
    print("="*95)
    print(metrics_df.to_string(index=False))
    print("="*95)
    
    best_model = metrics_df.iloc[0]
    print(f"\nCHAMPION BASELINE MODEL: {best_model['Model']}")
    print(f"Official WMAE: ${best_model['WMAE']:,.2f} | MAE: ${best_model['MAE']:,.2f} | RMSE: ${best_model['RMSE']:,.2f}")
    print(f"Holiday WMAE: ${best_model['WMAE_Holiday']:,.2f} | Non-Holiday WMAE: ${best_model['WMAE_NonHoliday']:,.2f}")
    print("="*95)
    
    return metrics_df, preds_df


if __name__ == "__main__":
    run_benchmark()
