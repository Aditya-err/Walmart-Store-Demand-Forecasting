"""
Walmart Sales Forecasting — Model Comparison & Benchmarking Orchestrator
Module: src/model_comparison.py
Author: Senior ML Engineer & Data Scientist

Orchestrates training, recursive multi-step forecasting, evaluation, and SHAP analysis:
- Benchmarks LightGBM, XGBoost, and CatBoost against Phase 4 Champion (Seasonal Naive: $1,988.53).
- Generates official WMAE, MAE, RMSE, Holiday WMAE, Non-holiday WMAE, Store Type, and Event slices.
- Performs SHAP feature attribution to explain key drivers of model decisions.
- Exports comprehensive artifacts to outputs/ml/.
"""

import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.ml_models import (
    BaseMLModel,
    LGBMModel,
    XGBoostModel,
    CatBoostModel,
    HAS_LIGHTGBM,
    HAS_XGBOOST,
    HAS_CATBOOST,
    HAS_SHAP
)
if HAS_SHAP:
    import shap

from src.ml_training import prepare_training_data
from src.ml_forecasting import RecursiveMultiStepForecaster
from src.preprocessing import HierarchicalFallbackManager
from src.evaluation import compute_wmae, compute_mae, compute_rmse, evaluate_forecast

logger = logging.getLogger("WalmartModelComparison")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path("E:/MCA projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting")
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs" / "ml"
MODEL_DIR = OUTPUT_DIR / "models"
BASELINE_OUTPUT_DIR = BASE_DIR / "outputs" / "baseline"

PHASE_4_CHAMPION_WMAE = 1988.53
PHASE_4_CHAMPION_NAME = "Seasonal Naive (Lag 52)"


def compute_shap_analysis(model: BaseMLModel, X_sample: pd.DataFrame, clean_name: str):
    """Compute and save mean absolute SHAP values using TreeExplainer on a sample of observations."""
    if not HAS_SHAP:
        logger.info("SHAP is not available; skipping SHAP analysis.")
        return

    logger.info(f"Computing SHAP feature attribution for {model.name} on {len(X_sample):,} sample rows...")
    try:
        # Use underlying tree model
        raw_model = model.model
        explainer = shap.TreeExplainer(raw_model)
        shap_values = explainer.shap_values(X_sample)
        
        # If output is a list (e.g. multi-class or raw), handle appropriately
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        shap_df = pd.DataFrame({
            "Feature": model.feature_names,
            "Mean_Abs_SHAP": mean_abs_shap
        }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
        
        total_shap = shap_df["Mean_Abs_SHAP"].sum()
        shap_df["SHAP_Importance_Pct"] = (shap_df["Mean_Abs_SHAP"] / total_shap) * 100.0 if total_shap > 0 else 0.0

        shap_path = OUTPUT_DIR / f"shap_summary_{clean_name}.csv"
        shap_df.to_csv(shap_path, index=False)
        logger.info(f"Saved SHAP summary to: {shap_path}")
        print(f"\n--- Top 10 SHAP Features for {model.name} ---")
        print(shap_df.head(10).to_string(index=False))
    except Exception as e:
        logger.warning(f"SHAP explanation failed for {model.name}: {e}")


def run_ml_comparison():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("=== STARTING PHASE 5 MACHINE LEARNING COMPARISON PIPELINE ===")

    # 1. Load Data
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

    # 2. Fit Fallback Manager
    logger.info("Fitting HierarchicalFallbackManager on training data...")
    fallback_mgr = HierarchicalFallbackManager().fit(train_df)

    # 3. Prepare Training and Early-Stopping Eval Set
    X_train, y_train, w_train, X_eval, y_eval, w_eval, feature_cols = prepare_training_data(train_df)
    eval_set = (X_eval, y_eval, w_eval)

    # 4. Instantiate Models
    models: List[BaseMLModel] = []
    if HAS_LIGHTGBM:
        models.append(LGBMModel(n_estimators=400, learning_rate=0.07, num_leaves=63, random_state=42))
    if HAS_XGBOOST:
        models.append(XGBoostModel(n_estimators=350, learning_rate=0.07, max_depth=6, random_state=42))
    if HAS_CATBOOST:
        models.append(CatBoostModel(iterations=350, learning_rate=0.07, depth=6, random_seed=42))

    # Base dictionary for predictions export
    predictions_dict = {
        "Store": val_df["Store"].values,
        "Dept": val_df["Dept"].values,
        "Date": val_df["Date"].values,
        "Type": val_df["Type"].values,
        "IsHoliday": val_df["IsHoliday"].values,
        "Weekly_Sales": y_true
    }

    # Include Seasonal Naive prediction if baseline output exists
    baseline_preds_path = BASELINE_OUTPUT_DIR / "baseline_predictions.parquet"
    if baseline_preds_path.exists():
        base_df = pd.read_parquet(baseline_preds_path)
        if "Pred_Seasonal_Naive_Lag_52" in base_df.columns:
            predictions_dict["Pred_Seasonal_Naive_Lag_52"] = base_df["Pred_Seasonal_Naive_Lag_52"].values

    metrics_list = []
    store_type_records = []
    holiday_records = []

    # Insert Phase 4 Champion into leaderboard for direct comparison
    metrics_list.append({
        "Model": f"[Phase 4 Champion] {PHASE_4_CHAMPION_NAME}",
        "WMAE": 1988.53,
        "MAE": 1917.65,
        "RMSE": 4178.57,
        "WMAE_Holiday": 2216.66,
        "WMAE_NonHoliday": 1892.47,
        "WMAE_vs_Champion_Delta": 0.00,
        "WMAE_Improvement_Pct": 0.00,
        "Fit_Time_Sec": 0.86,
        "Forecast_Time_Sec": 3.00,
        "Status": "Benchmark Baseline"
    })

    # Sample for SHAP
    np.random.seed(42)
    sample_indices = np.random.choice(len(val_df), size=min(2000, len(val_df)), replace=False)
    X_sample_val = val_df.iloc[sample_indices][feature_cols]

    for model in models:
        logger.info(f"\n==================== EVALUATING: {model.name} ====================")
        clean_name = model.name.lower().replace(" ", "_")
        
        # A. Train Model
        t0_fit = time.time()
        model.fit(X_train, y_train, sample_weight=w_train, eval_set=eval_set)
        fit_time = time.time() - t0_fit

        # Save artifact
        model_save_path = MODEL_DIR / f"{clean_name}.pkl"
        with open(model_save_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Saved model to: {model_save_path}")

        # B. Recursive Multi-Step Forecast
        forecaster = RecursiveMultiStepForecaster(model, fallback_mgr)
        forecaster.initialize_history_from_train(train_df)
        
        t0_fore = time.time()
        preds = forecaster.forecast_validation_horizon(val_df)
        forecast_time = time.time() - t0_fore

        # Verify output
        assert len(preds) == total_val_rows, f"Prediction count mismatch: {len(preds)}"
        nan_count = np.isnan(preds).sum()
        assert nan_count == 0, f"Found {nan_count} NaNs in {model.name} predictions!"

        # Save to predictions dictionary
        col_pred_name = "Pred_" + model.name.replace(" ", "_")
        predictions_dict[col_pred_name] = preds

        # C. Evaluate Overall Performance
        m = evaluate_forecast(y_true, preds, is_holiday)
        wmae_val = round(m["WMAE"], 2)
        delta_wmae = round(wmae_val - PHASE_4_CHAMPION_WMAE, 2)
        pct_imp = round(((PHASE_4_CHAMPION_WMAE - wmae_val) / PHASE_4_CHAMPION_WMAE) * 100.0, 2)
        status_str = "BEATS CHAMPION" if delta_wmae < 0 else "Behind Baseline"

        metrics_list.append({
            "Model": model.name,
            "WMAE": wmae_val,
            "MAE": round(m["MAE"], 2),
            "RMSE": round(m["RMSE"], 2),
            "WMAE_Holiday": round(m["WMAE_Holiday"], 2),
            "WMAE_NonHoliday": round(m["WMAE_NonHoliday"], 2),
            "WMAE_vs_Champion_Delta": delta_wmae,
            "WMAE_Improvement_Pct": pct_imp,
            "Fit_Time_Sec": round(fit_time, 2),
            "Forecast_Time_Sec": round(forecast_time, 2),
            "Status": status_str
        })

        # D. Store Type Breakdown
        for stype in ["A", "B", "C"]:
            type_mask = (val_df["Type"] == stype).values
            t_wmae = compute_wmae(y_true[type_mask], preds[type_mask], is_holiday[type_mask])
            t_mae = compute_mae(y_true[type_mask], preds[type_mask])
            store_type_records.append({
                "Model": model.name,
                "Store_Type": stype,
                "Observations": int(np.sum(type_mask)),
                "WMAE": round(t_wmae, 2),
                "MAE": round(t_mae, 2)
            })

        # E. Holiday Breakdown
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

        # F. Feature Importances & SHAP
        try:
            fi_df = model.get_feature_importances()
            fi_path = OUTPUT_DIR / f"feature_importance_{clean_name}.csv"
            fi_df.to_csv(fi_path, index=False)
            logger.info(f"Saved feature importances for {model.name} to: {fi_path}")
        except Exception as e:
            logger.warning(f"Could not compute feature importances for {model.name}: {e}")

        compute_shap_analysis(model, X_sample_val, clean_name)

        logger.info(f"--> {model.name} Final Result: WMAE = ${wmae_val:,.2f} "
                    f"(Delta vs Seasonal Naive: ${delta_wmae:+,.2f} | {pct_imp:+.2f}%) [{status_str}]")

    # 5. Export Summary Artifacts
    logger.info("\n=== SAVING MACHINE LEARNING COMPARISON ARTIFACTS ===")
    
    # A. Predictions Parquet
    preds_df = pd.DataFrame(predictions_dict)
    preds_parquet_path = OUTPUT_DIR / "ml_predictions.parquet"
    preds_df.to_parquet(preds_parquet_path, index=False)
    logger.info(f"Saved validation predictions to: {preds_parquet_path} ({preds_df.shape})")

    # B. Overall Comparison Metrics
    metrics_df = pd.DataFrame(metrics_list).sort_values("WMAE").reset_index(drop=True)
    metrics_csv_path = OUTPUT_DIR / "ml_comparison_metrics.csv"
    metrics_json_path = OUTPUT_DIR / "ml_comparison_metrics.json"
    metrics_df.to_csv(metrics_csv_path, index=False)
    metrics_df.to_json(metrics_json_path, orient="records", indent=2)
    logger.info(f"Saved metrics comparison to: {metrics_csv_path}")

    # C. Store Type Breakdown
    type_df = pd.DataFrame(store_type_records)
    type_csv_path = OUTPUT_DIR / "ml_by_store_type.csv"
    type_df.to_csv(type_csv_path, index=False)
    logger.info(f"Saved store type breakdown to: {type_csv_path}")

    # D. Holiday Breakdown
    hol_df = pd.DataFrame(holiday_records)
    hol_csv_path = OUTPUT_DIR / "ml_holiday_breakdown.csv"
    hol_df.to_csv(hol_csv_path, index=False)
    logger.info(f"Saved holiday breakdown to: {hol_csv_path}")

    # 6. Display Final Benchmark Leaderboard
    print("\n" + "="*115)
    print("WALMART SALES FORECASTING — PHASE 5 MACHINE LEARNING BENCHMARK LEADERBOARD")
    print("="*115)
    print(metrics_df.to_string(index=False))
    print("="*115)
    
    best_entry = metrics_df.iloc[0]
    print(f"\nOVERALL CHAMPION: {best_entry['Model']}")
    print(f"Official WMAE: ${best_entry['WMAE']:,.2f} | MAE: ${best_entry['MAE']:,.2f} | RMSE: ${best_entry['RMSE']:,.2f}")
    print(f"Holiday WMAE: ${best_entry['WMAE_Holiday']:,.2f} | Non-Holiday WMAE: ${best_entry['WMAE_NonHoliday']:,.2f}")
    print(f"Performance vs Seasonal Naive Baseline: {best_entry['WMAE_Improvement_Pct']:+.2f}%")
    print("="*115)

    return metrics_df, preds_df


if __name__ == "__main__":
    run_ml_comparison()
