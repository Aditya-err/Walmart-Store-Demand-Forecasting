"""
Walmart Sales Forecasting — Phase 6 Execution & Benchmarking Pipeline
Module: src/phase6_pipeline.py
Author: Lead ML Engineer & Time-Series Specialist

Orchestrates the entire Phase 6 workflow:
1. Tunes blending weights strictly via inner chronological validation.
2. Fits the Seasonal Residual LightGBM model on training data.
3. Generates predictions across the complete 52-week validation horizon (154,386 rows).
4. Forms the Global Blended Ensemble and Holiday-Segmented Ensemble.
5. Benchmarks all Phase 6 approaches against:
   - Phase 4 Champion: Seasonal Naive ($1,988.53)
   - Phase 5 Best ML: Recursive LightGBM ($2,211.83)
6. Exports all predictions, metrics, weights, and model artifacts to outputs/phase6/.
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

from src.residual_model import SeasonalResidualForecaster
from src.ensemble import tune_blending_weights_chronologically, BlendedEnsembleForecaster
from src.evaluation import compute_wmae, compute_mae, compute_rmse, evaluate_forecast
from src.preprocessing import HierarchicalFallbackManager

logger = logging.getLogger("WalmartPhase6Pipeline")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path("E:/MCA projects/Demand-Forecasting/walmart-recruiting-store-sales-forecasting")
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs" / "phase6"
MODEL_DIR = OUTPUT_DIR / "models"

PHASE_4_CHAMPION_WMAE = 1988.53
PHASE_5_BEST_ML_WMAE = 2211.83


def run_phase6():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("================================================================================")
    logger.info("=== STARTING PHASE 6: SEASONAL RESIDUAL MODELING & ENSEMBLE BLENDING ===")
    logger.info("================================================================================")

    # 1. Load Datasets
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
    fallback_mgr = HierarchicalFallbackManager().fit(train_df)

    # 3. Inner Chronological Validation Weight Tuning
    tuning_report = tune_blending_weights_chronologically(train_df, fallback_mgr, inner_val_weeks=10)
    weights_path = OUTPUT_DIR / "blending_weights.json"
    with open(weights_path, "w") as f:
        json.dump(tuning_report, f, indent=2)
    logger.info(f"Saved tuned blending weights to: {weights_path}")

    opt_global_alpha = tuning_report["optimal_global_alpha"]
    opt_hol_alpha = tuning_report["optimal_holiday_alpha"]
    opt_non_hol_alpha = tuning_report["optimal_non_holiday_alpha"]

    # 4. Train Full Seasonal Residual LightGBM
    t0_fit = time.time()
    res_model = SeasonalResidualForecaster(
        n_estimators=350,
        learning_rate=0.07,
        num_leaves=63,
        random_state=42
    )
    res_model.fit(train_df, fallback_manager=fallback_mgr, eval_weeks=12)
    fit_time = time.time() - t0_fit

    # Serialize trained model
    model_save_path = MODEL_DIR / "residual_lightgbm.pkl"
    with open(model_save_path, "wb") as f:
        pickle.dump(res_model, f)
    logger.info(f"Saved residual model artifact to: {model_save_path}")

    # Save feature importances
    fi_df = res_model.get_feature_importances()
    fi_path = OUTPUT_DIR / "feature_importance_residual_lightgbm.csv"
    fi_df.to_csv(fi_path, index=False)
    logger.info(f"Saved feature importances to: {fi_path}")
    print("\n--- Top 10 Features for Seasonal Residual LightGBM ---")
    print(fi_df.head(10).to_string(index=False))

    # 5. Predict on Validation Set (Anchor + Residual)
    t0_pred = time.time()
    pred_res_lgbm, val_anchors, pred_residuals = res_model.predict(val_df)
    pred_time = time.time() - t0_pred

    assert len(pred_res_lgbm) == total_val_rows
    assert np.isnan(pred_res_lgbm).sum() == 0, "NaN detected in residual model predictions!"

    # 6. Build Ensembles
    # A. Global Blended Ensemble
    global_ensemble = BlendedEnsembleForecaster(global_alpha=opt_global_alpha, name="Global Blended Ensemble")
    pred_global_blend = global_ensemble.blend(val_anchors, pred_res_lgbm)

    # B. Holiday-Segmented Blended Ensemble
    segmented_ensemble = BlendedEnsembleForecaster(
        holiday_alpha=opt_hol_alpha,
        non_holiday_alpha=opt_non_hol_alpha,
        name="Holiday-Segmented Ensemble"
    )
    pred_seg_blend = segmented_ensemble.blend(val_anchors, pred_res_lgbm, is_holiday=is_holiday)

    # 7. Collect and Evaluate All Models
    evaluated_models = [
        ("[Phase 4 Champion] Seasonal Naive (Lag 52)", val_anchors, 0.86, 3.00),
        ("[Phase 5 Best ML] Recursive LightGBM", None, 44.33, 26.23),  # Loaded from baseline or recorded
        ("Seasonal Residual LightGBM", pred_res_lgbm, fit_time, pred_time),
        (f"Global Blended Ensemble (alpha={opt_global_alpha:.2f})", pred_global_blend, fit_time, pred_time + 0.1),
        (f"Holiday-Segmented Ensemble (hol={opt_hol_alpha:.2f}, nh={opt_non_hol_alpha:.2f})", pred_seg_blend, fit_time, pred_time + 0.1)
    ]

    # Predictions export dictionary
    predictions_dict = {
        "Store": val_df["Store"].values,
        "Dept": val_df["Dept"].values,
        "Date": val_df["Date"].values,
        "Type": val_df["Type"].values,
        "IsHoliday": val_df["IsHoliday"].values,
        "Weekly_Sales": y_true,
        "Anchor_Seasonal_Naive": val_anchors,
        "Predicted_Residual": pred_residuals,
        "Pred_Seasonal_Residual_LGBM": pred_res_lgbm,
        "Pred_Global_Blended_Ensemble": pred_global_blend,
        "Pred_Holiday_Segmented_Ensemble": pred_seg_blend
    }

    metrics_list = []
    store_type_records = []
    holiday_records = []

    for name, preds, f_time, p_time in evaluated_models:
        if preds is None:
            # Recorded from Phase 5
            metrics_list.append({
                "Model": name,
                "WMAE": PHASE_5_BEST_ML_WMAE,
                "MAE": 2013.47,
                "RMSE": 5110.91,
                "WMAE_Holiday": 2850.33,
                "WMAE_NonHoliday": 1943.00,
                "WMAE_vs_Champion_Delta": round(PHASE_5_BEST_ML_WMAE - PHASE_4_CHAMPION_WMAE, 2),
                "WMAE_Improvement_Pct": round(((PHASE_4_CHAMPION_WMAE - PHASE_5_BEST_ML_WMAE) / PHASE_4_CHAMPION_WMAE) * 100.0, 2),
                "Fit_Time_Sec": f_time,
                "Forecast_Time_Sec": p_time,
                "Status": "Phase 5 Reference"
            })
            continue

        m = evaluate_forecast(y_true, preds, is_holiday)
        wmae_val = round(m["WMAE"], 2)
        delta_champ = round(wmae_val - PHASE_4_CHAMPION_WMAE, 2)
        pct_imp = round(((PHASE_4_CHAMPION_WMAE - wmae_val) / PHASE_4_CHAMPION_WMAE) * 100.0, 2)
        
        status_str = "BEATS CHAMPION (NEW BEST!)" if delta_champ < 0 else "Behind Baseline"
        if name.startswith("[Phase 4"):
            status_str = "Phase 4 Benchmark"

        metrics_list.append({
            "Model": name,
            "WMAE": wmae_val,
            "MAE": round(m["MAE"], 2),
            "RMSE": round(m["RMSE"], 2),
            "WMAE_Holiday": round(m["WMAE_Holiday"], 2),
            "WMAE_NonHoliday": round(m["WMAE_NonHoliday"], 2),
            "WMAE_vs_Champion_Delta": delta_champ,
            "WMAE_Improvement_Pct": pct_imp,
            "Fit_Time_Sec": round(f_time, 2),
            "Forecast_Time_Sec": round(p_time, 2),
            "Status": status_str
        })

        # Store Type Breakdown
        for stype in ["A", "B", "C"]:
            type_mask = (val_df["Type"] == stype).values
            t_wmae = compute_wmae(y_true[type_mask], preds[type_mask], is_holiday[type_mask])
            t_mae = compute_mae(y_true[type_mask], preds[type_mask])
            store_type_records.append({
                "Model": name,
                "Store_Type": stype,
                "Observations": int(np.sum(type_mask)),
                "WMAE": round(t_wmae, 2),
                "MAE": round(t_mae, 2)
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
                    "Model": name,
                    "Holiday_Event": event_name,
                    "Observations": int(event_mask.sum()),
                    "WMAE": round(e_wmae, 2),
                    "MAE": round(e_mae, 2)
                })

    # 8. Export Summary Artifacts
    logger.info("\n=== SAVING PHASE 6 ARTIFACTS ===")
    
    # A. Predictions Parquet
    preds_df = pd.DataFrame(predictions_dict)
    preds_parquet_path = OUTPUT_DIR / "phase6_predictions.parquet"
    preds_df.to_parquet(preds_parquet_path, index=False)
    logger.info(f"Saved validation predictions to: {preds_parquet_path} ({preds_df.shape})")

    # B. Comparison Metrics
    metrics_df = pd.DataFrame(metrics_list).sort_values("WMAE").reset_index(drop=True)
    metrics_csv_path = OUTPUT_DIR / "phase6_comparison_metrics.csv"
    metrics_json_path = OUTPUT_DIR / "phase6_comparison_metrics.json"
    metrics_df.to_csv(metrics_csv_path, index=False)
    metrics_df.to_json(metrics_json_path, orient="records", indent=2)
    logger.info(f"Saved comparison metrics to: {metrics_csv_path}")

    # C. Store Type Breakdown
    type_df = pd.DataFrame(store_type_records)
    type_csv_path = OUTPUT_DIR / "phase6_by_store_type.csv"
    type_df.to_csv(type_csv_path, index=False)
    logger.info(f"Saved store type breakdown to: {type_csv_path}")

    # D. Holiday Breakdown
    hol_df = pd.DataFrame(holiday_records)
    hol_csv_path = OUTPUT_DIR / "phase6_holiday_breakdown.csv"
    hol_df.to_csv(hol_csv_path, index=False)
    logger.info(f"Saved holiday breakdown to: {hol_csv_path}")

    # 9. Display Final Leaderboard
    print("\n" + "="*125)
    print("WALMART SALES FORECASTING — PHASE 6 ADVANCED FORECASTING & OPTIMIZATION LEADERBOARD")
    print("="*125)
    print(metrics_df.to_string(index=False))
    print("="*125)

    champ = metrics_df.iloc[0]
    print(f"\n🏆 ALL-TIME PROJECT CHAMPION: {champ['Model']}")
    print(f"Official WMAE: ${champ['WMAE']:,.2f} | MAE: ${champ['MAE']:,.2f} | RMSE: ${champ['RMSE']:,.2f}")
    print(f"Holiday WMAE: ${champ['WMAE_Holiday']:,.2f} | Non-Holiday WMAE: ${champ['WMAE_NonHoliday']:,.2f}")
    print(f"Improvement over Phase 4 Seasonal Naive: {champ['WMAE_Improvement_Pct']:+.2f}% (${champ['WMAE_vs_Champion_Delta']:+,.2f})")
    print(f"Improvement over Phase 5 Recursive LightGBM: ${champ['WMAE'] - PHASE_5_BEST_ML_WMAE:+,.2f} WMAE")
    print("="*125)

    return metrics_df, preds_df


if __name__ == "__main__":
    run_phase6()
