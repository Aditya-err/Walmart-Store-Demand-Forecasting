"""
Walmart Sales Forecasting — Machine Learning Training Pipeline
Module: src/ml_training.py
Author: Senior ML Engineer & Data Scientist

Trains global gradient boosted decision tree regressors (LightGBM, XGBoost, CatBoost):
- Uses official WMAE sample weights (Holiday = 5.0, Non-holiday = 1.0).
- Applies internal chronological validation split for early stopping without lookahead bias.
- Serializes trained model artifacts to outputs/ml/models/.
- Extracts and saves feature importance tables.
"""

import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
    get_feature_columns,
    HAS_LIGHTGBM,
    HAS_XGBOOST,
    HAS_CATBOOST
)

logger = logging.getLogger("WalmartMLTraining")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path("E:/MCA projects\Demand-Forecasting/walmart-recruiting-store-sales-forecasting")
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs" / "ml"
MODEL_DIR = OUTPUT_DIR / "models"


def prepare_training_data(train_df: pd.DataFrame,
                          val_fraction_weeks: int = 16) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray,
                                                                 pd.DataFrame, np.ndarray, np.ndarray,
                                                                 List[str]]:
    """
    Extract feature matrices and sample weights with an internal chronological split.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Historical training dataset (267,184 rows, 2010-02-05 to 2011-10-28)
    val_fraction_weeks : int
        Number of most recent training weeks to allocate to internal early-stopping validation (default: 16)
        
    Returns:
    --------
    X_train, y_train, w_train, X_eval, y_eval, w_eval, feature_cols
    """
    feature_cols = get_feature_columns(train_df)
    logger.info(f"Using {len(feature_cols)} predictor features for ML training.")

    # Chronological sort
    df_sorted = train_df.sort_values(["Date", "Store", "Dept"]).reset_index(drop=True)
    all_dates = sorted(df_sorted["Date"].unique())
    split_date = all_dates[-val_fraction_weeks]
    
    train_mask = df_sorted["Date"] < split_date
    eval_mask = df_sorted["Date"] >= split_date
    
    logger.info(f"Internal split: Train ({train_mask.sum():,} rows up to {all_dates[-val_fraction_weeks-1].strftime('%Y-%m-%d')}) "
                f"| Early-Stopping Eval ({eval_mask.sum():,} rows from {split_date.strftime('%Y-%m-%d')} to {all_dates[-1].strftime('%Y-%m-%d')})")

    # Sample weights: 5.0 for holiday, 1.0 for non-holiday
    weights = np.where(df_sorted["IsHoliday"] == 1, 5.0, 1.0)
    
    X_train = df_sorted.loc[train_mask, feature_cols]
    y_train = df_sorted.loc[train_mask, "Weekly_Sales"].values
    w_train = weights[train_mask]
    
    X_eval = df_sorted.loc[eval_mask, feature_cols]
    y_eval = df_sorted.loc[eval_mask, "Weekly_Sales"].values
    w_eval = weights[eval_mask]
    
    return X_train, y_train, w_train, X_eval, y_eval, w_eval, feature_cols


def train_all_models(train_df: pd.DataFrame) -> Dict[str, BaseMLModel]:
    """Train LightGBM, XGBoost, and CatBoost models and serialize them."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    X_train, y_train, w_train, X_eval, y_eval, w_eval, feature_cols = prepare_training_data(train_df)
    eval_set = (X_eval, y_eval, w_eval)

    models_to_train: List[BaseMLModel] = []
    
    if HAS_LIGHTGBM:
        models_to_train.append(LGBMModel(
            n_estimators=400,
            learning_rate=0.07,
            num_leaves=63,
            random_state=42
        ))
    else:
        logger.warning("LightGBM not installed; skipping LightGBM training.")

    if HAS_XGBOOST:
        models_to_train.append(XGBoostModel(
            n_estimators=350,
            learning_rate=0.07,
            max_depth=6,
            random_state=42
        ))
    else:
        logger.warning("XGBoost not installed; skipping XGBoost training.")

    if HAS_CATBOOST:
        models_to_train.append(CatBoostModel(
            iterations=350,
            learning_rate=0.07,
            depth=6,
            random_seed=42
        ))
    else:
        logger.warning("CatBoost not installed; skipping CatBoost training.")

    fitted_models: Dict[str, BaseMLModel] = {}

    for model in models_to_train:
        logger.info(f"\n==================== TRAINING: {model.name} ====================")
        t0 = time.time()
        model.fit(X_train, y_train, sample_weight=w_train, eval_set=eval_set)
        elapsed = time.time() - t0
        logger.info(f"{model.name} successfully trained in {elapsed:.2f} seconds.")

        # Save model artifact
        clean_name = model.name.lower().replace(" ", "_")
        model_save_path = MODEL_DIR / f"{clean_name}.pkl"
        with open(model_save_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Saved model artifact to: {model_save_path}")

        # Save feature importances
        try:
            fi_df = model.get_feature_importances()
            fi_path = OUTPUT_DIR / f"feature_importance_{clean_name}.csv"
            fi_df.to_csv(fi_path, index=False)
            logger.info(f"Saved top feature importances to: {fi_path}")
            print(f"\n--- Top 10 Features for {model.name} ---")
            print(fi_df.head(10).to_string(index=False))
        except Exception as e:
            logger.warning(f"Could not extract feature importances for {model.name}: {e}")

        fitted_models[model.name] = model

    return fitted_models


if __name__ == "__main__":
    train_path = DATA_DIR / "train_engineered.parquet"
    logger.info(f"Loading training data: {train_path}")
    train_df = pd.read_parquet(train_path)
    train_all_models(train_df)
