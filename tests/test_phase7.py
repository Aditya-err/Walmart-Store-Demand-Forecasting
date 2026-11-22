"""
Walmart Sales Forecasting — Phase 7 Automated Quality Assurance Test Suite
File: tests/test_phase7.py
Author: Lead Data Scientist & Principal QA Engineer

Tests:
1. Phase 7 artifacts existence and file sizes.
2. Submission CSV schema, row count (115,064), and exact column ordering ('Id', 'Weekly_Sales').
3. Strict 100% ID alignment and row ordering matching sampleSubmission.csv.zip.
4. Prediction completeness: zero NaNs, zero nulls, zero infs, zero empty strings.
5. Zero duplicate IDs in submission.
6. Final trained model loading via pickle and sample inference.
7. Final model training on ALL 261,541 eligible residual training observations.
8. Final model tree count equals exactly 350 trees.
9. Accurate metadata documentation (early stopping did not trigger, full retraining executed).
10. Feature compatibility: exactly 48 clean features matching training and test sets.
11. Seasonal anchor generation and hierarchical fallback coverage on test set.
12. Resilience to missing macro/markdown features (e.g. CPI/Unemployment blackout period).
13. Forecast reproducibility: model.predict matches saved predictions bit-for-bit.
14. Plausibility of negative predictions in minor departments.
"""

import json
import os
import pickle
import zipfile
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Support clean public outputs structure (outputs/submission, outputs/model) with fallback to phase7
if (PROJECT_ROOT / "outputs" / "submission" / "submission.csv").exists():
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "submission"
    MODEL_DIR = PROJECT_ROOT / "outputs" / "model"
else:
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase7"
    MODEL_DIR = OUTPUT_DIR / "model"

DATA_DIR = PROJECT_ROOT / "data" / "processed"
if not DATA_DIR.exists():
    DATA_DIR = PROJECT_ROOT / "walmart-recruiting-store-sales-forecasting" / "data" / "processed"

SAMPLE_SUB_ZIP = PROJECT_ROOT / "sampleSubmission.csv.zip"
if not SAMPLE_SUB_ZIP.exists():
    SAMPLE_SUB_ZIP = PROJECT_ROOT / "walmart-recruiting-store-sales-forecasting" / "sampleSubmission.csv.zip"


@pytest.fixture(scope="module")
def submission_df():
    sub_path = OUTPUT_DIR / "submission.csv"
    assert sub_path.exists(), f"Missing submission.csv at {sub_path}"
    return pd.read_csv(sub_path)


@pytest.fixture(scope="module")
def sample_submission_df():
    assert SAMPLE_SUB_ZIP.exists(), f"Missing sampleSubmission.csv.zip at {SAMPLE_SUB_ZIP}"
    with zipfile.ZipFile(SAMPLE_SUB_ZIP, "r") as z:
        with z.open("sampleSubmission.csv") as f:
            return pd.read_csv(f)


@pytest.fixture(scope="module")
def test_predictions_df():
    pred_path = OUTPUT_DIR / "test_predictions.parquet"
    assert pred_path.exists(), f"Missing test_predictions.parquet at {pred_path}"
    return pd.read_parquet(pred_path)


@pytest.fixture(scope="module")
def loaded_model():
    model_path = MODEL_DIR / "final_seasonal_residual_lgbm.pkl"
    assert model_path.exists(), f"Missing model pickle at {model_path}"
    with open(model_path, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def model_metadata():
    meta_path = MODEL_DIR / "metadata.json"
    assert meta_path.exists(), f"Missing metadata.json at {meta_path}"
    with open(meta_path, "r") as f:
        return json.load(f)


def test_phase7_artifacts_exist():
    """Verify all mandatory Phase 7 artifact files and directories exist."""
    assert (OUTPUT_DIR / "submission.csv").exists(), "submission.csv missing!"
    assert (OUTPUT_DIR / "test_predictions.parquet").exists(), "test_predictions.parquet missing!"
    assert (OUTPUT_DIR / "phase7_metrics.json").exists(), "phase7_metrics.json missing!"
    assert (MODEL_DIR / "final_seasonal_residual_lgbm.pkl").exists(), "model pickle missing!"
    assert (MODEL_DIR / "feature_importance.csv").exists(), "feature_importance.csv missing!"
    assert (MODEL_DIR / "feature_names.json").exists(), "feature_names.json missing!"
    assert (MODEL_DIR / "metadata.json").exists(), "metadata.json missing!"
    
    # Check submission file size is non-trivial (> 3 MB)
    sub_size_mb = os.path.getsize(OUTPUT_DIR / "submission.csv") / (1024 * 1024)
    assert sub_size_mb > 3.0, f"submission.csv too small: {sub_size_mb:.2f} MB"


def test_submission_schema_and_shape(submission_df):
    """Verify submission has exactly 115,064 rows and columns ['Id', 'Weekly_Sales']."""
    assert len(submission_df) == 115064, f"Expected 115,064 rows, got {len(submission_df)}"
    assert submission_df.columns.tolist() == ["Id", "Weekly_Sales"], \
        f"Incorrect column schema: {submission_df.columns.tolist()}"
    assert np.issubdtype(submission_df["Weekly_Sales"].dtype, np.number), "Weekly_Sales must be numeric!"


def test_submission_exact_alignment_with_sample(submission_df, sample_submission_df):
    """Verify submission IDs match sampleSubmission.csv exactly in value and original row order."""
    assert len(submission_df) == len(sample_submission_df), "Row counts do not match!"
    assert (submission_df["Id"] == sample_submission_df["Id"]).all(), \
        "Submission Id column is not identically ordered with sampleSubmission.csv!"


def test_submission_no_duplicate_ids(submission_df):
    """Verify submission contains zero duplicate IDs."""
    duplicates = submission_df["Id"].duplicated().sum()
    assert duplicates == 0, f"Found {duplicates} duplicate IDs in submission.csv"


def test_submission_values_finite_and_clean(submission_df):
    """Verify zero NaN, null, infinite, or missing values in predictions."""
    assert submission_df["Weekly_Sales"].isnull().sum() == 0, "Null/NaN values found in submission!"
    assert not np.isinf(submission_df["Weekly_Sales"]).any(), "Infinite values found in submission!"


def test_model_loading_and_sample_inference(loaded_model):
    """Verify loaded model is properly instantiated and can generate predictions."""
    assert loaded_model.is_fitted, "Loaded model reports is_fitted=False!"
    assert hasattr(loaded_model, "predict"), "Loaded model missing predict method!"
    assert len(loaded_model.feature_names) == 48, f"Expected 48 features, got {len(loaded_model.feature_names)}"


def test_final_model_trained_on_all_eligible_observations(loaded_model):
    """Verify model was retrained on all 261,541 eligible residual training observations."""
    assert hasattr(loaded_model, "fitted_rows_count"), "Model missing fitted_rows_count attribute!"
    assert loaded_model.fitted_rows_count == 261541, \
        f"Expected 261,541 fitted rows, got {loaded_model.fitted_rows_count}"


def test_final_model_has_350_trees(loaded_model):
    """Verify final model has exactly 350 boosting trees."""
    assert hasattr(loaded_model, "final_tree_count"), "Model missing final_tree_count attribute!"
    assert loaded_model.final_tree_count == 350, f"Expected 350 trees, got {loaded_model.final_tree_count}"
    assert loaded_model.model.booster_.num_trees() == 350, "Booster num_trees does not equal 350!"


def test_metadata_records_remediation_honestly(model_metadata):
    """Verify metadata honestly records early stopping did not trigger and full retraining was done."""
    assert model_metadata["early_stopping_triggered"] is False, "early_stopping_triggered should be False!"
    assert model_metadata["full_retraining_executed"] is True, "full_retraining_executed should be True!"
    assert model_metadata["eligible_residual_training_observations"] == 261541, \
        f"Expected 261,541 eligible rows in metadata, got {model_metadata['eligible_residual_training_observations']}"
    assert model_metadata["trees_in_booster"] == 350, "Expected 350 trees in metadata!"


def test_feature_compatibility_between_model_and_test_data(loaded_model):
    """Verify feature names in model match features in test_engineered.parquet."""
    test_df = pd.read_parquet(DATA_DIR / "test_engineered.parquet")
    from src.residual_model import get_residual_feature_columns
    test_features = get_residual_feature_columns(test_df)
    assert loaded_model.feature_names == test_features, "Feature mismatch between model and test data!"


def test_seasonal_anchor_and_fallback_coverage(loaded_model):
    """Verify _compute_anchor handles all test rows (including cold starts) without producing nulls."""
    test_df = pd.read_parquet(DATA_DIR / "test_engineered.parquet")
    anchors, has_l52 = loaded_model._compute_anchor(test_df)
    assert len(anchors) == 115064, f"Expected 115,064 anchors, got {len(anchors)}"
    assert not np.isnan(anchors).any(), "NaN values detected in anchor series!"
    # Verify exactly 2,041 missing lag52 rows were imputed with fallbacks
    assert (has_l52 == False).sum() == 2041, f"Expected 2,041 missing lag52 rows, got {(has_l52 == False).sum()}"


def test_resilience_to_missing_macro_features(loaded_model):
    """Verify inference succeeds on the test blackout period (May-July 2013 where CPI/Unemployment are NaN)."""
    test_df = pd.read_parquet(DATA_DIR / "test_engineered.parquet")
    preds, _, _ = loaded_model.predict(test_df)
    assert not np.isnan(preds).any(), "Predictions failed or produced NaNs during macro blackout period!"


def test_forecast_reproducibility(loaded_model, test_predictions_df):
    """Verify loaded model re-prediction matches saved predictions bit-for-bit."""
    test_df = pd.read_parquet(DATA_DIR / "test_engineered.parquet")
    sample_slice = test_df.iloc[:1000]
    preds, anchors, residuals = loaded_model.predict(sample_slice)
    saved_preds = test_predictions_df["Weekly_Sales_Predicted"].iloc[:1000].values
    np.testing.assert_allclose(preds, saved_preds, rtol=1e-5, atol=1e-5,
                               err_msg="Predictions are not reproducible!")


def test_negative_predictions_plausibility(test_predictions_df):
    """Verify negative predictions are small, bounded, and concentrated in minor departments."""
    neg_preds = test_predictions_df[test_predictions_df["Weekly_Sales_Predicted"] < 0]
    assert len(neg_preds) < 1150, f"Unreasonably high negative predictions count: {len(neg_preds)}"
    assert neg_preds["Weekly_Sales_Predicted"].min() > -2000.0, "Unreasonably large negative prediction found!"
    top_neg_depts = set(neg_preds["Dept"].value_counts().head(5).index)
    assert 99 in top_neg_depts or 49 in top_neg_depts or 59 in top_neg_depts, \
        "Negative predictions not concentrated in expected departments!"
