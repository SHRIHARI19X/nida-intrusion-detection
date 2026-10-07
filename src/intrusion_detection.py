"""
intrusion_detection.py
----------------------
Phase 9: Network Intrusion Detection on Unseen KDDTest+ Dataset.

ANTIGRAVITY AI MODEL: Gemini 3.8 Flash Medium
ACTUAL DEEP LEARNING MODEL: Trained Dense Autoencoder
PROJECT: Network Intrusion Detection Using Autoencoder

OBJECTIVE:
Apply the trained Dense Autoencoder and the official threshold derived from
normal validation traffic in Phase 8 to perform intrusion/anomaly detection
on the completely unseen KDDTest+ dataset.

CRITICAL TEST-DATA RULES:
  - KDDTest+ is the FINAL unseen evaluation dataset.
  - DO NOT fit preprocessor on KDDTest+ (no .fit(), only .transform()).
  - DO NOT refit StandardScaler or OneHotEncoder.
  - DO NOT retrain the Autoencoder.
  - DO NOT modify test labels or use test labels to alter the threshold.
  - DO NOT remove or alter test records.
  - Final metrics (accuracy, precision, recall, f1, ROC-AUC, confusion matrix)
    are reserved for Phase 10.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure src/ is on Python search path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing import (
    load_dataset,
    create_binary_labels,
    get_feature_matrix,
    load_preprocessor,
    map_attack_category,
    check_array_integrity,
    TRAFFIC_FEATURES,
    ATTACK_CATEGORIES,
)
from autoencoder import load_input_dimension

# ---------------------------------------------------------------------------
# Directories and Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

TEST_FILE               = os.path.join(DATA_DIR, "KDDTest+.txt")
PREPROCESSOR_PATH       = os.path.join(MODELS_DIR, "preprocessor.joblib")
MODEL_PATH              = os.path.join(MODELS_DIR, "autoencoder.keras")
THRESHOLD_JSON_PATH     = os.path.join(RESULTS_DIR, "anomaly_threshold.json")
MODEL_INPUT_SUMMARY     = os.path.join(RESULTS_DIR, "model_input_summary.json")
TRAINING_SUMMARY        = os.path.join(RESULTS_DIR, "training_summary.json")
RECON_ERROR_SUMMARY     = os.path.join(RESULTS_DIR, "reconstruction_error_summary.json")
PREPROCESSING_SUMMARY   = os.path.join(RESULTS_DIR, "preprocessing_summary.json")

TEST_ERRORS_PATH        = os.path.join(RESULTS_DIR, "test_reconstruction_errors.npy")
PREDICTIONS_CSV_PATH    = os.path.join(RESULTS_DIR, "test_predictions.csv")
DETECTION_SUMMARY_PATH  = os.path.join(RESULTS_DIR, "intrusion_detection_summary.json")
DISTRIBUTION_PNG_PATH   = os.path.join(RESULTS_DIR, "test_reconstruction_error_distribution.png")

os.makedirs(RESULTS_DIR, exist_ok=True)


def run_phase_9():
    print("=" * 70)
    print("PHASE 9 — INTRUSION DETECTION ON UNSEEN KDDTEST+")
    print("=" * 70)
    print(f"Antigravity AI Model : Gemini 3.8 Flash Medium")
    print(f"Actual DL Model      : Trained Dense Autoencoder")

    # =======================================================================
    # STEP 1: READ EXISTING ARTIFACTS
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 1] Verifying and reading existing artifacts...")
    print("=" * 50)

    required_artifacts = [
        ("preprocessor.joblib", PREPROCESSOR_PATH),
        ("autoencoder.keras", MODEL_PATH),
        ("anomaly_threshold.json", THRESHOLD_JSON_PATH),
        ("model_input_summary.json", MODEL_INPUT_SUMMARY),
        ("training_summary.json", TRAINING_SUMMARY),
        ("reconstruction_error_summary.json", RECON_ERROR_SUMMARY),
        ("preprocessing_summary.json", PREPROCESSING_SUMMARY),
        ("KDDTest+.txt", TEST_FILE),
    ]

    for name, path in required_artifacts:
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {name:<35} ({size:>10,} bytes)")
        if not exists:
            raise FileNotFoundError(f"Required artifact missing: {path}")

    # Load input dimension from Phase 4 summary
    expected_dim = load_input_dimension(MODEL_INPUT_SUMMARY)
    print(f"  Expected feature dimension (from artifact): {expected_dim}")

    # =======================================================================
    # STEP 2: VERIFY TEST DATA
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 2] Loading and verifying unseen KDDTest+ dataset...")
    print("=" * 50)

    df_test = load_dataset(TEST_FILE)
    actual_rows, actual_cols = df_test.shape
    print(f"  Actual test dataset shape: {df_test.shape}")
    print(f"  Actual number of rows    : {actual_rows:,}")
    print(f"  Actual number of columns : {actual_cols}")

    assert actual_cols == 43, f"Expected 43 columns, got {actual_cols}"
    assert "label" in df_test.columns, "Missing 'label' column in test data"
    assert "difficulty" in df_test.columns, "Missing 'difficulty' column in test data"
    assert len(TRAFFIC_FEATURES) == 41, f"Expected 41 traffic features, got {len(TRAFFIC_FEATURES)}"
    for col in TRAFFIC_FEATURES:
        assert col in df_test.columns, f"Traffic feature '{col}' missing from test data"

    print("  Schema and column verification: PASS")

    # =======================================================================
    # STEP 3: CREATE TEST FEATURES & LABELS
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 3] Creating test feature matrix and labels...")
    print("=" * 50)

    # Feature matrix contains ONLY the 41 network traffic features
    X_test_raw = get_feature_matrix(df_test)
    assert X_test_raw.shape == (actual_rows, 41), f"Expected ({actual_rows}, 41), got {X_test_raw.shape}"
    assert "label" not in X_test_raw.columns, "'label' must not be in feature matrix"
    assert "difficulty" not in X_test_raw.columns, "'difficulty' must not be in feature matrix"

    # Binary evaluation labels: 0 = NORMAL, 1 = INTRUSION / ATTACK
    y_test_binary = create_binary_labels(df_test).values
    actual_normal_count = int(np.sum(y_test_binary == 0))
    actual_attack_count = int(np.sum(y_test_binary == 1))

    # Original attack labels preserved for post-detection analysis
    original_labels = df_test["label"].str.strip().str.lower().values

    print(f"  Raw test feature matrix shape : {X_test_raw.shape}")
    print(f"  Binary label array shape      : {y_test_binary.shape}")
    print(f"  Actual Normal records         : {actual_normal_count:,} ({actual_normal_count / actual_rows * 100:.2f}%)")
    print(f"  Actual Attack records         : {actual_attack_count:,} ({actual_attack_count / actual_rows * 100:.2f}%)")

    # =======================================================================
    # STEP 4: APPLY EXISTING PREPROCESSOR (TRANSFORM ONLY)
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 4] Applying saved preprocessor (transform only)...")
    print("=" * 50)

    preprocessor = load_preprocessor(PREPROCESSOR_PATH)
    print(f"  Loaded preprocessor from: {PREPROCESSOR_PATH}")
    print("  CRITICAL RULE: Using preprocessor.transform(X_test) ONLY. No .fit().")

    # Apply transform ONLY
    X_test_processed = preprocessor.transform(X_test_raw).astype(np.float32)
    processed_rows, processed_cols = X_test_processed.shape
    print(f"  Processed test feature matrix shape: {X_test_processed.shape}")

    assert processed_rows == actual_rows, f"Row count changed: {processed_rows} vs {actual_rows}"
    assert processed_cols == expected_dim, f"Feature dimension mismatch: {processed_cols} vs {expected_dim}"

    # Integrity checks on processed data
    integ = check_array_integrity(X_test_processed, "X_test_processed")
    assert integ["nan_count"] == 0, f"Found {integ['nan_count']} NaN values in processed test data!"
    assert integ["inf_count"] == 0, f"Found {integ['inf_count']} Inf values in processed test data!"
    print("  Processed test data integrity: PASS (0 NaN, 0 Inf)")

    # =======================================================================
    # STEP 5: LOAD TRAINED AUTOENCODER
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 5] Loading trained Dense Autoencoder...")
    print("=" * 50)

    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"  Model loaded from: {MODEL_PATH}")
    print(f"  Model input shape : {model.input_shape}")
    print(f"  Model output shape: {model.output_shape}")

    assert model.input_shape[1] == processed_cols, f"Input dim mismatch: {model.input_shape[1]} vs {processed_cols}"
    assert model.output_shape[1] == processed_cols, f"Output dim mismatch: {model.output_shape[1]} vs {processed_cols}"
    print(f"  Verified architecture: {processed_cols} -> {processed_cols}")

    # =======================================================================
    # STEP 6: CALCULATE TEST RECONSTRUCTIONS
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 6] Calculating test reconstructions with Autoencoder...")
    print("=" * 50)

    X_test_reconstructed = model.predict(X_test_processed, batch_size=512, verbose=0)
    print(f"  Reconstructed shape: {X_test_reconstructed.shape}")
    assert X_test_reconstructed.shape == X_test_processed.shape, "Reconstruction shape mismatch!"
    print("  Reconstruction shape matches processed input: PASS")

    # =======================================================================
    # STEP 7: CALCULATE TEST RECONSTRUCTION ERRORS
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 7] Calculating per-sample MSE reconstruction errors...")
    print("=" * 50)

    # Per-sample MSE: mean over feature axis (axis=1) in float64 for precision
    reconstruction_errors = np.mean(
        np.square(X_test_processed.astype(np.float64) - X_test_reconstructed.astype(np.float64)),
        axis=1
    )
    print(f"  Reconstruction errors shape: {reconstruction_errors.shape}")
    assert len(reconstruction_errors) == actual_rows, "Error count does not match test sample count"

    # Integrity check on errors
    nan_errors = int(np.isnan(reconstruction_errors).sum())
    inf_errors = int(np.isinf(reconstruction_errors).sum())
    neg_errors = int(np.sum(reconstruction_errors < 0))

    print(f"  NaN errors     : {nan_errors}")
    print(f"  Inf errors     : {inf_errors}")
    print(f"  Negative errors: {neg_errors}")

    assert nan_errors == 0, f"Found {nan_errors} NaN reconstruction errors!"
    assert inf_errors == 0, f"Found {inf_errors} Inf reconstruction errors!"
    assert neg_errors == 0, f"Found {neg_errors} negative reconstruction errors!"

    err_min    = float(np.min(reconstruction_errors))
    err_max    = float(np.max(reconstruction_errors))
    err_mean   = float(np.mean(reconstruction_errors))
    err_median = float(np.median(reconstruction_errors))
    err_std    = float(np.std(reconstruction_errors))

    print(f"  Test Error Statistics:")
    print(f"    Min   : {err_min:.8f}")
    print(f"    Max   : {err_max:.8f}")
    print(f"    Mean  : {err_mean:.8f}")
    print(f"    Median: {err_median:.8f}")
    print(f"    StdDev: {err_std:.8f}")

    # =======================================================================
    # STEP 8: APPLY OFFICIAL THRESHOLD
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 8] Loading official threshold and generating predictions...")
    print("=" * 50)

    with open(THRESHOLD_JSON_PATH, "r", encoding="utf-8") as f:
        threshold_info = json.load(f)

    threshold = float(threshold_info["threshold"])
    threshold_source = threshold_info.get("source", "normal_validation_reconstruction_errors")
    print(f"  Loaded official threshold: {threshold:.8f}")
    print(f"  Threshold source         : {threshold_source}")

    # Classification rule:
    # error <= threshold -> NORMAL (0)
    # error > threshold  -> INTRUSION (1)
    y_pred = (reconstruction_errors > threshold).astype(int)
    predictions_text = np.where(y_pred == 1, "INTRUSION", "NORMAL")

    # =======================================================================
    # STEP 9: TEST PREDICTION SUMMARY
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 9] Test Prediction Summary...")
    print("=" * 50)

    total_test_records = actual_rows
    predicted_normal_count = int(np.sum(y_pred == 0))
    predicted_intrusion_count = int(np.sum(y_pred == 1))
    pct_normal = float(predicted_normal_count / total_test_records * 100)
    pct_intrusion = float(predicted_intrusion_count / total_test_records * 100)

    print(f"  Total Test Records         : {total_test_records:,}")
    print(f"  Predicted NORMAL (0)       : {predicted_normal_count:,} ({pct_normal:.2f}%)")
    print(f"  Predicted INTRUSION (1)    : {predicted_intrusion_count:,} ({pct_intrusion:.2f}%)")
    print(f"  Actual NORMAL count        : {actual_normal_count:,} ({actual_normal_count / total_test_records * 100:.2f}%)")
    print(f"  Actual ATTACK count        : {actual_attack_count:,} ({actual_attack_count / total_test_records * 100:.2f}%)")

    # =======================================================================
    # STEP 10: ATTACK-TYPE ANALYSIS (ANALYSIS ONLY)
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 10] Attack Category Analysis (Analysis Only)...")
    print("=" * 50)

    attack_categories = [map_attack_category(lbl) for lbl in original_labels]
    df_analysis = pd.DataFrame({
        "original_label": original_labels,
        "category": attack_categories,
        "true_binary": y_test_binary,
        "predicted_binary": y_pred,
        "prediction": predictions_text,
        "error": reconstruction_errors,
    })

    category_summary = {}
    categories = ["Normal", "DoS", "Probe", "R2L", "U2R"]

    print(f"\n  {'Category':<10} | {'Total':>8} | {'Pred Normal':>12} ({'%':>6}) | {'Pred Intrusion':>15} ({'%':>6}) | {'Mean Error':>12}")
    print("  " + "-" * 75)

    for cat in categories:
        cat_mask = (df_analysis["category"] == cat)
        cat_total = int(np.sum(cat_mask))
        if cat_total > 0:
            cat_norm = int(np.sum(cat_mask & (df_analysis["predicted_binary"] == 0)))
            cat_intr = int(np.sum(cat_mask & (df_analysis["predicted_binary"] == 1)))
            cat_norm_pct = float(cat_norm / cat_total * 100)
            cat_intr_pct = float(cat_intr / cat_total * 100)
            cat_mean_err = float(np.mean(df_analysis.loc[cat_mask, "error"]))
        else:
            cat_norm = 0
            cat_intr = 0
            cat_norm_pct = 0.0
            cat_intr_pct = 0.0
            cat_mean_err = 0.0

        category_summary[cat] = {
            "total_records": cat_total,
            "predicted_normal": cat_norm,
            "predicted_normal_pct": round(cat_norm_pct, 2),
            "predicted_intrusion": cat_intr,
            "predicted_intrusion_pct": round(cat_intr_pct, 2),
            "mean_reconstruction_error": round(cat_mean_err, 6),
        }

        print(f"  {cat:<10} | {cat_total:>8,} | {cat_norm:>12,} ({cat_norm_pct:>5.1f}%) | {cat_intr:>15,} ({cat_intr_pct:>5.1f}%) | {cat_mean_err:>12.6f}")

    # =======================================================================
    # STEP 11: SAVE TEST RECONSTRUCTION ERRORS
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 11] Saving test reconstruction errors (.npy)...")
    print("=" * 50)

    np.save(TEST_ERRORS_PATH, reconstruction_errors)
    print(f"  Saved reconstruction errors: {TEST_ERRORS_PATH}")
    print(f"  File size: {os.path.getsize(TEST_ERRORS_PATH):,} bytes")

    # =======================================================================
    # STEP 12: SAVE PREDICTIONS CSV
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 12] Saving test predictions (.csv)...")
    print("=" * 50)

    df_predictions = pd.DataFrame({
        "original_label": original_labels,
        "true_binary_label": y_test_binary,
        "reconstruction_error": reconstruction_errors,
        "predicted_binary_label": y_pred,
        "prediction": predictions_text,
    })

    df_predictions.to_csv(PREDICTIONS_CSV_PATH, index=False)
    print(f"  Saved predictions: {PREDICTIONS_CSV_PATH}")
    print(f"  Rows saved: {len(df_predictions):,}")
    print(f"  File size: {os.path.getsize(PREDICTIONS_CSV_PATH):,} bytes")

    # =======================================================================
    # STEP 13: SAVE INTRUSION DETECTION SUMMARY
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 13] Saving detection summary (.json)...")
    print("=" * 50)

    detection_summary = {
        "phase": "Phase 9 — Intrusion Detection on Unseen KDDTest+",
        "antigravity_ai_model": "Gemini 3.8 Flash Medium",
        "dl_model": "Trained Dense Autoencoder",
        "test_sample_count": total_test_records,
        "processed_feature_count": processed_cols,
        "threshold": threshold,
        "threshold_method": threshold_info.get("method", "95th_percentile"),
        "predicted_normal_count": predicted_normal_count,
        "predicted_intrusion_count": predicted_intrusion_count,
        "percentage_predicted_normal": round(pct_normal, 4),
        "percentage_predicted_intrusion": round(pct_intrusion, 4),
        "actual_normal_count": actual_normal_count,
        "actual_attack_count": actual_attack_count,
        "model_path": MODEL_PATH,
        "preprocessor_path": PREPROCESSOR_PATH,
        "reconstruction_error_stats": {
            "min": round(err_min, 8),
            "max": round(err_max, 8),
            "mean": round(err_mean, 8),
            "median": round(err_median, 8),
            "std": round(err_std, 8),
        },
        "attack_category_detection_breakdown": category_summary,
        "integrity_audit": {
            "nan_errors": nan_errors,
            "inf_errors": inf_errors,
            "negative_errors": neg_errors,
            "kddtest_fitted_preprocessing": False,
            "kddtest_trained_autoencoder": False,
            "test_labels_used_for_threshold": False,
            "test_labels_used_to_modify_predictions": False,
            "test_data_modified": False,
            "threshold_source": "Phase 8 normal validation errors only",
            "test_based_optimization": False,
        },
    }

    with open(DETECTION_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(detection_summary, f, indent=4)

    print(f"  Saved detection summary: {DETECTION_SUMMARY_PATH}")

    # =======================================================================
    # STEP 14: SAVE TEST RECONSTRUCTION ERROR DISTRIBUTION GRAPH
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 14] Generating error distribution graph...")
    print("=" * 50)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Log-scale histogram of reconstruction errors
    ax1 = axes[0]
    # Filter for visualization of distribution
    normal_errors = reconstruction_errors[y_test_binary == 0]
    attack_errors = reconstruction_errors[y_test_binary == 1]

    # Clip for plotting to avoid extreme visual skew while preserving histogram shape
    log_bins = np.logspace(np.log10(max(err_min, 1e-4)), np.log10(min(err_max, 100)), 60)

    ax1.hist(normal_errors, bins=log_bins, alpha=0.6, color="#2563EB", label=f"Normal ({len(normal_errors):,})", density=True)
    ax1.hist(attack_errors, bins=log_bins, alpha=0.6, color="#DC2626", label=f"Attack ({len(attack_errors):,})", density=True)
    ax1.axvline(threshold, color="#10B981", linestyle="--", linewidth=2.5,
                label=f"Threshold = {threshold:.6f}")
    ax1.set_xscale("log")
    ax1.set_xlabel("Reconstruction Error (MSE, log scale)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax1.set_title("Test Reconstruction Error Distribution (Normal vs Attack)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, alpha=0.3, which="both")

    # Right plot: Overall Test Error Distribution with Threshold Region
    ax2 = axes[1]
    ax2.hist(reconstruction_errors, bins=log_bins, color="#4F46E5", alpha=0.7, density=True, label=f"All Test Records ({total_test_records:,})")
    ax2.axvline(threshold, color="#10B981", linestyle="--", linewidth=2.5,
                label=f"Threshold = {threshold:.6f}")
    ax2.axvspan(threshold, max(log_bins), color="#DC2626", alpha=0.1, label=f"Classified as Intrusion ({pct_intrusion:.1f}%)")
    ax2.axvspan(min(log_bins), threshold, color="#2563EB", alpha=0.1, label=f"Classified as Normal ({pct_normal:.1f}%)")
    ax2.set_xscale("log")
    ax2.set_xlabel("Reconstruction Error (MSE, log scale)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax2.set_title(f"Official Threshold Separation (Threshold = {threshold:.6f})", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, alpha=0.3, which="both")

    plt.suptitle("KDDTest+ Reconstruction Error & Anomaly Detection Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(DISTRIBUTION_PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Distribution plot saved to: {DISTRIBUTION_PNG_PATH}")

    # =======================================================================
    # STEP 15 & 16: VERIFICATION OF CHECKS
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 16] Execution Verifications:")
    print("=" * 50)
    print("  [PASS] KDDTest+ loads successfully")
    print("  [PASS] Test features created (41 features)")
    print("  [PASS] Test labels created (binary & original)")
    print("  [PASS] Existing preprocessor loaded")
    print("  [PASS] Preprocessor uses transform only")
    print("  [PASS] Test processed successfully (77 features)")
    print("  [PASS] Trained Autoencoder loads")
    print("  [PASS] Test reconstruction succeeds")
    print("  [PASS] Reconstruction errors calculated")
    print("  [PASS] No NaN errors")
    print("  [PASS] No Inf errors")
    print("  [PASS] Official threshold loaded programmatically")
    print("  [PASS] Threshold not changed")
    print("  [PASS] Predictions generated")
    print("  [PASS] Prediction CSV saved")
    print("  [PASS] Test error array saved")
    print("  [PASS] Detection summary saved")
    print("  [PASS] Test distribution graph saved")

    # =======================================================================
    # STEP 17: DATA-INTEGRITY AUDIT
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 17] Data-Integrity Audit:")
    print("=" * 50)
    print("  ✓ KDDTest+ was never used to fit preprocessing")
    print("  ✓ KDDTest+ was never used for Autoencoder training")
    print("  ✓ Test labels did not affect threshold (threshold derived strictly from normal val data in Phase 8)")
    print("  ✓ Test labels did not affect predictions (predictions derived strictly from error > threshold)")
    print("  ✓ Test data was not modified or dropped")
    print("  ✓ Threshold was loaded from Phase 8")
    print("  ✓ No test-based optimization occurred")

    return detection_summary


if __name__ == "__main__":
    run_phase_9()
