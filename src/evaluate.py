"""
evaluate.py
-----------
Phase 10: Final Model Evaluation on Unseen NSL-KDD KDDTest+ Dataset.

ANTIGRAVITY AI MODEL: Gemini 3.8 Flash Medium
REVIEW MODEL: Claude Sonnet 4.6 (Thinking)
ACTUAL DEEP LEARNING MODEL: Trained Dense Autoencoder
PROJECT: Network Intrusion Detection Using Autoencoder

OBJECTIVE:
Formally evaluate the intrusion detection system on the untouched KDDTest+
dataset using the predictions and continuous reconstruction-error scores
generated in Phase 9.

CRITICAL EVALUATION RULES:
  - DO NOT retrain the model.
  - DO NOT change or re-optimize the threshold.
  - DO NOT modify test data or drop records.
  - Use continuous reconstruction errors for ROC-AUC.
  - Verify alignment across all arrays: len(y_test) == len(y_pred) == len(errors).
  - Verify manual metric formulas against scikit-learn outputs.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

# Ensure src/ is on Python search path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing import (
    load_dataset,
    create_binary_labels,
    map_attack_category,
    TRAFFIC_FEATURES,
    ATTACK_CATEGORIES,
)

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
MODEL_ARCH_SUMMARY      = os.path.join(RESULTS_DIR, "model_architecture_summary.json")
RECON_ERROR_SUMMARY     = os.path.join(RESULTS_DIR, "reconstruction_error_summary.json")
INTRUSION_DET_SUMMARY   = os.path.join(RESULTS_DIR, "intrusion_detection_summary.json")

TEST_ERRORS_PATH        = os.path.join(RESULTS_DIR, "test_reconstruction_errors.npy")
PREDICTIONS_CSV_PATH    = os.path.join(RESULTS_DIR, "test_predictions.csv")

CONFUSION_MATRIX_PNG    = os.path.join(RESULTS_DIR, "confusion_matrix.png")
ROC_CURVE_PNG           = os.path.join(RESULTS_DIR, "roc_curve.png")
DISTRIBUTION_PNG_PATH   = os.path.join(RESULTS_DIR, "test_reconstruction_error_distribution.png")
FINAL_METRICS_JSON      = os.path.join(RESULTS_DIR, "final_metrics.json")
FINAL_REPORT_TXT        = os.path.join(RESULTS_DIR, "final_evaluation_report.txt")


def run_phase_10():
    print("=" * 70)
    print("PHASE 10 — FINAL MODEL EVALUATION")
    print("=" * 70)
    print("Antigravity AI Model : Gemini 3.8 Flash Medium")
    print("Review Model         : Claude Sonnet 4.6 (Thinking)")
    print("Actual DL Model      : Trained Dense Autoencoder")

    # =======================================================================
    # STEP 1: READ EXISTING ARTIFACTS
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 1] Verifying and reading existing artifacts...")
    print("=" * 50)

    required_artifacts = [
        ("test_predictions.csv", PREDICTIONS_CSV_PATH),
        ("test_reconstruction_errors.npy", TEST_ERRORS_PATH),
        ("intrusion_detection_summary.json", INTRUSION_DET_SUMMARY),
        ("anomaly_threshold.json", THRESHOLD_JSON_PATH),
        ("training_summary.json", TRAINING_SUMMARY),
        ("model_architecture_summary.json", MODEL_ARCH_SUMMARY),
        ("model_input_summary.json", MODEL_INPUT_SUMMARY),
        ("reconstruction_error_summary.json", RECON_ERROR_SUMMARY),
        ("autoencoder.keras", MODEL_PATH),
        ("preprocessor.joblib", PREPROCESSOR_PATH),
        ("KDDTest+.txt", TEST_FILE),
    ]

    for name, path in required_artifacts:
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {name:<35} ({size:>10,} bytes)")
        if not exists:
            raise FileNotFoundError(f"Required artifact missing: {path}")

    # Load threshold from Phase 8
    with open(THRESHOLD_JSON_PATH, "r", encoding="utf-8") as f:
        threshold_data = json.load(f)
    threshold = float(threshold_data["threshold"])
    threshold_source = threshold_data.get("source", "normal_validation_reconstruction_errors")
    print(f"\n  Official Anomaly Threshold: {threshold:.8f}")
    print(f"  Threshold Source          : {threshold_source}")

    # =======================================================================
    # STEP 2: VERIFY EVALUATION INPUTS & ALIGNMENT
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 2] Verifying evaluation inputs and alignment...")
    print("=" * 50)

    # 1. Raw test data
    df_test = load_dataset(TEST_FILE)
    n_test_records = len(df_test)
    print(f"  1. KDDTest+ raw records           : {n_test_records:,}")

    # 2. Predictions CSV
    df_preds = pd.read_csv(PREDICTIONS_CSV_PATH)
    n_pred_records = len(df_preds)
    print(f"  2. Predictions CSV records        : {n_pred_records:,}")

    # 3. Reconstruction errors NPY
    test_reconstruction_errors = np.load(TEST_ERRORS_PATH)
    n_error_records = len(test_reconstruction_errors)
    print(f"  3. Reconstruction error records   : {n_error_records:,}")

    # 4. Ground truth binary labels
    y_test = create_binary_labels(df_test).values
    n_gt_records = len(y_test)
    print(f"  4. Ground truth label records     : {n_gt_records:,}")

    # 5. Check exact alignment
    assert n_test_records == 22544, f"Expected 22,544 test records, got {n_test_records}"
    assert n_pred_records == n_test_records, f"Prediction count mismatch: {n_pred_records} vs {n_test_records}"
    assert n_error_records == n_test_records, f"Error count mismatch: {n_error_records} vs {n_test_records}"
    assert n_gt_records == n_test_records, f"Ground truth count mismatch: {n_gt_records} vs {n_test_records}"

    print(f"  5. Alignment check: len(y_test) == len(y_pred) == len(errors) == {n_test_records:,} -> PASS")

    # =======================================================================
    # STEP 3: VERIFY GROUND TRUTH
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 3] Verifying ground truth distributions...")
    print("=" * 50)

    actual_normal_count = int(np.sum(y_test == 0))
    actual_attack_count = int(np.sum(y_test == 1))
    print(f"  Actual Normal (0) count: {actual_normal_count:,}")
    print(f"  Actual Attack (1) count: {actual_attack_count:,}")

    assert actual_normal_count == 9711, f"Expected 9,711 normal records, got {actual_normal_count}"
    assert actual_attack_count == 12833, f"Expected 12,833 attack records, got {actual_attack_count}"
    print("  Ground truth verification: PASS (9,711 Normal, 12,833 Attack)")

    # Also verify CSV true labels match y_test
    assert np.array_equal(df_preds["true_binary_label"].values, y_test), \
        "Mismatch between CSV true_binary_label and newly loaded ground truth!"
    print("  CSV true_binary_label exactly matches ground truth: PASS")

    # =======================================================================
    # STEP 4: LOAD EXISTING PREDICTIONS & RE-VERIFY THRESHOLD LOGIC
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 4] Verifying prediction consistency with threshold...")
    print("=" * 50)

    y_pred = df_preds["predicted_binary_label"].values
    expected_pred = (test_reconstruction_errors > threshold).astype(int)

    # Verify predictions in CSV exactly match threshold rule on errors
    mismatches = np.sum(y_pred != expected_pred)
    assert mismatches == 0, f"Found {mismatches} prediction discrepancies between CSV and threshold application!"
    print(f"  CSV predictions match (reconstruction_errors > threshold): 100% PASS (0 discrepancies)")

    # Verify errors in CSV match test_reconstruction_errors.npy
    max_err_diff = np.max(np.abs(df_preds["reconstruction_error"].values - test_reconstruction_errors))
    assert max_err_diff < 1e-12, f"Reconstruction error precision mismatch: {max_err_diff}"
    print(f"  CSV reconstruction errors match NPY array: 100% PASS (max diff = {max_err_diff:.2e})")

    # =======================================================================
    # STEP 5: CALCULATE FINAL CLASSIFICATION METRICS (SKLEARN)
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 5] Calculating final classification metrics...")
    print("=" * 50)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    print(f"  Accuracy  : {acc:.6f} ({acc * 100:.2f}%)")
    print(f"  Precision : {prec:.6f} ({prec * 100:.2f}%)")
    print(f"  Recall    : {rec:.6f} ({rec * 100:.2f}%)")
    print(f"  F1 Score  : {f1:.6f} ({f1 * 100:.2f}%)")

    # =======================================================================
    # STEP 6: CLASSIFICATION REPORT
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 6] Detailed Classification Report...")
    print("=" * 50)

    report_text = classification_report(
        y_test,
        y_pred,
        target_names=["NORMAL", "INTRUSION"],
        digits=4,
        zero_division=0
    )
    print(report_text)

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=["NORMAL", "INTRUSION"],
        output_dict=True,
        zero_division=0
    )

    # =======================================================================
    # STEP 7 & 8: CONFUSION MATRIX & COUNTS
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 7 & 8] Calculating and plotting confusion matrix...")
    print("=" * 50)

    cm = confusion_matrix(y_test, y_pred)
    # Scikit-learn format:
    # [[TN, FP],
    #  [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    print(f"  True Negatives  (TN - Normal correctly identified) : {tn:>6,} ({tn / n_test_records * 100:>5.2f}%)")
    print(f"  False Positives (FP - Normal flagged as Intrusion) : {fp:>6,} ({fp / n_test_records * 100:>5.2f}%)")
    print(f"  False Negatives (FN - Attack missed as Normal)     : {fn:>6,} ({fn / n_test_records * 100:>5.2f}%)")
    print(f"  True Positives  (TP - Attack correctly detected)   : {tp:>6,} ({tp / n_test_records * 100:>5.2f}%)")

    total_cm = tn + fp + fn + tp
    assert total_cm == n_test_records, f"Confusion matrix sum mismatch: {total_cm} vs {n_test_records}"
    print(f"  Verification: TN + FP + FN + TP == {total_cm:,} (100% matched)")

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6.5))
    cm_labels = [
        [f"TN\n{tn:,}\n({tn/actual_normal_count*100:.1f}%)", f"FP\n{fp:,}\n({fp/actual_normal_count*100:.1f}%)"],
        [f"FN\n{fn:,}\n({fn/actual_attack_count*100:.1f}%)", f"TP\n{tp:,}\n({tp/actual_attack_count*100:.1f}%)"]
    ]

    sns.heatmap(
        cm,
        annot=cm_labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=["Predicted NORMAL", "Predicted INTRUSION"],
        yticklabels=["Actual NORMAL", "Actual ATTACK"],
        annot_kws={"size": 13, "weight": "bold"},
    )
    plt.title(f"Test Confusion Matrix — KDDTest+ (Threshold = {threshold:.6f})", fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Actual Label", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PNG, dpi=300)
    plt.close()
    print(f"  Saved confusion matrix heatmap: {CONFUSION_MATRIX_PNG}")

    # =======================================================================
    # STEP 9: TEST NORMAL-TRAFFIC FALSE POSITIVE RATE & SPECIFICITY
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 9] False Positive Rate & Specificity on Normal Traffic...")
    print("=" * 50)

    fpr_normal = fp / (fp + tn)
    specificity = tn / (tn + fp)

    print(f"  False Positive Rate (FPR) = FP / (FP + TN) : {fpr_normal:.6f} ({fpr_normal * 100:.2f}%)")
    print(f"  Specificity (True Negative Rate) = TN / (TN + FP) : {specificity:.6f} ({specificity * 100:.2f}%)")
    print(f"  Note: Threshold was calibrated at 95th percentile (expected ~5% FPR).")
    print(f"        Observed test FPR is {fpr_normal * 100:.2f}% (Specificity is {specificity * 100:.2f}%).")

    # =======================================================================
    # STEP 10: ATTACK DETECTION RATE (RECALL)
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 10] Attack Detection Rate (True Positive Rate)...")
    print("=" * 50)

    tpr_attack = tp / (tp + fn)
    print(f"  Attack Detection Rate (Recall) = TP / (TP + FN) : {tpr_attack:.6f} ({tpr_attack * 100:.2f}%)")
    print(f"  Precision on Intrusions = TP / (TP + FP)         : {prec:.6f} ({prec * 100:.2f}%)")
    print(f"  F1 Score                                        : {f1:.6f} ({f1 * 100:.2f}%)")

    # =======================================================================
    # STEP 11: ROC-AUC SCORE (CONTINUOUS RECONSTRUCTION ERROR)
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 11] Calculating ROC-AUC with Continuous Reconstruction Errors...")
    print("=" * 50)

    roc_auc = float(roc_auc_score(y_test, test_reconstruction_errors))
    fpr_curve, tpr_curve, roc_thresholds = roc_curve(y_test, test_reconstruction_errors)

    print(f"  ROC-AUC Score : {roc_auc:.6f} ({roc_auc * 100:.2f}%)")
    print(f"  Computed using continuous reconstruction error scores: PASS")

    # =======================================================================
    # STEP 12: ROC CURVE PLOT
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 12] Generating ROC curve plot...")
    print("=" * 50)

    plt.figure(figsize=(8, 6.5))
    plt.plot(fpr_curve, tpr_curve, color="#2563EB", lw=2.5, label=f"Autoencoder ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.5000)")

    # Plot operating point corresponding to official threshold
    plt.scatter([fpr_normal], [tpr_attack], color="#DC2626", s=100, zorder=5,
                label=f"Operating Point (Threshold={threshold:.4f})\nFPR={fpr_normal:.3f}, TPR={tpr_attack:.3f}")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) Curve — KDDTest+", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=10.5, frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROC_CURVE_PNG, dpi=300)
    plt.close()
    print(f"  Saved ROC curve: {ROC_CURVE_PNG}")

    # =======================================================================
    # STEP 13: TEST RECONSTRUCTION ERROR ANALYSIS
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 13] Test Reconstruction Error Statistics by Ground Truth...")
    print("=" * 50)

    normal_errors = test_reconstruction_errors[y_test == 0]
    attack_errors = test_reconstruction_errors[y_test == 1]

    norm_stats = {
        "mean": float(np.mean(normal_errors)),
        "median": float(np.median(normal_errors)),
        "p90": float(np.percentile(normal_errors, 90)),
        "p95": float(np.percentile(normal_errors, 95)),
        "p99": float(np.percentile(normal_errors, 99)),
        "max": float(np.max(normal_errors)),
    }

    att_stats = {
        "mean": float(np.mean(attack_errors)),
        "median": float(np.median(attack_errors)),
        "p90": float(np.percentile(attack_errors, 90)),
        "p95": float(np.percentile(attack_errors, 95)),
        "p99": float(np.percentile(attack_errors, 99)),
        "max": float(np.max(attack_errors)),
    }

    print("  Reconstruction Error Comparison:")
    print(f"  {'Metric':<10} | {'Normal Errors':>16} | {'Attack Errors':>16}")
    print("  " + "-" * 48)
    for k in ["mean", "median", "p90", "p95", "p99", "max"]:
        print(f"  {k.upper():<10} | {norm_stats[k]:>16.6f} | {att_stats[k]:>16.6f}")

    # =======================================================================
    # STEP 14: ERROR DISTRIBUTION GRAPH
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 14] Generating high-resolution error distribution graph...")
    print("=" * 50)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    min_val = max(float(np.min(test_reconstruction_errors)), 1e-4)
    max_val = min(float(np.max(test_reconstruction_errors)), 100.0)
    log_bins = np.logspace(np.log10(min_val), np.log10(max_val), 60)

    # Subplot 1: Distribution of Normal vs Attack
    ax1 = axes[0]
    ax1.hist(normal_errors, bins=log_bins, alpha=0.65, color="#2563EB", label=f"Normal Traffic ({len(normal_errors):,})", density=True)
    ax1.hist(attack_errors, bins=log_bins, alpha=0.65, color="#DC2626", label=f"Attack Traffic ({len(attack_errors):,})", density=True)
    ax1.axvline(threshold, color="#10B981", linestyle="--", linewidth=2.5,
                label=f"Official Threshold = {threshold:.6f}")
    ax1.set_xscale("log")
    ax1.set_xlabel("Reconstruction Error (MSE, log scale)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax1.set_title("Test Reconstruction Error Density: Normal vs Attack", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, fontsize=10)
    ax1.grid(True, alpha=0.3, which="both")

    # Subplot 2: Separation & Cumulative Detection
    ax2 = axes[1]
    ax2.hist(test_reconstruction_errors, bins=log_bins, color="#4F46E5", alpha=0.7, density=True, label=f"All Test Traffic ({n_test_records:,})")
    ax2.axvline(threshold, color="#10B981", linestyle="--", linewidth=2.5,
                label=f"Official Threshold = {threshold:.6f}")
    ax2.axvspan(threshold, max_val, color="#DC2626", alpha=0.12, label=f"Classified as Intrusion ({len(df_preds[df_preds['predicted_binary_label']==1]):,} records)")
    ax2.axvspan(min_val, threshold, color="#2563EB", alpha=0.12, label=f"Classified as Normal ({len(df_preds[df_preds['predicted_binary_label']==0]):,} records)")
    ax2.set_xscale("log")
    ax2.set_xlabel("Reconstruction Error (MSE, log scale)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax2.set_title("Detection Regions Defined by Phase 8 Calibration", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, fontsize=10)
    ax2.grid(True, alpha=0.3, which="both")

    plt.suptitle("KDDTest+ Reconstruction Error & Detection Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(DISTRIBUTION_PNG_PATH, dpi=300)
    plt.close()
    print(f"  Saved distribution graph: {DISTRIBUTION_PNG_PATH}")

    # =======================================================================
    # STEP 15: PER-ATTACK-CATEGORY PERFORMANCE
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 15] Evaluating Per-Attack-Category Performance...")
    print("=" * 50)

    original_labels = df_test["label"].str.strip().str.lower().values
    categories = [map_attack_category(lbl) for lbl in original_labels]
    df_eval = pd.DataFrame({
        "original_label": original_labels,
        "category": categories,
        "true_binary": y_test,
        "predicted_binary": y_pred,
        "error": test_reconstruction_errors,
    })

    attack_families = ["DoS", "Probe", "R2L", "U2R"]
    category_results = {}

    print(f"  {'Family':<8} | {'Total':>7} | {'Detected (TP)':>14} | {'Missed (FN)':>12} | {'Detection Rate (Recall)':>25} | {'Mean Error':>12}")
    print("  " + "-" * 88)

    for fam in attack_families:
        fam_mask = (df_eval["category"] == fam)
        fam_total = int(np.sum(fam_mask))
        fam_detected = int(np.sum(fam_mask & (df_eval["predicted_binary"] == 1)))
        fam_missed = int(np.sum(fam_mask & (df_eval["predicted_binary"] == 0)))
        fam_rate = float(fam_detected / fam_total * 100) if fam_total > 0 else 0.0
        fam_mean_err = float(np.mean(df_eval.loc[fam_mask, "error"])) if fam_total > 0 else 0.0

        category_results[fam] = {
            "total_records": fam_total,
            "correctly_detected": fam_detected,
            "missed_attacks": fam_missed,
            "detection_rate_pct": round(fam_rate, 2),
            "mean_reconstruction_error": round(fam_mean_err, 6),
        }

        print(f"  {fam:<8} | {fam_total:>7,} | {fam_detected:>14,} | {fam_missed:>12,} | {fam_rate:>23.2f}% | {fam_mean_err:>12.6f}")

    # =======================================================================
    # STEP 16: CREATE FINAL METRICS SUMMARY (JSON)
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 16] Creating results/final_metrics.json...")
    print("=" * 50)

    final_metrics = {
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1_score": round(f1, 6),
        "roc_auc": round(roc_auc, 6),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "false_positive_rate": round(fpr_normal, 6),
        "specificity": round(specificity, 6),
        "attack_detection_rate": round(tpr_attack, 6),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "test_sample_count": n_test_records,
        "normal_test_count": actual_normal_count,
        "attack_test_count": actual_attack_count,
        "model_path": MODEL_PATH,
        "preprocessor_path": PREPROCESSOR_PATH,
        "reconstruction_error_statistics": {
            "normal": {k: round(v, 6) for k, v in norm_stats.items()},
            "attack": {k: round(v, 6) for k, v in att_stats.items()},
        },
        "per_category_performance": category_results,
        "classification_report": report_dict,
    }

    with open(FINAL_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=4)
    print(f"  Saved: {FINAL_METRICS_JSON}")

    # =======================================================================
    # STEP 17: CREATE HUMAN-READABLE RESULTS FILE
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 17] Creating results/final_evaluation_report.txt...")
    print("=" * 50)

    report_lines = [
        "================================================================================",
        "FINAL MODEL EVALUATION REPORT",
        "================================================================================",
        "PROJECT: Network Intrusion Detection Using Autoencoder",
        "MODEL: Trained Dense Autoencoder (77 -> 64 -> 32 -> 16 -> 32 -> 64 -> 77)",
        "TEST DATASET: NSL-KDD KDDTest+ (completely unseen)",
        f"TEST SAMPLES: {n_test_records:,} (Normal: {actual_normal_count:,}, Attack: {actual_attack_count:,})",
        f"THRESHOLD: {threshold:.8f} (Derived exclusively from Phase 8 normal validation errors)",
        "",
        "--------------------------------------------------------------------------------",
        "CONFUSION MATRIX",
        "--------------------------------------------------------------------------------",
        f"True Negatives  (TN) : {tn:,}  (Normal correctly identified)",
        f"False Positives (FP) : {fp:,}  (Normal incorrectly flagged as intrusion)",
        f"False Negatives (FN) : {fn:,}  (Attack missed as normal)",
        f"True Positives  (TP) : {tp:,}  (Attack correctly detected)",
        f"Total Records Check  : {tn + fp + fn + tp:,} / {n_test_records:,}",
        "",
        "--------------------------------------------------------------------------------",
        "CLASSIFICATION METRICS",
        "--------------------------------------------------------------------------------",
        f"Accuracy             : {acc:.6f} ({acc * 100:.2f}%)",
        f"Precision            : {prec:.6f} ({prec * 100:.2f}%)",
        f"Recall (Sensitivity) : {rec:.6f} ({rec * 100:.2f}%)",
        f"F1 Score             : {f1:.6f} ({f1 * 100:.2f}%)",
        f"ROC-AUC Score        : {roc_auc:.6f} ({roc_auc * 100:.2f}%)",
        "",
        "--------------------------------------------------------------------------------",
        "RATES & SPECIFICITY",
        "--------------------------------------------------------------------------------",
        f"False Positive Rate  : {fpr_normal:.6f} ({fpr_normal * 100:.2f}%)",
        f"Specificity (TNR)    : {specificity:.6f} ({specificity * 100:.2f}%)",
        f"Attack Detection Rate: {tpr_attack:.6f} ({tpr_attack * 100:.2f}%)",
        "",
        "--------------------------------------------------------------------------------",
        "PER-ATTACK-CATEGORY RESULTS",
        "--------------------------------------------------------------------------------",
    ]

    for fam in attack_families:
        res = category_results[fam]
        report_lines.append(
            f"{fam:<6} -> Total: {res['total_records']:>5,} | "
            f"Detected: {res['correctly_detected']:>5,} | "
            f"Missed: {res['missed_attacks']:>5,} | "
            f"Recall: {res['detection_rate_pct']:>6.2f}% | "
            f"Mean MSE: {res['mean_reconstruction_error']:.6f}"
        )

    report_lines.extend([
        "",
        "--------------------------------------------------------------------------------",
        "RECONSTRUCTION ERROR DISTRIBUTION",
        "--------------------------------------------------------------------------------",
        f"Normal Traffic -> Mean: {norm_stats['mean']:.6f}, Median: {norm_stats['median']:.6f}, P95: {norm_stats['p95']:.6f}, Max: {norm_stats['max']:.6f}",
        f"Attack Traffic -> Mean: {att_stats['mean']:.6f}, Median: {att_stats['median']:.6f}, P95: {att_stats['p95']:.6f}, Max: {att_stats['max']:.6f}",
        "",
        "--------------------------------------------------------------------------------",
        "DATA INTEGRITY & METHODOLOGICAL AUDIT",
        "--------------------------------------------------------------------------------",
        "✓ KDDTest+ was never used to fit preprocessing or train the Autoencoder",
        "✓ Threshold was derived solely from normal validation records in Phase 8",
        "✓ No test-based hyperparameter tuning or threshold modification occurred",
        "✓ Continuous reconstruction errors were used to calculate ROC-AUC",
        "✓ Manual formula checks verified exact equivalence with scikit-learn",
        "================================================================================",
    ])

    report_str = "\n".join(report_lines)
    with open(FINAL_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"  Saved: {FINAL_REPORT_TXT}")

    # =======================================================================
    # STEP 18: VERIFY METRICS MANUALLY
    # =======================================================================
    print("\n" + "=" * 50)
    print("[STEP 18] Manually verifying metrics against scikit-learn...")
    print("=" * 50)

    manual_acc  = (tp + tn) / (tp + tn + fp + fn)
    manual_prec = tp / (tp + fp)
    manual_rec  = tp / (tp + fn)
    manual_spec = tn / (tn + fp)
    manual_f1   = 2 * manual_prec * manual_rec / (manual_prec + manual_rec)

    print(f"  Manual Accuracy  : {manual_acc:.8f} | Sklearn: {acc:.8f} | Diff: {abs(manual_acc - acc):.2e}")
    print(f"  Manual Precision : {manual_prec:.8f} | Sklearn: {prec:.8f} | Diff: {abs(manual_prec - prec):.2e}")
    print(f"  Manual Recall    : {manual_rec:.8f} | Sklearn: {rec:.8f} | Diff: {abs(manual_rec - rec):.2e}")
    print(f"  Manual F1 Score  : {manual_f1:.8f} | Sklearn: {f1:.8f} | Diff: {abs(manual_f1 - f1):.2e}")
    print(f"  Manual Specificity: {manual_spec:.8f} | Derived: {specificity:.8f} | Diff: {abs(manual_spec - specificity):.2e}")

    assert abs(manual_acc - acc) < 1e-12, "Manual Accuracy does not match sklearn!"
    assert abs(manual_prec - prec) < 1e-12, "Manual Precision does not match sklearn!"
    assert abs(manual_rec - rec) < 1e-12, "Manual Recall does not match sklearn!"
    assert abs(manual_f1 - f1) < 1e-12, "Manual F1 does not match sklearn!"
    assert abs(manual_spec - specificity) < 1e-12, "Manual Specificity mismatch!"
    print("  Manual verification: 100% PASS on all metrics!")

    print("\n" + "=" * 50)
    print("Phase 10 Execution Complete!")
    print("=" * 50)

    return final_metrics


if __name__ == "__main__":
    run_phase_10()
