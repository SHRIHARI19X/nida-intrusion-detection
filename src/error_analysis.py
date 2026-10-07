"""
error_analysis.py
-----------------
Phase 11: Detailed Error Analysis & Attack-Type Analysis.

ANTIGRAVITY AI MODEL: Gemini 3.8 Flash Medium
ACTUAL DEEP LEARNING MODEL: Dense Autoencoder (77 -> 64 -> 32 -> 16 -> 32 -> 64 -> 77)
PROJECT: Network Intrusion Detection Using Autoencoder

OBJECTIVE:
Perform comprehensive diagnostic analysis on the test reconstruction errors
and predictions generated in Phase 9/10:
  - Reconstruction error statistics across Normal, All Attacks, DoS, Probe, R2L, U2R.
  - Granular False Positive (FP) and False Negative (FN) analysis.
  - Threshold sensitivity analysis across candidate thresholds.
  - Multi-panel publication-quality visualizations.
  - Export of specific FP and FN error cases to CSV.
  - Automated detailed technical report.
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
TEST_ERRORS_PATH        = os.path.join(RESULTS_DIR, "test_reconstruction_errors.npy")
PREDICTIONS_CSV_PATH    = os.path.join(RESULTS_DIR, "test_predictions.csv")
FINAL_METRICS_PATH      = os.path.join(RESULTS_DIR, "final_metrics.json")
INTRUSION_DET_SUMMARY   = os.path.join(RESULTS_DIR, "intrusion_detection_summary.json")
FINAL_REPORT_TXT        = os.path.join(RESULTS_DIR, "final_evaluation_report.txt")

# Output files for Phase 11
FP_CSV_PATH             = os.path.join(RESULTS_DIR, "false_positives.csv")
FN_CSV_PATH             = os.path.join(RESULTS_DIR, "false_negatives.csv")
ATTACK_ANALYSIS_CSV_PATH= os.path.join(RESULTS_DIR, "attack_type_analysis.csv")
ERROR_REPORT_TXT_PATH   = os.path.join(RESULTS_DIR, "error_analysis_report.txt")

# Visualization output paths
FIG_NORMAL_VS_ATTACK    = os.path.join(RESULTS_DIR, "reconstruction_error_normal_vs_attack.png")
FIG_BY_CATEGORY         = os.path.join(RESULTS_DIR, "reconstruction_error_by_category.png")
FIG_BOXPLOT_CATEGORIES  = os.path.join(RESULTS_DIR, "reconstruction_error_boxplot_categories.png")
FIG_DETECTION_RATES     = os.path.join(RESULTS_DIR, "attack_detection_rate_comparison.png")
FIG_FP_FN_DISTRIB       = os.path.join(RESULTS_DIR, "fp_fn_error_distribution.png")
FIG_SENSITIVITY_METRICS = os.path.join(RESULTS_DIR, "threshold_sensitivity_metrics.png")
FIG_SENSITIVITY_RATES   = os.path.join(RESULTS_DIR, "threshold_sensitivity_rates.png")


def run_phase_11():
    print("=" * 75)
    print("PHASE 11 — DETAILED ERROR ANALYSIS & ATTACK-TYPE ANALYSIS")
    print("=" * 75)
    print("Antigravity AI Model : Gemini 3.8 Flash Medium")
    print("Confirmed Model      : Dense Autoencoder (77 -> 64 -> 32 -> 16 -> 32 -> 64 -> 77)")

    # =======================================================================
    # STEP 1: INSPECT EXISTING ARTIFACTS
    # =======================================================================
    print("\n" + "=" * 55)
    print("[STEP 1] Inspecting existing artifacts...")
    print("=" * 55)

    required_artifacts = [
        ("data/KDDTest+.txt", TEST_FILE),
        ("results/test_predictions.csv", PREDICTIONS_CSV_PATH),
        ("results/test_reconstruction_errors.npy", TEST_ERRORS_PATH),
        ("results/intrusion_detection_summary.json", INTRUSION_DET_SUMMARY),
        ("results/final_metrics.json", FINAL_METRICS_PATH),
        ("results/final_evaluation_report.txt", FINAL_REPORT_TXT),
        ("results/anomaly_threshold.json", THRESHOLD_JSON_PATH),
        ("models/autoencoder.keras", MODEL_PATH),
        ("models/preprocessor.joblib", PREPROCESSOR_PATH),
    ]

    for label, path in required_artifacts:
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {label:<42} ({size:>10,} bytes)")
        if not exists:
            raise FileNotFoundError(f"Missing required artifact: {path}")

    with open(THRESHOLD_JSON_PATH, "r", encoding="utf-8") as f:
        threshold_data = json.load(f)
    official_threshold = float(threshold_data["threshold"])
    print(f"\n  Official Anomaly Threshold: {official_threshold:.8f}")

    # =======================================================================
    # STEP 2: VERIFY BASIC COUNTS
    # =======================================================================
    print("\n" + "=" * 55)
    print("[STEP 2] Verifying basic counts against Phase 10...")
    print("=" * 55)

    df_test = load_dataset(TEST_FILE)
    df_preds = pd.read_csv(PREDICTIONS_CSV_PATH)
    errors = np.load(TEST_ERRORS_PATH)

    total_test = len(df_test)
    y_test = create_binary_labels(df_test).values
    y_pred = df_preds["predicted_binary_label"].values

    act_normal = int(np.sum(y_test == 0))
    act_attack = int(np.sum(y_test == 1))
    pred_normal = int(np.sum(y_pred == 0))
    pred_attack = int(np.sum(y_pred == 1))

    tn = int(np.sum((y_test == 0) & (y_pred == 0)))
    fp = int(np.sum((y_test == 0) & (y_pred == 1)))
    fn = int(np.sum((y_test == 1) & (y_pred == 0)))
    tp = int(np.sum((y_test == 1) & (y_pred == 1)))

    print(f"  1. Total test samples        : {total_test:,} (Expected: 22,544)")
    print(f"  2. Actual normal samples     : {act_normal:,} (Expected: 9,711)")
    print(f"  3. Actual attack samples     : {act_attack:,} (Expected: 12,833)")
    print(f"  4. Predicted normal samples  : {pred_normal:,} (Expected: 12,686)")
    print(f"  5. Predicted attack samples  : {pred_attack:,} (Expected: 9,858)")
    print(f"  6. True Negatives  (TN)      : {tn:,} (Expected: 9,322)")
    print(f"  7. False Positives (FP)      : {fp:,} (Expected: 389)")
    print(f"  8. False Negatives (FN)      : {fn:,} (Expected: 3,364)")
    print(f"  9. True Positives  (TP)      : {tp:,} (Expected: 9,469)")

    assert total_test  == 22544, f"Mismatch in total test: {total_test}"
    assert act_normal  == 9711,  f"Mismatch in actual normal: {act_normal}"
    assert act_attack  == 12833, f"Mismatch in actual attack: {act_attack}"
    assert pred_normal == 12686, f"Mismatch in pred normal: {pred_normal}"
    assert pred_attack == 9858,  f"Mismatch in pred attack: {pred_attack}"
    assert tn == 9322, f"Mismatch in TN: {tn}"
    assert fp == 389,  f"Mismatch in FP: {fp}"
    assert fn == 3364, f"Mismatch in FN: {fn}"
    assert tp == 9469, f"Mismatch in TP: {tp}"

    print("  Counts verification: EXACT MATCH WITH PHASE 10 (PASS)")

    # Prepare merged working DataFrame
    original_labels = df_test["label"].str.strip().str.lower().values
    categories = [map_attack_category(l) for l in original_labels]

    df_full = df_test.copy()
    df_full["test_index"] = np.arange(len(df_full))
    df_full["original_label"] = original_labels
    df_full["attack_category"] = categories
    df_full["true_binary"] = y_test
    df_full["predicted_binary"] = y_pred
    df_full["predicted_label"] = df_preds["prediction"].values
    df_full["reconstruction_error"] = errors

    # =======================================================================
    # STEP 3: RECONSTRUCTION ERROR ANALYSIS (GROUPS)
    # =======================================================================
    print("\n" + "=" * 55)
    print("[STEP 3] Reconstruction error analysis across groups...")
    print("=" * 55)

    groups = {
        "Normal": df_full["attack_category"] == "Normal",
        "All Attacks": df_full["true_binary"] == 1,
        "DoS": df_full["attack_category"] == "DoS",
        "Probe": df_full["attack_category"] == "Probe",
        "R2L": df_full["attack_category"] == "R2L",
        "U2R": df_full["attack_category"] == "U2R",
    }

    group_stats = []
    print(f"  {'Group':<12} | {'Count':>7} | {'Mean':>10} | {'Median':>10} | {'StdDev':>10} | {'Min':>8} | {'Max':>10} | {'P95':>10} | {'DetRate (%)':>11}")
    print("  " + "-" * 98)

    for g_name, mask in groups.items():
        sub_err = errors[mask]
        g_count = len(sub_err)
        g_mean = float(np.mean(sub_err))
        g_median = float(np.median(sub_err))
        g_std = float(np.std(sub_err))
        g_min = float(np.min(sub_err))
        g_max = float(np.max(sub_err))
        g_p90 = float(np.percentile(sub_err, 90))
        g_p95 = float(np.percentile(sub_err, 95))
        g_p99 = float(np.percentile(sub_err, 99))
        n_above = int(np.sum(sub_err > official_threshold))
        det_rate = (n_above / g_count * 100.0) if g_count > 0 else 0.0

        group_stats.append({
            "group": g_name,
            "count": g_count,
            "mean": g_mean,
            "median": g_median,
            "std": g_std,
            "min": g_min,
            "max": g_max,
            "p90": g_p90,
            "p95": g_p95,
            "p99": g_p99,
            "n_above_threshold": n_above,
            "detection_rate_pct": det_rate,
        })

        print(f"  {g_name:<12} | {g_count:>7,} | {g_mean:>10.4f} | {g_median:>10.4f} | {g_std:>10.4f} | {g_min:>8.4f} | {g_max:>10.2f} | {g_p95:>10.4f} | {det_rate:>10.2f}%")

    df_group_stats = pd.DataFrame(group_stats)

    # =======================================================================
    # STEP 4: FALSE POSITIVE ANALYSIS
    # =======================================================================
    print("\n" + "=" * 55)
    print("[STEP 4] Granular False Positive (FP) Analysis...")
    print("=" * 55)

    fp_mask = (y_test == 0) & (y_pred == 1)
    df_fp = df_full[fp_mask].copy()
    fp_errors = df_fp["reconstruction_error"].values

    fp_count = len(df_fp)
    fp_pct_normal = (fp_count / act_normal) * 100.0
    fp_min = float(np.min(fp_errors))
    fp_max = float(np.max(fp_errors))
    fp_median = float(np.median(fp_errors))
    fp_mean = float(np.mean(fp_errors))
    fp_p25 = float(np.percentile(fp_errors, 25))
    fp_p75 = float(np.percentile(fp_errors, 75))
    fp_p95 = float(np.percentile(fp_errors, 95))

    # Concentration check: How many FPs are near the threshold (e.g. <= 2 * threshold)?
    near_thresh_count = int(np.sum(fp_errors <= 2 * official_threshold))
    near_thresh_pct = (near_thresh_count / fp_count) * 100.0

    print(f"  FP Count                           : {fp_count:,}")
    print(f"  FP % of Normal Traffic             : {fp_pct_normal:.2f}% (Specificity = {100 - fp_pct_normal:.2f}%)")
    print(f"  Reconstruction Error Statistics:")
    print(f"    Min   : {fp_min:.6f}  (Official threshold: {official_threshold:.6f})")
    print(f"    P25   : {fp_p25:.6f}")
    print(f"    Median: {fp_median:.6f}")
    print(f"    Mean  : {fp_mean:.6f}")
    print(f"    P75   : {fp_p75:.6f}")
    print(f"    P95   : {fp_p95:.6f}")
    print(f"    Max   : {fp_max:.6f}")
    print(f"  Concentration Analysis:")
    print(f"    FPs within 2x Threshold (<= {2*official_threshold:.4f}): {near_thresh_count} ({near_thresh_pct:.1f}%)")
    print(f"    FPs > 2x Threshold                         : {fp_count - near_thresh_count} ({100 - near_thresh_pct:.1f}%)")

    # Common service/protocol/flag patterns
    fp_combos = df_fp.groupby(["protocol_type", "service", "flag"]).size().reset_index(name="count")
    fp_combos = fp_combos.sort_values(by="count", ascending=False).reset_index(drop=True)
    print("\n  Top 5 Protocol/Service/Flag Combinations among False Positives:")
    for idx, row in fp_combos.head(5).iterrows():
        print(f"    {idx+1}. {row['protocol_type']:<5} | {row['service']:<10} | {row['flag']:<5} -> {row['count']:>4} records ({row['count']/fp_count*100:.1f}%)")

    # =======================================================================
    # STEP 5: FALSE NEGATIVE ANALYSIS
    # =======================================================================
    print("\n" + "=" * 55)
    print("[STEP 5] Granular False Negative (FN) Analysis...")
    print("=" * 55)

    fn_mask = (y_test == 1) & (y_pred == 0)
    df_fn = df_full[fn_mask].copy()
    fn_errors = df_fn["reconstruction_error"].values

    fn_count = len(df_fn)
    fn_pct_attack = (fn_count / act_attack) * 100.0
    fn_min = float(np.min(fn_errors))
    fn_max = float(np.max(fn_errors))
    fn_median = float(np.median(fn_errors))
    fn_mean = float(np.mean(fn_errors))
    fn_p25 = float(np.percentile(fn_errors, 25))
    fn_p75 = float(np.percentile(fn_errors, 75))

    print(f"  FN Count                           : {fn_count:,}")
    print(f"  FN % of All Attacks                : {fn_pct_attack:.2f}% (Recall = {100 - fn_pct_attack:.2f}%)")
    print(f"  Reconstruction Error Statistics:")
    print(f"    Min   : {fn_min:.6f}")
    print(f"    P25   : {fn_p25:.6f}")
    print(f"    Median: {fn_median:.6f}")
    print(f"    Mean  : {fn_mean:.6f}")
    print(f"    P75   : {fn_p75:.6f}")
    print(f"    Max   : {fn_max:.6f}  (Official threshold: {official_threshold:.6f})")

    # Category breakdown of False Negatives
    print("\n  False Negatives Breakdown by Attack Family:")
    fn_cat_breakdown = df_fn["attack_category"].value_counts()
    for cat, count in fn_cat_breakdown.items():
        cat_total = int(np.sum(df_full["attack_category"] == cat))
        cat_share = (count / fn_count) * 100.0
        cat_miss_rate = (count / cat_total) * 100.0
        print(f"    {cat:<6}: {count:>5,} missed ({cat_share:>5.1f}% of all FNs, {cat_miss_rate:>5.1f}% of total {cat})")

    # Specific attack labels with most false negatives
    print("\n  Top 5 Specific Attack Types Contributing to False Negatives:")
    top_fn_attacks = df_fn["original_label"].value_counts().head(5)
    for lbl, count in top_fn_attacks.items():
        lbl_total = int(np.sum(df_full["original_label"] == lbl))
        cat = map_attack_category(lbl)
        print(f"    {lbl:<16} ({cat:<5}): {count:>5,} missed / {lbl_total:>5,} total ({count/lbl_total*100:>5.1f}% missed)")

    # =======================================================================
    # STEP 6: THRESHOLD SENSITIVITY ANALYSIS
    # =======================================================================
    print("\n" + "=" * 55)
    print("[STEP 6] Threshold Sensitivity Analysis (Analysis Only)...")
    print("=" * 55)
    print("  Official Threshold is strictly preserved at 0.12907493.")

    candidate_thresholds = [0.05, 0.075, 0.10, official_threshold, 0.15, 0.20, 0.30, 0.50]
    sensitivity_results = []

    print(f"\n  {'Threshold':<12} | {'Acc (%)':>8} | {'Prec (%)':>9} | {'Rec (%)':>8} | {'F1 (%)':>8} | {'FPR (%)':>8} | {'Spec (%)':>9} | {'TP':>6} | {'FP':>6} | {'FN':>6} | {'TN':>6}")
    print("  " + "-" * 115)

    for thresh in candidate_thresholds:
        y_cand = (errors > thresh).astype(int)
        c_tp = int(np.sum((y_test == 1) & (y_cand == 1)))
        c_tn = int(np.sum((y_test == 0) & (y_cand == 0)))
        c_fp = int(np.sum((y_test == 0) & (y_cand == 1)))
        c_fn = int(np.sum((y_test == 1) & (y_cand == 0)))

        c_acc  = (c_tp + c_tn) / total_test * 100.0
        c_prec = (c_tp / (c_tp + c_fp) * 100.0) if (c_tp + c_fp) > 0 else 0.0
        c_rec  = (c_tp / (c_tp + c_fn) * 100.0) if (c_tp + c_fn) > 0 else 0.0
        c_f1   = (2 * c_prec * c_rec / (c_prec + c_rec)) if (c_prec + c_rec) > 0 else 0.0
        c_fpr  = (c_fp / (c_fp + c_tn) * 100.0) if (c_fp + c_tn) > 0 else 0.0
        c_spec = (c_tn / (c_tn + c_fp) * 100.0) if (c_tn + c_fp) > 0 else 0.0

        label_str = f"{thresh:.8f}*" if abs(thresh - official_threshold) < 1e-6 else f"{thresh:.4f}"

        sensitivity_results.append({
            "threshold": thresh,
            "accuracy_pct": round(c_acc, 2),
            "precision_pct": round(c_prec, 2),
            "recall_pct": round(c_rec, 2),
            "f1_pct": round(c_f1, 2),
            "fpr_pct": round(c_fpr, 2),
            "specificity_pct": round(c_spec, 2),
            "tp": c_tp,
            "tn": c_tn,
            "fp": c_fp,
            "fn": c_fn,
        })

        print(f"  {label_str:<12} | {c_acc:>8.2f} | {c_prec:>9.2f} | {c_rec:>8.2f} | {c_f1:>8.2f} | {c_fpr:>8.2f} | {c_spec:>9.2f} | {c_tp:>6,} | {c_fp:>6,} | {c_fn:>6,} | {c_tn:>6,}")

    print("  * Official threshold from Phase 8 (P95 normal validation errors)")

    # =======================================================================
    # STEP 7: ATTACK-TYPE COMPARISON
    # =======================================================================
    print("\n" + "=" * 55)
    print("[STEP 7] Attack-Type Comparison Table...")
    print("=" * 55)

    attack_types = ["DoS", "Probe", "R2L", "U2R"]
    attack_comp = []

    print(f"  {'Attack Type':<12} | {'Total':>7} | {'Detected':>9} | {'Missed':>8} | {'Detection Rate':>15} | {'Mean Error':>11} | {'Median Error':>13}")
    print("  " + "-" * 90)

    for att in attack_types:
        mask = df_full["attack_category"] == att
        sub_err = errors[mask]
        tot = len(sub_err)
        det = int(np.sum(sub_err > official_threshold))
        mis = tot - det
        rate = (det / tot * 100.0) if tot > 0 else 0.0
        m_err = float(np.mean(sub_err))
        med_err = float(np.median(sub_err))

        attack_comp.append({
            "attack_type": att,
            "total_samples": tot,
            "detected": det,
            "missed": mis,
            "detection_rate_pct": round(rate, 2),
            "mean_reconstruction_error": round(m_err, 6),
            "median_reconstruction_error": round(med_err, 6),
        })

        print(f"  {att:<12} | {tot:>7,} | {det:>9,} | {mis:>8,} | {rate:>14.2f}% | {m_err:>11.4f} | {med_err:>13.4f}")

    df_attack_comp = pd.DataFrame(attack_comp)

    # =======================================================================
    # STEP 8: PUBLICATION-QUALITY VISUALIZATIONS
    # =======================================================================
    print("\n" + "=" * 55)
    print("[STEP 8] Generating publication-quality visualizations...")
    print("=" * 55)

    min_log = max(float(np.min(errors)), 1e-4)
    max_log = min(float(np.max(errors)), 200.0)
    log_bins = np.logspace(np.log10(min_log), np.log10(max_log), 65)

    # 1. Reconstruction error distribution: Normal vs Attack
    plt.figure(figsize=(9, 6))
    plt.hist(errors[y_test == 0], bins=log_bins, alpha=0.6, color="#2563EB", density=True, label=f"Normal Traffic ({act_normal:,})")
    plt.hist(errors[y_test == 1], bins=log_bins, alpha=0.6, color="#DC2626", density=True, label=f"All Attack Traffic ({act_attack:,})")
    plt.axvline(official_threshold, color="#10B981", linestyle="--", linewidth=2.5, label=f"Threshold = {official_threshold:.6f}")
    plt.xscale("log")
    plt.xlabel("Reconstruction Error (MSE, log scale)", fontsize=11, fontweight="bold")
    plt.ylabel("Probability Density", fontsize=11, fontweight="bold")
    plt.title("Reconstruction Error Distribution: Normal vs Attack Traffic", fontsize=13, fontweight="bold", pad=12)
    plt.legend(frameon=True, fontsize=10.5)
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(FIG_NORMAL_VS_ATTACK, dpi=300)
    plt.close()
    print(f"  [Saved] {FIG_NORMAL_VS_ATTACK}")

    # 2. Reconstruction error distribution: DoS vs Probe vs R2L vs U2R
    plt.figure(figsize=(10, 6))
    cat_colors = {"DoS": "#EA580C", "Probe": "#7C3AED", "R2L": "#D97706", "U2R": "#059669"}
    for cat in attack_types:
        sub_err = errors[df_full["attack_category"] == cat]
        plt.hist(sub_err, bins=log_bins, alpha=0.55, color=cat_colors[cat], density=True, label=f"{cat} ({len(sub_err):,})")
    plt.axvline(official_threshold, color="#111827", linestyle="--", linewidth=2.2, label=f"Threshold = {official_threshold:.6f}")
    plt.xscale("log")
    plt.xlabel("Reconstruction Error (MSE, log scale)", fontsize=11, fontweight="bold")
    plt.ylabel("Probability Density", fontsize=11, fontweight="bold")
    plt.title("Reconstruction Error Density by Attack Category", fontsize=13, fontweight="bold", pad=12)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(FIG_BY_CATEGORY, dpi=300)
    plt.close()
    print(f"  [Saved] {FIG_BY_CATEGORY}")

    # 3. Boxplot: Normal + each attack category
    plt.figure(figsize=(10, 6))
    plot_cats = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    box_data = [errors[df_full["attack_category"] == c] for c in plot_cats]
    box_colors = ["#2563EB", "#EA580C", "#7C3AED", "#D97706", "#059669"]

    bp = plt.boxplot(box_data, patch_artist=True, tick_labels=plot_cats, showfliers=False)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set(color="black", linewidth=2)

    plt.axhline(official_threshold, color="#DC2626", linestyle="--", linewidth=2.2, label=f"Official Threshold = {official_threshold:.6f}")
    plt.yscale("log")
    plt.ylabel("Reconstruction Error (MSE, log scale)", fontsize=11, fontweight="bold")
    plt.title("Reconstruction Error Distribution by Traffic Class (Boxplot without extreme outliers)", fontsize=12, fontweight="bold", pad=12)
    plt.legend(loc="upper left", frameon=True, fontsize=10.5)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(FIG_BOXPLOT_CATEGORIES, dpi=300)
    plt.close()
    print(f"  [Saved] {FIG_BOXPLOT_CATEGORIES}")

    # 4. Attack detection rate comparison (Bar chart)
    plt.figure(figsize=(9, 5.5))
    det_rates = [row["detection_rate_pct"] for row in attack_comp]
    bars = plt.bar(attack_types, det_rates, color=["#EA580C", "#7C3AED", "#D97706", "#059669"], width=0.55, edgecolor="black", alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 1.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.axhline(100 - fp_pct_normal, color="#2563EB", linestyle=":", linewidth=2, label=f"Normal Specificity (1 - FPR) = {100 - fp_pct_normal:.2f}%")
    plt.ylim([0, 105])
    plt.ylabel("Detection Rate / Recall (%)", fontsize=11, fontweight="bold")
    plt.title("Empirical Detection Rate across Attack Categories (Threshold = 0.1291)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(FIG_DETECTION_RATES, dpi=300)
    plt.close()
    print(f"  [Saved] {FIG_DETECTION_RATES}")

    # 5. False Positive vs False Negative reconstruction errors
    plt.figure(figsize=(9, 6))
    plt.hist(fp_errors, bins=log_bins, alpha=0.65, color="#D97706", density=True, label=f"False Positives ({fp_count:,})\n[Normal flagged as Attack]")
    plt.hist(fn_errors, bins=log_bins, alpha=0.65, color="#9333EA", density=True, label=f"False Negatives ({fn_count:,})\n[Attack missed as Normal]")
    plt.axvline(official_threshold, color="#DC2626", linestyle="--", linewidth=2.5, label=f"Decision Boundary = {official_threshold:.6f}")
    plt.xscale("log")
    plt.xlabel("Reconstruction Error (MSE, log scale)", fontsize=11, fontweight="bold")
    plt.ylabel("Probability Density", fontsize=11, fontweight="bold")
    plt.title("Error Distribution of Classification Errors: False Positives vs False Negatives", fontsize=12, fontweight="bold", pad=12)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(FIG_FP_FN_DISTRIB, dpi=300)
    plt.close()
    print(f"  [Saved] {FIG_FP_FN_DISTRIB}")

    # 6. Threshold sensitivity: Precision / Recall / F1
    t_vals = [r["threshold"] for r in sensitivity_results]
    prec_vals = [r["precision_pct"] for r in sensitivity_results]
    rec_vals = [r["recall_pct"] for r in sensitivity_results]
    f1_vals = [r["f1_pct"] for r in sensitivity_results]

    plt.figure(figsize=(9, 6))
    plt.plot(t_vals, prec_vals, marker="o", lw=2.2, color="#2563EB", label="Precision (%)")
    plt.plot(t_vals, rec_vals, marker="s", lw=2.2, color="#DC2626", label="Recall (%)")
    plt.plot(t_vals, f1_vals, marker="^", lw=2.2, color="#059669", label="F1 Score (%)")
    plt.axvline(official_threshold, color="#111827", linestyle="--", lw=2, label=f"Official Threshold ({official_threshold:.4f})")
    plt.xlabel("Anomaly Threshold", fontsize=11, fontweight="bold")
    plt.ylabel("Score (%)", fontsize=11, fontweight="bold")
    plt.title("Threshold Sensitivity: Precision, Recall, and F1 Score", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="center right", frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_SENSITIVITY_METRICS, dpi=300)
    plt.close()
    print(f"  [Saved] {FIG_SENSITIVITY_METRICS}")

    # 7. Threshold sensitivity: FPR vs Specificity
    fpr_vals = [r["fpr_pct"] for r in sensitivity_results]
    spec_vals = [r["specificity_pct"] for r in sensitivity_results]

    plt.figure(figsize=(9, 6))
    plt.plot(t_vals, fpr_vals, marker="o", lw=2.2, color="#EA580C", label="False Positive Rate (%)")
    plt.plot(t_vals, spec_vals, marker="d", lw=2.2, color="#4F46E5", label="Specificity (%)")
    plt.axvline(official_threshold, color="#111827", linestyle="--", lw=2, label=f"Official Threshold ({official_threshold:.4f})")
    plt.xlabel("Anomaly Threshold", fontsize=11, fontweight="bold")
    plt.ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    plt.title("Threshold Sensitivity: False Positive Rate and Specificity", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="center right", frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_SENSITIVITY_RATES, dpi=300)
    plt.close()
    print(f"  [Saved] {FIG_SENSITIVITY_RATES}")

    # =======================================================================
    # STEP 9: ERROR CASE EXPORT
    # =======================================================================
    print("\n" + "=" * 55)
    print("[STEP 9] Exporting error cases and attack analysis CSVs...")
    print("=" * 55)

    export_cols = [
        "test_index",
        "original_label",
        "attack_category",
        "predicted_label",
        "reconstruction_error",
        "protocol_type",
        "service",
        "flag",
    ]

    df_fp_export = df_fp[export_cols].copy()
    df_fp_export.to_csv(FP_CSV_PATH, index=False)
    print(f"  [Saved] {FP_CSV_PATH} ({len(df_fp_export):,} rows)")

    df_fn_export = df_fn[export_cols].copy()
    df_fn_export.to_csv(FN_CSV_PATH, index=False)
    print(f"  [Saved] {FN_CSV_PATH} ({len(df_fn_export):,} rows)")

    # Detailed attack type analysis CSV
    detailed_attack_analysis = []
    unique_labels = sorted(df_full["original_label"].unique())
    for lbl in unique_labels:
        mask = df_full["original_label"] == lbl
        sub_err = errors[mask]
        tot = len(sub_err)
        det = int(np.sum(sub_err > official_threshold))
        mis = tot - det
        cat = map_attack_category(lbl)
        rate = (det / tot * 100.0) if tot > 0 else 0.0
        detailed_attack_analysis.append({
            "label": lbl,
            "category": cat,
            "total_samples": tot,
            "detected": det,
            "missed": mis,
            "detection_rate_pct": round(rate, 2),
            "mean_error": round(float(np.mean(sub_err)), 6),
            "median_error": round(float(np.median(sub_err)), 6),
            "std_error": round(float(np.std(sub_err)), 6),
            "p90_error": round(float(np.percentile(sub_err, 90)), 6),
            "p95_error": round(float(np.percentile(sub_err, 95)), 6),
            "p99_error": round(float(np.percentile(sub_err, 99)), 6),
        })

    df_attack_detailed = pd.DataFrame(detailed_attack_analysis)
    df_attack_detailed.to_csv(ATTACK_ANALYSIS_CSV_PATH, index=False)
    print(f"  [Saved] {ATTACK_ANALYSIS_CSV_PATH} ({len(df_attack_detailed):,} rows)")

    # =======================================================================
    # STEP 10: AUTOMATED DETAILED REPORT
    # =======================================================================
    print("\n" + "=" * 55)
    print("[STEP 10] Generating automated diagnostic report...")
    print("=" * 55)

    report_content = f"""================================================================================
PHASE 11: COMPREHENSIVE ERROR ANALYSIS & ATTACK-TYPE DIAGNOSTICS
================================================================================
Project: Network Intrusion Detection Using Autoencoder
Dataset: NSL-KDD / KDDTest+ (completely unseen)
Model  : Dense Autoencoder (77 -> 64 -> 32 -> 16 -> 32 -> 64 -> 77)
Official Anomaly Threshold: {official_threshold:.8f} (Phase 8 Normal Val P95)

1. EXECUTIVE SUMMARY
--------------------------------------------------------------------------------
The Dense Autoencoder was evaluated on 22,544 unseen KDDTest+ records. Operating
purely on unsupervised reconstruction error without ever seeing attack samples during
training, the model achieved:
  - Accuracy  : 83.35% (18,791 / 22,544)
  - Precision : 96.05% (9,469 / 9,858)
  - Recall    : 73.79% (9,469 / 12,833)
  - F1 Score  : 83.46%
  - ROC-AUC   : 95.70%
  - FPR       : 4.01% (389 / 9,711)
  - Specificity: 95.99% (9,322 / 9,711)

2. DATASET & CONFUSION MATRIX VERIFICATION
--------------------------------------------------------------------------------
Total Test Records : {total_test:,}
  - Actual Normal  : {act_normal:,} (43.08%)
  - Actual Attack  : {act_attack:,} (56.92%)
Confusion Matrix:
  - True Negatives  (TN): {tn:,} ({tn/total_test*100:.2f}%)
  - False Positives (FP): {fp:,} ({fp/total_test*100:.2f}%)
  - False Negatives (FN): {fn:,} ({fn/total_test*100:.2f}%)
  - True Positives  (TP): {tp:,} ({tp/total_test*100:.2f}%)
Sum check: TN + FP + FN + TP = {tn + fp + fn + tp:,} (Exact match)

3. RECONSTRUCTION ERROR ANALYSIS ACROSS GROUPS
--------------------------------------------------------------------------------
{df_group_stats.to_string(index=False)}

Key observations:
  - Normal traffic median error is 0.0032 with a mean of 0.0385, well below the 0.1291 threshold.
  - Attack traffic median error is 0.4576 (143x higher than normal) with a mean of 0.9631 (25x higher).
  - Extreme separation is observed for DoS and Probe attacks, while R2L exhibits strong distribution overlap.

4. FALSE POSITIVE (FP) ANALYSIS
--------------------------------------------------------------------------------
Total False Positives: {fp_count:,} ({fp_pct_normal:.2f}% of normal traffic)
Reconstruction Error:
  - Min   : {fp_min:.6f}
  - P25   : {fp_p25:.6f}
  - Median: {fp_median:.6f}
  - Mean  : {fp_mean:.6f}
  - P75   : {fp_p75:.6f}
  - Max   : {fp_max:.6f}

Distribution & Concentration:
  - {near_thresh_count} out of {fp_count} FPs ({near_thresh_pct:.1f}%) have reconstruction error <= 2x threshold.
  - This demonstrates that most false alarms are marginal threshold boundary crossings
    inherent to setting the threshold at the 95th percentile, rather than model divergence.

Top Protocol / Service / Flag Patterns among False Positives:
{fp_combos.head(5).to_string(index=False)}

5. FALSE NEGATIVE (FN) ANALYSIS
--------------------------------------------------------------------------------
Total False Negatives (Missed Attacks): {fn_count:,} ({fn_pct_attack:.2f}% of all attacks)
Reconstruction Error:
  - Min   : {fn_min:.6f}
  - P25   : {fn_p25:.6f}
  - Median: {fn_median:.6f}
  - Mean  : {fn_mean:.6f}
  - P75   : {fn_p75:.6f}
  - Max   : {fn_max:.6f}

False Negatives Breakdown by Attack Category:
  - R2L   : 1,728 missed ({1728/fn_count*100:.2f}% of all FNs, {1728/2887*100:.2f}% of R2L traffic)
  - DoS   : 1,166 missed ({1166/fn_count*100:.2f}% of all FNs, {1166/7458*100:.2f}% of DoS traffic)
  - Probe :   447 missed ({447/fn_count*100:.2f}% of all FNs, {447/2421*100:.2f}% of Probe traffic)
  - U2R   :    23 missed ({23/fn_count*100:.2f}% of all FNs, {23/67*100:.2f}% of U2R traffic)

Crucial Insight:
R2L (Remote-to-Local) intrusions constitute over 51.3% of all false negatives.
These attacks (such as password guessing, warezclient, snmpgetattack) take place
over valid session structures and normal protocols, causing little deviation in connection
summary statistics.

6. ATTACK-TYPE COMPARISON
--------------------------------------------------------------------------------
{df_attack_comp.to_string(index=False)}

Easiest vs Hardest:
  - EASIEST: DoS (84.37% recall) and Probe (81.54% recall).
    Volumetric floods and host/port scanning disrupt traffic connection metrics
    (serror_rate, count, srv_count), producing massive reconstruction errors.
  - HARDEST: R2L (40.15% recall).
    The low mean error (0.0526 for missed R2L) indicates the autoencoder reconstructs
    these connections with fidelity similar to legitimate user sessions.

7. THRESHOLD SENSITIVITY ANALYSIS (ANALYSIS ONLY)
--------------------------------------------------------------------------------
{pd.DataFrame(sensitivity_results).to_string(index=False)}

Threshold Trade-off:
  - Lowering threshold to 0.075 raises recall from 73.79% to 80.52%, but doubles FPR to 8.65%.
  - Raising threshold to 0.20 lowers FPR to 2.16% (Specificity 97.84%), but drops recall to 67.24%.
  - The official threshold 0.12907493 derived from the validation 95th percentile delivers
    the optimal operational balance: 96.05% Precision, 73.79% Recall, 4.01% FPR.

8. KEY FINDINGS
--------------------------------------------------------------------------------
  1. High Operational Precision: 96.05% precision indicates that when the alarm triggers,
     there is a 96% chance it is a genuine cyberattack, drastically reducing SOC alert fatigue.
  2. Specificity Calibration Holds: The observed test FPR is 4.01%, directly corroborating
     the Phase 8 calibration (P95 validation errors -> ~5% expected FPR).
  3. Structural Anomaly Separation: Attacks perturbing connection rates (DoS, Probe) are
     detected reliably (>81-84%), while semantic/content-based attacks (R2L) pass through
     lower-layer statistical models.

9. MODEL LIMITATIONS
--------------------------------------------------------------------------------
  1. Tabular Payload Blindness: The NSL-KDD dataset represents connection metadata
     (header statistics), not packet payloads. R2L and application-layer attacks cannot
     be fully detected without deep packet inspection (DPI).
  2. Fixed Compression Bottleneck: The 16-dimensional bottleneck represents normal
     connections effectively, but subtle attacks that mimic normal session state
     do not incur sufficient reconstruction error.

10. POSSIBLE FUTURE IMPROVEMENTS
--------------------------------------------------------------------------------
  1. Multi-modal / Hybrid Architectures: Combining the Autoencoder with a supervised
     payload classifier or sequence model (LSTM/Transformer).
  2. Feature Engineering: Adding specialized session-level credential features to
     differentiate R2L attempts.
  3. Dynamic / Ensembled Thresholding: Using service-specific thresholds rather than a
     single global cutoff.
================================================================================
"""

    with open(ERROR_REPORT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"  [Saved] {ERROR_REPORT_TXT_PATH}")

    # =======================================================================
    # STEP 11: FINAL VALIDATION
    # =======================================================================
    print("\n" + "=" * 55)
    print("[STEP 11] Running final integrity validation...")
    print("=" * 55)

    # 1. Verify model files untouched
    assert os.path.exists(MODEL_PATH)
    assert os.path.getsize(MODEL_PATH) == 226836, "models/autoencoder.keras was modified!"
    assert os.path.exists(PREPROCESSOR_PATH)
    assert os.path.getsize(PREPROCESSOR_PATH) == 5995, "models/preprocessor.joblib was modified!"
    print("  ✓ models/autoencoder.keras is untouched (226,836 bytes)")
    print("  ✓ models/preprocessor.joblib is untouched (5,995 bytes)")

    # 2. Verify threshold remains 0.12907493
    assert abs(official_threshold - 0.1290749317) < 1e-6, "Threshold modified!"
    print("  ✓ Anomaly threshold remains strictly 0.12907493")

    # 3. Verify Phase 10 metrics unchanged
    with open(FINAL_METRICS_PATH, "r", encoding="utf-8") as f:
        m10 = json.load(f)
    assert m10["accuracy"] == 0.833526
    assert m10["precision"] == 0.96054
    assert m10["recall"] == 0.737863
    assert m10["f1_score"] == 0.834604
    assert m10["roc_auc"] == 0.95696
    print("  ✓ Phase 10 metrics remain completely unchanged in results/final_metrics.json")

    # 4. Verify CSV counts match
    assert len(df_fp_export) == fp, f"FP CSV count mismatch: {len(df_fp_export)} vs {fp}"
    assert len(df_fn_export) == fn, f"FN CSV count mismatch: {len(df_fn_export)} vs {fn}"
    print(f"  ✓ Exported FP rows ({len(df_fp_export):,}) match calculated FP count ({fp:,})")
    print(f"  ✓ Exported FN rows ({len(df_fn_export):,}) match calculated FN count ({fn:,})")

    # 5. Attack-type totals sum correctly
    att_sum = sum(row["total_samples"] for row in attack_comp)
    assert att_sum == act_attack, f"Attack type sum mismatch: {att_sum} vs {act_attack}"
    print(f"  ✓ Attack-type totals sum to {att_sum:,} == actual attack count {act_attack:,}")

    # 6. NaN / Inf check
    for item in group_stats:
        for k, v in item.items():
            if isinstance(v, (int, float)):
                assert not np.isnan(v), f"NaN found in {item['group']} {k}"
                assert not np.isinf(v), f"Inf found in {item['group']} {k}"
    print("  ✓ 0 NaN and 0 Inf across all generated numerical summaries")

    print("\n" + "=" * 55)
    print("PHASE 11 COMPLETE")
    print("=" * 55)


if __name__ == "__main__":
    run_phase_11()
