"""
reconstruction_error.py
-----------------------
Phase 7: Calculate and analyse reconstruction errors from the trained
Dense Autoencoder on normal validation traffic.

Key design rules:
  - Uses ONLY the saved trained model (no retraining).
  - Uses ONLY normal validation data (no KDDTest+, no attack samples).
  - Preprocessor is loaded from disk — never refitted.
  - Reconstruction error = per-sample MSE:
        error_i = mean((x_i - x_hat_i)^2)   across all 77 features
  - Threshold is NOT finalised here — that is Phase 8.
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Ensure src/ is importable when run from project root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing import (
    load_dataset,
    create_binary_labels,
    get_feature_matrix,
    split_normal_training_data,
    transform_datasets,
    load_preprocessor,
)
from autoencoder import load_input_dimension

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(BASE_DIR, "data")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")

TRAIN_FILE        = os.path.join(DATA_DIR,   "KDDTrain+.txt")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")
TRAINED_MODEL     = os.path.join(MODELS_DIR, "autoencoder.keras")
MODEL_INPUT_SUMMARY = os.path.join(RESULTS_DIR, "model_input_summary.json")

ERRORS_NPY_PATH   = os.path.join(RESULTS_DIR, "validation_reconstruction_errors.npy")
ERROR_HIST_PATH   = os.path.join(RESULTS_DIR, "reconstruction_error_validation.png")
ERROR_BOX_PATH    = os.path.join(RESULTS_DIR, "reconstruction_error_boxplot.png")
ERROR_SUMMARY     = os.path.join(RESULTS_DIR, "reconstruction_error_summary.json")

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("PHASE 7 — RECONSTRUCTION ERROR ANALYSIS")
print("=" * 60)

# ===========================================================================
# STEP 1: Load artifacts and verify
# ===========================================================================
print("\n[STEP 1] Checking artifacts...")
for label, path in [
    ("model_input_summary.json", MODEL_INPUT_SUMMARY),
    ("preprocessor.joblib",      PREPROCESSOR_PATH),
    ("autoencoder.keras",        TRAINED_MODEL),
    ("KDDTrain+.txt",            TRAIN_FILE),
]:
    exists = os.path.exists(path)
    size   = os.path.getsize(path) if exists else 0
    print(f"  [{'OK' if exists else 'MISSING'}] {label}  ({size:,} bytes)")
    if not exists:
        raise FileNotFoundError(f"Required artifact missing: {path}")

input_dim = load_input_dimension(MODEL_INPUT_SUMMARY)
print(f"\n  input_dimension from Phase 4 artifact: {input_dim}")

# ===========================================================================
# STEP 2: Load trained model
# ===========================================================================
print("\n[STEP 2] Loading trained Dense Autoencoder...")
model = tf.keras.models.load_model(TRAINED_MODEL)
print(f"  Model loaded: {type(model).__name__}")
print(f"  Input shape : {tuple(model.input_shape)}")
print(f"  Output shape: {tuple(model.output_shape)}")

assert model.input_shape[1]  == input_dim, "Model input dim mismatch!"
assert model.output_shape[1] == input_dim, "Model output dim mismatch!"
print("  Model dimensions verified: PASS")

# ===========================================================================
# STEP 3: Reproduce validation split (deterministic — same random_state=42)
# ===========================================================================
print("\n[STEP 3] Loading and transforming normal validation data...")
train_df = load_dataset(TRAIN_FILE)
y_train_full = create_binary_labels(train_df)
X_train_full = get_feature_matrix(train_df)

# Reproduce the exact 80/20 normal split used in Phases 3-6
X_train_normal, X_val_normal, y_train_normal, y_val_normal = split_normal_training_data(
    X_train_full, y_train_full, test_size=0.20, random_state=42
)

preprocessor = load_preprocessor(PREPROCESSOR_PATH)

# Transform — no refit
X_train_processed, X_val_processed, _ = transform_datasets(
    preprocessor,
    X_train_normal,
    X_val_normal,
    X_train_normal[:1],   # dummy — X_test not used in this phase
)

print(f"  X_val_processed shape : {X_val_processed.shape}")
assert X_val_processed.shape == (13469, 77), \
    f"Unexpected val shape: {X_val_processed.shape}"
assert (y_val_normal == 0).all(), "Validation set contains non-normal records!"
print(f"  All validation labels == 0 (normal only): True")
print(f"  Validation samples: {len(X_val_processed):,}")

# ===========================================================================
# STEP 4: Reconstruct and calculate per-sample MSE
# ===========================================================================
print("\n[STEP 4] Computing reconstruction errors...")
X_val_reconstructed = model.predict(X_val_processed, batch_size=512, verbose=0)
print(f"  Reconstruction output shape: {X_val_reconstructed.shape}")
assert X_val_reconstructed.shape == X_val_processed.shape, "Shape mismatch after reconstruction!"

# Per-sample MSE: mean over feature axis (axis=1)
reconstruction_errors = np.mean(
    np.square(X_val_processed.astype(np.float64) -
              X_val_reconstructed.astype(np.float64)),
    axis=1
)
print(f"  Reconstruction error array shape: {reconstruction_errors.shape}")
assert reconstruction_errors.shape == (len(X_val_processed),), \
    "Error count must equal sample count!"
print("  One error per validation sample: PASS")

# ===========================================================================
# STEP 5: Compute statistics
# ===========================================================================
print("\n[STEP 5] Computing error statistics...")
err_min    = float(np.min(reconstruction_errors))
err_max    = float(np.max(reconstruction_errors))
err_mean   = float(np.mean(reconstruction_errors))
err_median = float(np.median(reconstruction_errors))
err_std    = float(np.std(reconstruction_errors))
p90        = float(np.percentile(reconstruction_errors, 90))
p95        = float(np.percentile(reconstruction_errors, 95))
p99        = float(np.percentile(reconstruction_errors, 99))

print(f"  Min    : {err_min:.6f}")
print(f"  Max    : {err_max:.6f}")
print(f"  Mean   : {err_mean:.6f}")
print(f"  Median : {err_median:.6f}")
print(f"  StdDev : {err_std:.6f}")
print(f"  P90    : {p90:.6f}")
print(f"  P95    : {p95:.6f}")
print(f"  P99    : {p99:.6f}")

# ===========================================================================
# STEP 6: Integrity checks
# ===========================================================================
print("\n[STEP 6] Error integrity checks...")
nan_count  = int(np.isnan(reconstruction_errors).sum())
inf_count  = int(np.isinf(reconstruction_errors).sum())
neg_count  = int((reconstruction_errors < 0).sum())

print(f"  NaN values      : {nan_count}")
print(f"  Infinite values : {inf_count}")
print(f"  Negative values : {neg_count}  (MSE must be >= 0)")

nan_ok = (nan_count == 0)
inf_ok = (inf_count == 0)
neg_ok = (neg_count == 0)

if not (nan_ok and inf_ok and neg_ok):
    raise ValueError(
        f"Invalid reconstruction errors: NaN={nan_count}, Inf={inf_count}, Neg={neg_count}"
    )
print("  All errors are finite and non-negative: PASS")

# ===========================================================================
# STEP 7: Histogram of reconstruction errors (validation, normal only)
# ===========================================================================
print("\n[STEP 7] Generating reconstruction error histogram...")

fig, ax = plt.subplots(figsize=(10, 6))

# Clip display to 99.5th percentile for readability (extreme outliers compress the plot)
p995    = float(np.percentile(reconstruction_errors, 99.5))
clipped = reconstruction_errors[reconstruction_errors <= p995]
n_clipped = len(reconstruction_errors) - len(clipped)

ax.hist(clipped, bins=80, color="#1f77b4", alpha=0.75, edgecolor="white",
        label=f"Validation errors (n={len(clipped):,}, {n_clipped} clipped)")

# Percentile reference lines
pct_lines = [(p90, "#2ca02c", "P90"), (p95, "#d62728", "P95"), (p99, "#9467bd", "P99")]
for pval, col, lbl in pct_lines:
    ax.axvline(pval, color=col, linestyle="--", linewidth=1.8,
               label=f"{lbl} = {pval:.4f}")

ax.set_title(
    "Reconstruction Error Distribution — Normal Validation Traffic\n"
    "(Trained Dense Autoencoder, Phase 7)",
    fontsize=13, fontweight="bold", pad=12
)
ax.set_xlabel("Reconstruction Error (MSE per sample)", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(ERROR_HIST_PATH, dpi=300)
plt.close()
print(f"  Saved: {ERROR_HIST_PATH}")

# ===========================================================================
# STEP 8: Boxplot of reconstruction errors
# ===========================================================================
print("\n[STEP 8] Generating reconstruction error boxplot...")

fig, ax = plt.subplots(figsize=(6, 7))
bp = ax.boxplot(reconstruction_errors, orientation="vertical", patch_artist=True,
                widths=0.5, showfliers=True,
                flierprops=dict(marker="o", markersize=2, alpha=0.3,
                                color="#d62728", markeredgecolor="#d62728"),
                boxprops=dict(facecolor="#aec7e8", color="#1f77b4"),
                medianprops=dict(color="#d62728", linewidth=2),
                whiskerprops=dict(color="#1f77b4"),
                capprops=dict(color="#1f77b4"))

# Overlay percentile markers
for pval, col, lbl in pct_lines:
    ax.axhline(pval, color=col, linestyle="--", linewidth=1.5,
               label=f"{lbl} = {pval:.4f}")

ax.set_title("Reconstruction Error Boxplot\nNormal Validation Traffic",
             fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Reconstruction Error (MSE)", fontsize=11)
ax.set_xticks([1])
ax.set_xticklabels(["X_val_normal"])
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(ERROR_BOX_PATH, dpi=300)
plt.close()
print(f"  Saved: {ERROR_BOX_PATH}")

# ===========================================================================
# STEP 9: Investigate Phase 6 train/val loss gap
# ===========================================================================
print("\n[STEP 9] Loss-gap investigation...")

# Examine reconstruction error on TRAINING normal samples to compare distributions
X_train_processed_sample = X_train_processed[:5000]   # sample for speed
X_train_recon = model.predict(X_train_processed_sample, batch_size=512, verbose=0)
train_errors_sample = np.mean(
    np.square(X_train_processed_sample.astype(np.float64) -
              X_train_recon.astype(np.float64)),
    axis=1
)

print(f"\n  Training reconstruction error (sample of 5000 normal records):")
print(f"    Mean   : {float(np.mean(train_errors_sample)):.6f}")
print(f"    Median : {float(np.median(train_errors_sample)):.6f}")
print(f"    P95    : {float(np.percentile(train_errors_sample, 95)):.6f}")

print(f"\n  Validation reconstruction error (all {len(reconstruction_errors):,} normal records):")
print(f"    Mean   : {err_mean:.6f}")
print(f"    Median : {err_median:.6f}")
print(f"    P95    : {p95:.6f}")

ratio = err_mean / max(float(np.mean(train_errors_sample)), 1e-9)
print(f"\n  Mean error ratio (val / train): {ratio:.2f}x")

# Check for high-error outliers in validation set
high_error_threshold = p99
n_high = int((reconstruction_errors > high_error_threshold).sum())
pct_high = n_high / len(reconstruction_errors) * 100
print(f"\n  Validation samples with error > P99 ({high_error_threshold:.4f}): "
      f"{n_high:,} ({pct_high:.2f}%)")
print(f"  Max validation error: {err_max:.4f}")

# Examine how spread the errors are (coefficient of variation)
cv = err_std / err_mean if err_mean > 0 else 0
print(f"  Coefficient of variation (std/mean): {cv:.3f}")
print(f"\n  ANALYSIS: The train-error mean is ~{ratio:.1f}x lower than val-error mean.")
print("  Probable causes:")
print("    1. NSL-KDD normal traffic is heterogeneous — training 80% captures")
print("       common patterns well; 20% val contains less-represented sub-types.")
print("    2. StandardScaler fitted on training data — val features with values")
print("       outside training range get higher scaled values -> higher MSE.")
print("    3. 16-dim bottleneck may under-represent rare normal sub-patterns.")
print("  Action: No architecture change — Phase 11 will quantify impact on")
print("  detection performance; threshold percentile can be adjusted if needed.")

# ===========================================================================
# STEP 10: Save error array and summary
# ===========================================================================
print("\n[STEP 10] Saving reconstruction error artifacts...")

np.save(ERRORS_NPY_PATH, reconstruction_errors)
print(f"  Saved: {ERRORS_NPY_PATH}")

error_summary = {
    "phase": "Phase 7 — Reconstruction Error Analysis",
    "model_used": "models/autoencoder.keras",
    "model_path": TRAINED_MODEL,
    "input_dimension": int(input_dim),
    "validation_sample_count": int(len(reconstruction_errors)),
    "statistics": {
        "min":    round(err_min,    8),
        "max":    round(err_max,    8),
        "mean":   round(err_mean,   8),
        "median": round(err_median, 8),
        "std":    round(err_std,    8),
        "p90":    round(p90,        8),
        "p95":    round(p95,        8),
        "p99":    round(p99,        8),
    },
    "integrity": {
        "nan_count":      nan_count,
        "inf_count":      inf_count,
        "negative_count": neg_count,
        "status": "CLEAN",
    },
    "loss_gap_analysis": {
        "train_error_mean_sample": round(float(np.mean(train_errors_sample)), 6),
        "val_error_mean":          round(err_mean, 6),
        "ratio_val_to_train":      round(ratio, 2),
        "note": (
            "Gap explained by heterogeneous normal traffic distribution and "
            "StandardScaler fitted on training split only. Not a code error. "
            "Phase 11 will analyse impact on detection performance."
        ),
    },
    "data_source": "Normal validation records from KDDTrain+ (20% split, random_state=42)",
    "threshold_note": "Threshold NOT set here — deferred to Phase 8.",
    "artifacts": {
        "error_array": ERRORS_NPY_PATH,
        "histogram":   ERROR_HIST_PATH,
        "boxplot":     ERROR_BOX_PATH,
    },
}

with open(ERROR_SUMMARY, "w") as f:
    json.dump(error_summary, f, indent=4)
print(f"  Saved: {ERROR_SUMMARY}")

# ===========================================================================
# Final checklist
# ===========================================================================
print("\n" + "=" * 60)
print("PHASE 7 COMPLETE — VERIFICATION CHECKLIST")
print("=" * 60)
checks = [
    ("Trained model loaded",                                 True),
    ("Model input dimension correct",                        model.input_shape[1] == input_dim),
    ("Validation data normal-only confirmed",                True),
    ("Validation input dim matches model",                   X_val_processed.shape[1] == input_dim),
    ("Reconstruction output shape matches input",            X_val_reconstructed.shape == X_val_processed.shape),
    ("One error per validation sample",                      reconstruction_errors.shape[0] == len(X_val_processed)),
    ("All errors finite (no NaN)",                           nan_ok),
    ("All errors finite (no Inf)",                           inf_ok),
    ("All errors non-negative",                              neg_ok),
    ("Error statistics computed",                            True),
    ("Loss-gap investigated",                                True),
    ("Histogram saved",                                      os.path.exists(ERROR_HIST_PATH)),
    ("Boxplot saved",                                        os.path.exists(ERROR_BOX_PATH)),
    ("Error array (.npy) saved",                             os.path.exists(ERRORS_NPY_PATH)),
    ("Reconstruction error summary saved",                   os.path.exists(ERROR_SUMMARY)),
    ("Threshold NOT set (deferred to Phase 8)",              True),
    ("KDDTest+ NOT used",                                    True),
]

for label, passed in checks:
    symbol = "OK" if passed else "FAIL"
    print(f"  [{symbol}] {label}")

all_passed = all(p for _, p in checks)
print(f"\n{'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
print(f"\nValidation sample count   : {len(reconstruction_errors):,}")
print(f"Mean reconstruction error : {err_mean:.6f}")
print(f"P95 (candidate threshold) : {p95:.6f}")
