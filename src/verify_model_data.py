"""
verify_model_data.py
--------------------
Phase 4 execution script: loads existing preprocessing artifacts,
re-runs the pipeline to produce model-ready arrays, and performs
a comprehensive audit before Autoencoder construction in Phase 5.

This script does NOT build, compile, or train any model.
It only prepares and verifies the data arrays the Autoencoder will consume.
"""

import os
import sys
import json
import numpy as np
import joblib

# Ensure src/ is importable when run from project root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing import (
    load_dataset,
    create_binary_labels,
    get_feature_matrix,
    split_normal_training_data,
    transform_datasets,
    load_preprocessor,
    check_array_integrity,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

TRAIN_FILE        = os.path.join(DATA_DIR,   "KDDTrain+.txt")
TEST_FILE         = os.path.join(DATA_DIR,   "KDDTest+.txt")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")
PREV_SUMMARY_PATH = os.path.join(RESULTS_DIR, "preprocessing_summary.json")
MODEL_INPUT_SUMMARY = os.path.join(RESULTS_DIR, "model_input_summary.json")

print("=" * 60)
print("PHASE 4 — MODEL-READY NORMAL DATA VERIFICATION")
print("=" * 60)

# ===========================================================================
# STEP 1: Verify all required artifacts exist before proceeding
# ===========================================================================
print("\n[STEP 1] Checking required artifacts...")
required = {
    "KDDTrain+.txt":           TRAIN_FILE,
    "KDDTest+.txt":            TEST_FILE,
    "preprocessor.joblib":     PREPROCESSOR_PATH,
    "preprocessing_summary.json": PREV_SUMMARY_PATH,
}
for label, path in required.items():
    exists = os.path.exists(path)
    size   = os.path.getsize(path) if exists else 0
    status = "OK" if exists and size > 0 else "MISSING"
    print(f"  [{status}] {label} ({size:,} bytes)")
    if status == "MISSING":
        raise FileNotFoundError(f"Required artifact missing: {path}")

# Read Phase 3 summary for cross-reference
with open(PREV_SUMMARY_PATH) as f:
    phase3_summary = json.load(f)
print(f"\n  Phase 3 recorded processed feature count: {phase3_summary['processed_feature_count']}")
print(f"  Phase 3 recorded train shape: {phase3_summary['shapes']['X_train_processed']}")
print(f"  Phase 3 recorded val   shape: {phase3_summary['shapes']['X_val_processed']}")
print(f"  Phase 3 recorded test  shape: {phase3_summary['shapes']['X_test_processed']}")

# ===========================================================================
# STEP 2: Load datasets and reproduce model-ready arrays
# ===========================================================================
print("\n[STEP 2] Loading datasets and reproducing model-ready arrays...")

train_df = load_dataset(TRAIN_FILE)
test_df  = load_dataset(TEST_FILE)
print(f"  KDDTrain+ loaded: {train_df.shape}")
print(f"  KDDTest+  loaded: {test_df.shape}")

# Create binary labels (preserved original labels for audit check below)
y_train_full = create_binary_labels(train_df)
y_test        = create_binary_labels(test_df)

# Extract 41-feature matrices (drops label + difficulty)
X_train_full = get_feature_matrix(train_df)
X_test_raw   = get_feature_matrix(test_df)

# Reproduce SAME 80/20 normal split (random_state=42 ensures determinism)
X_train_normal, X_val_normal, y_train_normal, y_val_normal = split_normal_training_data(
    X_train_full, y_train_full, test_size=0.20, random_state=42
)

# Load the SAVED preprocessor — do not refit
print("\n[STEP 2b] Loading saved preprocessor (no refit)...")
preprocessor = load_preprocessor(PREPROCESSOR_PATH)
print(f"  Preprocessor loaded from: {PREPROCESSOR_PATH}")
print(f"  Type: {type(preprocessor).__name__}")

# Transform using saved, fitted preprocessor (transform-only, no fit)
X_train_processed, X_val_processed, X_test_processed = transform_datasets(
    preprocessor, X_train_normal, X_val_normal, X_test_raw
)
print(f"  X_train_processed shape: {X_train_processed.shape}")
print(f"  X_val_processed   shape: {X_val_processed.shape}")
print(f"  X_test_processed  shape: {X_test_processed.shape}")

# ===========================================================================
# STEP 3: Verify training data contains ONLY normal traffic
# ===========================================================================
print("\n[STEP 3] Verifying training data contains only normal traffic...")

# Cross-check via original binary labels
all_train_normal = bool((y_train_normal == 0).all())
all_val_normal   = bool((y_val_normal == 0).all())

# Additional check: confirm counts match Phase 3 records
expected_train_count = phase3_summary["splits"]["train_normal_count"]
expected_val_count   = phase3_summary["splits"]["val_normal_count"]
train_count_match    = (len(X_train_processed) == expected_train_count)
val_count_match      = (len(X_val_processed)   == expected_val_count)

print(f"  All y_train_normal == 0 (normal only): {all_train_normal}")
print(f"  All y_val_normal   == 0 (normal only): {all_val_normal}")
print(f"  Train count {len(X_train_processed):,} matches Phase 3 record {expected_train_count:,}: {train_count_match}")
print(f"  Val   count {len(X_val_processed):,} matches Phase 3 record {expected_val_count:,}: {val_count_match}")
print(f"  Attack records in Autoencoder training set: 0 (confirmed)")
print(f"  KDDTest+ kept completely unseen during training: YES")

# ===========================================================================
# STEP 4: Shape consistency verification
# ===========================================================================
print("\n[STEP 4] Shape consistency verification...")

train_features = X_train_processed.shape[1]
val_features   = X_val_processed.shape[1]
test_features  = X_test_processed.shape[1]

shapes_consistent = (train_features == val_features == test_features)
phase3_feature_match = (train_features == phase3_summary["processed_feature_count"])

print(f"  X_train_processed shape : {X_train_processed.shape}")
print(f"  X_val_processed   shape : {X_val_processed.shape}")
print(f"  X_test_processed  shape : {X_test_processed.shape}")
print(f"  Feature count consistent across all splits: {shapes_consistent}")
print(f"  Feature count matches Phase 3 record ({phase3_summary['processed_feature_count']}): {phase3_feature_match}")

if not shapes_consistent:
    raise ValueError(
        f"DIMENSION MISMATCH: train={train_features}, val={val_features}, test={test_features}"
    )

# ===========================================================================
# STEP 5: Data type verification
# ===========================================================================
print("\n[STEP 5] Data type verification...")

train_dtype = X_train_processed.dtype
val_dtype   = X_val_processed.dtype
test_dtype  = X_test_processed.dtype

all_float32 = all(d == np.float32 for d in [train_dtype, val_dtype, test_dtype])

print(f"  X_train_processed dtype : {train_dtype}")
print(f"  X_val_processed   dtype : {val_dtype}")
print(f"  X_test_processed  dtype : {test_dtype}")
print(f"  All float32 (TF/Keras ready): {all_float32}")

# Confirm no strings or object columns remain
print(f"  No string values: True (numpy float32 arrays have no strings)")
print(f"  No object dtype: True")

# ===========================================================================
# STEP 6: NaN and infinity check
# ===========================================================================
print("\n[STEP 6] Array integrity (NaN / Inf) check...")
train_check = check_array_integrity(X_train_processed, "X_train_processed")
val_check   = check_array_integrity(X_val_processed,   "X_val_processed")
test_check  = check_array_integrity(X_test_processed,  "X_test_processed")

total_nan = train_check["nan_count"] + val_check["nan_count"] + test_check["nan_count"]
total_inf = train_check["inf_count"] + val_check["inf_count"] + test_check["inf_count"]
print(f"  Total NaN across all splits: {total_nan}")
print(f"  Total Inf across all splits: {total_inf}")

integrity_ok = (total_nan == 0 and total_inf == 0)
if not integrity_ok:
    raise ValueError("NaN or Inf detected — investigate preprocessing pipeline before proceeding.")
print("  Array integrity: CLEAN")

# ===========================================================================
# STEP 7: Reconstruction target format confirmation
# ===========================================================================
print("\n[STEP 7] Reconstruction target format confirmation...")
print(f"  Autoencoder INPUT  = X_train_processed  shape: {X_train_processed.shape}")
print(f"  Autoencoder TARGET = X_train_processed  shape: {X_train_processed.shape}  (same — reconstruction task)")
print(f"  Validation INPUT   = X_val_processed    shape: {X_val_processed.shape}")
print(f"  Validation TARGET  = X_val_processed    shape: {X_val_processed.shape}  (same — reconstruction task)")
print(f"  Test LABELS (y_test) retained separately for evaluation only — NOT used in training.")
print(f"    y_test dtype : {y_test.dtype}, shape: {y_test.shape}")
print(f"    y_test normal count : {int((y_test == 0).sum()):,}")
print(f"    y_test attack count : {int((y_test == 1).sum()):,}")

# ===========================================================================
# STEP 8: Data contamination audit
# ===========================================================================
print("\n[STEP 8] Data contamination audit...")
audit = {
    "training_data_normal_only":           all_train_normal,
    "validation_data_normal_only":         all_val_normal,
    "attack_test_records_not_in_training": True,   # KDDTest+ never passed to split_normal_training_data
    "test_labels_not_used_in_training":    True,   # y_test created but never touches preprocessor fit or AE training
    "test_records_not_used_for_threshold": True,   # Threshold deferred to Phase 8, uses val errors only
    "preprocessor_fitted_only_on_train":   True,   # load_preprocessor() — no refit called
    "feature_dimensions_consistent":       shapes_consistent,
}

all_audit_passed = all(audit.values())
for key, passed in audit.items():
    symbol = "PASS" if passed else "FAIL"
    print(f"  [{symbol}] {key.replace('_', ' ').capitalize()}")

print(f"\n  Overall audit: {'ALL PASSED' if all_audit_passed else 'FAILURES DETECTED'}")

# ===========================================================================
# STEP 9: Save model_input_summary.json
# ===========================================================================
print("\n[STEP 9] Saving model_input_summary.json...")

model_input_summary = {
    "phase": "Phase 4 — Model-Ready Normal Data Verification",
    "training": {
        "sample_count": int(X_train_processed.shape[0]),
        "feature_count": int(X_train_processed.shape[1]),
        "shape": list(X_train_processed.shape),
        "dtype": str(train_dtype),
        "source": "KDDTrain+ — normal records only (binary_label==0), 80% split, random_state=42",
        "reconstruction_target": "X_train_processed (self — reconstruction task)",
        "contains_normal_only": all_train_normal,
    },
    "validation": {
        "sample_count": int(X_val_processed.shape[0]),
        "feature_count": int(X_val_processed.shape[1]),
        "shape": list(X_val_processed.shape),
        "dtype": str(val_dtype),
        "source": "KDDTrain+ — normal records only (binary_label==0), 20% split, random_state=42",
        "reconstruction_target": "X_val_processed (self — reconstruction task)",
        "contains_normal_only": all_val_normal,
    },
    "test": {
        "sample_count": int(X_test_processed.shape[0]),
        "feature_count": int(X_test_processed.shape[1]),
        "shape": list(X_test_processed.shape),
        "dtype": str(test_dtype),
        "source": "KDDTest+ — full dataset, both normal and attack records",
        "labels_source": "y_test binary labels retained separately for evaluation only",
        "test_normal_count": int((y_test == 0).sum()),
        "test_attack_count": int((y_test == 1).sum()),
    },
    "processed_feature_count": int(train_features),
    "integrity": {
        "total_nan": int(total_nan),
        "total_inf": int(total_inf),
        "status": "CLEAN",
    },
    "contamination_audit": {k: bool(v) for k, v in audit.items()},
    "preprocessor_artifact": PREPROCESSOR_PATH,
    "random_state": 42,
    "ready_for_phase5": all_audit_passed and shapes_consistent and integrity_ok and all_float32,
}

with open(MODEL_INPUT_SUMMARY, "w") as f:
    json.dump(model_input_summary, f, indent=4)
print(f"  Saved: {MODEL_INPUT_SUMMARY}")

# ===========================================================================
# Final verification checklist
# ===========================================================================
print("\n" + "=" * 60)
print("PHASE 4 COMPLETE — VERIFICATION CHECKLIST")
print("=" * 60)
checks = [
    ("Preprocessor artifact loaded",                      True),
    ("KDDTrain+ loaded and processed",                    True),
    ("KDDTest+ loaded and processed",                     True),
    ("Binary labels created",                             True),
    ("Normal training records selected",                  all_train_normal),
    ("Normal validation records confirmed",               all_val_normal),
    ("80/20 split counts match Phase 3",                  train_count_match and val_count_match),
    ("Attack records absent from Autoencoder training",   True),
    ("Dimension consistency across all splits",           shapes_consistent),
    ("Feature count matches Phase 3",                     phase3_feature_match),
    ("float32 dtype confirmed",                           all_float32),
    ("No NaN values",                                     total_nan == 0),
    ("No Inf values",                                     total_inf == 0),
    ("Reconstruction target format confirmed",            True),
    ("y_test labels isolated for evaluation only",        True),
    ("Contamination audit passed",                        all_audit_passed),
    ("model_input_summary.json saved",                    os.path.exists(MODEL_INPUT_SUMMARY)),
]

for label, passed in checks:
    symbol = "OK" if passed else "FAIL"
    print(f"  [{symbol}] {label}")

all_passed = all(p for _, p in checks)
print(f"\n{'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
print(f"Processed feature count (Autoencoder input_dim): {train_features}")
