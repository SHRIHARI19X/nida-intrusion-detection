"""
preprocess_pipeline.py
----------------------
Phase 3 execution script: runs the complete leakage-free preprocessing
pipeline and produces verified, persistent artifacts ready for Phase 5.

Execution order:
  1. Load KDDTrain+ and KDDTest+
  2. Create binary labels
  3. Select normal training records
  4. Split normal records 80/20 (train / validation)
  5. Build ColumnTransformer (OneHotEncoder + StandardScaler)
  6. Fit preprocessor ONLY on X_train_normal
  7. Transform train, val, and test splits (no fit on val/test)
  8. Verify dimensions, NaN, Inf, and dtype
  9. Save preprocessor artifact
 10. Save preprocessing_summary.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# Ensure src/ is on the path when run from the project root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing import (
    # Schema
    TRAFFIC_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    METADATA_COLUMNS,
    # Phase 2 functions (preserved)
    load_dataset,
    # Phase 3 functions
    create_binary_labels,
    get_feature_matrix,
    split_normal_training_data,
    build_preprocessor,
    fit_preprocessor,
    transform_datasets,
    save_preprocessor,
    check_array_integrity,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(BASE_DIR, "data")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")

TRAIN_FILE   = os.path.join(DATA_DIR, "KDDTrain+.txt")
TEST_FILE    = os.path.join(DATA_DIR, "KDDTest+.txt")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "preprocessing_summary.json")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===========================================================================
print("=" * 60)
print("PHASE 3 — DATA PREPROCESSING PIPELINE")
print("=" * 60)

# ---------------------------------------------------------------------------
# Step 1: Load datasets
# ---------------------------------------------------------------------------
print("\n[STEP 1] Loading datasets...")
train_df = load_dataset(TRAIN_FILE)
test_df  = load_dataset(TEST_FILE)
print(f"  KDDTrain+ loaded: {train_df.shape}")
print(f"  KDDTest+  loaded: {test_df.shape}")

# ---------------------------------------------------------------------------
# Step 2: Feature type verification
# ---------------------------------------------------------------------------
print("\n[STEP 2] Feature type verification...")
print(f"  Total traffic features     : {len(TRAFFIC_FEATURES)}")
print(f"  Categorical features ({len(CATEGORICAL_FEATURES)})   : {CATEGORICAL_FEATURES}")
print(f"  Numerical features  ({len(NUMERICAL_FEATURES)})   : {NUMERICAL_FEATURES}")

# Programmatic dtype check to confirm categorical columns are object/string
for col in CATEGORICAL_FEATURES:
    dtype_str = str(train_df[col].dtype)
    print(f"    {col}: dtype={dtype_str}  (expected string/object)")
for col in NUMERICAL_FEATURES[:5]:
    dtype_str = str(train_df[col].dtype)
    print(f"    {col}: dtype={dtype_str}  (expected int/float)")

# ---------------------------------------------------------------------------
# Step 3: Create binary labels
# ---------------------------------------------------------------------------
print("\n[STEP 3] Creating binary labels...")
y_train_full = create_binary_labels(train_df)
y_test        = create_binary_labels(test_df)

train_normal_total = int((y_train_full == 0).sum())
train_attack_total = int((y_train_full == 1).sum())
test_normal_total  = int((y_test == 0).sum())
test_attack_total  = int((y_test == 1).sum())

print(f"  KDDTrain+ → binary_label 0 (normal): {train_normal_total:,}")
print(f"  KDDTrain+ → binary_label 1 (attack): {train_attack_total:,}")
print(f"  KDDTest+  → binary_label 0 (normal): {test_normal_total:,}")
print(f"  KDDTest+  → binary_label 1 (attack): {test_attack_total:,}")

# ---------------------------------------------------------------------------
# Step 4: Extract feature matrices (drops label and difficulty)
# ---------------------------------------------------------------------------
print("\n[STEP 4] Extracting 41-feature input matrices...")
X_train_full = get_feature_matrix(train_df)
X_test       = get_feature_matrix(test_df)
print(f"  X_train_full shape: {X_train_full.shape}  (label/difficulty excluded)")
print(f"  X_test       shape: {X_test.shape}")

# ---------------------------------------------------------------------------
# Step 5: Split normal training data 80/20
# ---------------------------------------------------------------------------
print("\n[STEP 5] Splitting normal KDDTrain+ records (80/20, random_state=42)...")
X_train_normal, X_val_normal, y_train_normal, y_val_normal = split_normal_training_data(
    X_train_full, y_train_full, test_size=0.20, random_state=42
)
print(f"  Normal training samples   (X_train_normal): {len(X_train_normal):,}")
print(f"  Normal validation samples (X_val_normal)  : {len(X_val_normal):,}")
print(f"  Confirm all y_train_normal == 0: {(y_train_normal == 0).all()}")
print(f"  Confirm all y_val_normal   == 0: {(y_val_normal == 0).all()}")

# ---------------------------------------------------------------------------
# Step 6: Build and fit the preprocessor
# ---------------------------------------------------------------------------
print("\n[STEP 6] Building and fitting ColumnTransformer...")
print("  Fitting ONLY on X_train_normal — no test or validation data used.")
preprocessor = build_preprocessor()
preprocessor  = fit_preprocessor(preprocessor, X_train_normal)
print("  ColumnTransformer fitted successfully.")

# Get processed feature count dynamically from the fitted transformer
processed_feature_count = preprocessor.transform(X_train_normal[:1]).shape[1]
print(f"  Processed feature count (from fitted transformer): {processed_feature_count}")

# Feature names breakdown
cat_transformer = preprocessor.named_transformers_["cat"]
ohe_feature_names = list(cat_transformer.get_feature_names_out(CATEGORICAL_FEATURES))
print(f"  OHE expanded categorical features: {len(ohe_feature_names)}")
print(f"    protocol_type variants : {[n for n in ohe_feature_names if n.startswith('protocol_type')]}")
print(f"    flag variants          : {[n for n in ohe_feature_names if n.startswith('flag')]}")
print(f"    service count          : {len([n for n in ohe_feature_names if n.startswith('service')])}")
print(f"  StandardScaler numerical features: {len(NUMERICAL_FEATURES)}")
print(f"  Total → OHE({len(ohe_feature_names)}) + Numerical({len(NUMERICAL_FEATURES)}) = {len(ohe_feature_names) + len(NUMERICAL_FEATURES)}")

# ---------------------------------------------------------------------------
# Step 7: Transform all splits
# ---------------------------------------------------------------------------
print("\n[STEP 7] Transforming all splits (transform only — no re-fitting)...")
X_train_processed, X_val_processed, X_test_processed = transform_datasets(
    preprocessor, X_train_normal, X_val_normal, X_test
)
print(f"  X_train_processed shape : {X_train_processed.shape}")
print(f"  X_val_processed   shape : {X_val_processed.shape}")
print(f"  X_test_processed  shape : {X_test_processed.shape}")

# ---------------------------------------------------------------------------
# Step 8: Dimension consistency check
# ---------------------------------------------------------------------------
print("\n[STEP 8] Dimension consistency check...")
assert X_train_processed.shape[1] == X_val_processed.shape[1] == X_test_processed.shape[1], (
    f"DIMENSION MISMATCH: train={X_train_processed.shape[1]}, "
    f"val={X_val_processed.shape[1]}, test={X_test_processed.shape[1]}"
)
print(f"  ✓ All three splits have identical feature count: {X_train_processed.shape[1]}")

# ---------------------------------------------------------------------------
# Step 9: NaN and Infinity checks
# ---------------------------------------------------------------------------
print("\n[STEP 9] Array integrity checks (NaN / Inf)...")
train_integrity = check_array_integrity(X_train_processed, "X_train_processed")
val_integrity   = check_array_integrity(X_val_processed,   "X_val_processed")
test_integrity  = check_array_integrity(X_test_processed,  "X_test_processed")

total_nan = train_integrity["nan_count"] + val_integrity["nan_count"] + test_integrity["nan_count"]
total_inf = train_integrity["inf_count"] + val_integrity["inf_count"] + test_integrity["inf_count"]
print(f"  Total NaN across all splits : {total_nan}")
print(f"  Total Inf across all splits : {total_inf}")

if total_nan > 0 or total_inf > 0:
    raise ValueError("CRITICAL: NaN or Inf values detected in processed arrays. Investigate preprocessing pipeline.")
print("  ✓ No NaN or Inf values detected.")

# ---------------------------------------------------------------------------
# Step 10: dtype check
# ---------------------------------------------------------------------------
print("\n[STEP 10] Data type verification...")
print(f"  X_train_processed dtype : {X_train_processed.dtype}")
print(f"  X_val_processed   dtype : {X_val_processed.dtype}")
print(f"  X_test_processed  dtype : {X_test_processed.dtype}")
assert X_train_processed.dtype == np.float32, "Expected float32 for TensorFlow compatibility"
print("  ✓ All arrays are float32 — TensorFlow/Keras compatible.")

# ---------------------------------------------------------------------------
# Step 11: Save preprocessor artifact
# ---------------------------------------------------------------------------
print("\n[STEP 11] Saving preprocessor artifact...")
save_preprocessor(preprocessor, PREPROCESSOR_PATH)
print(f"  ✓ Preprocessor saved: {PREPROCESSOR_PATH}")

# ---------------------------------------------------------------------------
# Step 12: Save preprocessing summary
# ---------------------------------------------------------------------------
print("\n[STEP 12] Saving preprocessing summary...")
summary = {
    "original_feature_count": len(TRAFFIC_FEATURES),
    "categorical_features": CATEGORICAL_FEATURES,
    "categorical_feature_count": len(CATEGORICAL_FEATURES),
    "numerical_feature_count": len(NUMERICAL_FEATURES),
    "ohe_expanded_feature_count": len(ohe_feature_names),
    "processed_feature_count": int(processed_feature_count),
    "splits": {
        "train_normal_count": int(len(X_train_normal)),
        "val_normal_count": int(len(X_val_normal)),
        "test_count": int(len(X_test)),
        "train_normal_pct": round(len(X_train_normal) / train_normal_total * 100, 2),
        "val_normal_pct": round(len(X_val_normal) / train_normal_total * 100, 2),
    },
    "shapes": {
        "X_train_processed": list(X_train_processed.shape),
        "X_val_processed": list(X_val_processed.shape),
        "X_test_processed": list(X_test_processed.shape),
    },
    "integrity": {
        "train_nan": train_integrity["nan_count"],
        "train_inf": train_integrity["inf_count"],
        "val_nan": val_integrity["nan_count"],
        "val_inf": val_integrity["inf_count"],
        "test_nan": test_integrity["nan_count"],
        "test_inf": test_integrity["inf_count"],
    },
    "dtype": str(X_train_processed.dtype),
    "preprocessor_artifact": PREPROCESSOR_PATH,
    "random_state": 42,
    "leakage_prevention": {
        "fit_data": "X_train_normal only (normal records from KDDTrain+, 80% split)",
        "val_transform": "transform() only — no fit()",
        "test_transform": "transform() only — no fit()",
        "threshold_source": "NOT YET — deferred to Phase 8 (validation errors only)",
    }
}

with open(SUMMARY_PATH, "w") as f:
    json.dump(summary, f, indent=4)
print(f"  ✓ Summary saved: {SUMMARY_PATH}")

# ---------------------------------------------------------------------------
# Final verification summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 3 COMPLETE — VERIFICATION CHECKLIST")
print("=" * 60)
checks = [
    ("KDDTrain+ loaded",                       True),
    ("KDDTest+ loaded",                        True),
    ("binary labels created",                  True),
    ("normal training records selected",       True),
    ("80/20 normal train-validation split",    True),
    ("ColumnTransformer built",                True),
    ("OneHotEncoder fitted (train only)",       True),
    ("StandardScaler fitted (train only)",     True),
    ("Training transformation",                True),
    ("Validation transformation",              True),
    ("Test transformation",                    True),
    ("Dimension consistency verified",         True),
    ("No NaN values",                          total_nan == 0),
    ("No Inf values",                          total_inf == 0),
    ("float32 dtype confirmed",                X_train_processed.dtype == np.float32),
    ("Preprocessor artifact saved",            os.path.exists(PREPROCESSOR_PATH)),
    ("Preprocessing summary saved",            os.path.exists(SUMMARY_PATH)),
]
for label, passed in checks:
    symbol = "✓" if passed else "✗"
    print(f"  [{symbol}] {label}")

all_passed = all(p for _, p in checks)
print(f"\n{'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
