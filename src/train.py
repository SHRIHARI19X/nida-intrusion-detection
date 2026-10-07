"""
train.py
--------
Phase 6: Train the Dense Autoencoder on NORMAL traffic only.

Training strategy (reconstruction learning):
    Input:  X_train_normal  ->  Target: X_train_normal
    Val:    X_val_normal    ->  Target: X_val_normal

The model learns to reconstruct normal traffic. Attack traffic
will produce high reconstruction error at inference time — that
gap IS the anomaly signal. No attack samples or test data are
used here in any way.
"""

import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on all platforms
import matplotlib.pyplot as plt

# Reproducibility seeds
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

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
from autoencoder import build_autoencoder, load_input_dimension

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR         = os.path.join(BASE_DIR, "data")
MODELS_DIR       = os.path.join(BASE_DIR, "models")
RESULTS_DIR      = os.path.join(BASE_DIR, "results")

TRAIN_FILE            = os.path.join(DATA_DIR,   "KDDTrain+.txt")
TEST_FILE             = os.path.join(DATA_DIR,   "KDDTest+.txt")
PREPROCESSOR_PATH     = os.path.join(MODELS_DIR, "preprocessor.joblib")
MODEL_INPUT_SUMMARY   = os.path.join(RESULTS_DIR, "model_input_summary.json")
TRAINED_MODEL_PATH    = os.path.join(MODELS_DIR, "autoencoder.keras")
LOSS_CURVE_PATH       = os.path.join(RESULTS_DIR, "loss_curve.png")
TRAINING_SUMMARY_PATH = os.path.join(RESULTS_DIR, "training_summary.json")
TRAINING_HISTORY_PATH = os.path.join(RESULTS_DIR, "training_history.json")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("PHASE 6 — DENSE AUTOENCODER TRAINING")
print("=" * 60)
print(f"TensorFlow : {tf.__version__}")

# Check compute device
gpus = tf.config.list_physical_devices("GPU")
training_device = "GPU" if gpus else "CPU"
print(f"Training device: {training_device}")
if not gpus:
    print("  No GPU detected — training on CPU (expected on native Windows TF 2.21).")

# ===========================================================================
# STEP 1: Verify input dimension from Phase 4 artifact — no hard-coding
# ===========================================================================
print("\n[STEP 1] Reading input_dimension from Phase 4 artifact...")
input_dim = load_input_dimension(MODEL_INPUT_SUMMARY)
print(f"  input_dimension: {input_dim}")
assert input_dim == 77, f"Unexpected input_dim: {input_dim} (expected 77)"

# ===========================================================================
# STEP 2: Load datasets and rebuild model-ready arrays
# ===========================================================================
print("\n[STEP 2] Loading datasets and reproducing model-ready arrays...")
train_df = load_dataset(TRAIN_FILE)
test_df  = load_dataset(TEST_FILE)        # loaded but NEVER used for training

y_train_full = create_binary_labels(train_df)
X_train_full = get_feature_matrix(train_df)
X_test_raw   = get_feature_matrix(test_df)
y_test        = create_binary_labels(test_df)   # isolated — evaluation only

# Reproduce the deterministic 80/20 normal split
X_train_normal, X_val_normal, y_train_normal, y_val_normal = split_normal_training_data(
    X_train_full, y_train_full, test_size=0.20, random_state=RANDOM_STATE
)

# Load the FITTED preprocessor (no refit)
preprocessor = load_preprocessor(PREPROCESSOR_PATH)

# Transform using saved preprocessor — transform only, no fit
X_train_processed, X_val_processed, X_test_processed = transform_datasets(
    preprocessor, X_train_normal, X_val_normal, X_test_raw
)
print(f"  X_train_processed : {X_train_processed.shape}")
print(f"  X_val_processed   : {X_val_processed.shape}")
print(f"  X_test_processed  : {X_test_processed.shape}  (isolated — not used in training)")

# ===========================================================================
# STEP 3: Verify NORMAL-ONLY training guarantee
# ===========================================================================
print("\n[STEP 3] Normal-only training verification...")
assert (y_train_normal == 0).all(), "Attack records found in training set!"
assert (y_val_normal   == 0).all(), "Attack records found in validation set!"
assert X_train_processed.shape[1] == input_dim, "Feature dimension mismatch!"
assert X_val_processed.shape[1]   == input_dim, "Feature dimension mismatch!"

print(f"  Training samples  (normal only): {len(X_train_processed):,}")
print(f"  Validation samples (normal only): {len(X_val_processed):,}")
print(f"  Attack samples in training : 0  (confirmed)")
print(f"  Test data used for training: No (completely isolated)")
print("  NORMAL-ONLY TRAINING: PASS")

# ===========================================================================
# STEP 4: Build autoencoder with dynamic input_dimension
# ===========================================================================
print(f"\n[STEP 4] Building Dense Autoencoder (input_dim={input_dim})...")
model = build_autoencoder(input_dim)
model.summary()
print(f"  Optimizer : {model.optimizer.__class__.__name__}")
print(f"  Loss      : {model.loss}")
print(f"  Trainable params : {model.count_params():,}")

# ===========================================================================
# STEP 5 & 6: Training configuration and execution
# ===========================================================================
EPOCHS     = 50
BATCH_SIZE = 256
PATIENCE   = 5

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=PATIENCE,
    restore_best_weights=True,
    verbose=1,
)

print(f"\n[STEP 5/6] Training configuration:")
print(f"  epochs       = {EPOCHS}")
print(f"  batch_size   = {BATCH_SIZE}")
print(f"  EarlyStopping: monitor=val_loss, patience={PATIENCE}, restore_best_weights=True")
print(f"\n  Training input  -> target : X_train_processed -> X_train_processed")
print(f"  Validation input -> target : X_val_processed   -> X_val_processed")
print("\nStarting model.fit()...\n")

t_start = time.time()
history = model.fit(
    X_train_processed,          # input
    X_train_processed,          # reconstruction target (same as input)
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val_processed, X_val_processed),
    callbacks=[early_stopping],
    verbose=1,
    shuffle=True,
)
t_elapsed = time.time() - t_start

# ===========================================================================
# STEP 7: Inspect training history
# ===========================================================================
print(f"\n[STEP 7] Training history inspection...")
train_losses = history.history["loss"]
val_losses   = history.history["val_loss"]
epochs_run   = len(train_losses)

best_val_loss    = float(min(val_losses))
final_train_loss = float(train_losses[-1])
final_val_loss   = float(val_losses[-1])

print(f"  Epochs actually trained : {epochs_run}")
print(f"  Final training loss     : {final_train_loss:.6f}")
print(f"  Final validation loss   : {final_val_loss:.6f}")
print(f"  Best validation loss    : {best_val_loss:.6f}")
print(f"  Training duration       : {t_elapsed:.1f}s")

# Sanity checks
assert all(np.isfinite(l) for l in train_losses), "NaN/Inf in training loss!"
assert all(np.isfinite(l) for l in val_losses),   "NaN/Inf in validation loss!"
print("  No NaN or Inf in any loss value: PASS")

# ===========================================================================
# STEP 8: Generate loss curve
# ===========================================================================
print(f"\n[STEP 8] Generating loss curve...")

fig, ax = plt.subplots(figsize=(9, 5))
epochs_axis = range(1, epochs_run + 1)

ax.plot(epochs_axis, train_losses, label="Training Loss",   color="#1f77b4", linewidth=2)
ax.plot(epochs_axis, val_losses,   label="Validation Loss", color="#d62728",
        linewidth=2, linestyle="--")

# Mark best validation epoch
best_epoch = val_losses.index(best_val_loss) + 1
ax.axvline(x=best_epoch, color="gray", linestyle=":", linewidth=1.2,
           label=f"Best val epoch ({best_epoch})")
ax.scatter([best_epoch], [best_val_loss], color="#d62728", s=80, zorder=5)

ax.set_title("Dense Autoencoder — MSE Loss (Normal Traffic Reconstruction)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Mean Squared Error", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(left=1)

plt.tight_layout()
plt.savefig(LOSS_CURVE_PATH, dpi=300)
plt.close()
print(f"  Saved: {LOSS_CURVE_PATH}")

# ===========================================================================
# STEP 9: Save trained model
# ===========================================================================
print(f"\n[STEP 9] Saving trained model to {TRAINED_MODEL_PATH}...")
model.save(TRAINED_MODEL_PATH)
model_size = os.path.getsize(TRAINED_MODEL_PATH)
print(f"  Saved: {model_size:,} bytes")

# ===========================================================================
# STEP 10: Save training summary and history
# ===========================================================================
print(f"\n[STEP 10] Saving training artifacts...")

training_summary = {
    "phase": "Phase 6 — Autoencoder Training",
    "model_type": "Dense Autoencoder",
    "framework": f"TensorFlow {tf.__version__}",
    "input_dimension": input_dim,
    "architecture": "77 -> 64 -> 32 -> 16 -> 32 -> 64 -> 77",
    "optimizer": model.optimizer.__class__.__name__,
    "loss_function": "mean_squared_error",
    "batch_size": BATCH_SIZE,
    "max_epochs": EPOCHS,
    "epochs_actually_trained": epochs_run,
    "early_stopping": {
        "monitor": "val_loss",
        "patience": PATIENCE,
        "restore_best_weights": True,
        "stopped_early": epochs_run < EPOCHS,
    },
    "training_samples": int(len(X_train_processed)),
    "validation_samples": int(len(X_val_processed)),
    "final_training_loss": round(final_train_loss, 8),
    "final_validation_loss": round(final_val_loss, 8),
    "best_validation_loss": round(best_val_loss, 8),
    "best_epoch": best_epoch,
    "training_duration_seconds": round(t_elapsed, 2),
    "training_device": training_device,
    "random_state": RANDOM_STATE,
    "normal_only_training": True,
    "test_data_used_in_training": False,
    "trained_model_path": TRAINED_MODEL_PATH,
    "loss_curve_path": LOSS_CURVE_PATH,
}

with open(TRAINING_SUMMARY_PATH, "w") as f:
    json.dump(training_summary, f, indent=4)
print(f"  Training summary saved: {TRAINING_SUMMARY_PATH}")

training_history = {
    "train_loss": [round(v, 8) for v in train_losses],
    "val_loss":   [round(v, 8) for v in val_losses],
}
with open(TRAINING_HISTORY_PATH, "w") as f:
    json.dump(training_history, f, indent=4)
print(f"  Training history saved: {TRAINING_HISTORY_PATH}")

# ===========================================================================
# STEP 11: Reload and verify saved model
# ===========================================================================
print(f"\n[STEP 11] Reloading and verifying saved model...")
loaded_model = tf.keras.models.load_model(TRAINED_MODEL_PATH)
print(f"  Loaded model type: {type(loaded_model).__name__}")
print(f"  Input shape : {tuple(loaded_model.input_shape)}")
print(f"  Output shape: {tuple(loaded_model.output_shape)}")

# Smoke test on a slice of validation data (normal only)
smoke_input  = X_val_processed[:4]
smoke_output = loaded_model.predict(smoke_input, verbose=0)
reload_smoke_ok = (smoke_input.shape == smoke_output.shape)
print(f"  Reload smoke test shape match: {reload_smoke_ok}")
print(f"  Output dtype: {smoke_output.dtype}")
assert reload_smoke_ok, "Reloaded model forward pass shape mismatch!"
print("  Model reload verification: PASS")

# ===========================================================================
# Final checklist
# ===========================================================================
print("\n" + "=" * 60)
print("PHASE 6 COMPLETE — TRAINING VERIFICATION CHECKLIST")
print("=" * 60)
checks = [
    ("input_dimension from Phase 4 artifact (no hard-coding)",  True),
    ("Training data is normal only",                            True),
    ("Validation data is normal only",                          True),
    ("Test data NOT used in training",                          True),
    ("model.fit() completed successfully",                      True),
    ("Training loss is finite",                                 all(np.isfinite(l) for l in train_losses)),
    ("Validation loss is finite",                               all(np.isfinite(l) for l in val_losses)),
    ("EarlyStopping applied",                                   True),
    ("Loss curve generated",                                    os.path.exists(LOSS_CURVE_PATH)),
    ("Trained model saved",                                     os.path.exists(TRAINED_MODEL_PATH)),
    ("Trained model reloads successfully",                      True),
    ("Reload forward pass correct",                             reload_smoke_ok),
    ("Training summary saved",                                  os.path.exists(TRAINING_SUMMARY_PATH)),
    ("Training history saved",                                  os.path.exists(TRAINING_HISTORY_PATH)),
]

for label, passed in checks:
    symbol = "OK" if passed else "FAIL"
    print(f"  [{symbol}] {label}")

all_passed = all(p for _, p in checks)
print(f"\n{'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
print(f"\nEpochs trained       : {epochs_run}")
print(f"Final train loss     : {final_train_loss:.6f}")
print(f"Final val loss       : {final_val_loss:.6f}")
print(f"Best val loss        : {best_val_loss:.6f}  (epoch {best_epoch})")
print(f"Training device      : {training_device}")
