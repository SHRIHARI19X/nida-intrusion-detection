"""
verify_autoencoder.py
---------------------
Phase 5 architecture verification runner.

Tasks:
  1. Read input_dimension from Phase 4 artifacts (no hard-coding).
  2. Build and compile the Dense Autoencoder.
  3. Print model.summary().
  4. Run a smoke test with a tiny dummy float32 batch.
  5. Verify input == output shape.
  6. Verify layer structure, activations, optimizer, loss.
  7. Save the untrained model to models/autoencoder_untrained.keras.
  8. Save results/model_architecture_summary.json.

This script does NOT call model.fit() — no training occurs here.
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf

# Ensure src/ is importable when run from project root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from autoencoder import build_autoencoder, load_input_dimension

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR    = os.path.join(BASE_DIR, "models")
RESULTS_DIR   = os.path.join(BASE_DIR, "results")
SUMMARY_PATH  = os.path.join(RESULTS_DIR, "model_input_summary.json")
UNTRAINED_MODEL_PATH = os.path.join(MODELS_DIR, "autoencoder_untrained.keras")
ARCH_SUMMARY_PATH    = os.path.join(RESULTS_DIR, "model_architecture_summary.json")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("PHASE 5 — DENSE AUTOENCODER ARCHITECTURE VERIFICATION")
print("=" * 60)

# -------------------------------------------------------------------------
# STEP 1: Read input_dimension from Phase 4 artifact — never hard-code
# -------------------------------------------------------------------------
print("\n[STEP 1] Reading input_dimension from model_input_summary.json...")
input_dim = load_input_dimension(SUMMARY_PATH)
print(f"  input_dimension (from Phase 4 artifact): {input_dim}")
print(f"  TensorFlow version: {tf.__version__}")
print(f"  Keras version:      {tf.keras.__version__}")

# -------------------------------------------------------------------------
# STEP 2: Build and compile the Autoencoder
# -------------------------------------------------------------------------
print(f"\n[STEP 2] Building Dense Autoencoder (input_dimension={input_dim})...")
model = build_autoencoder(input_dim)
print("  Model built and compiled successfully.")

# -------------------------------------------------------------------------
# STEP 3: Print model summary
# -------------------------------------------------------------------------
print("\n[STEP 3] Model Summary:")
model.summary()

# -------------------------------------------------------------------------
# STEP 4: Extract layer-level architecture details
# -------------------------------------------------------------------------
print("\n[STEP 4] Layer-level verification...")
layer_info = []
for layer in model.layers:
    cfg = layer.get_config()
    activation = cfg.get("activation", "N/A")
    units      = cfg.get("units", "N/A")
    layer_info.append({
        "name":       layer.name,
        "type":       type(layer).__name__,
        "units":      units,
        "activation": activation,
    })
    print(f"  Layer: {layer.name:20s}  type={type(layer).__name__:12s}  "
          f"units={str(units):6s}  activation={activation}")

# Verify encoder progression: input_dim -> 64 -> 32 -> 16
encoder_units = [64, 32, 16]
decoder_units = [32, 64, input_dim]

dense_layers = [l for l in model.layers if type(l).__name__ == "Dense"]
actual_units = [l.get_config()["units"] for l in dense_layers]
actual_activations = [l.get_config()["activation"] for l in dense_layers]

print(f"\n  Expected Dense layer units: {encoder_units + decoder_units}")
print(f"  Actual   Dense layer units: {actual_units}")
units_correct = (actual_units == encoder_units + decoder_units)
print(f"  Layer units match specification: {units_correct}")

# Verify activations: first 5 hidden = relu, last output = linear
expected_activations = ["relu", "relu", "relu", "relu", "relu", "linear"]
activations_correct = (actual_activations == expected_activations)
print(f"  Expected activations: {expected_activations}")
print(f"  Actual   activations: {actual_activations}")
print(f"  Activations match specification: {activations_correct}")

# -------------------------------------------------------------------------
# STEP 5: Verify input / output shape
# -------------------------------------------------------------------------
print("\n[STEP 5] Input / output shape verification...")
model_input_shape  = tuple(model.input_shape)
model_output_shape = tuple(model.output_shape)
print(f"  model.input_shape  : {model_input_shape}")
print(f"  model.output_shape : {model_output_shape}")

input_dim_correct  = (model_input_shape  == (None, input_dim))
output_dim_correct = (model_output_shape == (None, input_dim))
shapes_match       = (model_input_shape  == model_output_shape)
print(f"  Input  shape correct (None, {input_dim}): {input_dim_correct}")
print(f"  Output shape correct (None, {input_dim}): {output_dim_correct}")
print(f"  Input == Output shape (reconstruction): {shapes_match}")

# -------------------------------------------------------------------------
# STEP 6: Verify optimizer and loss
# -------------------------------------------------------------------------
print("\n[STEP 6] Optimizer and loss verification...")
optimizer_name = model.optimizer.__class__.__name__
loss_fn_name   = model.loss if isinstance(model.loss, str) else type(model.loss).__name__
print(f"  Optimizer : {optimizer_name}")
print(f"  Loss      : {loss_fn_name}")
optimizer_ok = (optimizer_name.lower() == "adam")
loss_ok      = (str(loss_fn_name).lower() in ("mse", "mean_squared_error"))
print(f"  Optimizer is Adam: {optimizer_ok}")
print(f"  Loss is MSE:       {loss_ok}")

# Verify no classification or recurrent layers
layer_types = [type(l).__name__ for l in model.layers]
forbidden   = {"Conv1D", "Conv2D", "LSTM", "GRU", "MultiHeadAttention",
               "TransformerBlock", "Bidirectional", "Conv3D"}
found_forbidden = [t for t in layer_types if t in forbidden]
no_forbidden = (len(found_forbidden) == 0)
print(f"  Forbidden layer types present: {found_forbidden if found_forbidden else 'None'}")
print(f"  No CNN/RNN/Transformer layers: {no_forbidden}")

# -------------------------------------------------------------------------
# STEP 7: Smoke test with dummy input
# -------------------------------------------------------------------------
print("\n[STEP 7] Smoke test (untrained forward pass with 2 dummy samples)...")
np.random.seed(42)
dummy_input  = np.random.randn(2, input_dim).astype(np.float32)
dummy_output = model.predict(dummy_input, verbose=0)
print(f"  Dummy input  shape: {dummy_input.shape}")
print(f"  Dummy output shape: {dummy_output.shape}")
smoke_pass = (dummy_input.shape == dummy_output.shape)
print(f"  Input shape == Output shape: {smoke_pass}")
print(f"  Output dtype: {dummy_output.dtype}")
print(f"  Output sample (first 6 values): {dummy_output[0, :6].round(6)}")

# -------------------------------------------------------------------------
# STEP 8: Parameter count
# -------------------------------------------------------------------------
print("\n[STEP 8] Parameter counts...")
total_params     = model.count_params()
trainable_params = sum(
    tf.size(w).numpy() for w in model.trainable_weights
)
non_trainable    = total_params - trainable_params
print(f"  Total parameters     : {total_params:,}")
print(f"  Trainable parameters : {trainable_params:,}")
print(f"  Non-trainable params : {non_trainable:,}")

# -------------------------------------------------------------------------
# STEP 9: Save untrained model
# -------------------------------------------------------------------------
print(f"\n[STEP 9] Saving untrained model to {UNTRAINED_MODEL_PATH}...")
model.save(UNTRAINED_MODEL_PATH)
saved_exists = os.path.exists(UNTRAINED_MODEL_PATH)
print(f"  Saved: {saved_exists}  ({os.path.getsize(UNTRAINED_MODEL_PATH):,} bytes)")

# -------------------------------------------------------------------------
# STEP 10: Save architecture summary JSON
# -------------------------------------------------------------------------
print(f"\n[STEP 10] Saving model_architecture_summary.json...")

arch_summary = {
    "model_type": "Dense Autoencoder",
    "framework": f"TensorFlow {tf.__version__} / Keras {tf.keras.__version__}",
    "input_dimension": input_dim,
    "latent_dimension": 16,
    "output_dimension": input_dim,
    "encoder_layers": [
        {"name": "encoder_1", "units": 64, "activation": "relu"},
        {"name": "encoder_2", "units": 32, "activation": "relu"},
        {"name": "latent",    "units": 16, "activation": "relu"},
    ],
    "decoder_layers": [
        {"name": "decoder_1",      "units": 32,        "activation": "relu"},
        {"name": "decoder_2",      "units": 64,        "activation": "relu"},
        {"name": "reconstruction", "units": input_dim, "activation": "linear"},
    ],
    "optimizer": optimizer_name,
    "loss": "mean_squared_error",
    "total_parameters": int(total_params),
    "trainable_parameters": int(trainable_params),
    "non_trainable_parameters": int(non_trainable),
    "input_shape": list(model_input_shape),
    "output_shape": list(model_output_shape),
    "architecture_verified": all([
        units_correct,
        activations_correct,
        input_dim_correct,
        output_dim_correct,
        shapes_match,
        optimizer_ok,
        loss_ok,
        no_forbidden,
        smoke_pass,
    ]),
    "untrained_model_path": UNTRAINED_MODEL_PATH,
    "note": (
        "Dense Autoencoder chosen for tabular NSL-KDD data. "
        "Learns normal traffic representation; high reconstruction error "
        "signals anomalous/attack traffic at inference time."
    ),
}

with open(ARCH_SUMMARY_PATH, "w") as f:
    json.dump(arch_summary, f, indent=4)
print(f"  Saved: {ARCH_SUMMARY_PATH}")

# -------------------------------------------------------------------------
# Final checklist
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 5 COMPLETE — ARCHITECTURE VERIFICATION CHECKLIST")
print("=" * 60)
checks = [
    ("input_dimension read from Phase 4 artifact (not hard-coded)", True),
    ("Model created successfully",                                   True),
    ("Input shape correct",                                          input_dim_correct),
    ("Output shape correct",                                         output_dim_correct),
    ("Output dim == Input dim (reconstruction)",                     shapes_match),
    ("Encoder reduces dimensionality (77->64->32->16)",              units_correct),
    ("Latent dimension is 16",                                       actual_units[2] == 16),
    ("Decoder restores dimensionality (16->32->64->77)",             units_correct),
    ("All hidden activations are relu",                              all(a == "relu" for a in actual_activations[:-1])),
    ("Output activation is linear",                                  actual_activations[-1] == "linear"),
    ("Optimizer is Adam",                                            optimizer_ok),
    ("Loss is MSE (not classification loss)",                        loss_ok),
    ("No CNN/LSTM/Transformer layers present",                       no_forbidden),
    ("Smoke test passed (input shape == output shape)",              smoke_pass),
    ("model.fit() NOT called (no training)",                         True),
    ("Untrained model saved",                                        saved_exists),
    ("Architecture summary JSON saved",                              os.path.exists(ARCH_SUMMARY_PATH)),
]

for label, passed in checks:
    symbol = "OK" if passed else "FAIL"
    print(f"  [{symbol}] {label}")

all_passed = all(p for _, p in checks)
print(f"\n{'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
print(f"\nAutoencoder input_dimension: {input_dim}")
print(f"Autoencoder latent_dimension: 16")
print(f"Autoencoder total parameters: {total_params:,}")
