"""
autoencoder.py
--------------
Dense Autoencoder architecture for Network Intrusion Detection.

WHY A DENSE (FULLY CONNECTED) AUTOENCODER?
    The NSL-KDD dataset is tabular network traffic data — a flat feature
    vector of 77 processed numerical features per sample. There is no
    spatial structure (as in images → CNN) and no meaningful temporal
    sequence (as in time-series → LSTM/GRU). A fully connected Dense
    Autoencoder is therefore the most appropriate architecture:

      - It learns a compact nonlinear representation of normal traffic.
      - The encoder compresses the input to a low-dimensional latent
        space; the decoder reconstructs it.
      - Anomaly score = per-sample Mean Squared Reconstruction Error.
        Normal traffic: low reconstruction error (the model knows it).
        Attack traffic: high reconstruction error (unseen distribution).

ARCHITECTURE (per Project.md specification):
    Input  (77)
      -> Dense(64, relu)        Encoder stage 1
      -> Dense(32, relu)        Encoder stage 2
      -> Dense(16, relu)        Latent representation
      -> Dense(32, relu)        Decoder stage 1
      -> Dense(64, relu)        Decoder stage 2
      -> Dense(77, linear)      Reconstruction output

    The output activation is 'linear' because the preprocessed input
    (StandardScaler output) can be negative — a sigmoid/relu output
    would incorrectly clamp negative values.

    input_dimension is determined DYNAMICALLY from the saved
    model_input_summary.json — it is never hard-coded.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


def build_autoencoder(input_dimension: int) -> Model:
    """
    Build and compile a Dense Autoencoder for anomaly detection.

    Parameters
    ----------
    input_dimension : int
        Number of processed features — read from model_input_summary.json,
        never hard-coded. Currently resolves to 77 for the NSL-KDD dataset
        after OneHotEncoding and StandardScaling.

    Returns
    -------
    model : keras.Model
        Compiled Autoencoder ready for training.

    Architecture rationale
    ----------------------
    Encoder (77 -> 64 -> 32 -> 16):
        Progressive compression forces the model to learn the most
        informative features of normal network traffic. The 16-dimensional
        bottleneck is the latent representation.

    Decoder (16 -> 32 -> 64 -> 77):
        Symmetric expansion reconstructs the full feature vector from
        the compressed representation. Reconstruction quality is the
        anomaly signal.

    Activation functions:
        - ReLU in hidden layers: fast, non-saturating, avoids vanishing
          gradients in deeper networks.
        - Linear in the output layer: preserves full real-valued range
          of StandardScaler-normalised features (including negatives).

    Optimizer: Adam — adaptive learning rate; well-suited for tabular data.
    Loss: Mean Squared Error — directly measures reconstruction fidelity.
    """
    if not isinstance(input_dimension, int) or input_dimension <= 0:
        raise ValueError(
            f"input_dimension must be a positive integer; got {input_dimension!r}"
        )

    # ------------------------------------------------------------------ #
    # Input layer                                                          #
    # ------------------------------------------------------------------ #
    inputs = layers.Input(shape=(input_dimension,), name="input")

    # ------------------------------------------------------------------ #
    # Encoder                                                              #
    # 77 -> 64 -> 32 -> 16 (latent)                                       #
    # Each Dense layer uses relu activation for nonlinear compression.    #
    # ------------------------------------------------------------------ #
    x = layers.Dense(64, activation="relu", name="encoder_1")(inputs)
    x = layers.Dense(32, activation="relu", name="encoder_2")(x)
    encoded = layers.Dense(16, activation="relu", name="latent")(x)

    # ------------------------------------------------------------------ #
    # Decoder                                                              #
    # 16 -> 32 -> 64 -> 77 (reconstruction)                              #
    # Linear output preserves the signed range of scaled features.       #
    # ------------------------------------------------------------------ #
    x = layers.Dense(32, activation="relu", name="decoder_1")(encoded)
    x = layers.Dense(64, activation="relu", name="decoder_2")(x)
    outputs = layers.Dense(
        input_dimension, activation="linear", name="reconstruction"
    )(x)

    # ------------------------------------------------------------------ #
    # Model definition                                                     #
    # ------------------------------------------------------------------ #
    model = Model(inputs=inputs, outputs=outputs, name="dense_autoencoder")

    # ------------------------------------------------------------------ #
    # Compile                                                              #
    # Optimizer: Adam (adaptive learning rate)                            #
    # Loss:      MSE  (reconstruction fidelity — NOT classification loss) #
    # ------------------------------------------------------------------ #
    model.compile(optimizer="adam", loss="mse")

    return model


def load_input_dimension(summary_path: str) -> int:
    """
    Read the processed feature count from the Phase 4 summary JSON.

    This ensures input_dimension is always derived from the actual
    fitted preprocessor — never hard-coded in the architecture code.
    """
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"model_input_summary.json not found at: {summary_path}\n"
            "Run Phase 4 (verify_model_data.py) before building the Autoencoder."
        )
    with open(summary_path) as f:
        summary = json.load(f)

    feature_count = summary.get("processed_feature_count")
    if not isinstance(feature_count, int) or feature_count <= 0:
        raise ValueError(
            f"Invalid processed_feature_count in summary: {feature_count!r}"
        )
    return feature_count
