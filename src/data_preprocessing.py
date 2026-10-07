"""
data_preprocessing.py
---------------------
Complete preprocessing module for the NSL-KDD Network Intrusion Detection project.

Phase 2 (preserved): Schema definitions, dataset loading, attack taxonomy.
Phase 3 (added):     Binary label creation, normal-traffic filtering, 80/20 split,
                     ColumnTransformer (OneHotEncoder + StandardScaler) pipeline,
                     strictly leakage-free fitting strategy, and persistence utilities.

DATA LEAKAGE PREVENTION DESIGN:
- The ColumnTransformer is fitted ONLY on X_train_normal (normal records, training split).
- X_val_normal and X_test are transformed using the ALREADY FITTED transformer.
- KDDTest+ NEVER influences fit() calls.
- Threshold determination (Phase 8) uses validation reconstruction errors only.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Schema: NSL-KDD 41 traffic features (in column order)
# ---------------------------------------------------------------------------

TRAFFIC_FEATURES = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

# Metadata columns appended by NSL-KDD files (not model inputs)
METADATA_COLUMNS = ["label", "difficulty"]

# Complete 43-column schema (41 features + label + difficulty)
COLUMN_NAMES = TRAFFIC_FEATURES + METADATA_COLUMNS

# Categorical features (3) — require OneHotEncoding
CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]

# Numerical features (38) — require StandardScaling
NUMERICAL_FEATURES = [f for f in TRAFFIC_FEATURES if f not in CATEGORICAL_FEATURES]

# ---------------------------------------------------------------------------
# Broad attack taxonomy for NSL-KDD (Phase 2, preserved)
# ---------------------------------------------------------------------------

ATTACK_CATEGORIES = {
    # Normal
    "normal": "Normal",
    # DoS
    "apache2": "DoS", "back": "DoS", "land": "DoS", "mailbomb": "DoS",
    "neptune": "DoS", "pod": "DoS", "processtable": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "udpstorm": "DoS",
    # Probe
    "ipsweep": "Probe", "mscan": "Probe", "nmap": "Probe",
    "portsweep": "Probe", "saint": "Probe", "satan": "Probe",
    # R2L
    "ftp_write": "R2L", "guess_passwd": "R2L", "httptunnel": "R2L",
    "imap": "R2L", "multihop": "R2L", "named": "R2L", "phf": "R2L",
    "sendmail": "R2L", "snmpgetattack": "R2L", "snmpguess": "R2L",
    "snmptrap": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "worm": "R2L", "xlock": "R2L", "xsnoop": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "ps": "U2R", "rootkit": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}


# ===========================================================================
# PHASE 2 FUNCTIONS (preserved)
# ===========================================================================

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load an NSL-KDD .txt file, assigning the correct 43-column schema.
    Files have no header row; column names are assigned explicitly.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    df = pd.read_csv(file_path, names=COLUMN_NAMES, header=None)
    return df


def map_attack_category(label: str) -> str:
    """Return the broad attack category for a given NSL-KDD label string."""
    return ATTACK_CATEGORIES.get(str(label).strip().lower(), "Unknown")


# ===========================================================================
# PHASE 3 FUNCTIONS
# ===========================================================================

def create_binary_labels(df: pd.DataFrame) -> pd.Series:
    """
    Create a binary target column from the 'label' column.

    Mapping:
        'normal'  → 0  (NORMAL traffic)
        anything else → 1  (INTRUSION / ATTACK)

    The original 'label' column is preserved in the DataFrame.
    This function returns only the binary Series.
    """
    return (df["label"].str.strip().str.lower() != "normal").astype(int)


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only the 41 traffic feature columns from a loaded DataFrame.

    Excludes 'label', 'difficulty', and any column not in TRAFFIC_FEATURES.
    This is the raw (un-encoded) input to the preprocessor.
    """
    # Verify all expected columns are present
    missing = [c for c in TRAFFIC_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Expected feature columns missing from DataFrame: {missing}")
    return df[TRAFFIC_FEATURES].copy()


def split_normal_training_data(
    X_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """
    Select only normal (binary_label == 0) records from KDDTrain+,
    then perform an 80/20 train/validation split.

    Leakage guarantee: KDDTest+ is never passed to this function.

    Returns
    -------
    X_train_normal : pd.DataFrame  — 80% of normal training records
    X_val_normal   : pd.DataFrame  — 20% of normal training records
    y_train_normal : pd.Series     — binary labels (all 0)
    y_val_normal   : pd.Series     — binary labels (all 0)
    """
    # Filter to normal records only — Autoencoder trains only on normal traffic
    normal_mask = (y_train_full == 0)
    X_normal = X_train_full[normal_mask]
    y_normal = y_train_full[normal_mask]

    X_train_normal, X_val_normal, y_train_normal, y_val_normal = train_test_split(
        X_normal,
        y_normal,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return X_train_normal, X_val_normal, y_train_normal, y_val_normal


def build_preprocessor() -> ColumnTransformer:
    """
    Build the ColumnTransformer pipeline.

    Categorical features → OneHotEncoder(handle_unknown='ignore')
        handle_unknown='ignore' ensures unseen categories in test data
        produce a zero-vector rather than raising an error.

    Numerical features → StandardScaler()
        Fitted exclusively on training data; test data is only transformed.

    remainder='drop' discards any columns not explicitly listed
    (this is a safeguard; all feature columns are explicitly named).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,  # cleaner feature names
    )
    return preprocessor


def fit_preprocessor(preprocessor: ColumnTransformer, X_train_normal: pd.DataFrame) -> ColumnTransformer:
    """
    Fit the ColumnTransformer ONLY on normal training data.

    CRITICAL: This function must never be called with validation or test data.
    Calling fit() on test/validation data would constitute data leakage.
    """
    preprocessor.fit(X_train_normal)
    return preprocessor


def transform_datasets(
    preprocessor: ColumnTransformer,
    X_train_normal: pd.DataFrame,
    X_val_normal: pd.DataFrame,
    X_test: pd.DataFrame,
    dtype=np.float32,
):
    """
    Apply the ALREADY FITTED preprocessor to all three splits.

    Only .transform() is called here — never .fit() or .fit_transform().
    This is the core leakage prevention: the scaler and encoder parameters
    come exclusively from X_train_normal.

    Returns float32 numpy arrays suitable for TensorFlow/Keras consumption.
    """
    X_train_processed = preprocessor.transform(X_train_normal).astype(dtype)
    X_val_processed   = preprocessor.transform(X_val_normal).astype(dtype)
    X_test_processed  = preprocessor.transform(X_test).astype(dtype)
    return X_train_processed, X_val_processed, X_test_processed


def save_preprocessor(preprocessor: ColumnTransformer, save_path: str) -> None:
    """
    Persist the fitted ColumnTransformer to disk using joblib.

    The saved artifact must be the SAME object used during evaluation
    and in the final predict.py / Streamlit app.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(preprocessor, save_path)
    print(f"Preprocessor saved → {save_path}")


def load_preprocessor(save_path: str) -> ColumnTransformer:
    """Load and return a previously saved ColumnTransformer."""
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Preprocessor artifact not found: {save_path}")
    return joblib.load(save_path)


def get_processed_feature_count(preprocessor: ColumnTransformer) -> int:
    """
    Return the number of features produced by the fitted preprocessor.
    Derived from the actual fitted transformer — never hard-coded.
    """
    return preprocessor.transform(
        pd.DataFrame(
            [["tcp", "http", "SF"] + [0] * len(NUMERICAL_FEATURES)],
            columns=TRAFFIC_FEATURES,
        )
    ).shape[1]


def check_array_integrity(array: np.ndarray, name: str) -> dict:
    """
    Check a processed numpy array for NaN and infinite values.
    Returns a dict with nan_count and inf_count.
    """
    nan_count = int(np.isnan(array).sum())
    inf_count = int(np.isinf(array).sum())
    print(f"  {name}: shape={array.shape}, dtype={array.dtype}, "
          f"NaN={nan_count}, Inf={inf_count}")
    return {"nan_count": nan_count, "inf_count": inf_count}
