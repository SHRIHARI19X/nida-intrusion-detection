"""
streamlit_app.py
----------------
AI-Powered Network Intrusion Detection System (NIDS)
Based on Dense Autoencoder & NSL-KDD Dataset.

Architecture: 77 -> 64 -> 32 -> 16 -> 32 -> 64 -> 77 (15,261 Trainable Parameters)
Official Anomaly Threshold: 0.12907493 (Calibrated from Phase 8 normal validation P95)

Features:
  1. Single Record Detection (Custom input or quick preset attack/normal scenarios)
  2. Batch CSV Detection (Upload, column verification, bulk inference, results export)
  3. Benchmark Dataset Demo (Inspect & evaluate real test records from KDDTest+)
  4. Validated Performance Dashboard (Phase 10 official benchmark metrics)
  5. Architecture & Theoretical Methodology Guide
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import streamlit as st

# Configure headless/quiet TensorFlow environment
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import joblib

# ---------------------------------------------------------------------------
# Robust Project Root Path Setup & Preprocessing Module Import
# ---------------------------------------------------------------------------
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Also ensure BASE_DIR string is available for path definitions
BASE_DIR = str(PROJECT_ROOT)

from src.data_preprocessing import (
    TRAFFIC_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    COLUMN_NAMES,
    ATTACK_CATEGORIES,
    map_attack_category,
)

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
MODEL_PATH      = os.path.join(BASE_DIR, "models", "autoencoder.keras")
PREPROC_PATH    = os.path.join(BASE_DIR, "models", "preprocessor.joblib")
THRESHOLD_PATH  = os.path.join(BASE_DIR, "results", "anomaly_threshold.json")
METRICS_PATH    = os.path.join(BASE_DIR, "results", "final_metrics.json")
TEST_DATA_PATH  = os.path.join(BASE_DIR, "data", "KDDTest+.txt")
CM_IMAGE_PATH   = os.path.join(BASE_DIR, "results", "confusion_matrix.png")
ROC_IMAGE_PATH  = os.path.join(BASE_DIR, "results", "roc_curve.png")
DIST_IMAGE_PATH = os.path.join(BASE_DIR, "results", "test_reconstruction_error_distribution.png")

# Page Configuration
st.set_page_config(
    page_title="AI Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for Cybersecurity Dashboard Aesthetic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-lbl {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-normal {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1.5px solid #10B981;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.25rem;
        display: inline-block;
        text-align: center;
    }
    .badge-intrusion {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1.5px solid #EF4444;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.25rem;
        display: inline-block;
        text-align: center;
    }
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Caching & Artifact Loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Deep Learning Model and Preprocessor...")
def load_system_artifacts():
    """Load and cache the trained Autoencoder, preprocessor pipeline, and threshold."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Autoencoder model not found at: {MODEL_PATH}")
    if not os.path.exists(PREPROC_PATH):
        raise FileNotFoundError(f"Preprocessor not found at: {PREPROC_PATH}")
    if not os.path.exists(THRESHOLD_PATH):
        raise FileNotFoundError(f"Threshold metadata not found at: {THRESHOLD_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROC_PATH)

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        thresh_info = json.load(f)
    official_threshold = float(thresh_info.get("threshold", 0.1290749317))

    metrics_info = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_info = json.load(f)

    # Extract processed feature names
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"feat_{i}" for i in range(77)]

    return model, preprocessor, official_threshold, metrics_info, feature_names


@st.cache_data(show_spinner="Loading test benchmark samples...")
def load_test_sample_dataset():
    """Load a sample subset of KDDTest+ for demo purposes."""
    if os.path.exists(TEST_DATA_PATH):
        col_names = TRAFFIC_FEATURES + ["label", "difficulty"]
        df_sample = pd.read_csv(TEST_DATA_PATH, header=None, names=col_names, nrows=1000)
        return df_sample
    return None


# ---------------------------------------------------------------------------
# Inference Utility Function
# ---------------------------------------------------------------------------
def predict_record(record_df: pd.DataFrame, model, preprocessor, threshold: float, feature_names: list):
    """
    Perform reconstruction-based intrusion detection on a single DataFrame record.
    Returns:
        prediction (str): "NORMAL" or "INTRUSION"
        mse_error (float): Mean Squared Reconstruction Error
        diff_from_threshold (float): Error minus official threshold
        top_deviations (pd.DataFrame): Top 5 features with largest reconstruction deviations
    """
    # 1. Transform record with preprocessor
    X_proc = preprocessor.transform(record_df[TRAFFIC_FEATURES]).astype(np.float32)

    # 2. Reconstruct with trained Autoencoder
    X_recon = model.predict(X_proc, verbose=0)

    # 3. Calculate per-sample MSE error
    error_vector = np.square(X_proc.astype(np.float64) - X_recon.astype(np.float64))
    mse_error = float(np.mean(error_vector))

    # 4. Classification decision
    is_intrusion = mse_error > threshold
    prediction = "INTRUSION" if is_intrusion else "NORMAL"
    distance = mse_error - threshold

    # 5. Top feature deviations
    abs_diffs = np.abs(X_proc[0] - X_recon[0])
    top_indices = np.argsort(abs_diffs)[::-1][:5]

    top_deviations = pd.DataFrame({
        "Feature": [feature_names[i] if i < len(feature_names) else f"Feature_{i}" for i in top_indices],
        "Input (Normalized)": [round(float(X_proc[0, i]), 4) for i in top_indices],
        "Reconstructed": [round(float(X_recon[0, i]), 4) for i in top_indices],
        "Absolute Error": [round(float(abs_diffs[i]), 4) for i in top_indices],
    })

    return prediction, mse_error, distance, top_deviations


# ---------------------------------------------------------------------------
# Load System
# ---------------------------------------------------------------------------
try:
    model, preprocessor, OFFICIAL_THRESHOLD, VALIDATED_METRICS, FEATURE_NAMES = load_system_artifacts()
except Exception as e:
    st.error(f"⚠️ Initialization Error: Could not load required artifacts: {e}")
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar Navigation & Information
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.markdown("### **Navigation & Control**")

    app_mode = st.radio(
        "Select Detection Mode:",
        [
            "🔍 Single Record Detection",
            "📁 Batch CSV Detection",
            "🧪 Benchmark Dataset Demo",
            "📊 Validated Performance",
            "🧠 Model Architecture",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### **Official Model Specs**")
    st.markdown(f"**Model:** Dense Autoencoder")
    st.markdown(f"**Architecture:** `77 → 64 → 32 → 16 → 32 → 64 → 77`")
    st.markdown(f"**Trainable Parameters:** `15,261`")
    st.markdown(f"**Official Threshold:** `{OFFICIAL_THRESHOLD:.8f}`")
    st.markdown(f"**Threshold Source:** P95 Normal Val MSE")

    st.markdown("---")
    st.markdown("### **Classification Rule**")
    st.info(f"""
    - **Reconstruction MSE $\\le$ {OFFICIAL_THRESHOLD:.4f}**  
      $\\implies$ **NORMAL TRAFFIC**
    - **Reconstruction MSE $>$ {OFFICIAL_THRESHOLD:.4f}**  
      $\\implies$ **INTRUSION DETECTED**
    """)
    st.caption("Autoencoder learns normal traffic; high reconstruction errors flag anomalies.")


# ---------------------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">🛡️ AI-Powered Network Intrusion Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Unsupervised Anomaly Detection using Dense Autoencoder on NSL-KDD Traffic</div>', unsafe_allow_html=True)


# ===========================================================================
# MODE 1: SINGLE RECORD DETECTION
# ===========================================================================
if app_mode == "🔍 Single Record Detection":
    st.markdown("### **Single Connection Anomaly Analysis**")
    st.write("Inspect a single network connection record across all 41 NSL-KDD traffic features.")

    # Preset Quick Loaders for Demonstration
    st.markdown("##### **Quick Preset Demonstrations (Viva & Demo):**")
    c_p1, c_p2, c_p3, c_p4, c_p5 = st.columns(5)

    preset_normal = {
        "protocol_type": "tcp", "service": "http", "flag": "SF", "duration": 0, "src_bytes": 240, "dst_bytes": 850,
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 1,
        "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0, "num_file_creations": 0,
        "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 5, "srv_count": 8, "serror_rate": 0.0, "srv_serror_rate": 0.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
        "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0, "dst_host_count": 180, "dst_host_srv_count": 255,
        "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 0.02,
        "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0
    }

    preset_dos = {
        "protocol_type": "tcp", "service": "private", "flag": "S0", "duration": 0, "src_bytes": 0, "dst_bytes": 0,
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 0,
        "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0, "num_file_creations": 0,
        "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 240, "srv_count": 12, "serror_rate": 1.0, "srv_serror_rate": 1.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
        "same_srv_rate": 0.05, "diff_srv_rate": 0.08, "srv_diff_host_rate": 0.0, "dst_host_count": 255, "dst_host_srv_count": 12,
        "dst_host_same_srv_rate": 0.05, "dst_host_diff_srv_rate": 0.08, "dst_host_same_src_port_rate": 0.0,
        "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 1.0, "dst_host_srv_serror_rate": 1.0,
        "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0
    }

    preset_probe = {
        "protocol_type": "tcp", "service": "private", "flag": "REJ", "duration": 0, "src_bytes": 0, "dst_bytes": 0,
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 0,
        "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0, "num_file_creations": 0,
        "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 180, "srv_count": 1, "serror_rate": 0.0, "srv_serror_rate": 0.0, "rerror_rate": 1.0, "srv_rerror_rate": 1.0,
        "same_srv_rate": 0.01, "diff_srv_rate": 0.85, "srv_diff_host_rate": 0.0, "dst_host_count": 255, "dst_host_srv_count": 1,
        "dst_host_same_srv_rate": 0.0, "dst_host_diff_srv_rate": 0.95, "dst_host_same_src_port_rate": 0.0,
        "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 1.0, "dst_host_srv_rerror_rate": 1.0
    }

    preset_u2r = {
        "protocol_type": "tcp", "service": "telnet", "flag": "SF", "duration": 45, "src_bytes": 2800, "dst_bytes": 4500,
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 3, "num_failed_logins": 0, "logged_in": 1,
        "num_compromised": 5, "root_shell": 1, "su_attempted": 1, "num_root": 5, "num_file_creations": 2,
        "num_shells": 1, "num_access_files": 0, "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 1, "srv_count": 1, "serror_rate": 0.0, "srv_serror_rate": 0.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
        "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0, "dst_host_count": 25, "dst_host_srv_count": 25,
        "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 0.05,
        "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0
    }

    preset_r2l = {
        "protocol_type": "tcp", "service": "ftp", "flag": "SF", "duration": 28, "src_bytes": 120, "dst_bytes": 280,
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 2, "num_failed_logins": 3, "logged_in": 0,
        "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0, "num_file_creations": 0,
        "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 1,
        "count": 1, "srv_count": 1, "serror_rate": 0.0, "srv_serror_rate": 0.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
        "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0, "dst_host_count": 150, "dst_host_srv_count": 30,
        "dst_host_same_srv_rate": 0.2, "dst_host_diff_srv_rate": 0.05, "dst_host_same_src_port_rate": 0.01,
        "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0
    }

    if "current_inputs" not in st.session_state:
        st.session_state.current_inputs = preset_normal.copy()

    if c_p1.button("🟢 Typical Normal (HTTP)"):
        st.session_state.current_inputs = preset_normal.copy()
        st.rerun()
    if c_p2.button("🔴 Neptune SYN Flood (DoS)"):
        st.session_state.current_inputs = preset_dos.copy()
        st.rerun()
    if c_p3.button("🟣 Port Scanning (Probe)"):
        st.session_state.current_inputs = preset_probe.copy()
        st.rerun()
    if c_p4.button("🟠 Privilege Escalation (U2R)"):
        st.session_state.current_inputs = preset_u2r.copy()
        st.rerun()
    if c_p5.button("🟡 Password Guessing (R2L)"):
        st.session_state.current_inputs = preset_r2l.copy()
        st.rerun()

    cur = st.session_state.current_inputs

    # Input Form
    with st.form("single_record_form"):
        st.markdown("#### **Network Connection Parameters**")

        # 1. Categorical Section
        col_c1, col_c2, col_c3 = st.columns(3)
        protocol_options = ["tcp", "udp", "icmp"]
        all_services = [
            "http", "private", "smtp", "domain_u", "ftp_data", "ftp", "eco_i", "ecr_i",
            "telnet", "finger", "auth", "pop_3", "uucp", "other", "IRC", "X11", "Z39_50",
            "aol", "bgp", "courier", "csnet_ns", "ctf", "daytime", "discard", "domain",
            "echo", "efs", "exec", "gopher", "harvest", "hostnames", "http_2784", "http_443",
            "http_8001", "imap4", "iso_tsap", "klogin", "kshell", "ldap", "link", "login",
            "mtp", "name", "netbios_dgm", "netbios_ns", "netbios_ssn", "netstat", "nnsp",
            "nntp", "ntp_u", "pm_dump", "pop_2", "printer", "red_i", "remote_job", "rje",
            "shell", "sql_net", "ssh", "sunrpc", "supdup", "systat", "tftp_u", "tim_i",
            "time", "urh_i", "urp_i", "uucp_path", "vmnet", "whois"
        ]
        all_flags = ["SF", "S0", "REJ", "RSTR", "RSTO", "S1", "SH", "S2", "RSTOS0", "S3", "OTH"]

        protocol_type = col_c1.selectbox(
            "Protocol Type", protocol_options,
            index=protocol_options.index(cur.get("protocol_type", "tcp"))
        )
        service = col_c2.selectbox(
            "Network Service", all_services,
            index=all_services.index(cur.get("service", "http")) if cur.get("service") in all_services else 0
        )
        flag = col_c3.selectbox(
            "Connection Status Flag", all_flags,
            index=all_flags.index(cur.get("flag", "SF")) if cur.get("flag") in all_flags else 0
        )

        st.markdown("---")
        st.markdown("##### **Basic Traffic & Payload Features**")
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        duration = col_b1.number_input("Duration (seconds)", min_value=0, value=int(cur.get("duration", 0)))
        src_bytes = col_b2.number_input("Source Bytes", min_value=0, value=int(cur.get("src_bytes", 240)))
        dst_bytes = col_b3.number_input("Destination Bytes", min_value=0, value=int(cur.get("dst_bytes", 850)))
        land = col_b4.selectbox("Land (src == dst)", [0, 1], index=int(cur.get("land", 0)))

        col_b5, col_b6, col_b7, col_b8 = st.columns(4)
        wrong_fragment = col_b5.number_input("Wrong Fragments", min_value=0, value=int(cur.get("wrong_fragment", 0)))
        urgent = col_b6.number_input("Urgent Packets", min_value=0, value=int(cur.get("urgent", 0)))
        hot = col_b7.number_input("Hot Indicators", min_value=0, value=int(cur.get("hot", 0)))
        num_failed_logins = col_b8.number_input("Failed Login Attempts", min_value=0, value=int(cur.get("num_failed_logins", 0)))

        col_b9, col_b10, col_b11, col_b12 = st.columns(4)
        logged_in = col_b9.selectbox("Logged In", [0, 1], index=int(cur.get("logged_in", 1)))
        num_compromised = col_b10.number_input("Num Compromised", min_value=0, value=int(cur.get("num_compromised", 0)))
        root_shell = col_b11.selectbox("Root Shell Obtained", [0, 1], index=int(cur.get("root_shell", 0)))
        su_attempted = col_b12.selectbox("SU Attempted", [0, 1, 2], index=int(cur.get("su_attempted", 0)))

        col_b13, col_b14, col_b15, col_b16 = st.columns(4)
        num_root = col_b13.number_input("Num Root Operations", min_value=0, value=int(cur.get("num_root", 0)))
        num_file_creations = col_b14.number_input("File Creations", min_value=0, value=int(cur.get("num_file_creations", 0)))
        num_shells = col_b15.number_input("Num Shells Spawned", min_value=0, value=int(cur.get("num_shells", 0)))
        num_access_files = col_b16.number_input("Access Control Files", min_value=0, value=int(cur.get("num_access_files", 0)))

        col_b17, col_b18, col_b19, col_b20 = st.columns(4)
        num_outbound_cmds = col_b17.number_input("Outbound Commands", min_value=0, value=int(cur.get("num_outbound_cmds", 0)))
        is_host_login = col_b18.selectbox("Is Host Login", [0, 1], index=int(cur.get("is_host_login", 0)))
        is_guest_login = col_b19.selectbox("Is Guest Login", [0, 1], index=int(cur.get("is_guest_login", 0)))
        count = col_b20.number_input("Connection Count (2s window)", min_value=0, value=int(cur.get("count", 5)))

        st.markdown("---")
        st.markdown("##### **Rate & Host-Based Traffic Features**")
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        srv_count = col_r1.number_input("Service Count", min_value=0, value=int(cur.get("srv_count", 8)))
        serror_rate = col_r2.slider("SYN Error Rate", 0.0, 1.0, float(cur.get("serror_rate", 0.0)), 0.01)
        srv_serror_rate = col_r3.slider("Srv SYN Error Rate", 0.0, 1.0, float(cur.get("srv_serror_rate", 0.0)), 0.01)
        rerror_rate = col_r4.slider("REJ Error Rate", 0.0, 1.0, float(cur.get("rerror_rate", 0.0)), 0.01)

        col_r5, col_r6, col_r7, col_r8 = st.columns(4)
        srv_rerror_rate = col_r5.slider("Srv REJ Error Rate", 0.0, 1.0, float(cur.get("srv_rerror_rate", 0.0)), 0.01)
        same_srv_rate = col_r6.slider("Same Service Rate", 0.0, 1.0, float(cur.get("same_srv_rate", 1.0)), 0.01)
        diff_srv_rate = col_r7.slider("Diff Service Rate", 0.0, 1.0, float(cur.get("diff_srv_rate", 0.0)), 0.01)
        srv_diff_host_rate = col_r8.slider("Srv Diff Host Rate", 0.0, 1.0, float(cur.get("srv_diff_host_rate", 0.0)), 0.01)

        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        dst_host_count = col_d1.number_input("Dst Host Count", min_value=0, max_value=255, value=int(cur.get("dst_host_count", 180)))
        dst_host_srv_count = col_d2.number_input("Dst Host Srv Count", min_value=0, max_value=255, value=int(cur.get("dst_host_srv_count", 255)))
        dst_host_same_srv_rate = col_d3.slider("Dst Host Same Srv Rate", 0.0, 1.0, float(cur.get("dst_host_same_srv_rate", 1.0)), 0.01)
        dst_host_diff_srv_rate = col_d4.slider("Dst Host Diff Srv Rate", 0.0, 1.0, float(cur.get("dst_host_diff_srv_rate", 0.0)), 0.01)

        col_d5, col_d6, col_d7, col_d8 = st.columns(4)
        dst_host_same_src_port_rate = col_d5.slider("Dst Host Same Src Port", 0.0, 1.0, float(cur.get("dst_host_same_src_port_rate", 0.02)), 0.01)
        dst_host_srv_diff_host_rate = col_d6.slider("Dst Host Srv Diff Host", 0.0, 1.0, float(cur.get("dst_host_srv_diff_host_rate", 0.0)), 0.01)
        dst_host_serror_rate = col_d7.slider("Dst Host SYN Error Rate", 0.0, 1.0, float(cur.get("dst_host_serror_rate", 0.0)), 0.01)
        dst_host_srv_serror_rate = col_d8.slider("Dst Host Srv SYN Error", 0.0, 1.0, float(cur.get("dst_host_srv_serror_rate", 0.0)), 0.01)

        col_d9, col_d10 = st.columns(2)
        dst_host_rerror_rate = col_d9.slider("Dst Host REJ Error Rate", 0.0, 1.0, float(cur.get("dst_host_rerror_rate", 0.0)), 0.01)
        dst_host_srv_rerror_rate = col_d10.slider("Dst Host Srv REJ Error", 0.0, 1.0, float(cur.get("dst_host_srv_rerror_rate", 0.0)), 0.01)

        submit_btn = st.form_submit_button("⚡ Analyze Network Traffic", type="primary", use_container_width=True)

    if submit_btn:
        # Build clean input dictionary
        record_dict = {
            "duration": duration, "protocol_type": protocol_type, "service": service, "flag": flag,
            "src_bytes": src_bytes, "dst_bytes": dst_bytes, "land": land, "wrong_fragment": wrong_fragment,
            "urgent": urgent, "hot": hot, "num_failed_logins": num_failed_logins, "logged_in": logged_in,
            "num_compromised": num_compromised, "root_shell": root_shell, "su_attempted": su_attempted,
            "num_root": num_root, "num_file_creations": num_file_creations, "num_shells": num_shells,
            "num_access_files": num_access_files, "num_outbound_cmds": num_outbound_cmds,
            "is_host_login": is_host_login, "is_guest_login": is_guest_login, "count": count,
            "srv_count": srv_count, "serror_rate": serror_rate, "srv_serror_rate": srv_serror_rate,
            "rerror_rate": rerror_rate, "srv_rerror_rate": srv_rerror_rate, "same_srv_rate": same_srv_rate,
            "diff_srv_rate": diff_srv_rate, "srv_diff_host_rate": srv_diff_host_rate,
            "dst_host_count": dst_host_count, "dst_host_srv_count": dst_host_srv_count,
            "dst_host_same_srv_rate": dst_host_same_srv_rate, "dst_host_diff_srv_rate": dst_host_diff_srv_rate,
            "dst_host_same_src_port_rate": dst_host_same_src_port_rate,
            "dst_host_srv_diff_host_rate": dst_host_srv_diff_host_rate,
            "dst_host_serror_rate": dst_host_serror_rate, "dst_host_srv_serror_rate": dst_host_srv_serror_rate,
            "dst_host_rerror_rate": dst_host_rerror_rate, "dst_host_srv_rerror_rate": dst_host_srv_rerror_rate
        }
        record_df = pd.DataFrame([record_dict])

        # Run Prediction
        pred, mse, dist, deviations = predict_record(
            record_df, model, preprocessor, OFFICIAL_THRESHOLD, FEATURE_NAMES
        )

        st.markdown("---")
        st.markdown("### **Detection Result & Diagnostics**")

        res_col1, res_col2, res_col3, res_col4 = st.columns(4)

        if pred == "NORMAL":
            res_col1.markdown('<div class="badge-normal">🟢 NORMAL TRAFFIC</div>', unsafe_allow_html=True)
        else:
            res_col1.markdown('<div class="badge-intrusion">🚨 INTRUSION DETECTED</div>', unsafe_allow_html=True)

        res_col2.metric("Reconstruction MSE", f"{mse:.6f}", delta=f"{dist:+.6f}", delta_color="inverse")
        res_col3.metric("Official Threshold", f"{OFFICIAL_THRESHOLD:.6f}", help="95th percentile normal validation error")
        res_col4.metric(
            "Status vs Threshold",
            "ABOVE THRESHOLD" if mse > OFFICIAL_THRESHOLD else "BELOW THRESHOLD",
            delta="Anomaly flagged" if mse > OFFICIAL_THRESHOLD else "Pass",
            delta_color="inverse"
        )

        # Technical Clarification Note
        st.caption("ℹ️ **Note on Scoring:** Reconstruction error is not an attack probability. The Dense Autoencoder produces a continuous Mean Squared Error measuring deviation from normal baseline traffic, not a calibrated attack likelihood. Classification is strictly determined by whether the reconstruction MSE exceeds the selected threshold (0.12907493).")

        # Reconstruction-based Explanation (Top 5 Deviations)
        st.markdown("#### **Reconstruction-Based Anomaly Indicators (Top 5 Deviations)**")
        st.write("These features contributed the largest reconstruction deviations for this record (reconstruction-based anomaly indicators; not definitive causal proof of an attack mechanism):")
        st.dataframe(deviations, use_container_width=True, hide_index=True)


# ===========================================================================
# MODE 2: BATCH CSV DETECTION
# ===========================================================================
elif app_mode == "📁 Batch CSV Detection":
    st.markdown("### **Batch Network Traffic Intrusion Detection**")
    st.write("Upload a CSV file containing network connections to perform bulk intrusion detection.")

    # Template download for ease of testing
    col_t1, col_t2 = st.columns([3, 1])
    with col_t2:
        df_sample_data = load_test_sample_dataset()
        if df_sample_data is not None:
            sample_csv_data = df_sample_data[TRAFFIC_FEATURES].head(25).to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download 25-Row Sample CSV",
                data=sample_csv_data,
                file_name="nsl_kdd_sample_batch.csv",
                mime="text/csv",
                help="Download a ready-to-test CSV with valid NSL-KDD schema"
            )

    uploaded_file = st.file_uploader(
        "Choose a CSV file containing network traffic records",
        type=["csv"],
        help="Must contain the 41 standard NSL-KDD traffic feature columns."
    )

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.success(f"File loaded successfully: **{len(df_upload):,} records** uploaded.")

            # Validate Required Columns
            missing_cols = [c for c in TRAFFIC_FEATURES if c not in df_upload.columns]
            if missing_cols:
                st.error(f"❌ Upload Failed: Missing {len(missing_cols)} required feature columns:")
                st.code(", ".join(missing_cols))
                st.stop()

            # Optional Ground Truth preservation if present
            has_gt = "label" in df_upload.columns
            ground_truth = df_upload["label"].values if has_gt else None

            # Process & Run Inference
            with st.spinner("Processing batch traffic records with Autoencoder..."):
                t_start = time.time()
                X_batch_raw = df_upload[TRAFFIC_FEATURES]
                X_batch_proc = preprocessor.transform(X_batch_raw).astype(np.float32)
                X_batch_recon = model.predict(X_batch_proc, batch_size=512, verbose=0)

                # Vectorized MSE calculation
                batch_errors = np.mean(
                    np.square(X_batch_proc.astype(np.float64) - X_batch_recon.astype(np.float64)),
                    axis=1
                )
                t_elapsed = time.time() - t_start

            # Predictions
            batch_preds_num = (batch_errors > OFFICIAL_THRESHOLD).astype(int)
            batch_preds_txt = np.where(batch_preds_num == 1, "INTRUSION", "NORMAL")

            # Dashboard Summary
            n_total = len(df_upload)
            n_normal = int(np.sum(batch_preds_num == 0))
            n_intrusion = int(np.sum(batch_preds_num == 1))
            pct_normal = (n_normal / n_total) * 100
            pct_intrusion = (n_intrusion / n_total) * 100

            st.markdown("---")
            st.markdown("#### **Batch Detection Summary**")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Records", f"{n_total:,}")
            m2.metric("Normal Passed", f"{n_normal:,}", f"{pct_normal:.1f}%")
            m3.metric("Intrusions Detected", f"{n_intrusion:,}", f"{pct_intrusion:.1f}%", delta_color="inverse")
            m4.metric("Throughput", f"{n_total / max(t_elapsed, 1e-4):,.0f} req/s")
            m5.metric("Inference Time", f"{t_elapsed:.2f}s")

            # Result DataFrame
            df_results = df_upload.copy()
            df_results.insert(0, "prediction", batch_preds_txt)
            df_results.insert(1, "reconstruction_error", np.round(batch_errors, 6))
            df_results.insert(2, "threshold", OFFICIAL_THRESHOLD)

            # Filter options for the display table
            st.markdown("#### **Detailed Predictions Table**")
            filter_choice = st.radio(
                "Filter View:",
                ["All Records", "Intrusions Only (Alerts)", "Normal Only"],
                horizontal=True
            )

            if filter_choice == "Intrusions Only (Alerts)":
                display_df = df_results[df_results["prediction"] == "INTRUSION"]
            elif filter_choice == "Normal Only":
                display_df = df_results[df_results["prediction"] == "NORMAL"]
            else:
                display_df = df_results

            st.dataframe(display_df, use_container_width=True)

            # Export Button
            csv_export = df_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Predictions CSV",
                data=csv_export,
                file_name="intrusion_detection_results.csv",
                mime="text/csv",
                type="primary"
            )

        except Exception as e:
            st.error(f"❌ Error processing batch file: {e}")


# ===========================================================================
# MODE 3: BENCHMARK DATASET DEMO
# ===========================================================================
elif app_mode == "🧪 Benchmark Dataset Demo":
    st.markdown("### **Live Demonstration on Unseen KDDTest+ Benchmark**")
    st.write("Select verified records from the official test dataset to evaluate live Autoencoder inference against ground-truth benchmark labels.")

    df_test_full = load_test_sample_dataset()
    if df_test_full is None:
        st.warning("⚠️ Benchmark dataset file `data/KDDTest+.txt` not found.")
        st.stop()

    st.markdown("##### **Select Preset Evaluation Cases:**")
    d_c1, d_c2, d_c3, d_c4, d_c5 = st.columns(5)

    # Known indices in KDDTest+
    selected_idx = 0
    if d_c1.button("Case 1: Normal (Row 3)"):
        selected_idx = 3
    elif d_c2.button("Case 2: Neptune DoS (Row 1)"):
        selected_idx = 1
    elif d_c3.button("Case 3: Saint Probe (Row 4)"):
        selected_idx = 4
    elif d_c4.button("Case 4: Guess Password R2L (Row 8)"):
        selected_idx = 8
    elif d_c5.button("Case 5: Normal HTTP (Row 6)"):
        selected_idx = 6

    row_index = st.number_input(
        "Or enter any test sample index (0 to 999):",
        min_value=0, max_value=len(df_test_full) - 1, value=int(selected_idx)
    )

    sample_row = df_test_full.iloc[[row_index]].copy()
    actual_label = sample_row["label"].values[0].strip().lower()
    actual_category = map_attack_category(actual_label)
    is_actual_attack = actual_label != "normal"

    st.markdown("---")
    st.markdown(f"#### **Inspecting Record #{row_index}**")

    # Run inference on record
    pred, mse, dist, deviations = predict_record(
        sample_row, model, preprocessor, OFFICIAL_THRESHOLD, FEATURE_NAMES
    )

    col_res1, col_res2, col_res3, col_res4 = st.columns(4)

    if pred == "NORMAL":
        col_res1.markdown('<div class="badge-normal">🟢 PREDICTED: NORMAL</div>', unsafe_allow_html=True)
    else:
        col_res1.markdown('<div class="badge-intrusion">🚨 PREDICTED: INTRUSION</div>', unsafe_allow_html=True)

    col_res2.metric("Reconstruction MSE", f"{mse:.6f}", delta=f"{dist:+.6f}", delta_color="inverse")
    col_res3.metric("Threshold", f"{OFFICIAL_THRESHOLD:.6f}")

    # Ground Truth Comparison (Demo mode only)
    is_correct = (pred == "INTRUSION" and is_actual_attack) or (pred == "NORMAL" and not is_actual_attack)
    if is_correct:
        col_res4.success(f"✅ Prediction Matches Benchmark Label")
    else:
        col_res4.error(f"⚠️ Mismatch with Benchmark Label")

    st.info(f"**Benchmark Ground Truth:** Original Label = `{actual_label}` | Attack Family = `{actual_category}` | True Class = `{'ATTACK' if is_actual_attack else 'NORMAL'}`")

    st.markdown("##### **Top 5 Feature Reconstruction Deviations for this Record:**")
    st.dataframe(deviations, use_container_width=True, hide_index=True)


# ===========================================================================
# MODE 4: VALIDATED MODEL PERFORMANCE (PHASE 10 METRICS)
# ===========================================================================
elif app_mode == "📊 Validated Performance":
    st.markdown("### **Official Validated Model Performance**")
    st.write("These metrics represent the formal, leakage-free evaluation results on the untouched **NSL-KDD KDDTest+** benchmark (22,544 records).")

    # Metrics Cards
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown('<div class="metric-card"><div class="metric-val">83.35%</div><div class="metric-lbl">Accuracy</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown('<div class="metric-card"><div class="metric-val">96.05%</div><div class="metric-lbl">Precision</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown('<div class="metric-card"><div class="metric-val">73.79%</div><div class="metric-lbl">Recall (Detection)</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown('<div class="metric-card"><div class="metric-val">83.46%</div><div class="metric-lbl">F1 Score</div></div>', unsafe_allow_html=True)
    with k5:
        st.markdown('<div class="metric-card"><div class="metric-val">95.70%</div><div class="metric-lbl">ROC-AUC</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    k6, k7, k8, k9 = st.columns(4)
    with k6:
        st.markdown('<div class="metric-card"><div class="metric-val">4.01%</div><div class="metric-lbl">False Positive Rate</div></div>', unsafe_allow_html=True)
    with k7:
        st.markdown('<div class="metric-card"><div class="metric-val">95.99%</div><div class="metric-lbl">Specificity (TNR)</div></div>', unsafe_allow_html=True)
    with k8:
        st.markdown('<div class="metric-card"><div class="metric-val">9,469</div><div class="metric-lbl">True Positives (TP)</div></div>', unsafe_allow_html=True)
    with k9:
        st.markdown('<div class="metric-card"><div class="metric-val">9,322</div><div class="metric-lbl">True Negatives (TN)</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### **Evaluation Artifacts & Diagnostic Plots**")

    tab1, tab2, tab3 = st.tabs(["Confusion Matrix Heatmap", "ROC Curve", "Error Distribution"])

    with tab1:
        if os.path.exists(CM_IMAGE_PATH):
            st.image(CM_IMAGE_PATH, caption="Confusion Matrix on KDDTest+ (22,544 samples)", use_container_width=True)
        else:
            st.info("Confusion matrix image not found.")

    with tab2:
        if os.path.exists(ROC_IMAGE_PATH):
            st.image(ROC_IMAGE_PATH, caption="Receiver Operating Characteristic Curve (AUC = 95.70%)", use_container_width=True)
        else:
            st.info("ROC Curve image not found.")

    with tab3:
        if os.path.exists(DIST_IMAGE_PATH):
            st.image(DIST_IMAGE_PATH, caption="Reconstruction Error Distribution (Normal vs Attack)", use_container_width=True)
        else:
            st.info("Distribution plot not found.")


# ===========================================================================
# MODE 5: MODEL ARCHITECTURE & METHODOLOGY
# ===========================================================================
elif app_mode == "🧠 Model Architecture":
    st.markdown("### **Dense Autoencoder Architecture & Theory**")

    col_a1, col_a2 = st.columns([3, 2])

    with col_a1:
        st.markdown("""
        #### **Architecture Design**
        The model is a fully connected **Symmetric Dense Autoencoder** designed specifically for tabular network connection telemetry:

        ```
        Input Layer       : 77 Features (OneHotEncoded + StandardScaled)
        Encoder Layer 1   : Dense(64, activation='relu')
        Encoder Layer 2   : Dense(32, activation='relu')
        Latent Bottleneck : Dense(16, activation='relu')  [Compression: 79.2%]
        Decoder Layer 1   : Dense(32, activation='relu')
        Decoder Layer 2   : Dense(64, activation='relu')
        Reconstruction    : Dense(77, activation='linear')
        ```

        - **Optimizer:** Adam (Adaptive Moment Estimation)
        - **Loss Function:** Mean Squared Error (MSE)
        - **Total Parameters:** 45,785 (Trainable: **15,261**)
        - **Training Target:** Identity Reconstruction ($X_{\text{train}} \to X_{\text{train}}$)
        """)

    with col_a2:
        st.markdown("#### **Layer Parameter Breakdown**")
        arch_df = pd.DataFrame([
            {"Stage": "Input", "Layer": "InputLayer", "Output Dimension": 77, "Parameters": 0},
            {"Stage": "Encoder 1", "Layer": "Dense (ReLU)", "Output Dimension": 64, "Parameters": "4,992"},
            {"Stage": "Encoder 2", "Layer": "Dense (ReLU)", "Output Dimension": 32, "Parameters": "2,080"},
            {"Stage": "Bottleneck", "Layer": "Dense (ReLU)", "Output Dimension": 16, "Parameters": "528"},
            {"Stage": "Decoder 1", "Layer": "Dense (ReLU)", "Output Dimension": 32, "Parameters": "544"},
            {"Stage": "Decoder 2", "Layer": "Dense (ReLU)", "Output Dimension": 64, "Parameters": "2,112"},
            {"Stage": "Reconstruction", "Layer": "Dense (Linear)", "Output Dimension": 77, "Parameters": "5,005"},
        ])
        st.dataframe(arch_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### **Why Unsupervised Reconstruction for NIDS?**")
    st.markdown("""
    1. **Potential for Unseen Anomaly Detection:** Unlike signature-based classifiers that require prior attack labels, the reconstruction-based approach can potentially identify previously unseen anomalous traffic when its statistical behavior differs sufficiently from normal training baselines.
    2. **Principled Validation Calibration:** Rather than selecting an arbitrary threshold, the selected threshold (`0.12907493`) was derived as the **95th percentile of normal validation reconstruction errors** (Phase 8), establishing an empirical reference false-positive rate of ~5% on benign validation traffic.
    3. **Observed Benchmark Specificity:** On the untouched KDDTest+ benchmark, the model attained **96.05% Precision** and a **4.01% False Positive Rate** (95.99% Specificity), demonstrating high discrimination on volumetric and scanning anomalies while identifying payload-agnostic sessions (R2L) as a notable limitation.
    """)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("AI-Powered Network Intrusion Detection System | Built with Streamlit, TensorFlow/Keras & Scikit-Learn | NSL-KDD Benchmark")
