# Network Intrusion Detection System — Streamlit Application

This directory contains the interactive web dashboard for the **AI-Powered Network Intrusion Detection System (NIDS)** built with Streamlit, TensorFlow/Keras, and Scikit-Learn.

---

## 1. How to Launch the Application

Ensure the Python environment with required dependencies is active, then execute:

```bash
streamlit run app/streamlit_app.py
```

Or from within the project root directory:

```bash
python -m streamlit run app/streamlit_app.py
```

The application will start and open automatically in your default browser at `http://localhost:8501`.

---

## 2. Required Artifacts & Files

The application requires the following existing project artifacts:

| Artifact | Location | Purpose |
|---|---|---|
| Trained Autoencoder | `models/autoencoder.keras` | Evaluates network traffic reconstruction ($77 \to 16 \to 77$). |
| Fitted Preprocessor | `models/preprocessor.joblib` | Encodes categorical & scales numerical features without leakage. |
| Anomaly Threshold | `results/anomaly_threshold.json` | Stores official threshold ($0.12907493$). |
| Benchmark Dataset | `data/KDDTest+.txt` | Provides live samples for the test benchmark demonstration. |
| Performance Metrics | `results/final_metrics.json` | Displays validated Phase 10 benchmark results. |
| Diagnostic Plots | `results/*.png` | Visualizes confusion matrix, ROC curve, and error distributions. |

---

## 3. Input Data Format

The application expects input records matching the **41 standard NSL-KDD traffic features**:

- **3 Categorical Features:**
  - `protocol_type` (`tcp`, `udp`, `icmp`)
  - `service` (`http`, `private`, `smtp`, `domain_u`, `ftp_data`, etc.)
  - `flag` (`SF`, `S0`, `REJ`, `RSTR`, `RSTO`, etc.)
- **38 Numerical Features:**
  - Connection durations, byte counts (`src_bytes`, `dst_bytes`), login statuses (`logged_in`), error rates (`serror_rate`, `rerror_rate`), and host-based traffic rates (`dst_host_count`, `dst_host_srv_count`, etc.).

---

## 4. Features & Modes

### A. Single Record Detection
- **Interactive Form:** Allows manual entry of all 41 traffic parameters.
- **Quick Preset Demonstrations:** Single-click buttons to load realistic connection profiles:
  - Typical Normal (HTTP browse)
  - Neptune SYN Flood (DoS)
  - Port Scanning (Probe)
  - Privilege Escalation (U2R)
  - Password Guessing (R2L)
- **Real-Time Classification:** Displays whether the connection is `NORMAL` or `INTRUSION DETECTED`, alongside the reconstruction MSE and distance from threshold.
- **Explainability:** Displays the top 5 features with the largest reconstruction deviations between the input and Autoencoder reconstruction.

### B. Batch CSV Detection
- **Bulk Upload:** Upload any CSV file containing the 41 network traffic features.
- **Validation:** Automatic check for missing or invalid columns.
- **Summary Dashboard:** Displays total records processed, normal passed, intrusions flagged, and processing throughput.
- **Exportable Results:** Download predictions as `intrusion_detection_results.csv`.
- **Sample Download:** Provides a quick 25-row sample CSV button to immediately test batch inference.

### C. Benchmark Dataset Demo
- Evaluates real test records directly from the official unseen `KDDTest+.txt` benchmark.
- Compares live predictions against ground-truth labels for demonstration and academic verification.

### D. Validated Performance Dashboard
- Visualizes official Phase 10 test results:
  - **Accuracy:** `83.35%`
  - **Precision:** `96.05%`
  - **Recall:** `73.79%`
  - **F1-Score:** `83.46%`
  - **ROC-AUC:** `95.70%`
  - **False Positive Rate:** `4.01%`
  - **Specificity:** `95.99%`

### E. Model Architecture & Theory Guide
- Full layer breakdown of the $77 \to 64 \to 32 \to 16 \to 32 \to 64 \to 77$ Dense Autoencoder.
- Explains the unsupervised reconstruction anomaly paradigm and how one-class learning can potentially identify previously unseen anomalous traffic when its statistical behavior differs sufficiently from normal training data.

---

## 5. Interpretation of Reconstruction Error

- The Autoencoder learns an identity mapping strictly on **normal network traffic**.
- **Reconstruction MSE** measures how accurately the model can compress and regenerate the input:
  $$\text{MSE} = \frac{1}{77} \sum_{j=1}^{77} (x_j - \hat{x}_j)^2$$
- **Low MSE:** The connection closely matches learned patterns of benign traffic.
- **High MSE:** The connection deviates from normal network behavior, indicating a structural anomaly or potential cyberattack.

> [!NOTE]
> Reconstruction error is not an attack probability. The Autoencoder produces an unsupervised reconstruction error (Mean Squared Error), not a calibrated probability score. The system does not output an "attack probability" because anomaly detection measures statistical deviation from benign baseline patterns, not class membership likelihood.

---

## 6. Meaning of the Threshold

- **Selected Anomaly Threshold:** `0.12907493` (Official threshold used in this experiment)
- **Derivation:** Empirically calculated as the **95th percentile of normal validation reconstruction errors** (Phase 8).
- **Operational Interpretation:** By definition, ~5% of normal validation traffic exceeds this threshold, targeting an expected benchmark False Positive Rate of ~5%. On the untouched KDDTest+ benchmark, the observed FPR is **4.01%** (Specificity = **95.99%**).
- **Classification Rule:**
  $$\text{MSE} \le 0.12907493 \implies \text{NORMAL}$$
  $$\text{MSE} > 0.12907493 \implies \text{INTRUSION}$$

---

## 7. Model Limitations

1. **Header Metadata Scope:** The NSL-KDD benchmark summarizes IP connection headers and packet counts rather than full deep packet payloads. Attacks that occur over legitimate user sessions without disrupting connection frequencies (e.g., R2L password guessing) produce lower reconstruction deviations.
2. **Tabular Focus:** The model is specialized for flat tabular telemetry and does not capture long-term sequential state across hours or days.
