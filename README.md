# AI-Powered Network Intrusion Detection Using Dense Autoencoder

**MSc Computer Science (AIML) / Deep Learning Project**  
**Dataset:** NSL-KDD (`KDDTrain+.txt` and `KDDTest+.txt`)  
**Core Model:** Symmetric Dense Autoencoder (`77 → 64 → 32 → 16 → 32 → 64 → 77`)  
**Trainable Parameters:** 15,261  
**Framework:** TensorFlow / Keras & Scikit-Learn

---

## 1. Problem Statement
Traditional Network Intrusion Detection Systems (NIDS) primarily depend on signature-based detection, rendering them vulnerable to novel or zero-day network threats. This project develops an anomaly-based intrusion detection pipeline using a Deep Learning **Dense Autoencoder**. Trained strictly on normal, benign network telemetry, the model learns the low-dimensional structural manifold of legitimate network behavior and flags anomalous deviations at test time via reconstruction fidelity.

---

## 2. Dataset
The project evaluates on the widely benchmarked **NSL-KDD** dataset, an improved version of KDD Cup 99 that removes duplicate records to eliminate evaluation bias:
- **`data/KDDTrain+.txt`:** 125,973 total training records (67,343 Normal, 58,630 Attacks).
- **`data/KDDTest+.txt`:** 22,544 total unseen test records (9,711 Normal, 12,833 Attacks across DoS, Probe, R2L, and U2R categories).
- **Schema:** 41 traffic features (3 categorical: `protocol_type`, `service`, `flag`; 38 numerical features) plus `label` and `difficulty`.

---

## 3. Leakage-Free Preprocessing Pipeline
To guarantee complete separation between training and evaluation:
1. **Feature Separation:** The metadata columns (`label`, `difficulty`, `binary_label`) are isolated and excluded from model inputs.
2. **ColumnTransformer:**
   - **Categorical (3 features):** Encoded with `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
   - **Numerical (38 features):** Normalized with `StandardScaler()`.
3. **Strict Zero-Leakage Guarantee:**
   - The preprocessor is fitted **exclusively** on the normal training split ($X_{\text{train\_normal}}$).
   - Validation ($X_{\text{val\_normal}}$) and test ($X_{\text{test}}$) splits are processed solely via `.transform()`.
   - Resulting input feature space: **77 processed numerical features**.

---

## 4. Normal-Only Training Strategy
- In an unsupervised one-class anomaly detection setup, the model must never observe attack records during training.
- Normal records from `KDDTrain+` (67,343 records) were split with a fixed random seed (`random_state=42`):
  - **Training Split (80%):** 53,874 records (Normal traffic only).
  - **Validation Split (20%):** 13,469 records (Normal traffic only).
- **Learning Objective:** Identity reconstruction ($X_{\text{train\_normal}} \to X_{\text{train\_normal}}$).

---

## 5. Autoencoder Architecture
A symmetric, fully connected Dense Autoencoder tailored for flat tabular telemetry:

```text
Input Layer        : 77 Features (Preprocessed)
Encoder Layer 1    : Dense(64, activation='relu')
Encoder Layer 2    : Dense(32, activation='relu')
Latent Bottleneck  : Dense(16, activation='relu')   [Compression Ratio: 79.2%]
Decoder Layer 1    : Dense(32, activation='relu')
Decoder Layer 2    : Dense(64, activation='relu')
Reconstruction     : Dense(77, activation='linear')
```

- **Optimizer:** Adam
- **Loss Function:** Mean Squared Error (MSE)
- **Trainable Parameters:** 15,261
- **Convergence:** Early stopping restored best validation loss of `0.313887` at epoch 8.

---

## 6. Reconstruction Error as Anomaly Score
For each input connection $x$, the anomaly score is computed as the per-sample Mean Squared Error across all 77 processed features:

$$\text{MSE} = \frac{1}{77} \sum_{j=1}^{77} (x_j - \hat{x}_j)^2$$

- **Normal traffic:** Low MSE (the model reconstructs familiar benign telemetry faithfully).
- **Anomalous traffic:** High MSE (unseen structural deviations fail compression through the bottleneck).

*Note: Reconstruction error represents continuous statistical deviation, not a calibrated attack probability.*

---

## 7. Selected Anomaly Threshold
Rather than using an arbitrary cutoff, the decision boundary was derived empirically from normal validation reconstruction errors:
- **Selected Threshold:** `0.12907493`
- **Methodology:** **95th percentile of normal validation errors** (Phase 8).
- **Target FPR:** Calibrated to yield an expected false-positive rate of ~5% on benign traffic.

---

## 8. Classification Rule
$$\text{Reconstruction MSE} \le 0.12907493 \implies \text{NORMAL (0)}$$
$$\text{Reconstruction MSE} > 0.12907493 \implies \text{INTRUSION (1)}$$

---

## 9. Validated Benchmark Performance (KDDTest+)
Evaluated on all 22,544 untouched records of `KDDTest+` (Phase 10 confirmed results):

| Metric | Score | Context / Operational Value |
|---|---|---|
| **Accuracy** | **83.35%** | 18,791 / 22,544 test connections correctly classified |
| **Precision** | **96.05%** | High alarm confidence; 9,469 / 9,858 flagged alerts are true intrusions |
| **Recall (TPR)** | **73.79%** | 9,469 / 12,833 total attacks detected without prior signature training |
| **F1 Score** | **83.46%** | Harmonic mean balancing high precision and sensitivity |
| **ROC-AUC** | **95.70%** | Measured using continuous per-sample reconstruction MSE scores |
| **False Positive Rate** | **4.01%** | 389 / 9,711 normal records flagged; closely tracks the 5% target calibration |
| **Specificity (TNR)** | **95.99%** | 9,322 / 9,711 normal connections correctly permitted |

### Confusion Matrix
- **True Negatives (TN):** `9,322`
- **False Positives (FP):** `389`
- **False Negatives (FN):** `3,364`
- **True Positives (TP):** `9,469`

### Attack-Family Detection Breakdown (Phase 11)
- **DoS (Denial of Service):** **84.37%** detected (6,292 / 7,458)
- **Probe (Surveillance / Scanning):** **81.54%** detected (1,974 / 2,421)
- **U2R (User-to-Root):** **65.67%** detected (44 / 67)
- **R2L (Remote-to-Local):** **40.15%** detected (1,159 / 2,887)

---

## 10. Interactive Streamlit Application
A full-featured web dashboard is provided in `app/streamlit_app.py` for demonstrations and viva presentations:
- **Single Record Detection:** Complete 41-feature form with quick preset buttons (Normal, Neptune DoS, Portsweep, Buffer Overflow, Password Guessing).
- **Explainability:** Displays top 5 feature reconstruction deviations.
- **Batch CSV Detection:** Bulk upload, column validation, live throughput calculation, filterable table, and CSV export.
- **Benchmark Demo:** Real-time inference on genuine `KDDTest+` records with ground-truth comparison.
- **Performance & Theory Dashboards:** Displays confirmed metrics, confusion matrix heatmap, ROC curve, and layer parameter breakdown.

---

## 11. Model Limitations
1. **Header-Level Telemetry:** The NSL-KDD benchmark summarizes IP connection metadata rather than full payload byte streams. Intrusions occurring within valid interactive application sessions (such as R2L password guessing) mimic legitimate user traffic patterns and produce lower reconstruction deviations.
2. **Fixed Bottleneck:** Subtle intrusions that do not disrupt host or service rate statistics incur small reconstruction errors and may fall below the global threshold.

---

## 12. How to Run the Project

### Environment Setup
```bash
pip install -r requirements.txt
```

### Reproduce Pipeline Phases
```bash
# 1. Dataset exploration
python src/explore_dataset.py

# 2. Leakage-free preprocessing pipeline
python src/preprocess_pipeline.py

# 3. Model verification & training
python src/train.py

# 4. Reconstruction error & threshold selection
python src/reconstruction_error.py
python src/threshold_selection.py

# 5. Test set inference & evaluation
python src/intrusion_detection.py
python src/evaluate.py

# 6. Detailed error analysis
python src/error_analysis.py
```

### Launch Interactive Web Application
```bash
streamlit run app/streamlit_app.py
```
Open `http://localhost:8501` in your browser.
