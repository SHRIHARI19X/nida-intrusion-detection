# Network Intrusion Detection Using Autoencoder

**MSc CS AIML — Deep Learning Project**
**Master instruction and documentation file for Google Antigravity**

---

## 0. How to Use This File

This file is the single source of truth for the project. The Antigravity agent must read it fully before starting and follow it phase by phase.

- The only deliverable at this stage is **this file (`Project.md`)**. Source code, `requirements.txt`, notebooks and a PPT are created later, phase by phase, as this plan directs.
- **No dataset statistic, model result, accuracy or other measurement is stated in this file.** Every such value must come from actual execution and be recorded in the phase log (Section 8).

**Primary objective:** Build a complete, working, reproducible network intrusion/anomaly detection system on the **NSL-KDD** dataset using a **Dense Autoencoder** implemented in **TensorFlow/Keras**.

---

## 1. Two Different Kinds of Models (Do Not Confuse)

### A) Antigravity AI Agent Models
These are the AI models that plan, implement, review, debug and verify the project.

| Phase | Name | Antigravity AI model |
|-------|------|----------------------|
| 0 | Planning | Claude Opus 4.6 (Thinking) |
| 1 | Project Setup | Gemini 3.8 Flash Medium |
| 2 | Dataset Acquisition & Exploration | Gemini 3.8 Flash Medium |
| 3 | Data Preprocessing | Claude Sonnet 4.6 (Thinking) |
| 4 | Normal-Traffic Preparation | Claude Sonnet 4.6 (Thinking) |
| 5 | Autoencoder Architecture | Claude Opus 4.6 (Thinking) |
| 6 | Autoencoder Training | Gemini 3.8 Flash Medium |
| 6 (Review) | Training review | Claude Sonnet 4.6 (Thinking) |
| 7 | Reconstruction Error | Claude Sonnet 4.6 (Thinking) |
| 8 | Threshold Selection | Claude Sonnet 4.6 (Thinking) |
| 9 | Intrusion Detection | Gemini 3.8 Flash Medium |
| 10 | Evaluation | Gemini 3.8 Flash Medium |
| 10 (Review) | Evaluation review | Claude Sonnet 4.6 (Thinking) |
| 11 | Error Analysis | Claude Opus 4.6 (Thinking) |
| 12 | Final Prediction Demo | Gemini 3.8 Flash Medium |
| 13 | Optional Streamlit UI | Gemini 3.8 Flash Medium |
| 14 | Final Code Review | Claude Opus 4.6 (Thinking) |
| 15 | End-to-End Verification | Gemini 3.8 Flash Medium + Claude Opus 4.6 (Thinking) |

**Models expected to be available in Antigravity:**
- Gemini 3.8 Flash Medium
- Gemini 3.7 Flash Medium
- Gemini 3.6 Flash Medium
- Gemini 3.1 Pro Low
- Claude Sonnet 4.6 (Thinking)
- Claude Opus 4.6 (Thinking)
- GPT-OSS 120B (Medium)

**Substitution rule:** If a named model is unavailable, use the closest available model and **explicitly record the substitution** (original model, substitute model, reason) in the phase log.

### B) The Actual Deep Learning Model

> **Dense Autoencoder** — framework: **TensorFlow / Keras**

This is the main and required model. **Do NOT replace it** with CNN, LSTM, GRU, Transformer, GAN, Random Forest or XGBoost unless explicitly requested later.

---

## 2. Project Concept

The system learns the representation of **normal** network traffic using an Autoencoder. The model reconstructs its input; the **reconstruction error** is the **anomaly score**. A threshold derived **only from normal validation reconstruction errors** classifies traffic as:

- `0` = Normal
- `1` = Intrusion / Attack

**Conceptual flow**

```
NSL-KDD
  ↓
Data Exploration
  ↓
Preprocessing
  ↓
OneHotEncoder + StandardScaler
  ↓
Normal Traffic Training Data
  ↓
Dense Autoencoder
  ↓
Reconstruction
  ↓
Reconstruction Error
  ↓
Anomaly Threshold
  ↓
Normal / Intrusion
  ↓
Evaluation
```

---

## 3. Dataset

**Dataset:** NSL-KDD

**Primary files:**
- `KDDTrain+.txt`
- `KDDTest+.txt`

**Important:** Actual dataset dimensions, class counts and label distributions **must be obtained by execution** and must never be invented in advance. Do not hard-code dataset statistics unless verified from the actual files.

**Important columns/features include:**
- `duration`
- `protocol_type`
- `service`
- `flag`
- `src_bytes`
- `dst_bytes`
- login / host / error / count related features
- `label`
- `difficulty`

**Categorical features:**
- `protocol_type`
- `service`
- `flag`

---

## 4. Phase-by-Phase Implementation Plan

Every phase lists: phase number, name, Antigravity AI model, actual Deep Learning model involved, objective, tasks, expected output, verification checks, and conditions for moving on.

---

### PHASE 0 — PLANNING

- **Antigravity AI model:** Claude Opus 4.6 (Thinking)
- **Actual Deep Learning model involved:** None yet
- **Objective:** Produce a concise, internally consistent implementation plan before any code is written.
- **Tasks:**
  - Understand the project requirements.
  - Finalize architecture and workflow.
  - Confirm tools and folder structure.
  - Identify data-leakage risks.
  - Define the evaluation methodology.
- **Expected output:** A written plan (recorded in the phase log) covering architecture, workflow, leakage risks and evaluation method.
- **Verification checks:** The plan exists, is concise, and has no internal contradictions with this file.
- **Move to next phase when:** The plan is complete and consistent.

---

### PHASE 1 — PROJECT SETUP

- **Antigravity AI model:** Gemini 3.8 Flash Medium
- **Actual Deep Learning model involved:** None yet
- **Objective:** Create the project skeleton and a working Python environment.
- **Tasks:**
  - Create the recommended folder structure:

```
Network-Intrusion-Detection/
├── data/
├── src/
├── models/
├── results/
├── notebooks/
├── app/
├── requirements.txt
└── README.md
```

  - Set up an environment with the expected libraries:
    - numpy
    - pandas
    - scikit-learn
    - tensorflow
    - matplotlib
    - seaborn
    - joblib
    - streamlit
  - Record Python and TensorFlow versions.
- **Expected output:** Folder structure, `requirements.txt`, working environment.
- **Verification checks:**
  - Python environment works.
  - All required libraries import successfully.
- **Move to next phase when:** All imports succeed.

---

### PHASE 2 — DATASET ACQUISITION & EXPLORATION

- **Antigravity AI model:** Gemini 3.8 Flash Medium
- **Actual Deep Learning model involved:** None yet
- **Objective:** Obtain NSL-KDD and understand its real structure.
- **Tasks:**
  - Obtain NSL-KDD and place `KDDTrain+.txt` and `KDDTest+.txt` in `data/`.
  - Verify the files exist and are readable.
  - Load with Pandas and assign the correct column names (including `label` and `difficulty`).
  - Inspect shape, data types, missing values, duplicates, label values and feature types.
  - Record all statistics from actual execution only.
- **Expected output:** An exploration summary file in `results/` containing the actual statistics.
- **Verification checks:**
  - Dataset loads correctly.
  - Column count and names match the expected schema.
  - Actual statistics are recorded; none are fabricated.
- **Move to next phase when:** Both files load correctly and actual statistics are saved.

---

### PHASE 3 — DATA PREPROCESSING

- **Antigravity AI model:** Claude Sonnet 4.6 (Thinking)
- **Actual Deep Learning model involved:** Not trained yet
- **Objective:** Build a leakage-free preprocessing pipeline.
- **Tasks:**
  - Use `OneHotEncoder(handle_unknown='ignore')` for categorical features.
  - Use `StandardScaler` for numerical features.
  - Combine them with `ColumnTransformer`.
  - Convert labels: `0` = Normal, `1` = Intrusion / Attack (stored as `binary_label`).
  - **Remove from model input:** `label`, `difficulty`, `binary_label`.
  - **CRITICAL:** Fit the pipeline on training data only. Never fit it on test data. Transform test data with the already-fitted pipeline.
  - Save the fitted pipeline with `joblib`.
- **Expected output:** A fitted, saved preprocessing pipeline and a transformed numeric training matrix.
- **Verification checks:**
  - Encoded data is fully numeric.
  - Test data is transformed using the already-fitted pipeline (no `fit` on test data).
  - Excluded columns are absent from the model input.
  - No obvious data leakage exists.
- **Move to next phase when:** All checks pass.

---

### PHASE 4 — NORMAL TRAFFIC PREPARATION

- **Antigravity AI model:** Claude Sonnet 4.6 (Thinking)
- **Actual Deep Learning model involved:** Dense Autoencoder will be used next
- **Objective:** Prepare normal-only training and validation data.
- **Tasks:**
  - Select normal traffic from the **training data** as the Autoencoder training data.
  - Split normal data **80% training / 20% validation** with `random_state=42`.
  - Do **NOT** train the Autoencoder on attack samples.
  - Do **NOT** use the test set for model training or threshold selection.
- **Expected output:** `X_train_normal` and `X_val_normal` arrays.
- **Verification checks:**
  - Actual sample counts are printed and checked.
  - Split ratio is approximately 80/20.
  - No attack samples are in either split.
- **Move to next phase when:** Counts are printed and verified.

---

### PHASE 5 — AUTOENCODER ARCHITECTURE

- **Antigravity AI model:** Claude Opus 4.6 (Thinking)
- **Actual Deep Learning model involved:** Dense Autoencoder (TensorFlow / Keras)
- **Objective:** Define the Dense Autoencoder.
- **Recommended architecture:**

```
Input
 ↓
Dense(64, activation='relu')
 ↓
Dense(32, activation='relu')
 ↓
Dense(16, activation='relu')
 ↓
Latent Representation
 ↓
Dense(32, activation='relu')
 ↓
Dense(64, activation='relu')
 ↓
Dense(input_dimension, activation='linear')
 ↓
Reconstructed Input
```

- **Tasks:**
  - Determine `input_dimension` **dynamically** after preprocessing. Do not hard-code it.
  - Optimizer: Adam. Loss: Mean Squared Error (MSE).
- **Expected output:** A compiled Keras model and a model-building function.
- **Verification checks:**
  - Model builds successfully.
  - `model.summary()` is available.
  - Output dimension matches the processed input dimension.
- **Move to next phase when:** All checks pass.

---

### PHASE 6 — AUTOENCODER TRAINING

- **Antigravity AI model:** Gemini 3.8 Flash Medium
- **Review model:** Claude Sonnet 4.6 (Thinking)
- **Actual Deep Learning model involved:** Dense Autoencoder
- **Objective:** Train the Autoencoder to reconstruct normal traffic.
- **Tasks:**
  - Train with `X_train_normal → X_train_normal`, using `X_val_normal → X_val_normal` for validation.
  - Recommended starting configuration: `epochs = 50`, `batch_size = 256`.
  - Use `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`.
  - Set NumPy and TensorFlow seeds.
  - Generate `results/loss_curve.png`.
  - Save the trained model to `models/`.
  - Record the **actual** epochs trained, final training loss and final validation loss.
- **Expected output:** Saved model, `results/loss_curve.png`, recorded training log.
- **Verification checks:**
  - Training completes without errors.
  - Loss values are produced.
  - Loss graph is generated.
  - Claude Sonnet 4.6 (Thinking) reviews the implementation for leakage or obvious errors.
- **Move to next phase when:** Training completes and the review finds no unresolved issues.

---

### PHASE 7 — RECONSTRUCTION ERROR

- **Antigravity AI model:** Claude Sonnet 4.6 (Thinking)
- **Actual Deep Learning model involved:** Trained Dense Autoencoder
- **Objective:** Compute the anomaly score.
- **Tasks:**
  - For each **normal validation** sample, reconstruct it and compute the per-sample MSE:
    `error = mean((original - reconstructed)^2)` (mean over features, per sample).
  - Use reconstruction error as the anomaly score.
  - Generate `results/reconstruction_error.png`.
- **Expected output:** An array of per-sample errors and the plot.
- **Verification checks:**
  - Error values come from actual model inference.
  - One error value per sample; no NaN or infinite values.
- **Move to next phase when:** Error values and plot are produced.

---

### PHASE 8 — ANOMALY THRESHOLD

- **Antigravity AI model:** Claude Sonnet 4.6 (Thinking)
- **Actual Deep Learning model involved:** Trained Dense Autoencoder
- **Objective:** Derive a principled detection threshold.
- **Tasks:**
  - Determine the threshold **ONLY** from normal validation reconstruction errors.
  - Recommended initial method: `threshold = 95th percentile of normal validation errors`.
  - Classification rule: `error <= threshold → Normal`, `error > threshold → Intrusion`.
  - Do **NOT** choose an arbitrary threshold (such as 0.05).
  - Do **NOT** use attack or test data to determine the threshold.
  - Store the threshold (e.g., JSON in `models/`).
- **Expected output:** A stored threshold value.
- **Verification checks:** The actual threshold is printed and stored, and was computed only from normal validation errors.
- **Move to next phase when:** Threshold is stored and verified.

---

### PHASE 9 — INTRUSION DETECTION

- **Antigravity AI model:** Gemini 3.8 Flash Medium
- **Actual Deep Learning model involved:** Trained Dense Autoencoder
- **Objective:** Generate predictions on the test set.
- **Flow:**

```
Test traffic → preprocessing → autoencoder → reconstruction
 → reconstruction error → threshold → prediction
```

- **Tasks:**
  - Apply the **same fitted** preprocessing pipeline to `KDDTest+`.
  - Compute reconstruction errors and apply the stored threshold.
  - Predictions: `0` = Normal, `1` = Intrusion.
- **Expected output:** Predictions and reconstruction errors for the full test set.
- **Verification checks:**
  - Test data is never used for retraining.
  - Predictions are generated for the full intended test data (count matches test set size).
- **Move to next phase when:** Predictions are complete.

---

### PHASE 10 — EVALUATION

- **Antigravity AI model:** Gemini 3.8 Flash Medium
- **Review model:** Claude Sonnet 4.6 (Thinking)
- **Actual Deep Learning model involved:** Dense Autoencoder
- **Objective:** Measure real performance on the test set.
- **Tasks:**
  - Calculate Accuracy, Precision, Recall, F1 Score and ROC-AUC.
  - For ROC-AUC, use the **continuous** reconstruction error scores (not binary predictions).
  - Generate `results/confusion_matrix.png`, `results/roc_curve.png` and `results/reconstruction_error.png`.
  - Save metrics to `results/` (e.g., JSON).
- **Expected output:** Metrics file and three plots.
- **Verification checks:**
  - All numbers are actual executed results.
  - Metrics are never fabricated or manually altered.
  - Claude Sonnet 4.6 (Thinking) reviews the evaluation logic.
- **Move to next phase when:** Metrics and plots are generated and reviewed.

---

### PHASE 11 — ERROR ANALYSIS

- **Antigravity AI model:** Claude Opus 4.6 (Thinking)
- **Actual Deep Learning model involved:** Dense Autoencoder
- **Objective:** Understand where and why the model errs.
- **Tasks:**
  - Analyze false positives, false negatives, the reconstruction-error distribution, threshold behavior and any unusual model behavior.
  - If results are poor, investigate the technical cause.
  - Do **not** modify the test set to improve metrics.
  - Any improvement (e.g., architecture or threshold-percentile change) must be legitimate, based on validation data only, and documented.
- **Expected output:** A written error-analysis section in `results/` or `README.md`.
- **Verification checks:** Any change is legitimate, documented, and re-run with fresh metrics.
- **Move to next phase when:** Analysis is documented.

---

### PHASE 12 — FINAL PREDICTION DEMO

- **Antigravity AI model:** Gemini 3.8 Flash Medium
- **Actual Deep Learning model involved:** Trained Dense Autoencoder
- **Objective:** Provide a reusable prediction function.
- **Tasks:**
  - Create `detect_intrusion(sample)`.
  - Flow: raw sample → preprocessing → autoencoder → reconstruction error → threshold → result.
  - Output: the reconstruction error and `NORMAL TRAFFIC` or `INTRUSION DETECTED`.
  - Demonstrate on sample records from the dataset.
- **Expected output:** Prediction function and a sample demonstration.
- **Verification checks:** The same preprocessing pipeline, model and threshold used in evaluation are loaded from disk and used here.
- **Move to next phase when:** The demo runs correctly.

---

### PHASE 13 — OPTIONAL STREAMLIT UI

- **Antigravity AI model:** Gemini 3.8 Flash Medium
- **Actual Deep Learning model involved:** Trained Dense Autoencoder
- **Objective:** A simple demonstration interface.
- **Condition:** Implement only after the core ML pipeline works.
- **Tasks / features:**
  - CSV upload
  - preprocessing
  - intrusion detection
  - reconstruction error
  - threshold
  - prediction
  - Do not overcomplicate the UI.
- **Expected output:** `app/` Streamlit app.
- **Verification checks:** The app launches and produces predictions from an uploaded CSV using the saved artifacts.
- **Move to next phase when:** The app works, or this phase is explicitly skipped and logged.

---

### PHASE 14 — FINAL CODE REVIEW

- **Antigravity AI model:** Claude Opus 4.6 (Thinking)
- **Actual Deep Learning model involved:** Dense Autoencoder
- **Objective:** Whole-codebase review.
- **Review for:**
  - correctness
  - maintainability
  - data leakage
  - training strategy
  - model architecture
  - threshold logic
  - evaluation logic
  - reproducibility
  - saving/loading model
  - prediction function
- **Tasks:** Fix legitimate problems found and re-run affected parts.
- **Expected output:** A review report and applied fixes.
- **Verification checks:** No unresolved critical issues remain.
- **Move to next phase when:** Review is complete and fixes are verified.

---

### PHASE 15 — END-TO-END VERIFICATION

- **Antigravity AI models:** Gemini 3.8 Flash Medium + Claude Opus 4.6 (Thinking)
- **Actual Deep Learning model involved:** Dense Autoencoder
- **Objective:** Prove the whole pipeline works from scratch.
- **Tasks:** Run the full pipeline from start to finish and verify:
  - dataset loads
  - preprocessing works
  - normal-only training works
  - model trains
  - loss graph is generated
  - reconstruction error works
  - threshold works
  - predictions work
  - confusion matrix works
  - ROC curve works
  - metrics are produced
  - model saves and reloads
  - prediction function works
  - optional Streamlit app works, if implemented
- **Expected output:** A final verification report.
- **Verification checks:** All items above pass.
- **Completion rule:** Do **not** declare the project complete until the full pipeline is verified.

---

## 5. Non-Negotiable Data Integrity Rules

**Never:**
- fabricate results
- fabricate metrics
- fabricate dataset statistics
- train on test data
- fit preprocessing using test data
- use test data to determine the threshold
- modify test labels
- remove difficult test samples to improve results
- report training metrics as final test metrics

All reported values must come from actual execution.

---

## 6. Reproducibility

- Use `random_state=42` where applicable.
- Set NumPy and TensorFlow seeds where appropriate.

**Record (from actual execution):**
- Python version
- TensorFlow version
- dataset dimensions
- original feature count
- processed feature count
- model architecture
- epochs actually trained
- batch size
- threshold
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

## 7. Project Outputs

The final project should produce:
- working source code
- preprocessing pipeline
- trained Dense Autoencoder
- saved model
- saved preprocessing pipeline where needed
- loss curve
- reconstruction-error plot
- confusion matrix
- ROC curve
- evaluation metrics
- sample prediction
- README
- optional Streamlit demonstration

### PPT — Do Not Create Yet

Do not create a PPT during the initial implementation. Only after the implementation is fully working and verified, prepare data for a **10-slide PPT**:

1. Title
2. Problem & Objective
3. Dataset
4. System Architecture
5. Data Preprocessing
6. Autoencoder Architecture
7. Model Training
8. Anomaly Detection
9. Results
10. Conclusion & Future Scope

The PPT must use only genuine results from the implementation.

---

## 8. Antigravity Workflow Rule

Always follow:

```
PLAN → IMPLEMENT → RUN → VERIFY → REVIEW → FIX → RUN AGAIN → DOCUMENT → NEXT PHASE
```

Do not generate one giant untested codebase.

**At the end of every phase, record the following (Phase Log template):**

| Field | Entry |
|-------|-------|
| Phase status | |
| Antigravity AI model used (note any substitution) | |
| Actual Deep Learning model involved | |
| Implementation completed | |
| Tests performed | |
| Actual results | |
| Errors found | |
| Fixes applied | |
| Passed verification? (Yes/No) | |

---

## 9. Final Project Statement

The completed system must implement:

```
NSL-KDD
  ↓
Preprocessing
  ↓
Normal Traffic
  ↓
Dense Autoencoder
  ↓
Reconstruction Error
  ↓
95th Percentile Threshold
  ↓
Normal / Intrusion
  ↓
Evaluation
```

The project must remain focused on a practical, academically explainable **Dense Autoencoder-based network intrusion detection** implementation.
