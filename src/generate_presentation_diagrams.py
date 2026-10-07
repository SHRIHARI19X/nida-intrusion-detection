"""
generate_presentation_diagrams.py
---------------------------------
Generates dedicated, publication-quality presentation diagrams for Phase 14:
  1. Data Preprocessing & Zero-Leakage Pipeline Diagram
  2. End-to-End System Workflow Flowchart
  3. Dense Autoencoder Layer-by-Layer Architecture Diagram
  4. Streamlit Application Workflow Diagram
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def create_preprocessing_diagram():
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    # Background
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")

    title_box = dict(boxstyle="round,pad=0.5", facecolor="#1E293B", edgecolor="#38BDF8", linewidth=1.5)
    ax.text(6, 5.9, "LEAKAGE-FREE DATA PREPROCESSING PIPELINE (NSL-KDD)",
            ha="center", va="center", color="#F8FAFC", fontsize=14, fontweight="bold", bbox=title_box)

    # Boxes
    boxes = [
        {"x": 0.5, "y": 3.6, "w": 2.2, "h": 1.4, "title": "Raw NSL-KDD Data\n(43 Columns)", "sub": "41 Traffic Features\n+ Label & Difficulty", "color": "#1E293B", "edge": "#94A3B8"},
        {"x": 3.4, "y": 4.1, "w": 2.4, "h": 1.2, "title": "Categorical Features (3)\nprotocol, service, flag", "sub": "OneHotEncoder(handle_unknown='ignore')", "color": "#1E293B", "edge": "#38BDF8"},
        {"x": 3.4, "y": 2.3, "w": 2.4, "h": 1.2, "title": "Numerical Features (38)\nDurations, byte counts, rates", "sub": "StandardScaler()", "color": "#1E293B", "edge": "#38BDF8"},
        {"x": 6.5, "y": 3.2, "w": 2.2, "h": 1.4, "title": "ColumnTransformer\nPipeline", "sub": "Fitted EXCLUSIVELY on\nNormal Training Split", "color": "#064E3B", "edge": "#10B981"},
        {"x": 9.3, "y": 3.2, "w": 2.2, "h": 1.4, "title": "Processed Matrix\n(77 Features)", "sub": "float32 Normalized Vector\nZero Contamination", "color": "#1E293B", "edge": "#F59E0B"},
    ]

    for b in boxes:
        rect = patches.FancyBboxPatch((b["x"], b["y"]), b["w"], b["h"], boxstyle="round,pad=0.15",
                                      facecolor=b["color"], edgecolor=b["edge"], linewidth=2)
        ax.add_patch(rect)
        ax.text(b["x"] + b["w"]/2, b["y"] + b["h"]*0.65, b["title"], ha="center", va="center",
                color="#F8FAFC", fontsize=9.5, fontweight="bold")
        ax.text(b["x"] + b["w"]/2, b["y"] + b["h"]*0.25, b["sub"], ha="center", va="center",
                color="#94A3B8" if b["edge"] != "#10B981" else "#A7F3D0", fontsize=8)

    # Arrows
    arrow_props = dict(arrowstyle="->", color="#38BDF8", lw=2, mutation_scale=15)
    ax.annotate("", xy=(3.4, 4.7), xytext=(2.7, 4.5), arrowprops=arrow_props)
    ax.annotate("", xy=(3.4, 2.9), xytext=(2.7, 4.1), arrowprops=arrow_props)
    ax.annotate("", xy=(6.5, 4.0), xytext=(5.8, 4.5), arrowprops=arrow_props)
    ax.annotate("", xy=(6.5, 3.8), xytext=(5.8, 2.9), arrowprops=arrow_props)
    ax.annotate("", xy=(9.3, 3.9), xytext=(8.7, 3.9), arrowprops=arrow_props)

    # Bottom notes banner
    leakage_box = dict(boxstyle="round,pad=0.4", facecolor="#1E293B", edgecolor="#10B981", linewidth=1.5)
    ax.text(6, 0.9, "✓ Strict Zero-Leakage Guarantee: Preprocessor fitted ONLY on X_train_normal (53,874 records).\n   Validation & Test sets are transformed with preprocessor.transform() only. Labels are strictly excluded.",
            ha="center", va="center", color="#E2E8F0", fontsize=9, bbox=leakage_box)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "diagram_preprocessing_pipeline.png")
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved: {out_path}")


def create_system_workflow_diagram():
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=300)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")

    title_box = dict(boxstyle="round,pad=0.5", facecolor="#1E293B", edgecolor="#38BDF8", linewidth=1.5)
    ax.text(6.5, 5.9, "UNSUPERVISED INTRUSION DETECTION SYSTEM WORKFLOW",
            ha="center", va="center", color="#F8FAFC", fontsize=14, fontweight="bold", bbox=title_box)

    steps = [
        {"x": 0.4, "y": 2.8, "w": 1.8, "h": 1.6, "title": "Network Traffic\nTelemetry", "sub": "41 raw features\nper connection", "color": "#1E293B", "edge": "#94A3B8"},
        {"x": 2.7, "y": 2.8, "w": 2.0, "h": 1.6, "title": "Fitted Pipeline\nTransform", "sub": "StandardScaler +\nOneHotEncoder", "color": "#1E293B", "edge": "#38BDF8"},
        {"x": 5.2, "y": 2.8, "w": 2.2, "h": 1.6, "title": "Dense Autoencoder\n(77 -> 16 -> 77)", "sub": "Normal manifold\nreconstruction", "color": "#1E293B", "edge": "#818CF8"},
        {"x": 7.9, "y": 2.8, "w": 2.0, "h": 1.6, "title": "Reconstruction\nMSE Error", "sub": "MSE = mean((x - x_hat)^2)\nover 77 features", "color": "#1E293B", "edge": "#F59E0B"},
    ]

    for s in steps:
        rect = patches.FancyBboxPatch((s["x"], s["y"]), s["w"], s["h"], boxstyle="round,pad=0.12",
                                      facecolor=s["color"], edgecolor=s["edge"], linewidth=2)
        ax.add_patch(rect)
        ax.text(s["x"] + s["w"]/2, s["y"] + s["h"]*0.65, s["title"], ha="center", va="center",
                color="#F8FAFC", fontsize=9.5, fontweight="bold")
        ax.text(s["x"] + s["w"]/2, s["y"] + s["h"]*0.25, s["sub"], ha="center", va="center",
                color="#94A3B8", fontsize=8)

    arrow_props = dict(arrowstyle="->", color="#38BDF8", lw=2, mutation_scale=15)
    ax.annotate("", xy=(2.7, 3.6), xytext=(2.2, 3.6), arrowprops=arrow_props)
    ax.annotate("", xy=(5.2, 3.6), xytext=(4.7, 3.6), arrowprops=arrow_props)
    ax.annotate("", xy=(7.9, 3.6), xytext=(7.4, 3.6), arrowprops=arrow_props)

    # Decision Diamond / Threshold comparator
    comp_box = patches.FancyBboxPatch((10.3, 2.9), 2.3, 1.4, boxstyle="round,pad=0.15",
                                      facecolor="#312E81", edgecolor="#A5B4FC", linewidth=2)
    ax.add_patch(comp_box)
    ax.text(11.45, 3.75, "Decision Boundary", ha="center", va="center", color="#F8FAFC", fontsize=9.5, fontweight="bold")
    ax.text(11.45, 3.3, "Threshold: 0.129075\n(P95 Normal Val MSE)", ha="center", va="center", color="#C7D2FE", fontsize=8)

    ax.annotate("", xy=(10.3, 3.6), xytext=(9.9, 3.6), arrowprops=arrow_props)

    # Output Branches
    norm_box = patches.FancyBboxPatch((10.3, 4.7), 2.3, 0.9, boxstyle="round,pad=0.12",
                                      facecolor="#064E3B", edgecolor="#10B981", linewidth=2)
    ax.add_patch(norm_box)
    ax.text(11.45, 5.15, "NORMAL (<= 0.1291)\n95.99% Specificity", ha="center", va="center", color="#A7F3D0", fontsize=9, fontweight="bold")

    att_box = patches.FancyBboxPatch((10.3, 1.6), 2.3, 0.9, boxstyle="round,pad=0.12",
                                     facecolor="#7F1D1D", edgecolor="#EF4444", linewidth=2)
    ax.add_patch(att_box)
    ax.text(11.45, 2.05, "INTRUSION (> 0.1291)\n96.05% Precision", ha="center", va="center", color="#FECACA", fontsize=9, fontweight="bold")

    branch_norm = dict(arrowstyle="->", color="#10B981", lw=2, mutation_scale=15)
    branch_att = dict(arrowstyle="->", color="#EF4444", lw=2, mutation_scale=15)
    ax.annotate("", xy=(11.45, 4.7), xytext=(11.45, 4.3), arrowprops=branch_norm)
    ax.annotate("", xy=(11.45, 2.5), xytext=(11.45, 2.9), arrowprops=branch_att)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "diagram_system_workflow.png")
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved: {out_path}")


def create_autoencoder_architecture_diagram():
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=300)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")

    title_box = dict(boxstyle="round,pad=0.5", facecolor="#1E293B", edgecolor="#38BDF8", linewidth=1.5)
    ax.text(6.5, 5.9, "SYMMETRIC DENSE AUTOENCODER ARCHITECTURE (15,261 PARAMETERS)",
            ha="center", va="center", color="#F8FAFC", fontsize=14, fontweight="bold", bbox=title_box)

    layers = [
        {"name": "Input Layer\n(77 Features)", "units": 77, "h": 4.2, "x": 0.8, "color": "#1E293B", "edge": "#38BDF8", "act": "Input", "params": "0"},
        {"name": "Encoder 1\nDense(64)", "units": 64, "h": 3.6, "x": 2.7, "color": "#1E293B", "edge": "#60A5FA", "act": "ReLU", "params": "4,992"},
        {"name": "Encoder 2\nDense(32)", "units": 32, "h": 2.8, "x": 4.6, "color": "#1E293B", "edge": "#818CF8", "act": "ReLU", "params": "2,080"},
        {"name": "Bottleneck\nDense(16)", "units": 16, "h": 1.8, "x": 6.5, "color": "#4338CA", "edge": "#A855F7", "act": "Latent Space\n79.2% Compress", "params": "528"},
        {"name": "Decoder 1\nDense(32)", "units": 32, "h": 2.8, "x": 8.4, "color": "#1E293B", "edge": "#818CF8", "act": "ReLU", "params": "544"},
        {"name": "Decoder 2\nDense(64)", "units": 64, "h": 3.6, "x": 10.3, "color": "#1E293B", "edge": "#60A5FA", "act": "ReLU", "params": "2,112"},
        {"name": "Reconstruction\nDense(77)", "units": 77, "h": 4.2, "x": 12.0, "color": "#1E293B", "edge": "#38BDF8", "act": "Linear Output", "params": "5,005"},
    ]

    for l in layers:
        y_pos = 3.2 - l["h"]/2
        rect = patches.FancyBboxPatch((l["x"] - 0.7, y_pos), 1.4, l["h"], boxstyle="round,pad=0.1",
                                      facecolor=l["color"], edgecolor=l["edge"], linewidth=2)
        ax.add_patch(rect)
        ax.text(l["x"], y_pos + l["h"]/2 + 0.2, l["name"], ha="center", va="center", color="#F8FAFC", fontsize=8.5, fontweight="bold")
        ax.text(l["x"], y_pos + l["h"]/2 - 0.25, f"[{l['act']}]", ha="center", va="center", color="#94A3B8" if l["x"] != 6.5 else "#F0ABFC", fontsize=7.5)
        ax.text(l["x"], 0.7, f"Params:\n{l['params']}", ha="center", va="center", color="#CBD5E1", fontsize=8, fontweight="bold")

    # Flow arrows
    arrow_props = dict(arrowstyle="->", color="#38BDF8", lw=1.8, mutation_scale=12)
    for i in range(len(layers) - 1):
        ax.annotate("", xy=(layers[i+1]["x"] - 0.75, 3.2), xytext=(layers[i]["x"] + 0.75, 3.2), arrowprops=arrow_props)

    # Brackets for Encoder and Decoder
    ax.text(2.7, 5.3, "ENCODER COMPRESSION (77 -> 16)", ha="center", va="center", color="#93C5FD", fontsize=9, fontweight="bold")
    ax.text(10.2, 5.3, "DECODER RECONSTRUCTION (16 -> 77)", ha="center", va="center", color="#93C5FD", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "diagram_autoencoder_architecture.png")
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved: {out_path}")


def create_streamlit_workflow_diagram():
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")

    title_box = dict(boxstyle="round,pad=0.5", facecolor="#1E293B", edgecolor="#38BDF8", linewidth=1.5)
    ax.text(6, 5.9, "STREAMLIT INTERACTIVE APPLICATION ARCHITECTURE",
            ha="center", va="center", color="#F8FAFC", fontsize=14, fontweight="bold", bbox=title_box)

    sections = [
        {"x": 0.5, "y": 2.2, "w": 3.0, "h": 2.8, "title": "User Interface Layer\n(app/streamlit_app.py)", "items": [
            "• Single Record Analysis Form",
            "• 5 Quick Attack/Normal Presets",
            "• Batch CSV Upload Engine",
            "• KDDTest+ Benchmark Demo",
            "• Validated Metrics Dashboard",
            "• Architecture & Theory Guide"
        ], "color": "#1E293B", "edge": "#38BDF8"},
        {"x": 4.5, "y": 2.2, "w": 3.2, "h": 2.8, "title": "Cached Inference Engine\n(@st.cache_resource)", "items": [
            "• models/autoencoder.keras",
            "  (Dense Autoencoder, 15,261 params)",
            "• models/preprocessor.joblib",
            "  (ColumnTransformer, 77 features)",
            "• results/anomaly_threshold.json",
            "  (Official Threshold: 0.129075)"
        ], "color": "#1E293B", "edge": "#818CF8"},
        {"x": 8.5, "y": 2.2, "w": 3.0, "h": 2.8, "title": "Real-Time Outputs\n& Explanations", "items": [
            "• NORMAL vs INTRUSION Flag",
            "• Continuous Reconstruction MSE",
            "• Distance from Threshold",
            "• Top-5 Feature Deviations",
            "• Bulk CSV Results Export",
            "• Performance Plot Views"
        ], "color": "#1E293B", "edge": "#10B981"},
    ]

    for s in sections:
        rect = patches.FancyBboxPatch((s["x"], s["y"]), s["w"], s["h"], boxstyle="round,pad=0.15",
                                      facecolor=s["color"], edgecolor=s["edge"], linewidth=2)
        ax.add_patch(rect)
        ax.text(s["x"] + s["w"]/2, s["y"] + s["h"] - 0.35, s["title"], ha="center", va="center",
                color="#F8FAFC", fontsize=10, fontweight="bold")
        y_text = s["y"] + s["h"] - 0.75
        for item in s["items"]:
            ax.text(s["x"] + 0.2, y_text, item, ha="left", va="center", color="#CBD5E1", fontsize=8)
            y_text -= 0.32

    arrow_props = dict(arrowstyle="->", color="#38BDF8", lw=2, mutation_scale=15)
    ax.annotate("", xy=(4.5, 3.6), xytext=(3.5, 3.6), arrowprops=arrow_props)
    ax.annotate("", xy=(8.5, 3.6), xytext=(7.7, 3.6), arrowprops=arrow_props)

    launch_box = dict(boxstyle="round,pad=0.35", facecolor="#1E293B", edgecolor="#F59E0B", linewidth=1.5)
    ax.text(6, 0.9, "Application Command: streamlit run app/streamlit_app.py   (Zero Retraining, Read-Only Cached Inference)",
            ha="center", va="center", color="#FDE68A", fontsize=9.5, fontweight="bold", bbox=launch_box)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "diagram_streamlit_workflow.png")
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    create_preprocessing_diagram()
    create_system_workflow_diagram()
    create_autoencoder_architecture_diagram()
    create_streamlit_workflow_diagram()
