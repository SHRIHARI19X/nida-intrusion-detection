"""
create_presentation.py
----------------------
Builds the final academic PowerPoint presentation (15 slides) and text documentation
for Phase 14: Network Intrusion Detection Using Dense Autoencoder.

Generates:
  1. presentation/Network_Intrusion_Detection_Autoencoder_Presentation.pptx
  2. presentation/speaker_notes.txt
  3. presentation/presentation_content.txt
"""

import os
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PRESENTATION_DIR = os.path.join(BASE_DIR, "presentation")
os.makedirs(PRESENTATION_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Color Palette Constants (Cybersecurity Dark Academic Theme)
# ---------------------------------------------------------------------------
BG_DARK      = RGBColor(15, 23, 42)      # #0F172A (Deep Slate / Navy)
CARD_BG      = RGBColor(30, 41, 59)      # #1E293B (Card Slate)
ACCENT_BLUE  = RGBColor(56, 189, 248)    # #38BDF8 (Sky Blue)
ACCENT_GREEN = RGBColor(16, 185, 129)    # #10B981 (Emerald)
ACCENT_RED   = RGBColor(239, 68, 68)     # #EF4444 (Crimson)
ACCENT_PURP  = RGBColor(168, 85, 247)    # #A855F7 (Purple)
TEXT_WHITE   = RGBColor(248, 250, 252)   # #F8FAFC (White/Off-white)
TEXT_MUTED   = RGBColor(148, 163, 184)   # #94A3B8 (Muted Gray)
BORDER_COLOR = RGBColor(51, 65, 85)      # #334155 (Subtle border)


def set_slide_background(slide, prs):
    """Set deep slate background color for 16:9 slide."""
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG_DARK
    bg.line.fill.background()
    return bg


def add_header(slide, title_text, category_text="MSC AI & ML DEEP LEARNING PROJECT"):
    """Add consistent academic header to slide."""
    # Category tag
    tag_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.5), Inches(0.35))
    tf_tag = tag_box.text_frame
    tf_tag.word_wrap = True
    p_tag = tf_tag.paragraphs[0]
    p_tag.text = category_text.upper()
    p_tag.font.size = Pt(10)
    p_tag.font.bold = True
    p_tag.font.color.rgb = ACCENT_BLUE

    # Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.5), Inches(0.8))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    p_title = tf_title.paragraphs[0]
    p_title.text = title_text
    p_title.font.size = Pt(24)
    p_title.font.bold = True
    p_title.font.color.rgb = TEXT_WHITE


def add_card(slide, left, top, width, height, title="", border_color=BORDER_COLOR):
    """Add a card container for clean content organization."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = CARD_BG
    card.line.color.rgb = border_color
    card.line.width = Pt(1.2)

    if title:
        tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.15), width - Inches(0.4), Inches(0.4))
        p = tb.text_frame.paragraphs[0]
        p.text = title
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = ACCENT_BLUE

    return card


def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    speaker_notes_dict = {}
    slide_content_dict = {}

    # =======================================================================
    # SLIDE 1: TITLE
    # =======================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1, prs)

    # Accent decorative bar
    bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.8), Inches(0.15), Inches(3.2))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT_BLUE
    bar.line.fill.background()

    # Title Box
    t_box = s1.shapes.add_textbox(Inches(1.2), Inches(1.7), Inches(11.2), Inches(2.2))
    tf1 = t_box.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "Network Intrusion Detection Using Autoencoder"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_WHITE

    p2 = tf1.add_paragraph()
    p2.text = "An Unsupervised Deep Learning Approach Using the NSL-KDD Dataset"
    p2.font.size = Pt(18)
    p2.font.color.rgb = ACCENT_BLUE
    p2.space_before = Pt(10)

    # Subtitle Details Card
    card_meta = add_card(s1, Inches(1.2), Inches(4.3), Inches(10.8), Inches(2.3), border_color=ACCENT_BLUE)
    meta_box = s1.shapes.add_textbox(Inches(1.5), Inches(4.5), Inches(10.2), Inches(2.0))
    tf_meta = meta_box.text_frame

    m_lines = [
        ("Candidate Name", "[Student Name]"),
        ("Roll Number", "[Roll Number]"),
        ("Program / Specialization", "MSc Computer Science (AI & ML)"),
        ("Institution", "[College / University Name]"),
        ("Project Supervisor", "[Guide / Supervisor Name]"),
        ("Academic Year", "2025 – 2026"),
    ]
    for lbl, val in m_lines:
        p = tf_meta.add_paragraph()
        run1 = p.add_run()
        run1.text = f"{lbl:<26}: "
        run1.font.bold = True
        run1.font.color.rgb = ACCENT_BLUE
        run1.font.size = Pt(12)
        run2 = p.add_run()
        run2.text = val
        run2.font.color.rgb = TEXT_WHITE
        run2.font.size = Pt(12)

    speaker_notes_dict[1] = (
        "Good morning respected evaluators and faculty. Today I present my deep learning project: "
        "'Network Intrusion Detection Using Autoencoder'. In this work, we address cyber threat detection "
        "as an unsupervised anomaly detection task on the NSL-KDD benchmark, training exclusively on normal traffic "
        "to detect intrusions based on reconstruction error.\n"
        "Key Takeaway: A leakage-free, unsupervised deep learning framework for network intrusion detection."
    )
    slide_content_dict[1] = "Slide 1: Title & Academic Metadata (Candidate, Guide, University, Program)"

    # =======================================================================
    # SLIDE 2: PROBLEM STATEMENT
    # =======================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2, prs)
    add_header(s2, "Problem Statement: The Challenge of Network Intrusion Detection")

    add_card(s2, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.1), title="Traditional NIDS Limitations")
    tb2_left = s2.shapes.add_textbox(Inches(1.0), Inches(2.3), Inches(5.2), Inches(4.3))
    tf2_l = tb2_left.text_frame
    tf2_l.word_wrap = True
    l_points = [
        ("Signature-Based Vulnerability:", "Conventional systems rely on pre-compiled databases of known attack signatures (e.g., Snort rules)."),
        ("Zero-Day Blindspot:", "Novel attack variants and modified payloads bypass static pattern-matching rules effortlessly."),
        ("Supervised ML Bottleneck:", "Supervised models require vast, balanced, perfectly labeled datasets containing all conceivable attack types."),
        ("Labeling Cost & Delay:", "Accurate cyberattack labeling in live enterprise networks is expensive, labor-intensive, and fundamentally reactive."),
    ]
    for h, b in l_points:
        p = tf2_l.add_paragraph()
        r1 = p.add_run()
        r1.text = f"• {h} "
        r1.font.bold = True
        r1.font.color.rgb = ACCENT_RED
        r1.font.size = Pt(12)
        r2 = p.add_run()
        r2.text = b
        r2.font.color.rgb = TEXT_WHITE
        r2.font.size = Pt(11.5)
        p.space_after = Pt(10)

    add_card(s2, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.1), title="The Anomaly Detection Alternative")
    tb2_right = s2.shapes.add_textbox(Inches(7.0), Inches(2.3), Inches(5.3), Inches(4.3))
    tf2_r = tb2_right.text_frame
    tf2_r.word_wrap = True
    r_points = [
        ("The Unsupervised Paradigm:", "Rather than modeling every attack, learn what legitimate, normal traffic looks like."),
        ("Deep Autoencoder Representation:", "Compress normal telemetry into a low-dimensional manifold and learn to reconstruct it."),
        ("Reconstruction Discrepancy:", "Anomalous traffic that deviates from normal statistical patterns cannot be reconstructed faithfully."),
        ("Core Research Question:", "Can an unsupervised Dense Autoencoder achieve high detection precision on unseen attacks while keeping false alarms strictly bounded?"),
    ]
    for h, b in r_points:
        p = tf2_r.add_paragraph()
        r1 = p.add_run()
        r1.text = f"• {h} "
        r1.font.bold = True
        r1.font.color.rgb = ACCENT_GREEN
        r1.font.size = Pt(12)
        r2 = p.add_run()
        r2.text = b
        r2.font.color.rgb = TEXT_WHITE
        r2.font.size = Pt(11.5)
        p.space_after = Pt(10)

    speaker_notes_dict[2] = (
        "Here we contrast traditional signature-based NIDS with anomaly detection. "
        "Signature-based tools and supervised classifiers struggle when facing novel attacks or unlabeled data. "
        "Our project asks: can an Autoencoder learn only normal traffic and flag anomalies via reconstruction failure?\n"
        "Key Takeaway: Anomaly detection shifts the paradigm from chasing attack signatures to establishing a normal baseline."
    )
    slide_content_dict[2] = "Slide 2: Problem Statement (Signature vs Anomaly Detection, Supervised vs Unsupervised)"

    # =======================================================================
    # SLIDE 3: MOTIVATION
    # =======================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3, prs)
    add_header(s3, "Project Motivation: Why Reconstruction-Based Anomaly Detection?")

    cards3 = [
        ("1. Dynamic Threat Landscape", "Cyberattacks constantly evolve through obfuscation, polymorphic payloads, and novel exploitation vectors, rendering static rule engines obsolete.", ACCENT_BLUE),
        ("2. Natural Abundance of Normal Telemetry", "In production networks, over 99% of raw network packets represent benign user traffic. Normal traffic provides an abundant, reliable baseline.", ACCENT_GREEN),
        ("3. Zero-Contamination One-Class Learning", "Training strictly on normal traffic eliminates the risk of model bias toward specific attack families represented in training sets.", ACCENT_PURP),
        ("4. Objective Metric for Alerting", "Reconstruction Mean Squared Error (MSE) provides a continuous, interpretable score measuring statistical deviation from normal behavior.", ACCENT_BLUE),
    ]

    for i, (title, desc, col) in enumerate(cards3):
        row = i // 2
        col_idx = i % 2
        left = Inches(0.8 + col_idx * 5.9)
        top = Inches(1.7 + row * 2.6)
        add_card(s3, left, top, Inches(5.6), Inches(2.3), title=title, border_color=col)
        tb = s3.shapes.add_textbox(left + Inches(0.25), top + Inches(0.7), Inches(5.1), Inches(1.4))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = desc
        p.font.size = Pt(12.5)
        p.font.color.rgb = TEXT_WHITE

    speaker_notes_dict[3] = (
        "Our motivation rests on four pillars: attacks evolve, normal data is abundant, one-class training avoids attack bias, "
        "and reconstruction error provides an objective anomaly signal. We emphasize that this approach does not guarantee "
        "detection of every stealthy intrusion, but reliably flags traffic that deviates statistically from normal baselines.\n"
        "Key Takeaway: Normal network traffic offers a rich, self-supervised learning signal for anomaly detection."
    )
    slide_content_dict[3] = "Slide 3: Project Motivation (Threat Evolution, Abundant Normal Data, One-Class Training, Objective Metric)"

    # =======================================================================
    # SLIDE 4: OBJECTIVES
    # =======================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4, prs)
    add_header(s4, "Project Objectives: Research & Engineering Goals")

    objs = [
        ("1. Design Leakage-Free Preprocessing Pipeline", "Construct a ColumnTransformer pipeline that standardizes 38 numerical features and one-hot encodes 3 categorical features, fitted strictly on normal training data without test contamination."),
        ("2. Develop Symmetric Dense Autoencoder Architecture", "Design a multi-stage fully connected autoencoder (77 -> 64 -> 32 -> 16 -> 32 -> 64 -> 77) with a 16-dimensional bottleneck to force optimal feature compression."),
        ("3. Implement Unsupervised Normal-Only Training", "Train the network exclusively on benign records from KDDTrain+ using Adam and MSE loss, completely isolating attack records during model parameter fitting."),
        ("4. Derive Principled Anomaly Decision Threshold", "Empirically select the decision boundary as the 95th percentile of normal validation reconstruction errors, guaranteeing a reference false-positive rate of ~5%."),
        ("5. Rigorous Evaluation on Unseen KDDTest+ Benchmark", "Quantify real-world generalization across 22,544 untouched test records, evaluating Accuracy, Precision, Recall, F1, ROC-AUC, and attack-family sensitivity."),
        ("6. Deploy Operational Streamlit Application", "Deliver an interactive cybersecurity dashboard supporting single-connection diagnostic inspection, batch CSV inference, and real-time feature deviation explanations."),
    ]

    for i, (title, body) in enumerate(objs):
        top_pos = Inches(1.6 + i * 0.9)
        card = add_card(s4, Inches(0.8), top_pos, Inches(11.7), Inches(0.8), border_color=ACCENT_BLUE)
        tb = s4.shapes.add_textbox(Inches(1.0), top_pos + Inches(0.08), Inches(11.3), Inches(0.65))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        r1 = p.add_run()
        r1.text = f"{title}: "
        r1.font.bold = True
        r1.font.size = Pt(11.5)
        r1.font.color.rgb = ACCENT_BLUE
        r2 = p.add_run()
        r2.text = body
        r2.font.size = Pt(10.5)
        r2.font.color.rgb = TEXT_WHITE

    speaker_notes_dict[4] = (
        "Here are our six core objectives spanning the entire machine learning lifecycle: "
        "leakage-free preprocessing, deep autoencoder modeling, normal-only training, validation-calibrated thresholding, "
        "comprehensive benchmark testing, and an operational web demonstration.\n"
        "Key Takeaway: Clear, verifiable objectives bridging theoretical modeling and practical cybersecurity deployment."
    )
    slide_content_dict[4] = "Slide 4: Project Objectives (Preprocessing, Architecture, Training, Threshold, Evaluation, Deployment)"

    # =======================================================================
    # SLIDE 5: DATASET: NSL-KDD
    # =======================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5, prs)
    add_header(s5, "Dataset: NSL-KDD Benchmark Characteristics")

    add_card(s5, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.2), title="Dataset Structure & Splits")
    tb5 = s5.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.2), Inches(4.5))
    tf5 = tb5.text_frame
    tf5.word_wrap = True

    d_info = [
        ("Benchmark Name", "NSL-KDD (Refined KDD Cup 99 without duplicates)"),
        ("Training Set (KDDTrain+)", "125,973 records (67,343 Normal, 58,630 Attacks)"),
        ("Unseen Test Set (KDDTest+)", "22,544 records (9,711 Normal, 12,833 Attacks)"),
        ("Original Traffic Features", "41 Features (3 Categorical, 38 Numerical)"),
        ("Categorical Fields", "protocol_type (3), service (70), flag (11)"),
        ("Numerical Fields", "Duration, byte counts, error rates, host statistics"),
        ("Metadata Columns (Isolated)", "label (specific attack), difficulty_level (score)"),
        ("Attack Taxonomy", "DoS, Probe, R2L, U2R (38 distinct attack types)"),
    ]
    for lbl, val in d_info:
        p = tf5.add_paragraph()
        r1 = p.add_run()
        r1.text = f"• {lbl}: "
        r1.font.bold = True
        r1.font.color.rgb = ACCENT_BLUE
        r1.font.size = Pt(11)
        r2 = p.add_run()
        r2.text = val
        r2.font.color.rgb = TEXT_WHITE
        r2.font.size = Pt(10.5)
        p.space_after = Pt(4)

    # Embed Actual Label Distribution Plot
    img_dist = os.path.join(RESULTS_DIR, "label_distribution.png")
    if os.path.exists(img_dist):
        add_card(s5, Inches(6.7), Inches(1.7), Inches(5.8), Inches(5.2), title="Traffic Label Distribution")
        s5.shapes.add_picture(img_dist, Inches(6.9), Inches(2.3), width=Inches(5.4))

    speaker_notes_dict[5] = (
        "Slide 5 presents the NSL-KDD dataset. KDDTrain+ contains 125,973 records, while KDDTest+ contains 22,544 records. "
        "The chart on the right shows the verified distribution between normal traffic and attack categories. "
        "Crucially, the 41 network traffic features were separated from labels and difficulty scores before any modeling.\n"
        "Key Takeaway: NSL-KDD provides an established benchmark with realistic feature diversity and novel test attacks."
    )
    slide_content_dict[5] = "Slide 5: Dataset NSL-KDD (KDDTrain+, KDDTest+, 41 Features, Label Distribution Plot)"

    # =======================================================================
    # SLIDE 6: DATA PREPROCESSING
    # =======================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6, prs)
    add_header(s6, "Data Preprocessing: Strict Leakage-Free Pipeline")

    img_pre = os.path.join(RESULTS_DIR, "diagram_preprocessing_pipeline.png")
    if os.path.exists(img_pre):
        s6.shapes.add_picture(img_pre, Inches(0.8), Inches(1.7), width=Inches(11.7))

    speaker_notes_dict[6] = (
        "Slide 6 illustrates our leakage-free preprocessing pipeline. "
        "OneHotEncoder processes the 3 categorical attributes, while StandardScaler scales the 38 numerical features. "
        "The ColumnTransformer was fitted strictly on normal training data (X_train_normal). "
        "Validation and test sets were transformed only, ensuring zero future information leaked into the model.\n"
        "Key Takeaway: Methodological rigor prevents optimistic bias by fitting preprocessors exclusively on training data."
    )
    slide_content_dict[6] = "Slide 6: Data Preprocessing Pipeline (One-Hot, Scaler, ColumnTransformer, Zero-Leakage)"

    # =======================================================================
    # SLIDE 7: PROPOSED METHODOLOGY
    # =======================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7, prs)
    add_header(s7, "Proposed Methodology: Unsupervised Anomaly Workflow")

    img_wf = os.path.join(RESULTS_DIR, "diagram_system_workflow.png")
    if os.path.exists(img_wf):
        s7.shapes.add_picture(img_wf, Inches(0.8), Inches(1.7), width=Inches(11.7))

    speaker_notes_dict[7] = (
        "Slide 7 walks through the end-to-end detection workflow. "
        "Raw telemetry is preprocessed into 77 dimensions, passed to the Autoencoder, and reconstructed. "
        "The per-sample Mean Squared Error measures reconstruction failure. If error exceeds the threshold of 0.1291, "
        "an intrusion alert is triggered; otherwise, it is classified as normal.\n"
        "Key Takeaway: An elegant, fully automated pipeline from raw network packet telemetry to binary alert decision."
    )
    slide_content_dict[7] = "Slide 7: Proposed Methodology (Telemetry -> Preprocessing -> Autoencoder -> MSE -> Threshold)"

    # =======================================================================
    # SLIDE 8: AUTOENCODER ARCHITECTURE
    # =======================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8, prs)
    add_header(s8, "Autoencoder Architecture: Deep Neural Network Design")

    img_arch = os.path.join(RESULTS_DIR, "diagram_autoencoder_architecture.png")
    if os.path.exists(img_arch):
        s8.shapes.add_picture(img_arch, Inches(0.8), Inches(1.7), width=Inches(11.7))

    speaker_notes_dict[8] = (
        "Slide 8 displays our confirmed Dense Autoencoder architecture. "
        "Starting with 77 inputs, the encoder compresses through 64 and 32 neurons down to a 16-dimensional bottleneck—a 79.2% compression ratio. "
        "The decoder symmetrically expands back to 77 linear output neurons. "
        "Total trainable parameters are exactly 15,261, optimized using Adam and MSE loss.\n"
        "Key Takeaway: The 16-neuron latent bottleneck acts as a compression barrier, preventing anomalies from reconstructing."
    )
    slide_content_dict[8] = "Slide 8: Autoencoder Architecture (77 -> 64 -> 32 -> 16 -> 32 -> 64 -> 77, 15,261 Params)"

    # =======================================================================
    # SLIDE 9: TRAINING & THRESHOLD SELECTION
    # =======================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9, prs)
    add_header(s9, "Model Training & Principled Threshold Calibration")

    # Embed Loss Curve & Anomaly Threshold plots side-by-side
    img_loss = os.path.join(RESULTS_DIR, "loss_curve.png")
    img_thresh = os.path.join(RESULTS_DIR, "anomaly_threshold.png")

    if os.path.exists(img_loss):
        add_card(s9, Inches(0.8), Inches(1.7), Inches(5.7), Inches(5.2), title="Training & Validation Loss Convergence")
        s9.shapes.add_picture(img_loss, Inches(1.0), Inches(2.3), width=Inches(5.3))

    if os.path.exists(img_thresh):
        add_card(s9, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.2), title="Threshold Calibration (Validation P95)")
        s9.shapes.add_picture(img_thresh, Inches(7.0), Inches(2.3), width=Inches(5.3))

    speaker_notes_dict[9] = (
        "Slide 9 demonstrates model training and threshold calibration. "
        "On the left, the loss curve shows smooth convergence down to a validation loss of 0.3138 at epoch 8 before early stopping. "
        "On the right, we plot the reconstruction error histogram on normal validation records. "
        "The threshold of 0.129075 was calibrated as the 95th percentile, mathematically fixing the expected false positive rate to ~5%.\n"
        "Key Takeaway: The threshold is grounded in validation error percentiles rather than an arbitrary heuristic cutoff."
    )
    slide_content_dict[9] = "Slide 9: Training Loss Curve & Normal Validation Threshold Calibration (P95 = 0.129075)"

    # =======================================================================
    # SLIDE 10: RECONSTRUCTION ERROR ANALYSIS
    # =======================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10, prs)
    add_header(s10, "Reconstruction Error Analysis: Normal vs Attack Separation")

    add_card(s10, Inches(0.8), Inches(1.7), Inches(5.4), Inches(5.2), title="Reconstruction Error Statistics")
    tb10 = s10.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.0), Inches(4.5))
    tf10 = tb10.text_frame
    tf10.word_wrap = True

    e_stats = [
        ("Normal Median MSE", "0.003158", TEXT_WHITE),
        ("Attack Median MSE", "0.457592", ACCENT_RED),
        ("Median Separation", "144.9x Higher for Attacks", ACCENT_GREEN),
        ("Normal Mean MSE", "0.038540", TEXT_WHITE),
        ("Attack Mean MSE", "0.963125", ACCENT_RED),
        ("Normal P95 MSE", "0.112638 (Below 0.1291 threshold)", TEXT_WHITE),
        ("Attack P95 MSE", "1.578233 (12.2x above threshold)", ACCENT_RED),
        ("Separation Principle", "Attacks violate normal feature correlations, yielding high reconstruction penalties.", ACCENT_BLUE),
    ]
    for lbl, val, col in e_stats:
        p = tf10.add_paragraph()
        r1 = p.add_run()
        r1.text = f"• {lbl}: "
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = ACCENT_BLUE
        r2 = p.add_run()
        r2.text = val
        r2.font.bold = (col != TEXT_WHITE)
        r2.font.color.rgb = col
        r2.font.size = Pt(11)
        p.space_after = Pt(4)

    img_err_dist = os.path.join(RESULTS_DIR, "reconstruction_error_normal_vs_attack.png")
    if os.path.exists(img_err_dist):
        add_card(s10, Inches(6.5), Inches(1.7), Inches(6.0), Inches(5.2), title="Empirical Error Distribution")
        s10.shapes.add_picture(img_err_dist, Inches(6.7), Inches(2.3), width=Inches(5.6))

    speaker_notes_dict[10] = (
        "Slide 10 highlights the core anomaly detection signal. "
        "The median reconstruction error for attack traffic (0.4576) is 144.9 times higher than normal traffic (0.0032). "
        "The log-scale distribution plot confirms clean separation: normal traffic concentrates heavily to the left of the green threshold line, "
        "while the attack distribution shifts decisively to the right.\n"
        "Key Takeaway: The autoencoder reliably discriminates anomalies due to massive reconstruction error disparities."
    )
    slide_content_dict[10] = "Slide 10: Reconstruction Error Analysis (144.9x Median Separation, Distribution Plot)"

    # =======================================================================
    # SLIDE 11: FINAL MODEL PERFORMANCE
    # =======================================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11, prs)
    add_header(s11, "Final Model Performance: Validated KDDTest+ Benchmark")

    # 4 Top KPI Cards
    kpis = [
        ("ACCURACY", "83.35%", "18,791 / 22,544 Correct", ACCENT_BLUE),
        ("PRECISION", "96.05%", "Low False Alarms (9,469 / 9,858)", ACCENT_GREEN),
        ("RECALL", "73.79%", "9,469 / 12,833 Attacks Flagged", ACCENT_PURP),
        ("F1 SCORE", "83.46%", "Harmonic Mean Balance", ACCENT_BLUE),
    ]
    for i, (lbl, val, sub, col) in enumerate(kpis):
        left = Inches(0.8 + i * 2.95)
        card = add_card(s11, left, Inches(1.7), Inches(2.8), Inches(1.8), border_color=col)
        tb = s11.shapes.add_textbox(left, Inches(1.85), Inches(2.8), Inches(1.5))
        tf = tb.text_frame
        tf.word_wrap = True
        p_val = tf.paragraphs[0]
        p_val.alignment = PP_ALIGN.CENTER
        p_val.text = val
        p_val.font.size = Pt(28)
        p_val.font.bold = True
        p_val.font.color.rgb = col

        p_lbl = tf.add_paragraph()
        p_lbl.alignment = PP_ALIGN.CENTER
        p_lbl.text = lbl
        p_lbl.font.size = Pt(11)
        p_lbl.font.bold = True
        p_lbl.font.color.rgb = TEXT_WHITE

        p_sub = tf.add_paragraph()
        p_sub.alignment = PP_ALIGN.CENTER
        p_sub.text = sub
        p_sub.font.size = Pt(9)
        p_sub.font.color.rgb = TEXT_MUTED

    # Bottom Detailed Metrics Table
    add_card(s11, Inches(0.8), Inches(3.8), Inches(11.7), Inches(3.1), title="Comprehensive Evaluation Metrics Summary")
    tb11_tbl = s11.shapes.add_textbox(Inches(1.1), Inches(4.3), Inches(11.1), Inches(2.4))
    tf11_t = tb11_tbl.text_frame
    tf11_t.word_wrap = True

    m_rows = [
        ("ROC-AUC Score", "95.70%", "Evaluated continuously across all operating thresholds (Phase 10 confirmed)."),
        ("False Positive Rate (FPR)", "4.01%", "389 / 9,711 benign test records flagged; closely tracks the 5% validation calibration target."),
        ("Specificity (TNR)", "95.99%", "9,322 / 9,711 normal connections correctly permitted without disruption."),
        ("Benchmark Integrity", "Zero Leakage", "Evaluated strictly on untouched KDDTest+ without retraining, tuning, or threshold alterations."),
    ]
    for m, v, c in m_rows:
        p = tf11_t.add_paragraph()
        r1 = p.add_run()
        r1.text = f"• {m:<28}: "
        r1.font.bold = True
        r1.font.color.rgb = ACCENT_BLUE
        r1.font.size = Pt(12)
        r2 = p.add_run()
        r2.text = f"{v:<10}  —  "
        r2.font.bold = True
        r2.font.color.rgb = ACCENT_GREEN if "%" in v or "Zero" in v else TEXT_WHITE
        r2.font.size = Pt(12)
        r3 = p.add_run()
        r3.text = c
        r3.font.color.rgb = TEXT_WHITE
        r3.font.size = Pt(11)
        p.space_after = Pt(4)

    speaker_notes_dict[11] = (
        "Slide 11 summarizes our official test metrics on KDDTest+. "
        "The model achieves 83.35% accuracy, an exceptional 96.05% precision, 73.79% recall, and 95.70% ROC-AUC. "
        "Crucially, the False Positive Rate is only 4.01%, validating that our 95th percentile validation threshold successfully "
        "constrained false alarms in unseen test traffic.\n"
        "Key Takeaway: High precision (96.05%) prevents alert fatigue while maintaining strong detection capability."
    )
    slide_content_dict[11] = "Slide 11: Final Model Performance (Accuracy 83.35%, Precision 96.05%, Recall 73.79%, ROC-AUC 95.70%)"

    # =======================================================================
    # SLIDE 12: CONFUSION MATRIX & ROC CURVE
    # =======================================================================
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12, prs)
    add_header(s12, "Confusion Matrix & Receiver Operating Characteristic")

    img_cm = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    img_roc = os.path.join(RESULTS_DIR, "roc_curve.png")

    if os.path.exists(img_cm):
        add_card(s12, Inches(0.8), Inches(1.7), Inches(5.7), Inches(5.2), title="Confusion Matrix (KDDTest+)")
        s12.shapes.add_picture(img_cm, Inches(1.0), Inches(2.3), width=Inches(5.3))

    if os.path.exists(img_roc):
        add_card(s12, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.2), title="ROC Curve (AUC = 95.70%)")
        s12.shapes.add_picture(img_roc, Inches(7.0), Inches(2.3), width=Inches(5.3))

    speaker_notes_dict[12] = (
        "Slide 12 presents the confusion matrix and ROC curve. "
        "On the left, out of 22,544 test connections, we correctly identify 9,322 True Negatives and 9,469 True Positives, "
        "with only 389 False Positives. "
        "On the right, the ROC curve shows an AUC of 95.70%, with the red dot marking our official operating point at FPR=4.01%, TPR=73.79%.\n"
        "Key Takeaway: Strong global discriminative power reflected by an ROC-AUC of 95.70%."
    )
    slide_content_dict[12] = "Slide 12: Confusion Matrix & ROC Curve (TN=9,322, FP=389, FN=3,364, TP=9,469, AUC=95.70%)"

    # =======================================================================
    # SLIDE 13: ATTACK-TYPE & ERROR ANALYSIS
    # =======================================================================
    s13 = prs.slides.add_slide(blank_layout)
    set_slide_background(s13, prs)
    add_header(s13, "Attack-Family Breakdown & Granular Error Diagnostics")

    add_card(s13, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.2), title="Detection Rates by Attack Family")
    tb13 = s13.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.2), Inches(4.5))
    tf13 = tb13.text_frame
    tf13.word_wrap = True

    att_summary = [
        ("DoS Attacks", "84.37% Recall (6,292 / 7,458)", "Volumetric floods (Neptune, Apache2) cause massive connection rate anomalies; mean MSE = 0.6511."),
        ("Probe Attacks", "81.54% Recall (1,974 / 2,421)", "Network scanning (Portsweep, Mscan) perturbs destination host distributions; mean MSE = 0.7465."),
        ("U2R Attacks", "65.67% Recall (44 / 67)", "Privilege escalation attempts incur severe structural reconstruction penalties; mean MSE = 30.2773."),
        ("R2L Attacks", "40.15% Recall (1,159 / 2,887)", "HARDEST FAMILY: Interactive password guessing (Guess_passwd) mimics valid user sessions at the header level; median MSE = 0.0435."),
        ("Error Distribution", "3,364 Total Missed (FN)", "R2L accounts for 51.4% of all missed attacks due to payload-agnostic tabular feature overlap."),
    ]
    for fam, rate, note in att_summary:
        p = tf13.add_paragraph()
        r1 = p.add_run()
        r1.text = f"• {fam}: "
        r1.font.bold = True
        r1.font.color.rgb = ACCENT_BLUE
        r1.font.size = Pt(11)
        r2 = p.add_run()
        r2.text = f"{rate}\n   "
        r2.font.bold = True
        r2.font.color.rgb = ACCENT_GREEN if "84" in rate or "81" in rate else (ACCENT_RED if "40" in rate else TEXT_WHITE)
        r2.font.size = Pt(10.5)
        r3 = p.add_run()
        r3.text = note
        r3.font.color.rgb = TEXT_MUTED
        r3.font.size = Pt(9.5)
        p.space_after = Pt(4)

    img_att_det = os.path.join(RESULTS_DIR, "attack_detection_rate_comparison.png")
    if os.path.exists(img_att_det):
        add_card(s13, Inches(6.7), Inches(1.7), Inches(5.8), Inches(5.2), title="Empirical Detection Rates vs Specificity")
        s13.shapes.add_picture(img_att_det, Inches(6.9), Inches(2.3), width=Inches(5.4))

    speaker_notes_dict[13] = (
        "Slide 13 breaks down performance across attack families. "
        "DoS and Probe are detected reliably at 84.4% and 81.5% because flooding and scanning disrupt host connection rates. "
        "In contrast, R2L is the hardest category at 40.15% detection, representing 51.4% of all missed attacks. "
        "This occurs because password guessing over valid sessions mimics legitimate telemetry in header metadata.\n"
        "Key Takeaway: Volumetric and scanning threats are detected easily, while content-based R2L intrusions reveal feature limitations."
    )
    slide_content_dict[13] = "Slide 13: Attack-Family Diagnostics (DoS 84.37%, Probe 81.54%, U2R 65.67%, R2L 40.15%)"

    # =======================================================================
    # SLIDE 14: STREAMLIT APPLICATION
    # =======================================================================
    s14 = prs.slides.add_slide(blank_layout)
    set_slide_background(s14, prs)
    add_header(s14, "Interactive Deployment: Streamlit NIDS Web Application")

    img_st_wf = os.path.join(RESULTS_DIR, "diagram_streamlit_workflow.png")
    if os.path.exists(img_st_wf):
        s14.shapes.add_picture(img_st_wf, Inches(0.8), Inches(1.7), width=Inches(11.7))

    speaker_notes_dict[14] = (
        "Slide 14 outlines our Streamlit web application. "
        "The system offers single-record analysis with 5 pre-loaded attack presets, batch CSV processing with live throughput, "
        "and real-time explainability highlighting the top 5 reconstruction deviations. "
        "Importantly, the UI clarifies that reconstruction error is an anomaly measure, not a calibrated attack probability.\n"
        "Key Takeaway: Operational prototype enabling interactive viva demonstration and batch telemetry processing."
    )
    slide_content_dict[14] = "Slide 14: Streamlit Application (Single Record, Batch CSV, Top-5 Deviations, Benchmark Demo)"

    # =======================================================================
    # SLIDE 15: CONCLUSION & FUTURE SCOPE
    # =======================================================================
    s15 = prs.slides.add_slide(blank_layout)
    set_slide_background(s15, prs)
    add_header(s15, "Conclusions & Future Research Directions")

    add_card(s15, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.2), title="Research Conclusions")
    tb15_l = s15.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.2), Inches(4.5))
    tf15_l = tb15_l.text_frame
    tf15_l.word_wrap = True
    concs = [
        ("Unsupervised Viability:", "Dense Autoencoders successfully establish a normal behavioral baseline using benign-only training data."),
        ("High Generalization Precision:", "Achieved 96.05% precision and 83.35% accuracy on 22,544 untouched test connections."),
        ("Principled Calibration:", "Deriving the decision threshold from the validation 95th percentile constrained test FPR to 4.01%."),
        ("Discriminative Power:", "Demonstrated 144.9x higher median reconstruction error for attacks compared to normal traffic."),
        ("Operational Deployment:", "Packaged as an interactive, explainable Streamlit application for cybersecurity workflows."),
    ]
    for h, b in concs:
        p = tf15_l.add_paragraph()
        r1 = p.add_run()
        r1.text = f"• {h} "
        r1.font.bold = True
        r1.font.color.rgb = ACCENT_GREEN
        r1.font.size = Pt(11.5)
        r2 = p.add_run()
        r2.text = b
        r2.font.color.rgb = TEXT_WHITE
        r2.font.size = Pt(11)
        p.space_after = Pt(6)

    add_card(s15, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.2), title="Future Scope & Improvements")
    tb15_r = s15.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.5))
    tf15_r = tb15_r.text_frame
    tf15_r.word_wrap = True
    futures = [
        ("Deep Packet Inspection (DPI) Hybridization:", "Combine tabular header autoencoders with byte-level payload NLP/CNN models to detect stealthy R2L intrusions."),
        ("Sequential Modeling (LSTM / Transformers):", "Capture multi-connection temporal sequences to detect low-and-slow probing campaigns across extended time windows."),
        ("Adaptive Service-Specific Thresholding:", "Replace single global threshold with service-conditioned cutoffs (e.g., separate thresholds for HTTP, DNS, SSH)."),
        ("Contemporary Dataset Validation:", "Evaluate generalization on modern benchmarks like CIC-IDS2017 and CSE-CIC-IDS2018."),
        ("Explainable AI (XAI) Attribution:", "Integrate SHAP / Integrated Gradients to trace feature contributions to reconstruction loss."),
    ]
    for h, b in futures:
        p = tf15_r.add_paragraph()
        r1 = p.add_run()
        r1.text = f"• {h} "
        r1.font.bold = True
        r1.font.color.rgb = ACCENT_BLUE
        r1.font.size = Pt(11.5)
        r2 = p.add_run()
        r2.text = b
        r2.font.color.rgb = TEXT_WHITE
        r2.font.size = Pt(11)
        p.space_after = Pt(6)

    speaker_notes_dict[15] = (
        "In conclusion, we have demonstrated that an unsupervised Dense Autoencoder can detect network intrusions "
        "with 83.35% accuracy, 96.05% precision, and an ROC-AUC of 95.70% on unseen benchmark traffic. "
        "For future work, incorporating payload inspection and sequential modeling will address R2L payload blindspots. "
        "Thank you, and I am now ready for questions.\n"
        "Key Takeaway: A successful, mathematically verified proof-of-concept for unsupervised deep learning intrusion detection."
    )
    slide_content_dict[15] = "Slide 15: Conclusions & Future Scope (Key Findings, DPI Integration, Sequential Modeling, Evaluation on Modern Benchmarks)"

    # Add speaker notes to slide objects
    for idx, slide in enumerate(prs.slides, start=1):
        if idx in speaker_notes_dict:
            notes_slide = slide.notes_slide
            tf_notes = notes_slide.notes_text_frame
            tf_notes.text = speaker_notes_dict[idx]

    # Save PPTX
    pptx_path = os.path.join(PRESENTATION_DIR, "Network_Intrusion_Detection_Autoencoder_Presentation.pptx")
    prs.save(pptx_path)
    print(f"Presentation saved successfully: {pptx_path}")

    # Save Speaker Notes Text File
    notes_txt_path = os.path.join(PRESENTATION_DIR, "speaker_notes.txt")
    with open(notes_txt_path, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("SPEAKER NOTES — NETWORK INTRUSION DETECTION USING AUTOENCODER\n")
        f.write("MSc Computer Science (AI & ML) Project Presentation & Viva Guide\n")
        f.write("================================================================================\n\n")
        for i in range(1, 16):
            f.write(f"--------------------------------------------------------------------------------\n")
            f.write(f"SLIDE {i}: {slide_content_dict[i]}\n")
            f.write(f"--------------------------------------------------------------------------------\n")
            f.write(speaker_notes_dict[i] + "\n\n")
    print(f"Speaker notes saved: {notes_txt_path}")

    # Save Presentation Content Outline Text File
    content_txt_path = os.path.join(PRESENTATION_DIR, "presentation_content.txt")
    with open(content_txt_path, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("PRESENTATION CONTENT OUTLINE — 15 SLIDES STRUCTURE\n")
        f.write("================================================================================\n\n")
        for i in range(1, 16):
            f.write(f"{i:2d}. {slide_content_dict[i]}\n")
            f.write(f"    Summary: {speaker_notes_dict[i].splitlines()[0]}\n\n")
    print(f"Presentation content outline saved: {content_txt_path}")

    return pptx_path, notes_txt_path, content_txt_path


if __name__ == "__main__":
    build_presentation()
