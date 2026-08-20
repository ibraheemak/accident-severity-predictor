"""Builds the final project-report .docx from the project's own documentation
(README.md, CLAUDE.md, docs/data_pipeline.md, docs/model_family_comparison.md,
CODE/experiments/full_stats19/README.md) and the original project_proposal.pdf.

Run from the repo root:
    python docs/report/build_final_report.py

Output: docs/report/Final_Report.docx
"""

import os

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

PRIMARY_BLUE = RGBColor(0x14, 0x54, 0x8C)
LIGHT_BLUE = RGBColor(0x2E, 0x74, 0xB5)
GRAY = RGBColor(0x60, 0x60, 0x60)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
HEADER_FILL = "14548C"

OUT_PATH = os.path.join(os.path.dirname(__file__), "Final_Report.docx")


# --------------------------------------------------------------------------
# Low-level helpers
# --------------------------------------------------------------------------

def shade_cell(cell, hex_color):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def add_page_break(doc):
    doc.add_page_break()


def add_heading(doc, text, level=1):
    h = doc.add_heading(level=level)
    run = h.add_run(text)
    run.font.color.rgb = PRIMARY_BLUE if level <= 2 else LIGHT_BLUE
    run.font.bold = True
    if level == 1:
        run.font.size = Pt(20)
        h.space_before = Pt(18)
    elif level == 2:
        run.font.size = Pt(15)
        h.space_before = Pt(12)
    else:
        run.font.size = Pt(12.5)
        h.space_before = Pt(8)
    return h


def add_para(doc, text="", italic=False, bold=False, size=11, color=None,
             align=None, space_after=8):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    if align is not None:
        p.alignment = align
    if text:
        run = p.add_run(text)
        run.italic = italic
        run.bold = bold
        run.font.size = Pt(size)
        if color is not None:
            run.font.color.rgb = color
    return p


def add_bullets(doc, items, size=11):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(4)
        if isinstance(item, tuple):
            lead, rest = item
            r1 = p.add_run(lead)
            r1.bold = True
            r1.font.size = Pt(size)
            r2 = p.add_run(rest)
            r2.font.size = Pt(size)
        else:
            r = p.add_run(item)
            r.font.size = Pt(size)


def add_table(doc, headers, rows, widths=None, header_size=10, body_size=10):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    hdr_cells = table.rows[0].cells
    for i, htext in enumerate(headers):
        hdr_cells[i].text = ""
        p = hdr_cells[i].paragraphs[0]
        run = p.add_run(htext)
        run.bold = True
        run.font.size = Pt(header_size)
        run.font.color.rgb = WHITE
        shade_cell(hdr_cells[i], HEADER_FILL)

    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            run = p.add_run(str(val))
            run.font.size = Pt(body_size)

    if widths:
        for row in table.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return table


def add_toc(doc):
    p = doc.add_paragraph()
    run = p.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = 'TOC \\o "1-3" \\h \\z \\u'
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    fld_text = OxmlElement("w:t")
    fld_text.text = "Right-click and choose “Update Field” to generate the table of contents."
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_sep)
    run._r.append(fld_text)
    run._r.append(fld_end)


def hr(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "8")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "2E74B5")
    pBdr.append(bottom)
    pPr.append(pBdr)


# --------------------------------------------------------------------------
# Document assembly
# --------------------------------------------------------------------------

def build():
    doc = Document()

    # Base style
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)

    section = doc.sections[0]
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)

    # ---------------- Title page ----------------
    add_para(doc, "Final Project Report", color=GRAY, align=WD_ALIGN_PARAGRAPH.CENTER,
              size=12, space_after=6)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("Predicting Road Accident Severity in the UK")
    run.font.size = Pt(28)
    run.font.bold = True
    run.font.color.rgb = PRIMARY_BLUE
    p.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("A Multi-Source Machine Learning Approach")
    run.italic = True
    run.font.size = Pt(16)
    run.font.color.rgb = LIGHT_BLUE
    p.paragraph_format.space_after = Pt(18)

    hr(doc)

    for line in [
        "Course: Project in Artificial Intelligence (02360502)",
        "Semester: Spring 2026",
        "Student 1: Ibraheem Akaree  |  ID: 212930846  |  ibraheema@campus.technion.ac.il",
        "Student 2: Sewar Abahry  |  ID: 325345734  |  sewar.abahry@campus.technion.ac.il",
        "Advisor: Yotam Gavish",
    ]:
        add_para(doc, line, align=WD_ALIGN_PARAGRAPH.CENTER, size=11.5, space_after=4)

    add_para(doc, "")
    add_para(
        doc,
        "This report supersedes the project proposal (docs/project_proposal.pdf) with the "
        "system as actually built and evaluated: the data pipeline, the eight frozen XGBoost "
        "severity classifiers, and the experimental results gathered to date. The Active "
        "Feature Acquisition reinforcement-learning agent described in the proposal is a "
        "designed but not-yet-implemented extension; its status is stated explicitly in "
        "Section 5.",
        italic=True, color=GRAY, align=WD_ALIGN_PARAGRAPH.CENTER, size=10,
    )

    add_page_break(doc)

    # ---------------- Table of contents ----------------
    add_heading(doc, "Table of Contents", level=1)
    add_toc(doc)
    add_page_break(doc)

    # ================= 1. Introduction =================
    add_heading(doc, "1. Introduction", level=1)

    add_heading(doc, "1.1 Problem Statement", level=2)
    add_para(
        doc,
        "Road traffic accidents impose a significant burden on emergency services, "
        "healthcare systems, and public safety infrastructure. In Great Britain alone, tens "
        "of thousands of personal-injury accidents are recorded annually by the police. When "
        "an accident is reported to emergency services, dispatchers must decide within "
        "seconds, and with incomplete information, what level of response to send: a single "
        "patrol car, an ambulance, or a full emergency response team.",
    )
    add_para(
        doc,
        "This project addresses the question: given only the information available in the "
        "first moments after a 999 call — location, time, road type, and whatever the "
        "caller can report — can accident severity be predicted accurately enough to be "
        "useful for dispatch? A reliable severity predictor at dispatch centres could help "
        "optimise emergency resource allocation, reduce response times for the most serious "
        "accidents, and ultimately save lives.",
    )

    add_heading(doc, "1.2 Approach", level=2)
    add_para(
        doc,
        "The problem is cast as supervised classification over the STATS19 "
        "collision_severity label (Fatal / Serious / Slight, and a binary "
        "Slight-vs-Severe collapse). Rather than relying on STATS19 alone, every accident "
        "record is enriched with three external, non-trivially-joined data sources — "
        "OpenStreetMap road infrastructure, DfT AADF traffic volume, and Open-Meteo historical "
        "weather — each of which may or may not be available, or worth the latency, at "
        "dispatch time. Eight classifiers are trained, one per subset of the three external "
        "sources, so that whichever combination of sources happens to be available at "
        "inference time, a matching model exists. The long-term goal (Section 5) is a "
        "reinforcement-learning agent that decides at runtime which of the three sources are "
        "worth acquiring before invoking the matching classifier; that agent is not yet "
        "built, and the classifiers described in this report are the environment it will "
        "eventually sit on top of.",
    )

    add_page_break(doc)

    # ================= 2. Data Pipeline =================
    add_heading(doc, "2. Data Pipeline & Dataset Construction", level=1)

    add_para(
        doc,
        "The predictive features are drawn from four sources: STATS19, the UK Department "
        "for Transport's official road-accident database, which supplies the primary "
        "records and the severity label; DfT AADF, annual traffic-volume counts joined to "
        "each accident by nearest count point; OpenStreetMap, road-infrastructure detail "
        "(speed cameras, lane count, medians) pulled via the Overpass API and Geofabrik "
        "extracts; and Open-Meteo, objective historical weather at the accident's exact "
        "location and hour, replacing STATS19's subjective officer-recorded weather field. "
        "STATS19, AADF, and OpenStreetMap are free; Open-Meteo requires a paid Professional "
        "plan (€99) for reliable historical-data volume. The pipeline below folds all four "
        "into a single feature table stage by stage.",
    )

    add_para(
        doc,
        "The full pipeline is a strict linear chain of CSV-in/CSV-out R stages, orchestrated "
        "by CODE/run_pipeline.R and documented in detail in docs/data_pipeline.md. Each stage "
        "is named after the sources folded in so far: stats19 → stats19+aadf → "
        "stats19+aadf+osm → ...+meteo → ...+risk → ...+cas, after which the table "
        "is split into train/valid/test, normalized, and expanded into masked copies for "
        "partial-observability training.",
    )

    add_heading(doc, "2.1 Severity-Aware Sampling", level=2)
    add_para(
        doc,
        "STATS19 is heavily imbalanced toward Slight-injury collisions in the real "
        "population. stats19_download.R downloads accident years 2016–2025 and applies a "
        "deliberate, disclosed sampling policy: every Fatal collision is kept, while Serious "
        "and Slight collisions are capped per year (SERIOUS_CAP / SLIGHT_CAP). This "
        "class-balance choice is central to the project's reported metrics and has been "
        "revisited twice; the current configuration and its history are summarised below.",
    )

    add_heading(doc, "2.2 Dataset-Difficulty History", level=2)
    add_para(
        doc,
        "Macro F1 on the binary Slight-vs-Severe task turns out to depend primarily on what "
        "fraction of the Severe class is Fatal — Fatal collisions are comparatively easy "
        "to classify, while Serious-vs-Slight is the genuinely hard boundary — which the "
        "Serious/Slight caps directly control. Three configurations have been built and "
        "evaluated:",
    )
    add_table(
        doc,
        ["Config", "Caps (Serious / Slight)", "Collisions", "Fatal-of-Severe", "Test macro F1"],
        [
            ["1 — original", "2,800 / 5,600", "89,791", "36.0%", "0.6967"],
            ["2 — scale-up (reverted)", "25,000 / 22,400", "415,550", "6.6% (≈ real population)", "0.6704"],
            ["3 — current (active)", "1,400 / 2,800", "51,989", "53.0%", "0.7322"],
        ],
        widths=[4.2, 4.0, 2.8, 3.5, 2.5],
    )
    add_para(
        doc,
        "Config 2 was a deliberate attempt to build a near-population-realistic benchmark "
        "(4.6× more data, Fatal-of-Severe close to the true ~7%): it produced a harder, more "
        "representative dataset on which macro F1 fell despite additional tuning and new "
        "features. A cap-sweep across seven configurations confirmed that score tracks "
        "Fatal-of-Severe share monotonically rather than dataset size — larger datasets "
        "scored worse here because every added row was a hard Serious case. Config 3, the "
        "currently active dataset, was then built by subsampling the archived config-2 data "
        "down to half the original caps, explicitly to buy a macro F1 above 0.70. This is a "
        "legitimate but material choice: Fatal is ≈53% of Severe in the active dataset "
        "against ≈7% in the real population, so every macro-F1 figure in this report should "
        "be read as a score on that benchmark, not an estimate of field performance "
        "(Section 6.6 discusses the population-realistic reconstruction). All three configs' "
        "data are preserved on disk / in git history for reproducibility.",
    )

    add_heading(doc, "2.3 Feature Engineering", level=2)
    add_para(
        doc,
        "Starting from the STATS19 fields available at call time, the pipeline successively "
        "adds vehicle-type flags, the AADF/OSM/Open-Meteo external features, spatial "
        "risk-density features, and dispatch-time casualty features, arriving at a "
        "61-column combined table (89,779 rows in the reference pipeline run).",
    )
    add_table(
        doc,
        ["Stage", "Adds", "Notes"],
        [
            ["STATS19 realtime (collisions)", "14 ML features (road type, junction, speed limit, time, …)",
             "Only fields available before post-collision investigation"],
            ["STATS19 realtime (vehicles)", "7 vehicle-type flags (has_hgv, has_car, …)",
             "Aggregated from one-row-per-vehicle via multi-hot flags"],
            ["AADF join", "aadf_traffic_volume, aadf_distance_m, aadf_match_type",
             "Nearest count point on the same road, falling back to nearest overall"],
            ["OSM join", "has_speed_camera, has_traffic_calming, has_traffic_signals, lane_count, has_median",
             "Two-stage ref-then-nearest road matching; 100% coverage"],
            ["Open-Meteo join", "temperature_2m_c, precipitation_mm, wind_speed_10m_kmh, weather_code, snowfall_cm, snow_depth_m",
             "Batched by date; objective replacement for the subjective STATS19 weather field"],
            ["Risk-density", "hist_acc_rate_500m, hist_severe_frac_500m, hist_fatal_rate_500m, dist_to_nearest_hist_acc_m",
             "Computed only from strictly prior-year accidents — no leakage; +0.005 macro F1"],
            ["Dispatch-time casualties", "n_casualties, n_pedestrian_casualties, youngest/oldest_casualty_age",
             "Single largest accuracy gain in the project (Section 6.3)"],
        ],
        widths=[3.6, 6.0, 5.5],
    )

    add_heading(doc, "2.4 Data Variants", level=2)
    add_bullets(doc, [
        ("raw ", "— the untransformed combined feature table, split into train/valid/test."),
        ("normalized ", "— cyclic-encoded (day-of-week, hour), log1p-transformed for "
                         "12 right-skewed columns, and z-scored (17 columns) using train-only "
                         "statistics; one row per collision. This is the variant currently used "
                         "for the project's headline binary results."),
        ("masked ", "— built from the normalized splits; each row is expanded into 11 "
                     "copies (1 unmasked + 10 with 0–23 of the 23 acquirable STATS19/vehicle/"
                     "casualty fields randomly blanked), simulating the partial information a "
                     "dispatcher actually has. This is why the masked variant has ≈11× the "
                     "row count of normalized/raw."),
    ])
    add_para(
        doc,
        "These three variants exist to separate three different questions: whether "
        "normalization matters, whether partial observability matters, and what the ceiling "
        "looks like with neither transformation. A hyperparameter configuration tuned on one "
        "variant does not automatically transfer to another — the masked variant's larger, "
        "noisier data needed heavier regularization than the clean normalized variant.",
    )

    add_page_break(doc)

    # ================= 3. Methodology =================
    add_heading(doc, "3. Model Architecture & Training", level=1)

    add_heading(doc, "3.1 Eight Source-Subset Classifiers", level=2)
    add_para(
        doc,
        "Because any of the three external sources may or may not be acquired before a "
        "prediction is needed, eight classifiers are trained — one per subset of "
        "{AADF, OSM, meteo}, from base-only (no external sources) up to full acquisition "
        "(all three). Each is an XGBoost classifier using native categorical and "
        "missing-value handling, trained via CODE/training/train/train_classifier.py, which "
        "defines the feature set through BASE_COLS (always present) plus SOURCE_COLS[source] "
        "(added when that source is acquired). The models are frozen once trained — they "
        "form the environment that the planned RL agent (Section 5) will later query.",
    )

    add_heading(doc, "3.2 Target Variable", level=2)
    add_para(
        doc,
        "Two target formulations are supported: the native 3-class STATS19 severity label "
        "(Fatal / Serious / Slight), and a binary Slight-vs-Severe collapse (Fatal and "
        "Serious merged into Severe). The project's current focus is the binary target on "
        "the normalized data variant, since it isolates the hardest and most operationally "
        "relevant boundary — whether a response needs to be escalated beyond a routine "
        "one — without the added noise of partial-observability masking.",
    )

    add_heading(doc, "3.3 Hyperparameter Tuning", level=2)
    add_para(
        doc,
        "Hyperparameters are tuned with Optuna (CODE/training/tune/"
        "tune_full_acquisition_classifier.py), separately per data variant and target "
        "formulation, and cached to models/best_hyperparams*.json for reuse. A re-tune on "
        "the current 54-column feature set converged to essentially the same configuration "
        "as before, but retraining with it still lifted the flagship 3-class classifier's "
        "uncalibrated test macro F1 from 0.4845 to 0.4928 (0.503 → 0.5028 after threshold "
        "calibration), driven mostly by the weak Serious class. The binary flagship stayed "
        "essentially flat (0.651 → 0.6517) under the same re-tune.",
    )

    add_heading(doc, "3.4 Threshold Calibration", level=2)
    add_para(
        doc,
        "CODE/training/train/calibrate_thresholds.py post-processes an already-trained "
        "model's predicted probabilities to pick per-class decision thresholds that improve "
        "macro F1 without retraining — useful because the severity-oversampled training "
        "distribution does not match the decision boundary that maximises macro F1 on a more "
        "balanced evaluation split.",
    )

    add_page_break(doc)

    # ================= 4. Model family comparison (moved before RL per numbering) ===
    add_heading(doc, "4. Choice of Model Family", level=1)
    add_para(
        doc,
        "Before committing to XGBoost as the deployed model, seventeen model families were "
        "compared on identical data: the normalized variant, binary target, full acquisition "
        "(aadf+osm+meteo, 52 features), the same train/valid/test split, balanced class "
        "weighting where supported, 3 seeds, scored on test.",
    )
    add_table(
        doc,
        ["Model", "Macro F1", "Balanced acc.", "AUC"],
        [
            ["XGBoost (deployed)", "0.7367", "0.7370", "0.8119"],
            ["CatBoost", "0.7300", "0.7302", "0.8068"],
            ["HistGradientBoosting", "0.7300", "0.7302", "0.8054"],
            ["LightGBM", "0.7269", "0.7270", "0.8059"],
            ["Random forest", "0.7245", "0.7244", "0.7986"],
            ["Neural net (MLP 128-64)", "0.7244", "0.7247", "0.8003"],
            ["Linear SVM / Logistic regression", "0.717 / 0.717", "0.718 / 0.717", "0.791 / 0.792"],
            ["Decision tree (depth 5)", "0.6707", "0.6707", "0.7298"],
            ["One-rule baseline (Holte's OneR)", "0.6047", "0.6102", "0.6102"],
            ["Random (stratified)", "0.5032", "0.5033", "0.5033"],
            ["Majority class", "0.3403", "0.5000", "0.5000"],
        ],
        widths=[6.0, 2.7, 3.0, 2.3],
    )
    add_bullets(doc, [
        ("Against a proper baseline, the model earns +0.132 macro F1 ",
         "over the one-rule (laminated-card) baseline, and +0.231 over random guessing by "
         "class prior — the comparison that measures what the model actually contributes."),
        ("Among learning algorithms the spread is small: ",
         "OLS thresholded at 0.5 reaches 0.7165, within 0.020 of the tuned booster. The "
         "engineered features have already absorbed most of the nonlinearity, so the pipeline, "
         "not the algorithm, is doing most of the work."),
        ("Four gradient boosters, a random forest, and a neural net all land in a 0.72–0.73 "
         "band ", "— the signature of an information ceiling set by the data, not the "
         "model family."),
        ("With an equal 120-trial tuning budget per family, ",
         "XGBoost's lead over the field narrows to +0.0015 macro F1 over LightGBM — inside "
         "seed noise. XGBoost is retained for a small, consistent edge plus roughly 11× faster "
         "training than tuned LightGBM."),
    ])
    add_para(
        doc,
        "Full methodology, all seventeen families, and seven supporting figures are in "
        "docs/model_family_comparison.md.",
        italic=True, color=GRAY, size=10,
    )

    add_page_break(doc)

    # ================= 6. RL / AFA =================
    add_heading(doc, "5. Active Feature Acquisition (Planned Extension)", level=1)
    add_para(
        doc,
        "The eventual goal of the project is a reinforcement-learning agent that decides, at "
        "runtime and per accident, which of the three external sources are worth acquiring "
        "before invoking the matching frozen classifier from Section 3 — trading prediction "
        "confidence against acquisition latency and API cost. This component has been fully "
        "designed but is not yet implemented; the eight classifiers described above are the "
        "environment it will be trained against. The design is summarised here so the "
        "planned system is documented alongside the parts that already exist.",
    )

    add_heading(doc, "5.1 MDP Formulation", level=2)
    add_para(
        doc,
        "State: sₜ = (x_base, x_acquired, mₜ, pₜ), where x_base is the STATS19 fields known "
        "automatically at call time, x_acquired is the partial feature vector built up as the "
        "agent acquires fields, mₜ is a binary mask of which features/sources have been "
        "obtained, and pₜ is the classifier's current predicted probability distribution over "
        "severity classes given what has been acquired so far.",
    )
    add_para(
        doc,
        "Actions: A0 (predict & terminate, returning the current pₜ); A1…Ak, one per "
        "individually acquirable STATS19/vehicle/casualty field (asking the dispatcher); and "
        "one grouped action per external API source (OSM, AADF, Open-Meteo), since all "
        "features from a given API arrive together in a single call.",
    )
    add_para(
        doc,
        "Reward: Reward = R_terminal + Σₜ R_step(t), with R_step(t) = α·ΔConfidenceₜ − Cᵢ "
        "(a small reward for the reduction in prediction entropy after acquiring feature i, "
        "minus that feature's acquisition cost) and R_terminal = β·Accuracy(ŷ, y) (macro F1 "
        "against ground truth, delivered only at termination). α is kept small relative to "
        "β so the dense per-step signal guides exploration without overshadowing the terminal "
        "objective; both are treated as tunable design parameters.",
    )

    add_heading(doc, "5.2 Planned Baselines", level=2)
    add_bullets(doc, [
        ("No-acquisition ", "— the base-only classifier; zero cost, lower-bound accuracy."),
        ("Full-acquisition ", "— always acquires all three sources; maximum cost, "
                               "upper-bound accuracy (see Section 6.5 for the closely related "
                               "full-STATS19 ceiling)."),
        ("Static feature priority ", "— a fixed acquisition order (e.g. weather → OSM → "
                                      "AADF) applied up to the same average cost budget as the "
                                      "learned agent, unable to adapt per accident."),
    ])
    add_para(
        doc,
        "The per-source ablation results in Section 6 are the intended basis for calibrating "
        "each source's acquisition cost Cᵢ in the reward function: a source with a large "
        "accuracy gain relative to its cost should be cheap to acquire; a source that "
        "contributes little should be expensive enough that the agent learns to skip it.",
    )

    add_page_break(doc)

    # ================= 7. Experiments & Results =================
    add_heading(doc, "6. Experiments & Results", level=1)

    add_heading(doc, "6.1 Headline Result", level=2)
    add_para(
        doc,
        "The current project focus, the full-acquisition (aadf+osm+meteo) binary "
        "Slight-vs-Severe classifier on the normalized data variant, reaches test macro F1 = "
        "0.7322 in its single-run training log and 0.7367 ± 0.001 averaged over three seeds "
        "in the model-family comparison (Section 4) — both measured on the current, "
        "1,400/2,800-cap dataset (Section 2.2). All macro-F1 figures in this report are tied "
        "to a specific dataset configuration; comparing across configurations without noting "
        "which one is in play is misleading, since the score is driven substantially by how "
        "Fatal-heavy the Severe class is (Section 2.2).",
    )

    add_heading(doc, "6.2 Dispatch-Time Ceiling: Full-STATS19 Upper Bound", level=2)
    add_para(
        doc,
        "To measure the accuracy cost of restricting the model to genuinely dispatch-time "
        "information, a reference “upper-bound” model was trained using every pre-label "
        "STATS19 field — forensic vehicle/casualty detail, driver demographics, road "
        "surface, whether police attended, and more — none of which a dispatcher actually "
        "has. This is explicitly not a deployment candidate (police_attended in particular is "
        "leakage-adjacent, since attendance is itself driven by how severe the collision "
        "looks).",
    )
    add_bullets(doc, [
        ("Full model: test macro F1 0.7625 / accuracy 0.7627 ",
         "versus the dispatch model's 0.7367 — a +0.026 ceiling on the current dataset."),
        ("The gap is size-invariant ", "(≈+0.027 across a 6× range of dataset sizes/severity "
         "mixes): the dispatch-time constraint is a fixed structural information cost, "
         "supporting the project's Active Feature Acquisition framing — closing the gap "
         "requires external sources, not more STATS19 rows."),
        ("Dedicated tuning of the 84-feature full model did not help ",
         "(0.7610 vs 0.7625 using the dispatch model's tuned hyperparameters) — the fifth "
         "independent search in this project to converge on the same configuration."),
        ("On a natural ~79%-Slight distribution ", "(clean split, no leakage) the full model "
         "reaches ~0.84 accuracy but only ~0.375 Severe recall — the honest version of the "
         "90%+ accuracy figures reported in some published STATS19 studies, which turn out to "
         "reflect majority-class accuracy on imbalanced data (and, in at least one paper, "
         "class rebalancing applied before the train/test split, a leakage artifact this "
         "project reproduced separately). Accuracy is reported only alongside per-class "
         "recall throughout this project for exactly this reason."),
    ])

    add_heading(doc, "6.3 Dispatch-Time Casualty Features", level=2)
    add_para(
        doc,
        "Aggregating the casualties table into four dispatch-time-plausible fields "
        "(n_casualties, n_pedestrian_casualties, youngest/oldest_casualty_age — the kind of "
        "detail a 999 caller volunteers, such as “a pedestrian's been hit” or “three people "
        "hurt”) was the single largest accuracy improvement found in the project: 3-class "
        "calibrated test macro F1 rose from 0.478 to 0.501 (Fatal F1 0.40 → 0.45, precision "
        "and recall both up), and binary Slight-vs-Severe macro F1 rose from 0.631 to 0.651. "
        "An extended set of five further demographic/mode-of-travel columns was trialled on "
        "top of these and removed after a controlled ablation showed only +0.0012 macro F1 on "
        "normalized (within noise) and +0.0052 on masked — not enough to justify the extra "
        "complexity.",
    )

    add_heading(doc, "6.4 Spatial Risk-Density Features", level=2)
    add_para(
        doc,
        "Four location-history features (accident rate, severe fraction, fatal rate, and "
        "distance to the nearest prior accident, all within 500 m) are computed using only "
        "strictly prior-year accidents, so they carry no future information. They add a "
        "small but consistent ~+0.005 macro F1 on both the 3-class and binary targets, on "
        "both validation and test.",
    )

    add_heading(doc, "6.5 Vehicle-Type Flags", level=2)
    add_para(
        doc,
        "Of fifteen candidate per-vehicle-type flags derived from the vehicles table, eight "
        "ultra-rare flags (each under 1.5% prevalence) were dropped after showing an "
        "overfitting signature — e.g. has_tram, at 0.02% prevalence, still appeared in 123 "
        "tree splits. The remaining seven were kept on the strength of a healthy "
        "split-to-prevalence ratio; has_hgv in particular carried the highest gain of any "
        "flag, and removing it cost real accuracy (test macro F1 0.697 → 0.692 on the "
        "normalized binary variant).",
    )

    add_heading(doc, "6.6 Population-Realistic Evaluation", level=2)
    add_para(
        doc,
        "Because the training/evaluation splits are deliberately severity-oversampled "
        "(≈52% Severe here vs ≈21% in the real population, and Fatal ≈53% of Severe here vs "
        "≈7% in reality), macro F1 and balanced accuracy — not plain accuracy — are the "
        "metrics trusted throughout this project; plain accuracy on these splits is "
        "systematically lower than field accuracy would be, and can be gamed upward by a "
        "model that leans toward the majority class. CODE/experiments/evaluate_population.py "
        "reconstructs a true population-accuracy estimate from the per-year sampling "
        "fractions recorded in each STATS19 download's metadata file, and is reported "
        "alongside minority-class recall to make clear how many severe accidents a "
        "deployment-realistic model would actually miss.",
    )

    add_page_break(doc)

    # ================= 8. Discussion =================
    add_heading(doc, "7. Discussion & Limitations", level=1)
    add_bullets(doc, [
        ("Class balance is deliberately non-natural. ",
         "Every headline metric in this report is measured on a Fatal-enriched benchmark "
         "chosen to make the classification problem tractable and comparable across "
         "experiments; it is not an estimate of real-world dispatch accuracy. Section 6.6 "
         "gives the reconstruction path back to a population-realistic estimate."),
        ("OSM is a present-day snapshot. ", "Road-infrastructure features reflect the "
         "network as it exists today, not as it was at each historical accident's date, "
         "since Geofabrik does not publish historical extracts at this granularity."),
        ("Open-Meteo weather is paid at production volume. ",
         "The free archive endpoint's rate limits (600 calls/min, 5,000/hour, 10,000/day) are "
         "workable for offline training-data construction but would need a paid plan for "
         "real-time dispatch-time acquisition at scale."),
        ("Model-family choice barely matters once the feature pipeline is fixed. ",
         "Section 4 shows the gap between the deployed XGBoost model and a plain thresholded "
         "linear regression is only 0.020 macro F1 — most of this project's measurable "
         "contribution is in the feature engineering, not the learning algorithm."),
        ("The Active Feature Acquisition agent is not yet built. ",
         "Section 5 documents its design; implementing, training, and evaluating it against "
         "the three baselines remains the primary open item in the project plan."),
    ])

    add_page_break(doc)

    # ================= 8. Conclusion =================
    add_heading(doc, "8. Conclusion & Future Work", level=1)
    add_para(
        doc,
        "This project has built and validated a complete, reproducible pipeline for "
        "predicting road-accident severity from dispatch-time-available information, enriched "
        "with three external data sources joined through non-trivial spatial and temporal "
        "logic. Eight frozen XGBoost classifiers cover every combination of external-source "
        "availability, and the flagship full-acquisition binary classifier reaches test macro "
        "F1 ≈ 0.73–0.74 on the current, disclosed, Fatal-enriched benchmark — +0.132 macro "
        "F1 over a non-learned one-rule baseline, with the dispatch-time restriction itself "
        "costing only ≈+0.026 macro F1 relative to a model given every pre-label STATS19 "
        "field. Model-family comparisons show this gap is set almost entirely by the "
        "engineered feature set rather than by the choice of learning algorithm.",
    )
    add_para(
        doc,
        "The immediate next steps are: (1) implement and train the Active Feature Acquisition "
        "reinforcement-learning agent designed in Section 5 against the frozen classifiers as "
        "its environment, calibrating per-source acquisition costs from the ablation results "
        "in Section 6; (2) evaluate the trained agent against the no-acquisition, "
        "full-acquisition, and static-priority baselines; and (3) extend the population-"
        "realistic reconstruction (Section 6.6) into a standard reporting step alongside every "
        "future macro-F1 result, so headline numbers and field-deployment expectations are "
        "never conflated.",
    )

    add_page_break(doc)

    # ================= Appendix =================
    add_heading(doc, "Appendix A: Repository & Reproducibility", level=1)
    add_para(
        doc,
        "The full codebase, all raw and processed data (where not gitignored for size), and "
        "the exact commands to reproduce every result in this report are in the project "
        "repository. Key commands (run from the repo root, with the project's .venv "
        "activated):",
    )
    add_table(
        doc,
        ["Task", "Command"],
        [
            ["Run the full R data pipeline", "Rscript CODE/run_pipeline.R"],
            ["Train the flagship binary classifier", "python CODE/training/train/train_classifier.py --sources aadf osm meteo --binary --data-variant normalized --use-best-params"],
            ["Tune hyperparameters", "python CODE/training/tune/tune_full_acquisition_classifier.py --n-trials 60 --binary --data-variant normalized"],
            ["Calibrate decision thresholds", "python CODE/training/train/calibrate_thresholds.py --sources aadf osm meteo --binary --data-variant normalized"],
            ["Model-family comparison", "python CODE/experiments/compare_model_families.py --seeds 42 43 44"],
            ["Population-accuracy reconstruction", "python CODE/experiments/evaluate_population.py"],
        ],
        widths=[4.5, 9.5],
        header_size=10, body_size=9,
    )
    add_para(
        doc,
        "There is no automated test suite, linter, or CI configuration in this repository; "
        "correctness is validated experimentally, via the ablations and comparisons "
        "summarised in Section 6.",
    )

    add_heading(doc, "Appendix B: References", level=1)
    add_bullets(doc, [
        "STATS19 (UK Department for Transport), via github.com/ropensci/stats19.",
        "DfT Annual Average Daily Flow traffic counts, roadtraffic.dft.gov.uk/downloads.",
        "OpenStreetMap, via the Overpass API (overpass-api.de) and Geofabrik Great Britain extracts (download.geofabrik.de).",
        "Open-Meteo Historical Weather API, open-meteo.com/en/docs/historical-weather-api.",
        "XGBoost: Chen & Guestrin, “XGBoost: A Scalable Tree Boosting System,” KDD 2016.",
        "Optuna: Akiba et al., “Optuna: A Next-generation Hyperparameter Optimization Framework,” KDD 2019.",
    ])

    doc.save(OUT_PATH)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    build()
