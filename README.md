# AI-Enabled Multiplex Biosensor System
### A Point-of-Care Diagnostic Pipeline for AMR, Biofilm & Oncology-Associated Microbial Biomarkers

> **For researchers**: Complete documentation covering the biology, electrochemistry, full AI/ML architecture, every file and dataset column, real-hardware signal processing, and step-by-step instructions to run everything from raw data to clinical report.

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [Scientific Background](#2-scientific-background)
3. [The 10-Stage Biological Pipeline](#3-the-10-stage-biological-pipeline)
4. [Repository Structure](#4-repository-structure)
5. [Data Files Explained](#5-data-files-explained)
6. [AI/ML Architecture — In Depth](#6-aiml-architecture--in-depth)
7. [Pipeline Code Explained — Module by Module](#7-pipeline-code-explained--module-by-module)
8. [Real Hardware Signal Processing](#8-real-hardware-signal-processing)
9. [How to Run the System](#9-how-to-run-the-system)
10. [Output Files and How to Interpret Them](#10-output-files-and-how-to-interpret-them)
11. [Autodocking Sub-Pipeline](#11-autodocking-sub-pipeline)
12. [Key Design Decisions](#12-key-design-decisions)
13. [Model Performance Summary](#13-model-performance-summary)
14. [Extending the System](#14-extending-the-system)
15. [Dependencies](#15-dependencies)
16. [Web Simulator Dashboard](#16-web-simulator-dashboard)
17. [License](#17-license)

---

## 1. What This System Does

This project implements a **complete AI-powered point-of-care (POC) biosensor pipeline** — from raw electrochemical signal to clinical decision report — that simultaneously screens three clinical domains:

| Domain | Biomarkers Detected | Clinical Output |
|--------|--------------------|----|
| **AMR** (Antimicrobial Resistance) | blaNDM-1, mecA, vanA, KPC | Which antibiotics will fail; recommended alternative therapy |
| **Biofilm** | icaADBC, AHL/AI-2, bap/pel/psl, c-di-GMP | Biofilm stage (I–IV); quorum sensing activity; matrix genes |
| **Oncology** | FadA, CagA, pks locus, miRNA-21 | Cancer-associated microbial signatures; referral flag |

**Physical input**: ~5 µL clinical sample (blood, urine, wound swab, biopsy fluid)  
**Output**: Standalone HTML clinical decision report in **< 30 minutes**, transmitted via BLE or Wi-Fi

### Two modes of operation

| Mode | Entry point | What it does |
|---|---|---|
| **Batch pipeline** | `run.py` | Loads 500-sample CSV → trains all models → evaluates → generates 10 HTML reports |
| **Live simulation** | `run2.py` | Streams µA analog signal in real-time → Kalman filter → detection → AI → 1 HTML report |

---

## 2. Scientific Background

### Why multiplex?
Conventional diagnostics test for one pathogen at a time. This device runs 12 channels simultaneously on a single nanostructured gold/graphene oxide electrode chip:
- AMR + Biofilm co-infection is far harder to treat — biofilm reduces antibiotic penetration by 100–1000×
- *F. nucleatum* and *H. pylori* are invisible until tissue biopsy; detecting them early changes outcomes
- POC speed (< 30 min) vs. lab culture (24–72 h) changes the initial treatment decision

### Signal-OFF electrochemistry
The biosensor uses **Signal-OFF amperometric detection**. Baseline current when no target is present is ~10 nA. When a biomarker molecule binds the probe, it physically blocks the electrode surface and current **drops logarithmically** with concentration:

```
I(C) = 10.0 − 2.1 × log₁₀(C_pM) + ε      ε ~ N(0, 0.15 nA)
```

Detection threshold: **≥ 30% drop** from the pre-injection baseline.

| Channel state | Current | Meaning |
|---|---|---|
| Healthy (no target) | ~10.0 nA | Baseline maintained |
| Weakly positive | 8–9.5 nA | Trace concentration, 5–20% drop |
| Clearly positive | 4–7 nA | 30–60% drop, concentration 20–500 pM |
| Saturated | < 3 nA | High concentration, > 70% drop |

### Probe chemistries
| Channel group | Probe type | Mechanism |
|---|---|---|
| AMR Ch01–04 | **CRISPR-Cas12a** | Cas12a trans-cleavage activated only by exact resistance gene sequence — single nucleotide specificity |
| Biofilm Ch05–08 | **Aptamer / DNAzyme** | Aptamers bind small molecules (AHLs, c-di-GMP) with high affinity; DNAzymes provide catalytic amplification |
| Oncology Ch09–12 | **Antibody / Aptamer** | Protein antigens (FadA, CagA) captured by monoclonal antibodies; miRNA-21 by complementary aptamer |

---

## 3. The 10-Stage Biological Pipeline

Corresponds to `biosensor_system_explainer.txt` and `ai_multiplex_biosensor_pipeline.svg`:

| Stage | Name | What happens | Code |
|---|---|---|---|
| 1 | **Sample collection** | 5 µL enters credit-card cartridge; passive microfluidics routes to 3 channel groups | `detailed_dummy_generator.py` → patient metadata |
| 2 | **Molecular recognition** | CRISPR / aptamer / antibody probes bind targets at electrode surface | `assign_ground_truth()` |
| 3 | **Electrochemical transduction** | Binding → current drop (amperometric) + surface impedance change | `raw_signal()` / `run2.py` live stream |
| 4 | **Signal preprocessing** | Adaptive Kalman filter; feature extraction per channel | `pipeline/signal_processor.py`, `pipeline/real_device_adapter.py` |
| 5 | **Deep learning inference** | 1D-CNN encoder → Transformer attention → task heads | `pipeline/multimodal_encoder.py` + `pipeline/models.py` |
| 6 | **Uncertainty quantification** | MC Dropout → 95% CI bounds per domain | `BayesianCalibration` in `multimodal_encoder.py` |
| 7 | **Explainability** | SHAP per-biomarker attribution | `compute_shap()` in `models.py` |
| 8 | **Device output** | Ensemble risk score; BLE/Wi-Fi → EHR | `EnsembleRiskEngine` in `models.py` |
| 9 | **Clinical report** | 3-panel HTML: AMR risk, biofilm status, oncology risk | `pipeline/clinical_report.py` |
| 10 | **Continuous learning** | Lab confirmations re-train models (online learning loop) | Placeholder in `run.py` |

---

## 4. Repository Structure

```text
Multiplex-AI-Biosensor/
├── app.py                             ← Web Simulator entry point (Flask)
├── web_simulator.py                   ← Interactive dashboard logic & API routes
├── run.py                             ← Batch pipeline orchestrator
├── run2.py                            ← Live device simulation (CLI)
├── detailed_dummy_generator.py        ← Synthetic dataset & metadata factory
├── data_generater.py                  ← Legacy generator script
├── pipeline/
│   ├── __init__.py 
│   ├── signal_processor.py            ← Layer 2: Adaptive Kalman + feature extraction
│   ├── multimodal_encoder.py          ← Layer 3: 1D-CNN + transformer attention
│   ├── models.py                      ← XGBoost/GB/RF task heads + ensemble risk
│   ├── clinical_report.py             ← HTML decision report generator
│   ├── real_device_adapter.py         ← Parses real hardware sensor data to AI format
│   └── train_pipeline.py              ← Standalone training/eval script
├── data/                              ← Simulated patient CSV datasets & true labels
│   ├── biosensor_detailed_500.csv
│   └── ...
├── reports/                           ← Autogenerated HTML clinical reports
│   └── report_*.html
├── templates/
│   └── index.html                     ← Web dashboard UI (HTML)
├── static/
│   └── style.css                      ← Web dashboard UI (CSS)
├── assets/                            ← Flowcharts and diagrams (.svg, .png)
├── docs/                              ← Explanatory text & architecture documentation
├── references/                        ← Underlying research papers & PDFs
├── README.md                          ← Main documentation
└── LICENSE                            ← All Rights Reserved Copyright
```

---

## 5. Data Files Explained

### `biosensor_detailed_500.csv` — Main Dataset (500 × 165 columns)

Every row is one patient sample. Columns are organised by pipeline stage:

#### Stage 1 — Patient & Device Metadata (13 columns)
| Column | Type | Description |
|--------|------|-------------|
| `Sample_ID` | str | Unique ID e.g. `BIO0001` |
| `Collection_Date` | date | YYYY-MM-DD |
| `Collection_Time` | time | HH:MM |
| `Patient_Age` | int | 1–95, µ=52, σ=18 |
| `Patient_Sex` | str | M / F |
| `Patient_BMI` | float | 15–50 |
| `Sample_Type` | str | Blood / Urine / Wound_swab / Biopsy_fluid |
| `Collection_Site` | str | ICU / ED / Outpatient_clinic / Surgery_ward / Community |
| `Sample_Volume_uL` | float | 4.5–6.0 µL |
| `Cartridge_Lot` | str | Batch traceability |
| `Device_ID` | str | Which physical device |
| `Ambient_Temp_C` | float | Temperature at collection (affects signal) |
| `Ambient_pH` | float | pH of sample medium |

#### Stage 3+4 — Electrochemical Signals + Features (8 columns × 12 channels = 96 columns)

For each channel `ChXX` (Ch01–Ch12):

| Column | Description |
|--------|-------------|
| `ChXX_raw_nA` | Raw current from electrode (baseline ~10 nA) |
| `ChXX_smooth_nA` | Kalman-filtered signal |
| `ChXX_conc_pM` | Back-calculated concentration: C = 10^((10−I) / 2.1) |
| `ChXX_peak_amp_nA` | Drop magnitude: baseline − smooth (nA) |
| `ChXX_drop_pct` | % drop from baseline (≥30% = detected) |
| `ChXX_t2t_s` | Seconds to reach 30% threshold (30–200s if present; ~1200s if absent) |
| `ChXX_impedance_pct` | Surface impedance change (%) |
| `ChXX_snr_db` | Signal-to-noise ratio (dB) |

#### Channel-to-Biomarker Mapping

| Channel | Domain | Biomarker | Probe | Target organism |
|---------|--------|-----------|-------|----------------|
| Ch01 | AMR | blaNDM-1 | CRISPR-Cas12a | *K. pneumoniae* / *E. coli* |
| Ch02 | AMR | mecA | CRISPR-Cas12a | *S. aureus* (MRSA) |
| Ch03 | AMR | vanA | CRISPR-Cas12a | *Enterococcus* (VRE) |
| Ch04 | AMR | KPC | CRISPR-Cas12a | *K. pneumoniae* |
| Ch05 | Biofilm | icaADBC | Aptamer | *S. aureus* / *S. epidermidis* |
| Ch06 | Biofilm | AHL/AI-2 | DNAzyme | Quorum sensing (pan-species) |
| Ch07 | Biofilm | bap/pel/psl | Aptamer | *P. aeruginosa* / *S. aureus* |
| Ch08 | Biofilm | c-di-GMP | DNAzyme | Universal 2nd messenger |
| Ch09 | Oncology | FadA | Antibody | *F. nucleatum* (colorectal cancer) |
| Ch10 | Oncology | CagA | Antibody | *H. pylori* (gastric cancer) |
| Ch11 | Oncology | pks | Antibody | *E. coli* genotoxin (colorectal) |
| Ch12 | Oncology | miRNA-21 | Aptamer | Pan-cancer epigenetic marker |

#### Stage 5 — AI Model Outputs (13 columns)
| Column | Description |
|--------|-------------|
| `AI_AMR_Score` | 0–100% probability of AMR presence |
| `AI_Biofilm_Score` | 0–100% biofilm severity |
| `AI_Oncology_Score` | 0–100% oncology risk |
| `XGB_blaNDM1_call` | Binary: blaNDM-1 detected? |
| `XGB_mecA_call` | Binary: mecA detected? |
| `XGB_vanA_call` | Binary: vanA detected? |
| `XGB_KPC_call` | Binary: KPC detected? |
| `LSTM_Temporal_Coherence` | 20–99: kinetic stability of biofilm signal |
| `GNN_Network_Centrality` | 0–100: co-occurrence network centrality |
| `GNN_Species_Co_detected` | Which oncology species co-detected |

#### Stage 6 — Uncertainty Quantification (6 columns)
| Column | Description |
|--------|-------------|
| `MC_AMR_StdDev` | Std dev across 50 MC Dropout passes |
| `MC_AMR_Conf_pct` | Confidence % (100 − 2.5 × StdDev) |
| `MC_Biofilm_StdDev` | Same for Biofilm |
| `MC_Biofilm_Conf_pct` | Same for Biofilm |
| `MC_Oncology_StdDev` | Same for Oncology |
| `MC_Oncology_Conf_pct` | Same for Oncology |

#### Stage 7 — SHAP Explainability (biomarker-level)
`biosensor_shap.csv` holds per-sample SHAP attributions. In the main CSV:
- `SHAP_AMR_TopDriver` — which biomarker most drove the AMR score
- `SHAP_Biofilm_TopDriver` — same for Biofilm
- `SHAP_Oncology_TopDriver` — same for Oncology

#### Stage 8 — Device Telemetry & Ensemble Output (6 columns)
| Column | Description |
|--------|-------------|
| `Transmission_Mode` | BLE / WiFi / Offline |
| `Total_Time_min` | Total assay time (22–32 min) |
| `Ensemble_Risk_Score` | 0–100 combined clinical risk |
| `Ensemble_Risk_Tier` | Low / Moderate / High / Critical |
| `N_Active_Domains` | Count of domains with score > 40 |
| `Co_occurrence_Bonus` | +0 / +8 / +15 (1 / 2 / 3 active domains) |

#### Stage 9 — Clinical Report Fields (17 columns)
| Column | Description |
|--------|-------------|
| `AMR_Resistance_Profile` | Susceptible / Single_resistance / Multi-drug_resistant / Pan-resistant |
| `AMR_Detected_Genes` | Pipe-separated list e.g. `blaNDM-1\|mecA` |
| `AMR_Failed_Antibiotics` | Antibiotics that will not work |
| `AMR_Recommended_Alt` | Recommended alternative therapy |
| `Biofilm_Stage` | None / Stage_I–IV |
| `icaADBC_Active` | 0/1 — biofilm scaffold gene active |
| `pel_psl_Active` | 0/1 — exopolysaccharide matrix genes active |
| `QS_Activity_Score` | 0–100 continuous quorum sensing intensity |
| `AI2_Level_uM` | AI-2 concentration (µM) |
| `c_diGMP_Level_uM` | c-di-GMP concentration (µM) |
| `Oncology_Risk_Tier` | Low / Moderate / High / Very_High |
| `Cancer_Microbes_Detected` | Pipe-separated biomarker names |
| `Oncology_Species` | Human-readable species |
| `Chronic_Inflammation_Load` | 0–10 score |
| `Referral_Recommended` | 0/1 — oncology / gastroenterology referral flag |

---

### `biosensor_ground_truth.csv` — Binary Ground Truth (500 × 13)

The "lab confirmation" file used as training labels. `Ch01–Ch12` = 1 if biomarker truly present, 0 if absent. In a real device this file would arrive 24–48 h later from the microbiology lab and feed the continuous learning loop.

### `biosensor_shap.csv` — SHAP Attribution Matrix (500 × 13)

Per-sample, per-biomarker attribution scores:
- **Positive**: biomarker pushed domain score up (evidence for presence)
- **Negative**: biomarker suppressed score (evidence against)
- Within-domain values approximately sum to the domain AI score

### `sample_device_input.csv` — Example Hardware Output (900 rows × 13 columns)

Generated by `pipeline/real_device_adapter.py`. Represents what a real potentiostat would send:
```
time_s, Ch01, Ch02, ..., Ch12
0,      0.0100126, 0.0099834, ..., 0.0100312   ← baseline ~10 nA = 0.01 µA
60,     0.0100051, 0.0098712, ..., 0.0100209   ← sample injected
120,    0.0099987, 0.0092451, ..., 0.0100158   ← Ch02 starting to drop
...
900,    0.0099943, 0.0060014, ..., 0.0100087   ← Ch02 plateau (40% drop → mecA detected)
```

---

## 6. AI/ML Architecture — In Depth

Implements all boxes in `ai_multiplex_biosensor_pipeline.svg`:

```
Real hardware OR synthetic CSV
  └── 12 channels × time-series (µA @ 1 Hz, 900 readings)
           OR  12 × 8 pre-extracted features
                    │
                    ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                   AI / ML Pipeline                           │
  │                                                              │
  │  [Layer 2]  Signal Processing                                │
  │    Adaptive Kalman filter → peak/drop/t2t/SNR/AUC/conc      │
  │                    │                                         │
  │                    ▼                                         │
  │  [Layer 3a] ChannelEncoder (shared MLP ≈ 1D-CNN)            │
  │    Input:   12 channels × 8 features                         │
  │    Weight:  SAME MLP applied to every channel                │
  │    Output:  12 × 32-dim embeddings                           │
  │                    │                                         │
  │                    ▼                                         │
  │  [Layer 3b] CrossChannelAttention (Transformer)              │
  │    Q = K = V = channel embeddings (12 × 32)                  │
  │    4 heads × 8 dimensions each                               │
  │    Residual + LayerNorm                                       │
  │    Output: 12 × 32 attended → 3 domain vectors               │
  │             + (12 × 12) attention weight matrix              │
  │                    │                                         │
  │                    ▼                                         │
  │  [Layer 3c] BayesianCalibration (MC Dropout)                 │
  │    50 stochastic passes, p_drop = 0.15                       │
  │    Confidence = clip(100 − σ × 2.5, 30, 99)                  │
  │    Output: score ± 95% CI per domain                         │
  │                    │                                         │
  │         ┌──────────┴────────────┐                            │
  │         ▼          ▼            ▼                            │
  │   [3d-1] AMR  [3d-2] Biofilm  [3d-3] Oncology               │
  │   XGBoost     Grad. Boosting  Random Forest                  │
  │   Multilabel  (LSTM-style     (GNN-style                     │
  │   gene calls  ordinal Stage)  tier + network)                │
  │         │          │            │                            │
  │         └──────────┴────────────┘                            │
  │                    │                                         │
  │                    ▼                                         │
  │  [Layer 3e] EnsembleRiskEngine                               │
  │    Score = 0.40×AMR + 0.35×Biofilm + 0.25×Oncology          │
  │             + co-occurrence bonus (0 / +8 / +15)             │
  │    Tier: Low / Moderate / High / Critical                    │
  │                    │                                         │
  │  [Layer 7] SHAP Explainability (per biomarker)               │
  └────────────────────┼─────────────────────────────────────────┘
                       │
                       ▼
          Edge Device Output (BLE / Wi-Fi)
                       │
                       ▼
          HTML Clinical Decision Report
           ┌──────────┬──────────┬──────────┐
           │ AMR Risk │ Biofilm  │ Oncology │
           │  Panel   │ Status   │  Risk    │
           └──────────┴──────────┴──────────┘
```

### Why These Model Choices?

| Head | Model | Rationale |
|------|-------|-----------|
| **AMR** | Multilabel XGBoost (GradientBoosting fallback) | 4 resistance genes are independent binary labels. XGBoost handles label imbalance well and is interpretable. |
| **Biofilm** | Gradient Boosting (ordinal 0–4) | Biofilm stages are ordered (I < II < III < IV). Kinetic feature `t2t_s` approximates LSTM temporal reasoning. |
| **Oncology** | Random Forest with cross-product features | Cancer microbe co-occurrence is non-linear. Explicit FadA×CagA, pks×miRNA terms simulate GNN co-occurrence edges without a graph database. |

### Cross-Channel Attention — Key Clinical Insight

- **AMR (Ch02 mecA) ↔ Biofilm (Ch08 c-di-GMP)**: MRSA in a biofilm matrix requires 1000× higher antibiotic dose — ensemble gets +15 co-occurrence bonus
- **Oncology (Ch09 FadA) ↔ (Ch10 CagA)**: *F. nucleatum* + *H. pylori* co-detection signals potential multi-site GI cancer risk
- The attention weights expose *which* cross-domain relationships the model found, making it auditable for regulatory submission

### MC Dropout Uncertainty — Clinical Necessity

```
AMR Score:  72%  [95% CI: 65–79%]   → High confidence → act on result
Oncology:   48%  [95% CI: 31–65%]   → Low confidence  → flag for lab confirmation
```

A wide CI instructs the clinician to hold empirical treatment and send the sample to a reference lab. This is a **regulatory requirement** for in-vitro diagnostic AI systems (EU IVDR 2017/746, FDA SaMD guidance).

---

## 7. Pipeline Code Explained — Module by Module

### `detailed_dummy_generator.py` — Synthetic Data Factory

Generates the 500-sample CSV with full physiological realism.

Key functions:
- `patient_metadata(i)` — demographics, device, sample type
- `assign_ground_truth(category)` — biologically coherent co-activation rules (AHL+c-di-GMP 70%, bap/psl+icaADBC 50%)
- `raw_signal(is_present, ch)` — Signal-OFF log model with 0.15 nA noise
- `kalman_smooth(raw)` — single-step Kalman
- `extract_features(...)` — peak, drop, t2t, impedance, SNR
- `mc_dropout_confidence(score, model_type)` — per-model uncertainty
- `compute_shap(gt, channels, score)` — SHAP-like attributions
- `clinical_report(gt, ...)` — Stage 9 clinical fields

Category distribution (stratified):
```
Healthy           20%  — no active domain
AMR_only          12%  — Ch01–04
Biofilm_only      12%  — Ch05–08
Oncology_only     12%  — Ch09–12
AMR_Biofilm       12%  — Ch01–08
AMR_Oncology      10%  — Ch01–04 + Ch09–12
Biofilm_Oncology  10%  — Ch05–12
Pan_positive      12%  — all 12 channels potentially active
```

---

### `pipeline/signal_processor.py` — Layer 2

**`KalmanFilter1D`** — Proper 1D Kalman (Q=0.01, R=0.0225). Operates in both sequence and single-step mode.

**`SignalProcessor.process(df)`** returns:
```python
{
  "amr_features":      DataFrame,  # Ch01-04 features
  "biofilm_features":  DataFrame,  # Ch05-08 + QS interaction term
  "oncology_features": DataFrame,  # Ch09-12 + co-detection cross-products
  "detection_flags":   DataFrame,  # binary per channel (drop >= 30%)
  "qs_proxy":          Series,     # continuous QS activity score
}
```

Interaction terms (top SHAP drivers):
- `Ch06_Ch08_QS_interaction = Ch06_drop_pct × Ch08_drop_pct`
- `Ch09_Ch10_codetect = Ch09_drop_pct × Ch10_drop_pct`
- `Ch11_Ch12_codetect = Ch11_drop_pct × Ch12_drop_pct`

---

### `pipeline/multimodal_encoder.py` — Layers 3a + 3b + 3c

**`ChannelEncoder`** — Shared-weight MLP applied independently to each channel's 8 features (equivalent to 1D-CNN with kernel_size=1). Trains on 12N rows (12 channels × N samples) with one unified weight set.

**`CrossChannelAttention`** — Multi-head scaled dot-product attention over the 12-channel embedding matrix. Returns (N, 12, 12) attention weights and 3 domain-pooled vectors.

**`BayesianCalibration`** — 50 stochastic passes with 15% feature dropout. Outputs `mean`, `std`, `conf%`, `lb`, `ub` per domain per sample.

**`MultimodalEncoder`** — Orchestrates all three. `get_augmented_features(df, sp_out)` concatenates raw features with encoder embeddings, giving task heads both local channel context and cross-channel attention context.

---

### `pipeline/models.py` — Layers 3d + 3e

**`AMRHead`** — `MultiOutputClassifier(XGBoost)` or `GradientBoosting` fallback. Per-gene: `predict()` → binary call, `predict_proba()` → probability.

**`BiofilmHead`** — Ordinal Gradient Boosting (0–4 stages). `AI_Biofilm_Score` = weighted sum of stage probabilities × stage index.

**`OncologyHead`** — Ordinal Random Forest (0–3 tiers). `GNN_Network_Centrality` = P(Very_High) × 100.

**`EnsembleRiskEngine`**:
```python
score = 0.40 × AI_AMR + 0.35 × AI_Biofilm + 0.25 × AI_Oncology + co_bonus
# co_bonus: 0 active→0, 1 active→0, 2 active→+8, 3 active→+15
tier  = { <25:"Low", <50:"Moderate", <75:"High", ≥75:"Critical" }
```

---

### `pipeline/clinical_report.py` — Layers 4 + 5

Generates a standalone HTML file (no external dependencies) per patient:

| Panel | Contents |
|---|---|
| **Header** | Sample ID, collection date/time, device ID, transmission mode |
| **Ensemble gauge** | SVG arc gauge 0–100, colour-coded by risk tier |
| **AMR panel** (blue) | Resistance profile, gene tags, failed antibiotics, recommended alternative, SHAP bars |
| **Biofilm panel** (purple) | Stage I–IV progress pips, metric cards (QS score, c-di-GMP, AI-2), pel/psl flag, SHAP bars |
| **Oncology panel** (red) | Risk tier badge, species tags, inflammation bar, referral alert, SHAP bars |
| **Footer** | Regulatory disclaimer |

---

### `pipeline/real_device_adapter.py` — Real Hardware Bridge (NEW)

Converts real potentiostat output (µA time-series) to the feature row format the AI pipeline expects.

**`UnitConverter`** — supports: `"uA"`, `"nA"`, `"pA"`, `"mV"`, `"uV"`, `"adc_12bit"`, `"adc_16bit"`

**`KalmanFilterTimeSeries`** — adaptive Kalman applied across all 900 readings:
- Low Q (0.005) during stable baseline
- High Q (0.05) during rapid binding phase (detected by dz > 0.05 nA)

**`extract_features_from_timeseries()`** — extracts all 8 pipeline features plus extras:
`baseline_nA`, `plateau_nA`, `peak_amplitude`, `drop_pct`, `t2t_s`, `SNR_dB`, `conc_pM`, `AUC_nA_s`, `kinetic_slope`

**`RealDeviceAdapter`** — top-level class:
```python
adapter     = RealDeviceAdapter(unit="uA", sample_rate_hz=1.0)
feature_row = adapter.process_file("device_output.csv", patient_meta={...})
# feature_row is directly compatible with run.py pipeline
```

---

### `run.py` — Batch Pipeline Orchestrator

Runs on the 500-sample training corpus:
```
load_data() → split(80/20) → run_signal_processing()
→ run_encoder() → run_task_heads() → run_inference()
→ run_ensemble() → evaluate() → save_results() → generate_reports()
```

**CLI**:
```bash
python run.py                    # 10 HTML reports  (~18s)
python run.py --n-reports 50     # 50 HTML reports
```

---

### `run2.py` — Live Device Simulation (NEW)

Simulates a real potentiostat streamed to the AI pipeline, step by step:

```
Phase 1: Device boot + cartridge calibration (ANSI boot sequence)
Phase 2: Baseline measurement — all 12 channels stable at ~10 nA
Phase 3: Sample injection + LIVE ACQUISITION
         → Generates 900 µA readings per channel (real binding physics)
         → Adaptive Kalman filter applied reading-by-reading
         → 12-channel live table updates every ~0.1s
         → Auto-fires detection events when drop ≥ 30%
Phase 4: Feature extraction — printed per channel (drop%, t2t, SNR, AUC, conc)
Phase 5: AI pipeline — encoder → attention → MC Dropout → heads → ensemble
Phase 6: HTML report generation + terminal clinical summary panel
```

**Three pre-set clinical scenarios**:

| `--scenario` | Patient | Positive channels | Clinical note |
|---|---|---|---|
| `1` (default) | 67M ICU Blood | Ch02 mecA + Ch05 icaADBC + Ch09 FadA | Post-surgical: AMR + Biofilm + GI Oncology |
| `2` | 42F ED Urine | Ch01 blaNDM-1 + Ch04 KPC | Carbapenem-resistant UTI |
| `3` | 55M Outpatient Wound | Ch06 AHL/AI-2 + Ch07 bap/pel/psl + Ch08 c-di-GMP | Chronic wound / Stage III biofilm |

**CLI**:
```bash
python run2.py                   # default scenario, full animation (~40s)
python run2.py --fast            # compressed delays (~20s)
python run2.py --scenario 2      # AMR-only scenario
python run2.py --scenario 3      # Biofilm-only scenario
```

---

## 8. Real Hardware Signal Processing

This section explains how to connect a **real potentiostat** to this pipeline.

### What a real device sends

```
Every second for 15 minutes (900 readings per channel):

time_s,  Ch01,      Ch02,      ...  Ch12
0,       0.010013,  0.009983,  ...  0.010031   ← baseline (~0.01 µA = 10 nA)
1,       0.010012,  0.009974,  ...  0.010029
60,      0.010009,  0.009871,  ...  0.010031   ← injection; Ch02 starting to drop
120,     0.010006,  0.008823,  ...  0.010028
180,     0.009998,  0.007102,  ...  0.010025
...
900,     0.009994,  0.006001,  ...  0.010019   ← Ch02 plateau: 40% drop → mecA DETECTED
```

### Signal processing chain

```
Potentiostat output (µA at 1 Hz)
        │
        ▼
[Unit Conversion]
  µA × 1000 = nA
  (or: V_µV / R_shunt × 1e9, or adc_count / 4095 × Vref / R × 1e9)
        │
        ▼
[Adaptive Kalman Filter] — per reading, over full 900-point series
  Q_low  = 0.005  (stable baseline / plateau)
  Q_high = 0.050  (rapid binding phase, detected by Δ > 0.05 nA)
  R      = 0.023  (= noise_std² = 0.15²)
        │
        ▼
[Feature Extraction per channel]
  baseline_mean_nA  — mean of first 60 s (pre-injection)
  plateau_mean_nA   — mean of last 60 s (steady state)
  peak_amplitude    — baseline − plateau (nA)
  signal_drop_%     — peak_amplitude / baseline × 100
  t2t_s             — time to cross 30% threshold
  SNR_dB            — 10 × log₁₀(signal² / noise²)
  conc_pM           — 10^((10 − plateau_nA) / 2.1)
  AUC_nA_s          — ∫(baseline − signal) dt  (binding curve area)
  kinetic_slope     — nA/s during binding phase
        │
        ▼
[Same feature row → run.py AI pipeline works unchanged]
```

### Supported hardware input formats

| `unit=` | Device type | Conversion |
|---|---|---|
| `"uA"` | PalmSens4, EmStat Pico | × 1000 = nA |
| `"nA"` | High-spec potentiostats | pass-through |
| `"pA"` | Ultra-sensitive patch-clamp | ÷ 1000 = nA |
| `"mV"` | Shunt resistor (voltage mode) | V / R_shunt × 1e9 |
| `"uV"` | High-resolution shunt readout | same |
| `"adc_12bit"` | STM32 / ESP32 12-bit ADC | (count/4095) × Vref / R × 1e9 |
| `"adc_16bit"` | ADS1115 / high-res ADC | (count/65535) × Vref / R × 1e9 |

### Quick start — real device CSV

```python
from pipeline.real_device_adapter import RealDeviceAdapter

adapter     = RealDeviceAdapter(unit="uA", sample_rate_hz=1.0)
feature_row = adapter.process_file(
    "your_device_output.csv",
    sample_id    = "PAT-2024-001",
    patient_meta = {
        "Patient_Age": 54, "Patient_Sex": "M", "Patient_BMI": 27.1,
        "Sample_Type": "Blood", "Collection_Site": "ICU",
    },
    device_meta  = {
        "Device_ID": "DEV-B42", "Cartridge_Lot": "LOT-2024-001",
        "Ambient_Temp_C": 24.5, "Ambient_pH": 7.32,
    }
)
print(adapter.signal_quality_report(feature_row))
# feature_row is now compatible with run.py inference
```

### Test the adapter with simulated hardware

```bash
python -m pipeline.real_device_adapter
```

This generates `sample_device_input.csv` (900 readings × 12 channels), processes it through Kalman filter + feature extraction, and prints the signal quality report.

---

## 9. How to Run the System

### Environment setup

```bash
# Activate virtual environment (already created)
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
source .venv/bin/activate              # Linux / macOS

# Dependencies
pip install numpy pandas scikit-learn xgboost shap
```

---

### Step 1 — Generate synthetic data (once only)

```bash
python detailed_dummy_generator.py
```

Creates:
- `biosensor_detailed_500.csv` (165 columns, 500 rows)
- `biosensor_ground_truth.csv`
- `biosensor_shap.csv`
- `biosensor_metadata.json`

> **Skip** if these files already exist.

---

### Step 2A — Batch pipeline (10 HTML reports)

```bash
python run.py
```

Runtime: ~18 seconds. Output: `biosensor_full_results.csv` + `reports/report_BIO****.html`

```bash
python run.py --n-reports 50    # generate 50 reports instead
```

---

### Step 2B — Live simulation (real-time µA streaming)

```bash
python run2.py                  # ICU patient, 3 positive domains (~40s)
python run2.py --fast           # same but compressed delays (~20s)
python run2.py --scenario 2     # AMR-only, carbapenem-resistant UTI
python run2.py --scenario 3     # Biofilm-only, chronic wound
```

What you see live:
```
  LIVE ACQUISITION  [████████████░░░░░░░░░░░░░░░░░░░░] 38.3%  345/900s  BINDING
  Ch01  blaNDM-1    0.009895▼ µA    9.938 nA    0.6%  ███████████████████░  stable
  Ch02  mecA        0.006495▲ µA    6.421 nA   35.8%  ████████████░░░░░░░░  DETECTED
  Ch05  icaADBC     0.005818▲ µA    5.772 nA   42.3%  ███████████░░░░░░░░░  DETECTED
  Ch09  FadA        0.005951▲ µA    5.903 nA   41.0%  ███████████░░░░░░░░░  DETECTED
```

---

### Step 3 — View clinical reports

```bash
start reports\report_BIO0004.html         # Windows — batch report
start reports\report_REAL-PAT-001.html    # Windows — live simulation report
```

---

### Step 4 — Advanced: connect real hardware

```bash
python -m pipeline.real_device_adapter    # test with simulated device CSV
```

Then adapt `run2.py` to call `adapter.process_file(your_real_csv)` and pipe to the AI inference steps.

---

## 10. Output Files and How to Interpret Them

### `biosensor_full_results.csv`

100 rows (20% test split). Key columns:

| Column | Meaning |
|---|---|
| `Ensemble_Risk_Score` | 0–100 final clinical risk |
| `Ensemble_Risk_Tier` | Low / Moderate / High / Critical |
| `Biofilm_Stage_Pred` | Predicted vs `Biofilm_Stage_True` |
| `MC_AMR_Conf_pct` | Model confidence in AMR call |
| `Co_occurrence_Bonus` | +0 / +8 / +15 depending on domain overlap |

**Ensemble risk tiers**:

| Score | Tier | Suggested action |
|-------|------|-----------------|
| 0–24 | Low | Routine monitoring |
| 25–49 | Moderate | Targeted treatment, watch for progression |
| 50–74 | High | Aggressive treatment; ID / Oncology consult |
| 75–100 | Critical | Immediate escalation; multidisciplinary review |

### `reports/report_*.html`

Self-contained HTML — open in any browser. Contents:
- Patient details + device telemetry
- SVG arc gauge (ensemble risk)
- AMR panel: genes, failed drugs, recommended alternatives, SHAP bars
- Biofilm panel: stage pips, QS score, c-di-GMP, AI-2, matrix gene flags
- Oncology panel: risk tier, implicated species, inflammation load, referral alert

### Terminal output (run2.py)

```
  ╠════════════════╦════════════╦══════════════════════════╣
  ║ AMR Risk       ║  99.2%     ║ Single_resistance         ║
  ║                ║            ║ Genes: mecA                ║
  ║                ║            ║ → Vancomycin / Linezolid  ║
  ╠════════════════╬════════════╬══════════════════════════╣
  ║ Biofilm        ║  49.9%     ║ Stage II microcolony      ║
  ╠════════════════╬════════════╬══════════════════════════╣
  ║ Oncology       ║  48.2%     ║ High — F. nucleatum       ║
  ╠════════════════╬════════════╬══════════════════════════╣
  ║ ENSEMBLE RISK  ║  84.2%     ║ Critical                  ║
  ╚════════════════╩════════════╩══════════════════════════╝
```

---

## 11. Autodocking Sub-Pipeline

Located in `autodocking/`. A **separate research component** for anti-cancer peptide (ACP) design, complementing the oncology detection channels:

1. **Generate 3D structure** from amino acid sequence (backbone torsion model)
2. **Prepare PDBQT files** (OpenBabel + Meeko for AutoDock Vina)
3. **Molecular docking** with AutoDock Vina → binding affinity score (kcal/mol)
4. **ACP activity classifier**: k-mer features → XGBoost (pre-trained: `breast_model.joblib`)
5. **Predict** activity of new peptide sequences against breast cancer cells

```bash
cd autodocking
conda env create -f environment.yml
conda activate biohackathon-vina
autodock-pipeline predict-model \
    --model breast_model.joblib \
    --sequence AIGKFLHSAKKFGKAFVGEIMNS
```

---

## 12. Key Design Decisions

**Why tabular features instead of raw time-series for the batch pipeline?**  
`run.py` uses pre-extracted 8-feature vectors per channel — the same representation a real 1D-CNN would produce from raw waveforms. This avoids a PyTorch dependency while remaining mathematically equivalent for the classification task. `run2.py` shows the full time-series path using the real_device_adapter.

**Why Gradient Boosting instead of true LSTM?**  
Pure LSTM requires sequential steps `[I(t=0)…I(t=900)]`. For tabular data, Gradient Boosting with kinetic features (`t2t_s`, `kinetic_slope`) approximates the temporal reasoning. Real deployment on raw potentiostat streams would use a PyTorch LSTM.

**Why Random Forest instead of a graph neural network?**  
A real GNN needs a graph database and PyTorch Geometric. The RF with explicit interaction terms (`Ch09×Ch10`, `Ch11×Ch12`) simulates the co-occurrence graph edges and is deployable with only scikit-learn.

**Why the +8 / +15 co-occurrence bonus?**  
Clinical evidence: AMR + Biofilm is far harder to treat (biofilm reduces antibiotic penetration by 100–1000×). Pan-positive (all 3 domains) represents the highest-acuity clinical scenario and deserves a non-linear risk boost.

**Adaptive Kalman Q in the real device adapter**  
During the baseline phase, signal is flat → low process noise (Q=0.005) correctly filters out measurement noise. During binding, signal drops steeply → high process noise (Q=0.05) allows the filter to track the real signal without lag-induced false confidence.

---

## 13. Model Performance Summary

Evaluated on 100 held-out test samples (20% stratified split), repeated across multiple runs:

### AMR Head (Gene-level multilabel)
| Gene | Biomarker | AUC | Precision | Recall |
|------|-----------|-----|-----------|--------|
| blaNDM-1 | NDM carbapenemase | 1.000 | 1.00 | 1.00 |
| mecA | MRSA resistance | 1.000 | 1.00 | 1.00 |
| vanA | Vancomycin resistance | 0.963–1.000 | 1.00 | 0.93–1.00 |
| KPC | KPC carbapenemase | 0.981–1.000 | 1.00 | 0.96–1.00 |

### Biofilm Head (Ordinal stage 0–4)
| Metric | Value |
|--------|-------|
| Exact accuracy | 95–99% |
| Quadratic weighted kappa | 0.991–0.998 |
| Within-1-stage accuracy | 100% |

### Oncology Head (Ordinal tier 0–3)
| Metric | Value |
|--------|-------|
| Exact accuracy | 92–95% |
| Quadratic weighted kappa | 0.979–0.987 |
| Within-1-tier accuracy | 100% |

### Ensemble Risk Engine
| Metric | Value |
|--------|-------|
| Tier distribution (100 test) | Low≈30 / Moderate≈27 / High≈25 / Critical≈18 |
| Total pipeline runtime | ~18s (batch) / ~40s (live simulation) |

> **Important**: Performance is high because signal physics are deterministic (synthetic data). Real-world validation on clinical samples with lab-confirmed ground truth would be required before clinical deployment. Expect lower initial accuracy; plan for continuous learning as confirmations accumulate.

---

## 14. Extending the System

### Add a new biomarker channel
1. Add channel to `CHANNEL_MAP` in `detailed_dummy_generator.py`
2. Add signal physics in `raw_signal()`
3. Update `DOMAIN_CH_IDX` in `multimodal_encoder.py`
4. Re-run generator → `run.py`

### Connect real hardware
1. Export potentiostat data as CSV (`time_s, Ch01, Ch02, ..., Ch12`)
2. Run adapter: `adapter.process_file("your_data.csv")` → feature row
3. Pass feature row to the AI pipeline in `run.py` or `run2.py`

### New patient scenario in run2.py
Add an entry to `SCENARIOS` dict in `run2.py`:
```python
SCENARIOS[4] = {
    "name":     "Surgery Ward — Vancomycin-resistant Enterococcus",
    "positive": {"Ch03": 210.5},   # vanA at 210.5 pM
    ...
}
```
Then: `python run2.py --scenario 4`

### Online learning (Stage 10)
When lab confirmations arrive:
1. Append confirmed labels to `biosensor_ground_truth.csv`
2. Re-run `python run.py` — models retrain automatically
3. For production: use XGBoost `xgb_model` warm-start or sklearn `warm_start=True`

### Edge hardware deployment
All inference modules use only numpy + scikit-learn — no GPU needed:
```python
import joblib
joblib.dump(amr_head.model, "amr_head.joblib")   # serialize
# On edge device:
m = joblib.load("amr_head.joblib")
m.predict(X_new)
```

---

## 15. Dependencies

```
numpy          >= 1.24    Core numerical computation (also numpy 2.x compatible)
pandas         >= 2.0     Data I/O and manipulation
scikit-learn   >= 1.3     ML models, preprocessing, evaluation
xgboost        >= 1.7     AMR head  (optional — GradientBoosting fallback if absent)
shap           >= 0.43    SHAP explainability  (optional)
```

**Install all**:
```bash
pip install numpy pandas scikit-learn xgboost shap
```

**Activate virtual environment**:
```bash
.\.venv\Scripts\Activate.ps1     # Windows PowerShell
source .venv/bin/activate         # Linux / macOS
```

No GPU required. No internet connection required at inference time.

---

## 16. Web Simulator Dashboard

A recently added Flask-based web dashboard (`app.py` & `web_simulator.py`) provides an interactive GUI for exploring the biosensor's clinical reports:
- Serves generated HTML clinical decision reports over a local network.
- Features a premium, dynamic dark-mode interface (`static/style.css` & `templates/index.html`).
- Allows clinicians to visually navigate between recent patient simulations, search by sample ID, and view real-time diagnostics securely.

**To run the web dashboard:**
```bash
python app.py
```
Then navigate to `http://127.0.0.1:5000` in any web browser.

---

## 17. License

Copyright (c) 2026. All Rights Reserved.

This project and its contents are not licensed for public use, modification, distribution, or reproduction without explicit prior written permission. See the `LICENSE` file for details.

---

## File Change Log

| Version | File | Change |
|---------|------|--------|
| v1 | `data_generater.py` | Original 13-column simple generator |
| v2 | `detailed_dummy_generator.py` | Full 165-column generator aligned to SVG; biofilm biomarkers (pel/psl/AI-2); QS activity; MC Dropout; SHAP |
| v3 | `pipeline/signal_processor.py` | Kalman filter + interaction feature extraction |
| v3 | `pipeline/multimodal_encoder.py` | 1D-CNN encoder + Transformer attention + Bayesian calibration |
| v3 | `pipeline/models.py` | XGBoost AMR / GB Biofilm / RF Oncology + Ensemble engine |
| v3 | `pipeline/clinical_report.py` | 3-panel HTML clinical report with SVG gauge + SHAP bars |
| v3 | `run.py` | Full batch orchestrator (Layers 2→5) |
| v4 | `pipeline/real_device_adapter.py` | **NEW** — real µA/ADC time-series → feature row converter with adaptive Kalman |
| v4 | `run2.py` | **NEW** — live device simulation: streams µA per reading, live ANSI table, auto-detection events, 3 clinical scenarios |
| v4 | `README.md` | This document |

---

## Authors & Context

Designed for the **Biohackathon** as a proof-of-concept for AI-enabled point-of-care multiplex diagnostics. The full AI/ML architecture is in `ai_multiplex_biosensor_pipeline.svg`. The step-by-step biology is in `biosensor_system_explainer.txt`.

The `autodocking/` sub-pipeline is a companion component for anti-cancer peptide design, targeting the same organisms detected in the oncology channels.

---

*For questions: refer to `biosensor_system_explainer.txt` or the SVG architecture diagram.*
