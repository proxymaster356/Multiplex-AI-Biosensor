"""
Detailed Dummy Data Generator -- AI-Enabled Multiplex Biosensor System
=======================================================================
Version 3.0  --  SVG-aligned update
Changes from v2.0:
  - Biofilm channel map updated to match SVG exactly:
      Ch06 = AHL / AI-2 combined quorum signal (was: AHL only)
      Ch07 = bap / pel / psl biofilm matrix genes (was: bap only)
    + Added derived sub-fields: AI2_Level_uM, pel_Active, psl_Active
  - Classification head model types now simulated per SVG spec:
      AMR      --> Multilabel XGBoost   (precise at extremes, hard cutoffs)
      Biofilm  --> LSTM temporal model  (smooth progression, stage-coherent)
      Oncology --> GNN microbial network (co-occurrence amplification)
  - New clinical fields from SVG Stage 9:
      QS_Activity_Score (0-100 continuous, combines AHL + AI-2 signals)
      AI2_Level_uM      (second messenger level)
      pel_psl_Active    (binary matrix gene flag)
  - Ensemble risk engine updated: adds co-occurrence bonus for
    multi-domain positives (matches SVG "AMR x biofilm x oncology scoring")
  - SHAP keys cleaned to match biomarker short names

Output files
------------
  biosensor_detailed_500.csv   -- main wide table (~140 columns)
  biosensor_ground_truth.csv   -- binary biomarker presence per channel
  biosensor_shap.csv           -- per-biomarker SHAP contribution matrix
  biosensor_metadata.json      -- full column schema / data dictionary
"""

import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

N          = 500
START_DATE = datetime(2024, 1, 1)

# ── Channel map  (SVG-aligned) ───────────────────────────────────────────────
# 12 electrochemical channels; biofilm domain now reflects full SVG biomarker list.
# Ch06 covers both AHLs and AI-2 (same aptamer/DNAzyme probe group on channel 6).
# Ch07 covers bap, pel, psl (biofilm matrix / exopolysaccharide genes).
CHANNEL_MAP = {
    # --- AMR domain (CRISPR-Cas12a probes) ---
    "Ch01": {"domain": "AMR",      "biomarker": "blaNDM-1",      "gene": True,
             "probe": "CRISPR-Cas12a",  "species": "Klebsiella / E. coli"},
    "Ch02": {"domain": "AMR",      "biomarker": "mecA",           "gene": True,
             "probe": "CRISPR-Cas12a",  "species": "S. aureus (MRSA)"},
    "Ch03": {"domain": "AMR",      "biomarker": "vanA",           "gene": True,
             "probe": "CRISPR-Cas12a",  "species": "Enterococcus (VRE)"},
    "Ch04": {"domain": "AMR",      "biomarker": "KPC",            "gene": True,
             "probe": "CRISPR-Cas12a",  "species": "Klebsiella pneumoniae"},
    # --- Biofilm domain (Aptamer / DNAzyme probes) ---
    "Ch05": {"domain": "Biofilm",  "biomarker": "icaADBC",        "gene": True,
             "probe": "Aptamer",        "species": "S. aureus / S. epidermidis"},
    "Ch06": {"domain": "Biofilm",  "biomarker": "AHL_AI2",        "gene": False,
             "probe": "DNAzyme",        "species": "Quorum sensing (pan-species)"},
    "Ch07": {"domain": "Biofilm",  "biomarker": "bap_pel_psl",    "gene": True,
             "probe": "Aptamer",        "species": "P. aeruginosa / S. aureus"},
    "Ch08": {"domain": "Biofilm",  "biomarker": "c-diGMP",        "gene": False,
             "probe": "DNAzyme",        "species": "Universal 2nd messenger"},
    # --- Oncology domain (Antibody probes) ---
    "Ch09": {"domain": "Oncology", "biomarker": "FadA",           "gene": True,
             "probe": "Antibody",       "species": "F. nucleatum (colorectal)"},
    "Ch10": {"domain": "Oncology", "biomarker": "CagA",           "gene": True,
             "probe": "Antibody",       "species": "H. pylori (gastric)"},
    "Ch11": {"domain": "Oncology", "biomarker": "pks",            "gene": True,
             "probe": "Antibody",       "species": "E. coli (colorectal)"},
    "Ch12": {"domain": "Oncology", "biomarker": "miRNA-21",       "gene": False,
             "probe": "Aptamer",        "species": "Pan-cancer epigenetic marker"},
}

DOMAINS           = ["AMR", "Biofilm", "Oncology"]
AMR_CHANNELS      = [c for c, v in CHANNEL_MAP.items() if v["domain"] == "AMR"]
BIOFILM_CHANNELS  = [c for c, v in CHANNEL_MAP.items() if v["domain"] == "Biofilm"]
ONCOLOGY_CHANNELS = [c for c, v in CHANNEL_MAP.items() if v["domain"] == "Oncology"]
DOMAIN_CHANNELS   = {"AMR": AMR_CHANNELS,
                      "Biofilm": BIOFILM_CHANNELS,
                      "Oncology": ONCOLOGY_CHANNELS}

# ── Clinical categories & priors ─────────────────────────────────────────────
CATEGORIES      = ["Healthy", "AMR_only", "Biofilm_only", "Oncology_only",
                    "AMR_Biofilm", "AMR_Oncology", "Biofilm_Oncology", "Pan_positive"]
CATEGORY_PROBS  = [0.20, 0.12, 0.12, 0.12, 0.12, 0.10, 0.10, 0.12]

SAMPLE_TYPES    = ["Blood", "Urine", "Wound_swab", "Biopsy_fluid"]
SAMPLE_PROBS    = [0.40,   0.25,    0.20,          0.15]

COLLECTION_SITES = ["ICU", "ED", "Outpatient_clinic", "Surgery_ward", "Community"]

# ── Antibiotic lookup ─────────────────────────────────────────────────────────
ANTIBIOTICS = {
    "blaNDM-1": ["Carbapenems", "Penicillins", "Cephalosporins"],
    "mecA":     ["Methicillin", "Oxacillin", "Flucloxacillin"],
    "vanA":     ["Vancomycin", "Teicoplanin"],
    "KPC":      ["Carbapenems", "Aztreonam"],
}
ANTIBIOTIC_ALTERNATIVES = {
    "blaNDM-1": "Colistin / Tigecycline",
    "mecA":     "Vancomycin / Linezolid",
    "vanA":     "Daptomycin / Linezolid",
    "KPC":      "Ceftazidime-avibactam",
}

BIOFILM_STAGES      = ["Stage_I_early_attachment",
                        "Stage_II_microcolony",
                        "Stage_III_maturation",
                        "Stage_IV_dispersion"]
ONCOLOGY_RISK_TIERS = ["Low", "Moderate", "High", "Very_High"]


# =============================================================================
# Stage 1 -- Patient metadata
# =============================================================================
def patient_metadata(i: int) -> dict:
    """Demographics, sample logistics, and device info."""
    dt = START_DATE + timedelta(
        days=random.randint(0, 364),
        hours=random.randint(6, 22),
        minutes=random.randint(0, 59)
    )
    age = int(np.clip(np.random.normal(52, 18), 1, 95))
    bmi = round(np.clip(np.random.normal(26.5, 5.0), 15.0, 50.0), 1)
    return {
        "Sample_ID":        f"BIO{i:04d}",
        "Collection_Date":  dt.strftime("%Y-%m-%d"),
        "Collection_Time":  dt.strftime("%H:%M"),
        "Patient_Age":      age,
        "Patient_Sex":      random.choice(["M", "F"]),
        "Patient_BMI":      bmi,
        "Sample_Type":      np.random.choice(SAMPLE_TYPES, p=SAMPLE_PROBS),
        "Collection_Site":  random.choice(COLLECTION_SITES),
        "Sample_Volume_uL": round(np.random.uniform(4.5, 6.0), 2),
        "Cartridge_Lot":    f"LOT-{random.randint(1000, 9999)}",
        "Device_ID":        f"DEV-{random.choice(['A','B','C'])}{random.randint(10,99)}",
        "Ambient_Temp_C":   round(np.random.uniform(18.0, 30.0), 1),
        "Ambient_pH":       round(np.random.uniform(6.8, 7.6), 2),
    }


# =============================================================================
# Stage 2 -- Ground truth biomarker assignment
# =============================================================================
def assign_ground_truth(category: str) -> dict:
    """Binary presence/absence per channel, biologically coherent."""
    gt = {ch: False for ch in CHANNEL_MAP}

    active_domains = []
    if "AMR"      in category: active_domains.append("AMR")
    if "Biofilm"  in category: active_domains.append("Biofilm")
    if "Oncology" in category: active_domains.append("Oncology")
    if "Pan"      in category: active_domains = list(DOMAINS)

    for domain in active_domains:
        chs = DOMAIN_CHANNELS[domain]
        n_active = np.random.randint(1, len(chs) + 1)
        for ch in random.sample(chs, n_active):
            gt[ch] = True

    # Biological constraint: AHL/AI-2 (Ch06) tends to co-activate with c-di-GMP (Ch08)
    if gt["Ch06"] and np.random.rand() < 0.70:
        gt["Ch08"] = True
    # bap/pel/psl (Ch07) co-activates with icaADBC (Ch05) at ~50%
    if gt["Ch07"] and np.random.rand() < 0.50:
        gt["Ch05"] = True

    # Healthy: rare stray positives (instrument noise)
    if category == "Healthy":
        for ch in CHANNEL_MAP:
            gt[ch] = (np.random.rand() < 0.02)

    return gt


# =============================================================================
# Stage 3 -- Raw electrochemical signal
# =============================================================================
def raw_signal(is_present: bool, ch_name: str) -> tuple:
    """
    Signal-OFF logarithmic decay model.
    Baseline ~10 nA. Target binding drops signal toward ~0.5 nA.
    Signaling molecules (AHL_AI2, c-diGMP, miRNA-21) use wider concentration range.
    Returns (raw_nA, concentration_pM).
    """
    is_molecule = not CHANNEL_MAP[ch_name]["gene"]
    noise       = np.random.normal(0, 0.15)

    if is_present:
        if is_molecule:
            conc_pM = np.random.uniform(5, 1000)    # broader range for small molecules
        else:
            conc_pM = np.random.uniform(10, 500)
        raw_nA = 10.0 - 2.1 * np.log10(conc_pM) + noise
    else:
        conc_pM = 0.0
        raw_nA  = 10.0 + noise

    return round(float(np.clip(raw_nA, 0.3, 11.0)), 3), round(conc_pM, 2)


# =============================================================================
# Stage 4 -- Kalman filter + feature extraction
# =============================================================================
def kalman_smooth(raw: float) -> float:
    """Single-step Kalman smoother (measurement noise ~0.15 nA)."""
    process_var = 0.01
    measure_var = 0.15 ** 2
    K           = process_var / (process_var + measure_var)
    smooth      = raw - 0.55 * (raw - (raw * (1 - K) + raw * K)) + np.random.normal(0, 0.04)
    return round(float(np.clip(smooth, 0.3, 11.0)), 3)


def extract_features(raw_nA: float, smooth_nA: float, is_present: bool) -> dict:
    """
    Feature vector per channel:
      peak_amplitude_nA   : absolute drop from 10 nA baseline
      signal_drop_pct     : % drop from baseline
      time_to_threshold_s : seconds to cross 30% drop threshold
      impedance_change_pct: % change in surface impedance
      snr_db              : signal-to-noise ratio in dB
    """
    baseline       = 10.0
    peak_amp       = round(baseline - smooth_nA, 3)
    signal_drop    = round(peak_amp / baseline * 100, 2)

    t2t = round(np.random.uniform(30,  180), 1) if is_present \
     else round(np.random.uniform(800, 1200), 1)

    imp_change = round(float(np.clip(
        peak_amp * np.random.uniform(1.8, 2.5) + np.random.normal(0, 0.3),
        0.0, 20.0)), 3)

    # SNR: positive signal well above noise floor gives high SNR
    signal_power = peak_amp ** 2
    noise_power  = 0.15 ** 2
    snr_db = round(10 * np.log10(max(signal_power, 1e-6) / noise_power), 2)

    return {
        "peak_amplitude_nA":    peak_amp,
        "signal_drop_pct":      signal_drop,
        "time_to_threshold_s":  t2t,
        "impedance_change_pct": imp_change,
        "snr_db":               snr_db,
    }


# =============================================================================
# Stage 5 -- AI model outputs  (model-type specific per SVG)
# =============================================================================
def xgboost_amr_score(gt: dict) -> dict:
    """
    Multilabel XGBoost head for AMR.
    Outputs: overall AMR probability + per-gene binary calls.
    XGBoost characteristic: sharp at extremes, low uncertainty in the middle.
    """
    n_active = sum(gt[ch] for ch in AMR_CHANNELS)
    base     = n_active / len(AMR_CHANNELS)
    # XGBoost sharpening: push scores toward 0/1
    sharpened = float(np.clip(base + np.random.normal(0, 0.04), 0, 1))
    score     = round(sharpened * 100, 1)

    # Per-gene binary call (with small false-call rate ~3%)
    gene_calls = {}
    for ch in AMR_CHANNELS:
        bmk = CHANNEL_MAP[ch]["biomarker"].replace("-", "").replace("/", "")
        true_pos  = gt[ch]
        false_call = (np.random.rand() < 0.03)
        gene_calls[f"XGB_{bmk}_call"] = int(true_pos or false_call)

    return {"AI_AMR_Score": score, **gene_calls}


def lstm_biofilm_score(gt: dict) -> dict:
    """
    LSTM temporal model head for Biofilm.
    Outputs: biofilm probability + temporal stage coherence score.
    LSTM characteristic: smooth progression, stage-coherent outputs.
    """
    n_active = sum(gt[ch] for ch in BIOFILM_CHANNELS)
    base     = n_active / len(BIOFILM_CHANNELS)

    # LSTM smoothing adds temporal noise (smaller std than XGB)
    score = float(np.clip(base * 100 + np.random.normal(0, 3.5), 0, 100))
    score = round(score, 1)

    # Temporal coherence score: how stable is the signal over the measurement window
    # (simulated as inversely proportional to noise)
    temporal_coherence = round(float(np.clip(
        100 - abs(np.random.normal(0, 8)) - (4 - n_active) * 5,
        20, 99)), 1)

    return {
        "AI_Biofilm_Score":          score,
        "LSTM_Temporal_Coherence":   temporal_coherence,
    }


def gnn_oncology_score(gt: dict) -> dict:
    """
    Graph Neural Network head for Oncology.
    Models microbial co-occurrence network; multi-species co-detection
    amplifies risk score (non-linear interaction).
    """
    n_active    = sum(gt[ch] for ch in ONCOLOGY_CHANNELS)
    base        = n_active / len(ONCOLOGY_CHANNELS)
    # GNN co-occurrence bonus: 2+ species present amplify score
    co_bonus    = 0.15 if n_active >= 2 else 0.0
    score       = float(np.clip(
        (base + co_bonus) * 100 + np.random.normal(0, 4.5),
        0, 100))
    score       = round(score, 1)

    # Network centrality score: how connected the detected microbes are
    # in the known microbial co-occurrence network
    species_present = [CHANNEL_MAP[ch]["species"].split("(")[0].strip()
                       for ch in ONCOLOGY_CHANNELS if gt[ch]]
    network_centrality = round(float(np.clip(
        n_active * 28 + np.random.normal(0, 8),
        0, 100)), 1)

    return {
        "AI_Oncology_Score":       score,
        "GNN_Network_Centrality":  network_centrality,
        "GNN_Species_Co_detected": "|".join(species_present) if species_present else "None",
    }


# =============================================================================
# Stage 6 -- Uncertainty quantification (MC Dropout)
# =============================================================================
def mc_dropout_confidence(score: float, model_type: str = "generic") -> dict:
    """
    Simulates 50 forward passes with dropout enabled.
    Variability peaks near decision boundary (50%).
    XGBoost head naturally less uncertain at extremes.
    """
    boundary_proximity = 1.0 - abs(score - 50) / 50.0

    if model_type == "xgboost":
        # XGB: sharper, less uncertain
        base_std = 1.5 + 5.0 * boundary_proximity
    elif model_type == "lstm":
        # LSTM: moderate uncertainty, influenced by temporal coherence
        base_std = 2.0 + 7.0 * boundary_proximity
    elif model_type == "gnn":
        # GNN: network effects add slight extra uncertainty
        base_std = 2.5 + 8.0 * boundary_proximity
    else:
        base_std = 2.0 + 8.0 * boundary_proximity

    std_dev        = round(float(np.clip(base_std + np.random.uniform(-1, 1), 0.5, 10)), 2)
    confidence_pct = float(np.clip(round(100 - std_dev * 2.5, 1), 30, 99))
    return {"mc_std_dev": std_dev, "confidence_pct": confidence_pct}


# =============================================================================
# Stage 7 -- SHAP explainability
# =============================================================================
def compute_shap(gt: dict, channels: list, domain_score: float) -> dict:
    """
    Shapley-value attribution. Active biomarkers get positive contribution;
    inactive get small negative (suppressive) values.
    Contributions are normalised so they sum ~to the domain score.
    """
    weights = []
    for ch in channels:
        w = np.random.uniform(0.15, 0.40) if gt[ch] else np.random.uniform(-0.05, 0.05)
        weights.append(w)

    total = sum(abs(w) for w in weights) or 1.0
    weights = [w / total for w in weights]

    return {
        f"SHAP_{CHANNEL_MAP[ch]['biomarker'].replace('-','').replace('/','_')}":
        round(w * domain_score, 2)
        for ch, w in zip(channels, weights)
    }


# =============================================================================
# Stage 8 -- Device output & ensemble risk
# =============================================================================
def device_output(score_amr: float, score_bio: float, score_onc: float,
                  gt: dict) -> dict:
    """
    Integrated clinical risk engine (SVG 3e):
    Ensemble = weighted sum + co-occurrence bonus (multi-domain positives
    are clinically more dangerous, e.g. biofilm-embedded MRSA with oncology co-signal).
    """
    # Count how many domains are active (score > 40)
    n_active_domains = sum([
        score_amr > 40,
        score_bio  > 40,
        score_onc  > 40,
    ])

    # Co-occurrence bonus: up to +15 for pan-positive
    co_bonus        = {0: 0, 1: 0, 2: 8, 3: 15}[n_active_domains]
    ensemble_raw    = 0.40 * score_amr + 0.35 * score_bio + 0.25 * score_onc
    ensemble_risk   = round(float(np.clip(ensemble_raw + co_bonus, 0, 100)), 1)
    risk_tier_label = ("Critical" if ensemble_risk >= 75 else
                       "High"     if ensemble_risk >= 50 else
                       "Moderate" if ensemble_risk >= 25 else "Low")

    return {
        "Transmission_Mode":        random.choice(["BLE", "WiFi", "Offline"]),
        "Total_Time_min":           round(np.random.uniform(22, 32), 1),
        "Ensemble_Risk_Score":      ensemble_risk,
        "Ensemble_Risk_Tier":       risk_tier_label,
        "N_Active_Domains":         n_active_domains,
        "Co_occurrence_Bonus":      co_bonus,
    }


# =============================================================================
# Stage 9 -- Clinical report
# =============================================================================
def clinical_report(gt: dict, score_bio: float, score_onc: float,
                    channel_signals: dict) -> dict:
    """
    Three-panel clinical output (SVG Layer 5).
    Now includes: per-gene AMR calls, QS_Activity_Score, AI2_Level_uM,
    pel_psl_Active, Biofilm_Stage (stage-coherent), referral recommendation.
    """
    # ── Panel 1: AMR ──────────────────────────────────────────────────────────
    detected_genes     = [CHANNEL_MAP[ch]["biomarker"]
                          for ch in AMR_CHANNELS if gt[ch]]
    failed_antibitics  = list(set(
        ab for gene in detected_genes for ab in ANTIBIOTICS.get(gene, [])))
    recommended_alts   = list(set(
        ANTIBIOTIC_ALTERNATIVES[gene] for gene in detected_genes
        if gene in ANTIBIOTIC_ALTERNATIVES))

    n_amr_genes = len(detected_genes)
    amr_risk_label = ("Pan-resistant" if n_amr_genes >= 3 else
                      "Multi-drug_resistant" if n_amr_genes == 2 else
                      "Single_resistance"    if n_amr_genes == 1 else "Susceptible")

    # ── Panel 2: Biofilm ─────────────────────────────────────────────────────
    n_bf_active = sum(gt[ch] for ch in BIOFILM_CHANNELS)
    bf_stage    = (["None"] + BIOFILM_STAGES)[min(n_bf_active, 4)]

    # QS activity score: continuous 0–100, derived from Ch06 (AHL/AI-2) signal drop
    # and Ch08 (c-di-GMP) concentration
    ch06_drop = channel_signals.get("Ch06_drop_pct", 0)  # present = high drop
    ch08_conc = channel_signals.get("Ch08_conc_pM",  0)
    qs_activity = round(float(np.clip(
        ch06_drop * 0.6 + (ch08_conc / 10) * 0.4 + np.random.normal(0, 3),
        0, 100)), 1)

    ai2_level_uM = round(np.random.uniform(0.5, 8.0), 3) if gt["Ch06"] \
                   else round(np.random.uniform(0, 0.3), 3)
    cdgmp_level  = round(np.random.uniform(0.5, 5.0), 3) if gt["Ch08"] \
                   else round(np.random.uniform(0, 0.4), 3)
    pel_psl_active = int(gt["Ch07"])   # Ch07 = bap/pel/psl matrix genes
    ica_active     = int(gt["Ch05"])   # Ch05 = icaADBC

    # Quorum sensing active flag (binary)
    qs_active = int(gt["Ch06"])

    # ── Panel 3: Oncology risk ───────────────────────────────────────────────
    onc_tier    = ("Very_High" if score_onc >= 75 else
                   "High"      if score_onc >= 50 else
                   "Moderate"  if score_onc >= 25 else "Low")
    referral    = int(score_onc >= 50)

    onc_microbes = [CHANNEL_MAP[ch]["biomarker"]
                    for ch in ONCOLOGY_CHANNELS if gt[ch]]
    onc_species  = [CHANNEL_MAP[ch]["species"].split("(")[0].strip()
                    for ch in ONCOLOGY_CHANNELS if gt[ch]]
    inflam_load  = round(np.random.uniform(2, 10), 2) if onc_microbes \
                   else round(np.random.uniform(0, 2), 2)

    return {
        # -- AMR panel --
        "AMR_Resistance_Profile":    amr_risk_label,
        "AMR_Detected_Genes":        "|".join(detected_genes)    or "None",
        "AMR_Failed_Antibiotics":    "|".join(failed_antibitics) or "None",
        "AMR_Recommended_Alt":       "|".join(recommended_alts)  or "None",
        "AMR_N_Genes_Detected":      n_amr_genes,
        # -- Biofilm panel --
        "Biofilm_Stage":             bf_stage,
        "icaADBC_Active":            ica_active,
        "pel_psl_Active":            pel_psl_active,
        "Quorum_Sensing_Active":     qs_active,
        "QS_Activity_Score":         qs_activity,
        "AI2_Level_uM":              ai2_level_uM,
        "c_diGMP_Level_uM":          cdgmp_level,
        # -- Oncology panel --
        "Oncology_Risk_Tier":        onc_tier,
        "Cancer_Microbes_Detected":  "|".join(onc_microbes) or "None",
        "Oncology_Species":          "|".join(onc_species)  or "None",
        "Chronic_Inflammation_Load": inflam_load,
        "Referral_Recommended":      referral,
    }


# =============================================================================
# Main generation loop
# =============================================================================
def generate(n: int = N) -> tuple:
    rows_main = []
    rows_gt   = []
    rows_shap = []

    for i in range(1, n + 1):
        category = np.random.choice(CATEGORIES, p=CATEGORY_PROBS)

        # Stage 1
        row = patient_metadata(i)
        row["Clinical_Category"] = category

        # Stage 2
        gt = assign_ground_truth(category)
        rows_gt.append({"Sample_ID": row["Sample_ID"],
                        **{ch: int(v) for ch, v in gt.items()}})

        # Stages 3 & 4 -- signals, Kalman smooth, feature extraction
        channel_signals = {}   # used later by clinical_report
        for ch_name in CHANNEL_MAP:
            is_present = gt[ch_name]
            raw, conc  = raw_signal(is_present, ch_name)
            smooth     = kalman_smooth(raw)
            feats      = extract_features(raw, smooth, is_present)

            row[f"{ch_name}_raw_nA"]           = raw
            row[f"{ch_name}_smooth_nA"]         = smooth
            row[f"{ch_name}_conc_pM"]           = conc
            row[f"{ch_name}_peak_amp_nA"]        = feats["peak_amplitude_nA"]
            row[f"{ch_name}_drop_pct"]           = feats["signal_drop_pct"]
            row[f"{ch_name}_t2t_s"]             = feats["time_to_threshold_s"]
            row[f"{ch_name}_impedance_pct"]     = feats["impedance_change_pct"]
            row[f"{ch_name}_snr_db"]            = feats["snr_db"]

            channel_signals[f"{ch_name}_drop_pct"] = feats["signal_drop_pct"]
            channel_signals[f"{ch_name}_conc_pM"]  = conc

        # Stage 5 -- model-specific AI scores
        amr_out = xgboost_amr_score(gt)
        bio_out = lstm_biofilm_score(gt)
        onc_out = gnn_oncology_score(gt)

        score_amr = amr_out["AI_AMR_Score"]
        score_bio = bio_out["AI_Biofilm_Score"]
        score_onc = onc_out["AI_Oncology_Score"]

        row.update(amr_out)
        row.update(bio_out)
        row.update(onc_out)

        # Stage 6 -- per-model MC Dropout confidence
        for domain, score, mtype in [
            ("AMR",      score_amr, "xgboost"),
            ("Biofilm",  score_bio, "lstm"),
            ("Oncology", score_onc, "gnn"),
        ]:
            mc = mc_dropout_confidence(score, mtype)
            row[f"MC_{domain}_StdDev"]    = mc["mc_std_dev"]
            row[f"MC_{domain}_Conf_pct"]  = mc["confidence_pct"]

        # Stage 7 -- SHAP attributions
        shap_row = {"Sample_ID": row["Sample_ID"]}
        for domain, channels, score in [
            ("AMR",      AMR_CHANNELS,      score_amr),
            ("Biofilm",  BIOFILM_CHANNELS,  score_bio),
            ("Oncology", ONCOLOGY_CHANNELS, score_onc),
        ]:
            shap_vals = compute_shap(gt, channels, score)
            shap_row.update(shap_vals)
            top_driver = (max(shap_vals, key=lambda k: abs(shap_vals[k]))
                          .replace("SHAP_", "") if shap_vals else "N/A")
            row[f"SHAP_{domain}_TopDriver"] = top_driver

        rows_shap.append(shap_row)

        # Stage 8 -- device output + ensemble risk
        dev = device_output(score_amr, score_bio, score_onc, gt)
        row.update(dev)

        # Stage 9 -- clinical report
        cli = clinical_report(gt, score_bio, score_onc, channel_signals)
        row.update(cli)

        row["Diagnostic_Label"] = category
        rows_main.append(row)

    df_main = pd.DataFrame(rows_main)
    df_gt   = pd.DataFrame(rows_gt)
    df_shap = pd.DataFrame(rows_shap)
    return df_main, df_gt, df_shap


# =============================================================================
# Schema / data dictionary
# =============================================================================
def build_schema() -> dict:
    return {
        "description":   "AI-Enabled Multiplex Biosensor -- Synthetic Dataset v3.0 (SVG-aligned)",
        "version":       "3.0",
        "n_samples":     N,
        "generated_at":  datetime.now().isoformat(),
        "changes_from_v2": [
            "Ch06 biomarker: AHL -> AHL_AI2 (combined quorum sensing channel)",
            "Ch07 biomarker: bap -> bap_pel_psl (biofilm matrix genes)",
            "AMR head: Multilabel XGBoost simulation with per-gene binary calls",
            "Biofilm head: LSTM temporal model with temporal coherence score",
            "Oncology head: GNN microbial network with co-occurrence bonus and network centrality",
            "Added QS_Activity_Score (continuous 0-100)",
            "Added AI2_Level_uM field",
            "Added pel_psl_Active binary field",
            "Added icaADBC_Active binary field",
            "Ensemble risk includes co-occurrence bonus for multi-domain positives",
            "Added snr_db feature per channel",
            "Added AMR_Resistance_Profile label and AMR_N_Genes_Detected count",
        ],
        "pipeline_stages": {
            "Stage1_Metadata":       ["Sample_ID", "Collection_Date", "Collection_Time",
                                      "Patient_Age", "Patient_Sex", "Patient_BMI",
                                      "Sample_Type", "Collection_Site", "Sample_Volume_uL",
                                      "Cartridge_Lot", "Device_ID", "Ambient_Temp_C", "Ambient_pH"],
            "Stage2_GroundTruth":    "biosensor_ground_truth.csv -- Ch01..Ch12 binary presence/absence",
            "Stage3_RawSignal":      "ChXX_raw_nA -- Signal-OFF log model (baseline=10 nA)",
            "Stage4_Features":       ["ChXX_smooth_nA", "ChXX_conc_pM", "ChXX_peak_amp_nA",
                                      "ChXX_drop_pct", "ChXX_t2t_s",
                                      "ChXX_impedance_pct", "ChXX_snr_db"],
            "Stage5_AI_Scores":      {
                "AMR_head":      "Multilabel XGBoost -- AI_AMR_Score + XGB_*_call per gene",
                "Biofilm_head":  "LSTM temporal -- AI_Biofilm_Score + LSTM_Temporal_Coherence",
                "Oncology_head": "GNN network -- AI_Oncology_Score + GNN_Network_Centrality",
            },
            "Stage6_Uncertainty":    ["MC_AMR_StdDev", "MC_AMR_Conf_pct",
                                      "MC_Biofilm_StdDev", "MC_Biofilm_Conf_pct",
                                      "MC_Oncology_StdDev", "MC_Oncology_Conf_pct"],
            "Stage7_SHAP":           "biosensor_shap.csv -- per-biomarker SHAP contributions",
            "Stage8_Device":         ["Transmission_Mode", "Total_Time_min",
                                      "Ensemble_Risk_Score", "Ensemble_Risk_Tier",
                                      "N_Active_Domains", "Co_occurrence_Bonus"],
            "Stage9_ClinicalReport": {
                "AMR_Panel":      ["AMR_Resistance_Profile", "AMR_Detected_Genes",
                                   "AMR_Failed_Antibiotics", "AMR_Recommended_Alt",
                                   "AMR_N_Genes_Detected"],
                "Biofilm_Panel":  ["Biofilm_Stage", "icaADBC_Active", "pel_psl_Active",
                                   "Quorum_Sensing_Active", "QS_Activity_Score",
                                   "AI2_Level_uM", "c_diGMP_Level_uM"],
                "Oncology_Panel": ["Oncology_Risk_Tier", "Cancer_Microbes_Detected",
                                   "Oncology_Species", "Chronic_Inflammation_Load",
                                   "Referral_Recommended"],
            },
        },
        "channel_map":   CHANNEL_MAP,
        "categories":    dict(zip(CATEGORIES, CATEGORY_PROBS)),
    }


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    print(f"Generating {N} samples (v3.0 SVG-aligned)...")
    df_main, df_gt, df_shap = generate(N)

    schema = build_schema()

    out_main   = "data/biosensor_detailed_500.csv"
    out_gt     = "data/biosensor_ground_truth.csv"
    out_shap   = "data/biosensor_shap.csv"
    out_schema = "data/biosensor_metadata.json"

    df_main.to_csv(out_main,  index=False)
    df_gt.to_csv(out_gt,      index=False)
    df_shap.to_csv(out_shap,  index=False)
    with open(out_schema, "w") as f:
        json.dump(schema, f, indent=2)

    print(f"\n[OK] Files written:")
    print(f"  {out_main}    -- {df_main.shape[0]} rows x {df_main.shape[1]} cols")
    print(f"  {out_gt}  -- {df_gt.shape[0]} rows x {df_gt.shape[1]} cols")
    print(f"  {out_shap}    -- {df_shap.shape[0]} rows x {df_shap.shape[1]} cols")
    print(f"  {out_schema}")

    print(f"\nCategory distribution:")
    print(df_main["Clinical_Category"].value_counts().to_string())

    print(f"\nNew fields confirmed present:")
    new_cols = ["QS_Activity_Score", "AI2_Level_uM", "pel_psl_Active",
                "icaADBC_Active", "LSTM_Temporal_Coherence",
                "GNN_Network_Centrality", "Ensemble_Risk_Tier",
                "Co_occurrence_Bonus", "AMR_Resistance_Profile"]
    for col in new_cols:
        status = "[OK]" if col in df_main.columns else "[MISSING]"
        print(f"  {status}  {col}")

    print(f"\nSample preview:")
    print(df_main[[
        "Sample_ID", "Clinical_Category",
        "AI_AMR_Score", "AI_Biofilm_Score", "AI_Oncology_Score",
        "QS_Activity_Score", "Ensemble_Risk_Score", "Ensemble_Risk_Tier",
        "Referral_Recommended"
    ]].head(3).to_string())
