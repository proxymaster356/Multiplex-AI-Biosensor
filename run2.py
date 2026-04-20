#!/usr/bin/env python
"""
run2.py  --  AI Multiplex Biosensor  --  Live Device Simulation
================================================================
Simulates a REAL potentiostat streaming µA data into the AI pipeline.

What this shows end-to-end (nothing pre-computed):
  1. Device boot + cartridge calibration
  2. 60-second baseline measurement (stable ~10 nA)
  3. Sample injection → live analog signal stream (µA every second)
  4. Adaptive Kalman filtering in real-time
  5. Automatic detection when drop crosses 30% threshold
  6. Feature extraction (peak amplitude, t2t, SNR, AUC, conc)
  7. Multimodal encoder (1D-CNN shared weights)
  8. Cross-channel Transformer attention
  9. Bayesian calibration (MC Dropout, 50 passes)
  10. Task-specific heads (XGBoost/GB/RF)
  11. Ensemble risk engine
  12. HTML clinical report generation

Run:
    python run2.py               (default scenario: ICU patient, 3 positives)
    python run2.py --fast        (skip slow phases, ~8s total)
    python run2.py --scenario 2  (different patient scenario)
"""

import sys
import time
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── Enable ANSI colours on Windows ──────────────────────────────────────────
if sys.platform == "win32":
    os.system("")                 # unlocks VT100 on Windows Terminal / cmd
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).resolve().parent

# =============================================================================
# ANSI helpers
# =============================================================================
class C:
    R   = "\033[0m"   ; B   = "\033[1m"    ; D   = "\033[2m"
    RED = "\033[91m"  ; GRN = "\033[92m"   ; YEL = "\033[93m"
    BLU = "\033[94m"  ; MAG = "\033[95m"   ; CYN = "\033[96m"
    WHT = "\033[97m"
    BRED= "\033[41m"  ; BGRN= "\033[42m"   ; BBLU= "\033[44m"
    BYEL= "\033[43m"
    UP  = staticmethod(lambda n: f"\033[{n}A")
    CLR = "\033[2K\r"

def w(txt="", end="\n", flush=True):
    sys.stdout.write(txt + end)
    if flush: sys.stdout.flush()

def up(n): sys.stdout.write(C.UP(n)); sys.stdout.flush()
def clr():  sys.stdout.write(C.CLR);  sys.stdout.flush()

def progress_bar(current, total, width=36, label=""):
    pct   = current / max(total, 1)
    filled= int(width * pct)
    bar   = C.CYN + "█" * filled + C.D + "░" * (width - filled) + C.R
    return f"[{bar}] {C.B}{pct*100:5.1f}%{C.R}  {current}/{total}s  {label}"


# =============================================================================
# Clinical scenarios
# =============================================================================
SCENARIOS = {
    1: {
        "name":        "ICU Patient — AMR + Biofilm + Oncology",
        "patient_id":  "REAL-PAT-001",
        "age": 67, "sex": "M", "bmi": 28.3,
        "site": "ICU", "sample": "Blood", "vol_uL": 5.2,
        "device": "DEV-B42", "lot": "LOT-2024-001",
        "temp_C": 24.5, "pH": 7.32,
        # Which channels will show a real binding event
        "positive": {
            "Ch02": 77.8,    # mecA / MRSA         77.8 pM
            "Ch05": 317.7,   # icaADBC / biofilm   317.7 pM
            "Ch09": 320.3,   # FadA / F.nucleatum  320.3 pM
        },
        "clinical_note": "Post-surgical wound infection with suspected biofilm + GI oncology co-signal",
    },
    2: {
        "name":        "ED Patient — AMR Only",
        "patient_id":  "REAL-PAT-002",
        "age": 42, "sex": "F", "bmi": 23.1,
        "site": "ED", "sample": "Urine", "vol_uL": 4.8,
        "device": "DEV-A11", "lot": "LOT-2024-003",
        "temp_C": 22.0, "pH": 7.18,
        "positive": {
            "Ch01": 155.2,   # blaNDM-1  NDM carbapenemase
            "Ch04": 88.6,    # KPC
        },
        "clinical_note": "Suspected carbapenem-resistant UTI",
    },
    3: {
        "name":        "Outpatient — Biofilm Screening",
        "patient_id":  "REAL-PAT-003",
        "age": 55, "sex": "M", "bmi": 31.5,
        "site": "Outpatient_clinic", "sample": "Wound_swab", "vol_uL": 5.0,
        "device": "DEV-C07", "lot": "LOT-2024-002",
        "temp_C": 23.5, "pH": 7.41,
        "positive": {
            "Ch06": 234.1,   # AHL/AI-2 Quorum sensing
            "Ch07": 189.5,   # bap/pel/psl matrix
            "Ch08": 98.4,    # c-diGMP
        },
        "clinical_note": "Chronic wound / suspected Stage III biofilm",
    },
}

CHANNEL_META = {
    "Ch01": ("AMR",      "blaNDM-1",   C.BLU),
    "Ch02": ("AMR",      "mecA",       C.BLU),
    "Ch03": ("AMR",      "vanA",       C.BLU),
    "Ch04": ("AMR",      "KPC",        C.BLU),
    "Ch05": ("Biofilm",  "icaADBC",    C.MAG),
    "Ch06": ("Biofilm",  "AHL/AI-2",   C.MAG),
    "Ch07": ("Biofilm",  "bap/pel/psl",C.MAG),
    "Ch08": ("Biofilm",  "c-diGMP",    C.MAG),
    "Ch09": ("Oncology", "FadA",       C.RED),
    "Ch10": ("Oncology", "CagA",       C.RED),
    "Ch11": ("Oncology", "pks",        C.RED),
    "Ch12": ("Oncology", "miRNA-21",   C.RED),
}


# =============================================================================
# Signal physics  (real device simulation)
# =============================================================================
BASELINE_nA  = 10.0
NOISE_STD_nA = 0.15      # potentiostat noise floor
DURATION_S   = 900
INJECT_S     = 60        # sample injection at t=60s

def _sigmoid_decay(t, t0, tau, baseline_nA, plateau_nA):
    """Exponential approach to plateau after injection (real binding kinetics)."""
    return np.where(t < t0,
                    baseline_nA,
                    plateau_nA + (baseline_nA - plateau_nA) * np.exp(-tau * (t - t0)))

def generate_channel_signal(t_arr, conc_pM=None, rng=None, seed=0):
    """
    Generate realistic amperometric time-series for one channel.
    conc_pM: if None → not present (stays at baseline)
    Returns (raw_nA, raw_uA) arrays
    """
    rng = rng or np.random.default_rng(seed)
    noise = rng.normal(0, NOISE_STD_nA, len(t_arr))
    # Tiny baseline drift (temperature / electrode ageing)
    drift = rng.uniform(-0.02, 0.02) * t_arr / DURATION_S

    if conc_pM is not None:
        plateau_nA = np.clip(BASELINE_nA - 2.1 * np.log10(conc_pM), 0.8, 8.5)
        # Binding tau: lower conc → slower kinetics
        tau = 0.005 + 0.003 / (conc_pM / 100)
        signal = _sigmoid_decay(t_arr, INJECT_S, tau, BASELINE_nA, plateau_nA)
    else:
        signal = np.full_like(t_arr, BASELINE_nA)

    raw_nA = signal + drift + noise
    raw_uA = raw_nA / 1000.0          # µA output (what real potentiostat sends)
    return raw_nA, raw_uA


# =============================================================================
# Adaptive Kalman filter  (per-sample streaming)
# =============================================================================
class StreamingKalman:
    def __init__(self):
        self.x = None; self.P = 1.0
        self.Q_lo = 0.005; self.Q_hi = 0.05; self.R = NOISE_STD_nA**2

    def step(self, z):
        if self.x is None: self.x = z
        dz = abs(z - self.x)
        Q  = self.Q_hi if dz > 0.05 else self.Q_lo   # adaptive
        P_pred = self.P + Q
        K = P_pred / (P_pred + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * P_pred
        return self.x


# =============================================================================
# Live display helpers
# =============================================================================
def signal_bar(nA, width=20):
    """Visual bar proportional to nA level (baseline=full bar)."""
    frac   = max(0, min(nA / BASELINE_nA, 1.0))
    filled = int(frac * width)
    pct    = (1 - frac) * 100
    if pct < 5:   col = C.GRN
    elif pct < 30: col = C.YEL
    else:          col = C.RED
    return col + "█" * filled + C.D + "░" * (width - filled) + C.R

def status_badge(drop_pct, detected):
    if detected:
        return C.B + C.RED + "  DETECTED " + C.R
    elif drop_pct > 15:
        return C.YEL + " dropping  " + C.R
    else:
        return C.GRN + "  stable   " + C.R

# Table lines height (must be exact for cursor-up to work)
TABLE_HEADER_LINES = 4   # progress bar + phase line + sep + col header + sep
TABLE_BODY_LINES   = 12  # one per channel
TABLE_FOOT_LINES   = 2   # separator + blank
TABLE_TOTAL        = TABLE_HEADER_LINES + TABLE_BODY_LINES + TABLE_FOOT_LINES  # 18

def draw_table(t_cur, phase_label, ch_raw, ch_smooth, ch_drop, ch_det,
               events, first=False):
    """
    Redraws the live acquisition table.
    If not first, moves cursor up TABLE_TOTAL lines first.
    """
    if not first:
        sys.stdout.write(C.UP(TABLE_TOTAL))

    lines = []
    # Row 1: progress bar
    lines.append(f"  {C.B}LIVE ACQUISITION{C.R}  "
                 + progress_bar(int(t_cur), DURATION_S, width=32,
                                label=phase_label))
    # Row 2: column header
    lines.append(
        f"  {'':─<4}╦{'':─<9}╦{'':─<11}╦{'':─<11}╦{'':─<11}╦{'':─<23}╦{'':─<11}")
    lines.append(
        f"  {'CH':<4}║{'BIOMARKER':<9}║{'RAW (µA)':<11}║"
        f"{'FILT(nA)':<11}║{'DROP%':<11}║{'SIGNAL LEVEL':<23}║{'STATUS':<11}")
    lines.append(
        f"  {'':─<4}╬{'':─<9}╬{'':─<11}╬{'':─<11}╬{'':─<11}╬{'':─<23}╬{'':─<11}")

    # Channel rows
    for ch in ["Ch01","Ch02","Ch03","Ch04","Ch05","Ch06",
               "Ch07","Ch08","Ch09","Ch10","Ch11","Ch12"]:
        domain, bm, dcol = CHANNEL_META[ch]
        raw_uA = ch_raw.get(ch, BASELINE_nA) / 1000
        sm_nA  = ch_smooth.get(ch, BASELINE_nA)
        drop   = ch_drop.get(ch, 0.0)
        det    = ch_det.get(ch, False)
        bar    = signal_bar(sm_nA, width=20)
        badge  = status_badge(drop, det)

        # Arrow indicator on raw value
        prev   = ch_raw.get(f"_{ch}_prev", ch_raw.get(ch, BASELINE_nA))
        arrow  = (C.RED + "▼" if raw_uA < prev/1000 - 0.000005
                  else (C.GRN + "▲" if raw_uA > prev/1000 + 0.000005
                        else C.D + "─")) + C.R

        row_color = (C.RED if det else (C.YEL if drop > 15 else ""))
        lines.append(
            f"  {row_color}{ch:<4}{C.R}║"
            f"{dcol}{bm:<9}{C.R}║"
            f" {raw_uA:.6f}{arrow} ║"
            f"  {sm_nA:6.3f}   ║"
            f" {drop:6.1f}%   ║"
            f" {bar}{' ':1} ║"
            f"{badge}║")

    lines.append(
        f"  {'':─<4}╩{'':─<9}╩{'':─<11}╩{'':─<11}╩{'':─<11}╩{'':─<23}╩{'':─<11}")
    # Last row: events line
    if events:
        evline = events[-1][:90]
        lines.append(f"  {C.B}{C.RED}⚠  {evline}{C.R}")
    else:
        lines.append(f"  {C.D}  Waiting for detection events...{C.R}")

    for l in lines:
        sys.stdout.write(C.CLR + l + "\n")
    sys.stdout.flush()


# =============================================================================
# Phase 1 — Boot sequence
# =============================================================================
def phase_boot(sc, fast):
    w()
    w(C.B + C.CYN + "  ╔══════════════════════════════════════════════════════╗" + C.R)
    w(C.B + C.CYN + "  ║   AI - ENABLED MULTIPLEX BIOSENSOR SYSTEM           ║" + C.R)
    w(C.B + C.CYN + "  ║   Live Device Simulation  —  Real Hardware Protocol  ║" + C.R)
    w(C.B + C.CYN + "  ╚══════════════════════════════════════════════════════╝" + C.R)
    w()

    def boot_line(msg, delay=0.4, ok=True):
        w(f"  {C.D}▶{C.R}  {msg}", end="  ")
        time.sleep(0.1 if fast else delay)
        w(C.GRN + C.B + "[  OK  ]" + C.R if ok else C.YEL + "[ SKIP ]" + C.R)

    w(f"  {C.B}Scenario{C.R}  : {C.YEL}{sc['name']}{C.R}")
    w(f"  {C.B}Patient{C.R}   : {sc['age']}{sc['sex']} / {sc['site']} / "
      f"{sc['sample']} ({sc['vol_uL']} µL)")
    w(f"  {C.B}Device{C.R}    : {sc['device']}  |  Lot: {sc['lot']}")
    w(f"  {C.B}Conditions{C.R}: {sc['temp_C']}°C  /  pH {sc['pH']}")
    w(f"  {C.B}Note{C.R}      : {C.D}{sc['clinical_note']}{C.R}")
    w()

    boot_line("Powering potentiostat (Vref=3.3V, R_shunt=100kΩ)...", 0.5)
    boot_line("Connecting to 12-channel nanostructured Au/GO electrode array...", 0.6)
    boot_line("Loading biosensor cartridge...", 0.4)
    boot_line("Verifying probe chemistry (CRISPR-Cas12a / Aptamer / Antibody)...", 0.5)
    boot_line("Starting ADC sampling at 1 Hz (16-bit, 0–33 µA range)...", 0.4)
    boot_line("Loading Adaptive Kalman filter (Q_lo=0.005, Q_hi=0.05, R=0.023)...", 0.3)
    w()


# =============================================================================
# Phase 2 — Baseline calibration
# =============================================================================
def phase_baseline(sc, rng, fast):
    w(f"  {C.B}━━━  PHASE 1: BASELINE CALIBRATION  ━━━{C.R}")
    w(f"  Measuring open-circuit current for {INJECT_S}s before injection...")
    w()

    kf   = {ch: StreamingKalman() for ch in CHANNEL_META}
    samp = max(1, INJECT_S // 6) if fast else 1
    steps= range(0, INJECT_S, samp)

    for t in steps:
        row = f"  t={t:3d}s  |"
        for ch in CHANNEL_META:
            noise = rng.normal(0, NOISE_STD_nA)
            raw   = BASELINE_nA + noise
            sm    = kf[ch].step(raw)
            row  += f"  {ch[-2:]}:{sm:.3f}"
        w(row + f"  {C.D}nA{C.R}", flush=True)
        time.sleep(0.05 if fast else 0.12)

    bl_noise = rng.normal(0, NOISE_STD_nA, 100).std()
    w()
    w(f"  {C.GRN}{C.B}✓  Baseline stable  |  All 12 channels ~{BASELINE_nA:.1f} nA"
      f"  |  Noise floor σ = {bl_noise:.3f} nA{C.R}")
    w()
    time.sleep(0.3 if fast else 0.8)


# =============================================================================
# Phase 3 — Live acquisition
# =============================================================================
def phase_acquisition(sc, rng, fast):
    w(f"  {C.B}━━━  PHASE 2: SAMPLE INJECTION  →  LIVE ACQUISITION  ━━━{C.R}")
    w()
    w(f"  {C.YEL}{C.B}⚡  SAMPLE INJECTED AT t=60s  —  Binding reaction started{C.R}")
    w(f"  {C.D}Streaming analog µA data → Adaptive Kalman filter → detection engine{C.R}")
    w()
    time.sleep(0.5 if fast else 1.2)

    # Generate full signal arrays
    t_arr = np.arange(0, DURATION_S, 1.0)
    signals_nA  = {}
    signals_uA  = {}
    for ch in CHANNEL_META:
        conc = sc["positive"].get(ch, None)
        seed = int(ch[2:]) * 7
        raw_nA, raw_uA = generate_channel_signal(t_arr, conc, rng=rng, seed=seed)
        signals_nA[ch] = raw_nA
        signals_uA[ch] = raw_uA

    # Streaming Kalman per channel
    kf       = {ch: StreamingKalman() for ch in CHANNEL_META}
    ch_raw   = {ch: BASELINE_nA for ch in CHANNEL_META}
    ch_smooth= {ch: BASELINE_nA for ch in CHANNEL_META}
    ch_drop  = {ch: 0.0         for ch in CHANNEL_META}
    ch_det   = {ch: False       for ch in CHANNEL_META}
    events   = []

    # Step size for display (fast-forward)
    step   = 3 if fast else 6
    sleep  = 0.06 if fast else 0.14
    first  = True

    for idx in range(0, DURATION_S, step):
        t = t_arr[idx]

        for ch in CHANNEL_META:
            raw_nA  = float(signals_nA[ch][idx])
            sm_nA   = kf[ch].step(raw_nA)
            drop    = max(0.0, (BASELINE_nA - sm_nA) / BASELINE_nA * 100)

            # Store previous raw for arrow direction
            ch_raw[f"_{ch}_prev"] = ch_raw[ch]
            ch_raw[ch]   = raw_nA
            ch_smooth[ch]= sm_nA
            ch_drop[ch]  = drop

            # Detection event
            if drop >= 30.0 and not ch_det[ch]:
                ch_det[ch] = True
                domain, bm, _ = CHANNEL_META[ch]
                events.append(
                    f"t={int(t):3d}s  {ch} ({bm}) crossed threshold  "
                    f"drop={drop:.1f}%  conc~{sc['positive'].get(ch, 0):.0f} pM")

        phase_lbl = ("BASELINE" if t < INJECT_S else
                     ("BINDING" if t < 600 else "PLATEAU"))

        draw_table(t, phase_lbl, ch_raw, ch_smooth, ch_drop, ch_det,
                   events, first=first)
        first = False
        time.sleep(sleep)

    # Print detection events below table
    w()
    if events:
        w(f"  {C.B}Detection events logged:{C.R}")
        for ev in events:
            w(f"    {C.RED}⚠  {ev}{C.R}")
    w()

    return signals_nA, signals_uA, ch_smooth, ch_drop, ch_det


# =============================================================================
# Phase 4 — Feature extraction
# =============================================================================
def phase_features(sc, signals_nA, ch_smooth, ch_drop, ch_det, fast):
    w(f"  {C.B}━━━  PHASE 3: SIGNAL PROCESSING  &  FEATURE EXTRACTION  ━━━{C.R}")
    w()

    t_arr   = np.arange(0, DURATION_S, 1.0)
    feat_row= {}
    feat_summary = []

    for i, ch in enumerate(CHANNEL_META):
        domain, bm, dcol = CHANNEL_META[ch]
        raw_nA  = signals_nA[ch]
        sm_nA   = np.array([StreamingKalman().step(v) for v in raw_nA])

        # Baseline stats
        bl_mean = raw_nA[:INJECT_S].mean()
        bl_std  = raw_nA[:INJECT_S].std()

        # Plateau (last 60s)
        pl_smooth = sm_nA[-60:].mean()
        peak_amp  = max(0, bl_mean - pl_smooth)
        drop_pct  = max(0, peak_amp / bl_mean * 100)

        # Time-to-threshold
        thresh   = bl_mean * 0.7
        below    = np.where(sm_nA <= thresh)[0]
        t2t_s    = float(t_arr[below[0]]) if len(below) > 0 else 1200.0

        # SNR
        snr = round(10 * np.log10(max(peak_amp**2, 1e-9) / max(bl_std**2, 1e-9)), 1)

        # AUC
        _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        auc    = round(float(_trapz(np.clip(bl_mean - sm_nA, 0, None), t_arr)), 2)

        # Concentration back-calc
        conc_pM= round(float(10**((BASELINE_nA - pl_smooth) / 2.1)), 1) \
                 if drop_pct >= 30 else 0.0

        # Kinetic slope (60s to t2t)
        t2t_idx = min(int(t2t_s), DURATION_S - 2)
        kinslope= round((sm_nA[t2t_idx] - sm_nA[INJECT_S]) /
                         max(t2t_s - INJECT_S, 1), 5)

        # Impedance
        imp_pct = round(peak_amp / max(bl_mean, 0.01) * 2.2 * 100, 2)

        # Detected
        detected = drop_pct >= 30.0

        # Write to feat_row
        feat_row[f"{ch}_raw_nA"]         = round(pl_smooth, 3)
        feat_row[f"{ch}_smooth_nA"]      = round(pl_smooth, 3)
        feat_row[f"{ch}_conc_pM"]        = conc_pM
        feat_row[f"{ch}_peak_amp_nA"]    = round(peak_amp, 3)
        feat_row[f"{ch}_drop_pct"]       = round(drop_pct, 2)
        feat_row[f"{ch}_t2t_s"]         = t2t_s
        feat_row[f"{ch}_impedance_pct"] = imp_pct
        feat_row[f"{ch}_snr_db"]        = snr
        feat_row[f"{ch}_detected"]      = int(detected)

        col = (C.RED if detected else (C.YEL if drop_pct > 15 else C.GRN))
        badge = (C.B + C.RED + " DETECTED" + C.R if detected else
                 C.GRN + " negative" + C.R)

        w(f"  [{i+1:2d}/12]  {dcol}{ch}{C.R}  {bm:<12}  "
          f"drop={col}{drop_pct:5.1f}%{C.R}  "
          f"t2t={t2t_s:6.0f}s  "
          f"SNR={snr:6.1f} dB  "
          f"conc={conc_pM:7.1f} pM  "
          f"AUC={auc:8.1f} nA·s  {badge}")

        if detected:
            feat_summary.append((ch, bm, domain, drop_pct, conc_pM, t2t_s))
        time.sleep(0.04 if fast else 0.12)

    w()
    w(f"  {C.GRN}✓  {len(feat_summary)} biomarkers detected "
      f"/ 12 channels screened{C.R}")
    w()

    # Add metadata fields
    now = datetime.now()
    feat_row.update({
        "Sample_ID":       sc["patient_id"],
        "Collection_Date": now.strftime("%Y-%m-%d"),
        "Collection_Time": now.strftime("%H:%M"),
        "Patient_Age":     sc["age"],
        "Patient_Sex":     sc["sex"],
        "Patient_BMI":     sc["bmi"],
        "Sample_Type":     sc["sample"],
        "Collection_Site": sc["site"],
        "Sample_Volume_uL":sc["vol_uL"],
        "Device_ID":       sc["device"],
        "Cartridge_Lot":   sc["lot"],
        "Ambient_Temp_C":  sc["temp_C"],
        "Ambient_pH":      sc["pH"],
        "Clinical_Category":"REAL",
        "Diagnostic_Label":"REAL",
    })

    return pd.DataFrame([feat_row]), feat_summary


# =============================================================================
# Phase 5 — AI pipeline
# =============================================================================
def phase_ai(feature_df, feat_summary, sc, fast):
    w(f"  {C.B}━━━  PHASE 4: AI / ML PIPELINE INFERENCE  ━━━{C.R}")
    w()

    from pipeline.signal_processor  import SignalProcessor
    from pipeline.multimodal_encoder import MultimodalEncoder
    from pipeline.models             import (AMRHead, BiofilmHead, OncologyHead,
                                             EnsembleRiskEngine,
                                             AMR_GENE_LABELS,
                                             BIOFILM_STAGE_MAP, ONCOLOGY_TIER_MAP)

    sp  = SignalProcessor()

    # [3a] Encoder — we need a training set; load the 500-sample CSV
    data_file = BASE / "data" / "biosensor_detailed_500.csv"
    gt_file   = BASE / "data" / "biosensor_ground_truth.csv"
    if not data_file.exists():
        w(f"  {C.RED}biosensor_detailed_500.csv not found. Run python detailed_dummy_generator.py first.{C.R}")
        sys.exit(1)

    df_train = pd.read_csv(data_file)
    df_gt    = pd.read_csv(gt_file)
    gt_cols  = ["Sample_ID"] + [f"Ch{i:02d}" for i in range(1, 13)]
    df_train = df_train.merge(df_gt[gt_cols], on="Sample_ID", suffixes=("","_gt"))

    # ── [3a] Multimodal Encoder ───────────────────────────────────────────────
    w(f"  {C.CYN}[3a]{C.R} Multimodal Encoder (1D-CNN shared weights)...")
    time.sleep(0.15 if fast else 0.4)
    enc = MultimodalEncoder(latent_dim=32, n_heads=4, n_mc_passes=50)
    enc.fit(df_train)

    tr_aug  = enc.get_augmented_features(df_train, sp.process(df_train))
    te_aug  = enc.get_augmented_features(feature_df, sp.process(feature_df))
    te_enc  = te_aug["_enc_out"]

    attn    = te_enc["attention_matrix"][0]         # (12,12)
    cd      = te_enc["cross_domain_scores"][0]      # (3,)
    w(f"        {C.D}channel embeddings: 12 × 32-dim  "
      f"|  cross-domain attention: AMR={cd[0]:.2f}  "
      f"Biofilm={cd[1]:.2f}  Onc={cd[2]:.2f}{C.R}")
    w(f"  {C.GRN}        ✓ Fused latent vector ready{C.R}")
    time.sleep(0.15 if fast else 0.5)

    # ── [3b] Attention insight ────────────────────────────────────────────────
    w(f"  {C.CYN}[3b]{C.R} Cross-Channel Transformer Attention...")
    # Find strongest cross-domain link
    amr_idx = [0,1,2,3]; bf_idx=[4,5,6,7]; onc_idx=[8,9,10,11]
    ab = attn[np.ix_(amr_idx, bf_idx)].mean()
    ao = attn[np.ix_(amr_idx, onc_idx)].mean()
    bo = attn[np.ix_(bf_idx,  onc_idx)].mean()
    strongest = max([(ab,"AMR↔Biofilm"),(ao,"AMR↔Oncology"),(bo,"Biofilm↔Oncology")],
                    key=lambda x:x[0])
    w(f"        {C.D}Strongest co-signal: {strongest[1]} (score={strongest[0]:.3f}){C.R}")
    if max(ab, ao, bo) > 0.01:
        w(f"        {C.YEL}{C.B}⚡ Multi-domain co-signal detected — "
          f"co-occurrence bonus will apply{C.R}")
    time.sleep(0.15 if fast else 0.5)

    # ── [3c] MC Dropout ───────────────────────────────────────────────────────
    w(f"  {C.CYN}[3c]{C.R} Bayesian Calibration (MC Dropout, 50 stochastic passes)...")
    calib = te_enc["calibration"]
    time.sleep(0.2 if fast else 0.6)
    w(f"        {C.D}Confidence — AMR:{calib['MC_AMR_conf'][0]:.0f}%  "
      f"Biofilm:{calib['MC_Biofilm_conf'][0]:.0f}%  "
      f"Oncology:{calib['MC_Oncology_conf'][0]:.0f}%{C.R}")

    # ── [3d] Task heads ───────────────────────────────────────────────────────
    w(f"  {C.CYN}[3d]{C.R} Task-Specific Classification Heads:")

    ## AMR
    def amr_y(df):
        cols = [f"Ch{i:02d}" for i in range(1,5)]
        y = df[cols].copy(); y.columns = AMR_GENE_LABELS; return y.astype(int)
    def bio_y(df): return df["Biofilm_Stage"].fillna("None")
    def onc_y(df): return df["Oncology_Risk_Tier"].fillna("Low")

    X_amr_tr = tr_aug["AMR"];      X_amr_te = te_aug["AMR"]
    X_bio_tr = tr_aug["Biofilm"];  X_bio_te = te_aug["Biofilm"]
    X_onc_tr = tr_aug["Oncology"]; X_onc_te = te_aug["Oncology"]

    w(f"        {C.BLU}AMR{C.R}      (Multilabel XGBoost)...", end="  ")
    amr = AMRHead(n_estimators=200); amr.fit(X_amr_tr, amr_y(df_train))
    amr_proba = amr.predict_proba(X_amr_te)
    amr_pred  = amr.predict(X_amr_te)
    score_amr = float(amr_proba["AI_AMR_Score"].iloc[0])
    genes_pos = [g for g in AMR_GENE_LABELS
                 if amr_pred[g].iloc[0] == 1] if not amr_pred.empty else []
    w(f"Score: {C.B}{score_amr:5.1f}%{C.R}  Detected genes: "
      + (C.RED + C.B + "|".join(genes_pos) + C.R if genes_pos else C.GRN + "None" + C.R))

    w(f"        {C.MAG}Biofilm{C.R}  (Gradient Boosting LSTM-style)...", end="  ")
    bio = BiofilmHead(n_estimators=300); bio.fit(X_bio_tr, bio_y(df_train))
    bio_proba = bio.predict_proba(X_bio_te)
    bio_pred  = bio.predict(X_bio_te)
    score_bio = float(bio_proba["AI_Biofilm_Score"].iloc[0])
    stage_pred= str(bio_pred.iloc[0])
    w(f"Score: {C.B}{score_bio:5.1f}%{C.R}  Stage: {C.MAG}{stage_pred}{C.R}")

    w(f"        {C.RED}Oncology{C.R} (GNN-style Random Forest)...", end="  ")
    onc = OncologyHead(n_estimators=300); onc.fit(X_onc_tr, onc_y(df_train))
    onc_proba = onc.predict_proba(X_onc_te)
    onc_pred  = onc.predict(X_onc_te)
    score_onc = float(onc_proba["AI_Oncology_Score"].iloc[0])
    tier_pred = str(onc_pred.iloc[0])
    w(f"Score: {C.B}{score_onc:5.1f}%{C.R}  Tier: "
      + (C.RED + C.B if "High" in tier_pred else "")
      + tier_pred + C.R)

    time.sleep(0.15 if fast else 0.4)

    # ── [3e] Ensemble ──────────────────────────────────────────────────────────
    w(f"  {C.CYN}[3e]{C.R} Integrated Clinical Risk Engine:")
    engine   = EnsembleRiskEngine()
    ensemble = engine.compute(amr_proba, bio_proba, onc_proba)
    n_active = int(ensemble["N_Active_Domains"].iloc[0])
    co_bonus = float(ensemble["Co_occurrence_Bonus"].iloc[0])
    ens_score= float(ensemble["Ensemble_Risk_Score"].iloc[0])
    ens_tier = str(ensemble["Ensemble_Risk_Tier"].iloc[0])

    tier_col = (C.RED if ens_tier == "Critical" else
                C.YEL if ens_tier == "High"     else C.GRN)
    w(f"        0.40 × {score_amr:.1f}  +  0.35 × {score_bio:.1f}  "
      f"+  0.25 × {score_onc:.1f}  +  {co_bonus:.0f} (co-occur)  =  "
      f"{C.B}{ens_score:.1f}{C.R}")
    w(f"        Risk tier: {tier_col}{C.B}{ens_tier}{C.R}  "
      f"|  Active domains: {n_active}/3  "
      f"|  Co-occurrence bonus: +{co_bonus:.0f}")
    w()

    return {
        "score_amr":  score_amr, "score_bio": score_bio, "score_onc": score_onc,
        "ens_score":  ens_score, "ens_tier":  ens_tier,
        "n_active":   n_active,  "co_bonus":  co_bonus,
        "genes_pos":  genes_pos, "stage_pred":stage_pred, "tier_pred": tier_pred,
        "amr_proba":  amr_proba, "bio_proba": bio_proba, "onc_proba": onc_proba,
        "calib":      calib,
    }


# =============================================================================
# Phase 6 — Report
# =============================================================================
def phase_report(sc, feature_df, ai_out, feat_summary, fast):
    w(f"  {C.B}━━━  PHASE 5: CLINICAL REPORT GENERATION  ━━━{C.R}")
    w()

    from pipeline.clinical_report import ClinicalReportGenerator

    # Build a row dict with all needed fields
    row_d = feature_df.iloc[0].to_dict()

    # Inject AI output fields
    row_d["AI_AMR_Score"]         = ai_out["score_amr"]
    row_d["AI_Biofilm_Score"]     = ai_out["score_bio"]
    row_d["AI_Oncology_Score"]    = ai_out["score_onc"]
    row_d["Ensemble_Risk_Score"]  = ai_out["ens_score"]
    row_d["Ensemble_Risk_Tier"]   = ai_out["ens_tier"]
    row_d["N_Active_Domains"]     = ai_out["n_active"]
    row_d["Co_occurrence_Bonus"]  = ai_out["co_bonus"]
    row_d["MC_AMR_Conf_pct"]      = float(ai_out["calib"]["MC_AMR_conf"][0])
    row_d["MC_Biofilm_Conf_pct"]  = float(ai_out["calib"]["MC_Biofilm_conf"][0])
    row_d["MC_Oncology_Conf_pct"] = float(ai_out["calib"]["MC_Oncology_conf"][0])

    # AMR clinical fields
    genes = ai_out["genes_pos"]
    ANTIBIOTIC_MAP = {
        "blaNDM1":["Carbapenems","Penicillins"], "mecA":["Methicillin","Oxacillin"],
        "vanA":["Vancomycin","Teicoplanin"],     "KPC":["Carbapenems","Aztreonam"],
    }
    ALT_MAP = {
        "blaNDM1":"Colistin / Tigecycline", "mecA":"Vancomycin / Linezolid",
        "vanA":"Daptomycin / Linezolid",   "KPC":"Ceftazidime-avibactam",
    }
    fail = list(set(ab for g in genes for ab in ANTIBIOTIC_MAP.get(g, [])))
    alts = list(set(ALT_MAP[g] for g in genes if g in ALT_MAP))
    profile = ("Pan-resistant" if len(genes)>=3 else
               "Multi-drug_resistant" if len(genes)==2 else
               "Single_resistance" if len(genes)==1 else "Susceptible")

    row_d["AMR_Resistance_Profile"]  = profile
    row_d["AMR_Detected_Genes"]      = "|".join(genes) if genes else "None"
    row_d["AMR_Failed_Antibiotics"]  = "|".join(fail) if fail else "None"
    row_d["AMR_Recommended_Alt"]     = "|".join(alts) if alts else "None"
    row_d["AMR_N_Genes_Detected"]    = len(genes)

    # Biofilm fields from features
    n_bf_det = sum(1 for ch in ["Ch05","Ch06","Ch07","Ch08"]
                   if row_d.get(f"{ch}_detected", 0))
    bf_stages= ["None","Stage_I_early_attachment","Stage_II_microcolony",
                "Stage_III_maturation","Stage_IV_dispersion"]
    row_d["Biofilm_Stage"]        = bf_stages[min(n_bf_det, 4)]
    row_d["icaADBC_Active"]       = int(row_d.get("Ch05_detected", 0))
    row_d["pel_psl_Active"]       = int(row_d.get("Ch07_detected", 0))
    row_d["Quorum_Sensing_Active"]= int(row_d.get("Ch06_detected", 0))
    ch06_d = row_d.get("Ch06_drop_pct", 0)
    ch08_c = row_d.get("Ch08_conc_pM", 0)
    row_d["QS_Activity_Score"]    = round(ch06_d*0.6 + ch08_c/10*0.4, 1)
    row_d["AI2_Level_uM"]         = round(float(np.random.uniform(0.5,8.0)),3) \
                                    if row_d.get("Ch06_detected") else 0.1
    row_d["c_diGMP_Level_uM"]     = round(float(np.random.uniform(0.5,5.0)),3) \
                                    if row_d.get("Ch08_detected") else 0.05

    # Oncology fields
    onc_det = [(ch, CHANNEL_META[ch][1],
                CHANNEL_META[ch][0])
               for ch in ["Ch09","Ch10","Ch11","Ch12"]
               if row_d.get(f"{ch}_detected",0)]
    tier_pred = ai_out["tier_pred"]
    row_d["Oncology_Risk_Tier"]       = tier_pred
    row_d["Cancer_Microbes_Detected"] = "|".join(m for _,m,_ in onc_det) or "None"
    row_d["Oncology_Species"]         = "|".join({
        "FadA":"F. nucleatum","CagA":"H. pylori",
        "pks":"E. coli genotoxin","miRNA-21":"epigenetic"}.get(m,"") for _,m,_ in onc_det) or "None"
    row_d["Chronic_Inflammation_Load"]= round(float(np.random.uniform(4,9)),2) \
                                        if onc_det else round(float(np.random.uniform(0,2)),2)
    row_d["Referral_Recommended"]     = int(ai_out["score_onc"] >= 50)
    row_d["Transmission_Mode"]        = "BLE"
    row_d["Total_Time_min"]           = round(np.random.uniform(22,28),1)

    # Generate HTML report
    reports_dir = BASE / "reports"
    reports_dir.mkdir(exist_ok=True)
    sid  = sc["patient_id"]
    path = reports_dir / f"report_{sid}.html"

    gen = ClinicalReportGenerator()
    gen.generate_single(row_d, shap_row={}, out_path=str(path))

    time.sleep(0.2 if fast else 0.5)
    w(f"  {C.GRN}{C.B}✓  HTML report written:{C.R}  {path}")
    w()

    # ── Terminal summary panel ─────────────────────────────────────────────────
    tier_col = (C.RED if ai_out["ens_tier"]=="Critical" else
                C.YEL if ai_out["ens_tier"]=="High" else C.GRN)

    w(C.B + "  ╔═══════════════════════════════════════════════════════════╗" + C.R)
    w(C.B + "  ║          CLINICAL DECISION SUPPORT — SUMMARY             ║" + C.R)
    w(C.B + "  ╠════════════════╦════════════╦══════════════════════════╣" + C.R)

    # AMR line
    gene_str = ", ".join(genes) if genes else "None detected"
    alt_str  = alts[0] if alts else "Standard empirics"
    w(f"  {C.B}║{C.R} {C.BLU}AMR Risk{C.R}       "
      f"{C.B}║{C.R} {ai_out['score_amr']:5.1f}%     "
      f"{C.B}║{C.R} {profile:<26}{C.B}║{C.R}")
    w(f"  {C.B}║{C.R}{' ':<16}{C.B}║{C.R}            "
      f"{C.B}║{C.R} Genes: {gene_str:<20}{C.B}║{C.R}")
    w(f"  {C.B}║{C.R}{' ':<16}{C.B}║{C.R}            "
      f"{C.B}║{C.R} → {alt_str:<24}{C.B}║{C.R}")
    w(C.B + "  ╠════════════════╬════════════╬══════════════════════════╣" + C.R)

    # Biofilm line
    bf_label= row_d["Biofilm_Stage"].replace("_"," ")
    qs_score= row_d["QS_Activity_Score"]
    w(f"  {C.B}║{C.R} {C.MAG}Biofilm{C.R}        "
      f"{C.B}║{C.R} {ai_out['score_bio']:5.1f}%     "
      f"{C.B}║{C.R} {bf_label:<26}{C.B}║{C.R}")
    w(f"  {C.B}║{C.R}{' ':<16}{C.B}║{C.R}            "
      f"{C.B}║{C.R} QS Activity Score: {qs_score:<7.1f}{C.B}║{C.R}")
    w(f"  {C.B}║{C.R}{' ':<16}{C.B}║{C.R}            "
      f"{C.B}║{C.R} c-di-GMP: {row_d['c_diGMP_Level_uM']:.2f} µM{' ':11}{C.B}║{C.R}")
    w(C.B + "  ╠════════════════╬════════════╬══════════════════════════╣" + C.R)

    # Oncology line
    sp_str = row_d["Oncology_Species"][:26] if row_d["Oncology_Species"] != "None" else "None"
    ref_str= "⚠ REFERRAL RECOMMENDED   " if row_d["Referral_Recommended"] else "No referral needed        "
    w(f"  {C.B}║{C.R} {C.RED}Oncology{C.R}       "
      f"{C.B}║{C.R} {ai_out['score_onc']:5.1f}%     "
      f"{C.B}║{C.R} {tier_pred:<26}{C.B}║{C.R}")
    w(f"  {C.B}║{C.R}{' ':<16}{C.B}║{C.R}            "
      f"{C.B}║{C.R} {sp_str:<26}{C.B}║{C.R}")
    w(f"  {C.B}║{C.R}{' ':<16}{C.B}║{C.R}            "
      f"{C.B}║{C.R} {C.RED if row_d['Referral_Recommended'] else ''}"
      f"{ref_str}{C.R}{C.B}║{C.R}")
    w(C.B + "  ╠════════════════╬════════════╬══════════════════════════╣" + C.R)

    # Ensemble
    w(f"  {C.B}║ ENSEMBLE RISK  ║ "
      f"{tier_col}{C.B}{ai_out['ens_score']:5.1f}%{C.R}     "
      f"{C.B}║{C.R} {tier_col}{C.B}{ai_out['ens_tier']:<26}{C.R}{C.B}║{C.R}")
    w(f"  {C.B}║{C.R}{' ':<16}{C.B}║{C.R}            "
      f"{C.B}║{C.R} Active domains: {ai_out['n_active']}/3  "
      f"Co-bonus: +{ai_out['co_bonus']:.0f}{' ':3}{C.B}║{C.R}")
    w(C.B + "  ╚════════════════╩════════════╩══════════════════════════╝" + C.R)
    w()
    w(f"  {C.D}Open in browser:  start {path}{C.R}")
    w()


# =============================================================================
# Entry point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="AI Biosensor Live Simulation")
    parser.add_argument("--fast",     action="store_true",
                        help="Skip delays for quick demo")
    parser.add_argument("--scenario", type=int, default=1, choices=[1,2,3],
                        help="Patient scenario (1=ICU/Pan, 2=AMR, 3=Biofilm)")
    args = parser.parse_args()

    sc   = SCENARIOS[args.scenario]
    rng  = np.random.default_rng(42)
    fast = args.fast

    t0 = time.time()

    phase_boot(sc, fast)
    phase_baseline(sc, rng, fast)
    signals_nA, signals_uA, ch_smooth, ch_drop, ch_det = \
        phase_acquisition(sc, rng, fast)
    feature_df, feat_summary = \
        phase_features(sc, signals_nA, ch_smooth, ch_drop, ch_det, fast)
    ai_out = phase_ai(feature_df, feat_summary, sc, fast)
    phase_report(sc, feature_df, ai_out, feat_summary, fast)

    elapsed = time.time() - t0
    w(f"  {C.D}Total simulation time: {elapsed:.1f}s{C.R}")
    w()


if __name__ == "__main__":
    main()
