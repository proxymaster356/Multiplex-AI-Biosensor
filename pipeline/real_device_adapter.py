"""
pipeline/real_device_adapter.py
================================
Real Biosensor Signal Adapter

Bridges between REAL hardware output and the AI pipeline.

What a real biosensor sends
----------------------------
A potentiostat (e.g. Palmsens4, EmStat Pico, custom ASIC) connected to the
12-channel electrode array samples current continuously:

  - Resolution : 1–100 pA (picoampere)
  - Sample rate: 1–10 Hz (1 reading per second is common for amperometry)
  - Duration   : 15–30 minutes per assay = 900–1800 readings per channel
  - Unit sent  : µA (microampere) or raw ADC integer (12/16-bit)
  - Format     : serial UART, USB-CDC, BLE GATT, or SD-card CSV

  Example raw output (one row per second, 12 channels):
    time_s, Ch01_uA, Ch02_uA, ..., Ch12_uA
    0,      0.01001, 0.00998, ..., 0.01003    ← baseline period
    1,      0.01002, 0.00997, ..., 0.01001
    ...
    60,     0.00921, 0.00998, ..., 0.01002    ← Ch01 dropping (target binding)
    ...
    900,    0.00503, 0.00995, ..., 0.00821    ← Ch01 and Ch12 at plateau

This adapter:
  1. Reads the raw time-series CSV (or receives the streaming array)
  2. Converts units  (µA → nA,  or ADC_counts → nA via calibration)
  3. Applies Kalman filter across the FULL time-series (not just one point)
  4. Detects baseline, injection point, and plateau
  5. Extracts 8 features per channel (same features the AI pipeline expects)
  6. Returns a single-row DataFrame compatible with run.py / pipeline/

Usage (from a real device CSV):
    from pipeline.real_device_adapter import RealDeviceAdapter
    adapter = RealDeviceAdapter()
    feature_row = adapter.process_file("device_output_2024_01_15.csv",
                                       sample_id="PAT-001",
                                       patient_meta={"age":45, "sex":"F", ...})
    # Then pass to run.py style inference:
    import run
    run.run_inference_single(feature_row)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from datetime import datetime

# ── Constants ────────────────────────────────────────────────────────────────
CHANNELS      = [f"Ch{i:02d}" for i in range(1, 13)]
BASELINE_nA   = 10.0          # expected healthy baseline (~10 nA)
THRESHOLD_DROP= 0.30          # 30% drop = biomarker detected
BASELINE_WINDOW_S = 60        # first 60 seconds = baseline period
PLATEAU_WINDOW_S  = 60        # last 60 seconds = plateau (steady state)


# =============================================================================
# Unit conversion  (real device → nA)
# =============================================================================
class UnitConverter:
    """
    Converts raw hardware output to nanoampere (nA).

    Supported input types:
      'uA'       : microampere  (most potentiostats, e.g. PalmSens)
      'nA'       : nanoampere   (already correct, pass-through)
      'pA'       : picoampere   (high-sensitivity instruments)
      'mV'       : millivolt    (when reading voltage across shunt resistor)
      'uV'       : microvolt    (high-resolution shunt measurement)
      'adc_12bit': raw 12-bit ADC integer (0–4095)
      'adc_16bit': raw 16-bit ADC integer (0–65535)

    For ADC:
      nA = (adc_count / adc_max) × Vref / R_shunt × unit_scale
      Default: Vref=3.3V, R_shunt=100 kΩ → full-scale ≈ 33 µA
    """
    CONVERSIONS = {
        "uA"       : 1e3,     # µA × 1000 = nA
        "nA"       : 1.0,     # already nA
        "pA"       : 1e-3,    # pA / 1000 = nA
        "mV"       : None,    # needs shunt resistor
        "uV"       : None,    # needs shunt resistor
        "adc_12bit": None,    # needs ADC calibration
        "adc_16bit": None,    # needs ADC calibration
    }

    def __init__(self, unit: str = "uA",
                 shunt_ohm: float = 100_000,    # 100 kΩ shunt
                 vref_V: float = 3.3,           # 3.3 V reference
                 adc_bits: int = 12,
                 baseline_offset_nA: float = 0.0):
        self.unit             = unit.lower()
        self.shunt_ohm        = shunt_ohm
        self.vref_V           = vref_V
        self.adc_bits         = adc_bits
        self.baseline_offset  = baseline_offset_nA

    def convert(self, raw: np.ndarray) -> np.ndarray:
        """Convert raw hardware values to nA."""
        if self.unit == "ua":
            nA = raw * 1e3
        elif self.unit == "na":
            nA = raw.copy()
        elif self.unit == "pa":
            nA = raw * 1e-3
        elif self.unit in ("mv", "uv"):
            # V = raw (convert to volts first)
            scale = 1e-3 if self.unit == "mv" else 1e-6
            V_volts = raw * scale
            I_amps  = V_volts / self.shunt_ohm
            nA      = I_amps * 1e9
        elif self.unit in ("adc_12bit", "adc_16bit"):
            adc_max = (2 ** self.adc_bits) - 1
            V_volts = (raw / adc_max) * self.vref_V
            I_amps  = V_volts / self.shunt_ohm
            nA      = I_amps * 1e9
        else:
            raise ValueError(f"Unknown unit: {self.unit}. "
                             f"Supported: {list(self.CONVERSIONS.keys())}")

        return nA - self.baseline_offset


# =============================================================================
# Kalman filter over full time-series  (higher quality than single-step)
# =============================================================================
class KalmanFilterTimeSeries:
    """
    Full Kalman filter applied across the 900-1800 reading time-series.
    Automatically adapts to fast signal changes (target binding) vs. noise.

    Uses adaptive Q: increases process noise when signal is changing fast
    (during the binding reaction phase) to track the real signal.
    """

    def __init__(self, Q_base: float = 0.005,
                 Q_dynamic: float = 0.05,
                 R: float = 0.0225,          # measurement noise var (0.15 nA std)²
                 change_threshold: float = 0.05):
        self.Q_base  = Q_base
        self.Q_dyn   = Q_dynamic
        self.R       = R
        self.thresh  = change_threshold

    def filter(self, measurements: np.ndarray) -> np.ndarray:
        """
        measurements: (T,) array of nA readings
        returns:      (T,) smoothed nA readings
        """
        T        = len(measurements)
        smoothed = np.zeros(T)
        x        = measurements[0]   # initial state estimate
        P        = 1.0               # initial covariance

        for t in range(T):
            z  = measurements[t]
            # Adaptive Q: larger when signal is changing rapidly
            dz = abs(z - x)
            Q  = self.Q_dyn if dz > self.thresh else self.Q_base

            # Predict
            P_pred = P + Q

            # Update
            K    = P_pred / (P_pred + self.R)
            x    = x + K * (z - x)
            P    = (1 - K) * P_pred

            smoothed[t] = x

        return smoothed


# =============================================================================
# Feature extractor from time-series
# =============================================================================
@dataclass
class ChannelTimeSeriesResult:
    """All extracted features for one channel from a full time-series assay."""
    channel:              str
    baseline_mean_nA:     float   # mean of first 60 s (pre-injection)
    baseline_std_nA:      float   # noise level of baseline
    plateau_mean_nA:      float   # mean of last 60 s (steady state)
    raw_nA:               float   # = plateau_mean (equivalent to generator's raw_nA)
    smooth_nA:            float   # Kalman-smoothed plateau
    peak_amplitude_nA:    float   # baseline − plateau (Signal-OFF drop)
    signal_drop_pct:      float   # % drop from baseline
    time_to_threshold_s:  float   # seconds to hit 30% drop (or >1200 if not hit)
    impedance_change_pct: float   # estimated from conductance change
    snr_db:               float   # signal power / noise floor
    conc_pM:              float   # back-calculated concentration (inverse log model)
    detected:             bool    # True if drop > 30%
    kinetic_slope:        float   # nA/s during binding phase (binding rate)
    auc_nA_s:             float   # area under response curve (integral)


def extract_features_from_timeseries(
        ch: str,
        time_s: np.ndarray,
        raw_nA: np.ndarray,
        smooth_nA: np.ndarray,
        sample_rate_hz: float = 1.0) -> ChannelTimeSeriesResult:
    """
    Extracts all 8 pipeline features + extras from a full time-series.

    time_s   : (T,) array of timestamps in seconds
    raw_nA   : (T,) raw current readings after unit conversion
    smooth_nA: (T,) Kalman-smoothed readings
    """
    T              = len(time_s)
    baseline_pts   = int(BASELINE_WINDOW_S * sample_rate_hz)
    plateau_pts    = int(PLATEAU_WINDOW_S  * sample_rate_hz)

    # ── Baseline (pre-injection, first BASELINE_WINDOW_S seconds) ──────────
    bl_raw    = raw_nA[:min(baseline_pts, T)]
    bl_mean   = float(bl_raw.mean())
    bl_std    = float(bl_raw.std())

    # Normalise to expected ~10 nA baseline (device calibration offset)
    offset    = BASELINE_nA - bl_mean
    raw_nA    = raw_nA + offset
    smooth_nA = smooth_nA + offset
    bl_mean  += offset

    # ── Plateau (last PLATEAU_WINDOW_S seconds) ─────────────────────────────
    pl_raw    = raw_nA[-min(plateau_pts, T):]
    pl_mean   = float(pl_raw.mean())
    pl_smooth = float(smooth_nA[-min(plateau_pts, T):].mean())

    # ── Core features ────────────────────────────────────────────────────────
    peak_amp  = max(0.0, bl_mean - pl_smooth)
    drop_pct  = round(peak_amp / bl_mean * 100, 2) if bl_mean > 0 else 0.0
    detected  = drop_pct >= (THRESHOLD_DROP * 100)

    # ── Time to threshold (t2t) ──────────────────────────────────────────────
    threshold_nA = bl_mean * (1 - THRESHOLD_DROP)
    below        = np.where(smooth_nA <= threshold_nA)[0]
    t2t_s        = float(time_s[below[0]]) if len(below) > 0 else 1200.0

    # ── Impedance change estimate ─────────────────────────────────────────────
    # ΔZ/Z ≈ ΔI/I × (geometric factor ~2.2)  from probe binding layer theory
    delta_I_over_I    = peak_amp / max(bl_mean, 1e-6)
    impedance_chg_pct = round(float(np.clip(delta_I_over_I * 2.2 * 100, 0, 20)), 3)

    # ── SNR ──────────────────────────────────────────────────────────────────
    signal_power = peak_amp ** 2
    noise_power  = max(bl_std ** 2, 1e-9)
    snr_db       = round(10 * np.log10(signal_power / noise_power), 2)

    # ── Concentration back-calculation (inverse Signal-OFF log model) ─────────
    # I = 10 - 2.1 × log₁₀(C_pM)  →  C_pM = 10^((10 - I) / 2.1)
    if detected and pl_smooth > 0.5:
        conc_pM = round(float(10 ** ((BASELINE_nA - pl_smooth) / 2.1)), 2)
    else:
        conc_pM = 0.0

    # ── Kinetic slope (binding rate) ──────────────────────────────────────────
    # Slope during the main response window (60s to t2t or midpoint)
    mid_start = min(baseline_pts, T - 1)
    mid_end   = min(int(t2t_s * sample_rate_hz) + 60, T - 1)
    if mid_end > mid_start + 1:
        dy = smooth_nA[mid_end] - smooth_nA[mid_start]
        dt = time_s[mid_end]   - time_s[mid_start]
        kinetic_slope = round(float(dy / max(dt, 1e-6)), 5)
    else:
        kinetic_slope = 0.0

    # ── Area under response curve ─────────────────────────────────────────────
    # AUC relative to baseline (negative = below baseline = positive signal)
    _trapz   = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    auc_nA_s = round(float(_trapz(bl_mean - smooth_nA, time_s)), 2)

    return ChannelTimeSeriesResult(
        channel              = ch,
        baseline_mean_nA     = round(bl_mean, 4),
        baseline_std_nA      = round(bl_std, 4),
        plateau_mean_nA      = round(pl_mean, 4),
        raw_nA               = round(pl_mean, 3),
        smooth_nA            = round(pl_smooth, 3),
        peak_amplitude_nA    = round(peak_amp, 3),
        signal_drop_pct      = drop_pct,
        time_to_threshold_s  = round(t2t_s, 1),
        impedance_change_pct = impedance_chg_pct,
        snr_db               = snr_db,
        conc_pM              = conc_pM,
        detected             = detected,
        kinetic_slope        = kinetic_slope,
        auc_nA_s             = auc_nA_s,
    )


# =============================================================================
# Device CSV parser
# =============================================================================
def parse_device_csv(filepath: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Parse a real device output CSV.

    Expected format:
      time_s, Ch01_uA, Ch02_uA, ..., Ch12_uA   (header row required)
      0,      0.01001, 0.00998, ..., 0.01003
      1,      0.01002, 0.00997, ..., 0.01001
      ...

    OR (alternative, time as index):
      time_s, Ch01, Ch02, ..., Ch12

    Returns:
      time_s : (T,) array of timestamps
      df     : DataFrame with columns Ch01..Ch12 (raw hardware values)
    """
    df = pd.read_csv(filepath)

    # Find time column
    time_col = None
    for c in ["time_s", "time", "Time_s", "Time", "t", "timestamp"]:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        # Assume first column is time
        time_col = df.columns[0]

    time_s = df[time_col].values.astype(float)

    # Find channel columns
    ch_cols = {}
    for ch in CHANNELS:
        # Try exact match, then partial match (e.g. "Ch01_uA", "ch01", "CH01")
        for col in df.columns:
            if col.upper().replace("_UA","").replace("_NA","").replace("_MV","") \
               == ch.upper():
                ch_cols[ch] = col
                break

    if not ch_cols:
        raise ValueError(
            f"Could not find channel columns in {filepath}.\n"
            f"Expected columns like: Ch01, Ch01_uA, Ch01_nA, ch01, ...\n"
            f"Found: {list(df.columns)}")

    ch_df = pd.DataFrame({ch: df[col].values for ch, col in ch_cols.items()})
    return time_s, ch_df


# =============================================================================
# Main adapter class
# =============================================================================
class RealDeviceAdapter:
    """
    Full adapter: raw device output → AI pipeline feature row.

    Accepts:
      - A CSV file from the device (parse_device_csv format)
      - A numpy array (time_s, channels_array)
      - A streaming callback (real-time via BLE/serial)

    Converts units, applies Kalman filter, extracts features.
    Returns a single-row DataFrame compatible with run.py.

    Example:
        adapter = RealDeviceAdapter(unit="uA")
        row_df  = adapter.process_file(
            "device_2024_01_15.csv",
            sample_id    = "BIO-REAL-001",
            patient_meta = {"Patient_Age": 54, "Patient_Sex": "M",
                            "Sample_Type": "Blood", "Collection_Site": "ICU"})
        # Now run through the AI pipeline:
        import run
        run.run_inference_row(row_df)   # see run.py for this function
    """

    def __init__(self, unit: str = "uA",
                 shunt_ohm: float = 100_000,
                 vref_V: float = 3.3,
                 adc_bits: int = 12,
                 sample_rate_hz: float = 1.0):
        self.converter    = UnitConverter(unit, shunt_ohm, vref_V, adc_bits)
        self.kf           = KalmanFilterTimeSeries()
        self.sample_rate  = sample_rate_hz

    # ── From CSV file ─────────────────────────────────────────────────────────
    def process_file(self, filepath: str,
                     sample_id: str           = "REAL-001",
                     patient_meta: dict       = None,
                     device_meta: dict        = None) -> pd.DataFrame:
        """
        Process a complete device CSV file.
        Returns a one-row DataFrame with all 96+ feature columns expected by run.py.
        """
        time_s, ch_df = parse_device_csv(filepath)
        return self._build_feature_row(time_s, ch_df,
                                       sample_id, patient_meta, device_meta)

    # ── From numpy arrays (streaming / real-time) ─────────────────────────────
    def process_array(self, time_s: np.ndarray,
                      channels: np.ndarray,
                      sample_id: str      = "REAL-001",
                      patient_meta: dict  = None,
                      device_meta: dict   = None) -> pd.DataFrame:
        """
        Process numpy arrays directly.
        channels : (T, 12) array of raw hardware values
        """
        ch_df = pd.DataFrame(channels, columns=CHANNELS)
        return self._build_feature_row(time_s, ch_df,
                                       sample_id, patient_meta, device_meta)

    # ── Core processing ───────────────────────────────────────────────────────
    def _build_feature_row(self, time_s, ch_df,
                           sample_id, patient_meta, device_meta) -> pd.DataFrame:
        now  = datetime.now()
        row  = {
            "Sample_ID":        sample_id,
            "Collection_Date":  now.strftime("%Y-%m-%d"),
            "Collection_Time":  now.strftime("%H:%M"),
            "Patient_Age":      patient_meta.get("Patient_Age", 0) if patient_meta else 0,
            "Patient_Sex":      patient_meta.get("Patient_Sex", "U") if patient_meta else "U",
            "Patient_BMI":      patient_meta.get("Patient_BMI", 0) if patient_meta else 0,
            "Sample_Type":      patient_meta.get("Sample_Type", "Unknown") if patient_meta else "Unknown",
            "Collection_Site":  patient_meta.get("Collection_Site", "Unknown") if patient_meta else "Unknown",
            "Sample_Volume_uL": patient_meta.get("Sample_Volume_uL", 5.0) if patient_meta else 5.0,
            "Cartridge_Lot":    device_meta.get("Cartridge_Lot", "REAL") if device_meta else "REAL",
            "Device_ID":        device_meta.get("Device_ID", "DEV-REAL") if device_meta else "DEV-REAL",
            "Ambient_Temp_C":   device_meta.get("Ambient_Temp_C", 25.0) if device_meta else 25.0,
            "Ambient_pH":       device_meta.get("Ambient_pH", 7.2) if device_meta else 7.2,
            "Clinical_Category":"REAL",
            "Diagnostic_Label": "REAL",
        }

        results: Dict[str, ChannelTimeSeriesResult] = {}

        for ch in CHANNELS:
            if ch not in ch_df.columns:
                continue

            raw_hw  = ch_df[ch].values.astype(float)
            # Unit conversion: hardware units → nA
            raw_nA  = self.converter.convert(raw_hw)
            # Kalman filter over full time-series
            sm_nA   = self.kf.filter(raw_nA)
            # Feature extraction
            result  = extract_features_from_timeseries(
                ch, time_s, raw_nA, sm_nA, self.sample_rate)
            results[ch] = result

            # Write features into row (same keys as biosensor_detailed_500.csv)
            row[f"{ch}_raw_nA"]           = result.raw_nA
            row[f"{ch}_smooth_nA"]        = result.smooth_nA
            row[f"{ch}_conc_pM"]          = result.conc_pM
            row[f"{ch}_peak_amp_nA"]      = result.peak_amplitude_nA
            row[f"{ch}_drop_pct"]         = result.signal_drop_pct
            row[f"{ch}_t2t_s"]           = result.time_to_threshold_s
            row[f"{ch}_impedance_pct"]   = result.impedance_change_pct
            row[f"{ch}_snr_db"]          = result.snr_db
            # Bonus real-device features
            row[f"{ch}_baseline_nA"]     = result.baseline_mean_nA
            row[f"{ch}_baseline_std"]    = result.baseline_std_nA
            row[f"{ch}_kinetic_slope"]   = result.kinetic_slope
            row[f"{ch}_auc_nA_s"]        = result.auc_nA_s
            row[f"{ch}_detected"]        = int(result.detected)

        # QS proxy from feature values
        ch06_drop = row.get("Ch06_drop_pct", 0)
        ch08_conc = row.get("Ch08_conc_pM",  0)
        row["QS_Activity_Score"] = round(
            float(np.clip(ch06_drop * 0.6 + ch08_conc / 10 * 0.4, 0, 100)), 1)

        return pd.DataFrame([row])

    # ── Signal quality report ─────────────────────────────────────────────────
    def signal_quality_report(self, feature_row: pd.DataFrame) -> str:
        """Print a human-readable signal quality summary."""
        lines = [
            "=" * 60,
            f"  Signal Quality Report — {feature_row['Sample_ID'].iloc[0]}",
            "=" * 60,
            f"  {'Channel':<8}  {'Domain':<10}  {'Drop%':>6}  "
            f"{'SNR(dB)':>8}  {'t2t(s)':>7}  {'Conc(pM)':>9}  {'Status'}",
            "-" * 60,
        ]
        domains = {
            **{f"Ch{i:02d}":"AMR"      for i in range(1,5)},
            **{f"Ch{i:02d}":"Biofilm"  for i in range(5,9)},
            **{f"Ch{i:02d}":"Oncology" for i in range(9,13)},
        }
        for ch in CHANNELS:
            drop  = feature_row.get(f"{ch}_drop_pct",   [0]).iloc[0]
            snr   = feature_row.get(f"{ch}_snr_db",     [0]).iloc[0]
            t2t   = feature_row.get(f"{ch}_t2t_s",      [9999]).iloc[0]
            conc  = feature_row.get(f"{ch}_conc_pM",    [0]).iloc[0]
            det   = feature_row.get(f"{ch}_detected",   [0]).iloc[0]
            status= "DETECTED" if det else "not detected"
            lines.append(
                f"  {ch:<8}  {domains[ch]:<10}  {drop:>6.1f}  "
                f"{snr:>8.1f}  {t2t:>7.0f}  {conc:>9.1f}  {status}")
        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# Synthetic real-device data generator  (for testing the adapter)
# =============================================================================
def generate_sample_device_csv(filepath: str = "sample_device_input.csv",
                                duration_s: int = 900,
                                sample_rate_hz: float = 1.0,
                                positive_channels: list = None,
                                unit: str = "uA",
                                seed: int = 42):
    """
    Generates a realistic device output CSV to test the adapter.

    Simulates:
      - Stable baseline for first 60 s
      - Logarithmic decay (Signal-OFF) for positive channels starting at 60 s
      - Gaussian noise (σ = 0.15 nA = 0.00015 µA)
      - Plateau after ~600 s

    positive_channels: list like ["Ch01", "Ch05", "Ch09"] — which channels bind
    unit             : unit of CSV output ("uA", "nA", "pA")

    Output CSV format (ready for RealDeviceAdapter):
      time_s, Ch01_uA, Ch02_uA, ..., Ch12_uA
    """
    rng = np.random.default_rng(seed)
    if positive_channels is None:
        positive_channels = ["Ch01", "Ch05", "Ch09"]

    t = np.arange(0, duration_s, 1.0 / sample_rate_hz)
    T = len(t)

    BASELINE_uA = BASELINE_nA / 1e3   # 10 nA → 0.010 µA

    data = {"time_s": t}
    for ch in CHANNELS:
        noise_uA = rng.normal(0, 0.00015, T)   # 0.15 nA noise in µA

        if ch in positive_channels:
            # Randomly pick a concentration
            conc_pM = rng.uniform(20, 400)
            # Signal-OFF plateau current (nA)
            plateau_nA = 10.0 - 2.1 * np.log10(conc_pM)
            plateau_uA = plateau_nA / 1e3

            # Sigmoid-like binding curve
            # - Baseline phase: 0–60 s
            # - Binding phase : 60–600 s (sigmoid approach to plateau)
            # - Plateau phase : 600–900 s
            signal_uA = np.where(
                t < 60,
                BASELINE_uA,                                     # baseline
                np.where(
                    t < 600,
                    plateau_uA + (BASELINE_uA - plateau_uA) *   # sigmoid decay
                        np.exp(-0.008 * (t - 60)),
                    plateau_uA                                    # plateau
                )
            )
        else:
            # Healthy channel: stays at baseline with minor drift
            drift = rng.uniform(-0.0001, 0.0001) * (t / duration_s)
            signal_uA = BASELINE_uA + drift

        # Apply noise
        signal_uA = signal_uA + noise_uA

        # Convert to requested unit for output
        if unit == "uA":
            data[f"{ch}"] = signal_uA
        elif unit == "nA":
            data[f"{ch}"] = signal_uA * 1e3
        elif unit == "pA":
            data[f"{ch}"] = signal_uA * 1e6

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False, float_format="%.7f")
    print(f"[OK] Sample device CSV written: {filepath}")
    print(f"     Duration: {duration_s}s | Rate: {sample_rate_hz} Hz | "
          f"{int(T)} readings | Unit: {unit}")
    print(f"     Positive channels: {positive_channels}")
    return filepath


# =============================================================================
# Quick test / demo
# =============================================================================
if __name__ == "__main__":
    import sys

    print("\n" + "="*60)
    print("  Real Device Adapter — Demo")
    print("="*60)

    # 1. Generate a fake device CSV (simulates a real sensor)
    csv_path = generate_sample_device_csv(
        filepath          = "sample_device_input.csv",
        duration_s        = 900,
        sample_rate_hz    = 1.0,
        positive_channels = ["Ch02", "Ch05", "Ch08", "Ch09"],
        unit              = "uA",
        seed              = 99,
    )

    # 2. Run the adapter
    adapter = RealDeviceAdapter(unit="uA", sample_rate_hz=1.0)
    feature_row = adapter.process_file(
        csv_path,
        sample_id    = "REAL-PAT-001",
        patient_meta = {
            "Patient_Age":      67,
            "Patient_Sex":      "M",
            "Patient_BMI":      28.3,
            "Sample_Type":      "Blood",
            "Collection_Site":  "ICU",
            "Sample_Volume_uL": 5.2,
        },
        device_meta  = {
            "Device_ID":       "DEV-B42",
            "Cartridge_Lot":   "LOT-2024-001",
            "Ambient_Temp_C":  24.5,
            "Ambient_pH":      7.3,
        }
    )

    # 3. Print signal quality report
    print(adapter.signal_quality_report(feature_row))

    # 4. Show key features extracted
    print("\n  Features extracted (key columns):")
    key_cols = [c for c in feature_row.columns if "_drop_pct" in c or "_detected" in c]
    print(feature_row[key_cols].T.to_string())

    print("\n  Feature row shape:", feature_row.shape)
    print("\n  [NEXT] Pass feature_row to run.py for AI inference:")
    print("         import run")
    print("         run.run_inference_row(feature_row)")
