"""
pipeline/signal_processor.py
==============================
SVG Layer 2 -- Signal Transduction & Preprocessing

Implements:
  1. Electrochemical readout   -- load raw nA values from channels
  2. Noise filtering           -- Kalman filter on each channel
  3. Feature extraction        -- peak amplitude, signal drop, time-to-threshold,
                                  impedance change, SNR (dB), area-under-curve

Input:  pandas DataFrame with columns like Ch01_raw_nA, Ch02_raw_nA, ...
Output: pandas DataFrame with all raw + smoothed + feature columns
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List

# ── Channel definitions (matches CHANNEL_MAP in generator) ──────────────────
CHANNEL_DOMAINS = {
    "Ch01": "AMR",      "Ch02": "AMR",      "Ch03": "AMR",      "Ch04": "AMR",
    "Ch05": "Biofilm",  "Ch06": "Biofilm",  "Ch07": "Biofilm",  "Ch08": "Biofilm",
    "Ch09": "Oncology", "Ch10": "Oncology", "Ch11": "Oncology", "Ch12": "Oncology",
}
CHANNELS       = list(CHANNEL_DOMAINS.keys())
BASELINE_nA    = 10.0          # healthy baseline current (Signal-OFF model)
THRESHOLD_DROP = 0.30          # 30% drop = detection threshold


# =============================================================================
# Kalman Filter (1-D, per channel)
# =============================================================================
class KalmanFilter1D:
    """
    Noise filtering layer (SVG box: 'Noise filtering (Kalman)').

    Single-variable steady-state Kalman filter.
    Assumes constant process model; optimised for electrochemical nA signals
    with ~0.15 nA measurement noise std.

    Args:
        process_variance : model uncertainty (Q)
        measurement_noise: sensor noise variance (R)
    """

    def __init__(self, process_variance: float = 0.01,
                 measurement_noise: float = 0.0225):
        self.Q    = process_variance
        self.R    = measurement_noise
        self._P   = 1.0    # initial estimate covariance
        self._x   = None   # current state estimate

    def reset(self, initial_value: float = BASELINE_nA):
        self._x = initial_value
        self._P = 1.0

    def update(self, measurement: float) -> float:
        if self._x is None:
            self.reset(measurement)

        # Predict
        P_pred = self._P + self.Q

        # Update (Kalman gain)
        K      = P_pred / (P_pred + self.R)
        self._x = self._x + K * (measurement - self._x)
        self._P = (1 - K) * P_pred

        return self._x

    def filter_sequence(self, measurements: List[float]) -> List[float]:
        """Filter a time-series of measurements."""
        self.reset(measurements[0] if measurements else BASELINE_nA)
        return [self.update(m) for m in measurements]

    def filter_single(self, measurement: float,
                      prior: float = BASELINE_nA) -> float:
        """Filter a single measurement (stateless, uses prior as starting point)."""
        self.reset(prior)
        return self.update(measurement)


# =============================================================================
# Feature Extractor (per-channel)
# =============================================================================
@dataclass
class ChannelFeatures:
    """All extracted features for one channel from one sample."""
    channel:             str
    domain:              str
    raw_nA:              float
    smooth_nA:           float
    concentration_pM:    float
    peak_amplitude_nA:   float     # baseline – smooth  (positive = target present)
    signal_drop_pct:     float     # % drop from baseline
    time_to_threshold_s: float     # simulated; real device would measure this
    impedance_change_pct:float     # correlated with peak amplitude
    snr_db:              float     # signal power / noise floor
    above_threshold:     bool      # True if drop > 30%


def extract_channel_features(ch: str, raw_nA: float, smooth_nA: float,
                             conc_pM: float, t2t_s: float,
                             impedance_pct: float) -> ChannelFeatures:
    """
    Stage 4 feature extraction for a single channel.
    Called during inference; mirrors what the generator pre-computes.
    """
    peak_amp   = max(0.0, BASELINE_nA - smooth_nA)
    drop_pct   = round(peak_amp / BASELINE_nA * 100, 2)
    signal_pwr = peak_amp ** 2
    noise_pwr  = 0.15 ** 2
    snr_db     = round(10 * np.log10(max(signal_pwr, 1e-9) / noise_pwr), 2)

    return ChannelFeatures(
        channel             = ch,
        domain              = CHANNEL_DOMAINS[ch],
        raw_nA              = raw_nA,
        smooth_nA           = smooth_nA,
        concentration_pM    = conc_pM,
        peak_amplitude_nA   = round(peak_amp, 3),
        signal_drop_pct     = drop_pct,
        time_to_threshold_s = t2t_s,
        impedance_change_pct= impedance_pct,
        snr_db              = snr_db,
        above_threshold     = drop_pct >= (THRESHOLD_DROP * 100),
    )


# =============================================================================
# Signal Processor -- processes a full DataFrame
# =============================================================================
class SignalProcessor:
    """
    SVG Layer 2 orchestrator.

    Accepts a DataFrame (one row = one patient sample) containing raw nA
    readings and pre-extracted features (as produced by the data generator,
    or from a real device), and returns:
      - smoothed signal columns (Kalman)
      - standardised feature matrix ready for ML heads
      - per-domain detection flags

    Usage:
        sp   = SignalProcessor()
        Xfeat, flags = sp.process(df)
    """

    # Feature columns extracted per channel
    FEATURE_SUFFIXES = [
        "_raw_nA", "_smooth_nA", "_conc_pM",
        "_peak_amp_nA", "_drop_pct", "_t2t_s",
        "_impedance_pct", "_snr_db",
    ]

    def __init__(self):
        self._kalman = {ch: KalmanFilter1D() for ch in CHANNELS}

    # ── Re-smooth (apply Kalman) if only raw_nA present ─────────────────────
    def re_smooth(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Kalman filter to raw_nA columns and overwrite smooth_nA columns.
        Useful if the DataFrame contains only raw readings (real device input).
        """
        df = df.copy()
        for ch in CHANNELS:
            raw_col    = f"{ch}_raw_nA"
            smooth_col = f"{ch}_smooth_nA"
            if raw_col in df.columns:
                kf = KalmanFilter1D()
                df[smooth_col] = df[raw_col].apply(
                    lambda v: kf.filter_single(v, BASELINE_nA))
        return df

    # ── Build domain feature matrices ────────────────────────────────────────
    def _domain_features(self, df: pd.DataFrame, domain: str) -> pd.DataFrame:
        """Return feature columns for one domain (AMR / Biofilm / Oncology)."""
        domain_chs = [ch for ch, d in CHANNEL_DOMAINS.items() if d == domain]
        cols = []
        for ch in domain_chs:
            for suf in self.FEATURE_SUFFIXES:
                col = f"{ch}{suf}"
                if col in df.columns:
                    cols.append(col)
        return df[cols].copy()

    def get_amr_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._domain_features(df, "AMR")

    def get_biofilm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        bf = self._domain_features(df, "Biofilm")
        # Add cross-channel quorum interaction: AHL_AI2 X c-diGMP product
        if "Ch06_drop_pct" in df.columns and "Ch08_drop_pct" in df.columns:
            bf["Ch06_Ch08_QS_interaction"] = df["Ch06_drop_pct"] * df["Ch08_drop_pct"]
        return bf

    def get_oncology_features(self, df: pd.DataFrame) -> pd.DataFrame:
        onc = self._domain_features(df, "Oncology")
        # GNN-inspired: FadA x CagA co-occurrence interaction term
        if "Ch09_drop_pct" in df.columns and "Ch10_drop_pct" in df.columns:
            onc["Ch09_Ch10_codetect"] = df["Ch09_drop_pct"] * df["Ch10_drop_pct"]
        # pks x miRNA co-signal (colorectal compound risk)
        if "Ch11_drop_pct" in df.columns and "Ch12_drop_pct" in df.columns:
            onc["Ch11_Ch12_codetect"] = df["Ch11_drop_pct"] * df["Ch12_drop_pct"]
        return onc

    # ── Full pipeline process ────────────────────────────────────────────────
    def process(self, df: pd.DataFrame) -> dict:
        """
        Run the full signal processing layer on a DataFrame.

        Returns a dict of:
          'amr_features'     : DataFrame  (AMR domain features)
          'biofilm_features' : DataFrame  (Biofilm domain features + interactions)
          'oncology_features': DataFrame  (Oncology domain features + interactions)
          'detection_flags'  : DataFrame  (per-channel binary above-threshold flags)
          'qs_proxy'         : Series     (quorum sensing activity proxy 0-100)
        """
        # Re-smooth if needed
        if any(f"Ch01_smooth_nA" not in df.columns for _ in [1]):
            df = self.re_smooth(df)

        # Per-channel detection flags (drop > 30%)
        flag_cols = {}
        for ch in CHANNELS:
            drop_col = f"{ch}_drop_pct"
            if drop_col in df.columns:
                flag_cols[f"{ch}_detected"] = (df[drop_col] >= 30.0).astype(int)
        det_flags = pd.DataFrame(flag_cols, index=df.index)

        # QS activity proxy (continuous)
        qs_proxy = pd.Series(0.0, index=df.index)
        if "Ch06_drop_pct" in df.columns and "Ch08_conc_pM" in df.columns:
            qs_proxy = (df["Ch06_drop_pct"] * 0.6 +
                        df["Ch08_conc_pM"].clip(0, 1000) / 10 * 0.4).clip(0, 100)

        return {
            "amr_features":      self.get_amr_features(df),
            "biofilm_features":  self.get_biofilm_features(df),
            "oncology_features": self.get_oncology_features(df),
            "detection_flags":   det_flags,
            "qs_proxy":          qs_proxy,
        }

    # ── Summary stats ────────────────────────────────────────────────────────
    def signal_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Quick summary: mean drop % and detection rate per channel."""
        rows = []
        for ch in CHANNELS:
            drop_col = f"{ch}_drop_pct"
            if drop_col in df.columns:
                detected = (df[drop_col] >= 30.0)
                rows.append({
                    "Channel":        ch,
                    "Domain":         CHANNEL_DOMAINS[ch],
                    "Mean_drop_pct":  round(df[drop_col].mean(), 2),
                    "Max_drop_pct":   round(df[drop_col].max(), 2),
                    "Detection_rate": round(detected.mean() * 100, 1),
                    "N_detected":     int(detected.sum()),
                })
        return pd.DataFrame(rows)
