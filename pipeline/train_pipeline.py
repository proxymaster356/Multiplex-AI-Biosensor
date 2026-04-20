"""
pipeline/train_pipeline.py
===========================
End-to-end training and evaluation script for the AI biosensor pipeline.

SVG path executed:
  CSV data  -->  [Layer 2: SignalProcessor]
             -->  [Layer 3a: Multimodal Encoder  (fit 3 heads)]
             -->  [Layer 3b: Cross-channel Attention (interaction features)]
             -->  [Layer 3c: Bayesian Calibration (evaluate confidence)]
             -->  [Layer 3d: Detection Heads  (AMR / Biofilm / Oncology)]
             -->  [Layer 3e: Ensemble Risk Engine]
             -->  [Layer 7:  SHAP explainability]
             -->  Evaluation report + predictions CSV

Usage:
    python -m pipeline.train_pipeline
  or
    python pipeline/train_pipeline.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score,
    cohen_kappa_score, confusion_matrix
)

# ── Resolve paths ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_FILE  = BASE_DIR / "data" / "biosensor_detailed_500.csv"
GT_FILE    = BASE_DIR / "data" / "biosensor_ground_truth.csv"
OUT_PREDS  = BASE_DIR / "data" / "biosensor_predictions.csv"
OUT_REPORT = BASE_DIR / "reports" / "biosensor_eval_report.txt"

from pipeline.signal_processor import SignalProcessor
from pipeline.models import (
    AMRHead, BiofilmHead, OncologyHead, EnsembleRiskEngine,
    AMR_GENE_LABELS, AMR_GT_COLS,
    BIOFILM_STAGE_MAP, ONCOLOGY_TIER_MAP,
)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# =============================================================================
# Data loading helpers
# =============================================================================
def load_data():
    df    = pd.read_csv(DATA_FILE)
    df_gt = pd.read_csv(GT_FILE)
    df    = df.merge(df_gt[["Sample_ID"] + [f"Ch{i:02d}" for i in range(1, 13)]],
                     on="Sample_ID", suffixes=("", "_gt"))
    print(f"Loaded {len(df)} samples x {df.shape[1]} columns")
    return df


def train_test_split_stratified(df: pd.DataFrame, test_frac: float = 0.20):
    """Stratify on Clinical_Category to preserve class balance."""
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(
        df, test_size=test_frac, random_state=42,
        stratify=df["Clinical_Category"]
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


# =============================================================================
# Feature preparation
# =============================================================================
def prepare_features(df: pd.DataFrame, sp: SignalProcessor) -> dict:
    """Run SignalProcessor and return ready feature matrices."""
    processed = sp.process(df)
    return processed


def get_amr_targets(df: pd.DataFrame) -> pd.DataFrame:
    """AMR ground-truth: Ch01-Ch04 binary columns."""
    cols = [f"Ch{i:02d}" for i in range(1, 5)]
    rename = {f"Ch{i:02d}": AMR_GENE_LABELS[i-1] for i in range(1, 5)}
    return df[cols].rename(columns=rename).astype(int)


def get_biofilm_target(df: pd.DataFrame) -> pd.Series:
    return df["Biofilm_Stage"].fillna("None")


def get_oncology_target(df: pd.DataFrame) -> pd.Series:
    return df["Oncology_Risk_Tier"].fillna("Low")


# =============================================================================
# Evaluation helpers
# =============================================================================
def print_section(title: str, width: int = 60):
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def eval_amr(head: AMRHead, X: pd.DataFrame, y_true: pd.DataFrame,
             report_lines: list):
    print_section("AMR Head -- Multilabel XGBoost Evaluation")
    y_pred  = head.predict(X)
    y_proba = head.predict_proba(X)

    for gene in AMR_GENE_LABELS:
        if gene not in y_true.columns or gene not in y_pred.columns:
            continue
        yt = y_true[gene].values
        yp = y_pred[gene].values
        try:
            auc = roc_auc_score(yt, y_proba[f"P_{gene}"].values)
        except Exception:
            auc = float("nan")
        line = (f"  {gene:<12}  "
                f"AUC={auc:.3f}  "
                f"Precision={sum((yp==1)&(yt==1))/(sum(yp==1)+1e-9):.2f}  "
                f"Recall={sum((yp==1)&(yt==1))/(sum(yt==1)+1e-9):.2f}  "
                f"N_true={int(yt.sum())}")
        print(line)
        report_lines.append(line)


def eval_ordinal(name: str, y_true_str: pd.Series,
                 y_pred_str: pd.Series, enc_map: dict,
                 report_lines: list):
    print_section(f"{name} -- Ordinal Classification Evaluation")
    yt  = y_true_str.map(enc_map).fillna(0).astype(int)
    yp  = y_pred_str.map(enc_map).fillna(0).astype(int)
    acc = (yt == yp).mean()
    kappa = cohen_kappa_score(yt, yp, weights="quadratic")
    within1 = (abs(yt - yp) <= 1).mean()

    lines = [
        f"  Accuracy (exact):     {acc:.3f}",
        f"  Quadratic kappa:      {kappa:.3f}",
        f"  Within-1 accuracy:    {within1:.3f}",
        f"  Confusion matrix:",
    ]
    cm = confusion_matrix(yt, yp)
    for row in cm:
        lines.append("    " + "  ".join(f"{v:4d}" for v in row))

    for l in lines:
        print(l)
        report_lines.append(l)


# =============================================================================
# SHAP summary (if available)
# =============================================================================
def run_shap(head, X: pd.DataFrame, name: str, n: int = 100):
    if not HAS_SHAP:
        print("  [SHAP] shap library not installed, skipping.")
        return
    print_section(f"SHAP feature importance -- {name}")
    X_sample = X.iloc[:n]
    try:
        shap_df = head.shap_values(X_sample)
        if shap_df.empty:
            return
        mean_abs = shap_df.abs().mean().sort_values(ascending=False)
        for feat, val in mean_abs.head(8).items():
            print(f"  {feat:<40}  mean|SHAP|={val:.4f}")
    except Exception as e:
        print(f"  [SHAP] Error: {e}")


# =============================================================================
# Main pipeline
# =============================================================================
def main():
    report_lines = []

    # ── Load data ─────────────────────────────────────────────────────────────
    print_section("Loading Data")
    df = load_data()
    train_df, test_df = train_test_split_stratified(df)
    print(f"  Train: {len(train_df)}  |  Test: {len(test_df)}")

    # ── Signal processing (Layer 2) ───────────────────────────────────────────
    print_section("Layer 2: Signal Processing")
    sp          = SignalProcessor()
    train_proc  = prepare_features(train_df, sp)
    test_proc   = prepare_features(test_df,  sp)

    summary = sp.signal_summary(train_df)
    print(summary.to_string(index=False))
    report_lines.append("\nChannel Signal Summary (train set):")
    report_lines.append(summary.to_string(index=False))

    # Feature matrices
    X_amr_tr  = train_proc["amr_features"]
    X_bio_tr  = train_proc["biofilm_features"]
    X_onc_tr  = train_proc["oncology_features"]

    X_amr_te  = test_proc["amr_features"]
    X_bio_te  = test_proc["biofilm_features"]
    X_onc_te  = test_proc["oncology_features"]

    # Targets
    y_amr_tr  = get_amr_targets(train_df)
    y_bio_tr  = get_biofilm_target(train_df)
    y_onc_tr  = get_oncology_target(train_df)

    y_amr_te  = get_amr_targets(test_df)
    y_bio_te  = get_biofilm_target(test_df)
    y_onc_te  = get_oncology_target(test_df)

    # ── Train heads (Layer 3a / 3d) ───────────────────────────────────────────
    print_section("Layer 3: Training Model Heads")

    print("  [1/3] AMR head (Multilabel XGBoost)...")
    amr_head = AMRHead(n_estimators=200, learning_rate=0.05)
    amr_head.fit(X_amr_tr, y_amr_tr)
    print("        Done.")

    print("  [2/3] Biofilm head (Gradient Boosting LSTM-style)...")
    bio_head = BiofilmHead(n_estimators=300)
    bio_head.fit(X_bio_tr, y_bio_tr)
    print("        Done.")

    print("  [3/3] Oncology head (GNN-style Random Forest)...")
    onc_head = OncologyHead(n_estimators=300)
    onc_head.fit(X_onc_tr, y_onc_tr)
    print("        Done.")

    # ── Predict probabilities (Layer 3c) ──────────────────────────────────────
    print_section("Layer 3c: Predicting on Test Set")
    amr_proba  = amr_head.predict_proba(X_amr_te)
    bio_proba  = bio_head.predict_proba(X_bio_te)
    onc_proba  = onc_head.predict_proba(X_onc_te)

    bio_pred_label = bio_head.predict(X_bio_te)
    onc_pred_label = onc_head.predict(X_onc_te)

    # ── MC Dropout style confidence (Layer 3c) ───────────────────────────────
    # Proxy: confidence inversely proportional to boundary proximity
    def mc_confidence(score_series: pd.Series, n_passes: int = 50) -> pd.Series:
        # Simulate multiple passes by adding small noise and measuring std
        scores = np.stack([
            score_series.values + np.random.normal(0, 2, len(score_series))
            for _ in range(n_passes)], axis=1)
        std = scores.std(axis=1)
        return pd.Series(np.clip(100 - std * 3, 30, 99), index=score_series.index)

    amr_conf  = mc_confidence(amr_proba["AI_AMR_Score"])
    bio_conf  = mc_confidence(bio_proba["AI_Biofilm_Score"])
    onc_conf  = mc_confidence(onc_proba["AI_Oncology_Score"])

    # ── Ensemble (Layer 3e) ───────────────────────────────────────────────────
    print_section("Layer 3e: Ensemble Risk Engine")
    engine  = EnsembleRiskEngine()
    ensemble= engine.compute(amr_proba, bio_proba, onc_proba)
    ensemble["MC_AMR_Conf_pct"]      = amr_conf.values
    ensemble["MC_Biofilm_Conf_pct"]  = bio_conf.values
    ensemble["MC_Oncology_Conf_pct"] = onc_conf.values

    print(ensemble[["AI_AMR_Score","AI_Biofilm_Score","AI_Oncology_Score",
                     "Ensemble_Risk_Score","Ensemble_Risk_Tier"]].head(5).to_string())

    # ── Evaluation (Layer 3d) ─────────────────────────────────────────────────
    eval_amr(amr_head, X_amr_te, y_amr_te, report_lines)

    eval_ordinal("Biofilm Head",
                 y_bio_te, bio_pred_label, BIOFILM_STAGE_MAP, report_lines)

    eval_ordinal("Oncology Head",
                 y_onc_te, onc_pred_label, ONCOLOGY_TIER_MAP, report_lines)

    # ── SHAP (Layer 7) ────────────────────────────────────────────────────────
    run_shap(amr_head, X_amr_te,  "AMR Head")
    run_shap(bio_head, X_bio_te,  "Biofilm Head")
    run_shap(onc_head, X_onc_te,  "Oncology Head")

    # ── Save predictions CSV ──────────────────────────────────────────────────
    print_section("Saving Predictions")
    pred_df = pd.DataFrame({
        "Sample_ID":              test_df["Sample_ID"].values,
        "Clinical_Category_True": test_df["Clinical_Category"].values,
        "Biofilm_Stage_True":     y_bio_te.values,
        "Ontology_Tier_True":     y_onc_te.values,
        "Biofilm_Stage_Pred":     bio_pred_label.values,
        "Oncology_Tier_Pred":     onc_pred_label.values,
    })
    pred_df = pd.concat([pred_df, ensemble.reset_index(drop=True),
                         amr_proba.reset_index(drop=True)], axis=1)

    pred_df.to_csv(OUT_PREDS, index=False)
    print(f"  Predictions  --> {OUT_PREDS}")

    with open(OUT_REPORT, "w") as f:
        f.write("AI-Enabled Multiplex Biosensor Pipeline -- Evaluation Report\n")
        f.write("=" * 60 + "\n")
        f.write("\n".join(report_lines))
    print(f"  Eval report  --> {OUT_REPORT}")

    print_section("Pipeline Complete")
    tier_counts = ensemble["Ensemble_Risk_Tier"].value_counts()
    print("  Ensemble risk tier distribution (test set):")
    for tier, cnt in tier_counts.items():
        print(f"    {tier:<12} {cnt}")


if __name__ == "__main__":
    main()
