"""
run.py  --  Full end-to-end pipeline orchestrator
===================================================
"If the electrical raw data is created, do everything that is needed."

Executes the complete SVG pipeline from raw biosensor CSV to clinical HTML reports:

  [Layer 2]  Signal Transduction & Preprocessing
               Kalman filter  |  Feature extraction  |  QS proxy
  [Layer 3a] Multimodal Encoder          (1D-CNN per channel, shared weights)
  [Layer 3b] Cross-Channel Attention     (Transformer fusion)
  [Layer 3c] Bayesian Calibration        (MC Dropout uncertainty)
  [Layer 3d] Task-Specific Heads         (XGBoost AMR | GB Biofilm | RF Oncology)
  [Layer 3e] Integrated Risk Engine      (Ensemble + co-occurrence bonus)
  [Layer 7]  SHAP Explainability         (per-biomarker attribution)
  [Layer 4]  Edge Device Output          (BLE/WiFi, transmission summary)
  [Layer 5]  Clinical Decision Report    (HTML, one per sample)

Output:
  reports/report_BIO0001.html  ...  report_BIO000N.html
  biosensor_full_results.csv
  biosensor_eval_report.txt

Usage:
  python run.py                    # run on all 500 samples
  python run.py --n-samples 10     # first 10 only
  python run.py --reports-only 5   # generate 5 HTML reports (skip re-training)
"""

import argparse
import warnings
import time
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

BASE_DIR  = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "biosensor_detailed_500.csv"
GT_FILE   = BASE_DIR / "data" / "biosensor_ground_truth.csv"
SHAP_FILE = BASE_DIR / "data" / "biosensor_shap.csv"
OUT_CSV   = BASE_DIR / "data" / "biosensor_full_results.csv"
REPORT_DIR= BASE_DIR / "reports"

from pipeline.signal_processor  import SignalProcessor
from pipeline.multimodal_encoder import MultimodalEncoder
from pipeline.models             import (AMRHead, BiofilmHead, OncologyHead,
                                         EnsembleRiskEngine,
                                         AMR_GENE_LABELS,
                                         BIOFILM_STAGE_MAP, ONCOLOGY_TIER_MAP)
from pipeline.clinical_report    import ClinicalReportGenerator


# ── Pretty printer ─────────────────────────────────────────────────────────────
def sec(title: str):
    print(f"\n{'='*62}\n  {title}\n{'='*62}")


def tick(msg: str):
    print(f"  {msg}")


# =============================================================================
# Load & prepare data
# =============================================================================
def load_data():
    sec("Loading Data")
    df    = pd.read_csv(DATA_FILE)
    df_gt = pd.read_csv(GT_FILE)
    df_sh = pd.read_csv(SHAP_FILE)
    # Merge ground truth channel columns
    gt_cols = ["Sample_ID"] + [f"Ch{i:02d}" for i in range(1, 13)]
    df = df.merge(df_gt[gt_cols], on="Sample_ID", suffixes=("", "_gt"))
    tick(f"Loaded {len(df)} samples | {df.shape[1]} columns")
    return df, df_gt, df_sh


def split(df, test_frac=0.20):
    tr, te = train_test_split(df, test_size=test_frac, random_state=42,
                               stratify=df["Clinical_Category"])
    return tr.reset_index(drop=True), te.reset_index(drop=True)


# =============================================================================
# Layer 2 -- Signal Processing + QS proxy
# =============================================================================
def run_signal_processing(df_tr, df_te):
    sec("Layer 2 | Signal Transduction & Preprocessing")
    sp       = SignalProcessor()
    tr_proc  = sp.process(df_tr)
    te_proc  = sp.process(df_te)
    summary  = sp.signal_summary(df_tr)
    tick("Channel detection summary (train):")
    print(summary.to_string(index=False, justify="left"))
    return sp, tr_proc, te_proc


# =============================================================================
# Layer 3a/3b/3c -- Multimodal Encoder
# =============================================================================
def run_encoder(df_tr, df_te, tr_proc, te_proc):
    sec("Layer 3a-3c | Multimodal Encoder + Attention + Bayesian Calibration")
    enc = MultimodalEncoder(latent_dim=32, n_heads=4, n_mc_passes=50)
    enc.fit(df_tr)

    tick("Encoding train set...")
    tr_aug = enc.get_augmented_features(df_tr, tr_proc)
    tick("Encoding test set...")
    te_aug = enc.get_augmented_features(df_te, te_proc)

    # Show cross-domain attention scores (mean over test)
    te_enc   = te_aug["_enc_out"]
    cd_scores = te_enc["cross_domain_scores"].mean(axis=0)
    tick(f"Mean cross-domain attention scores (test):")
    for di, d in enumerate(["AMR","Biofilm","Oncology"]):
        tick(f"    {d:<12}  {cd_scores[di]:.2f}")

    return enc, tr_aug, te_aug


# =============================================================================
# Layer 3d -- Task-specific heads
# =============================================================================
def run_task_heads(df_tr, df_te, tr_aug, te_aug):
    sec("Layer 3d | Task-Specific Classification Heads")

    # AMR targets (ground truth gene columns)
    def amr_y(df):
        gt_cols = [f"Ch{i:02d}" for i in range(1, 5)]
        y = df[gt_cols].copy()
        y.columns = AMR_GENE_LABELS
        return y.astype(int)

    def bio_y(df):  return df["Biofilm_Stage"].fillna("None")
    def onc_y(df):  return df["Oncology_Risk_Tier"].fillna("Low")

    # Feature matrices (augmented = raw signals + encoder embeddings)
    X_amr_tr = tr_aug["AMR"]
    X_bio_tr = tr_aug["Biofilm"]
    X_onc_tr = tr_aug["Oncology"]

    X_amr_te = te_aug["AMR"]
    X_bio_te = te_aug["Biofilm"]
    X_onc_te = te_aug["Oncology"]

    tick("[1/3] AMR head (Multilabel XGBoost)...")
    amr = AMRHead(n_estimators=200); amr.fit(X_amr_tr, amr_y(df_tr))

    tick("[2/3] Biofilm head (Gradient Boosting - LSTM temporal style)...")
    bio = BiofilmHead(n_estimators=300); bio.fit(X_bio_tr, bio_y(df_tr))

    tick("[3/3] Oncology head (Random Forest - GNN network style)...")
    onc = OncologyHead(n_estimators=300); onc.fit(X_onc_tr, onc_y(df_tr))

    tick("All heads trained.")
    return (amr, bio, onc,
            X_amr_te, X_bio_te, X_onc_te,
            amr_y(df_te), bio_y(df_te), onc_y(df_te))


# =============================================================================
# Layer 3c continued -- MC Dropout + predict
# =============================================================================
def run_inference(amr, bio, onc,
                  X_amr_te, X_bio_te, X_onc_te, te_aug):
    sec("Layer 3c | Bayesian Calibration (MC Dropout Inference)")

    amr_proba  = amr.predict_proba(X_amr_te)
    bio_proba  = bio.predict_proba(X_bio_te)
    onc_proba  = onc.predict_proba(X_onc_te)

    bio_pred   = bio.predict(X_bio_te)
    onc_pred   = onc.predict(X_onc_te)
    amr_pred   = amr.predict(X_amr_te)

    # MC Dropout confidence from calibrator
    te_enc     = te_aug["_enc_out"]
    calib      = te_enc["calibration"]

    for d in ["AMR", "Biofilm", "Oncology"]:
        mc_mean = calib[f"MC_{d}_mean"]
        mc_conf = calib[f"MC_{d}_conf"]
        tick(f"MC {d:<10}  mean={mc_mean.mean():.1f}  "
             f"avg conf={mc_conf.mean():.1f}%")

    return amr_proba, bio_proba, onc_proba, bio_pred, onc_pred, amr_pred, calib


# =============================================================================
# Layer 3e -- Ensemble Risk Engine
# =============================================================================
def run_ensemble(amr_proba, bio_proba, onc_proba, calib, df_te):
    sec("Layer 3e | Integrated Clinical Risk Engine")
    engine   = EnsembleRiskEngine()
    ensemble = engine.compute(amr_proba, bio_proba, onc_proba)

    # Attach MC Dropout confidence
    for d in ["AMR","Biofilm","Oncology"]:
        ensemble[f"MC_{d}_Conf_pct"] = calib[f"MC_{d}_conf"]
        ensemble[f"MC_{d}_LB"]       = calib[f"MC_{d}_lb"]
        ensemble[f"MC_{d}_UB"]       = calib[f"MC_{d}_ub"]

    tick("Ensemble risk tier distribution (test set):")
    for tier, cnt in ensemble["Ensemble_Risk_Tier"].value_counts().items():
        bar = "#" * cnt
        tick(f"    {tier:<12}  {cnt:>3}  {bar}")

    return ensemble


# =============================================================================
# Evaluation
# =============================================================================
def evaluate(amr, bio, onc,
             X_amr_te, X_bio_te, X_onc_te,
             y_amr_te, y_bio_te, y_onc_te,
             amr_proba, bio_pred, onc_pred):
    sec("Evaluation Metrics")

    # AMR per-gene
    amr_pred = amr.predict(X_amr_te)
    tick("AMR Head (gene-level):")
    for gene in AMR_GENE_LABELS:
        if gene not in y_amr_te.columns: continue
        yt  = y_amr_te[gene].values
        yp  = amr_pred[gene].values
        prec = (sum((yp==1)&(yt==1))) / (sum(yp==1)+1e-9)
        rec  = (sum((yp==1)&(yt==1))) / (sum(yt==1)+1e-9)
        try:
            from sklearn.metrics import roc_auc_score
            auc  = roc_auc_score(yt, amr_proba[f"P_{gene}"].values)
        except: auc = float("nan")
        tick(f"    {gene:<12}  AUC={auc:.3f}  P={prec:.2f}  R={rec:.2f}  N={int(yt.sum())}")

    # Biofilm ordinal
    yt_bio = y_bio_te.map(BIOFILM_STAGE_MAP).fillna(0).astype(int)
    yp_bio = bio_pred.map(BIOFILM_STAGE_MAP).fillna(0).astype(int)
    tick(f"\nBiofilm Head:  "
         f"Acc={( yt_bio==yp_bio).mean():.3f}  "
         f"Kappa={cohen_kappa_score(yt_bio, yp_bio, weights='quadratic'):.3f}  "
         f"Within-1={(abs(yt_bio-yp_bio)<=1).mean():.3f}")

    # Oncology ordinal
    yt_onc = y_onc_te.map(ONCOLOGY_TIER_MAP).fillna(0).astype(int)
    yp_onc = onc_pred.map(ONCOLOGY_TIER_MAP).fillna(0).astype(int)
    tick(f"Oncology Head: "
         f"Acc={(yt_onc==yp_onc).mean():.3f}  "
         f"Kappa={cohen_kappa_score(yt_onc, yp_onc, weights='quadratic'):.3f}  "
         f"Within-1={(abs(yt_onc-yp_onc)<=1).mean():.3f}")


# =============================================================================
# Save full results CSV
# =============================================================================
def save_results(df_te, ensemble, bio_pred, onc_pred, amr_pred, amr_proba):
    sec("Saving Full Results")
    out = pd.DataFrame({
        "Sample_ID":              df_te["Sample_ID"].values,
        "Clinical_Category_True": df_te["Clinical_Category"].values,
        "Biofilm_Stage_True":     df_te["Biofilm_Stage"].values,
        "Oncology_Tier_True":     df_te["Oncology_Risk_Tier"].values,
        "Biofilm_Stage_Pred":     bio_pred.values,
        "Oncology_Tier_Pred":     onc_pred.values,
    })
    for gene in AMR_GENE_LABELS:
        if gene in amr_pred.columns:
            out[f"AMR_{gene}_pred"] = amr_pred[gene].values
        if f"P_{gene}" in amr_proba.columns:
            out[f"AMR_{gene}_prob"] = amr_proba[f"P_{gene}"].values

    out = pd.concat([out, ensemble.reset_index(drop=True)], axis=1)
    out.to_csv(OUT_CSV, index=False)
    tick(f"Full results    -> {OUT_CSV}")
    return out


# =============================================================================
# Layer 4+5 -- Clinical HTML reports
# =============================================================================
def generate_reports(df_main, df_shap, results_df, n_reports: int = 10):
    sec(f"Layers 4+5 | Clinical Decision Support Reports (x{n_reports})")
    REPORT_DIR.mkdir(exist_ok=True)
    gen    = ClinicalReportGenerator()
    paths  = []
    sample_ids = results_df["Sample_ID"].tolist()[:n_reports]

    for sid in sample_ids:
        row_main = df_main[df_main["Sample_ID"]==sid]
        row_res  = results_df[results_df["Sample_ID"]==sid]
        row_shap = df_shap[df_shap["Sample_ID"]==sid] if df_shap is not None else None

        if row_main.empty: continue
        row_d = row_main.iloc[0].to_dict()
        # Merge AI output fields from results
        if not row_res.empty:
            for col in ["Ensemble_Risk_Score","Ensemble_Risk_Tier",
                        "N_Active_Domains","Co_occurrence_Bonus",
                        "AI_AMR_Score","AI_Biofilm_Score","AI_Oncology_Score",
                        "MC_AMR_Conf_pct","MC_Biofilm_Conf_pct","MC_Oncology_Conf_pct"]:
                if col in row_res.columns:
                    row_d[col] = row_res.iloc[0][col]

        shap_d = row_shap.iloc[0].to_dict() if (row_shap is not None and not row_shap.empty) else {}
        fpath  = REPORT_DIR / f"report_{sid}.html"
        gen.generate_single(row_d, shap_d, out_path=fpath)
        paths.append(str(fpath))

    tick(f"Generated {len(paths)} HTML reports -> {REPORT_DIR}/")
    for p in paths:
        tick(f"    {Path(p).name}")
    return paths


# =============================================================================
# Main
# =============================================================================
def main(n_reports: int = 10):
    t0 = time.time()
    print("\n" + "="*62)
    print("  AI-Enabled Multiplex Biosensor -- Full Pipeline")
    print("  SVG: ai_multiplex_biosensor_pipeline.svg")
    print("="*62)

    # Load
    df, df_gt, df_sh = load_data()
    df_tr, df_te     = split(df)

    # Layer 2
    sp, tr_proc, te_proc = run_signal_processing(df_tr, df_te)

    # Layers 3a-3c (encoder)
    enc, tr_aug, te_aug  = run_encoder(df_tr, df_te, tr_proc, te_proc)

    # Layer 3d (heads)
    (amr, bio, onc,
     X_amr_te, X_bio_te, X_onc_te,
     y_amr_te, y_bio_te, y_onc_te) = run_task_heads(df_tr, df_te, tr_aug, te_aug)

    # Layer 3c inference
    (amr_proba, bio_proba, onc_proba,
     bio_pred, onc_pred, amr_pred, calib) = run_inference(
        amr, bio, onc, X_amr_te, X_bio_te, X_onc_te, te_aug)

    # Layer 3e ensemble
    ensemble = run_ensemble(amr_proba, bio_proba, onc_proba, calib, df_te)

    # Evaluation
    evaluate(amr, bio, onc,
             X_amr_te, X_bio_te, X_onc_te,
             y_amr_te, y_bio_te, y_onc_te,
             amr_proba, bio_pred, onc_pred)

    # Save results
    results_df = save_results(df_te, ensemble, bio_pred, onc_pred, amr_pred, amr_proba)

    # Layers 4+5 -- reports
    generate_reports(df, df_sh, results_df, n_reports=n_reports)

    elapsed = time.time() - t0
    sec("Pipeline Complete")
    tick(f"Total pipeline time: {elapsed:.1f}s")
    tick(f"Outputs:")
    tick(f"  {OUT_CSV}")
    tick(f"  {REPORT_DIR}/report_BIO*.html  ({n_reports} files)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Biosensor Full Pipeline")
    parser.add_argument("--n-reports", type=int, default=10,
                        help="Number of HTML clinical reports to generate (default 10)")
    args = parser.parse_args()
    main(n_reports=args.n_reports)
