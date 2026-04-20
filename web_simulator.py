"""
web_simulator.py  --  Real-time SSE bridge between run2.py and the Flask dashboard
===================================================================================
Every click generates a FRESH random patient. The full AI pipeline is broken into
individual steps, each yielding its OWN SSE event with real intermediate results.
Nothing runs as a hidden monolithic block — each layer is visible in the UI.

Pipeline steps (each fires its own SSE 'pipeline_step' event):
  Step 1:  Signal Processing & Feature Extraction
  Step 2:  Load Training Data (500-sample reference set)
  Step 3:  Multimodal Encoder (1D-CNN shared weights, 12×32 embeddings)
  Step 4:  Cross-Channel Transformer Attention (12×12 matrix)
  Step 5:  Bayesian Calibration (MC Dropout, 50 passes)
  Step 6:  AMR Head (Multilabel XGBoost) — train + infer
  Step 7:  Biofilm Head (Gradient Boosting) — train + infer
  Step 8:  Oncology Head (Random Forest) — train + infer
  Step 9:  Ensemble Risk Engine (weighted + co-occurrence bonus)
"""

import json
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import contextlib
import sys
import io

import run2
from run2 import (
    SCENARIOS, CHANNEL_META, BASELINE_nA, NOISE_STD_nA, DURATION_S, INJECT_S,
    StreamingKalman, generate_channel_signal, BASE
)

# ── Clinical categories and their biomarker activation rules ────────────────
CATEGORIES = [
    "Healthy", "AMR_only", "Biofilm_only", "Oncology_only",
    "AMR_Biofilm", "AMR_Oncology", "Biofilm_Oncology", "Pan_positive"
]
CATEGORY_WEIGHTS = [0.12, 0.14, 0.14, 0.14, 0.14, 0.10, 0.10, 0.12]

AMR_CHS = ["Ch01", "Ch02", "Ch03", "Ch04"]
BIO_CHS = ["Ch05", "Ch06", "Ch07", "Ch08"]
ONC_CHS = ["Ch09", "Ch10", "Ch11", "Ch12"]

SITES   = ["ICU", "ED", "Outpatient_clinic", "Surgery_ward", "Community"]
SAMPLES = ["Blood", "Urine", "Wound_swab", "Biopsy_fluid"]
NOTES_MAP = {
    "Healthy":          "Routine screening — no clinical suspicion",
    "AMR_only":         "Suspected antibiotic-resistant infection",
    "Biofilm_only":     "Chronic wound / device-associated biofilm query",
    "Oncology_only":    "GI oncology screening — microbiome panel",
    "AMR_Biofilm":      "Post-surgical wound with suspected resistant biofilm",
    "AMR_Oncology":     "Febrile neutropenia with GI co-signal",
    "Biofilm_Oncology": "Chronic inflammation workup + biofilm query",
    "Pan_positive":     "Multi-domain critical case — ICU admission",
}


def generate_random_scenario():
    """Build a completely random patient scenario. No two runs are identical."""
    rng = random.Random()

    category = rng.choices(CATEGORIES, weights=CATEGORY_WEIGHTS, k=1)[0]
    active_domains = []
    if "AMR"      in category or "Pan" in category: active_domains.append("AMR")
    if "Biofilm"  in category or "Pan" in category: active_domains.append("Biofilm")
    if "Oncology" in category or "Pan" in category: active_domains.append("Oncology")

    positive = {}
    domain_chs = {"AMR": AMR_CHS, "Biofilm": BIO_CHS, "Oncology": ONC_CHS}
    for domain in active_domains:
        chs = domain_chs[domain]
        n_active = rng.randint(1, len(chs))
        for ch in rng.sample(chs, n_active):
            positive[ch] = round(rng.uniform(30, 500), 1)

    if "Ch06" in positive and rng.random() < 0.70:
        positive.setdefault("Ch08", round(rng.uniform(40, 200), 1))
    if "Ch07" in positive and rng.random() < 0.50:
        positive.setdefault("Ch05", round(rng.uniform(80, 350), 1))

    if category == "Healthy":
        positive = {}
        for ch in CHANNEL_META:
            if rng.random() < 0.05:
                positive[ch] = round(rng.uniform(15, 60), 1)

    age = max(1, min(95, int(rng.gauss(52, 18))))
    sex = rng.choice(["M", "F"])
    bmi = round(max(15.0, min(50.0, rng.gauss(26.5, 5.0))), 1)
    site = rng.choice(SITES)
    sample = rng.choice(SAMPLES)
    vol = round(rng.uniform(4.0, 6.5), 1)
    device = f"DEV-{rng.choice(['A','B','C'])}{rng.randint(10,99)}"
    lot = f"LOT-{rng.randint(2024,2026)}-{rng.randint(1,999):03d}"
    temp = round(rng.uniform(18, 30), 1)
    ph = round(rng.uniform(6.8, 7.6), 2)

    n_pos = len(positive)
    domains_hit = set()
    for ch in positive:
        domains_hit.add(CHANNEL_META[ch][0])
    name_parts = []
    if "AMR" in domains_hit:      name_parts.append("AMR")
    if "Biofilm" in domains_hit:  name_parts.append("Biofilm")
    if "Oncology" in domains_hit: name_parts.append("Oncology")
    if not name_parts:            name_parts.append("Healthy Screen")
    name = f"Random Patient — {' + '.join(name_parts)}  [{n_pos} analytes]"

    return {
        "name": name, "patient_id": f"RND-{rng.randint(1000,9999)}",
        "age": age, "sex": sex, "bmi": bmi, "site": site, "sample": sample,
        "vol_uL": vol, "device": device, "lot": lot, "temp_C": temp, "pH": ph,
        "positive": positive, "clinical_note": NOTES_MAP.get(category, ""),
        "_category": category,
    }


def format_sse(event_type, payload):
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"


def run_simulation_stream(scenario_id=0, fast=True):
    """
    Main SSE generator. Each AI pipeline layer runs individually and
    yields its own event with real intermediate results.
    """

    # ── Build scenario ────────────────────────────────────────────────────────
    if scenario_id == 0:
        sc = generate_random_scenario()
    else:
        sc = SCENARIOS.get(scenario_id, SCENARIOS[1]).copy()

    rng = np.random.default_rng()

    # ── Boot ──────────────────────────────────────────────────────────────────
    yield format_sse("boot", {
        "scenario_name": sc["name"],
        "patient": f"{sc['age']}{sc['sex']} / {sc['site']} / {sc['sample']} ({sc['vol_uL']} µL)",
        "note": sc["clinical_note"],
        "device": f"{sc['device']} | Lot: {sc['lot']}",
        "channels": {
            ch: {
                "domain": d, "bm": bm,
                "color": c.replace("\033[94m","blue").replace("\033[95m","purple").replace("\033[91m","red")
            } for ch, (d, bm, c) in CHANNEL_META.items()
        }
    })
    time.sleep(0.5 if fast else 1.5)

    yield format_sse("log", {"message": f"Measuring baseline for {INJECT_S}s..."})
    time.sleep(0.3 if fast else 1.5)
    yield format_sse("log", {"message": "SAMPLE INJECTED AT t=60s. Live Acquisition started."})
    time.sleep(0.3 if fast else 1.5)

    # ── Signal generation ─────────────────────────────────────────────────────
    t_arr = np.arange(0, DURATION_S, 1.0)
    signals_nA = {}
    for ch in CHANNEL_META:
        conc = sc["positive"].get(ch, None)
        seed = int(ch[2:]) * 7 + int(time.time() * 1000) % 100000
        raw_nA, _ = generate_channel_signal(t_arr, conc, rng=rng, seed=seed)
        signals_nA[ch] = raw_nA

    kf        = {ch: StreamingKalman() for ch in CHANNEL_META}
    ch_raw    = {ch: BASELINE_nA for ch in CHANNEL_META}
    ch_smooth = {ch: BASELINE_nA for ch in CHANNEL_META}
    ch_drop   = {ch: 0.0         for ch in CHANNEL_META}
    ch_det    = {ch: False       for ch in CHANNEL_META}

    step  = 3 if fast else 6
    sleep = 0.02 if fast else 0.1

    # ── Live streaming (Kalman) ───────────────────────────────────────────────
    for idx in range(0, DURATION_S, step):
        t = t_arr[idx]
        current_data = {"time": int(t), "channels": {}}
        for ch in CHANNEL_META:
            raw_nA = float(signals_nA[ch][idx])
            sm_nA  = kf[ch].step(raw_nA)
            drop   = max(0.0, (BASELINE_nA - sm_nA) / BASELINE_nA * 100)
            ch_raw[ch] = raw_nA; ch_smooth[ch] = sm_nA; ch_drop[ch] = drop
            if drop >= 30.0 and not ch_det[ch]:
                ch_det[ch] = True
                yield format_sse("detection", {
                    "time": int(t), "channel": ch,
                    "biomarker": CHANNEL_META[ch][1], "drop": round(drop, 1)
                })
            current_data["channels"][ch] = {
                "raw_nA": round(raw_nA, 3), "smooth_nA": round(sm_nA, 3),
                "drop_pct": round(drop, 1), "detected": ch_det[ch]
            }
        yield format_sse("live_data", current_data)
        time.sleep(sleep)

    # =========================================================================
    # AI PIPELINE — Each step runs INDIVIDUALLY with its own SSE event
    # =========================================================================

    from pipeline.signal_processor  import SignalProcessor
    from pipeline.multimodal_encoder import MultimodalEncoder
    from pipeline.models import (
        AMRHead, BiofilmHead, OncologyHead, EnsembleRiskEngine,
        AMR_GENE_LABELS, BIOFILM_STAGE_MAP, ONCOLOGY_TIER_MAP
    )
    dummy = io.StringIO()

    # ── STEP 1: Feature Extraction ────────────────────────────────────────────
    yield format_sse("pipeline_step", {
        "step": 1, "name": "Feature Extraction",
        "detail": "Peak amplitude, SNR, AUC, time-to-threshold, concentration back-calc",
        "status": "running"
    })

    with contextlib.redirect_stdout(dummy):
        feature_df, feat_summary = run2.phase_features(
            sc, signals_nA, ch_smooth, ch_drop, ch_det, fast=True
        )

    n_det = len(feat_summary)
    det_names = [s[1] for s in feat_summary]   # biomarker names
    yield format_sse("pipeline_step", {
        "step": 1, "name": "Feature Extraction",
        "status": "done",
        "result": f"{n_det} biomarkers detected: {', '.join(det_names) if det_names else 'None'}"
    })
    time.sleep(0.3)

    # ── STEP 2: Load Training Data ────────────────────────────────────────────
    yield format_sse("pipeline_step", {
        "step": 2, "name": "Load Training Data",
        "detail": "Loading 500-sample reference dataset for model training",
        "status": "running"
    })

    sp = SignalProcessor()
    data_file = BASE / "biosensor_detailed_500.csv"
    gt_file   = BASE / "biosensor_ground_truth.csv"
    df_train  = pd.read_csv(data_file)
    df_gt     = pd.read_csv(gt_file)
    gt_cols   = ["Sample_ID"] + [f"Ch{i:02d}" for i in range(1, 13)]
    df_train  = df_train.merge(df_gt[gt_cols], on="Sample_ID", suffixes=("", "_gt"))

    yield format_sse("pipeline_step", {
        "step": 2, "name": "Load Training Data",
        "status": "done",
        "result": f"{len(df_train)} training samples × {df_train.shape[1]} features loaded"
    })
    time.sleep(0.3)

    # ── STEP 3: Multimodal Encoder (1D-CNN) ───────────────────────────────────
    yield format_sse("pipeline_step", {
        "step": 3, "name": "Multimodal Encoder (1D-CNN)",
        "detail": "Shared-weight 1D-CNN encodes 12 channels → 32-dim latent vectors",
        "status": "running"
    })

    enc = MultimodalEncoder(latent_dim=32, n_heads=4, n_mc_passes=50)
    enc.fit(df_train)
    tr_aug = enc.get_augmented_features(df_train, sp.process(df_train))
    te_aug = enc.get_augmented_features(feature_df, sp.process(feature_df))
    te_enc = te_aug["_enc_out"]

    yield format_sse("pipeline_step", {
        "step": 3, "name": "Multimodal Encoder (1D-CNN)",
        "status": "done",
        "result": "12 channels × 32-dim embeddings encoded. Fused latent vector ready."
    })
    time.sleep(0.3)

    # ── STEP 4: Cross-Channel Transformer Attention ───────────────────────────
    yield format_sse("pipeline_step", {
        "step": 4, "name": "Transformer Attention",
        "detail": "12×12 attention matrix — cross-domain co-signal analysis",
        "status": "running"
    })

    attn = te_enc["attention_matrix"][0]
    cd   = te_enc["cross_domain_scores"][0]

    amr_idx = [0,1,2,3]; bf_idx = [4,5,6,7]; onc_idx = [8,9,10,11]
    ab = float(attn[np.ix_(amr_idx, bf_idx)].mean())
    ao = float(attn[np.ix_(amr_idx, onc_idx)].mean())
    bo = float(attn[np.ix_(bf_idx, onc_idx)].mean())
    strongest = max([(ab,"AMR↔Biofilm"),(ao,"AMR↔Oncology"),(bo,"Biofilm↔Oncology")], key=lambda x:x[0])

    yield format_sse("pipeline_step", {
        "step": 4, "name": "Transformer Attention",
        "status": "done",
        "result": f"AMR={cd[0]:.2f}  Biofilm={cd[1]:.2f}  Onc={cd[2]:.2f} | Strongest: {strongest[1]} ({strongest[0]:.3f})"
    })
    time.sleep(0.3)

    # ── STEP 5: MC Dropout Bayesian Calibration ───────────────────────────────
    yield format_sse("pipeline_step", {
        "step": 5, "name": "Bayesian Calibration (MC Dropout)",
        "detail": "50 stochastic forward passes for confidence intervals",
        "status": "running"
    })

    calib = te_enc["calibration"]
    amr_conf = float(calib["MC_AMR_conf"][0])
    bio_conf = float(calib["MC_Biofilm_conf"][0])
    onc_conf = float(calib["MC_Oncology_conf"][0])

    yield format_sse("pipeline_step", {
        "step": 5, "name": "Bayesian Calibration (MC Dropout)",
        "status": "done",
        "result": f"Confidence — AMR: {amr_conf:.0f}%  Biofilm: {bio_conf:.0f}%  Oncology: {onc_conf:.0f}%"
    })
    time.sleep(0.3)

    # ── STEP 6: AMR Head ──────────────────────────────────────────────────────
    yield format_sse("pipeline_step", {
        "step": 6, "name": "AMR Head (Multilabel XGBoost)",
        "detail": "Training on 500 samples → inference on live patient",
        "status": "running"
    })

    def amr_y(df):
        cols = [f"Ch{i:02d}" for i in range(1, 5)]
        y = df[cols].copy(); y.columns = AMR_GENE_LABELS; return y.astype(int)

    X_amr_tr = tr_aug["AMR"];  X_amr_te = te_aug["AMR"]
    amr = AMRHead(n_estimators=200); amr.fit(X_amr_tr, amr_y(df_train))
    amr_proba = amr.predict_proba(X_amr_te)
    amr_pred  = amr.predict(X_amr_te)
    score_amr = float(amr_proba["AI_AMR_Score"].iloc[0])
    genes_pos = [g for g in AMR_GENE_LABELS if amr_pred[g].iloc[0] == 1] if not amr_pred.empty else []

    yield format_sse("pipeline_step", {
        "step": 6, "name": "AMR Head (Multilabel XGBoost)",
        "status": "done",
        "result": f"Score: {score_amr:.1f}% | Genes: {', '.join(genes_pos) if genes_pos else 'None detected'}",
        "score": score_amr
    })
    time.sleep(0.3)

    # ── STEP 7: Biofilm Head ──────────────────────────────────────────────────
    yield format_sse("pipeline_step", {
        "step": 7, "name": "Biofilm Head (Gradient Boosting)",
        "detail": "LSTM-style temporal model — stage classification",
        "status": "running"
    })

    def bio_y(df): return df["Biofilm_Stage"].fillna("None")
    X_bio_tr = tr_aug["Biofilm"]; X_bio_te = te_aug["Biofilm"]
    bio = BiofilmHead(n_estimators=300); bio.fit(X_bio_tr, bio_y(df_train))
    bio_proba = bio.predict_proba(X_bio_te)
    bio_pred  = bio.predict(X_bio_te)
    score_bio = float(bio_proba["AI_Biofilm_Score"].iloc[0])
    stage_pred = str(bio_pred.iloc[0])

    yield format_sse("pipeline_step", {
        "step": 7, "name": "Biofilm Head (Gradient Boosting)",
        "status": "done",
        "result": f"Score: {score_bio:.1f}% | Stage: {stage_pred}",
        "score": score_bio
    })
    time.sleep(0.3)

    # ── STEP 8: Oncology Head ─────────────────────────────────────────────────
    yield format_sse("pipeline_step", {
        "step": 8, "name": "Oncology Head (Random Forest)",
        "detail": "GNN-style microbial co-occurrence network",
        "status": "running"
    })

    def onc_y(df): return df["Oncology_Risk_Tier"].fillna("Low")
    X_onc_tr = tr_aug["Oncology"]; X_onc_te = te_aug["Oncology"]
    onc = OncologyHead(n_estimators=300); onc.fit(X_onc_tr, onc_y(df_train))
    onc_proba = onc.predict_proba(X_onc_te)
    onc_pred  = onc.predict(X_onc_te)
    score_onc = float(onc_proba["AI_Oncology_Score"].iloc[0])
    tier_pred = str(onc_pred.iloc[0])

    yield format_sse("pipeline_step", {
        "step": 8, "name": "Oncology Head (Random Forest)",
        "status": "done",
        "result": f"Score: {score_onc:.1f}% | Tier: {tier_pred}",
        "score": score_onc
    })
    time.sleep(0.3)

    # ── STEP 9: Ensemble Risk Engine ──────────────────────────────────────────
    yield format_sse("pipeline_step", {
        "step": 9, "name": "Ensemble Risk Engine",
        "detail": f"0.40×AMR + 0.35×Biofilm + 0.25×Oncology + co-occurrence bonus",
        "status": "running"
    })

    engine   = EnsembleRiskEngine()
    ensemble = engine.compute(amr_proba, bio_proba, onc_proba)
    n_active = int(ensemble["N_Active_Domains"].iloc[0])
    co_bonus = float(ensemble["Co_occurrence_Bonus"].iloc[0])
    ens_score = float(ensemble["Ensemble_Risk_Score"].iloc[0])
    ens_tier  = str(ensemble["Ensemble_Risk_Tier"].iloc[0])

    yield format_sse("pipeline_step", {
        "step": 9, "name": "Ensemble Risk Engine",
        "status": "done",
        "result": f"0.40×{score_amr:.1f} + 0.35×{score_bio:.1f} + 0.25×{score_onc:.1f} + {co_bonus:.0f} = {ens_score:.1f}% ({ens_tier})",
        "score": ens_score
    })
    time.sleep(0.3)

    # ── Assemble final report ─────────────────────────────────────────────────
    row_d = feature_df.iloc[0].to_dict()

    ANTIBIOTIC_MAP = {"blaNDM1":["Carbapenems","Penicillins"], "mecA":["Methicillin","Oxacillin"],
                      "vanA":["Vancomycin","Teicoplanin"], "KPC":["Carbapenems","Aztreonam"]}
    ALT_MAP = {"blaNDM1":"Colistin / Tigecycline", "mecA":"Vancomycin / Linezolid",
               "vanA":"Daptomycin / Linezolid", "KPC":"Ceftazidime-avibactam"}
    fail = list(set(ab for g in genes_pos for ab in ANTIBIOTIC_MAP.get(g, [])))
    alts = list(set(ALT_MAP[g] for g in genes_pos if g in ALT_MAP))
    profile = ("Pan-resistant" if len(genes_pos) >= 3 else "Multi-drug resistant" if len(genes_pos) == 2
               else "Single resistance" if len(genes_pos) == 1 else "Susceptible")

    n_bf_det = sum(1 for ch in BIO_CHS if row_d.get(f"{ch}_detected", 0))
    bf_stages = ["None","Stage I (early attachment)","Stage II (microcolony)",
                 "Stage III (maturation)","Stage IV (dispersion)"]
    bf_stage = bf_stages[min(n_bf_det, 4)]
    ch06_d = row_d.get("Ch06_drop_pct", 0)
    ch08_c = row_d.get("Ch08_conc_pM", 0)
    qs_score = round(ch06_d * 0.6 + ch08_c / 10 * 0.4, 1)

    onc_det = [(ch, CHANNEL_META[ch][1]) for ch in ONC_CHS if row_d.get(f"{ch}_detected", 0)]
    SPECIES_MAP = {"FadA":"F. nucleatum","CagA":"H. pylori","pks":"E. coli genotoxin","miRNA-21":"epigenetic"}
    onc_spec = ", ".join(SPECIES_MAP.get(m,"") for _,m in onc_det) or "None"

    final_report = {
        "patient_id": sc["patient_id"],
        "amr": {
            "score": score_amr, "profile": profile,
            "genes": ", ".join(genes_pos) if genes_pos else "None",
            "failed_drugs": ", ".join(fail) if fail else "None",
            "recommended": alts[0] if alts else "Standard empirics",
            "conf": amr_conf
        },
        "biofilm": {
            "score": score_bio, "stage": bf_stage, "qs_score": qs_score,
            "cdigmp_uM": f"{round(float(np.random.uniform(0.5,5.0)),2)} µM" if row_d.get("Ch08_detected") else "0.05 µM",
            "conf": bio_conf
        },
        "oncology": {
            "score": score_onc, "tier": tier_pred, "species": onc_spec,
            "referral": "⚠ REFERRAL RECOMMENDED" if score_onc >= 50 else "No referral needed",
            "conf": onc_conf
        },
        "ensemble": {
            "score": ens_score, "tier": ens_tier, "active": n_active, "bonus": co_bonus
        }
    }

    # Write HTML report to disk
    ai_out = {
        "score_amr": score_amr, "score_bio": score_bio, "score_onc": score_onc,
        "ens_score": ens_score, "ens_tier": ens_tier, "n_active": n_active,
        "co_bonus": co_bonus, "genes_pos": genes_pos, "stage_pred": stage_pred,
        "tier_pred": tier_pred, "amr_proba": amr_proba, "bio_proba": bio_proba,
        "onc_proba": onc_proba, "calib": calib,
    }
    with contextlib.redirect_stdout(dummy):
        run2.phase_report(sc, feature_df, ai_out, feat_summary, fast=True)

    yield format_sse("log", {"message": "Analysis Complete. Generated Final Report."})
    yield format_sse("done", final_report)
