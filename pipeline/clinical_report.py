"""
pipeline/clinical_report.py
=============================
SVG Layers 4 + 5 -- Edge Device Output + Clinical Decision Support Report

Generates a full professional HTML clinical report with:
  - Patient metadata header
  - Device telemetry (BLE/Wi-Fi, time, cartridge lot)
  - 3-panel clinical output (AMR / Biofilm / Oncology)
  - Ensemble risk gauge
  - SHAP driver bar chart (inline SVG)
  - MC Dropout confidence intervals
  - Referral flag + treatment guidance
"""

from datetime import datetime
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent


# =============================================================================
# Colour palette (matches SVG dark theme translated to clinical white/light)
# =============================================================================
COLORS = {
    "amr":     {"bg": "#EBF3FD", "border": "#2563EB", "text": "#1E3A8A", "accent": "#3B82F6"},
    "biofilm": {"bg": "#EDE9FE", "border": "#7C3AED", "text": "#3B0764", "accent": "#8B5CF6"},
    "oncology":{"bg": "#FEF2F2", "border": "#DC2626", "text": "#7F1D1D", "accent": "#EF4444"},
    "ensemble":{"bg": "#FFFBEB", "border": "#D97706", "text": "#451A03", "accent": "#F59E0B"},
    "healthy": {"bg": "#F0FDF4", "border": "#16A34A", "text": "#14532D", "accent": "#22C55E"},
}

RISK_GAUGE_COLORS = {
    "Low":      "#22C55E",
    "Moderate": "#F59E0B",
    "High":     "#EF4444",
    "Critical": "#7F1D1D",
    "Very_High":"#7F1D1D",
}


def _risk_color(tier: str) -> str:
    return RISK_GAUGE_COLORS.get(tier, "#6B7280")


def _confidence_badge(conf: float) -> str:
    if conf >= 85:
        color = "#22C55E"; label = "High confidence"
    elif conf >= 65:
        color = "#F59E0B"; label = "Moderate confidence"
    else:
        color = "#EF4444"; label = "Low confidence — flag for review"
    return (f'<span style="background:{color}20;color:{color};'
            f'border:1px solid {color};border-radius:12px;'
            f'padding:2px 10px;font-size:12px;font-weight:600;">'
            f'{label} ({conf:.0f}%)</span>')


def _shap_bar(label: str, value: float, max_val: float, color: str) -> str:
    pct = min(abs(value) / max(max_val, 0.01) * 100, 100)
    direction = "+" if value >= 0 else "−"
    return f"""
    <div style="display:flex;align-items:center;gap:10px;margin:4px 0;">
      <span style="min-width:130px;font-size:12px;color:#374151;text-align:right">{label}</span>
      <div style="flex:1;background:#F3F4F6;border-radius:4px;height:14px;overflow:hidden;">
        <div style="width:{pct:.1f}%;background:{color};height:100%;border-radius:4px;
                    transition:width .4s;"></div>
      </div>
      <span style="font-size:12px;color:{color};font-weight:600;min-width:40px">
        {direction}{abs(value):.2f}</span>
    </div>"""


def _panel_amr(row: dict, shap: dict) -> str:
    tier  = str(row.get("AMR_Resistance_Profile", "Susceptible") or "Susceptible")
    genes = str(row.get("AMR_Detected_Genes", "None") or "None")
    fail  = str(row.get("AMR_Failed_Antibiotics", "None") or "None")
    alt   = str(row.get("AMR_Recommended_Alt", "None") or "None")
    score = float(row.get("AI_AMR_Score", 0) or 0)
    conf  = float(row.get("MC_AMR_Conf_pct", 70) or 70)
    c     = COLORS["amr"]

    amr_shap_keys = [k for k in shap if any(g in k for g in ["blaNDM","mecA","vanA","KPC"])]
    max_shap = max((abs(shap[k]) for k in amr_shap_keys), default=1)
    shap_bars = "".join(
        _shap_bar(k.replace("SHAP_",""), shap[k], max_shap, c["accent"])
        for k in amr_shap_keys
    )

    gene_tags = ("".join(
        f'<span style="background:{c["accent"]}20;color:{c["text"]};'
        f'border:1px solid {c["border"]};border-radius:8px;'
        f'padding:2px 10px;font-size:12px;margin:2px;display:inline-block">{g}</span>'
        for g in genes.split("|")) if genes != "None" else
        '<span style="color:#6B7280;font-size:12px;">No resistance genes detected</span>')

    return f"""
    <div style="background:{c['bg']};border:1.5px solid {c['border']};
                border-radius:14px;padding:20px;flex:1;min-width:240px;">
      <div style="display:flex;justify-content:space-between;align-items:center;
                  margin-bottom:12px;">
        <h3 style="color:{c['text']};margin:0;font-size:16px;">&#x1F9EC; AMR Risk</h3>
        <span style="background:{c['accent']};color:white;border-radius:8px;
                     padding:4px 12px;font-size:13px;font-weight:700;">
          {score:.0f}%</span>
      </div>
      {_confidence_badge(conf)}
      <hr style="border:none;border-top:1px solid {c['border']};margin:12px 0;opacity:.3">

      <div style="margin-bottom:10px;">
        <div style="color:{c['text']};font-size:11px;font-weight:600;
                    text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px">
          Resistance Profile</div>
        <div style="font-size:14px;font-weight:600;color:{c['text']}">{tier}</div>
      </div>

      <div style="margin-bottom:10px;">
        <div style="color:{c['text']};font-size:11px;font-weight:600;
                    text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px">
          Genes Detected</div>
        <div>{gene_tags}</div>
      </div>

      <div style="margin-bottom:10px;">
        <div style="color:{c['text']};font-size:11px;font-weight:600;
                    text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px">
          Predicted Resistance</div>
        <div style="font-size:13px;color:#374151">
          {fail if fail != "None" else "No resistance predicted"}</div>
      </div>

      <div style="background:white;border-radius:8px;padding:10px;
                  border:1px solid {c['border']}30;margin-bottom:10px;">
        <div style="font-size:11px;font-weight:600;color:{c['text']};margin-bottom:4px">
          Recommended Alternative</div>
        <div style="font-size:13px;color:#374151;font-style:{'italic' if alt == 'None' else 'normal'}">
          {alt if alt != "None" else "Standard empirical therapy"}</div>
      </div>

      <div style="margin-top:12px;">
        <div style="font-size:11px;font-weight:600;color:{c['text']};
                    text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px">
          SHAP Drivers</div>
        {shap_bars if shap_bars else '<span style="font-size:12px;color:#9CA3AF">No significant drivers</span>'}
      </div>
    </div>"""


def _panel_biofilm(row: dict, shap: dict) -> str:
    stage = str(row.get("Biofilm_Stage", "None") or "None")
    qs    = float(row.get("QS_Activity_Score", 0) or 0)
    ai2   = float(row.get("AI2_Level_uM", 0) or 0)
    cdgmp = float(row.get("c_diGMP_Level_uM", 0) or 0)
    ica   = int(row.get("icaADBC_Active", 0) or 0)
    pelps = int(row.get("pel_psl_Active", 0) or 0)
    score = float(row.get("AI_Biofilm_Score", 0) or 0)
    conf  = float(row.get("MC_Biofilm_Conf_pct", 70) or 70)
    c     = COLORS["biofilm"]

    stage_num = {"None":0,"Stage_I_early_attachment":1,"Stage_II_microcolony":2,
                 "Stage_III_maturation":3,"Stage_IV_dispersion":4}.get(stage, 0)
    stage_label = stage.replace("_", " ")
    stages = ["None","I","II","III","IV"]
    stage_pips = "".join(
        f'<div style="width:28px;height:28px;border-radius:50%;'
        f'background:{c["accent"] if i <= stage_num else "#E5E7EB"};'
        f'color:{"white" if i <= stage_num else "#9CA3AF"};'
        f'display:flex;align-items:center;justify-content:center;'
        f'font-size:11px;font-weight:700">{stages[i]}</div>'
        for i in range(5))

    bf_shap_keys = [k for k in shap if any(g in k for g in ["ica","AHL","bap","cdiGMP","AHL_AI2","bap_pel"])]
    max_shap = max((abs(shap[k]) for k in bf_shap_keys), default=1)
    shap_bars = "".join(
        _shap_bar(k.replace("SHAP_",""), shap[k], max_shap, c["accent"])
        for k in bf_shap_keys)

    return f"""
    <div style="background:{c['bg']};border:1.5px solid {c['border']};
                border-radius:14px;padding:20px;flex:1;min-width:240px;">
      <div style="display:flex;justify-content:space-between;align-items:center;
                  margin-bottom:12px;">
        <h3 style="color:{c['text']};margin:0;font-size:16px;">&#x1F9AB; Biofilm Status</h3>
        <span style="background:{c['accent']};color:white;border-radius:8px;
                     padding:4px 12px;font-size:13px;font-weight:700;">
          {score:.0f}%</span>
      </div>
      {_confidence_badge(conf)}
      <hr style="border:none;border-top:1px solid {c['border']};margin:12px 0;opacity:.3">

      <div style="margin-bottom:12px;">
        <div style="font-size:11px;font-weight:600;color:{c['text']};
                    text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px">
          Biofilm Stage</div>
        <div style="display:flex;gap:6px;align-items:center">{stage_pips}</div>
        <div style="font-size:13px;color:{c['text']};margin-top:6px;font-weight:500">
          {stage_label}</div>
      </div>

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px;">
        <div style="background:white;border-radius:8px;padding:8px;
                    border:1px solid {c['border']}30;text-align:center">
          <div style="font-size:18px;font-weight:700;color:{c['text']}">{qs:.0f}</div>
          <div style="font-size:10px;color:#6B7280">QS Activity Score</div>
        </div>
        <div style="background:white;border-radius:8px;padding:8px;
                    border:1px solid {c['border']}30;text-align:center">
          <div style="font-size:18px;font-weight:700;color:{c['text']}">{cdgmp:.2f}</div>
          <div style="font-size:10px;color:#6B7280">c-di-GMP (µM)</div>
        </div>
        <div style="background:white;border-radius:8px;padding:8px;
                    border:1px solid {c['border']}30;text-align:center">
          <div style="font-size:18px;font-weight:700;color:{c['text']}">{ai2:.2f}</div>
          <div style="font-size:10px;color:#6B7280">AI-2 Level (µM)</div>
        </div>
        <div style="background:white;border-radius:8px;padding:8px;
                    border:1px solid {c['border']}30;text-align:center">
          <div style="font-size:18px;font-weight:700;
                      color:{'#DC2626' if ica else '#22C55E'}">
            {'&#x2714;' if ica else '&#x2715;'}</div>
          <div style="font-size:10px;color:#6B7280">icaADBC</div>
        </div>
      </div>

      <div style="background:white;border-radius:8px;padding:8px;
                  border:1px solid {c['border']}30;margin-bottom:12px">
        <span style="font-size:12px;color:{c['text']};font-weight:600">Matrix genes: </span>
        <span style="font-size:12px;color:{'#DC2626' if pelps else '#6B7280'}">
          pel / psl {'&#x2714; Active' if pelps else 'Not detected'}</span>
      </div>

      <div>
        <div style="font-size:11px;font-weight:600;color:{c['text']};
                    text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px">
          SHAP Drivers</div>
        {shap_bars if shap_bars else '<span style="font-size:12px;color:#9CA3AF">No significant drivers</span>'}
      </div>
    </div>"""


def _panel_oncology(row: dict, shap: dict) -> str:
    tier    = str(row.get("Oncology_Risk_Tier", "Low") or "Low")
    species = str(row.get("Oncology_Species", "None") or "None")
    microbes= str(row.get("Cancer_Microbes_Detected", "None") or "None")
    inflam  = float(row.get("Chronic_Inflammation_Load", 0) or 0)
    referral= int(row.get("Referral_Recommended", 0) or 0)
    score   = float(row.get("AI_Oncology_Score", 0) or 0)
    conf    = float(row.get("MC_Oncology_Conf_pct", 70) or 70)
    c       = COLORS["oncology"]

    tier_color = _risk_color(tier)
    sp_tags = ("".join(
        f'<span style="background:{c["accent"]}20;color:{c["text"]};'
        f'border:1px solid {c["border"]};border-radius:8px;'
        f'padding:2px 10px;font-size:12px;margin:2px;display:inline-block">{s.strip()}</span>'
        for s in species.split("|")) if species != "None" else
        '<span style="color:#6B7280;font-size:12px;">No oncology species detected</span>')

    onc_shap_keys = [k for k in shap if any(g in k for g in ["FadA","CagA","pks","miRNA"])]
    max_shap = max((abs(shap[k]) for k in onc_shap_keys), default=1)
    shap_bars = "".join(
        _shap_bar(k.replace("SHAP_",""), shap[k], max_shap, c["accent"])
        for k in onc_shap_keys)

    inflam_pct = min(inflam / 10 * 100, 100)

    referral_box = f"""
    <div style="background:#FEF2F2;border:2px solid #DC2626;border-radius:10px;
                padding:12px;margin-top:10px;display:flex;align-items:center;gap:10px">
      <span style="font-size:24px">&#x1F6A8;</span>
      <div>
        <div style="font-weight:700;color:#7F1D1D;font-size:14px">Referral Recommended</div>
        <div style="font-size:12px;color:#991B1B">Oncology / Gastroenterology review advised</div>
      </div>
    </div>""" if referral else ""

    return f"""
    <div style="background:{c['bg']};border:1.5px solid {c['border']};
                border-radius:14px;padding:20px;flex:1;min-width:240px;">
      <div style="display:flex;justify-content:space-between;align-items:center;
                  margin-bottom:12px;">
        <h3 style="color:{c['text']};margin:0;font-size:16px;">&#x1F52C; Oncology Risk</h3>
        <span style="background:{c['accent']};color:white;border-radius:8px;
                     padding:4px 12px;font-size:13px;font-weight:700;">
          {score:.0f}%</span>
      </div>
      {_confidence_badge(conf)}
      <hr style="border:none;border-top:1px solid {c['border']};margin:12px 0;opacity:.3">

      <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
        <div style="background:{tier_color};color:white;border-radius:10px;
                    padding:8px 16px;font-weight:700;font-size:14px;letter-spacing:.04em">
          {tier.replace("_"," ")}</div>
        <div style="font-size:12px;color:#6B7280">Risk Tier</div>
      </div>

      <div style="margin-bottom:12px;">
        <div style="font-size:11px;font-weight:600;color:{c['text']};
                    text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px">
          Implicated Species</div>
        {sp_tags}
      </div>

      <div style="margin-bottom:12px;">
        <div style="font-size:11px;font-weight:600;color:{c['text']};margin-bottom:4px">
          Chronic Inflammation Load</div>
        <div style="background:#F3F4F6;border-radius:6px;height:10px;overflow:hidden;">
          <div style="width:{inflam_pct:.1f}%;background:{tier_color};height:100%;
                      border-radius:6px;"></div>
        </div>
        <div style="font-size:12px;color:#6B7280;margin-top:4px">{inflam:.1f} / 10</div>
      </div>

      <div>
        <div style="font-size:11px;font-weight:600;color:{c['text']};
                    text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px">
          SHAP Drivers</div>
        {shap_bars if shap_bars else '<span style="font-size:12px;color:#9CA3AF">No significant drivers</span>'}
      </div>
      {referral_box}
    </div>"""


def _ensemble_gauge(ensemble_score: float, tier: str,
                    n_domains: int, co_bonus: float) -> str:
    color  = _risk_color(tier)
    pct    = min(ensemble_score, 100)
    # SVG arc gauge
    r, cx, cy = 60, 80, 80
    circ = 2 * 3.14159 * r
    dash = circ * pct / 100
    return f"""
    <div style="background:#FFFBEB;border:1.5px solid #D97706;border-radius:14px;
                padding:20px;margin-bottom:20px;display:flex;
                align-items:center;gap:24px;flex-wrap:wrap">
      <div style="text-align:center">
        <svg width="160" height="160" viewBox="0 0 160 160">
          <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
                  stroke="#E5E7EB" stroke-width="14" stroke-dasharray="{circ}" />
          <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
                  stroke="{color}" stroke-width="14"
                  stroke-dasharray="{dash:.1f} {circ:.1f}"
                  stroke-linecap="round"
                  transform="rotate(-90 {cx} {cy})" />
          <text x="{cx}" y="{cy-8}" text-anchor="middle"
                font-size="22" font-weight="700" fill="{color}">{pct:.0f}</text>
          <text x="{cx}" y="{cy+14}" text-anchor="middle"
                font-size="11" fill="#6B7280">/ 100</text>
        </svg>
        <div style="font-size:13px;color:#6B7280;margin-top:-8px">Ensemble Risk Score</div>
      </div>
      <div style="flex:1">
        <div style="font-size:22px;font-weight:800;color:{color};margin-bottom:4px">
          {tier.replace("_"," ")}</div>
        <div style="font-size:13px;color:#374151;margin-bottom:12px">
          Overall clinical risk tier</div>
        <div style="display:flex;gap:16px;flex-wrap:wrap">
          <div style="background:white;border-radius:8px;padding:8px 16px;
                      border:1px solid #E5E7EB;text-align:center">
            <div style="font-size:18px;font-weight:700;color:#374151">{n_domains}</div>
            <div style="font-size:11px;color:#9CA3AF">Active Domains</div>
          </div>
          <div style="background:white;border-radius:8px;padding:8px 16px;
                      border:1px solid #E5E7EB;text-align:center">
            <div style="font-size:18px;font-weight:700;color:#D97706">+{co_bonus:.0f}</div>
            <div style="font-size:11px;color:#9CA3AF">Co-occurrence Bonus</div>
          </div>
        </div>
        <div style="margin-top:12px;font-size:12px;color:#6B7280">
          Ensemble = 0.40 x AMR + 0.35 x Biofilm + 0.25 x Oncology + Co-occurrence
        </div>
      </div>
    </div>"""


# =============================================================================
# Main Report Generator
# =============================================================================

class ClinicalReportGenerator:
    """
    SVG Layers 4 + 5.
    Generates a standalone HTML clinical decision support report
    for one patient sample.
    """

    def generate_single(self, row: dict, shap_row: dict = None,
                        out_path: Path = None) -> str:
        """
        row      : dict of all columns for one sample
        shap_row : dict of SHAP values for that sample
        out_path : if provided, save HTML to file
        Returns  : HTML string
        """
        shap = shap_row or {}
        now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Header meta
        sample_id  = row.get("Sample_ID", "UNKNOWN")
        age        = row.get("Patient_Age", "—")
        sex        = row.get("Patient_Sex", "—")
        bmi        = row.get("Patient_BMI", "—")
        stype      = row.get("Sample_Type", "—")
        site       = row.get("Collection_Site", "—")
        cdate      = row.get("Collection_Date", "—")
        ctime      = row.get("Collection_Time", "—")
        device_id  = row.get("Device_ID", "—")
        lot        = row.get("Cartridge_Lot", "—")
        t_mode     = row.get("Transmission_Mode", "—")
        t_total    = row.get("Total_Time_min", "—")

        ensemble_score = float(row.get("Ensemble_Risk_Score", 0))
        ensemble_tier  = str(row.get("Ensemble_Risk_Tier", "Low"))
        n_domains      = int(row.get("N_Active_Domains", 0))
        co_bonus       = float(row.get("Co_occurrence_Bonus", 0))
        t_mode_icon    = ("&#x1F4F6;" if t_mode == "BLE" else
                           "&#x1F4F6;" if t_mode == "WiFi" else "&#x1F4BB;")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Biosensor Clinical Report — {sample_id}</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
          background:#F9FAFB;color:#111827;line-height:1.5; }}
  .container {{ max-width:1100px;margin:0 auto;padding:24px; }}
  .card {{ background:white;border-radius:16px;box-shadow:0 1px 3px rgba(0,0,0,.08);
           border:1px solid #E5E7EB;padding:24px;margin-bottom:20px; }}
  table {{ width:100%;border-collapse:collapse; }}
  td,th {{ padding:6px 12px;text-align:left;font-size:13px; }}
  th {{ color:#6B7280;font-weight:500;font-size:11px;text-transform:uppercase;letter-spacing:.06em; }}
  @media print {{ body {{ background:white }} .container {{ padding:0 }} .no-print {{ display:none !important }} }}
</style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <div style="background:linear-gradient(135deg,#1E3A8A,#3730A3);color:white;
              border-radius:16px;padding:24px;margin-bottom:20px;
              display:flex;justify-content:space-between;flex-wrap:wrap;gap:12px">
    <div>
      <div style="font-size:11px;opacity:.7;text-transform:uppercase;letter-spacing:.08em">
        AI-Enabled Multiplex Biosensor System</div>
      <h1 style="font-size:22px;font-weight:800;margin:4px 0">Clinical Decision Support Report</h1>
      <div style="font-size:13px;opacity:.8">Sample ID: <strong>{sample_id}</strong></div>
    </div>
    <div style="text-align:right;opacity:.85">
      <div style="font-size:12px">Report generated: {now}</div>
      <div style="font-size:12px">Collection: {cdate} {ctime}</div>
      <div style="font-size:12px">{t_mode_icon} Transmitted via {t_mode} &bull;
        Total time: {t_total} min</div>
      <button class="no-print" onclick="window.print()" style="margin-top:12px; background:rgba(255,255,255,0.15); border:1px solid rgba(255,255,255,0.3); color:white; padding:6px 14px; border-radius:6px; cursor:pointer; font-size:12px; font-weight:600; display:inline-flex; align-items:center; gap:6px; transition:all 0.2s;" onmouseover="this.style.background='rgba(255,255,255,0.25)'" onmouseout="this.style.background='rgba(255,255,255,0.15)'">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 6 2 18 2 18 9"></polyline><path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"></path><rect x="6" y="14" width="12" height="8"></rect></svg>
        Print Report
      </button>
    </div>
  </div>

  <!-- Patient + Device info -->
  <div class="card" style="margin-bottom:20px">
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;flex-wrap:wrap">
      <div>
        <div style="font-size:12px;font-weight:700;color:#374151;
                    text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px">
          Patient Information</div>
        <table>
          <tr><th>Age</th><td>{age}</td><th>Sex</th><td>{sex}</td></tr>
          <tr><th>BMI</th><td>{bmi}</td><th>Sample type</th><td>{stype}</td></tr>
          <tr><th>Collection site</th><td colspan="3">{site}</td></tr>
        </table>
      </div>
      <div>
        <div style="font-size:12px;font-weight:700;color:#374151;
                    text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px">
          Device Telemetry</div>
        <table>
          <tr><th>Device ID</th><td>{device_id}</td></tr>
          <tr><th>Cartridge lot</th><td>{lot}</td></tr>
          <tr><th>Total assay time</th><td>{t_total} min</td></tr>
          <tr><th>Transmission</th><td>{t_mode}</td></tr>
        </table>
      </div>
    </div>
  </div>

  <!-- Ensemble Gauge -->
  <div class="card">
    <div style="font-size:14px;font-weight:700;color:#374151;margin-bottom:14px">
      Integrated Clinical Risk Engine</div>
    {_ensemble_gauge(ensemble_score, ensemble_tier, n_domains, co_bonus)}
  </div>

  <!-- 3-Panel Clinical Report -->
  <div class="card">
    <div style="font-size:14px;font-weight:700;color:#374151;margin-bottom:14px">
      Clinical Decision Support — Three-Panel Report</div>
    <div style="display:flex;gap:16px;flex-wrap:wrap">
      {_panel_amr(row, shap)}
      {_panel_biofilm(row, shap)}
      {_panel_oncology(row, shap)}
    </div>
  </div>

  <!-- Footer -->
  <div style="text-align:center;color:#9CA3AF;font-size:11px;padding:16px 0">
    AI-Enabled Multiplex Biosensor &bull; For clinical decision support only &bull;
    Always verify with laboratory confirmation for critical decisions.
  </div>

</div>
</body>
</html>"""

        if out_path:
            Path(out_path).write_text(html, encoding="utf-8")

        return html

    def generate_batch(self, df_main, df_shap=None,
                       out_dir: Path = None, n_samples: int = 5) -> list:
        """Generate reports for the first n_samples rows."""
        out_dir = Path(out_dir) if out_dir else BASE_DIR / "reports"
        out_dir.mkdir(exist_ok=True)
        paths = []
        for i in range(min(n_samples, len(df_main))):
            row      = df_main.iloc[i].to_dict()
            shap_row = df_shap.iloc[i].to_dict() if df_shap is not None else {}
            sid      = row.get("Sample_ID", f"S{i:04d}")
            fpath    = out_dir / f"report_{sid}.html"
            self.generate_single(row, shap_row, out_path=fpath)
            paths.append(str(fpath))
        return paths
