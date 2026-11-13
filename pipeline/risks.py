"""
CareGuard standard-score risk states for four ICU risk domains. Pure rules, no model.

Domain      Standards used (all from published definitions)                              Tier mapping (0 Low, 1 Watch, 2 High, 3 Critical)
sepsis      SIRS, qSOFA, SOFA (Sepsis-3), septic-shock proxy (vasopressor + lactate>2)    shock proxy=3, qSOFA>=2 or (SOFA>=2 & SIRS>=2)=2, SIRS>=2 or qSOFA>=1=1
arf         ROX index (Roca 2019: <2.85/<3.47/<3.85 at 2/6/12 h, >=4.88 low risk),       P/F<100 or ventilated P/F<200 or ROX<2.85=3; P/F<200, S/F<235, ROX below time threshold,
            P/F and S/F bands, NEWS2 (RCP: >=5 urgent, >=7 emergency)                    NEWS2>=7 with respiratory points=2; P/F<300, S/F<315, ROX<4.88, SpO2<92, RR>=25=1
aki         KDIGO 2012 (creatinine ratio / 0.3 mg/dL rise, urine output 0.5/0.3 mL/kg/h)  KDIGO stage 3=3, 2=2, 1=1
shock       SCAI shock stage (2019, 2022 update), shock index, lactate, vasopressors      SCAI D/E=3, C=2, B=1

Limits, stated plainly:
 * ROX was validated on high-flow nasal cannula in pneumonia; here it is applied to any supplemental oxygen as a proxy.
 * No body weight in the data: urine-output thresholds assume 70 kg (0.5 mL/kg/h = 35 mL/h, 0.3 mL/kg/h = 21 mL/h).
 * SCAI staging is for cardiogenic shock and needs bedside/hemodynamic data (cardiac index, PA catheter, CPR, MCS) we do not have.
   We use a proxy from MAP/SBP, shock index, lactate, pressors and inotropes; it does not tell cardiogenic from septic shock.
 * No dialysis flag in the data, so KDIGO stage 3 by RRT is not detected.
"""
from __future__ import annotations
import numpy as np, pandas as pd

TIER_NAMES = ["Low", "Watch", "High", "Critical"]
PRESSORS = ["rx_norepi", "rx_epi", "rx_dopamine", "rx_vasopressin", "rx_phenylephrine"]
WEIGHT_KG = 70.0
DOMAINS = {"sepsis": "Sepsis", "arf": "Acute respiratory failure", "aki": "Acute kidney injury", "shock": "Circulatory shock"}
STANDARD = {"sepsis": "SIRS / qSOFA / SOFA", "arf": "ROX index / P-F / NEWS2", "aki": "KDIGO", "shock": "SCAI-style stage / shock index"}
SCAI = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}


def _col(p, c, default=np.nan):
    return p[c] if c in p else pd.Series(default, index=p.index)


def _roll(g, col, w, fn, minp):
    return g[col].transform(lambda s: getattr(s.rolling(w, min_periods=minp), fn)())


def add_risk_states(p: pd.DataFrame) -> pd.DataFrame:
    """Adds rox, kdigo, scai, and tier columns t_sepsis/t_arf/t_aki/t_shock (0..3). Input = hourly panel."""
    p = p.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    g = p.groupby("stay_id", sort=False)
    n_press = sum((_col(p, c, 0).fillna(0) > 0).astype(int) for c in PRESSORS)
    inotrope = _col(p, "rx_dobutamine", 0).fillna(0) > 0
    p["n_pressors"] = n_press
    lact = _col(p, "lactate").where(_col(p, "lactate_age_h", 0) <= 24)              # only a fresh lactate counts

    # ---- sepsis
    shock_s = (n_press > 0) & (lact > 2)
    high = (p.qsofa >= 2) | ((p.sofa >= 2) & (p.sirs >= 2))
    watch = (p.sirs >= 2) | (p.qsofa >= 1)
    p["t_sepsis"] = np.select([shock_s, high, watch], [3, 2, 1], 0)

    # ---- acute respiratory failure
    supp = _col(p, "o2_supp", 0).fillna(0) > 0
    vent = _col(p, "vent", 0).fillna(0) > 0
    run = supp.astype(int).groupby([p.stay_id, (~supp).cumsum()]).cumsum()          # consecutive hours on oxygen support
    fio2 = _col(p, "fio2") / 100.0
    p["rox"] = np.where(supp & ~vent & (fio2 > 0) & (p.rr > 0), (p.spo2 / fio2) / p.rr, np.nan)
    thr = np.where(run >= 12, 3.85, np.where(run >= 6, 3.47, 2.85))                 # Roca 2019 failure cut-offs by hours on support
    rox_fail = p.rox < thr
    pf, sf = _col(p, "pf_ratio"), _col(p, "sf_ratio")
    resp_pts = (_col(p, "n2_rr", 0).fillna(0) + _col(p, "n2_spo2", 0).fillna(0)) >= 3
    crit = (pf < 100) | (vent & (pf < 200)) | (p.rox < 2.85)
    high = (pf < 200) | (sf < 235) | rox_fail | ((p.news2 >= 7) & resp_pts)
    watch = (pf < 300) | (sf < 315) | (p.rox < 4.88) | (p.spo2 < 92) | (p.rr >= 25) | ((p.news2 >= 5) & resp_pts)
    p["t_arf"] = np.select([crit, high, watch], [3, 2, 1], 0)

    # ---- acute kidney injury (KDIGO)
    cr = p.creatinine
    base7 = _roll(g, "creatinine", 168, "min", 1)
    min48 = _roll(g, "creatinine", 48, "min", 1)
    ratio, rise = cr / base7, cr - min48
    s_cr = np.select([(ratio >= 3) | ((cr >= 4.0) & (rise >= 0.3)), ratio >= 2, (ratio >= 1.5) | (rise >= 0.3)], [3, 2, 1], 0)
    uo6, uo12, uo24 = (_roll(g, "urine_ml", w, "mean", int(w * .75)) for w in (6, 12, 24))
    lo, vlo = 0.5 * WEIGHT_KG, 0.3 * WEIGHT_KG
    s_uo = np.select([(uo24 < vlo) | (uo12 <= 0), uo12 < lo, uo6 < lo], [3, 2, 1], 0)
    p["kdigo"] = np.maximum(s_cr, s_uo)
    p["kdigo_cr"], p["kdigo_uo"], p["cr_ratio"] = s_cr, s_uo, ratio
    p["t_aki"] = p["kdigo"]

    # ---- circulatory shock, SCAI-style stage (0=A ... 4=E)
    hypo = (p.sbp < 90) | (p["map"] < 60) | (_col(p, "shock_index") >= 0.9)
    hypoperf = (lact >= 2) | (p.kdigo >= 2) | (p.gcs < 13) & hypo
    on_support = (n_press > 0) | inotrope
    lact_rise = lact - g["lactate"].shift(6)
    stage = np.zeros(len(p), dtype=int)
    stage = np.where(hypo, 1, stage)                                                # B: relative hypotension / tachycardia, no hypoperfusion
    stage = np.where(hypoperf & (hypo | on_support), 2, stage)                      # C: hypoperfusion needing more than volume
    stage = np.where((stage >= 2) & ((n_press >= 2) | ((n_press >= 1) & (lact_rise > 0.5) & (lact >= 4))), 3, stage)   # D: failing / escalating
    stage = np.where((n_press >= 2) & ((lact >= 8) | (_col(p, "ph") <= 7.2)), 4, stage)                                # E: extremis proxy
    p["scai"] = stage
    p["t_shock"] = np.select([stage >= 3, stage == 2, stage == 1], [3, 2, 1], 0)
    return p


# ---------------------------------------------------------------- human-readable evidence (computed only for rows we show)
def _f(x, nd=1):
    return "n/a" if pd.isna(x) else f"{x:.{nd}f}"


def evidence_backend(r, domain):
    if domain == "sepsis":
        s = f"qSOFA {_f(r.qsofa,0)}/3, SIRS {_f(r.sirs,0)}/4, SOFA {_f(r.sofa,0)}"
        if r.t_sepsis == 3:
            s += f", vasopressor with lactate {_f(r.lactate)}"
        return s
    if domain == "arf":
        bits = []
        if pd.notna(r.pf_ratio):
            bits.append(f"P/F {_f(r.pf_ratio,0)}")
        elif pd.notna(r.sf_ratio):
            bits.append(f"S/F {_f(r.sf_ratio,0)}")
        if pd.notna(r.rox):
            bits.append(f"ROX {_f(r.rox,2)}")
        bits += [f"SpO2 {_f(r.spo2,0)}%", f"RR {_f(r.rr,0)}", f"NEWS2 {_f(r.news2,0)}"]
        return ", ".join(bits)
    if domain == "aki":
        return f"stage {int(r.kdigo)} (creatinine {_f(r.creatinine,2)}, x{_f(r.cr_ratio,1)} of baseline; urine {_f(r.urine_ml,0)} mL last hour; 70 kg assumed)"
    if domain == "shock":
        bits = [f"SCAI-style {SCAI[int(r.scai)]}", f"shock index {_f(r.shock_index,2)}", f"MAP {_f(r['map'],0)}"]
        if pd.notna(r.lactate):
            bits.append(f"lactate {_f(r.lactate)}")
        if r.n_pressors:
            bits.append(f"{int(r.n_pressors)} vasopressor(s)")
        return ", ".join(bits)
    return ""


# ---------------------------------------------------------------- plain-language reasons (what patients and doctors see)
# Only measured vitals, labs and treatments. Index names and scores (qSOFA, ROX, KDIGO, SCAI...) stay in the backend columns.
def _ok(x):
    return pd.notna(x)


def reason_text(r, domain, max_items=5):
    """Abnormal measurements behind the tier, most relevant first, e.g. 'resp rate 28/min, SpO2 92%, lactate 3.4 mmol/L'."""
    it = []
    def add(cond, txt):
        if cond:
            it.append(txt)
    g = lambda c: r[c] if c in r.index else np.nan
    if domain == "sepsis":
        add(_ok(g("lactate")) and g("lactate") > 2, f"lactate {g('lactate'):.1f} mmol/L")
        add(g("n_pressors") > 0, "on vasopressor medication")
        add(_ok(g("sbp")) and g("sbp") <= 100, f"systolic BP {g('sbp'):.0f}")
        add(_ok(g("rr")) and g("rr") >= 22, f"resp rate {g('rr'):.0f}/min")
        add(_ok(g("gcs")) and g("gcs") < 15, f"reduced consciousness (GCS {g('gcs'):.0f})")
        add(_ok(g("temp_c")) and (g("temp_c") > 38 or g("temp_c") < 36), f"temperature {g('temp_c'):.1f} C")
        add(_ok(g("hr")) and g("hr") > 90, f"heart rate {g('hr'):.0f}")
        add(_ok(g("wbc")) and (g("wbc") > 12 or g("wbc") < 4), f"WBC {g('wbc'):.1f}")
        add(_ok(g("platelets")) and g("platelets") < 150, f"platelets {g('platelets'):.0f}")
        add(_ok(g("creatinine")) and g("creatinine") > 1.2, f"creatinine {g('creatinine'):.1f}")
    elif domain == "arf":
        add(_ok(g("spo2")) and g("spo2") < 92, f"SpO2 {g('spo2'):.0f}%")
        add(_ok(g("pao2")) and g("pao2") < 80, f"PaO2 {g('pao2'):.0f} mmHg")
        add(_ok(g("fio2")) and g("fio2") > 21 and g("o2_supp") > 0, f"needs {g('fio2'):.0f}% oxygen")
        add(_ok(g("rr")) and g("rr") >= 25, f"resp rate {g('rr'):.0f}/min")
        add(g("vent") > 0, "on ventilator")
        add(_ok(g("paco2")) and g("paco2") > 50, f"CO2 {g('paco2'):.0f} mmHg")
    elif domain == "aki":
        add(_ok(g("creatinine")) and g("cr_ratio") >= 1.5, f"creatinine {g('creatinine'):.1f} mg/dL, {g('cr_ratio'):.1f}x the patient's lowest recent value")
        add(g("kdigo_uo") >= 1 and _ok(g("urine_ml")), f"low urine output (last hour {g('urine_ml'):.0f} mL)")
        add(_ok(g("bun")) and g("bun") > 40, f"BUN {g('bun'):.0f}")
        add(_ok(g("potassium")) and g("potassium") > 5.5, f"potassium {g('potassium'):.1f}")
    elif domain == "shock":
        add(_ok(g("lactate")) and g("lactate") >= 2, f"lactate {g('lactate'):.1f} mmol/L")
        add(_ok(g("map")) and g("map") < 65, f"mean BP {g('map'):.0f}")
        add(_ok(g("sbp")) and g("sbp") < 90, f"systolic BP {g('sbp'):.0f}")
        add(_ok(g("shock_index")) and g("shock_index") >= 0.9 and not (_ok(g("sbp")) and g("sbp") < 90), f"heart rate {g('hr'):.0f} with systolic BP {g('sbp'):.0f}")
        add(_ok(g("hr")) and g("hr") > 100 and _ok(g("shock_index")) and g("shock_index") < 0.9, f"heart rate {g('hr'):.0f}")
        add(g("kdigo") >= 2, "kidney function worsening")
        add(g("n_pressors") > 0, f"{int(g('n_pressors'))} vasopressor(s) running")
        add(_ok(g("rx_dobutamine")) and g("rx_dobutamine") > 0, "inotrope running")
        add(_ok(g("gcs")) and g("gcs") < 13, f"reduced consciousness (GCS {g('gcs'):.0f})")
        add(_ok(g("urine_ml")) and g("urine_ml") < 30, f"low urine output ({g('urine_ml'):.0f} mL last hour)")
    return ", ".join(it[:max_items]) if it else "values near the alert range"
