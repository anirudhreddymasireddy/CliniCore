"""
scores.py - standard clinical scores on MIMIC-IV-schema tables (real MIMIC-IV or synthetic).

NEWS2 (RCP 2017, SpO2 scale 1; scale 2 optional), SOFA (6 organs), qSOFA, SIRS,
and a Sepsis-3 proxy (suspected infection + SOFA rise >= 2).

Design rules
  * Works only on MIMIC-IV schema tables. It never looks at synthetic-only columns, so the
    same code runs unchanged on the real data.
  * Never silently invents values. Missing inputs give NaN points and are counted
    (n_missing); every approximation is named in APPROXIMATIONS below.
  * Implausible values (HR 0, MAP 801, temp 99 C ...) are set to NaN and counted in a QC report,
    not dropped silently.

Entry points
  news2_points(...)            vectorised, works on arrays/Series
  sofa_components(...)         vectorised
  qsofa / sirs                 vectorised
  build_hourly_panel(tables)   hourly ICU panel (one row per stay-hour) with all signals + scores
  sepsis3_proxy(tables, panel) per-stay suspected-infection time + sepsis onset time
  add_sepsis_label(panel, onset, horizon=6)  label_6h (onset within next H hours, onset-time-aware)
  news2_from_triage(triage)    ED stage: NEWS2 from triage vitals (consciousness/O2 unknown)
"""
from __future__ import annotations
import re
import warnings
import numpy as np
import pandas as pd

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

APPROXIMATIONS = {
    "news2_consciousness": "GCS<15 treated as NOT 'Alert' (3 points). MIMIC has no ACVPU field.",
    "news2_o2": "Supplemental O2 = recorded O2 delivery device other than none/room air (itemid 226732), "
                "falling back to FiO2>21%.",
    "sofa_resp": "PaO2/FiO2 when a PaO2 within 24h exists; otherwise SpO2/FiO2 surrogate "
                 "(Pandharipande 2009 thresholds 148/235/315/357, SpO2<=97 only).",
    "sofa_vent": "Ventilatory support = endotracheal tube, tracheostomy, BiPAP/CPAP in O2 delivery device.",
    "sofa_cv": "Norepi/epi/dopamine/dobutamine per SOFA table (mcg/kg/min). Vasopressin or phenylephrine "
               "alone are scored 3 (not in the original table).",
    "sofa_renal": "Creatinine; urine output <500 mL/24h scores 3 and <200 mL/24h scores 4 only when "
                  ">=20 h of the last 24 h have urine charted.",
    "sofa_missing": "A missing SOFA component counts 0 (standard practice), but n_sofa_missing is reported.",
    "sepsis3": "Suspected infection = antibiotic and culture within 24h (abx first) / 72h (culture first), "
               "Seymour 2016. Sepsis onset = first hour in [-48h,+24h] around it when SOFA >= baseline+2 "
               "(baseline = min SOFA in the 24h before the window start, else 0).",
}

# ----------------------------------------------------------------------------- itemids
ITEM = dict(
    hr=[220045], rr=[220210, 224690], spo2=[220277],
    sbp_art=[220050], sbp_nibp=[220179], dbp_art=[220051], dbp_nibp=[220180],
    map_art=[220052, 225312], map_nibp=[220181],
    temp_f=[223761], temp_c=[223762],
    gcs_eye=[220739], gcs_verbal=[223900], gcs_motor=[223901],
    fio2=[223835], o2_device=[226732],
)
LAB = dict(  # itemid -> name (blood)
    creatinine=[50912], bilirubin=[50885], platelets=[51265], wbc=[51301], lactate=[50813],
    pao2=[50821], paco2=[50818], ph=[50820], hco3=[50882], hgb=[51222], inr=[51237], glucose=[50931],
    sodium=[50983], potassium=[50971], bun=[51006], alt=[50861], ast=[50878],
)
PRESSOR = {221906: "norepi", 221289: "epi", 221662: "dopamine", 221653: "dobutamine",
           222315: "vasopressin", 221749: "phenylephrine"}
URINE_ITEMS = [226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557, 226558, 227488, 227489]
ABX_RE = re.compile(r"vancomycin|piperacillin|cefepime|ceftriaxone|ceftazidime|cefazolin|meropenem|imipenem|"
                    r"ertapenem|azithromycin|levofloxacin|ciprofloxacin|metronidazole|ampicillin|gentamicin|"
                    r"tobramycin|amikacin|linezolid|daptomycin|clindamycin|doxycycline|sulfamethoxazole|"
                    r"aztreonam|penicillin|nafcillin|oxacillin|cefpodoxime|cephalexin|amoxicillin", re.I)
INVASIVE_VENT_RE = re.compile(r"endotracheal|tracheostomy|bipap|cpap|vent", re.I)
NO_O2_RE = re.compile(r"^(none|room air|ra|nasal cannula off)?$", re.I)

PLAUSIBLE = {  # (lo, hi) inclusive; outside -> NaN (counted in qc)
    "hr": (10, 300), "rr": (2, 80), "spo2": (40, 100), "sbp": (30, 300), "dbp": (10, 200),
    "map": (20, 250), "temp_c": (25, 45), "fio2": (21, 100), "gcs_eye": (1, 4), "gcs_verbal": (1, 5),
    "gcs_motor": (1, 6),
    "creatinine": (0.1, 30), "bilirubin": (0.05, 80), "platelets": (1, 2000), "wbc": (0.1, 500),
    "lactate": (0.2, 40), "pao2": (15, 700), "paco2": (5, 200), "ph": (6.5, 7.9), "hco3": (2, 60),
    "hgb": (2, 25), "inr": (0.5, 20), "glucose": (10, 2000),
    "sodium": (90, 190), "potassium": (1.0, 10), "bun": (1, 300), "alt": (1, 20000), "ast": (1, 20000),
}


# ----------------------------------------------------------------------------- NEWS2
def _pts(x, edges, pts_list):
    """x < edges[i] -> pts_list[i]; else pts_list[-1]. NaN stays NaN."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, pts_list[-1], dtype=float)
    for e, p in reversed(list(zip(edges, pts_list[:-1]))):
        out = np.where(x < e, p, out)
    return np.where(np.isnan(x), np.nan, out)


def news2_points(rr, spo2, sbp, hr, temp_c, gcs=None, o2_supp=None, scale2=False, consciousness_abnormal=None):
    """Return DataFrame with component points, total (NaN-skipping), n_missing and risk band.
    Bands: 0-4 low, any single 3 -> low-medium, 5-6 medium, >=7 high.
    scale2=True uses SpO2 scale 2 (target 88-92 for hypercapnic failure): the caller must decide."""
    rr, spo2, sbp, hr, temp_c = [np.asarray(v, dtype=float) for v in (rr, spo2, sbp, hr, temp_c)]
    p_rr = _pts(rr, [9, 12, 21, 25], [3, 1, 0, 2, 3])
    if scale2:
        # <=83:3, 84-85:2, 86-87:1, 88-92 (or >=93 on air):0, 93-94 on O2:1, 95-96 on O2:2, >=97 on O2:3
        on_o2 = np.asarray(o2_supp, dtype=float) == 1 if o2_supp is not None else np.zeros_like(spo2, bool)
        p_sp = np.select(
            [np.isnan(spo2), spo2 < 84, spo2 < 86, spo2 < 88, spo2 < 93, (spo2 < 95) & on_o2,
             (spo2 < 97) & on_o2, on_o2],
            [np.nan, 3, 2, 1, 0, 1, 2, 3], default=0)
    else:
        p_sp = _pts(spo2, [92, 94, 96], [3, 2, 1, 0])
    p_sbp = _pts(sbp, [91, 101, 111, 220], [3, 2, 1, 0, 3])
    p_hr = _pts(hr, [41, 51, 91, 111, 131], [3, 1, 0, 1, 2, 3])
    t = temp_c
    p_t = np.select([np.isnan(t), t <= 35.0, t <= 36.0, t <= 38.0, t <= 39.0], [np.nan, 3, 1, 0, 1], default=2)
    if consciousness_abnormal is not None:
        p_c = np.where(np.isnan(np.asarray(consciousness_abnormal, float)), np.nan,
                       np.asarray(consciousness_abnormal, float) * 3)
    elif gcs is not None:
        g = np.asarray(gcs, dtype=float)
        p_c = np.where(np.isnan(g), np.nan, np.where(g < 15, 3, 0))
    else:
        p_c = np.full(rr.shape, np.nan)
    if o2_supp is None:
        p_o = np.full(rr.shape, np.nan)
    else:
        o = np.asarray(o2_supp, dtype=float)
        p_o = np.where(np.isnan(o), np.nan, np.where(o == 1, 2, 0))
    comp = pd.DataFrame({"n2_rr": p_rr, "n2_spo2": p_sp, "n2_sbp": p_sbp, "n2_hr": p_hr, "n2_temp": p_t,
                         "n2_cons": p_c, "n2_o2": p_o})
    comp["news2"] = comp.sum(axis=1, min_count=1)
    comp["news2_n_missing"] = comp[["n2_rr", "n2_spo2", "n2_sbp", "n2_hr", "n2_temp", "n2_cons", "n2_o2"]].isna().sum(axis=1)
    mx = comp[["n2_rr", "n2_spo2", "n2_sbp", "n2_hr", "n2_temp", "n2_cons"]].max(axis=1)
    comp["news2_band"] = np.select([comp.news2.isna(), comp.news2 >= 7, comp.news2 >= 5, mx >= 3],
                                   ["unknown", "high", "medium", "low_medium"], default="low")
    return comp


def news2_from_triage(tr: pd.DataFrame) -> pd.DataFrame:
    """ED triage table (temperature in F). Consciousness and O2 unknown -> reported as missing (3 of 7 vitals
    needed to score is NOT enforced; the caller sees news2_n_missing)."""
    t_raw = pd.to_numeric(tr["temperature"], errors="coerce")
    # real MIMIC-IV-ED has Celsius values (e.g. 36.5) in the Fahrenheit column: 25-45 cannot be Fahrenheit -> keep as Celsius
    t_c = pd.Series(np.where((t_raw >= 25) & (t_raw <= 45), t_raw, (t_raw - 32) * 5 / 9), index=tr.index)
    t_c = t_c.where((t_c >= PLAUSIBLE["temp_c"][0]) & (t_c <= PLAUSIBLE["temp_c"][1]))
    d = news2_points(_clean("rr", tr["resprate"]), _clean("spo2", tr["o2sat"]), _clean("sbp", tr["sbp"]),
                     _clean("hr", tr["heartrate"]), t_c)
    d.index = tr.index
    return d


def _clean(name, s):
    lo, hi = PLAUSIBLE[name]
    s = pd.to_numeric(s, errors="coerce")
    return s.where((s >= lo) & (s <= hi))


# ----------------------------------------------------------------------------- SOFA
def sofa_components(pf=None, sf=None, vent=None, platelets=None, bilirubin=None, map_=None, norepi=None,
                    epi=None, dopamine=None, dobutamine=None, other_pressor=None, gcs=None, creatinine=None,
                    urine_24h=None, urine_hours=None):
    """All args array-like of the same length (NaN = missing). Returns DataFrame of 6 component scores (0-4)
    plus sofa_total and n_sofa_missing."""
    n = len(next(v for v in (pf, sf, platelets, bilirubin, map_, gcs, creatinine) if v is not None))
    f = lambda v: np.full(n, np.nan) if v is None else np.asarray(v, dtype=float)
    pf, sf, vent, plt_, bil, mp, ne, ep, dp, db, op, g, cr = map(f, (pf, sf, vent, platelets, bilirubin, map_, norepi, epi,
                                                                         dopamine, dobutamine, other_pressor, gcs, creatinine))
    vent1 = np.nan_to_num(vent) == 1
    # respiration: P/F preferred, else S/F surrogate
    resp_pf = np.select([np.isnan(pf), (pf < 100) & vent1, (pf < 200) & vent1, pf < 300, pf < 400],
                        [np.nan, 4, 3, 2, 1], default=0)
    s = f(sf)
    resp_sf = np.select([np.isnan(s), (s <= 148) & vent1, (s <= 235) & vent1, s <= 315, s <= 357],
                        [np.nan, 4, 3, 2, 1], default=0)
    resp = np.where(np.isnan(resp_pf), resp_sf, resp_pf)
    coag = np.select([np.isnan(plt_), plt_ < 20, plt_ < 50, plt_ < 100, plt_ < 150], [np.nan, 4, 3, 2, 1], default=0)
    liver = np.select([np.isnan(bil), bil >= 12, bil >= 6, bil >= 2, bil >= 1.2], [np.nan, 4, 3, 2, 1], default=0)
    z = lambda a: np.nan_to_num(a, nan=0.0)
    cv_known = ~np.isnan(mp) | ((z(ne) + z(ep) + z(dp) + z(db) + z(op)) > 0)
    cv = np.select(
        [z(dp) > 15, z(ep) > 0.1, z(ne) > 0.1,
         z(dp) > 5, (z(ep) > 0) , (z(ne) > 0), z(op) > 0,
         (z(dp) > 0) | (z(db) > 0), mp < 70],
        [4, 4, 4, 3, 3, 3, 3, 2, 1], default=0).astype(float)
    cv = np.where(cv_known, cv, np.nan)
    cns = np.select([np.isnan(g), g < 6, g < 10, g < 13, g < 15], [np.nan, 4, 3, 2, 1], default=0)
    renal_cr = np.select([np.isnan(cr), cr >= 5, cr >= 3.5, cr >= 2, cr >= 1.2], [np.nan, 4, 3, 2, 1], default=0)
    if urine_24h is not None:
        u = np.asarray(urine_24h, float); uh = np.asarray(urine_hours, float) if urine_hours is not None else np.full(n, 24.0)
        ok = (uh >= 20) & ~np.isnan(u)
        renal_uo = np.where(ok & (u < 200), 4, np.where(ok & (u < 500), 3, 0))
        renal = np.fmax(np.nan_to_num(renal_cr, nan=-1), renal_uo)
        renal = np.where((renal < 0) | (np.isnan(renal_cr) & ~ok), np.nan, renal)
    else:
        renal = renal_cr
    out = pd.DataFrame({"sofa_resp": resp, "sofa_coag": coag, "sofa_liver": liver, "sofa_cv": cv,
                        "sofa_cns": cns, "sofa_renal": renal})
    out["n_sofa_missing"] = out.isna().sum(axis=1)
    out["sofa"] = out[["sofa_resp", "sofa_coag", "sofa_liver", "sofa_cv", "sofa_cns", "sofa_renal"]].fillna(0).sum(axis=1)
    return out


def qsofa(rr, sbp, gcs):
    rr, sbp, g = (np.asarray(v, float) for v in (rr, sbp, gcs))
    comp = pd.DataFrame({"q_rr": np.where(np.isnan(rr), np.nan, rr >= 22), "q_sbp": np.where(np.isnan(sbp), np.nan, sbp <= 100),
                         "q_gcs": np.where(np.isnan(g), np.nan, g < 15)})
    comp["qsofa"] = comp.sum(axis=1, min_count=1)
    comp["qsofa_pos"] = comp.qsofa >= 2
    return comp


def sirs(temp_c, hr, rr, wbc, paco2=None):
    t, h, r, w = (np.asarray(v, float) for v in (temp_c, hr, rr, wbc))
    pc = np.full(len(t), np.nan) if paco2 is None else np.asarray(paco2, float)
    c = pd.DataFrame({"s_temp": np.where(np.isnan(t), np.nan, (t > 38) | (t < 36)),
                      "s_hr": np.where(np.isnan(h), np.nan, h > 90),
                      "s_rr": np.where(np.isnan(r) & np.isnan(pc), np.nan, (r > 20) | (pc < 32)),
                      "s_wbc": np.where(np.isnan(w), np.nan, (w > 12) | (w < 4))})
    c["sirs"] = c.sum(axis=1, min_count=1)
    c["sirs_pos"] = c.sirs >= 2
    return c


# ----------------------------------------------------------------------------- hourly panel
def _t(df, col):
    return pd.to_datetime(df[col], errors="coerce")


def _hourly_signal(ev, hours_idx_df):
    return ev


def build_hourly_panel(tables: dict, vital_ffill_h: int = 6, lab_ffill_h: int = 48):
    """tables: dict with icustays, chartevents, labevents, inputevents, outputevents (DataFrames, MIMIC-IV schema).
    Returns (panel, qc). panel: one row per (stay_id, hour) with raw signals, measured flags,
    hours-since-measured, NEWS2, SOFA, qSOFA, SIRS. qc: counts of values removed as implausible."""
    ic = tables["icustays"].copy()
    ic["intime"] = _t(ic, "intime"); ic["outtime"] = _t(ic, "outtime")
    ic = ic.dropna(subset=["intime", "outtime"])
    ic = ic[ic.outtime > ic.intime]
    ch = tables["chartevents"][["stay_id", "charttime", "itemid", "value", "valuenum"]].copy()
    ch["charttime"] = _t(ch, "charttime")
    ch = ch.dropna(subset=["charttime"])
    qc = {}

    def pick(ids, name, conv=None):
        d = ch[ch.itemid.isin(ids)][["stay_id", "charttime", "valuenum"]].copy()
        d["v"] = pd.to_numeric(d.valuenum, errors="coerce")
        if conv is not None:
            d["v"] = conv(d["v"])
        lo, hi = PLAUSIBLE[name.split("__")[0]]
        bad = d.v.notna() & ((d.v < lo) | (d.v > hi))
        qc[name] = int(bad.sum())
        d.loc[bad, "v"] = np.nan
        d["hour"] = d.charttime.dt.floor("h")
        return d.dropna(subset=["v"]).groupby(["stay_id", "hour"]).v.median().rename(name)

    sig = {
        "hr": pick(ITEM["hr"], "hr"), "rr": pick(ITEM["rr"], "rr"), "spo2": pick(ITEM["spo2"], "spo2"),
        "sbp_a": pick(ITEM["sbp_art"], "sbp").rename("sbp_a"), "sbp_n": pick(ITEM["sbp_nibp"], "sbp__n").rename("sbp_n"),
        "dbp_a": pick(ITEM["dbp_art"], "dbp").rename("dbp_a"), "dbp_n": pick(ITEM["dbp_nibp"], "dbp__n").rename("dbp_n"),
        "map_a": pick(ITEM["map_art"], "map").rename("map_a"), "map_n": pick(ITEM["map_nibp"], "map__n").rename("map_n"),
        "fio2": pick(ITEM["fio2"], "fio2"),
        "gcs_eye": pick(ITEM["gcs_eye"], "gcs_eye"), "gcs_verbal": pick(ITEM["gcs_verbal"], "gcs_verbal"),
        "gcs_motor": pick(ITEM["gcs_motor"], "gcs_motor"),
    }
    # temperature: F items converted, C items kept; implausible C (e.g. 99) removed AFTER conversion
    tf = pick(ITEM["temp_f"], "temp_c__f", conv=lambda s: (s - 32) * 5 / 9)
    tc = pick(ITEM["temp_c"], "temp_c__c")
    sig["temp_c"] = pd.concat([tf, tc]).groupby(level=[0, 1]).median().rename("temp_c")
    # fixed qc keys
    qc["temp_c"] = qc.pop("temp_c__f") + qc.pop("temp_c__c")
    # O2 device (string)
    od = ch[ch.itemid.isin(ITEM["o2_device"])][["stay_id", "charttime", "value"]].copy()
    od["hour"] = od.charttime.dt.floor("h")
    od["o2_on"] = (~od.value.fillna("").str.strip().str.match(NO_O2_RE)).astype(float)
    od["vent"] = od.value.fillna("").str.contains(INVASIVE_VENT_RE).astype(float)
    o2 = od.groupby(["stay_id", "hour"])[["o2_on", "vent"]].max()

    # hourly grid
    grids = []
    for sid, a, b in zip(ic.stay_id.values, ic.intime.dt.floor("h").values, ic.outtime.dt.ceil("h").values):
        h = pd.date_range(a, b, freq="h")
        grids.append(pd.DataFrame({"stay_id": sid, "hour": h}))
    P = pd.concat(grids, ignore_index=True).set_index(["stay_id", "hour"])
    for name, s in sig.items():
        P = P.join(s, how="left")
    P = P.join(o2, how="left")
    # combined signals: arterial preferred, else NIBP
    P["sbp"] = P["sbp_a"].fillna(P["sbp_n"])
    P["dbp"] = P["dbp_a"].fillna(P["dbp_n"])
    P["map"] = P["map_a"].fillna(P["map_n"])
    # derive MAP when missing and SBP/DBP exist
    P["map"] = P["map"].fillna((P["sbp"] + 2 * P["dbp"]) / 3)
    gcs_parts = P[["gcs_eye", "gcs_verbal", "gcs_motor"]]
    P["gcs"] = np.where(gcs_parts.notna().all(axis=1), gcs_parts.sum(axis=1), np.nan)

    # labs
    lb = tables["labevents"][["hadm_id", "charttime", "itemid", "valuenum"]].copy()
    lb["charttime"] = _t(lb, "charttime")
    lb = lb.dropna(subset=["charttime"])
    lbm = lb.merge(ic[["stay_id", "hadm_id", "intime", "outtime"]], on="hadm_id")
    lbm = lbm[(lbm.charttime >= lbm.intime - pd.Timedelta(hours=24)) & (lbm.charttime <= lbm.outtime)]
    labcols = []
    for name, ids in LAB.items():
        d = lbm[lbm.itemid.isin(ids)].copy()
        d["v"] = pd.to_numeric(d.valuenum, errors="coerce")
        lo, hi = PLAUSIBLE[name]
        bad = d.v.notna() & ((d.v < lo) | (d.v > hi)); qc[name] = int(bad.sum())
        d.loc[bad, "v"] = np.nan
        d["hour"] = d.charttime.dt.floor("h")
        s = d.dropna(subset=["v"]).groupby(["stay_id", "hour"]).v.last().rename(name)
        P = P.join(s, how="left"); labcols.append(name)
    # pre-ICU labs (up to 24h before intime) enter via first-hour carry: reindex-free approach -> add as hour of intime
    P = P.reset_index().sort_values(["stay_id", "hour"])
    P["hour_idx"] = P.groupby("stay_id").cumcount()
    # inject pre-ICU lab values at first hour
    pre = lbm[lbm.charttime < lbm.intime]
    for name, ids in LAB.items():
        d = pre[pre.itemid.isin(ids)].sort_values("charttime")
        d = d.assign(v=pd.to_numeric(d.valuenum, errors="coerce")).dropna(subset=["v"])
        lo, hi = PLAUSIBLE[name]; d = d[(d.v >= lo) & (d.v <= hi)]
        last = d.groupby("stay_id").v.last()
        m = (P.hour_idx == 0) & P[name].isna()
        P.loc[m, name] = P.loc[m, "stay_id"].map(last)

    # measured flags + hours since measured + forward fill with limits
    vit = ["hr", "rr", "spo2", "sbp", "dbp", "map", "temp_c", "gcs", "fio2"]
    P = P.set_index(["stay_id", "hour"])
    for c in vit + labcols:
        meas = P[c].notna()
        P[c + "_meas"] = meas.astype(int)
        lim = vital_ffill_h if c in vit else lab_ffill_h
        if c == "gcs":
            lim = 12
        P[c + "_raw"] = P[c]
        P[c] = P.groupby(level=0)[c].ffill(limit=lim)
        last_t = P.index.get_level_values(1).to_series(index=P.index).where(meas)
        last_t = last_t.groupby(level=0).ffill()
        P[c + "_age_h"] = (P.index.get_level_values(1).to_series(index=P.index) - last_t).dt.total_seconds() / 3600
    for c in ["o2_on", "vent"]:
        P[c] = P.groupby(level=0)[c].ffill(limit=12)

    # vasopressors from inputevents (rate in mcg/kg/min), weight-normalised if rateuom is mcg/min
    ie = tables["inputevents"]
    ie = ie[ie.itemid.isin(PRESSOR)].copy()
    ie["start"] = _t(ie, "starttime"); ie["end"] = _t(ie, "endtime")
    ie = ie.dropna(subset=["start", "end", "rate"])
    ie["rate"] = pd.to_numeric(ie.rate, errors="coerce")
    w = pd.to_numeric(ie.get("patientweight"), errors="coerce")
    ie["rate"] = np.where(ie.rateuom.fillna("").str.contains("kg"), ie.rate, ie.rate / w.fillna(80))
    for it, nm in PRESSOR.items():
        P["rx_" + nm] = 0.0
    if len(ie):
        rows = []
        for r in ie.itertuples():
            if pd.isna(r.start) or pd.isna(r.end) or r.end < r.start:
                continue
            for h in pd.date_range(r.start.floor("h"), r.end.floor("h"), freq="h"):
                rows.append((r.stay_id, h, PRESSOR[r.itemid], r.rate))
        if rows:
            pr = pd.DataFrame(rows, columns=["stay_id", "hour", "drug", "rate"])
            pv = pr.groupby(["stay_id", "hour", "drug"]).rate.max().unstack("drug")
            for nm in pv.columns:
                idx = P.index.intersection(pv.index)
                P.loc[idx, "rx_" + nm] = pv.loc[idx, nm].fillna(0).values

    # urine output (hourly sum) + rolling 24h
    oe = tables["outputevents"]
    oe = oe[oe.itemid.isin(URINE_ITEMS)].copy()
    oe["hour"] = _t(oe, "charttime").dt.floor("h")
    oe["v"] = pd.to_numeric(oe.value, errors="coerce").clip(lower=0)
    uo = oe.groupby(["stay_id", "hour"]).v.sum().rename("urine_ml")
    P = P.join(uo, how="left")
    P["uo_meas"] = P.urine_ml.notna().astype(int)
    P["urine_24h"] = P.groupby(level=0).urine_ml.transform(lambda s: s.fillna(0).rolling(24, min_periods=1).sum())
    P["urine_hours"] = P.groupby(level=0).uo_meas.transform(lambda s: s.rolling(24, min_periods=1).sum())

    P = P.reset_index().copy()
    P = _add_scores(P)
    return P, qc


def _worst24(P, col, how):
    g = P.groupby("stay_id")[col]
    return g.transform(lambda s: getattr(s.rolling(24, min_periods=1), how)())


def _add_scores(P: pd.DataFrame) -> pd.DataFrame:
    # NEWS2 at each hour from latest-known (ffilled) vitals
    o2 = np.where(P.o2_on.isna() & P.fio2.notna(), (P.fio2 > 21).astype(float), P.o2_on)
    n2 = news2_points(P.rr, P.spo2, P.sbp, P.hr, P.temp_c, gcs=P.gcs, o2_supp=o2)
    for c in n2.columns:
        P[c] = n2[c].values
    # SOFA (worst over trailing 24h)
    P["o2_supp"] = o2
    pf = P.pao2 / (P.fio2 / 100.0)
    pf = pf.where(P.pao2_age_h <= 24)
    sf = (P.spo2 / (P.fio2 / 100.0)).where((P.spo2 <= 97) & (P.fio2 >= 21))
    P["pf_ratio"] = pf; P["sf_ratio"] = sf
    w = pd.DataFrame({
        "pf": _worst24(P.assign(x=pf), "x", "min"), "sf": _worst24(P.assign(x=sf), "x", "min"),
        "vent": _worst24(P, "vent", "max"), "plt": _worst24(P, "platelets", "min"),
        "bil": _worst24(P, "bilirubin", "max"), "map": _worst24(P, "map", "min"),
        "gcs": _worst24(P, "gcs", "min"), "cr": _worst24(P, "creatinine", "max"),
        "ne": _worst24(P, "rx_norepi", "max"), "ep": _worst24(P, "rx_epi", "max"),
        "dp": _worst24(P, "rx_dopamine", "max"), "db": _worst24(P, "rx_dobutamine", "max"),
        "op": _worst24(P.assign(x=P[["rx_vasopressin", "rx_phenylephrine"]].max(axis=1)), "x", "max"),
    })
    # first hours: carry only what exists. pressor columns are 0 (not NaN) when none -> fine.
    s = sofa_components(pf=w["pf"], sf=w["sf"], vent=w["vent"], platelets=w["plt"], bilirubin=w["bil"], map_=w["map"],
                        norepi=w["ne"], epi=w["ep"], dopamine=w["dp"], dobutamine=w["db"], other_pressor=w["op"],
                        gcs=w["gcs"], creatinine=w["cr"], urine_24h=P["urine_24h"], urine_hours=P["urine_hours"])
    for c in s.columns:
        P[c] = s[c].values
    q = qsofa(P.rr, P.sbp, P.gcs)
    for c in q.columns:
        P[c] = q[c].values
    si = sirs(P.temp_c, P.hr, P.rr, P.wbc, P.paco2)
    for c in si.columns:
        P[c] = si[c].values
    P["shock_index"] = P.hr / P.sbp
    return P


# ----------------------------------------------------------------------------- Sepsis-3 proxy
def _attach_hadm(df: pd.DataFrame, adm):
    """Real MIMIC-IV microbiology rows often have hadm_id NaN (cultures drawn in the ED / outpatient).
    Fill hadm_id from the admission whose [admittime-24h, dischtime] window contains the row time."""
    df = df.copy()
    if adm is None or "subject_id" not in df:
        return df
    miss = df.hadm_id.isna()
    if not miss.any():
        return df
    a = adm[["subject_id", "hadm_id", "admittime", "dischtime"]].copy()
    a["admittime"] = _t(a, "admittime"); a["dischtime"] = _t(a, "dischtime")
    m = df[miss].drop(columns="hadm_id").reset_index().merge(a, on="subject_id", how="left")
    ok = (m.t >= m.admittime - pd.Timedelta(hours=24)) & (m.t <= m.dischtime)
    m = m[ok].sort_values("admittime").drop_duplicates("index").set_index("index")
    df.loc[m.index, "hadm_id"] = m.hadm_id
    return df


def suspected_infection(tables: dict) -> pd.DataFrame:
    """Per hadm: first suspected-infection time (Seymour 2016 rules) from antibiotics + cultures."""
    abx = []
    rx = tables.get("prescriptions")
    if rx is not None and len(rx):
        r = rx[rx.drug.fillna("").str.contains(ABX_RE)]
        abx.append(pd.DataFrame({"hadm_id": r.hadm_id, "t": _t(r, "starttime")}))
    ie = tables.get("inputevents")
    if ie is not None and len(ie) and "ordercategoryname" in ie:
        r = ie[ie.ordercategoryname.fillna("").str.contains("Antibiotics")]
        abx.append(pd.DataFrame({"hadm_id": r.hadm_id, "t": _t(r, "starttime")}))
    if not abx:
        return pd.DataFrame(columns=["hadm_id", "suspected_infection_time"])
    abx = pd.concat(abx).dropna()
    mb = tables.get("microbiologyevents")
    if mb is None or not len(mb):
        return pd.DataFrame(columns=["hadm_id", "suspected_infection_time"])
    cul = pd.DataFrame({"subject_id": mb.subject_id, "hadm_id": mb.hadm_id,
                        "t": pd.to_datetime(mb.charttime.fillna(mb.chartdate), errors="coerce")}).dropna(subset=["t"])
    cul = _attach_hadm(cul, tables.get("admissions"))[["hadm_id", "t"]].dropna()
    m = abx.rename(columns={"t": "abx"}).merge(cul.rename(columns={"t": "cul"}), on="hadm_id")
    d = (m.cul - m.abx).dt.total_seconds() / 3600
    ok = ((d >= 0) & (d <= 24)) | ((d < 0) & (d >= -72))  # abx first within 24h; culture first within 72h
    m = m[ok]
    m["onset"] = m[["abx", "cul"]].min(axis=1)
    return m.groupby("hadm_id").onset.min().rename("suspected_infection_time").reset_index()


def sepsis3_proxy(tables: dict, panel: pd.DataFrame) -> pd.DataFrame:
    ic = tables["icustays"][["stay_id", "hadm_id", "intime"]].copy(); ic["intime"] = _t(ic, "intime")
    inf = suspected_infection(tables)
    d = ic.merge(inf, on="hadm_id", how="left")
    rows = []
    P = panel.set_index(["stay_id", "hour"]).sort_index()
    for r in d.itertuples():
        onset = pd.NaT; base = np.nan
        if pd.notna(r.suspected_infection_time) and r.stay_id in P.index.get_level_values(0):
            s = P.loc[r.stay_id].sofa
            t0 = r.suspected_infection_time
            lo, hi = t0 - pd.Timedelta(hours=48), t0 + pd.Timedelta(hours=24)
            pre = s[(s.index < lo)]
            base = float(pre.iloc[-24:].min()) if len(pre) else 0.0
            if np.isnan(base):
                base = 0.0
            win = s[(s.index >= lo) & (s.index <= hi)]
            hit = win[win >= base + 2]
            if len(hit):
                onset = hit.index[0]
        rows.append((r.stay_id, r.hadm_id, r.suspected_infection_time, base, onset))
    out = pd.DataFrame(rows, columns=["stay_id", "hadm_id", "suspected_infection_time", "sofa_baseline", "sepsis3_onset"])
    out["sepsis3"] = out.sepsis3_onset.notna()
    return out


def add_sepsis_label(panel: pd.DataFrame, onset: pd.DataFrame, horizon: int = 6) -> pd.DataFrame:
    """label_Hh = 1 if sepsis onset lies in (t, t+H]; label_active = 1 if onset <= t.
    Rows at/after onset are flagged in 'post_onset' so training code can drop them."""
    P = panel.merge(onset[["stay_id", "sepsis3_onset"]], on="stay_id", how="left")
    dt = (P.sepsis3_onset - P.hour).dt.total_seconds() / 3600
    P[f"label_{horizon}h"] = ((dt > 0) & (dt <= horizon)).astype(int)
    P["post_onset"] = (dt <= 0).astype(int)
    return P.drop(columns=["sepsis3_onset"])
