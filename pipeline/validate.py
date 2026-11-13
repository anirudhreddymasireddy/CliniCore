#!/usr/bin/env python3
"""
validate.py - one command that answers: "does the pipeline survive this dataset?"

  python validate.py --data /path/to/mimic_or_synth_dir [--truth /path/to/synth_truth]

Checks (all run on real MIMIC-IV too; the truth part is skipped when no synth_truth/ folder exists)
  1. schema    : required tables present, required columns present, extra columns allowed
  2. integrity : ICU stays link to admissions, labs/chart rows link to stays, duplicate keys, time order
  3. scoring   : NEWS2 / SOFA / qSOFA / SIRS / Sepsis-3 proxy run end to end without error, QC counts
  4. truth     : (synthetic only) sepsis detection vs planted truth, planted-artifact coverage
Exit code 0 = pipeline ran; findings are printed and returned as a dict (also written to --report json).
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scores as S                                   # noqa: E402
from load_tables import load_tables, find            # noqa: E402

REQUIRED_COLS = {
    "icustays": ["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
    "chartevents": ["stay_id", "charttime", "itemid", "value", "valuenum"],
    "labevents": ["hadm_id", "charttime", "itemid", "valuenum"],
    "inputevents": ["stay_id", "starttime", "endtime", "itemid", "rate", "rateuom"],
    "outputevents": ["stay_id", "charttime", "itemid", "value"],
    "admissions": ["subject_id", "hadm_id", "admittime", "dischtime"],
    "patients": ["subject_id", "gender", "anchor_age"],
    "edstays": ["subject_id", "stay_id", "intime", "outtime"],
    "triage": ["stay_id", "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "acuity", "chiefcomplaint"],
    "vitalsign": ["stay_id", "charttime", "heartrate", "resprate", "o2sat", "sbp", "dbp"],
}
OPTIONAL_TABLES = ["diagnosis", "medrecon", "pyxis", "prescriptions", "microbiologyevents", "transfers", "diagnoses_icd"]


def validate(data_dir, truth_dir=None, quiet=False):
    rep = {"data": data_dir, "schema": {}, "integrity": {}, "scoring": {}, "truth": {}}
    say = (lambda *a: None) if quiet else print
    # 1. schema
    missing_tables, missing_cols = [], {}
    for t, cols in REQUIRED_COLS.items():
        p = find(data_dir, t)
        if p is None:
            missing_tables.append(t); continue
        head = pd.read_csv(p, nrows=5)
        mc = [c for c in cols if c not in head.columns]
        if mc:
            missing_cols[t] = mc
    rep["schema"] = {"missing_tables": missing_tables, "missing_columns": missing_cols,
                     "optional_present": [t for t in OPTIONAL_TABLES if find(data_dir, t)]}
    say("schema:", rep["schema"])
    if missing_tables or missing_cols:
        rep["ok"] = False
        return rep
    T = load_tables(data_dir)
    ic, adm, ch = T["icustays"], T["admissions"], T["chartevents"]
    # 2. integrity
    it = {}
    it["icu_stays"] = int(len(ic)); it["admissions"] = int(len(adm)); it["ed_stays"] = int(len(T["edstays"]))
    it["icu_hadm_not_in_admissions"] = int((~ic.hadm_id.isin(adm.hadm_id)).sum())
    it["chart_stay_not_in_icustays"] = int((~ch.stay_id.isin(ic.stay_id)).sum())
    it["icu_stays_without_chartevents"] = int((~ic.stay_id.isin(ch.stay_id)).sum())
    it["duplicate_stay_id"] = int(ic.stay_id.duplicated().sum())
    ic_t = ic.assign(a=pd.to_datetime(ic.intime), b=pd.to_datetime(ic.outtime))
    it["icu_outtime_before_intime"] = int((ic_t.b <= ic_t.a).sum())
    c_t = pd.to_datetime(ch.charttime, errors="coerce")
    it["chartevents_bad_charttime"] = int(c_t.isna().sum())
    it["chartevents_exact_duplicates"] = int(ch.duplicated(["stay_id", "charttime", "itemid", "value"]).sum())
    if "storetime" in ch:
        it["chartevents_storetime_before_charttime"] = int((pd.to_datetime(ch.storetime, errors="coerce") < c_t - pd.Timedelta(minutes=1)).sum())
    mm = ch.merge(ic[["stay_id", "intime", "outtime"]], on="stay_id")
    tt = pd.to_datetime(mm.charttime, errors="coerce")
    it["chartevents_outside_icu_window"] = int(((tt < pd.to_datetime(mm.intime)) | (tt > pd.to_datetime(mm.outtime))).sum())
    rep["integrity"] = it
    say("integrity:", it)
    # 3. scoring
    P, qc = S.build_hourly_panel(T)
    onset = S.sepsis3_proxy(T, P)
    PL = S.add_sepsis_label(P, onset)
    sc = {"panel_rows": int(len(P)), "stays": int(P.stay_id.nunique()), "qc_values_set_to_nan": {k: v for k, v in qc.items() if v},
          "news2_band_share": P.news2_band.value_counts(normalize=True).round(3).to_dict(),
          "sofa_stay_max": P.groupby("stay_id").sofa.max().describe().round(2).to_dict(),
          "sofa_component_nan_share": P[["sofa_resp", "sofa_coag", "sofa_liver", "sofa_cv", "sofa_cns", "sofa_renal"]].isna().mean().round(3).to_dict(),
          "measured_share": {c: round(float(P[c + "_meas"].mean()), 3) for c in ["hr", "map", "gcs", "creatinine", "lactate"]},
          "qsofa_pos_share": round(float(P.qsofa_pos.mean()), 3), "sirs_pos_share": round(float(P.sirs_pos.mean()), 3),
          "suspected_infection_stays": int(onset.suspected_infection_time.notna().sum()), "sepsis3_stays": int(onset.sepsis3.sum()),
          "label_6h_positive_rate": round(float(PL.label_6h.mean()), 4)}
    rep["scoring"] = sc
    say("scoring:", json.dumps(sc, indent=1, default=str))
    # NEWS2 on triage (stage 1)
    n2 = S.news2_from_triage(T["triage"])
    rep["scoring"]["triage_news2_missing_share"] = round(float(n2.news2.isna().mean()), 3)
    # 4. truth (synthetic only)
    td = truth_dir or os.path.join(data_dir, "synth_truth")
    f = os.path.join(td, "truth_icu.csv.gz")
    if os.path.exists(f):
        tr = pd.read_csv(f)
        m = tr.merge(onset, on="stay_id")
        tp = int(((m.is_sepsis_truth == 1) & m.sepsis3).sum()); fn = int(((m.is_sepsis_truth == 1) & ~m.sepsis3).sum())
        fp = int(((m.is_sepsis_truth == 0) & m.sepsis3).sum()); tn = int(((m.is_sepsis_truth == 0) & ~m.sepsis3).sum())
        mt = m[(m.is_sepsis_truth == 1) & m.sepsis3]
        err = (pd.to_datetime(mt.sepsis3_onset) - pd.to_datetime(mt.sepsis_truth_onset)).dt.total_seconds() / 3600
        # planted artifacts vs what QC caught
        tags = tr.tags.fillna("").str.split("|").explode()
        planted = tags[tags != ""].value_counts().to_dict()
        rep["truth"] = {"sepsis_tp": tp, "sepsis_fn": fn, "sepsis_fp": fp, "sepsis_tn": tn,
                        "onset_error_h_median": round(float(err.median()), 2) if len(err) else None,
                        "onset_within_6h": round(float((err.abs() <= 6).mean()), 3) if len(err) else None,
                        "archetype_counts": tr.archetype.value_counts().to_dict(), "planted_edge_cases": planted,
                        "qc_caught": {k: v for k, v in qc.items() if v}}
        say("truth:", json.dumps(rep["truth"], indent=1, default=str))
    rep["ok"] = True
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--truth", default=None); ap.add_argument("--report", default=None)
    a = ap.parse_args()
    r = validate(a.data, a.truth)
    if a.report:
        json.dump(r, open(a.report, "w"), indent=1, default=str)
    sys.exit(0 if r.get("ok") else 1)
