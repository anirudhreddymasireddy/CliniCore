#!/usr/bin/env python3
"""MIMIC-IV-ED (real, credentialed PhysioNet access) -> the same Stage 1 frame as nhamcs.py, so Stage 1 trains on it unchanged.

  python mimic_ed.py --dir "C:\\data\\mimic-iv-ed" --hosp "C:\\data\\mimic-iv\\hosp" --cache mimic_cache
  python nhamcs_stage1.py --cache mimic_cache --split random            # trains + tests, writes the model the app uses

Needs ed/edstays, ed/triage, ed/diagnosis (and hosp/patients for age). Label = first ED diagnosis (seq_num 1), ICD-10 codes mapped to the
same 17 diagnosis groups as NHAMCS; ICD-9 visits are skipped. MIMIC data must never be committed to git or shared (PhysioNet DUA)."""
from __future__ import annotations
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nhamcs as NH
import scores as S
from load_tables import load_tables


def build(ed: dict, patients: pd.DataFrame | None) -> pd.DataFrame:
    es, tr, dx = ed["edstays"], ed["triage"], ed["diagnosis"]
    d = tr.merge(es[["stay_id", "subject_id", "gender", "arrival_transport", "intime"]], on=["stay_id", "subject_id"], how="inner")
    if patients is not None and len(patients):
        d = d.merge(patients[["subject_id", "anchor_age", "anchor_year"]], on="subject_id", how="left")
        d["anchor_age"] = d.anchor_age + (pd.to_datetime(d.intime, errors="coerce").dt.year - d.anchor_year).clip(lower=0)
    else:
        d["anchor_age"] = np.nan
    d = d[d.anchor_age.isna() | (d.anchor_age >= 18)].copy()
    t = pd.to_numeric(d.temperature, errors="coerce")
    d["temp_c"] = np.where((t >= 25) & (t <= 45), t, (t - 32) * 5 / 9)
    for c, nm in [("heartrate", "hr"), ("resprate", "rr"), ("o2sat", "spo2"), ("sbp", "sbp"), ("dbp", "dbp")]:
        d[nm] = S._clean(nm, pd.to_numeric(d[c], errors="coerce"))
    d["acuity"] = pd.to_numeric(d.acuity, errors="coerce").where(lambda s: s.between(1, 5))
    d["pain"] = pd.to_numeric(d.pain, errors="coerce").where(lambda s: s.between(0, 10))
    n2 = S.news2_from_triage(pd.DataFrame(dict(temperature=d.temp_c * 9 / 5 + 32, resprate=d.rr, o2sat=d.spo2, sbp=d.sbp, heartrate=d.hr), index=d.index))
    d["news2_triage"] = n2.news2.values
    d["male"] = (d.gender == "M").astype(float); d["ambulance"] = d.arrival_transport.fillna("").str.upper().eq("AMBULANCE").astype(float)
    d["n_hist_dx"] = np.nan                                                    # prior history needs hosp/diagnoses_icd; left missing here
    d["cc_text"] = d.chiefcomplaint.fillna("").astype(str)
    d["cc"] = NH.complaint_words(d.cc_text); d["rfv_main"] = np.nan
    first = dx[(dx.seq_num == 1) & (dx.icd_version == 10)][["stay_id", "icd_code"]]
    d = d.merge(first, on="stay_id", how="inner")
    d["dx_code"] = d.icd_code.astype(str); d["dx_group"] = d.dx_code.map(NH.dx_group)
    d = d[d.dx_group.notna()].copy()
    d["year"] = pd.to_datetime(d.intime, errors="coerce").dt.year                # shifted dates: use --split random, not by year
    keep = ["stay_id", "year", "anchor_age", "male", "ambulance", "temp_c", "hr", "rr", "spo2", "sbp", "dbp", "acuity", "news2_triage", "n_hist_dx", "pain",
            "cc", "cc_text", "rfv_main", "dx_code", "dx_group"]
    return d[keep].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True, help="MIMIC-IV-ED folder (with ed/ or the csv files)")
    ap.add_argument("--hosp", default=None, help="MIMIC-IV hosp folder (for patient age)"); ap.add_argument("--cache", default="mimic_cache"); a = ap.parse_args()
    ed = load_tables(a.dir, names=["edstays", "triage", "diagnosis"], optional=())
    pts = load_tables(a.hosp, names=["patients"], optional=())["patients"] if a.hosp else None
    d = build(ed, pts); os.makedirs(a.cache, exist_ok=True)
    d.to_csv(os.path.join(a.cache, "nhamcs_ed.csv.gz"), index=False)       # same file name, so nhamcs_stage1.py reads it unchanged
    print(f"{len(d)} adult ED visits with an ICD-10 first diagnosis -> {a.cache}/nhamcs_ed.csv.gz"); print(d.dx_group.value_counts().to_string())


if __name__ == "__main__":
    main()
