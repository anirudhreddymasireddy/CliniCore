#!/usr/bin/env python3
"""NHAMCS ED (2018-2022, Stata files) -> Stage 1 frame: adult visits with a usable first diagnosis, triage vitals, reason-for-visit codes, diagnosis GROUP.

  python nhamcs.py --dir "C:\\path\\with\\dta\\files"            # builds nhamcs_cache/nhamcs_ed.csv.gz (+ Exasol table with --exasol)
Diagnosis groups come from the first ICD-10-CM code (DIAG1). Administrative Z-codes, blank and unknown diagnoses are dropped (nothing to learn from)."""
from __future__ import annotations
import argparse, glob, os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scores as S

COLS = ["YEAR", "AGE", "SEX", "ARREMS", "TEMPF", "PULSE", "RESPR", "BPSYS", "BPDIAS", "POPCT", "PAINSCALE", "IMMEDR", "RFV1", "RFV2", "RFV3", "DIAG1",
        "CHF", "COPD", "CAD", "CKD", "CANCER", "CEBVD", "ESRD", "HTN", "DIABETES", "ASTHMA", "DEPRN", "OBESITY", "HYPLIPID", "PATWT"]
COMORB = ["CHF", "COPD", "CAD", "CKD", "CANCER", "CEBVD", "ESRD", "HTN", "DIABETES", "ASTHMA", "DEPRN", "OBESITY", "HYPLIPID"]


GROUP_LABELS = {
    "viral_covid_flu": "Viral infection (COVID-19 / flu)", "respiratory_infection": "Respiratory infection (incl. pneumonia)",
    "asthma_copd_other_respiratory": "Asthma / COPD / other breathing", "chest_pain_cardiac": "Chest pain / coronary",
    "heart_rhythm_failure_bp_vascular": "Heart rhythm, heart failure, blood pressure", "neuro_stroke_headache_syncope": "Neurological (stroke, headache, fainting)",
    "abdominal_gi": "Abdominal / digestive", "urinary_kidney": "Urinary / kidney", "gynecologic_obstetric": "Gynecologic / pregnancy",
    "skin_soft_tissue": "Skin / soft tissue", "sepsis_fever_other_infection": "Sepsis, fever or other infection", "injury_poisoning": "Injury / poisoning",
    "mental_substance": "Mental health / substance use", "metabolic_endocrine_blood": "Metabolic, diabetes, blood", "musculoskeletal_pain": "Muscle, joint, back pain",
    "eye_ear_cancer_other": "Eye, ear, cancer, other", "other_symptoms_signs": "Other symptoms"}


def dx_group(code: str):
    """ICD-10-CM first diagnosis -> coarse group (None = not usable)."""
    c = str(code).strip().upper().replace(".", "")
    if not c or c in ("-9", "-8", "NAN", "BLANK") or not c[0].isalpha() or c[0] == "Z":
        return None
    L, n = c[0], int(re.sub(r"\D", "", c[1:3]) or 0)
    if c.startswith(("U07", "B34", "B97", "B33")) or (L == "J" and 9 <= n <= 11):
        return "viral_covid_flu"
    if L == "J" and (n <= 6 or 12 <= n <= 22):
        return "respiratory_infection"
    if L == "J":
        return "asthma_copd_other_respiratory"
    if L == "R" and c[:3] == "R07" or (L == "I" and 20 <= n <= 25):
        return "chest_pain_cardiac"
    if L == "I" and not (60 <= n <= 69):
        return "heart_rhythm_failure_bp_vascular"
    if (L == "I" and 60 <= n <= 69) or L == "G" or c[:3] in ("R51", "R55", "R56", "R41", "R42", "R20", "R29", "R47", "R26"):
        return "neuro_stroke_headache_syncope"
    if L == "K" or c[:3] in ("R10", "R11", "R19", "R12", "R13", "R14", "R15", "R16", "R17", "R18"):
        return "abdominal_gi"
    if L == "N" and n < 70 or c[:3] in ("R30", "R31", "R32", "R33", "R34", "R35", "R36", "R39"):
        return "urinary_kidney"
    if L in ("O", "N"):
        return "gynecologic_obstetric"
    if L == "L" or c[:3] in ("A46", "B35", "B36", "B00", "B02", "B07"):
        return "skin_soft_tissue"
    if c[:3] in ("A40", "A41", "R65", "R57", "R50", "R68", "B99") or L in ("A", "B"):
        return "sepsis_fever_other_infection"
    if L in ("S", "T", "V", "W", "X", "Y"):
        return "injury_poisoning"
    if L == "F":
        return "mental_substance"
    if L in ("E", "D"):
        return "metabolic_endocrine_blood"
    if L == "M":
        return "musculoskeletal_pain"
    if L in ("H", "C"):
        return "eye_ear_cancer_other"
    if L == "R":
        return "other_symptoms_signs"
    return None


RFV_LABELS: dict = {}


def complaint_words(text: pd.Series) -> pd.Series:
    """Lower-case words of a complaint text ('Chest pain; Shortness of breath' -> 'chest pain shortness of breath')."""
    return text.fillna("").astype(str).str.lower().str.replace(r"[^a-z ]+", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()


def read_years(folder: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(folder, "**", "*.dta"), recursive=True))
    if not files:
        sys.exit(f"No .dta files found under {folder}")
    parts = []
    for f in files:
        r = pd.io.stata.StataReader(f); have = set(r.variable_labels())
        if not RFV_LABELS:
            try:
                r.read(nrows=1); RFV_LABELS.update({int(k): str(v) for k, v in r.value_labels().get("RFVF", {}).items()})   # reason-for-visit code -> readable text
            except Exception:
                pass
        d = pd.read_stata(f, columns=[c for c in COLS if c in have], convert_categoricals=False)
        for c in COLS:
            if c not in d.columns:
                d[c] = np.nan
        print(f"  {os.path.basename(f)}: {len(d)} visits"); parts.append(d)
    return pd.concat(parts, ignore_index=True)


def build_frame(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.copy(); n0 = len(d)
    num = lambda c: pd.to_numeric(d[c], errors="coerce")
    d["anchor_age"] = num("AGE"); d = d[d.anchor_age >= 18].copy(); n_adult = len(d)
    miss = lambda s, lo, hi: s.where((s >= lo) & (s <= hi))                                  # negative codes (-9 blank, -8 unknown) and impossible values -> NaN
    d["temp_f"] = miss(num("TEMPF") / 10.0, 85, 110); d["temp_c"] = (d.temp_f - 32) * 5 / 9
    d["hr"] = miss(num("PULSE"), 20, 250); d["rr"] = miss(num("RESPR"), 4, 80); d["spo2"] = miss(num("POPCT"), 50, 100)
    d["sbp"] = miss(num("BPSYS"), 50, 300); d["dbp"] = miss(num("BPDIAS"), 20, 200)
    d["acuity"] = num("IMMEDR").where(num("IMMEDR").between(1, 5))                           # 1 = immediate ... 5 = non-urgent; 0/7/blank = no triage
    d["male"] = (num("SEX") == 2).astype(float); d["ambulance"] = num("ARREMS").map({1: 1.0, 2: 0.0})
    d["pain"] = miss(num("PAINSCALE"), 0, 10)
    n2 = S.news2_from_triage(pd.DataFrame(dict(temperature=d.temp_f.values, resprate=d.rr.values, o2sat=d.spo2.values, sbp=d.sbp.values, heartrate=d.hr.values), index=d.index))
    d["news2_triage"] = n2.news2.values
    d["n_hist_dx"] = sum((num(c) == 1).astype(int) for c in COMORB)                          # known chronic conditions recorded at the visit
    def toks(row):
        out = []
        for k, p in (("RFV1", "r"), ("RFV2", "s"), ("RFV3", "s")):
            v = row[k]
            if pd.notna(v) and v > 0:
                out.append(f"{p}{int(v)}"); out.append(f"g{int(v) // 10}")          # detailed code + broader 4-digit group
        return " ".join(out)
    d["cc"] = d[["RFV1", "RFV2", "RFV3"]].apply(toks, axis=1)
    def label(row):
        out = [RFV_LABELS.get(int(row[k]), f"code {int(row[k])}") for k in ("RFV1", "RFV2", "RFV3") if pd.notna(row[k]) and row[k] > 0]
        return "; ".join(out)
    d["cc_text"] = d[["RFV1", "RFV2", "RFV3"]].apply(label, axis=1)
    d["cc"] = d.cc + " " + complaint_words(d.cc_text)                       # codes + plain words: lets free-text complaints (MIMIC-IV-ED, live) use the same model
    d["rfv_main"] = num("RFV1").where(num("RFV1") > 0)
    d["dx_group"] = d.DIAG1.map(dx_group); d["dx_code"] = d.DIAG1.astype(str)
    d = d[d.dx_group.notna()].copy()
    d["stay_id"] = (d.YEAR.astype(int) * 1_000_000 + np.arange(len(d))).astype(np.int64); d["year"] = d.YEAR.astype(int)
    keep = ["stay_id", "year", "anchor_age", "male", "ambulance", "temp_c", "hr", "rr", "spo2", "sbp", "dbp", "acuity", "news2_triage", "n_hist_dx", "pain", "cc", "cc_text", "rfv_main", "dx_code", "dx_group"]
    print(f"visits {n0} -> adults {n_adult} -> with a usable diagnosis {len(d)}")
    return d[keep].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--cache", default="nhamcs_cache")
    ap.add_argument("--exasol", action="store_true"); ap.add_argument("--schema", default="CAREGUARD_NHAMCS"); a = ap.parse_args()
    d = build_frame(read_years(a.dir)); os.makedirs(a.cache, exist_ok=True)
    d.to_csv(os.path.join(a.cache, "nhamcs_ed.csv.gz"), index=False); print("cached ->", os.path.join(a.cache, "nhamcs_ed.csv.gz"))
    used = pd.to_numeric(d.rfv_main, errors="coerce").dropna().astype(int).value_counts()
    import json
    json.dump({str(k): RFV_LABELS.get(int(k), f"code {k}") for k in used.index}, open(os.path.join(a.cache, "rfv_labels.json"), "w"), indent=0)   # reasons seen as main complaint, most common first
    print(d.groupby("year").size().to_dict()); print(d.dx_group.value_counts().to_string())
    if a.exasol:
        from stage1_pipeline import Source, write_exasol
        con = Source("exasol", a.schema).con; write_exasol(con, a.schema, "NHAMCS_ED", d, replace=True); print("written to Exasol", a.schema + ".NHAMCS_ED")


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------ one live intake (same features as training)
RFV_TEXT: dict = {}            # reason code -> label text, set by the app from rfv_labels.json
INTAKE_NAMES = {"age": "Age", "temp_c": "Temperature", "hr": "Heart rate", "rr": "Breathing rate", "spo2": "Oxygen saturation", "sbp": "Systolic blood pressure",
                "dbp": "Diastolic blood pressure", "pain": "Pain", "acuity": "Triage level"}
INTAKE_LIMITS = {"age": (18, 110), "temp_c": (30, 43), "hr": (20, 250), "rr": (4, 80), "spo2": (50, 100), "sbp": (50, 300), "dbp": (20, 200), "pain": (0, 10), "acuity": (1, 5)}
CHRONIC = {"CHF": "Heart failure", "COPD": "COPD", "CAD": "Coronary artery disease", "CKD": "Chronic kidney disease", "CANCER": "Cancer",
           "CEBVD": "Stroke / cerebrovascular disease", "ESRD": "End-stage renal disease", "HTN": "High blood pressure", "DIABETES": "Diabetes",
           "ASTHMA": "Asthma", "DEPRN": "Depression", "OBESITY": "Obesity", "HYPLIPID": "High cholesterol"}


def validate_intake(x: dict):
    """Returns (clean dict, list of problems). Impossible values are errors; missing vitals are allowed (the model handles missing)."""
    errs, out = [], {}
    if x.get("age") in (None, ""):
        errs.append("age is required (adults 18+ only)")
    for k, (lo, hi) in INTAKE_LIMITS.items():
        v = x.get(k)
        if v in (None, ""):
            out[k] = np.nan; continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            errs.append(f"{INTAKE_NAMES.get(k, k)}: not a number"); continue
        if not lo <= v <= hi:
            errs.append(f"{INTAKE_NAMES.get(k, k)} {v:g} is outside the possible range {lo:g}-{hi:g}" + (" (CliniCore ED intake is for adults)" if k == "age" else "")); continue
        out[k] = v
    if not np.isnan(out.get("sbp", np.nan)) and not np.isnan(out.get("dbp", np.nan)) and out["dbp"] >= out["sbp"]:
        errs.append("diastolic BP must be lower than systolic BP")
    reasons = [int(r) for r in (x.get("reasons") or []) if str(r).strip().lstrip("-").isdigit() and int(r) > 0][:3]
    out["complaint_text"] = str(x.get("complaint_text") or "").strip()[:300]
    if not reasons and not out["complaint_text"]:
        errs.append("at least one reason for visit is required")
    out["reasons"] = reasons; out["male"] = 1.0 if str(x.get("sex", "")).lower() in ("m", "male", "2") else 0.0
    out["ambulance"] = 1.0 if x.get("ambulance") else 0.0; out["chronic"] = [c for c in (x.get("chronic") or []) if c in CHRONIC]
    return out, errs


def frame_from_intake(c: dict, visit_id: int) -> pd.DataFrame:
    t_f = c["temp_c"] * 9 / 5 + 32 if not np.isnan(c["temp_c"]) else np.nan
    n2 = S.news2_from_triage(pd.DataFrame(dict(temperature=[t_f], resprate=[c["rr"]], o2sat=[c["spo2"]], sbp=[c["sbp"]], heartrate=[c["hr"]])))
    toks = []
    for i, r in enumerate(c["reasons"]):
        toks += [f"{'r' if i == 0 else 's'}{r}", f"g{r // 10}"]
    words = complaint_words(pd.Series([" ".join(RFV_TEXT.get(r, "") for r in c["reasons"]) + " " + c.get("complaint_text", "")])).iloc[0]
    return pd.DataFrame([dict(stay_id=visit_id, anchor_age=c["age"], male=c["male"], ambulance=c["ambulance"], temp_c=c["temp_c"], hr=c["hr"], rr=c["rr"], spo2=c["spo2"],
                              sbp=c["sbp"], dbp=c["dbp"], acuity=c["acuity"], news2_triage=float(n2.news2.iloc[0]) if pd.notna(n2.news2.iloc[0]) else np.nan,
                              n_hist_dx=float(len(c["chronic"])), pain=c["pain"], cc=(" ".join(toks) + " " + words).strip(), rfv_main=c["reasons"][0] if c["reasons"] else -1)])
