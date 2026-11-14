#!/usr/bin/env python3
"""
CareGuard Stage 3 (multi-risk) - ranked ICU risks per patient with standard scores, history and a model for each risk.

For every ICU hour and each of four risks (sepsis, acute respiratory failure, AKI, circulatory shock):
  1. CURRENT STATE from the published standard (risks.py): tier Low / Watch / High / Critical + the evidence behind it.
  2. EARLY WARNING from a LightGBM head: "will this risk reach HIGH within the next H hours?" (only for patients not already High).
Risks are ranked by current tier first, then by the model flag. The top 3 are shown with their standard and evidence.

  python stage3_multirisk.py --source csv --dir data/samples/synth_default --shift-dir data/samples/synth_hospital_c
  python stage3_multirisk.py --source exasol --schema CAREGUARD_A --shift-schema CAREGUARD_C

Train on hospital A, thresholds from A validation stays, test on hospital B. Synthetic numbers only prove the code works.
"""
from __future__ import annotations
import argparse, json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scores as S, risks as R
import stage3_pipeline as P
from stage1_pipeline import Source, write_exasol
from sklearn.metrics import roc_auc_score, average_precision_score

TABLES = P.TABLES + ["diagnoses_icd"]
HORIZON = {"sepsis": 6, "arf": 6, "aki": 12, "shock": 6}          # hours ahead each head looks
CHRONIC = {                                                           # past-history flags: ICD-10 prefixes, ICD-9 prefixes
    "hx_htn": (("I10", "I11", "I12", "I13"), ("401", "402", "403")),
    "hx_dm": (("E10", "E11", "E13"), ("250",)),
    "hx_hf": (("I50",), ("428",)),
    "hx_cad": (("I25",), ("414",)),
    "hx_af": (("I48",), ("42731",)),
    "hx_copd": (("J44", "J43"), ("496", "491", "492")),
    "hx_ckd": (("N18",), ("585",)),
    "hx_liver": (("K70", "K74", "K76"), ("571",)),
    "hx_cancer": (tuple(f"C{i:02d}" for i in range(0, 97)), tuple(str(i) for i in range(140, 209))),
}


def load(src: Source):
    T = {}
    for t in TABLES:
        try:
            T[t] = src.read(t)
        except Exception as e:
            if t in ("icustays", "chartevents"):
                raise RuntimeError(f"Cannot read table {t} from '{src.where}'. Is that schema/folder loaded? "
                                   f"Load it first: python load_to_exasol.py <folder> <SCHEMA>.  ({str(e).strip()[:120]})")
            T[t] = pd.DataFrame()
    panel, _ = S.build_hourly_panel(T)
    onset = S.sepsis3_proxy(T, panel)
    p = S.add_sepsis_label(panel, onset)
    p = p[~p.post_onset.fillna(False).astype(bool)]
    p = R.add_risk_states(p)
    return p, T


def history(T, stays, include_current_chronic=True):
    """Static per-stay history: age, sex, admission type, prior admissions, chronic conditions known before ICU.
    Prior-admission diagnoses are always safe. Current-admission codes are limited to CHRONIC conditions (a proxy for the
    past-medical-history list recorded at admission); acute diagnoses such as sepsis or MI are never used."""
    ic, pt, ad = T["icustays"], T["patients"], T["admissions"]
    d = ic[["stay_id", "subject_id", "hadm_id", "intime", "first_careunit"]].copy()
    d = d.merge(pt[["subject_id", "gender", "anchor_age"]], on="subject_id", how="left")
    d = d.merge(ad[["hadm_id", "admittime", "admission_type"]], on="hadm_id", how="left")
    d["intime"] = pd.to_datetime(d.intime, errors="coerce"); d["admittime"] = pd.to_datetime(d.admittime, errors="coerce")
    H = pd.DataFrame({"stay_id": d.stay_id, "age": d.anchor_age, "male": (d.gender == "M").astype(float),
                      "adm_emergency": d.admission_type.fillna("").str.contains("EMER|URGENT", case=False).astype(float),
                      "h_admit_to_icu": (d.intime - d.admittime).dt.total_seconds() / 3600.0,
                      "icu_micu": d.first_careunit.fillna("").str.contains("MICU|Medical", case=False).astype(float),
                      "icu_cardiac": d.first_careunit.fillna("").str.contains("CCU|CVICU|Cardiac", case=False).astype(float)})
    a2 = ad.assign(admittime=pd.to_datetime(ad.admittime, errors="coerce"))[["subject_id", "hadm_id", "admittime"]]
    cur = d[["stay_id", "subject_id", "hadm_id", "admittime"]].merge(a2, on="subject_id", suffixes=("", "_o"))
    prior = cur[(cur.admittime_o < cur.admittime)]
    H["prior_admissions"] = H.stay_id.map(prior.groupby("stay_id").hadm_id_o.nunique()).fillna(0)
    dx = T.get("diagnoses_icd", pd.DataFrame())
    for k in CHRONIC:
        H[k] = 0.0
    if len(dx):
        dx = dx.assign(icd_code=dx.icd_code.astype(str).str.replace(".", "", regex=False), icd_version=dx.icd_version)
        def flag(df, key_col):
            out = {}
            for k, (v10, v9) in CHRONIC.items():
                m = ((df.icd_version == 10) & df.icd_code.str.startswith(v10)) | ((df.icd_version == 9) & df.icd_code.str.startswith(v9))
                out[k] = df[m].groupby(key_col).size()
            return out
        pf = flag(dx.merge(prior[["stay_id", "hadm_id_o"]], left_on="hadm_id", right_on="hadm_id_o"), "stay_id")
        for k, s in pf.items():
            H[k] = np.maximum(H[k], H.stay_id.map(s).notna().astype(float))
        if include_current_chronic:
            cf = flag(dx.merge(d[["stay_id", "hadm_id"]], on="hadm_id"), "stay_id")
            for k, s in cf.items():
                H[k] = np.maximum(H[k], H.stay_id.map(s).notna().astype(float))
    return H


def build_X(p, H, drop_sofa=False):
    p = p.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    X = P.features(p, drop_sofa)
    for c in ["rox", "kdigo", "kdigo_cr", "kdigo_uo", "cr_ratio", "scai", "n_pressors", "t_sepsis", "t_arf", "t_aki", "t_shock"]:
        X[c] = p[c]
    Hs = H.set_index("stay_id")
    for c in Hs.columns:
        X[c] = p.stay_id.map(Hs[c]).values
    return p, X


def future_event(p, tier_col, h):
    g = p.groupby("stay_id", sort=False)[tier_col]
    fut = g.transform(lambda s: s.shift(-1)[::-1].rolling(h, min_periods=1).max()[::-1])
    return ((fut >= 2) & (p[tier_col] < 2)).astype(int)


def labels(p):
    L, M = {}, {}
    for d, h in HORIZON.items():
        if d == "sepsis":
            L[d] = p.label_6h.fillna(0).astype(int)                      # Sepsis-3 proxy onset (label_6h)
            M[d] = p.t_sepsis < 3
        else:
            L[d] = future_event(p, f"t_{d}", h)
            M[d] = p[f"t_{d}"] < 2
    return L, M


SEV = np.array([0.0, 0.15, 0.45, 0.80])      # severity weight of a risk at tier Low / Watch / High / Critical
FLAG_BUMP = 0.10                              # extra weight when the model expects a Low/Watch risk to reach High
LEVEL_CUTS = (15, 40, 70)                     # overall score -> Watch / High / Critical


def criticality(df, flags):
    """Overall criticality 0-100 = 1 - prod(1 - severity_d): several concurrent risks compound, one Critical risk alone is >= 80.
    CareGuard's own composite of the standard-score tiers; it is NOT a validated clinical score (see criticality check)."""
    surv = np.ones(len(df))
    for d in R.DOMAINS:
        t = df[f"t_{d}"].values.astype(int)
        s = SEV[t] + np.where(flags[d].values & (t < 2), FLAG_BUMP, 0.0)
        surv *= (1 - s)
    score = 100 * (1 - surv)
    level = np.select([score >= LEVEL_CUTS[2], score >= LEVEL_CUTS[1], score >= LEVEL_CUTS[0]], [3, 2, 1], 0)
    return np.round(score, 1), level


CLASSIC = {"sepsis": "qsofa", "arf": "news2", "aki": "cr_ratio", "shock": "shock_index"}


def head_metrics(df, y, m, score, domain):
    d = df[m.values]; yy = y[m.values]
    if yy.nunique() < 2:
        return {"rows": int(m.sum()), "events": int(yy.sum())}
    out = {"rows": int(m.sum()), "events": int(yy.sum()), "prevalence": round(float(yy.mean()), 4),
           "auroc": round(roc_auc_score(yy, d[score]), 3), "auprc": round(average_precision_score(yy, d[score]), 4),
           "auroc_current_tier": round(roc_auc_score(yy, d[f"t_{domain}"]), 3)}
    cs = d[CLASSIC[domain]]
    ok = cs.notna()
    if ok.sum() > 50 and yy[ok].nunique() > 1:
        out[f"auroc_{CLASSIC[domain]}"] = round(roc_auc_score(yy[ok], cs[ok]), 3)
    return out


def ranked(row, flags):
    """List of (domain, tier, flag) most critical first."""
    items = []
    for d in R.DOMAINS:
        t = int(row[f"t_{d}"]); f = bool(flags[d])
        if t >= 1 or f:
            items.append((d, t, f))
    return sorted(items, key=lambda x: (-x[1], -int(x[2])))[:3]


def describe(row, items, backend=False):
    """Patient/doctor text uses only vitals and labs. backend=True gives the index/score text for audit columns."""
    out = []
    for d, t, f in items:
        tag = R.TIER_NAMES[t].upper()
        if backend:
            txt = f"{R.DOMAINS[d]} {tag} [{R.STANDARD[d]}: {R.evidence_backend(row, d)}]"
        else:
            txt = f"{R.DOMAINS[d]} - {tag}: {R.reason_text(row, d)}"
            if f and t < 2:
                txt += f" (model expects this to worsen within {HORIZON[d]} h)"
        out.append(txt)
    return out


def train_heads(pa, Xa, La, Ma, tr, va, sets, target_rate):
    """One LightGBM early-warning model per risk, trained on the training stays of hospital A; alert threshold set on A's
    validation stays at `target_rate` alerts per patient-day; metrics for every evaluation set."""
    models, thr, rows = {}, {}, []
    for d in R.DOMAINS:
        m_tr = (tr & Ma[d].values)
        if La[d][m_tr].nunique() < 2:                       # no events for this risk in this data (e.g. no urine/pressor data)
            print(f"   {d}: no events in training data -> rule tiers only, no early-warning model")
            continue
        models[d] = P.fit(Xa[m_tr], La[d][m_tr])
        pa[f"risk_{d}"] = models[d].predict_proba(Xa)[:, 1]
        vdf = pa[va & Ma[d].values]
        thr[d] = P.threshold_for_rate(vdf, f"risk_{d}", target_rate)
        for name, (df, X, L, M, mask) in sets.items():
            if name != "A_validation":
                df[f"risk_{d}"] = models[d].predict_proba(X)[:, 1]
            sub = df[mask]; r = head_metrics(sub, L[d][mask], M[d][mask], f"risk_{d}", d)
            r.update(dict(set=name, risk=d, standard=R.STANDARD[d], horizon_h=HORIZON[d], threshold=round(thr[d], 3)))
            rows.append(r)
    cols = ["set", "risk", "standard", "horizon_h", "rows", "events", "prevalence", "auroc", "auprc",
            "auroc_current_tier", "auroc_qsofa", "auroc_news2", "auroc_cr_ratio", "auroc_shock_index"]
    met = pd.DataFrame(rows).reindex(columns=cols)
    return models, thr, met


def save_heads(models, thr, cols, trained_on, path="models/stage3_heads.joblib"):
    """Everything stage3_score.py needs to score new (live) data without retraining."""
    import joblib
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(dict(models=models, thresholds=thr, columns=list(cols), trained_on=trained_on, horizon_h=HORIZON), path)


def board(df, thr, out):
    """ICU board outputs for the stays in df (one row per stay-hour, with t_* tiers and risk_* model scores):
    stage3_risks_snapshot.csv (latest hour per stay, ranked risks with plain-language reasons) and stage3_risk_timeline.csv."""
    df = df.reset_index(drop=True)
    for d in R.DOMAINS:
        if f"risk_{d}" not in df:
            df[f"risk_{d}"] = 0.0
    flags = pd.DataFrame({d: (df[f"risk_{d}"] >= thr.get(d, 2.0)) & (df[f"t_{d}"] < 2) for d in R.DOMAINS})
    top = [ranked(r, flags.loc[i]) for i, r in df.iterrows()]
    df["top_domain"] = [t[0][0] if t else "" for t in top]
    df["top_tier"] = [t[0][1] if t else 0 for t in top]
    df["criticality"], df["crit_level"] = criticality(df, flags)
    prev = df.groupby("stay_id").criticality.shift(6)
    df["crit_change_6h"] = (df.criticality - prev).round(1)
    df["crit_trend"] = np.select([prev.isna(), df.crit_change_6h >= 10, df.crit_change_6h <= -10], ["new", "rising", "falling"], "steady")
    last = df.groupby("stay_id").tail(1).index
    snap = []
    for i in last:
        items = top[i]; txt = describe(df.loc[i], items) + ["", "", ""]; bk = describe(df.loc[i], items, True) + ["", "", ""]
        snap.append(dict(stay_id=df.at[i, "stay_id"], hour=str(df.at[i, "hour"]), hour_idx=df.at[i, "hour_idx"],
                         criticality=df.at[i, "criticality"], risk_level=R.TIER_NAMES[int(df.at[i, "crit_level"])],
                         trend_6h=df.at[i, "crit_trend"], change_6h=df.at[i, "crit_change_6h"], n_risks=len(items),
                         risk_1=txt[0], risk_2=txt[1], risk_3=txt[2], backend_1=bk[0], backend_2=bk[1], backend_3=bk[2],
                         domain_1=(items[0][0] if len(items) > 0 else ""), domain_2=(items[1][0] if len(items) > 1 else ""), domain_3=(items[2][0] if len(items) > 2 else ""),
                         tier_1=(items[0][1] if len(items) > 0 else 0), tier_2=(items[1][1] if len(items) > 1 else 0), tier_3=(items[2][1] if len(items) > 2 else 0),
                         **{f"v_{c}": (round(float(df.at[i, c]), 2) if c in df and pd.notna(df.at[i, c]) else None) for c in ["hr", "spo2", "sbp", "dbp", "rr", "temp_c"]}))
    snap = pd.DataFrame(snap).sort_values("criticality", ascending=False).reset_index(drop=True)   # ICU board order: sickest first
    snap.to_csv(os.path.join(out, "stage3_risks_snapshot.csv"), index=False)
    # timeline: only hours where the top risk or its tier changes (keeps the dashboard table small)
    chg = (df.groupby("stay_id").top_domain.shift() != df.top_domain) | (df.groupby("stay_id").top_tier.shift() != df.top_tier) | (df.groupby("stay_id").crit_level.shift() != df.crit_level)
    tl = df[chg & (df.top_tier >= 1)][["stay_id", "hour", "hour_idx", "top_domain", "top_tier", "criticality", "crit_level"]].copy()
    tl["hour"] = tl.hour.astype(str); tl["top_tier_name"] = tl.top_tier.map(dict(enumerate(R.TIER_NAMES)))
    tl.to_csv(os.path.join(out, "stage3_risk_timeline.csv"), index=False)
    vitals_table(df).to_csv(os.path.join(out, "stage3_vitals.csv"), index=False)
    return df, snap, tl


VITAL_COLS = ["hr", "spo2", "sbp", "dbp", "map", "rr", "temp_c", "fio2", "lactate"]


def vitals_table(df, hours=48):
    """Last `hours` hourly values per stay (forward-filled as the panel shows them) for the live vital-sign view."""
    keep = df[df.groupby("stay_id").hour_idx.transform("max") - df.hour_idx < hours]
    cols = [c for c in VITAL_COLS if c in keep]
    return keep[["stay_id", "hour_idx"] + cols].round(2)


def write_board_exasol(con, schema, snap, tl, out):
    """Board tables in Exasol (same files as the CSV board)."""
    write_exasol(con, schema, "ST3_RISKS", snap, replace=True)
    write_exasol(con, schema, "ST3_RISK_TIMELINE", tl, replace=True)
    f = os.path.join(out, "stage3_vitals.csv")
    if os.path.exists(f):
        write_exasol(con, schema, "ST3_VITALS", pd.read_csv(f), replace=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["exasol", "csv"], default="csv")
    ap.add_argument("--schema", default="CAREGUARD_A"); ap.add_argument("--shift-schema", default=None)
    ap.add_argument("--dir"); ap.add_argument("--shift-dir"); ap.add_argument("--out-schema", default="CAREGUARD_RESULTS")
    ap.add_argument("--target-rate", type=float, default=0.15, help="alerts per patient-day allowed per risk head")
    ap.add_argument("--no-history", action="store_true", help="ablation: drop history features")
    ap.add_argument("--prior-only", action="store_true", help="history from earlier admissions only (no current-admission chronic codes)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--data-label", default="Demo patients (synthetic)", help='shown on the ICU board, e.g. "MIMIC-IV patients"')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    A = Source(a.source, a.dir if a.source == "csv" else a.schema)
    B = Source(a.source, a.shift_dir if a.source == "csv" else a.shift_schema) if (a.shift_dir or a.shift_schema) else None
    print("[1/5] loading hospital A and building hourly features (the slow part, 1-2 min)...", flush=True)
    pa, Ta = load(A)
    Ha = history(Ta, pa.stay_id.unique(), not a.prior_only)
    pa, Xa = build_X(pa, Ha)
    if a.no_history:
        Xa = Xa[[c for c in Xa.columns if c not in Ha.columns]]
    cols = list(Xa.columns)
    La, Ma = labels(pa)
    val_ids, tr_ids = P.stay_split(pa)
    tr, va = pa.stay_id.isin(tr_ids).values, pa.stay_id.isin(val_ids).values

    sets = {"A_validation": (pa, Xa, La, Ma, va)}; Tmap = {"A_validation": Ta}
    if B is not None:
        print("[2/5] loading the shifted hospital...", flush=True)
        pb, Tb = load(B)
        pb, Xb = build_X(pb, history(Tb, pb.stay_id.unique(), not a.prior_only))
        Xb = Xb.reindex(columns=cols)
        Lb, Mb = labels(pb)
        sets["B_shifted_hospital"] = (pb, Xb, Lb, Mb, np.ones(len(pb), bool)); Tmap["B_shifted_hospital"] = Tb

    print("[3/5] training the four risk models...", flush=True)
    models, thr, met = train_heads(pa, Xa, La, Ma, tr, va, sets, a.target_rate)
    met.to_csv(os.path.join(a.out, "stage3_multirisk_metrics.csv"), index=False)

    print("[4/5] ranking risks and criticality for each patient...", flush=True)
    tname = "B_shifted_hospital" if B is not None else "A_validation"
    df, X, L, M, mask = sets[tname]; df = df[mask].reset_index(drop=True)
    df, snap, tl = board(df, thr, a.out)
    save_heads(models, thr, cols, a.data_label)
    src = dict(data=a.data_label, label=a.data_label, patients_on_board=int(len(snap)))
    json.dump(src, open(os.path.join(a.out, "stage3_source.json"), "w"), indent=1)
    # criticality check: does the highest score in a stay track in-hospital death better than the single standard scores?
    try:
        from sklearn.metrics import roc_auc_score as _auc
        T = Tmap[tname]; ic, ad = T["icustays"], T["admissions"]
        died = ic.merge(ad[["hadm_id", "hospital_expire_flag"]], on="hadm_id").set_index("stay_id").hospital_expire_flag
        g = df.groupby("stay_id")
        chk = pd.DataFrame({"max_criticality": g.criticality.max(), "max_news2": g.news2.max(), "max_sofa": g.sofa.max()})
        chk["died"] = died.reindex(chk.index)
        chk = chk.dropna()
        ck = {k: round(_auc(chk.died, chk[k]), 3) for k in ["max_criticality", "max_news2", "max_sofa"]} if chk.died.nunique() > 1 else {}
        ck.update(stays=int(len(chk)), deaths=int(chk.died.sum()))
        pd.DataFrame([ck]).to_csv(os.path.join(a.out, "stage3_criticality_check.csv"), index=False)
        print("criticality check (AUROC for in-hospital death, stay level):", ck)
    except Exception as e:
        print("criticality check skipped:", e)
    json.dump({"thresholds": thr, "criticality": {"severity_by_tier": SEV.tolist(), "model_flag_bump": FLAG_BUMP, "level_cuts": LEVEL_CUTS}, "horizon_h": HORIZON, "history_features": list(Ha.columns[1:]), "weight_assumed_kg": R.WEIGHT_KG},
              open(os.path.join(a.out, "stage3_multirisk_config.json"), "w"), indent=2)

    pd.set_option("display.width", 200)
    print(met.drop(columns=["rows", "standard"]).to_string(index=False))
    imp = pd.Series(models["shock"].feature_importances_, cols).sort_values(ascending=False) if "shock" in models else pd.Series(dtype=float)
    print("history features used by the shock head:", {k: int(v) for k, v in imp[[c for c in Ha.columns[1:] if c in imp.index]].items() if v > 0})
    if a.source == "exasol" and A.con is not None:
        print("[5/5] writing results to Exasol...", flush=True)
        write_exasol(A.con, a.out_schema, "ST3_RISK_METRICS", met, replace=True)
        write_board_exasol(A.con, a.out_schema, snap, tl, a.out)
        write_exasol(A.con, a.out_schema, "ST3_SOURCE", pd.DataFrame([src]), replace=True)
        print(f"written to Exasol schema {a.out_schema}: ST3_RISK_METRICS, ST3_RISKS, ST3_RISK_TIMELINE")


if __name__ == "__main__":
    main()
