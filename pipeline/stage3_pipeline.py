#!/usr/bin/env python3
"""
CareGuard Stage 3 - ICU deterioration / sepsis early warning.

Hourly panel (scores.build_hourly_panel) -> features -> LightGBM risk -> alerts with reasons.
Trains on hospital A, tests on hospital B (never tuned on B). Compares against NEWS2 / qSOFA / SOFA rules.

  python stage3_pipeline.py --source csv --dir data/samples/synth_default --shift-dir data/samples/synth_hospital_c
  python stage3_pipeline.py --source exasol --schema CAREGUARD_A --shift-schema CAREGUARD_C

Label = Sepsis-3 proxy, onset within next 6 h (label_6h). Rows after onset are dropped (post_onset).
NOTE: on synthetic data the label is built from SOFA, so SOFA-derived features partly "see" the label.
Real-data numbers (PhysioNet Sepsis 2019, set A -> set B) are the only ones to quote.
"""
from __future__ import annotations
import argparse, json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scores as S
from stage1_pipeline import Source, write_exasol
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb

TABLES = ["icustays", "chartevents", "labevents", "inputevents", "outputevents", "prescriptions",
          "microbiologyevents", "patients", "admissions"]
VITALS = ["hr", "rr", "spo2", "sbp", "map", "temp_c", "gcs"]
LABS = ["lactate", "creatinine", "platelets", "wbc", "bilirubin", "ph", "bun", "glucose"]
SCORES = ["news2", "sofa", "qsofa", "sirs", "shock_index", "pf_ratio", "sf_ratio"]
PRESS = ["rx_norepi", "rx_epi", "rx_dopamine", "rx_dobutamine", "rx_vasopressin", "rx_phenylephrine"]
NICE = {"hr": "heart rate", "rr": "resp rate", "spo2": "SpO2", "sbp": "systolic BP", "map": "mean BP", "temp_c": "temperature",
        "gcs": "GCS", "lactate": "lactate", "creatinine": "creatinine", "platelets": "platelets", "wbc": "WBC",
        "bilirubin": "bilirubin", "ph": "pH", "bun": "BUN", "glucose": "glucose", "news2": "NEWS2", "sofa": "SOFA",
        "qsofa": "qSOFA", "sirs": "SIRS", "shock_index": "shock index", "pf_ratio": "P/F ratio", "sf_ratio": "S/F ratio",
        "pressor_any": "vasopressor", "urine_ml": "urine output", "hour_idx": "time in ICU"}


# ------------------------------------------------------------------ data
def load_panel(src: Source) -> pd.DataFrame:
    T = {}
    for t in TABLES:
        try:
            T[t] = src.read(t)
        except Exception:
            T[t] = pd.DataFrame()
    panel, _ = S.build_hourly_panel(T)
    onset = S.sepsis3_proxy(T, panel)
    p = S.add_sepsis_label(panel, onset)
    return p.merge(onset[["stay_id", "sepsis3_onset"]], on="stay_id", how="left")


def features(p: pd.DataFrame, drop_sofa=False) -> pd.DataFrame:
    p = p.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    g = p.groupby("stay_id", sort=False)
    X = pd.DataFrame(index=p.index)
    for c in VITALS + LABS + SCORES + ["urine_ml"]:
        if c in p:
            X[c] = p[c]
    for c in VITALS + LABS:
        if c + "_age_h" in p:
            X[c + "_age_h"] = p[c + "_age_h"]
    X["pressor_any"] = p[[c for c in PRESS if c in p]].fillna(0).astype(float).max(axis=1)
    X["hour_idx"] = p["hour_idx"]
    for c in ["hr", "rr", "map", "spo2", "temp_c", "lactate", "sofa", "news2"]:
        if c in p:
            X[c + "_d6"] = p[c] - g[c].shift(6)                       # change over 6 h
            X[c + "_max6"] = g[c].transform(lambda s: s.rolling(6, min_periods=1).max())
            X[c + "_min6"] = g[c].transform(lambda s: s.rolling(6, min_periods=1).min())
    if drop_sofa:
        X = X[[c for c in X.columns if not c.startswith(("sofa", "pf_ratio", "sf_ratio"))]]
    return X


# ------------------------------------------------------------------ baselines and metrics
BASELINES = {"NEWS2>=5": ("news2", 5), "NEWS2>=7": ("news2", 7), "qSOFA>=2": ("qsofa", 2), "SOFA>=2": ("sofa", 2)}


def alert_times(df, score, thr, refractory=12):
    """First-alert style episodes: alert when score>=thr and no alert in the previous `refractory` hours."""
    out = []
    for sid, d in df.groupby("stay_id", sort=False):
        last = -999
        for h, s in zip(d.hour_idx.values, d[score].values):
            if s >= thr and h - last >= refractory:
                out.append((sid, h, s)); last = h
    return pd.DataFrame(out, columns=["stay_id", "hour_idx", "score"])


def evaluate_alerts(df, alerts, onset_idx):
    """alerts per patient-day, share of septic stays alerted >=0h before onset, median lead (h)."""
    days = len(df) / 24.0
    first = alerts.groupby("stay_id").hour_idx.min()
    septic = onset_idx.dropna()
    leads = []
    for sid, o in septic.items():
        a = alerts[(alerts.stay_id == sid) & (alerts.hour_idx <= o)]
        if len(a):
            leads.append(o - a.hour_idx.min())
    n_false = len(alerts) - sum(len(alerts[(alerts.stay_id == s) & (alerts.hour_idx <= o + 0)]) > 0 for s, o in septic.items())
    return {"alerts_per_patient_day": round(len(alerts) / days, 3),
            "septic_stays": int(len(septic)), "septic_caught_before_onset": int(len(leads)),
            "recall_stays": round(len(leads) / max(len(septic), 1), 3),
            "median_lead_h": float(np.median(leads)) if leads else None, "alerts_total": int(len(alerts))}


def utility(df, score, thr, onset_idx):
    """Approximate PhysioNet/CinC-2019 utility, normalised (0 = never alert, 1 = ideal). Use the official script on real data."""
    dt_e, dt_o, dt_l, fp, fn_min = -12, -6, 3, -0.05, -2.0
    def u_tp(t):  return np.where(t < dt_e, 0, np.where(t <= dt_o, (t - dt_e) / (dt_o - dt_e), np.where(t <= dt_l, 1 - (t - dt_o) / (dt_l - dt_o), 0)))
    def u_fn(t):  return np.where(t <= dt_o, 0, np.where(t <= dt_l, fn_min * (t - dt_o) / (dt_l - dt_o), fn_min))
    obs = inact = best = 0.0
    for sid, d in df.groupby("stay_id", sort=False):
        pred = (d[score].values >= thr)
        o = onset_idx.get(sid, np.nan)
        if np.isnan(o):
            obs += fp * pred.sum()                              # best = inaction = 0 for non-septic
            continue
        t = d.hour_idx.values - o
        tp, fn = u_tp(t), u_fn(t)
        obs += np.where(pred, np.where(t >= dt_e, tp, fp), np.where(t >= dt_o, fn, 0)).sum()
        inact += np.where(t >= dt_o, fn, 0).sum()
        best += np.where(t >= dt_e, tp, 0).sum()
    return round((obs - inact) / (best - inact), 3) if best != inact else None


def threshold_for_rate(df, score, target):
    """Smallest threshold whose episode alert rate <= target alerts per patient-day."""
    cand = np.unique(np.quantile(df[score].dropna(), np.linspace(0.5, 0.9995, 60)))
    for thr in cand:
        if len(alert_times(df, score, thr)) / (len(df) / 24.0) <= target:
            return float(thr)
    return float(cand[-1])


def report(df, score, thr, onset_idx):
    m = df.dropna(subset=[score])
    r = {"auroc": round(roc_auc_score(m.label_6h, m[score]), 3) if m.label_6h.nunique() > 1 else None,
         "auprc": round(average_precision_score(m.label_6h, m[score]), 3) if m.label_6h.nunique() > 1 else None}
    al = alert_times(df, score, thr)
    r.update(evaluate_alerts(df, al, onset_idx)); r["utility"] = utility(df, score, thr, onset_idx); r["threshold"] = round(float(thr), 3)
    return r


# ------------------------------------------------------------------ model + reasons
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=2.0, verbose=-1, n_estimators=300, random_state=0)


def fit(Xtr, ytr, groups=None):
    pos = max(ytr.sum(), 1)
    m = lgb.LGBMClassifier(**PARAMS, scale_pos_weight=min((len(ytr) - pos) / pos, 20))
    m.fit(Xtr, ytr)
    return m


def reasons(model, X, rows, p, k=3):
    """Top-k drivers per row from LightGBM SHAP contributions; adds a stale-data note when a driver is old."""
    contrib = model.booster_.predict(X.loc[rows], pred_contrib=True)[:, :-1]
    names = list(X.columns); out = []; skip = {"hour_idx"}
    for i, r in enumerate(rows):
        order = np.argsort(-contrib[i])
        txt = []; seen = set()
        for j in order:
            if contrib[i, j] <= 0 or len(txt) >= k:
                break
            n = names[j]
            if n in skip:
                continue
            if n.endswith("_age_h"):
                base = n[:-6]; v = X.at[r, n]
                if not (v >= 12):
                    continue                                  # only flag genuinely stale data
                txt.append(f"{NICE.get(base, base)} not measured for {v:.0f} h"); continue
            base = n.split("_d6")[0].split("_max6")[0].split("_min6")[0]
            val = X.at[r, n]
            kind = "rising" if n.endswith("_d6") and val > 0 else "falling" if n.endswith("_d6") else ""
            if base in seen:
                continue
            seen.add(base)
            label = NICE.get(base, base)
            txt.append(f"{label} {kind} ({val:+.1f} over 6 h)" if n.endswith("_d6") else f"{label} {val:.1f}")
        out.append("; ".join(txt))
    return out


# ------------------------------------------------------------------ main
def stay_split(p, frac=0.25, seed=0):
    ids = p.stay_id.drop_duplicates().sample(frac=1, random_state=seed).values
    n = int(len(ids) * frac)
    return set(ids[:n]), set(ids[n:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["exasol", "csv"], default="csv")
    ap.add_argument("--schema", default="CAREGUARD_A"); ap.add_argument("--shift-schema", default=None)
    ap.add_argument("--dir"); ap.add_argument("--shift-dir"); ap.add_argument("--out-schema", default="CAREGUARD_RESULTS")
    ap.add_argument("--target-rate", type=float, default=0.30, help="alerts per patient-day allowed (sets the threshold on validation stays)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-sofa-features", action="store_true", help="ablation: drop SOFA/PF/SF inputs (label on synthetic data is built from SOFA)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    A = Source(a.source, a.dir if a.source == "csv" else a.schema)
    B = Source(a.source, (a.shift_dir if a.source == "csv" else a.shift_schema)) if (a.shift_dir or a.shift_schema) else None

    pa = load_panel(A); pa = pa[~pa.post_onset.fillna(False).astype(bool)].reset_index(drop=True)
    Xa = features(pa, a.no_sofa_features); pa = pa.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    pa["hour_idx"] = pa["hour_idx"].astype(float)
    val_ids, tr_ids = stay_split(pa)
    tr, va = pa.stay_id.isin(tr_ids).values, pa.stay_id.isin(val_ids).values
    model = fit(Xa[tr], pa.label_6h[tr].astype(int))
    pa["risk"] = model.predict_proba(Xa)[:, 1]
    thr = threshold_for_rate(pa[va], "risk", a.target_rate)                 # threshold from validation stays of hospital A only

    def onset_of(df):
        o = df.groupby("stay_id").apply(lambda d: d.hour_idx[d.label_6h == 1].max() + 1 if (d.label_6h == 1).any() else np.nan)
        return o

    res = {"threshold_rule": f"risk >= {thr:.3f} (set on hospital-A validation stays for <= {a.target_rate} alerts per patient-day)", "sets": {}}
    sets = {"A_validation": pa[va]}
    alerts_out = []
    if B is not None:
        pb = load_panel(B); pb = pb[~pb.post_onset.fillna(False).astype(bool)].sort_values(["stay_id", "hour"]).reset_index(drop=True)
        pb["hour_idx"] = pb["hour_idx"].astype(float)
        Xb = features(pb, a.no_sofa_features)[Xa.columns]
        pb["risk"] = model.predict_proba(Xb)[:, 1]
        sets["B_shifted_hospital"] = pb
    for name, df in sets.items():
        oi = onset_of(df)
        rows = {"LightGBM": report(df, "risk", thr, oi)}
        for bn, (col, t) in BASELINES.items():
            if col in df:
                rows[bn] = report(df, col, t, oi)
                # same-alert-budget comparison: tune the rule's own threshold to LightGBM's alert rate on this set
                tm = threshold_for_rate(df, col, rows["LightGBM"]["alerts_per_patient_day"])
                rows[bn + " (matched alert rate)"] = report(df, col, tm, oi)
        res["sets"][name] = rows
    # alerts with reasons on the target set (or validation set)
    tgt_name = "B_shifted_hospital" if B is not None else "A_validation"
    tdf = sets[tgt_name]; Xt = (Xb if B is not None else Xa[va]).copy()
    al = alert_times(tdf, "risk", thr)
    key = tdf.reset_index().set_index(["stay_id", "hour_idx"])["index"]
    idx = [key[(s, h)] for s, h in zip(al.stay_id, al.hour_idx)]
    Xt.index = tdf.index if B is not None else tdf.index
    al["reason"] = reasons(model, features(pb, a.no_sofa_features)[Xa.columns] if B is not None else Xa, idx, None) if len(al) else []
    al["hour"] = [tdf.at[i, "hour"] for i in idx]
    al["septic_later"] = al.stay_id.map(lambda s: bool((tdf[tdf.stay_id == s].label_6h == 1).any()))
    al.to_csv(os.path.join(a.out, "stage3_alerts.csv"), index=False)
    json.dump(res, open(os.path.join(a.out, "stage3_results.json"), "w"), indent=2)

    flat = [dict(set=s, model=m, **v) for s, d in res["sets"].items() for m, v in d.items()]
    tab = pd.DataFrame(flat); tab.to_csv(os.path.join(a.out, "stage3_metrics.csv"), index=False)
    print(res["threshold_rule"])
    print(tab[["set", "model", "auroc", "auprc", "alerts_per_patient_day", "recall_stays", "median_lead_h", "utility"]].to_string(index=False))
    if a.source == "exasol" and A.con is not None:
        write_exasol(A.con, a.out_schema, "ST3_METRICS", tab, replace=True)
        al["hour"] = al["hour"].astype(str)
        write_exasol(A.con, a.out_schema, "ST3_ALERTS", al[["stay_id", "hour", "hour_idx", "score", "reason", "septic_later"]], replace=True)
        print(f"written to Exasol schema {a.out_schema}: ST3_METRICS, ST3_ALERTS")


if __name__ == "__main__":
    main()
