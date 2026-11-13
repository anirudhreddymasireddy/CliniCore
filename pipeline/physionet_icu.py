#!/usr/bin/env python3
"""Real ICU board from PhysioNet/CinC 2019 (40,336 real ICU patients from two hospital systems).

Trains the Stage 3 early-warning models on hospital system A and shows real hospital-B patients on the CliniCore ICU board.
Same code path as MIMIC-IV / synthetic data: the hourly data is mapped onto the Stage 3 panel, then the same rule tiers
(risks.py), features, models, ranking and board files are produced.

  python physionet_icu.py --source exasol                 # reads CAREGUARD_SEPSIS19.SEPSIS2019 (sepsis2019_load.py), writes CAREGUARD_RESULTS
  python physionet_icu.py --source csv                    # reads sepsis2019_cache/sepsis2019.csv.gz

What this data has and lacks (stated on the board and in the docs):
  has   heart rate, SpO2, temperature, blood pressure, resp rate, FiO2, blood gases, creatinine, lactate, WBC, platelets, bilirubin...
  lacks GCS / consciousness, vasopressors, urine output, ventilation flag, PaO2 (only SaO2).
  -> sepsis uses the real Sepsis-3 label of the challenge; kidney injury uses creatinine only; shock uses blood pressure,
     heart rate and lactate only (no drug data); respiratory failure uses SpO2/FiO2 and breathing rate.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import risks as R
import scores as S
import stage3_multirisk as M
import stage3_pipeline as P
from stage1_pipeline import Source, write_exasol

MAP = {"hr": "hr", "o2sat": "spo2", "temp": "temp_c", "sbp": "sbp", "dbp": "dbp", "map": "map", "resp": "rr", "fio2": "fio2",
       "creatinine": "creatinine", "bilirubin_total": "bilirubin", "platelets": "platelets", "wbc": "wbc", "lactate": "lactate",
       "paco2": "paco2", "ph": "ph", "hco3": "hco3", "hgb": "hgb", "glucose": "glucose", "potassium": "potassium", "bun": "bun", "ast": "ast"}
VITALS = ["hr", "rr", "spo2", "sbp", "dbp", "map", "temp_c", "gcs", "fio2"]
LABS = ["creatinine", "bilirubin", "platelets", "wbc", "lactate", "pao2", "paco2", "ph", "hco3", "hgb", "inr", "glucose", "sodium", "potassium", "bun", "alt", "ast"]
T0 = pd.Timestamp("2019-01-01")


def load_raw(a) -> pd.DataFrame:
    if a.source == "exasol":
        con = Source("exasol", a.schema).con
        d = con.export_to_pandas(f'SELECT * FROM "{a.schema}"."SEPSIS2019"')
    else:
        d = pd.read_csv(os.path.join(a.cache, "sepsis2019.csv.gz"))
    d.columns = [c.lower() for c in d.columns]
    return d


def to_panel(d: pd.DataFrame):
    """PhysioNet hourly rows -> Stage 3 hourly panel (same columns as scores.build_hourly_panel) + history table."""
    d = d.sort_values(["patient_id", "iculos"]).reset_index(drop=True)
    sid = d.patient_id.astype(str).str.extract(r"(\d+)")[0].astype(np.int64) + np.where(d.set_name.eq("B"), 500_000_000, 0)
    p = pd.DataFrame({"stay_id": sid.values, "hour": T0 + pd.to_timedelta(d.iculos.astype(float) - 1, unit="h"),
                      "hour_idx": (d.iculos.astype(int) - 1).values})
    for src, dst in MAP.items():
        p[dst] = pd.to_numeric(d[src], errors="coerce").values
    p["fio2"] = np.where(p.fio2 <= 1.0, p.fio2 * 100, p.fio2)                       # fraction -> %
    for c in p.columns:                                                            # same plausibility limits as the MIMIC loader
        if c in S.PLAUSIBLE:
            lo, hi = S.PLAUSIBLE[c]; p[c] = p[c].where((p[c] >= lo) & (p[c] <= hi))
    p["map"] = p["map"].fillna((p.sbp + 2 * p.dbp) / 3)
    for c in ["gcs", "pao2", "inr", "sodium", "alt"]:
        p[c] = np.nan                                                              # not in this dataset
    p["o2_on"], p["vent"] = np.nan, np.nan
    p = p.set_index(["stay_id", "hour"])
    hours = p.index.get_level_values(1).to_series(index=p.index)
    for c in VITALS + LABS:                                                        # same forward-fill rules as the MIMIC panel
        meas = p[c].notna()
        p[c + "_meas"] = meas.astype(int); p[c + "_raw"] = p[c]
        lim = 12 if c == "gcs" else (6 if c in VITALS else 48)
        p[c] = p.groupby(level=0)[c].ffill(limit=lim)
        last = hours.where(meas).groupby(level=0).ffill()
        p[c + "_age_h"] = (hours - last).dt.total_seconds() / 3600
    for nm in ["norepi", "epi", "dopamine", "dobutamine", "vasopressin", "phenylephrine"]:
        p["rx_" + nm] = 0.0                                                         # no drug data in PhysioNet 2019
    p["urine_ml"], p["uo_meas"], p["urine_24h"], p["urine_hours"] = np.nan, 0, 0.0, 0
    p = S._add_scores(p.reset_index())
    # Sepsis-3 label of the challenge: SepsisLabel = 1 from 6 h before onset -> onset = first labelled hour + 6
    lab = pd.Series(d.sepsislabel.fillna(0).values, index=p.index)
    first = p.hour.where(lab == 1).groupby(p.stay_id).transform("min")
    onset = first + pd.Timedelta(hours=6)
    dt = (onset - p.hour).dt.total_seconds() / 3600
    p["label_6h"] = ((dt > 0) & (dt <= 6)).astype(int)
    p["post_onset"] = (dt <= 0).astype(int)
    p = p[p.post_onset == 0]
    p = R.add_risk_states(p)
    st = d.groupby(sid.values).first()
    H = pd.DataFrame({"stay_id": st.index.values, "age": pd.to_numeric(st.age, errors="coerce").values, "male": (pd.to_numeric(st.gender, errors="coerce") == 1).astype(float).values,
                      "adm_emergency": np.nan, "h_admit_to_icu": -pd.to_numeric(st.hospadmtime, errors="coerce").values,
                      "icu_micu": pd.to_numeric(st.unit1, errors="coerce").values, "icu_cardiac": np.nan, "prior_admissions": np.nan})
    for k in M.CHRONIC:
        H[k] = np.nan                                                              # no diagnosis history in this dataset
    return p, H, dict(zip(sid.values, d.set_name.values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["exasol", "csv"], default="exasol"); ap.add_argument("--schema", default="CAREGUARD_SEPSIS19")
    ap.add_argument("--cache", default="sepsis2019_cache"); ap.add_argument("--out", default="results"); ap.add_argument("--out-schema", default="CAREGUARD_RESULTS")
    ap.add_argument("--train-patients", type=int, default=12000, help="hospital-A patients used for training (all = 20336)")
    ap.add_argument("--test-patients", type=int, default=6000, help="hospital-B patients used for the reported metrics")
    ap.add_argument("--board", type=int, default=200, help="hospital-B patients shown on the ICU board")
    ap.add_argument("--target-rate", type=float, default=0.15); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time(); rng = np.random.RandomState(a.seed)
    print("[1/6] reading PhysioNet 2019 ...", flush=True)
    d = load_raw(a)
    pid = d.drop_duplicates("patient_id")[["patient_id", "set_name"]]
    A = rng.permutation(pid[pid.set_name == "A"].patient_id.values)[: a.train_patients]
    B = rng.permutation(pid[pid.set_name == "B"].patient_id.values)[: max(a.test_patients, a.board)]
    d = d[d.patient_id.isin(set(A) | set(B))]
    print(f"[2/6] mapping {d.patient_id.nunique()} patients ({len(d)} hours) onto the Stage 3 panel ... ({time.time()-t0:.0f}s)", flush=True)
    p, H, hosp = to_panel(d)
    isB = p.stay_id.map(hosp).eq("B").values
    pa, pb = p[~isB].copy(), p[isB].copy()
    print(f"[3/6] features ... ({time.time()-t0:.0f}s)", flush=True)
    pa, Xa = M.build_X(pa, H); pb, Xb = M.build_X(pb, H); Xb = Xb.reindex(columns=Xa.columns)
    La, Ma = M.labels(pa); Lb, Mb = M.labels(pb)
    val_ids, tr_ids = P.stay_split(pa); tr, va = pa.stay_id.isin(tr_ids).values, pa.stay_id.isin(val_ids).values
    sets = {"A_validation": (pa, Xa, La, Ma, va), "B_shifted_hospital": (pb, Xb, Lb, Mb, np.ones(len(pb), bool))}
    print(f"[4/6] training the risk models on hospital A ... ({time.time()-t0:.0f}s)", flush=True)
    models, thr, met = M.train_heads(pa, Xa, La, Ma, tr, va, sets, a.target_rate)
    met["data"] = "PhysioNet 2019 (real)"
    M.save_heads(models, thr, Xa.columns, "PhysioNet 2019 hospital A (real)")
    met.to_csv(os.path.join(a.out, "stage3_multirisk_metrics.csv"), index=False)
    pd.set_option("display.width", 200); print(met[["set", "risk", "events", "auroc", "auprc", "auroc_current_tier", "auroc_qsofa", "auroc_news2"]].to_string(index=False))
    # board: random hospital-B patients; half of the beds go to patients who later developed sepsis so the demo shows warnings
    stays = pb.groupby("stay_id").label_6h.max()
    sep, non = stays[stays == 1].index.values, stays[stays == 0].index.values
    k = min(len(sep), a.board // 4)
    show = np.r_[rng.permutation(sep)[:k], rng.permutation(non)[: a.board - k]]
    print(f"[5/6] building the ICU board for {len(show)} hospital-B patients ({k} who later developed sepsis) ... ({time.time()-t0:.0f}s)", flush=True)
    _, snap, tl = M.board(pb[pb.stay_id.isin(show)], thr, a.out)
    json.dump(dict(train=[str(x) for x in A], test=[str(x) for x in B], board_stay_ids=[int(x) for x in show]),
              open(os.path.join(a.out, "stage3_physionet_ids.json"), "w"))                       # retrain.py --stage3-physionet uses the same patients
    src = dict(data="PhysioNet 2019 (real ICU patients)", label="Real ICU patients · PhysioNet 2019", hospital="B (never used for training)",
               patients_on_board=int(len(snap)), with_later_sepsis=int(k), trained_on=f"{len(A)} hospital-A patients",
               missing=["consciousness level", "vasopressors", "urine output", "ventilation flag", "PaO2"],
               note="Kidney injury from creatinine only; shock from blood pressure, heart rate and lactate only.")
    json.dump(src, open(os.path.join(a.out, "stage3_source.json"), "w"), indent=1)
    json.dump({"thresholds": thr, "models": sorted(models)}, open(os.path.join(a.out, "stage3_physionet_config.json"), "w"), indent=1)
    if a.source == "exasol":
        print("[6/6] writing to Exasol ...", flush=True)
        con = Source("exasol", a.out_schema).con
        write_exasol(con, a.out_schema, "ST3_RISK_METRICS", met, replace=True)
        write_exasol(con, a.out_schema, "ST3_RISKS", snap, replace=True)
        write_exasol(con, a.out_schema, "ST3_RISK_TIMELINE", tl, replace=True)
        write_exasol(con, a.out_schema, "ST3_SOURCE", pd.DataFrame([{k: (", ".join(v) if isinstance(v, list) else v) for k, v in src.items()}]), replace=True)
    print(f"done in {time.time()-t0:.0f}s: {len(snap)} real patients on the board; levels {snap.risk_level.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
