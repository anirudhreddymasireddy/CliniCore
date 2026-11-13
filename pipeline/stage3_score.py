#!/usr/bin/env python3
"""Score NEW ICU data with the saved Stage 3 models (no retraining) and refresh the ICU board. This is the live path.

Any source with MIMIC-IV-shaped tables works (icustays, chartevents, labevents, inputevents, outputevents, patients, admissions):
a MIMIC-IV extract, or a hospital feed written into an Exasol schema by an ETL job (HL7/FHIR -> those tables).

  python stage3_score.py --source exasol --schema ICU_LIVE                  # e.g. every 15 min from the Exasol scheduler / Task Scheduler
  python stage3_score.py --source csv --dir path\\to\\mimic-iv-folder --label "MIMIC-IV patients"
Models come from models/stage3_heads.joblib (written by stage3_multirisk.py or physionet_icu.py)."""
from __future__ import annotations
import argparse, json, os, sys, time
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage3_multirisk as M
from stage1_pipeline import Source, write_exasol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["exasol", "csv"], default="exasol"); ap.add_argument("--schema", default="ICU_LIVE"); ap.add_argument("--dir")
    ap.add_argument("--models", default="models/stage3_heads.joblib"); ap.add_argument("--out", default="results"); ap.add_argument("--out-schema", default="CAREGUARD_RESULTS")
    ap.add_argument("--label", default="Live ICU patients"); ap.add_argument("--last-hours", type=int, default=72, help="keep only the most recent hours per stay")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    H = joblib.load(a.models)
    src = Source(a.source, a.dir if a.source == "csv" else a.schema)
    p, T = M.load(src)                                                        # same hourly panel + rule tiers as training
    p, X = M.build_X(p, M.history(T, p.stay_id.unique()))
    X = X.reindex(columns=H["columns"])
    for d, m in H["models"].items():
        p[f"risk_{d}"] = m.predict_proba(X)[:, 1]
    p = p[p.groupby("stay_id").hour_idx.transform("max") - p.hour_idx < a.last_hours]
    _, snap, tl = M.board(p, H["thresholds"], a.out)
    stamp = time.strftime("%Y-%m-%d %H:%M")
    info = dict(data=a.label, label=f"{a.label} · scored {stamp}", patients_on_board=int(len(snap)), models_trained_on=H.get("trained_on", ""))
    json.dump(info, open(os.path.join(a.out, "stage3_source.json"), "w"), indent=1)
    if a.source == "exasol":
        write_exasol(src.con, a.out_schema, "ST3_RISKS", snap, replace=True)
        write_exasol(src.con, a.out_schema, "ST3_RISK_TIMELINE", tl, replace=True)
        write_exasol(src.con, a.out_schema, "ST3_SOURCE", pd.DataFrame([info]), replace=True)
    print(f"scored {len(snap)} ICU stays in {time.time()-t0:.0f}s; levels {snap.risk_level.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
