#!/usr/bin/env python3
"""Load the PhysioNet/CinC 2019 Sepsis data (Kaggle copy or the original .psv folders) into one table and, optionally, into Exasol.

  python sepsis2019_load.py                      # downloads with kagglehub, caches sepsis2019_cache/sepsis2019.csv.gz, writes Exasol table
  python sepsis2019_load.py --path "C:\\path\\to\\dataset" --no-exasol
Table CAREGUARD_SEPSIS19.SEPSIS2019: patient_id, set_name (A or B = the two hospital systems), then the 40 original columns."""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
COLS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium", "Chloride",
        "Creatinine", "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen",
        "Platelets", "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS", "SepsisLabel"]


def set_of(path):
    p = path.lower().replace("\\", "/")
    return "A" if ("seta" in p or "set_a" in p or "set-a" in p) else ("B" if ("setb" in p or "set_b" in p or "set-b" in p) else "?")


def read_any(root):
    psv = sorted(glob.glob(os.path.join(root, "**", "*.psv"), recursive=True))
    if psv:
        print(f"found {len(psv)} .psv files")
        parts = []
        for i, f in enumerate(psv):
            d = pd.read_csv(f, sep="|"); d.insert(0, "patient_id", os.path.splitext(os.path.basename(f))[0]); d.insert(1, "set_name", set_of(f)); parts.append(d)
            if (i + 1) % 5000 == 0:
                print(f"  read {i+1}/{len(psv)}")
        return pd.concat(parts, ignore_index=True)
    csvs = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
    print("no .psv files; csv files found:", [os.path.relpath(c, root) for c in csvs][:10])
    if not csvs:
        sys.exit("No .psv or .csv files under " + root)
    d = pd.concat([pd.read_csv(c).assign(_src=c) for c in csvs], ignore_index=True)
    low = {c.lower(): c for c in d.columns}
    pid = next((low[k] for k in ("patient_id", "patient", "pid", "id") if k in low), None)
    if pid is None:                                                           # rebuild patient ids from ICULOS restarting at 1
        d["patient_id"] = (d["ICULOS"].values == 1).cumsum(); pid = "patient_id"
    d = d.rename(columns={pid: "patient_id"}); d["patient_id"] = d.patient_id.astype(str)
    d["set_name"] = d["_src"].map(set_of) if "_src" in d else "?"
    return d.drop(columns=[c for c in ("_src", "Unnamed: 0") if c in d.columns])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--path"); ap.add_argument("--cache", default="sepsis2019_cache")
    ap.add_argument("--schema", default="CAREGUARD_SEPSIS19"); ap.add_argument("--no-exasol", action="store_true"); a = ap.parse_args()
    path = a.path
    if not path:
        import kagglehub
        path = kagglehub.dataset_download("farjanayesmin/the-physionet-challenge-2019-dataset"); print("downloaded to", path)
    d = read_any(path)
    miss = [c for c in COLS if c not in d.columns]
    if miss:
        sys.exit(f"Columns missing in this copy of the data: {miss}. Columns found: {list(d.columns)}")
    d = d[["patient_id", "set_name"] + COLS]
    print(f"{d.patient_id.nunique()} patients, {len(d)} hourly rows, sepsis hours {int(d.SepsisLabel.sum())}; sets: {d.groupby('set_name').patient_id.nunique().to_dict()}")
    os.makedirs(a.cache, exist_ok=True); d.to_csv(os.path.join(a.cache, "sepsis2019.csv.gz"), index=False); print("cached ->", os.path.join(a.cache, "sepsis2019.csv.gz"))
    if not a.no_exasol:
        from stage1_pipeline import Source, write_exasol
        con = Source("exasol", a.schema).con
        print("writing to Exasol (this takes a few minutes) ...")
        write_exasol(con, a.schema, "SEPSIS2019", d, replace=True)
        print("rows in Exasol:", con.execute(f'SELECT COUNT(*) FROM "{a.schema}"."SEPSIS2019"').fetchone()[0])


if __name__ == "__main__":
    main()
