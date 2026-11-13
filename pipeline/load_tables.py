"""Load MIMIC-IV-schema tables from a directory (real MIMIC-IV demo/full, or synthetic output).
Accepts layouts  <dir>/{hosp,icu,ed}/*.csv[.gz]   or flat <dir>/*.csv[.gz]."""
from __future__ import annotations
import os, glob
import pandas as pd

NEEDED = ["icustays", "chartevents", "labevents", "inputevents", "outputevents", "prescriptions",
          "microbiologyevents", "patients", "admissions", "transfers", "diagnoses_icd",
          "edstays", "triage", "vitalsign", "diagnosis", "medrecon", "pyxis"]


def find(root, name):
    for pat in (f"{root}/*/{name}.csv.gz", f"{root}/*/{name}.csv", f"{root}/{name}.csv.gz", f"{root}/{name}.csv"):
        g = glob.glob(pat)
        if g:
            return g[0]
    return None


def load_tables(root, names=NEEDED, optional=("medrecon", "pyxis", "transfers", "diagnoses_icd", "diagnosis", "microbiologyevents", "prescriptions")):
    out = {}
    for n in names:
        p = find(root, n)
        if p is None:
            if n in optional:
                continue
            raise FileNotFoundError(f"{n} not found under {root}")
        out[n] = pd.read_csv(p, low_memory=False)
    return out
