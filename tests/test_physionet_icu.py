"""PhysioNet 2019 adapter: hourly .psv-style rows -> Stage 3 panel -> tiers, features, models, ICU board (fake patients, no download)."""
import os, sys, json, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
import physionet_icu as PN
import stage3_multirisk as M
from sepsis2019_load import COLS


def fake(n_per_set=60, seed=0):
    rng, rows = np.random.RandomState(seed), []
    for s in ("A", "B"):
        for p in range(n_per_set):
            T = rng.randint(20, 60); sep = rng.rand() < 0.3; on = rng.randint(12, T) if sep else None
            d = pd.DataFrame(np.nan, index=range(T), columns=COLS)
            drift = np.clip(np.arange(T) - (on - 10), 0, None) if sep else np.zeros(T)
            d["HR"] = 80 + 2 * drift + rng.randn(T) * 3; d["Resp"] = 16 + drift + rng.randn(T); d["O2Sat"] = 97 - 0.5 * drift
            d["SBP"] = 120 - 2 * drift; d["MAP"] = 85 - drift; d["DBP"] = 70 - drift; d["Temp"] = 37 + 0.1 * drift
            d.loc[::8, "Creatinine"] = 1.0 + 0.05 * drift[::8]; d.loc[::8, "Lactate"] = 1.2 + 0.2 * drift[::8]; d.loc[::12, "WBC"] = 9 + drift[::12]
            d["Age"], d["Gender"], d["Unit1"], d["Unit2"], d["HospAdmTime"] = 65, 1, 1, 0, -5
            d["ICULOS"] = np.arange(1, T + 1); d["SepsisLabel"] = 0
            if sep:
                d.loc[d.ICULOS >= on - 6, "SepsisLabel"] = 1
            d.insert(0, "patient_id", f"p{s}{p:05d}"); d.insert(1, "set_name", s); rows.append(d)
    d = pd.concat(rows, ignore_index=True); d.columns = [c.lower() for c in d.columns]
    return d


def test_panel_has_stage3_columns_and_no_post_onset_rows():
    p, H, hosp = PN.to_panel(fake())
    for c in ["t_sepsis", "t_arf", "t_aki", "t_shock", "news2", "sofa", "qsofa", "sirs", "label_6h", "hr_age_h"]:
        assert c in p.columns
    assert p.post_onset.sum() == 0 and p.label_6h.sum() > 0 and set(hosp.values()) == {"A", "B"}
    assert p.gcs.isna().all() and (p.n_pressors == 0).all()          # not in this dataset: never invented


def test_full_board_from_physionet(tmp_path, monkeypatch):
    raw = fake(); cache = tmp_path / "c"; cache.mkdir()
    raw.to_csv(cache / "sepsis2019.csv.gz", index=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "--source", "csv", "--cache", str(cache), "--out", "out", "--board", "20"])
    PN.main()
    snap = pd.read_csv("out/stage3_risks_snapshot.csv"); src = json.load(open("out/stage3_source.json"))
    assert len(snap) == 20 and snap.criticality.is_monotonic_decreasing and "PhysioNet" in src["label"]
    assert not snap.risk_1.fillna("").str.contains("GCS|qSOFA|SOFA|NEWS2|KDIGO|SCAI|ROX").any()   # staff see measurements, not score names
    ids = json.load(open("out/stage3_physionet_ids.json")); assert len(ids["board_stay_ids"]) == 20
    assert os.path.exists("models/stage3_heads.joblib")
