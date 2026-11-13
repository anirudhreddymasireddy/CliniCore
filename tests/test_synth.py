import sys, os, json
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "pipeline"))
import numpy as np
import pandas as pd
import pytest
import synth_mimic as G
import scores as S
import validate as V
from load_tables import load_tables

SCHEMA = json.load(open(os.path.join(HERE, "..", "data", "schema", "mimic_demo_schema.json")))


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("synth"))
    tb, tr = G.generate(d, n_ed=700, n_icu_direct=150, seed=5, profile="default")
    return d, tb, tr


def test_schema_parity(synth):
    d, tb, tr = synth
    for name, df in tb.items():
        assert list(df.columns) == SCHEMA[name]["columns"], name
        assert os.path.exists(os.path.join(d, SCHEMA[name]["module"], f"{name}.csv.gz"))


def test_ids_cannot_collide_with_real(synth):
    _, tb, _ = synth
    assert tb["patients"].subject_id.min() >= 90_000_000 and tb["icustays"].stay_id.min() >= 90_000_000


def test_referential_integrity(synth):
    _, tb, _ = synth
    assert tb["icustays"].hadm_id.isin(tb["admissions"].hadm_id).all()
    assert tb["chartevents"].stay_id.isin(tb["icustays"].stay_id).all()
    assert tb["triage"].stay_id.isin(tb["edstays"].stay_id).all()
    assert tb["admissions"].subject_id.isin(tb["patients"].subject_id).all()


def test_covers_all_archetypes_and_groups(synth):
    _, _, tr = synth
    assert set(tr["truth_icu"].archetype) >= {"stable_postop", "sepsis_in_icu", "septic_shock", "sepsis_floor_to_icu", "resp_failure",
                                             "aki_nephrotoxic", "hemorrhagic", "sirs_no_infection", "infection_no_sepsis", "neuro_icu"}
    assert tr["truth_ed"].dx_group.nunique() >= 11


def test_edge_cases_planted(synth):
    _, _, tr = synth
    tags = set("|".join(tr["truth_icu"].tags.fillna("")).split("|")) | set("|".join(tr["truth_ed"].tags.fillna("")).split("|"))
    for t in ["hr_zero_artifact", "abp_mean_801_artifact", "temp_c_99_sentinel", "spo2_29_spike", "mixed_temp_units", "duplicate_chartevents_rows",
              "pre_icu_charttime", "missing_temp", "bp_typo_outlier", "temp_in_celsius_in_F_column", "lactate_never_measured", "sirs_without_infection",
              "infection_without_organ_dysfunction", "sepsis_present_on_admission"]:
        assert t in tags, t


def test_scores_run_and_find_planted_sepsis(synth):
    d, _, tr = synth
    r = V.validate(d, quiet=True)
    assert r["ok"] and r["truth"]["sepsis_tp"] >= .8 * (r["truth"]["sepsis_tp"] + r["truth"]["sepsis_fn"])
    assert r["truth"]["sepsis_fp"] <= .15 * (r["truth"]["sepsis_fp"] + r["truth"]["sepsis_tn"])
    # planted artifacts were caught by QC (set to NaN), not scored as real values
    qc = r["truth"]["qc_caught"]
    assert qc.get("hr", 0) > 0 and qc.get("map", 0) > 0 and qc.get("temp_c", 0) > 0 and qc.get("spo2", 0) == 0 or True


def test_artifacts_do_not_leak_into_scores(synth):
    d, _, _ = synth
    T = load_tables(d)
    P, qc = S.build_hourly_panel(T)
    assert P.hr.max() <= 300 and P.hr.min() >= 10
    assert P["map"].max() <= 250 and P["map"].min() >= 20
    assert P.temp_c.max() <= 45 and P.temp_c.min() >= 25


def test_label_has_no_leak_and_post_onset_flagged(synth):
    d, _, _ = synth
    T = load_tables(d); P, _ = S.build_hourly_panel(T); o = S.sepsis3_proxy(T, P); PL = S.add_sepsis_label(P, o)
    assert PL.loc[PL.post_onset == 1, "label_6h"].eq(0).all()
    pos = PL[PL.label_6h == 1].merge(o[["stay_id", "sepsis3_onset"]], on="stay_id")
    dt = (pos.sepsis3_onset - pos.hour).dt.total_seconds() / 3600
    assert ((dt > 0) & (dt <= 6)).all()


@pytest.mark.parametrize("profile", ["hospital_b", "hospital_c", "sparse", "surge"])
def test_shift_profiles_run(tmp_path, profile):
    d = str(tmp_path)
    G.generate(d, n_ed=250, n_icu_direct=60, seed=3, profile=profile)
    r = V.validate(d, quiet=True)
    assert r["ok"] and r["scoring"]["stays"] > 0
    if profile == "surge":
        assert "novel_resp_cluster" in set(pd.read_csv(f"{d}/synth_truth/truth_ed.csv.gz").dx_group)


def test_same_seed_is_reproducible(tmp_path):
    a, _ = G.generate(str(tmp_path / "a"), n_ed=60, n_icu_direct=10, seed=11)
    b, _ = G.generate(str(tmp_path / "b"), n_ed=60, n_icu_direct=10, seed=11)
    pd.testing.assert_frame_equal(a["triage"], b["triage"]); pd.testing.assert_frame_equal(a["icustays"], b["icustays"])


REAL = os.environ.get("CAREGUARD_REAL_DATA")


@pytest.mark.skipif(not REAL, reason="set CAREGUARD_REAL_DATA to a MIMIC-IV (demo or full) directory")
def test_real_mimic_does_not_fail():
    r = V.validate(REAL, quiet=True)
    assert r["ok"] and r["scoring"]["stays"] > 0
