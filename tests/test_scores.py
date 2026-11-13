import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
import numpy as np
import pandas as pd
import pytest
import scores as S


def n2(**kw):
    base = dict(rr=16, spo2=98, sbp=120, hr=70, temp_c=37.0, gcs=15, o2_supp=0)
    base.update(kw)
    return S.news2_points(**{k: np.array([v], float) for k, v in base.items()})


@pytest.mark.parametrize("rr,pts", [(8, 3), (9, 1), (11, 1), (12, 0), (20, 0), (21, 2), (24, 2), (25, 3)])
def test_news2_rr(rr, pts):
    assert n2(rr=rr).n2_rr[0] == pts


@pytest.mark.parametrize("sp,pts", [(91, 3), (92, 2), (93, 2), (94, 1), (95, 1), (96, 0), (100, 0)])
def test_news2_spo2(sp, pts):
    assert n2(spo2=sp).n2_spo2[0] == pts


@pytest.mark.parametrize("sbp,pts", [(90, 3), (91, 2), (100, 2), (101, 1), (110, 1), (111, 0), (219, 0), (220, 3)])
def test_news2_sbp(sbp, pts):
    assert n2(sbp=sbp).n2_sbp[0] == pts


@pytest.mark.parametrize("hr,pts", [(40, 3), (41, 1), (50, 1), (51, 0), (90, 0), (91, 1), (110, 1), (111, 2), (130, 2), (131, 3)])
def test_news2_hr(hr, pts):
    assert n2(hr=hr).n2_hr[0] == pts


@pytest.mark.parametrize("t,pts", [(35.0, 3), (35.1, 1), (36.0, 1), (36.1, 0), (38.0, 0), (38.1, 1), (39.0, 1), (39.1, 2)])
def test_news2_temp(t, pts):
    assert n2(temp_c=t).n2_temp[0] == pts


def test_news2_total_bands_and_missing():
    assert n2().news2[0] == 0 and n2().news2_band[0] == "low"
    assert n2(o2_supp=1).news2[0] == 2
    assert n2(gcs=14).n2_cons[0] == 3 and n2(gcs=14).news2_band[0] == "low_medium"
    worst = n2(rr=26, spo2=90, sbp=85, hr=135, temp_c=34, gcs=8, o2_supp=1)
    assert worst.news2[0] == 3 * 6 + 2 and worst.news2_band[0] == "high"
    m = S.news2_points(rr=[np.nan], spo2=[98], sbp=[120], hr=[70], temp_c=[37.0], gcs=[15], o2_supp=[0])
    assert np.isnan(m.n2_rr[0]) and m.news2_n_missing[0] == 1 and m.news2[0] == 0
    assert n2(rr=22, hr=95, temp_c=38.5).news2_band[0] == "low"          # 2+1+1 = 4


def test_news2_scale2():
    f = lambda sp, o2: S.news2_points([16], [sp], [120], [70], [37.0], [15], [o2], scale2=True).n2_spo2[0]
    assert f(88, 0) == 0 and f(92, 0) == 0 and f(86, 0) == 1 and f(84, 0) == 2 and f(83, 0) == 3
    assert f(94, 1) == 1 and f(96, 1) == 2 and f(98, 1) == 3 and f(98, 0) == 0


def sofa(**kw):
    n = 1
    args = {k: np.array([v], float) for k, v in kw.items()}
    return S.sofa_components(**args)


def test_sofa_resp():
    assert sofa(pf=90, vent=1).sofa_resp[0] == 4 and sofa(pf=150, vent=1).sofa_resp[0] == 3
    assert sofa(pf=150, vent=0).sofa_resp[0] == 2 and sofa(pf=250, vent=0).sofa_resp[0] == 2
    assert sofa(pf=350, vent=0).sofa_resp[0] == 1 and sofa(pf=450, vent=0).sofa_resp[0] == 0
    assert sofa(sf=140, vent=1).sofa_resp[0] == 4          # S/F surrogate when no PaO2


@pytest.mark.parametrize("plt,pts", [(19, 4), (49, 3), (99, 2), (149, 1), (150, 0)])
def test_sofa_coag(plt, pts):
    assert sofa(platelets=plt).sofa_coag[0] == pts


@pytest.mark.parametrize("b,pts", [(12, 4), (6, 3), (2, 2), (1.2, 1), (1.19, 0)])
def test_sofa_liver(b, pts):
    assert sofa(bilirubin=b).sofa_liver[0] == pts


def test_sofa_cv():
    assert sofa(map_=69).sofa_cv[0] == 1 and sofa(map_=70).sofa_cv[0] == 0
    assert sofa(map_=60, dopamine=4).sofa_cv[0] == 2 and sofa(map_=60, dobutamine=5).sofa_cv[0] == 2
    assert sofa(map_=60, dopamine=6).sofa_cv[0] == 3 and sofa(map_=60, norepi=.1).sofa_cv[0] == 3
    assert sofa(map_=60, norepi=.11).sofa_cv[0] == 4 and sofa(map_=60, epi=.2).sofa_cv[0] == 4
    assert sofa(map_=80, dopamine=16).sofa_cv[0] == 4 and sofa(map_=60, other_pressor=1).sofa_cv[0] == 3


@pytest.mark.parametrize("g,pts", [(15, 0), (14, 1), (13, 1), (12, 2), (10, 2), (9, 3), (6, 3), (5, 4), (3, 4)])
def test_sofa_cns(g, pts):
    assert sofa(gcs=g).sofa_cns[0] == pts


@pytest.mark.parametrize("c,pts", [(1.1, 0), (1.2, 1), (1.9, 1), (2.0, 2), (3.4, 2), (3.5, 3), (4.9, 3), (5.0, 4)])
def test_sofa_renal_creatinine(c, pts):
    assert sofa(creatinine=c).sofa_renal[0] == pts


def test_sofa_renal_urine_and_missing():
    assert sofa(creatinine=1.0, urine_24h=150, urine_hours=24).sofa_renal[0] == 4
    assert sofa(creatinine=1.0, urine_24h=400, urine_hours=24).sofa_renal[0] == 3
    assert sofa(creatinine=1.0, urine_24h=150, urine_hours=6).sofa_renal[0] == 0   # not enough hours charted -> ignored
    r = sofa(platelets=100)
    assert r.n_sofa_missing[0] == 5 and r.sofa[0] == 1   # missing components count 0 but are reported


def test_qsofa_sirs():
    q = S.qsofa([22], [100], [15]); assert q.qsofa[0] == 2 and bool(q.qsofa_pos[0])
    q = S.qsofa([21], [101], [15]); assert q.qsofa[0] == 0
    s = S.sirs([38.5], [95], [18], [13]); assert s.sirs[0] == 3 and bool(s.sirs_pos[0])
    s = S.sirs([37.0], [80], [18], [8]); assert s.sirs[0] == 0


def test_triage_news2_handles_celsius_and_outliers():
    tr = pd.DataFrame(dict(temperature=[98.6, 37.0, 99.0, np.nan], heartrate=[80, 80, 0, 80], resprate=[16, 16, 16, 16],
                           o2sat=[98, 98, 10, 98], sbp=[120, 120, 120, 120]))
    d = S.news2_from_triage(tr)
    assert len(d) == 4
    assert np.isnan(d.n2_hr[2]) and np.isnan(d.n2_spo2[2])              # 0 bpm / SpO2 10 are artifacts -> NaN, not scored
    assert d.n2_temp[1] == 0 and d.n2_temp[0] == 0   # 37.0 in the F column is rescued as Celsius, 98.6 F converted
