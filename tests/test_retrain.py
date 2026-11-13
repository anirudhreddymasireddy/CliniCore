"""Trigger logic of automatic retraining (no model training involved)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
import pandas as pd
from retrain import Store, Service, REG


def make(tmp_path, promote=True, **kw):
    calls = {1: [], 3: []}
    def f(stage):
        def fn(fb, v):
            calls[stage].append((len(fb), v)); return dict(promoted=promote, detail={})
        return fn
    st = Store("csv", str(tmp_path)); return st, Service(st, f(1), f(3), **kw), calls


def doc(st, n, wrong=True):
    for i in range(n):
        st.add_doctor_label(i, "A", "B" if wrong else "A", 0.5, "low_confidence")


def test_stage1_fires_at_exactly_50(tmp_path):
    st, svc, calls = make(tmp_path)
    doc(st, 49); assert svc.check() == [] and not calls[1]
    doc(st, 1); ev = svc.check()
    assert len(ev) == 1 and ev[0]["stage"] == 1 and ev[0]["n_new"] == 50 and calls[1] == [(50, 1)]


def test_counter_resets_and_no_retrain_loop(tmp_path):
    st, svc, calls = make(tmp_path)
    doc(st, 50); svc.check(); assert svc.check() == []            # same labels never trigger twice
    doc(st, 49); assert svc.check() == []
    doc(st, 1); assert len(svc.check()) == 1 and calls[1][-1] == (100, 2)


def test_stage3_counts_only_false_alarms(tmp_path):
    st, svc, calls = make(tmp_path)
    for i in range(120): st.add_alert_verdict(1, i, "sepsis", "true_alarm")
    for i in range(49): st.add_alert_verdict(2, i, "sepsis", "false_alarm", reason_code="expected_for_this_patient")
    assert svc.check() == []                                       # confirmed alarms do not count
    st.add_alert_verdict(2, 99, "aki", "false_alarm", reason_code="expected_for_this_patient"); ev = svc.check()
    assert len(ev) == 1 and ev[0]["stage"] == 3 and ev[0]["n_new"] == 50 and calls[3][0][0] == 170   # but all of them train the model


def test_rejected_candidate_is_logged_and_restarts_counter(tmp_path):
    st, svc, calls = make(tmp_path, promote=False)
    doc(st, 50); ev = svc.check()
    assert ev[0]["promoted"] is False and ev[0]["version_after"] == 0     # old model stays live
    assert svc.check() == []
    reg = st.read(REG); assert len(reg) == 1 and not bool(reg.promoted.iloc[0])


def test_stages_are_independent(tmp_path):
    st, svc, calls = make(tmp_path)
    doc(st, 50)
    for i in range(10): st.add_alert_verdict(1, i, "arf", "false_alarm", reason_code="expected_for_this_patient")
    ev = svc.check(); assert [e["stage"] for e in ev] == [1] and not calls[3]


def test_corrections_only_option(tmp_path):
    st, svc, calls = make(tmp_path, corrections_only=True)
    doc(st, 80, wrong=False); assert svc.check() == []
    doc(st, 50, wrong=True); assert len(svc.check()) == 1


def test_false_alarm_needs_a_reason(tmp_path):
    import pytest
    st = Store("csv", str(tmp_path))
    with pytest.raises(ValueError):
        st.add_alert_verdict(1, 1, "sepsis", "false_alarm")
    with pytest.raises(ValueError):
        st.add_alert_verdict(1, 1, "sepsis", "false_alarm", reason_code="other")
    st.add_alert_verdict(1, 1, "sepsis", "false_alarm", reason_code="other", note="patient was agitated")
    st.add_alert_verdict(1, 2, "sepsis", "true_alarm")
    fb = st.read(__import__("retrain").FB3); assert list(fb.reason_code.fillna("")) == ["other", ""] and fb.note.iloc[0] == "patient was agitated"
