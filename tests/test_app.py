import os, sys, tempfile
import pandas as pd, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
from appdata import Data
ROOT = os.path.join(os.path.dirname(__file__), "..")


def _base(tmp):
    pd.DataFrame(dict(visit_id=[1, 2, 3], model_version="v0", predicted_group=["a", "b", "a"], confidence=[.5, .95, .6], margin=[.1, .9, .2],
                      route=["DOCTOR_QUEUE", "PATIENT_VIEW", "DOCTOR_QUEUE"])).to_csv(os.path.join(tmp, "stage1_predictions.csv"), index=False)
    os.makedirs(os.path.join(tmp, "results"))
    pd.DataFrame(dict(stay_id=[10, 11], hour_idx=[5, 6], criticality=[30., 90.], risk_level=["Watch", "Critical"], trend_6h=["steady", "rising"], change_6h=[0., 12.],
                      risk_1=["Sepsis - HIGH: x", "Shock - CRITICAL: y"], domain_1=["sepsis", "shock"])).to_csv(os.path.join(tmp, "results", "stage3_risks_snapshot.csv"), index=False)


def test_queue_labels_and_counters():
    tmp = tempfile.mkdtemp(); _base(tmp); d = Data("csv", tmp, os.path.join(tmp, "st"))
    assert list(d.doctor_queue().visit_id) == [1, 3]                       # least confident first
    assert d.patient_view(1)["kind"] == "doctor" and d.patient_view(2)["kind"] == "answer"
    d.label_visit(1, "b"); assert list(d.doctor_queue().visit_id) == [3] and d.counter(1) == 1
    assert d.icu_board().stay_id.iloc[0] == 11                            # sickest first
    d.give_verdict(11, 6, "shock", "true_alarm"); assert d.counter(3) == 0   # real warnings do not count toward the 50
    d.give_verdict(10, 5, "sepsis", "false_alarm", reason_code="sensor_or_measurement_error"); assert d.counter(3) == 1
    assert len(d.false_alarm_log()) == 1 and "Sensor" in d.false_alarm_log().reason.iloc[0]

