"""CliniCore API tests (Stage 1 intake/queue/labels, Stage 3 board/verdicts, trust). Needs the trained pipeline outputs
(models/stage1_live.json, stage1_predictions.csv, results/stage3_risks_snapshot.csv, nhamcs_cache/). Feedback goes to a temp folder."""
import importlib, os, sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
PIPE = Path(os.environ.get("CLINICORE_PIPELINE", HERE.parents[1]))
pytestmark = pytest.mark.skipif(not (PIPE / "models" / "stage1_live.json").exists(), reason="pipeline not trained here")


@pytest.fixture()
def c(tmp_path, monkeypatch):
    monkeypatch.setenv("CLINICORE_STATE", str(tmp_path)); monkeypatch.setenv("CAREPRIORITY_DATA", "snapshot")
    import clinicore_api, main
    importlib.reload(clinicore_api); importlib.reload(main)
    return TestClient(main.app)


GOOD = dict(age=67, sex="male", ambulance=True, temp_c=38.9, hr=118, rr=26, spo2=91, sbp=98, dbp=60, pain=3, reasons=[14400, 10100], chronic=["COPD"])


def test_intake_validation_rejects_impossible_values(c):
    r = c.post("/api/ed/intake", json=dict(age=12, sex="female", hr=900, reasons=[]))
    assert r.status_code == 422 and any("Heart rate" in m for m in r.json()["detail"]) and any("reason" in m for m in r.json()["detail"])
    assert c.post("/api/ed/intake", json=dict(GOOD, sbp=80, dbp=90)).status_code == 422          # diastolic above systolic


def test_intake_routes_and_patient_never_sees_unsure_guess(c):
    r = c.post("/api/ed/intake", json=GOOD).json()
    assert r["clinician"]["route"] in ("PATIENT_VIEW", "DOCTOR_QUEUE") and len(r["clinician"]["top"]) == 3
    if r["clinician"]["route"] == "DOCTOR_QUEUE":
        assert r["patient"]["kind"] == "doctor" and "could be" not in r["patient"]["headline"]
    q = c.get("/api/ed/queue").json()
    assert r["clinician"]["route"] == "PATIENT_VIEW" or any(x["visit_id"] == r["visit_id"] for x in q["live"])


def test_doctor_label_counts_and_patient_sees_review(c):
    v = c.post("/api/ed/intake", json=GOOD).json()["visit_id"]
    assert c.post("/api/ed/label", json=dict(visit_id=v, label="not_a_group")).status_code == 422
    r = c.post("/api/ed/label", json=dict(visit_id=v, label="respiratory_infection")).json()
    assert r["counter"]["n"] == 1
    assert c.get(f"/api/ed/patient/{v}").json()["kind"] == "reviewed"
    assert all(x["visit_id"] != v for x in c.get("/api/ed/queue").json()["live"])


def test_icu_false_alarm_needs_reason_and_counts(c):
    b = c.get("/api/icu/board").json(); bed = b["beds"][0]; risk = bed["risks"][0]["domain"]
    assert b["beds"][0]["criticality"] >= b["beds"][-1]["criticality"]                              # sickest first
    assert c.post("/api/icu/verdict", json=dict(stay_id=bed["stay_id"], risk=risk, verdict="false_alarm")).status_code == 422
    ok = c.post("/api/icu/verdict", json=dict(stay_id=bed["stay_id"], risk=risk, verdict="false_alarm", reason_code="other", note="patient coughing")).json()
    assert ok["counter"]["n"] == 1 and c.get("/api/icu/false-alarms").json()[0]["note"] == "patient coughing"
    c.post("/api/icu/verdict", json=dict(stay_id=bed["stay_id"], risk=risk, verdict="true_alarm"))
    assert c.get("/api/icu/board").json()["counter"]["n"] == 1                                     # real warnings do not count


def test_trust_and_web_app(c):
    t = c.get("/api/trust").json(); assert "rules" in t and t["rules"]["retrain_after"] == 50
    assert c.get("/").status_code == 200 and c.get("/api/presentation").status_code == 200


def test_notify_doctor_sbar_ack_and_auto(c):
    bed = c.get("/api/icu/board").json()["beds"][0]
    p = c.get(f"/api/notify/preview?stay_id={bed['stay_id']}&urgency=emergency").json()
    assert p["suggested"][0] == "icu_attending" and p["message"].startswith("S: Bed") and "\nR: Come now" in p["message"]
    assert c.post("/api/notify", json=dict(stay_id=bed["stay_id"], doctors=["nobody"], urgency="urgent")).status_code == 422
    r = c.post("/api/notify", json=dict(stay_id=bed["stay_id"], doctors=["icu_attending", "nephrology"], urgency="urgent", note="check urine")).json()
    assert r["sent"] == 2 and "check urine" in r["message"]
    inbox = c.get("/api/notify?doctor=icu_attending").json(); assert inbox[0]["status"] == "sent"
    c.post("/api/notify/ack", json=dict(notify_id=inbox[0]["notify_id"], doctor="icu_attending", action="on_my_way"))
    b = next(x for x in c.get("/api/icu/board").json()["beds"] if x["stay_id"] == bed["stay_id"])
    st = {d["doctor"]: d["status"] for d in b["notified"]["doctors"]}
    assert st["ICU attending on call"] == "on_my_way" and st["Nephrology on call"] == "sent"
    first = c.post("/api/notify/auto", json=dict(enabled=True)).json()["sent_now"]
    assert c.post("/api/notify/auto", json=dict(enabled=True)).json()["sent_now"] == 0      # never twice while open
    assert first >= 0
    c.post("/api/notify/auto", json=dict(enabled=False))
