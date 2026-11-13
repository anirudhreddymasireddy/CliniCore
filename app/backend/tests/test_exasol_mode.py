"""CliniCore in Exasol mode, against an in-memory stand-in for Exasol (no database needed).

The stand-in is strict where Exasol is strict: tables must exist before they are read, imports must match the table's
columns, text longer than the column's VARCHAR size is rejected, and some column names are refused as reserved words
(more than Exasol actually refuses, so the rename path is exercised). Run the real thing with tests/selftest_exasol.py."""
import contextlib, importlib, os, re, sys, threading, types
from pathlib import Path
import pandas as pd
import pytest
from fastapi.testclient import TestClient

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
PIPE = Path(os.environ.get("CLINICORE_PIPELINE", HERE.parents[1]))
pytestmark = pytest.mark.skipif(not (PIPE / "models" / "stage1_live.json").exists(), reason="pipeline not trained here")
RESERVED = {"LEVEL", "ACTION", "STATUS", "AUTO", "MESSAGE"}


class Res:
    def __init__(self, rows): self.rows = rows
    def fetchone(self): return self.rows[0]
    def fetchall(self): return self.rows


class FakeExasol:
    def __init__(self):
        self.tables, self.types, self.schema, self.lock, self.busy = {}, {}, None, threading.Lock(), False

    @contextlib.contextmanager
    def _guard(self):                                       # a real pyexasol connection breaks if two threads use it at once
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("concurrent use of one connection")
        try:
            yield
        finally:
            self.lock.release()

    def execute(self, q):
        with self._guard():
            q = " ".join(q.split())
            if m := re.match(r'CREATE SCHEMA IF NOT EXISTS "(\w+)"', q): return Res([(0,)])
            if m := re.match(r'OPEN SCHEMA "(\w+)"', q): self.schema = m.group(1); return Res([(0,)])
            if m := re.search(r"EXA_ALL_TABLES WHERE TABLE_SCHEMA='(\w+)' AND TABLE_NAME='(\w+)'", q):
                return Res([(int((m.group(1), m.group(2)) in self.tables),)])
            if m := re.search(r"EXA_ALL_COLUMNS WHERE COLUMN_SCHEMA='(\w+)' AND COLUMN_TABLE='(\w+)'", q):
                return Res([(c,) for c in self.tables[(m.group(1), m.group(2))].columns])
            if m := re.match(r'CREATE OR REPLACE TABLE "(\w+)" \((.*)\)$', q):
                cols = re.findall(r'"(\w+)" (\w+(?:\(\d+\))?)', m.group(2))
                for c, _ in cols:
                    if c in RESERVED:
                        raise RuntimeError(f"{c} not allowed as column name.")
                key = (self.schema, m.group(1))
                self.tables[key] = pd.DataFrame(columns=[c for c, _ in cols]); self.types[key] = dict(cols)
                return Res([(0,)])
            if m := re.match(r'DROP TABLE (?:IF EXISTS )?"?(\w+)"?\."?(\w+)"?', q):
                self.tables.pop((m.group(1), m.group(2)), None); return Res([(0,)])
            raise AssertionError(f"unexpected SQL: {q}")

    def import_from_pandas(self, d, target):
        with self._guard():
            key = tuple(target)
            assert key in self.tables, f"import into missing table {key}"
            assert list(d.columns) == list(self.tables[key].columns), f"column mismatch {list(d.columns)} vs {list(self.tables[key].columns)}"
            for c, t in self.types[key].items():
                if m := re.match(r"VARCHAR\((\d+)\)", t):
                    too_long = d[c].dropna().astype(str).str.len() > int(m.group(1))
                    if too_long.any():
                        raise RuntimeError(f"data exception - string too long for {key}.{c} ({t})")
            self.tables[key] = pd.concat([self.tables[key], d], ignore_index=True) if len(self.tables[key]) else d.reset_index(drop=True)

    def export_to_pandas(self, q):
        with self._guard():
            m = re.search(r'FROM "(\w+)"\."(\w+)"', q)
            key = (m.group(1), m.group(2))
            if key not in self.tables:
                raise RuntimeError(f"object {key} not found")
            return self.tables[key].copy()


@pytest.fixture()
def exa(tmp_path, monkeypatch):
    fake = FakeExasol()
    monkeypatch.setitem(sys.modules, "pyexasol", types.SimpleNamespace(connect=lambda **k: fake))
    monkeypatch.setenv("EXA_PASS", "x"); monkeypatch.setenv("CLINICORE_SOURCE", "exasol"); monkeypatch.setenv("CLINICORE_STATE", str(tmp_path))
    monkeypatch.setenv("CAREPRIORITY_DATA", "snapshot")
    sys.path.insert(0, str(PIPE / "pipeline"))
    from stage1_pipeline import write_exasol
    for table, f in {"ST1_PREDICTIONS": "stage1_predictions.csv", "ST3_RISKS": "results/stage3_risks_snapshot.csv",
                     "ST3_RISK_TIMELINE": "results/stage3_risk_timeline.csv", "ST3_RISK_METRICS": "results/stage3_multirisk_metrics.csv"}.items():
        write_exasol(fake, "CAREGUARD_RESULTS", table, pd.read_csv(PIPE / f), replace=True)       # what the pipelines write
    write_exasol(fake, "CAREGUARD_RESULTS", "ST3_SOURCE", pd.DataFrame([dict(data="x", label="Real ICU patients · test", missing="urine output, vasopressors")]), replace=True)
    import appdata, clinicore_api, main
    appdata.Data._cache.clear()
    importlib.reload(clinicore_api); importlib.reload(main)
    yield TestClient(main.app), fake
    importlib.reload(clinicore_api)                         # back to CSV mode for other tests (env restored by monkeypatch)


GOOD = dict(age=67, sex="male", ambulance=True, temp_c=38.9, hr=118, rr=26, spo2=91, sbp=98, dbp=60, pain=3, reasons=[14400, 10100], chronic=["COPD"],
            history_note="Lives alone; " + "long history note " * 20)


def test_full_staff_flow_in_exasol_mode(exa):
    c, fake = exa
    assert c.get("/api/clinicore/health").json()["source"] == "exasol"
    b = c.get("/api/icu/board").json()
    assert b["data_note"] == "Real ICU patients · test" and b["source"]["missing"] == ["urine output", "vasopressors"]
    # ED: two intakes (the first row decides the table layout; the second has longer text)
    v1 = c.post("/api/ed/intake", json=dict(GOOD, history_note="")).json()["visit_id"]
    v2 = c.post("/api/ed/intake", json=GOOD).json()["visit_id"]
    assert v2 == v1 + 1 and ("CAREGUARD_RESULTS", "ST1_INTAKE") in fake.tables
    r = c.post("/api/ed/label", json=dict(visit_id=v1, label="respiratory_infection", detail="Community-acquired pneumonia, right lower lobe")).json()
    assert r["counter"]["n"] == 1
    assert "pneumonia" in c.get(f"/api/ed/patient/{v1}").json()["headline"]
    q = c.get("/api/ed/queue").json(); assert all(x["visit_id"] != v1 for x in q["live"]) and q["waiting_total"] > 0
    # ICU: false alarm with a short note first, then a long one (must not hit the first row's text size)
    bed = b["beds"][0]; risk = bed["risks"][0]["domain"]
    c.post("/api/icu/verdict", json=dict(stay_id=bed["stay_id"], risk=risk, verdict="false_alarm", reason_code="sensor_or_measurement_error"))
    ok = c.post("/api/icu/verdict", json=dict(stay_id=bed["stay_id"], risk=risk, verdict="false_alarm", reason_code="other", note="n" * 300))
    assert ok.status_code == 200 and ok.json()["counter"]["n"] == 2
    assert c.get("/api/icu/false-alarms").json()[0]["note"] == "n" * 300
    # notifications: reserved-looking column names survive the round trip
    sent = c.post("/api/notify", json=dict(stay_id=bed["stay_id"], doctors=["icu_attending"], urgency="emergency", note="BP falling")).json()
    assert sent["sent"] == 1
    n = c.get("/api/notify?doctor=icu_attending").json()[0]
    assert n["risk_level"] == bed["level"] and n["auto"] in (0, False) and n["message"].startswith("S: Bed")
    c.post("/api/notify/ack", json=dict(notify_id=n["notify_id"], doctor="icu_attending", action="on_my_way"))
    assert c.get("/api/notify?doctor=icu_attending").json()[0]["status"] == "on_my_way"


def test_parallel_requests_do_not_share_the_connection(exa):
    c, _ = exa
    errors = []

    def hit(path):
        try:
            for _ in range(5):
                assert c.get(path).status_code == 200
        except Exception as e:                              # noqa: BLE001
            errors.append(repr(e))
    ts = [threading.Thread(target=hit, args=(p,)) for p in ["/api/icu/board", "/api/ed/queue", "/api/notify?doctor=icu_attending", "/api/icu/false-alarms"] * 2]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert not errors, errors


def test_retraining_counters_read_exasol_feedback(exa):
    c, fake = exa
    import retrain
    store = retrain.Store("exasol", schema="CAREGUARD_RESULTS", con=fake)
    for i in range(3):
        store.add_alert_verdict(90000000 + i, 1, "sepsis", "false_alarm", reason_code="already_being_treated")
    svc = retrain.Service(store, None, None)
    assert svc.pending(3)[0] == 3
