#!/usr/bin/env python3
"""Check CliniCore against YOUR Exasol (run on the machine with the database). Prints PASS/FAIL per step; changes nothing
outside the temporary schema CLINICORE_SELFTEST, which it drops at the end.

  python scripts/selftest_exasol.py                      # uses EXA_DSN (default 127.0.0.1:8563), EXA_USER (sys), EXA_PASS
"""
import os, sys, traceback
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline")); sys.path.insert(0, str(ROOT / "app" / "backend")); os.chdir(ROOT)
results = []


def step(name, fn):
    try:
        msg = fn(); results.append(True); print(f"PASS  {name}" + (f"  - {msg}" if msg else ""))
    except Exception as e:                                                  # noqa: BLE001
        results.append(False); print(f"FAIL  {name}  - {str(e).strip()[:300]}")
        if os.environ.get("SELFTEST_DEBUG"):
            traceback.print_exc()


from stage1_pipeline import Source, write_exasol
import retrain
con = None


def connect():
    global con
    con = Source("exasol", "CAREGUARD_RESULTS").con
    return f"connected as {con.execute('SELECT CURRENT_USER').fetchone()[0]}"


def result_tables():
    need = ["ST1_PREDICTIONS", "ST3_RISKS", "ST3_RISK_TIMELINE"]
    have = {r[0]: r[1] for r in con.execute("SELECT TABLE_NAME, TABLE_ROW_COUNT FROM EXA_ALL_TABLES WHERE TABLE_SCHEMA='CAREGUARD_RESULTS'").fetchall()}
    miss = [t for t in need if t not in have]
    if miss:
        raise RuntimeError(f"missing {miss} in CAREGUARD_RESULTS - run nhamcs_stage1.py --exasol and physionet_icu.py --source exasol first")
    return ", ".join(f"{t} {have[t]} rows" for t in need + [t for t in ("ST3_SOURCE",) if t in have])


def roundtrip():
    st = retrain.Store("exasol", schema="CLINICORE_SELFTEST", con=con)
    st.append("T_APPEND", pd.DataFrame([dict(id=1, level="High", note="", action="x")]))                 # short first row, reserved-looking names
    st.append("T_APPEND", pd.DataFrame([dict(id=2, level="Critical", note="n" * 300, action="y")]))       # longer text later
    d = st.read("T_APPEND")
    assert list(d.columns) == ["id", "level", "note", "action"], d.columns
    assert len(d) == 2 and d.note.str.len().max() == 300
    st.add_alert_verdict(1, 1, "sepsis", "false_alarm", reason_code="other", note="selftest")
    return "append, reserved words, long text and feedback all round-trip"


def app_endpoints():
    os.environ["CLINICORE_SOURCE"] = "exasol"; os.environ["CAREPRIORITY_DATA"] = "snapshot"
    from fastapi.testclient import TestClient
    import main
    c = TestClient(main.app); out = []
    for p in ["/api/clinicore/health", "/api/ed/meta", "/api/ed/queue", "/api/icu/board", "/api/notify/meta", "/api/icu/false-alarms"]:
        r = c.get(p)
        assert r.status_code == 200, f"{p} -> {r.status_code} {r.text[:200]}"
        out.append(p.split("/")[-1])
    b = c.get("/api/icu/board").json()
    return f"{len(b['beds'])} ICU patients ({b['data_note']}); endpoints ok: {', '.join(out)}"


step("connect to Exasol", connect)
if con is not None:
    step("pipeline result tables present", result_tables)
    step("write/read staff tables (temporary schema)", roundtrip)
    step("CliniCore API in Exasol mode (read-only calls)", app_endpoints)
    try:
        con.execute('DROP SCHEMA IF EXISTS "CLINICORE_SELFTEST" CASCADE')
    except Exception:
        pass
print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
