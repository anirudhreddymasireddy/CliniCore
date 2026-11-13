"""Backend tests: run in both data modes (live needs Exasol; snapshot does not).

    uv run --group dev pytest -q tests            (live + snapshot)
    CAREPRIORITY_DATA=snapshot uv run --group dev pytest -q tests -k snapshot
"""
import importlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
ID_LEAK = re.compile(r"(EICU|MIMIC3|MIMIC4)-\d")
CONFIG = json.loads((HERE.parent / "config" / "demo_patients.json").read_text())


def make_client(mode, tmp_path, monkeypatch):
    monkeypatch.setenv("CAREPRIORITY_DATA", mode)
    import main
    main = importlib.reload(main)
    monkeypatch.setattr(main, "FEEDBACK_LOG", tmp_path / "feedback_log.csv")
    return TestClient(main.app), main


@pytest.fixture(params=["snapshot", "live"])
def client(request, tmp_path, monkeypatch):
    return make_client(request.param, tmp_path, monkeypatch)


def test_health(client):
    c, main = client
    r = c.get("/api/health").json()
    assert r["status"] == "ok" and r["data_mode"] == main.DATA_MODE
    if main.DATA_MODE == "live":
        assert r["db_user"] == "MCP_READONLY"


def test_presentation_from_snapshot(client):
    c, _ = client
    p = c.get("/api/presentation").json()
    assert 8 <= len(p["beds"]) <= 10
    roles = {b["role"] for b in p["beds"]}
    assert {"critical", "medium", "calm"} <= roles
    crit = p["patients"][CONFIG["critical"]["token"]]
    alert = [h for h in crit["hours"] if h["hour_index"] == CONFIG["critical"]["alert_hour"]][0]
    assert alert["tier"] == "CRITICAL" and alert["alert_fired"] == 1 and "shock" in alert["trigger_components"]
    pre = [v for v in crit["vitals"]["points"] if (CONFIG["critical"]["alert_hour"] - 5) * 60 <= v["t"] <= (CONFIG["critical"]["alert_hour"] + 1) * 60]
    assert sum(v["map"] is not None for v in pre) >= 60, "5-minute MAP before the alert"
    assert not ID_LEAK.search(json.dumps(p))


def test_ward_and_patient(client):
    c, _ = client
    for wid in ("A", "B"):
        w = c.get(f"/api/ward/{wid}").json()
        assert len(w["beds"]) == 12
        tok = w["beds"][0]["token"]
        pt = c.get(f"/api/patient/{tok}?upto=30").json()
        assert pt["hours"] and max(h["hour_index"] for h in pt["hours"]) <= 30
        assert all(v["t"] <= 31 * 60 for v in pt["vitals"]["points"])
        assert re.fullmatch(r"P-[0-9A-F]{4}", pt["anon_id"])
        assert not ID_LEAK.search(json.dumps(w) + json.dumps(pt))
    assert c.get("/api/patient/doesnotexist").status_code == 404


def test_live_matches_snapshot(tmp_path, monkeypatch):
    live, _ = make_client("live", tmp_path, monkeypatch)
    a = live.get("/api/ward/A").json()
    snap, _ = make_client("snapshot", tmp_path, monkeypatch)
    b = snap.get("/api/ward/A").json()
    assert [x["token"] for x in a["beds"]] == [x["token"] for x in b["beds"]]
    assert a["beds"][0]["hours"]["24"]["tier"] == b["beds"][0]["hours"]["24"]["tier"] if "24" in a["beds"][0]["hours"] else True


def test_feedback(client):
    c, main = client
    ok = c.post("/api/feedback", json={"ward": "P", "bed": 4, "anon_id": "P-ABCD", "replay_hour": 22, "alert_hour": 22,
                                       "tier": "CRITICAL", "components": "shock", "action": "call doctor"})
    assert ok.status_code == 200
    bad = c.post("/api/feedback", json={"ward": "P", "bed": 4, "anon_id": "P-ABCD", "replay_hour": 22, "action": "false alarm"})
    assert bad.status_code == 422
    fa = c.post("/api/feedback", json={"ward": "P", "bed": 4, "anon_id": "P-ABCD", "replay_hour": 22,
                                       "action": "false alarm", "reason": "Artifact"})
    assert fa.status_code == 200
    rows = c.get("/api/feedback?anon_id=P-ABCD&ward=P").json()
    assert [r["action"] for r in rows] == ["call doctor", "false alarm"]
    assert c.post("/api/feedback", json={"ward": "P", "bed": 1, "anon_id": "EICU-1", "replay_hour": 1, "action": "acknowledge"}).status_code == 422


def test_snapshot_mode_never_touches_db(tmp_path, monkeypatch):
    c, main = make_client("snapshot", tmp_path, monkeypatch)
    assert "repo_live" not in main.repo.__name__
    assert c.get("/api/ward/B").status_code == 200
