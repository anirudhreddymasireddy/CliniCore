"""Snapshot data access: serves the same payloads as repo_live from ui/data/snapshot.json, without Exasol."""
import json
from pathlib import Path

SNAPSHOT = Path(__file__).resolve().parents[1] / "data" / "snapshot.json"
_data = None


def _load():
    global _data
    if _data is None:
        if not SNAPSHOT.exists():
            raise FileNotFoundError(f"{SNAPSHOT} missing - run: uv run python export_snapshot.py")
        _data = json.loads(SNAPSHOT.read_text())
    return _data


def meta():
    return {**_load()["meta"], "snapshot_generated_at": _load()["generated_at"]}


def patient(token, upto_hour=72):
    p = _load()["patients"].get(token)
    if p is None:
        raise KeyError(f"unknown patient token {token}")
    upto = min(int(upto_hour), p["upto_hour"])
    hi = (upto + 1) * 60
    return {**p, "upto_hour": upto, "hours": [h for h in p["hours"] if h["hour_index"] <= upto],
            "vitals": {**p["vitals"], "points": [v for v in p["vitals"]["points"] if v["t"] <= hi],
                       "artifact_times": [t for t in p["vitals"].get("artifact_times", []) if t <= hi]}}


def ward(ward_id):
    w = _load()["wards"].get(ward_id)
    if w is None:
        raise KeyError(f"unknown ward {ward_id}")
    return w


def presentation(config=None):
    return _load()["presentation"]
