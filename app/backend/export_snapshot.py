"""Export the anonymized demo data from Exasol (read-only) to ui/data/snapshot.json.

Contains: the presentation scenario (config/demo_patients.json), replay wards A and B, and every patient of
them up to ICU hour 72 - exactly the payloads the live backend serves, so CAREPRIORITY_DATA=snapshot works
without the database. Verifies that no raw stay identifier (EICU-/MIMIC3-/MIMIC4-<id>) is in the file.

Usage: uv run python export_snapshot.py
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import repo_live

UI = Path(__file__).resolve().parents[1]
OUT = UI / "data" / "snapshot.json"
CONFIG = UI / "config" / "demo_patients.json"


def main():
    config = json.loads(CONFIG.read_text())
    presentation = repo_live.presentation(config)
    wards = {w: repo_live.ward(w) for w in ("A", "B")}
    patients = {}
    for t, p in presentation["patients"].items():
        patients[t] = repo_live.patient(t, 72)
    for w in wards.values():
        for b in w["beds"]:
            if b["token"] not in patients:
                patients[b["token"]] = repo_live.patient(b["token"], 72)
    snap = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "meta": repo_live.meta(), "presentation": presentation, "wards": wards, "patients": patients}
    text = json.dumps(snap, separators=(",", ":"), ensure_ascii=False)
    leak = re.search(r"(EICU|MIMIC3|MIMIC4)-\d", text)
    if leak:
        raise SystemExit(f"refusing to write snapshot: raw identifier pattern found ({leak.group(0)}...)")
    OUT.write_text(text)
    n_pts = sum(len(p["vitals"]["points"]) for p in patients.values())
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB): presentation beds {len(presentation['beds'])}, "
          f"wards {', '.join(f'{k}={len(v['beds'])} beds' for k, v in wards.items())}, patients {len(patients)}, "
          f"raw vital points {n_pts}; no raw identifiers")


if __name__ == "__main__":
    main()
