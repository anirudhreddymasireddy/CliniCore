"""CliniCore backend (FastAPI). Serves the built web app and two APIs:
  /api/ed, /api/icu, /api/trust  -> CliniCore Stage 1 + Stage 3 (clinicore_api.py, writes only staff feedback and intakes)
  /api/presentation, /api/ward.. -> the ICU central-monitoring demo (originally CarePriority, read-only)

Run (Windows or Linux):  python -m uvicorn main:app --host 127.0.0.1 --port 8765     then open http://127.0.0.1:8765

Data source switch: CAREPRIORITY_DATA=live (default; Exasol as MCP_READONLY) or snapshot (ui/data/snapshot.json,
no database). The presentation scenario is always served from the snapshot so the pitch cannot depend on Exasol.
Nurse actions are appended to ui/feedback_log.csv only.

Run: uv run uvicorn main:app --host 127.0.0.1 --port 8765
"""
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import repo_snapshot

UI = Path(__file__).resolve().parents[1]
FEEDBACK_LOG = UI / "feedback_log.csv"
LOG_FIELDS = ["logged_at_utc", "ward", "bed", "anon_id", "source", "replay_hour", "alert_hour", "tier",
              "components", "action", "reason"]
ACTIONS = {"call doctor", "acknowledge", "acknowledge (notification)", "snooze 30 min", "false alarm"}
FALSE_ALARM_REASONS = {"Artifact", "Already known & treated", "Not clinically relevant"}
DATA_MODE = os.environ.get("CAREPRIORITY_DATA", "snapshot").lower()
if DATA_MODE not in ("live", "snapshot"):
    raise SystemExit("CAREPRIORITY_DATA must be 'live' or 'snapshot'")

if DATA_MODE == "live":
    import repo_live as repo
else:
    repo = repo_snapshot

app = FastAPI(title="CliniCore", version="1.0")

import clinicore_api  # noqa: E402
app.include_router(clinicore_api.router)

import time as _time  # noqa: E402
import base64 as _b64, hmac as _hmac  # noqa: E402
from fastapi.responses import Response as _Resp  # noqa: E402

_PW = os.environ.get("CLINICORE_PASSWORD", "")               # set it on any public deployment; empty = no login (local use)


@app.middleware("http")
async def _shared_password(request, call_next):
    """Optional shared password (browser login box). User name: anything; password: CLINICORE_PASSWORD."""
    if not _PW:
        return await call_next(request)
    auth = request.headers.get("authorization", "")
    if auth.startswith("Basic "):
        try:
            given = _b64.b64decode(auth[6:]).decode("utf-8", "ignore").split(":", 1)[-1]
            if _hmac.compare_digest(given, _PW):
                return await call_next(request)
        except Exception:
            pass
    return _Resp("CliniCore - login required", status_code=401, headers={"WWW-Authenticate": 'Basic realm="CliniCore", charset="UTF-8"'})



@app.middleware("http")
async def _log_slow(request, call_next):
    """Prints every API call slower than 300 ms in the app window (for finding lag)."""
    t0 = _time.perf_counter()
    response = await call_next(request)
    ms = (_time.perf_counter() - t0) * 1000
    if ms > 300 and request.url.path.startswith("/api"):
        print(f"SLOW {ms:6.0f} ms  {request.method} {request.url.path}", flush=True)
    return response


def _call(fn, *args):
    try:
        return fn(*args)
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'"))
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except Exception as e:  # database unreachable etc.: never leak connection details
        raise HTTPException(503, f"data source unavailable ({DATA_MODE}): {type(e).__name__}")


@app.get("/api/health")
def health():
    out = {"data_mode": DATA_MODE, "snapshot_available": repo_snapshot.SNAPSHOT.exists(), "status": "ok"}
    if DATA_MODE == "live":
        import db
        try:
            out["db_user"] = db.current_user()
        except Exception as e:
            out.update(status="degraded", db_error=type(e).__name__)
    return out


@app.get("/api/meta")
def meta():
    return _call(repo.meta)


@app.get("/api/presentation")
def presentation():
    data = _call(repo_snapshot.presentation)
    return {**data, "meta": _call(repo_snapshot.meta)}


@app.get("/api/ward/{ward_id}")
def ward(ward_id: str):
    if ward_id not in ("A", "B"):
        raise HTTPException(404, "ward must be A or B")
    return _call(repo.ward, ward_id)


@app.get("/api/patient/{token}")
def patient(token: str, upto: int = 72, demo: bool = False):
    src = repo_snapshot if demo else repo
    return _call(src.patient, token, max(0, min(upto, 72)))


class Feedback(BaseModel):
    ward: str = Field(max_length=10)
    bed: int
    anon_id: str = Field(pattern=r"^P-[0-9A-F]{4}$")
    source: str = Field(default="", max_length=10)
    replay_hour: int
    alert_hour: Optional[int] = None
    tier: str = Field(default="", max_length=10)
    components: str = Field(default="", max_length=60)
    action: str
    reason: str = ""


@app.post("/api/feedback")
def add_feedback(fb: Feedback):
    if fb.action not in ACTIONS:
        raise HTTPException(422, f"unknown action {fb.action}")
    if fb.action == "false alarm" and fb.reason not in FALSE_ALARM_REASONS:
        raise HTTPException(422, "a false alarm needs a reason: " + ", ".join(sorted(FALSE_ALARM_REASONS)))
    new = not FEEDBACK_LOG.exists()
    row = {"logged_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), **fb.model_dump()}
    with FEEDBACK_LOG.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in LOG_FIELDS})
    return {"logged": True, "file": str(FEEDBACK_LOG.name)}


@app.get("/api/feedback")
def list_feedback(anon_id: str, ward: str):
    if not FEEDBACK_LOG.exists():
        return []
    with FEEDBACK_LOG.open(newline="") as fh:
        return [r for r in csv.DictReader(fh) if r["anon_id"] == anon_id and r["ward"] == ward]


# ------------------------------------------------------------------ the built web app (frontend/dist) on the same port
from fastapi.staticfiles import StaticFiles  # noqa: E402
DIST = UI / "frontend" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="web")
