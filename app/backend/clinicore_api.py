"""CliniCore API: Stage 1 (ED intake -> diagnosis group, confidence routing, doctor queue) and Stage 3 (ICU ranked risks,
staff verdicts) on top of the CareGuard pipeline. Writes only staff feedback and intakes, through retrain.Store (CSV folder or Exasol).

Settings (environment variables):
  CLINICORE_PIPELINE  repository root (with pipeline/, models/, results/)          (default: two folders above this file)
  CLINICORE_SOURCE    csv | exasol                                                    (default csv)
  CLINICORE_STATE     csv mode: folder for feedback and intakes                       (default <pipeline>/app_state)
  CLINICORE_SCHEMA    Exasol results schema                                           (default CAREGUARD_RESULTS)
  CLINICORE_NHAMCS    folder with nhamcs_ed.csv.gz and rfv_labels.json                (default <pipeline>/nhamcs_cache)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

PIPE = Path(os.environ.get("CLINICORE_PIPELINE", Path(__file__).resolve().parents[2]))       # repository root
sys.path.insert(0, str(PIPE / "pipeline") if (PIPE / "pipeline").exists() else str(PIPE))
os.chdir(PIPE)                                                     # pipeline modules use paths relative to their folder

import joblib                                                      # noqa: E402
import stage1_pipeline as S1                                       # noqa: E402
import nhamcs as NH                                                # noqa: E402
from appdata import Data, DOMAIN_NAMES                             # noqa: E402
from retrain import FALSE_ALARM_REASONS, THRESHOLD, FB1, FB3, REG  # noqa: E402

SOURCE = os.environ.get("CLINICORE_SOURCE", "csv").lower()
SCHEMA = os.environ.get("CLINICORE_SCHEMA", "CAREGUARD_RESULTS")
STATE = os.environ.get("CLINICORE_STATE", str(PIPE / "app_state"))
NHAMCS = Path(os.environ.get("CLINICORE_NHAMCS", PIPE / "nhamcs_cache"))
INTAKE = "ST1_INTAKE"
INTAKE_ID0 = 900_000_000
AUDIT_RATE = 0.05
TIER = ["Low", "Watch", "High", "Critical"]

router = APIRouter(prefix="/api")
_lock = threading.RLock()                                          # one Exasol session / one CSV writer at a time
_con = None
if SOURCE == "exasol":
    _con = S1.Source("exasol", SCHEMA).con
D = Data(SOURCE, str(PIPE), STATE, SCHEMA, _con)
if SOURCE == "exasol":                                             # one pyexasol connection is not thread-safe: serialise all access
    import time as _time
    _raw_read, _raw_append = D.store.read, D.store.append
    _cache, _tables = {}, {"t": 0.0, "names": set()}
    STAFF_TTL, TABLES_TTL = 3.0, 15.0                              # several pages poll the same tables every few seconds

    def _exists(name):
        if _time.time() - _tables["t"] > TABLES_TTL:
            rows = _con.execute(f"SELECT TABLE_NAME FROM EXA_ALL_TABLES WHERE TABLE_SCHEMA='{SCHEMA}'").fetchall()
            _tables.update(t=_time.time(), names={r[0] for r in rows})
        return name.upper() in _tables["names"]

    def _locked_read(name):
        with _lock:
            hit = _cache.get(name)
            if hit and _time.time() - hit[0] < STAFF_TTL:
                return hit[1].copy()
            df = _raw_read(name) if _exists(name) else pd.DataFrame()   # a table nobody has written yet costs no round trip
            _cache[name] = (_time.time(), df)
            return df.copy()

    def _locked_append(name, df):
        with _lock:
            r = _raw_append(name, df)
            _cache.pop(name, None); _tables["names"].add(name.upper())
            return r
    D.store.read, D.store.append = _locked_read, _locked_append


def pretty(g: str) -> str:
    return NH.GROUP_LABELS.get(str(g), str(g).replace("_", " ").capitalize())


def jsonable(v):
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if (math.isnan(v) or math.isinf(v)) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def records(df: pd.DataFrame) -> list:
    return [jsonable(r) for r in df.to_dict("records")]


def is_audit(visit_id: int) -> bool:
    return int(hashlib.md5(f"audit-{int(visit_id)}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < AUDIT_RATE


# ------------------------------------------------------------------ Stage 1 model (reloads automatically after a promoted retrain)
class Stage1Model:
    def __init__(self):
        self._stamp, self.m, self.meta, self.version = None, None, {}, 0

    def get(self):
        ptr = PIPE / "models" / "stage1_live.json"
        if not ptr.exists():
            raise HTTPException(503, "Stage 1 model not trained yet: run  python nhamcs_stage1.py --cache nhamcs_cache")
        stamp = ptr.stat().st_mtime
        if stamp != self._stamp:
            p = json.loads(ptr.read_text())
            self.m, self.version, self._stamp = joblib.load(PIPE / p["path"]), int(p["version"]), stamp
            mp = PIPE / "models" / "stage1_nhamcs_meta.json"
            self.meta = json.loads(mp.read_text()) if mp.exists() else {}
            self.known = set(self.meta.get("known_tokens", []))
        return self


MODEL = Stage1Model()
RFV = json.loads((NHAMCS / "rfv_labels.json").read_text()) if (NHAMCS / "rfv_labels.json").exists() else {}
NH.RFV_TEXT.update({int(k): v for k, v in RFV.items()})


def assess(frame: pd.DataFrame) -> dict:
    """Prediction + every reason the case may need a person. Used for live intakes."""
    M = MODEL.get(); m, meta = M.m, M.meta
    pred, conf, margin, P = m.predict(frame)
    p, cls = P[0], np.array(m.classes_)
    order = np.argsort(-p)
    q = meta.get("conformal_qhat")
    plausible = [str(cls[i]) for i in order if q is not None and p[i] >= 1 - q]
    flags = []
    r = frame.iloc[0]
    names = {"anchor_age": "age", "temp_c": "temperature", "hr": "heart rate", "rr": "breathing rate", "spo2": "oxygen saturation",
             "sbp": "systolic BP", "dbp": "diastolic BP"}
    for c, (lo, hi) in meta.get("ranges", {}).items():
        v = r.get(c)
        if v is None or pd.isna(v) or lo <= float(v) <= hi:
            continue
        if c == "news2_triage":                                  # combined vital-sign score: shown to staff in plain words only
            flags.append("vital signs are abnormal together more than in 99% of patients the model learned from")
        else:
            flags.append(f"{names.get(c, c)} {float(v):.3g} is outside what 99% of patients the model learned from had ({lo:.3g}-{hi:.3g})")
    unknown = [t for t in str(r.cc).split() if t.startswith("r") and t not in M.known]
    if unknown:
        flags.append("main reason for visit was rare or unseen in training data")
    vitals_missing = sum(pd.isna(r.get(c)) for c in ("temp_c", "hr", "rr", "spo2", "sbp"))
    if vitals_missing >= 3:
        flags.append(f"{vitals_missing} of 5 key vital signs missing")
    tau, mg = meta.get("tau", S1.TAU), meta.get("margin", S1.MARGIN)
    confident = bool(conf[0] >= tau and margin[0] >= mg)
    if confident and flags:
        why = "unusual case: " + flags[0]
    elif confident:
        why = f"confident (probability {conf[0]:.0%} >= {tau:.0%}, clear gap to the 2nd guess)"
    else:
        why = f"not confident enough (probability {conf[0]:.0%}, gap to 2nd guess {margin[0]:.0%})"
    route = "PATIENT_VIEW" if confident and not flags else "DOCTOR_QUEUE"
    return dict(predicted_group=str(pred[0]), confidence=float(conf[0]), margin=float(margin[0]), route=route, route_reason=why, flags=flags,
                top=[dict(group=str(cls[i]), label=pretty(cls[i]), prob=round(float(p[i]), 3)) for i in order[:3]],
                plausible_set=[dict(group=g, label=pretty(g)) for g in plausible], model_version=M.version)


# ------------------------------------------------------------------ Stage 1 endpoints
class Intake(BaseModel):
    age: float
    sex: str = Field(pattern=r"^(male|female|other)$")
    ambulance: bool = False
    temp_c: Optional[float] = None
    hr: Optional[float] = None
    rr: Optional[float] = None
    spo2: Optional[float] = None
    sbp: Optional[float] = None
    dbp: Optional[float] = None
    pain: Optional[float] = None
    acuity: Optional[float] = None
    reasons: List[int] = Field(default_factory=list, max_length=3)
    complaint_text: str = Field(default="", max_length=300)        # free-text complaint (e.g. from another system); reasons or text required
    chronic: List[str] = Field(default_factory=list, max_length=13)
    history_note: str = Field(default="", max_length=500)


class Label(BaseModel):
    visit_id: int
    label: str = Field(max_length=60)
    detail: str = Field(default="", max_length=300)          # the doctor's own words (specific diagnosis / note); the group is what the model learns
    role: str = Field(default="doctor", max_length=30)


LABEL_NOTES = "ST1_LABEL_NOTES"


def label_details() -> dict:
    n = D.store.read(LABEL_NOTES)
    if n.empty:
        return {}
    n.columns = [c.lower() for c in n.columns]
    return {int(v): str(d) for v, d in zip(n.visit_id, n.detail)}


def counter(stage: int) -> dict:
    return dict(n=min(D.counter(stage), THRESHOLD), threshold=THRESHOLD)


def intakes() -> pd.DataFrame:
    with _lock:
        it = D.store.read(INTAKE)
    if not it.empty:
        it.columns = [c.lower() for c in it.columns]
    return it


@router.get("/ed/meta")
def ed_meta():
    M = MODEL.get(); meta = {k: v for k, v in M.meta.items() if k not in ("known_tokens", "ranges")}
    groups = [dict(id=str(c), label=pretty(c)) for c in M.m.classes_]
    reasons = [dict(code=int(k), label=v) for k, v in list(RFV.items())[:400]]
    return jsonable(dict(model=meta, model_version=M.version, groups=groups, reasons=reasons, chronic=[dict(id=k, label=v) for k, v in NH.CHRONIC.items()],
                         counter=counter(1), audit_rate=AUDIT_RATE))


@router.post("/ed/intake")
def ed_intake(x: Intake):
    clean, errs = NH.validate_intake(x.model_dump())
    if errs:
        raise HTTPException(422, errs)
    with _lock:
        it = intakes(); vid = INTAKE_ID0 + (int(it.visit_id.max()) - INTAKE_ID0 + 1 if len(it) else 1)
        frame = NH.frame_from_intake(clean, vid)
        a = assess(frame)
        audit = a["route"] == "PATIENT_VIEW" and is_audit(vid)
        row = frame.iloc[0].to_dict()
        row.update(visit_id=vid, ts=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), cc_text="; ".join([RFV.get(str(r), f"code {r}") for r in clean["reasons"]] + ([clean["complaint_text"]] if clean["complaint_text"] else [])),
                   chronic="; ".join(NH.CHRONIC[c] for c in clean["chronic"]), history_note=x.history_note.strip(), predicted_group=a["predicted_group"],
                   confidence=round(a["confidence"], 4), margin=round(a["margin"], 4), route=a["route"], route_reason=a["route_reason"], flags=" | ".join(a["flags"]),
                   top1_group=a["top"][0]["group"], top1_prob=a["top"][0]["prob"], top2_group=a["top"][1]["group"], top2_prob=a["top"][1]["prob"],
                   top3_group=a["top"][2]["group"], top3_prob=a["top"][2]["prob"], plausible=";".join(p["group"] for p in a["plausible_set"]),
                   audit=int(audit), model_version=f"v{a['model_version']}")
        D.store.append(INTAKE, pd.DataFrame([row]))
    return jsonable(dict(visit_id=vid, patient=patient_view_from(row), clinician=a, audit=audit))


def patient_view_from(r: dict) -> dict:
    if r.get("route") == "PATIENT_VIEW":
        return dict(kind="answer", headline=f"This could be: {pretty(r['predicted_group'])}",
                    detail="This is a first guide from your triage information, not a diagnosis. A clinician will still see you.",
                    confidence=r.get("confidence"))
    return dict(kind="doctor", headline="A doctor will review your case",
                detail="Your information has been passed to a doctor. We show no computer guess here on purpose: when the system is not sure, a person decides.")


def visit_row(vid: int) -> Optional[dict]:
    it = intakes()
    if len(it) and (it.visit_id.astype(int) == vid).any():
        r = it[it.visit_id.astype(int) == vid].iloc[-1].to_dict(); r["source"] = "live intake"; return r
    p = D.table("ST1_PREDICTIONS")
    if len(p) and (p.visit_id.astype(int) == vid).any():
        r = p[p.visit_id.astype(int) == vid].iloc[0].to_dict(); r["source"] = "earlier ED visit"; return r
    return None


def card(r: dict, labelled: dict) -> dict:
    vid = int(r["visit_id"])
    tops = [dict(group=str(r[f"top{k}_group"]), label=pretty(r[f"top{k}_group"]), prob=r.get(f"top{k}_prob")) for k in (1, 2, 3) if pd.notna(r.get(f"top{k}_group"))]
    vit = {k: r.get(k) for k in ("temp_c", "hr", "rr", "spo2", "sbp", "dbp", "pain", "acuity")}
    why = r.get("route_reason") if isinstance(r.get("route_reason"), str) else (
        "5% random re-check of a confident answer" if r.get("route") == "PATIENT_VIEW" else
        f"not confident enough (probability {float(r['confidence']):.0%}, gap {float(r['margin']):.0%})")
    flags = [f for f in str(r.get("flags") or "").split(" | ") if f and f != "nan"]
    plaus = [g for g in str(r.get("plausible") or "").split(";") if g and g != "nan"]
    return jsonable(dict(visit_id=vid, source=r.get("source", ""), ts=r.get("ts"), age=r.get("anchor_age"), sex="male" if r.get("male") == 1 else "female",
                         ambulance=bool(r.get("ambulance") == 1), complaint=r.get("chief_complaint") if pd.notna(r.get("chief_complaint", np.nan)) else r.get("cc_text"),
                         chronic=r.get("chronic") if isinstance(r.get("chronic"), str) else "", history_note=r.get("history_note") if isinstance(r.get("history_note"), str) else "",
                         vitals=vit, predicted=dict(group=str(r["predicted_group"]), label=pretty(r["predicted_group"])), confidence=r.get("confidence"),
                         margin=r.get("margin"), route=r.get("route"), why=why, flags=flags, top=tops, plausible=[dict(group=g, label=pretty(g)) for g in plaus],
                         labelled=labelled.get(vid)))


@router.get("/ed/queue")
def ed_queue(limit: int = 40):
    limit = max(1, min(limit, 200))
    fb = D.table(FB1); labelled = {int(v): l for v, l in zip(fb.visit_id, fb.doctor_label)} if len(fb) else {}
    it = intakes(); live = []
    if len(it):
        it = it[~it.visit_id.astype(int).isin(labelled)]
        it = it[(it.route == "DOCTOR_QUEUE") | (it.audit.astype(int) == 1)].iloc[::-1]
        live = [dict(r, source="live intake", chief_complaint=np.nan) for r in it.to_dict("records")]
    dq = D.doctor_queue()
    hist = dq.head(limit).to_dict("records")
    audit = D.audit_sample(AUDIT_RATE).head(10).to_dict("records")
    return jsonable(dict(live=[card(r, labelled) for r in live], waiting=[card(dict(r, source="earlier ED visit"), labelled) for r in hist],
                         audit=[card(dict(r, source="earlier ED visit"), labelled) for r in audit], counter=counter(1),
                         waiting_total=int(len(dq)), live_total=len(live)))


@router.get("/ed/visit/{vid}")
def ed_visit(vid: int):
    r = visit_row(vid)
    if r is None:
        raise HTTPException(404, f"unknown visit {vid}")
    fb = D.table(FB1); labelled = {int(v): l for v, l in zip(fb.visit_id, fb.doctor_label)} if len(fb) else {}
    return card(r, labelled)


@router.get("/ed/patient/{vid}")
def ed_patient(vid: int):
    r = visit_row(vid)
    if r is None:
        raise HTTPException(404, f"unknown visit {vid}")
    fb = D.table(FB1)
    if len(fb) and (fb.visit_id.astype(int) == vid).any():
        lab = fb[fb.visit_id.astype(int) == vid].doctor_label.iloc[-1]
        det = label_details().get(vid)
        return dict(kind="reviewed", headline=f"Reviewed by a doctor: {det or pretty(lab)}", detail="A doctor has looked at your information. Please follow the instructions of the care team.")
    return jsonable(patient_view_from(r))


@router.post("/ed/label")
def ed_label(x: Label):
    M = MODEL.get()
    if x.label not in set(map(str, M.m.classes_)):
        raise HTTPException(422, f"unknown diagnosis group {x.label}")
    r = visit_row(x.visit_id)
    if r is None:
        raise HTTPException(404, f"unknown visit {x.visit_id}")
    reason = "audit_confident" if r.get("route") == "PATIENT_VIEW" else "low_confidence"
    with _lock:
        D.store.add_doctor_label(x.visit_id, x.label, r["predicted_group"], float(r["confidence"]), reason, x.role, M.version)
        if x.detail.strip():
            D.store.append(LABEL_NOTES, pd.DataFrame([dict(visit_id=int(x.visit_id), ts=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), group_label=x.label,
                                                           detail=x.detail.strip(), staff_role=x.role)]))
    return dict(saved=True, counter=counter(1), model_was_right=str(r["predicted_group"]) == x.label)


@router.get("/ed/recent")
def ed_recent(limit: int = 10):
    it = intakes()
    if it.empty:
        return []
    return [dict(visit_id=int(r["visit_id"]), ts=r["ts"], route=r["route"], predicted=pretty(r["predicted_group"]), confidence=jsonable(r["confidence"]))
            for r in it.iloc[::-1].head(limit).to_dict("records")]


# ------------------------------------------------------------------ Stage 3 endpoints
class Verdict(BaseModel):
    stay_id: int
    risk: str = Field(max_length=20)
    verdict: str = Field(pattern=r"^(false_alarm|true_alarm)$")
    role: str = Field(default="nurse", max_length=30)
    reason_code: str = Field(default="", max_length=40)
    note: str = Field(default="", max_length=300)


def risk_items(r: dict) -> list:
    out = []
    for k in (1, 2, 3):
        txt, dom = r.get(f"risk_{k}"), r.get(f"domain_{k}")
        if isinstance(txt, str) and txt:
            tier = int(r.get(f"tier_{k}") or 0) if pd.notna(r.get(f"tier_{k}")) else 0
            body = txt.split(": ", 1)[1] if ": " in txt else txt
            out.append(dict(rank=k, domain=dom, name=DOMAIN_NAMES.get(dom, dom), tier=TIER[min(tier, 3)], evidence=body, text=txt,
                            early_warning="model expects" in txt))
    return out


_SRC = {"t": 0.0, "v": None}


def icu_source() -> dict:
    """Where the ICU board patients come from (written by stage3_multirisk.py or physionet_icu.py). Cached 30 s."""
    import time as _t
    if _SRC["v"] is not None and _t.time() - _SRC["t"] < 30:
        return _SRC["v"]
    _SRC.update(t=_t.time(), v=_icu_source())
    return _SRC["v"]


def _icu_source() -> dict:
    try:
        if SOURCE == "exasol":
            t = D.store.read("ST3_SOURCE")
            if len(t):
                t.columns = [c.lower() for c in t.columns]; r = t.iloc[0].to_dict()
                r["missing"] = [x.strip() for x in str(r.get("missing") or "").split(",") if x.strip() and x != "nan"]
                return jsonable(r)
        f = PIPE / "results" / "stage3_source.json"
        if f.exists():
            return json.loads(f.read_text())
    except Exception:
        pass
    return dict(label="Demo patients (synthetic)")


@router.get("/icu/board")
def icu_board():
    b = D.icu_board(); v = D.verdicts()
    nt = notifications() if "notifications" in globals() else pd.DataFrame()
    beds = []
    for i, r in enumerate(b.to_dict("records")):
        sid = int(r["stay_id"]); mine = v[v.stay_id.astype(int) == sid] if len(v) else v
        beds.append(dict(bed=i + 1, stay_id=sid, criticality=r.get("criticality"), level=r.get("risk_level"), trend=r.get("trend_6h"), change_6h=r.get("change_6h"),
                         hour=r.get("hour_idx"), risks=risk_items(r), vitals={k[2:]: r.get(k) for k in r if k.startswith("v_")}, verdicts=records(mine[["risk", "verdict", "reason_code"]]) if len(mine) and "reason_code" in mine else []))
        if len(nt):
            mn = nt[nt.stay_id.astype(int) == sid]
            if len(mn):
                mn = mn.sort_values("notify_id"); last = mn.iloc[-1]
                per = mn.groupby("doctor_label").tail(1)
                beds[-1]["notified"] = dict(ts=last.ts, doctor=last.doctor_label, status=last.status, urgency=last.urgency, auto=bool(int(last.auto)), n=int(len(mn)),
                                            any_open=bool((per.status == "sent").any()),
                                            doctors=[dict(doctor=r.doctor_label, status=r.status, ts=r.ts, ack_ts=r.ack_ts) for r in per.itertuples()])
    if "_auto_notify" in globals():
        _auto_notify(beds)
    return jsonable(dict(beds=beds, counter=counter(3), reasons=[dict(code=k, label=l) for k, l in FALSE_ALARM_REASONS.items()],
                         data_note=(src := icu_source()).get("label", "Demo patients"), source=src))


@router.get("/icu/stay/{sid}")
def icu_stay(sid: int):
    b = D.icu_board()
    row = b[b.stay_id.astype(int) == sid]
    if row.empty:
        raise HTTPException(404, f"unknown stay {sid}")
    r = row.iloc[0].to_dict(); t = D.timeline(sid)
    v = D.table("ST3_VITALS")
    vit = records(v[v.stay_id.astype(int) == sid].sort_values("hour_idx").drop(columns=["stay_id"])) if len(v) else []
    return jsonable(dict(stay_id=sid, criticality=r.get("criticality"), level=r.get("risk_level"), trend=r.get("trend_6h"), risks=risk_items(r), vitals=vit,
                         timeline=records(t[["hour_idx", "criticality", "crit_level", "top_domain", "top_tier"]]) if len(t) else []))


@router.get("/icu/timelines")
def icu_timelines():
    """All stays' criticality changes (hour, score, level, top risk) for the wall replay."""
    t = D.table("ST3_RISK_TIMELINE")
    if t.empty:
        return {}
    t = t.sort_values(["stay_id", "hour_idx"])
    return {str(int(s)): [[float(r.hour_idx), round(float(r.criticality), 1), TIER[min(3, int(r.crit_level))] if str(r.crit_level).isdigit() else str(r.crit_level), str(r.top_domain) if pd.notna(r.top_domain) else "", int(r.top_tier) if pd.notna(r.top_tier) else 0]
                          for r in g.itertuples()] for s, g in t.groupby("stay_id")}


@router.post("/icu/verdict")
def icu_verdict(x: Verdict):
    b = D.icu_board(); row = b[b.stay_id.astype(int) == x.stay_id]
    if row.empty:
        raise HTTPException(404, f"unknown stay {x.stay_id}")
    try:
        with _lock:
            D.give_verdict(x.stay_id, float(row.iloc[0].hour_idx), x.risk, x.verdict, x.role, x.reason_code, x.note)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return dict(saved=True, counter=counter(3))


@router.get("/icu/false-alarms")
def icu_false_alarms(limit: int = 15):
    g = D.false_alarm_log(limit)
    if g.empty:
        return []
    return records(g.assign(risk_name=g.risk.map(lambda x: DOMAIN_NAMES.get(x, x)))[["ts", "stay_id", "risk_name", "reason", "note", "staff_role"]])


# ------------------------------------------------------------------ trust / model page
def _read_json(p):
    p = PIPE / p
    return json.loads(p.read_text()) if p.exists() else None


def _read_csv(p):
    p = PIPE / p
    return records(pd.read_csv(p)) if p.exists() else None


@router.get("/trust")
def trust():
    reg = D.registry()
    groups = _read_csv("results/nhamcs_stage1_per_group.csv") or []
    for g in groups:
        g["group"] = pretty(g["group"])
    return jsonable(dict(stage1=_read_json("results/nhamcs_stage1_metrics.json"), stage1_groups=groups,
                         sepsis_benchmark=_read_csv("results/sepsis19_benchmark.csv"), stage3=_read_csv("results/stage3_multirisk_metrics.csv"),
                         registry=records(reg.tail(20)) if len(reg) else [], counters={"stage1": counter(1), "stage3": counter(3)},
                         rules=dict(tau=S1.TAU, margin=S1.MARGIN, audit_rate=AUDIT_RATE, retrain_after=THRESHOLD)))


@router.get("/clinicore/health")
def health():
    ok = {"source": SOURCE, "pipeline": str(PIPE)}
    try:
        MODEL.get(); ok["stage1_model"] = f"v{MODEL.version}"
    except HTTPException as e:
        ok["stage1_model"] = e.detail
    ok["icu_patients"] = int(len(D.icu_board()))
    return ok


# ------------------------------------------------------------------ doctor notifications (nurse -> concerned on-call doctor)
NOTIFY, NOTIFY_ACK = "ST3_NOTIFY", "ST3_NOTIFY_ACK"
DOCTORS = [
    dict(id="icu_attending", label="ICU attending on call", covers=["sepsis", "arf", "aki", "shock"]),
    dict(id="infectious", label="Infectious diseases on call", covers=["sepsis"]),
    dict(id="pulmonology", label="Pulmonology on call", covers=["arf"]),
    dict(id="nephrology", label="Nephrology on call", covers=["aki"]),
    dict(id="cardiology", label="Cardiology on call", covers=["shock"]),
]
DOC = {d["id"]: d for d in DOCTORS}
URGENCY = {"emergency": "Come now (emergency)", "urgent": "Review within 30 min", "routine": "Review at next round"}
AUTO_FILE = Path(STATE) / "auto_notify.json" if SOURCE == "csv" else PIPE / "app_state" / "auto_notify.json"


def suggested_doctors(risks: list) -> list:
    """ICU attending always; plus the specialist for the top risk."""
    out = ["icu_attending"]
    if risks:
        spec = next((d["id"] for d in DOCTORS[1:] if risks[0]["domain"] in d["covers"]), None)
        if spec:
            out.append(spec)
    return out


def sbar(bed: dict, risks: list, urgency: str, note: str) -> str:
    """Deterministic SBAR message: same input -> same words; only measured values, no model probabilities."""
    s = f"S: Bed {bed['bed']} (patient #{bed['stay_id']}) is {bed['level']} {num0(bed['criticality'])}/100, {bed.get('trend') or 'steady'}."
    b = f"B: Hour {num0(bed.get('hour'))} in ICU" + (f"; score changed {bed['change_6h']:+.0f} in the last 6 h." if isinstance(bed.get("change_6h"), (int, float)) and not pd.isna(bed.get("change_6h")) else ".")
    a = ("A:\n" + "\n".join(f"- {r['name']} {r['tier'].upper()}: {r['evidence']}" for r in risks)) if risks else "A: no active risk listed."
    r = f"R: {URGENCY.get(urgency, urgency)}." + (f" Nurse note: {note.strip()}" if note.strip() else "")
    return "\n".join([s, b, a, r])


def num0(v):
    try:
        return f"{float(v):.0f}"
    except (TypeError, ValueError):
        return "?"


def notifications() -> pd.DataFrame:
    with _lock:
        n, a = D.store.read(NOTIFY), D.store.read(NOTIFY_ACK)
    if n.empty:
        return n
    n.columns = [c.lower() for c in n.columns]
    n["status"], n["ack_by"], n["ack_ts"] = "sent", "", ""
    if len(a):
        a.columns = [c.lower() for c in a.columns]
        last = a.sort_values("ack_id").groupby("notify_id").tail(1).set_index("notify_id")
        m = n.notify_id.isin(last.index)
        n.loc[m, "status"] = n.loc[m, "notify_id"].map(last.ack_action if "ack_action" in last else last.action)
        n.loc[m, "ack_by"] = n.loc[m, "notify_id"].map(last.doctor)
        n.loc[m, "ack_ts"] = n.loc[m, "notify_id"].map(last.ts)
    return n


def _send(bed: dict, doctor: str, urgency: str, risk_domains: list, note: str, sent_by: str, auto: bool) -> dict:
    risks = [r for r in bed["risks"] if r["domain"] in risk_domains] or bed["risks"]
    with _lock:
        n = D.store.read(NOTIFY); nid = int(n[[c for c in n.columns if c.lower() == "notify_id"][0]].max()) + 1 if len(n) else 1
        row = dict(notify_id=nid, ts=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), stay_id=int(bed["stay_id"]), bed=int(bed["bed"]),
                   risk_level=bed["level"], criticality=float(bed["criticality"] or 0), doctor=doctor, doctor_label=DOC[doctor]["label"], urgency=urgency,
                   risks=";".join(r["domain"] for r in risks), message=sbar(bed, risks, urgency, note), sent_by=sent_by, auto=int(auto))
        D.store.append(NOTIFY, pd.DataFrame([row]))
    return row


def _auto_settings() -> dict:
    try:
        return json.loads(AUTO_FILE.read_text())
    except Exception:
        return dict(enabled=False, urgency="emergency")


def _auto_notify(beds: list):
    """When switched on: every Critical patient whose score is rising or new and who has no open notification gets an
    automatic pop-up to the ICU attending. Never fires twice for the same patient while a notification is open."""
    s = _auto_settings()
    if not s.get("enabled"):
        return
    n = notifications()
    open_ids = set(n[n.status.isin(["sent"])].stay_id.astype(int)) if len(n) else set()
    for b in beds:
        if b["level"] == "Critical" and b.get("trend") in ("rising", "new") and int(b["stay_id"]) not in open_ids:
            _send(b, "icu_attending", s.get("urgency", "emergency"), [r["domain"] for r in b["risks"]], "", "CliniCore auto-notify", True)


class Notify(BaseModel):
    stay_id: int
    doctors: List[str] = Field(min_length=1, max_length=5)
    urgency: str = Field(pattern=r"^(emergency|urgent|routine)$")
    risks: List[str] = Field(default_factory=list, max_length=4)
    note: str = Field(default="", max_length=300)
    role: str = Field(default="nurse", max_length=30)


class Ack(BaseModel):
    notify_id: int
    doctor: str = Field(max_length=30)
    action: str = Field(pattern=r"^(acknowledged|on_my_way|declined)$")
    note: str = Field(default="", max_length=300)


class AutoNotify(BaseModel):
    enabled: bool
    urgency: str = Field(default="emergency", pattern=r"^(emergency|urgent|routine)$")


def _bed(sid: int) -> dict:
    """One bed, without rebuilding the whole board (no feedback/notification reads)."""
    b = D.icu_board()
    pos = b.index[b.stay_id.astype(int) == int(sid)]
    if not len(pos):
        raise HTTPException(404, f"unknown stay {sid}")
    r = b.loc[pos[0]].to_dict()
    return jsonable(dict(bed=int(pos[0]) + 1, stay_id=int(sid), criticality=r.get("criticality"), level=r.get("risk_level"), trend=r.get("trend_6h"),
                         change_6h=r.get("change_6h"), hour=r.get("hour_idx"), risks=risk_items(r)))


@router.get("/notify/meta")
def notify_meta():
    return dict(doctors=DOCTORS, urgency=[dict(id=k, label=v) for k, v in URGENCY.items()], auto=_auto_settings())


@router.get("/notify/preview")
def notify_preview(stay_id: int, urgency: str = "urgent", risks: str = "", note: str = ""):
    b = _bed(stay_id); doms = [x for x in risks.split(",") if x]
    sel = [r for r in b["risks"] if r["domain"] in doms] or b["risks"]
    return dict(message=sbar(b, sel, urgency, note), suggested=suggested_doctors(b["risks"]))


@router.post("/notify")
def notify_send(x: Notify):
    bad = [d for d in x.doctors if d not in DOC]
    if bad:
        raise HTTPException(422, f"unknown doctor {bad}")
    b = _bed(x.stay_id)
    sent = [_send(b, d, x.urgency, x.risks, x.note, x.role, False) for d in dict.fromkeys(x.doctors)]
    return jsonable(dict(sent=len(sent), to=[s["doctor_label"] for s in sent], message=sent[0]["message"]))


@router.get("/notify")
def notify_list(doctor: str = "", stay_id: int = 0, limit: int = 50):
    n = notifications()
    if n.empty:
        return []
    if doctor:
        n = n[n.doctor == doctor]
    if stay_id:
        n = n[n.stay_id.astype(int) == stay_id]
    return records(n.sort_values("notify_id", ascending=False).head(max(1, min(limit, 200))))


@router.post("/notify/ack")
def notify_ack(x: Ack):
    n = notifications()
    if n.empty or not (n.notify_id.astype(int) == x.notify_id).any():
        raise HTTPException(404, f"unknown notification {x.notify_id}")
    with _lock:
        a = D.store.read(NOTIFY_ACK); aid = int(a[[c for c in a.columns if c.lower() == "ack_id"][0]].max()) + 1 if len(a) else 1
        D.store.append(NOTIFY_ACK, pd.DataFrame([dict(ack_id=aid, notify_id=x.notify_id, ts=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                                                      doctor=x.doctor, ack_action=x.action, note=x.note.strip())]))
    return dict(saved=True)


@router.get("/notify/auto")
def auto_get():
    return _auto_settings()


@router.post("/notify/auto")
def auto_set(x: AutoNotify):
    AUTO_FILE.parent.mkdir(parents=True, exist_ok=True); AUTO_FILE.write_text(json.dumps(x.model_dump()))
    sent_before = len(notifications())
    _auto_notify(icu_board()["beds"])
    return dict(**x.model_dump(), sent_now=len(notifications()) - sent_before)
