"""Live data access (read-only, MCP_READONLY). Every payload is anonymized: patients are identified only by
a token (first 12 hex of HASH_MD5('carepriority-' || STAY_ID)), a display ID ("P-" + first 4 hex) and a bed
number. Stay IDs are used inside SQL only and never returned.

Sources:
    ANALYTICS.PREDICTIONS             tiers, calibrated risks, alerts, alert texts (eligible stay-hours)
    ANALYTICS.PATIENT_HOURLY_FEATURES age, sex, unit type, hourly vitals and labs for the same stay-hours
    RAW_EICU.VITALPERIODIC            eICU 5-minute monitor values (HR, arterial MAP, RR, SpO2)
    RAW_EICU.VITALAPERIODIC           eICU non-invasive cuff MAP (used when there is no arterial line)
    RAW_MIMIC3/4.CHARTEVENTS, ICUSTAYS MIMIC charted values at native resolution, aligned to ICU admission
Raw values outside the CORE plausibility bounds are dropped as artifacts (counted per patient).
"""
import json
from pathlib import Path

import db

HERE = Path(__file__).resolve().parent
ML = HERE.parents[1] / "ml"
TOKEN_SQL = "SUBSTR(HASH_MD5('carepriority-' || {col}), 1, 12)"
ANON_SQL = "'P-' || UPPER(SUBSTR(HASH_MD5('carepriority-' || {col}), 1, 4))"
PLAUSIBLE = {"hr": (0, 300), "map": (20, 250), "rr": (1, 60), "spo2": (50, 100)}  # CORE range filters
MIMIC_ITEMS = {  # same itemids as sql/phase2_core.sql
    "hr": (211, 220045), "map": (52, 456, 220052, 220181, 225312),
    "rr": (615, 618, 220210, 224690), "spo2": (646, 220277),
}
HOURLY_COLS = """p.HOUR_INDEX, p.TIER, p.ALERT_FIRED, p.ALERT_LEVEL, p.TRIGGER_COMPONENTS, p.ALERT_TEXT, p.MODIFIED_NEWS2,
       p.P_SHOCK, p.P_RESP, p.P_RENAL, f.AGE, f.GENDER, f.UNIT_TYPE,
       f.HEART_RATE, f.MAP_MMHG, f.RESP_RATE, f.SPO2, f.TEMP_C, f.GCS,
       f.LACTATE, f.CREATININE, f.BUN, f.POTASSIUM, f.SODIUM, f.BICARBONATE, f.MAGNESIUM, f.GLUCOSE, f.WBC,
       f.HEMOGLOBIN, f.PLATELETS, f.PH_ARTERIAL, f.PAO2, f.PACO2, f.TROPONIN, f.TROPONIN_ORDERED_FLAG, f.BNP,
       f.BILIRUBIN_TOTAL, f.ALBUMIN"""
WARD_ELIG = """
    SELECT STAY_ID, MIN(SOURCE_SYSTEM) AS SOURCE_SYSTEM FROM ANALYTICS.PREDICTIONS
    WHERE HOUR_INDEX <= 72 AND (({ward!s} = 'A' AND SOURCE_SYSTEM = 'EICU') OR ({ward!s} = 'B' AND SOURCE_SYSTEM <> 'EICU'))
    GROUP BY STAY_ID HAVING SUM(ALERT_FIRED) >= 1 AND COUNT(*) >= 36"""


def _num(v):
    return None if v is None else (int(v) if float(v).is_integer() and not isinstance(v, float) else float(v))


def _clean(row):
    return {k.lower(): (_num(v) if not isinstance(v, str) else v) for k, v in row.items()}


def meta():
    thr = json.loads((ML / "models/final/alert_thresholds.json").read_text())
    setting_a = next(v for k, v in thr["settings"].items() if k.startswith("A"))
    ranges = {k: v for k, v in json.loads((ML / "clinical_ranges.json").read_text()).items() if not k.startswith("_")}
    return {"thresholds": {"med": setting_a["thr_med"], "crit": thr["thr_crit"]}, "ranges": ranges,
            "horizon_hours": 6, "model_version": "carepriority-2026-11-13-setA-v2text"}


def _resolve(token):
    rows = db.query(f"""SELECT STAY_ID, MIN(SOURCE_SYSTEM) AS SOURCE_SYSTEM FROM ANALYTICS.PREDICTIONS
                        WHERE {TOKEN_SQL.format(col='STAY_ID')} = {{t}} GROUP BY STAY_ID""", {"t": token})
    if len(rows) != 1:
        raise KeyError(f"unknown patient token {token}")
    return rows[0]["STAY_ID"], rows[0]["SOURCE_SYSTEM"]


def _raw_vitals(stay_id, source, upto_hour):
    native = int(stay_id.split("-", 1)[1])
    hi = (upto_hour + 1) * 60
    points = []
    if source == "EICU":
        rows = db.query(f"""SELECT OBSERVATIONOFFSET AS T, HEARTRATE AS HR, SYSTEMICMEAN AS MBP, RESPIRATION AS RR, SAO2 AS SPO2
                            FROM RAW_EICU.VITALPERIODIC WHERE PATIENTUNITSTAYID = {native}
                              AND OBSERVATIONOFFSET BETWEEN -60 AND {hi} ORDER BY OBSERVATIONOFFSET""")
        art = {int(r["T"]) for r in rows if r["MBP"] is not None}
        points = [{"t": int(r["T"]), "hr": r["HR"], "map": r["MBP"], "rr": r["RR"], "spo2": r["SPO2"],
                   "bp_source": "ART" if r["MBP"] is not None else None} for r in rows]
        nibp = db.query(f"""SELECT OBSERVATIONOFFSET AS T, NONINVASIVEMEAN AS MBP FROM RAW_EICU.VITALAPERIODIC
                            WHERE PATIENTUNITSTAYID = {native} AND NONINVASIVEMEAN IS NOT NULL
                              AND OBSERVATIONOFFSET BETWEEN -60 AND {hi} ORDER BY OBSERVATIONOFFSET""")
        if len(art) < 3:  # no arterial line: blood pressure from the non-invasive cuff
            points += [{"t": int(r["T"]), "hr": None, "map": r["MBP"], "rr": None, "spo2": None, "bp_source": "NIBP"}
                       for r in nibp]
        resolution = "5-minute monitor values (eICU VITALPERIODIC)" + ("" if len(art) >= 3 else "; MAP from NIBP cuff")
    else:
        schema, sid_col = ("RAW_MIMIC3", "ICUSTAY_ID") if source == "MIMIC3" else ("RAW_MIMIC4", "STAY_ID")
        items = ", ".join(str(i) for ids in MIMIC_ITEMS.values() for i in ids)
        err = "AND COALESCE(c.ERROR, 0) = 0" if source == "MIMIC3" else ""
        rows = db.query(f"""SELECT ROUND(MINUTES_BETWEEN(c.CHARTTIME, s.INTIME)) AS T, c.ITEMID, c.VALUENUM AS V
                            FROM {schema}.CHARTEVENTS c JOIN {schema}.ICUSTAYS s ON s.{sid_col} = c.{sid_col}
                            WHERE c.{sid_col} = {native} AND c.ITEMID IN ({items}) AND c.VALUENUM IS NOT NULL {err}
                              AND MINUTES_BETWEEN(c.CHARTTIME, s.INTIME) BETWEEN -60 AND {hi}
                            ORDER BY T""")
        var_of = {i: v for v, ids in MIMIC_ITEMS.items() for i in ids}
        by_t = {}
        for r in rows:
            p = by_t.setdefault(int(r["T"]), {"t": int(r["T"]), "hr": None, "map": None, "rr": None, "spo2": None,
                                              "bp_source": None})
            p[var_of[int(r["ITEMID"])]] = r["V"]
            if var_of[int(r["ITEMID"])] == "map":
                p["bp_source"] = "ART" if int(r["ITEMID"]) in (52, 220052, 225312) else "NIBP"
        points = [by_t[t] for t in sorted(by_t)]
        resolution = "charted values at native resolution (MIMIC CHARTEVENTS)"
    dropped = []  # minute of each value dropped as implausible (so the UI can count per displayed window)
    for p in points:
        for k, (lo, hi_) in PLAUSIBLE.items():
            v = p.get(k)
            if v is not None:
                v = float(v)
                if not (lo < v <= hi_) if k == "hr" else not (lo <= v <= hi_):
                    p[k] = None
                    dropped.append(p["t"])
                else:
                    p[k] = v
    points.sort(key=lambda p: p["t"])
    return {"points": points, "resolution": resolution, "artifacts_dropped": len(dropped), "artifact_times": dropped}


def patient(token, upto_hour=72):
    stay_id, source = _resolve(token)
    hourly = db.query(f"""SELECT {HOURLY_COLS}, {ANON_SQL.format(col='p.STAY_ID')} AS ANON_ID
                          FROM ANALYTICS.PREDICTIONS p
                          LEFT JOIN ANALYTICS.PATIENT_HOURLY_FEATURES f ON f.STAY_ID = p.STAY_ID AND f.HOUR_INDEX = p.HOUR_INDEX
                          WHERE p.STAY_ID = {{s}} AND p.HOUR_INDEX BETWEEN 0 AND {int(upto_hour)}
                          ORDER BY p.HOUR_INDEX""", {"s": stay_id})
    if not hourly:
        raise KeyError(f"no scored hours for {token}")
    first = hourly[0]
    return {"token": token, "anon_id": first["ANON_ID"], "source": source,
            "age": _num(first["AGE"]), "sex": first["GENDER"], "unit_type": first["UNIT_TYPE"],
            "admission_reason": None,  # not part of the model tables (only in raw eICU / MIMIC admissions)
            "hours": [{k: v for k, v in _clean(r).items() if k not in ("anon_id", "age", "gender", "unit_type")}
                      for r in hourly],
            "vitals": _raw_vitals(stay_id, source, int(upto_hour)), "upto_hour": int(upto_hour)}


def ward(ward_id):
    rows = db.query(f"""
        WITH elig AS ({WARD_ELIG}),
        beds AS (SELECT STAY_ID, SOURCE_SYSTEM, ROW_NUMBER() OVER (ORDER BY HASH_MD5('ward-' || STAY_ID)) AS BED_NO FROM elig)
        SELECT b.BED_NO, b.SOURCE_SYSTEM, {TOKEN_SQL.format(col='b.STAY_ID')} AS TOKEN, {ANON_SQL.format(col='b.STAY_ID')} AS ANON_ID,
               p.HOUR_INDEX, p.TIER, p.ALERT_FIRED, p.ALERT_LEVEL, p.TRIGGER_COMPONENTS, p.ALERT_TEXT,
               p.P_SHOCK, p.P_RESP, p.P_RENAL, p.MODIFIED_NEWS2, f.HEART_RATE, f.MAP_MMHG, f.RESP_RATE, f.SPO2
        FROM beds b
        JOIN ANALYTICS.PREDICTIONS p ON p.STAY_ID = b.STAY_ID AND p.HOUR_INDEX BETWEEN 0 AND 72
        LEFT JOIN ANALYTICS.PATIENT_HOURLY_FEATURES f ON f.STAY_ID = p.STAY_ID AND f.HOUR_INDEX = p.HOUR_INDEX
        WHERE b.BED_NO <= 12
        ORDER BY b.BED_NO, p.HOUR_INDEX""".replace("{ward!s}", "{ward}"), {"ward": ward_id})
    beds = {}
    for r in rows:
        b = beds.setdefault(int(r["BED_NO"]), {"bed": int(r["BED_NO"]), "token": r["TOKEN"], "anon_id": r["ANON_ID"],
                                               "source": r["SOURCE_SYSTEM"], "hours": {}})
        b["hours"][int(r["HOUR_INDEX"])] = {k: v for k, v in _clean(r).items()
                                            if k not in ("bed_no", "token", "anon_id", "source_system", "hour_index")}
    return {"ward": ward_id, "name": {"A": "Ward A · eICU", "B": "Ward B · MIMIC-III / IV"}[ward_id],
            "max_hour": 72, "selection": "12 stays with at least one alert and >= 36 scored hours in ICU hours 0-72",
            "beds": [beds[k] for k in sorted(beds)]}


def presentation(config):
    beds = []
    for c in config["calm_beds"]:
        beds.append({"bed": c["bed"], "token": c["token"], "role": "calm", "display_hour": c["display_hour"]})
    for role in ("critical", "medium"):
        c = config[role]
        beds.append({"bed": c["bed"], "token": c["token"], "role": role, "display_hour": c["alert_hour"] - 1,
                     "alert_hour": c["alert_hour"]})
    beds.sort(key=lambda b: b["bed"])
    patients = {}
    for b in beds:
        upto = b.get("alert_hour", b["display_hour"])
        p = patient(b["token"], upto)
        patients[b["token"]] = p
        b["anon_id"], b["source"] = p["anon_id"], p["source"]
    return {"ward_name": config.get("ward_name", "ICU (demo)"), "trigger_delay_seconds": config.get("trigger_delay_seconds", 2),
            "beds": beds, "patients": patients,
            "critical": {"token": config["critical"]["token"], "alert_hour": config["critical"]["alert_hour"]},
            "medium": {"token": config["medium"]["token"], "alert_hour": config["medium"]["alert_hour"]}}
