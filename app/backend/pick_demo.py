"""Find presentation-mode demo patients in ANALYTICS.PREDICTIONS (read-only, MCP_READONLY).

CRITICAL shock candidates (eICU, so 5-minute monitor data exists in RAW_EICU.VITALPERIODIC):
    a shock alert that fires as CRITICAL right after a NORMAL hour, with an alert text naming 2-3
    concrete drivers, and in the 6 h before the alert: arterial MAP (SYSTEMICMEAN) drifting down and
    HR drifting up (least-squares slope on the 5-minute values), dense MAP data, no artifact values.
MEDIUM candidates: a respiratory or renal alert, MEDIUM right after NORMAL, 2-3 concrete drivers.

Prints ranked candidates; the chosen ones are written to ../config/demo_patients.json by --save
(anonymized token only: first 12 hex characters of HASH_MD5('carepriority-' || STAY_ID)).

Usage: uv run python pick_demo.py [--save]
"""
import json
import re
import sys
from pathlib import Path

import db

CONFIG = Path(__file__).resolve().parents[1] / "config" / "demo_patients.json"
TOKEN = "SUBSTR(HASH_MD5('carepriority-' || {col}), 1, 12)"


def drivers(text):
    part = text.split(":", 1)[1] if ":" in text else ""
    return [d.strip() for d in part.split(";")[0].split(", ") if d.strip()]


ALERTS = f"""
WITH p AS (
    SELECT STAY_ID, HOUR_INDEX, TIER, ALERT_FIRED, TRIGGER_COMPONENTS, ALERT_TEXT, P_SHOCK, P_RESP, P_RENAL,
           LAG(TIER) OVER (PARTITION BY STAY_ID ORDER BY HOUR_INDEX) AS PREV_TIER,
           LAG(HOUR_INDEX) OVER (PARTITION BY STAY_ID ORDER BY HOUR_INDEX) AS PREV_HOUR
    FROM ANALYTICS.PREDICTIONS WHERE SOURCE_SYSTEM = 'EICU'
)
SELECT {TOKEN.format(col='p.STAY_ID')} AS TOKEN, CAST(SUBSTR(p.STAY_ID, 6) AS DECIMAL(18,0)) AS NATIVE_ID,
       p.HOUR_INDEX, p.TIER, p.TRIGGER_COMPONENTS, p.ALERT_TEXT, p.P_SHOCK, p.P_RESP, p.P_RENAL
FROM p
WHERE p.ALERT_FIRED = 1 AND p.PREV_TIER = 'NORMAL' AND p.PREV_HOUR = p.HOUR_INDEX - 1
  AND p.ALERT_TEXT NOT LIKE '%pattern-based%'
  AND ((p.TIER = 'CRITICAL' AND p.TRIGGER_COMPONENTS = 'shock')
       OR (p.TIER = 'MEDIUM' AND p.TRIGGER_COMPONENTS IN ('resp', 'renal')))
"""

VITAL_STATS = """
SELECT COUNT(SYSTEMICMEAN) AS N_MAP, COUNT(HEARTRATE) AS N_HR, COUNT(RESPIRATION) AS N_RR, COUNT(SAO2) AS N_SPO2,
       60 * REGR_SLOPE(SYSTEMICMEAN, OBSERVATIONOFFSET) AS MAP_SLOPE_H, 60 * REGR_SLOPE(HEARTRATE, OBSERVATIONOFFSET) AS HR_SLOPE_H,
       60 * REGR_SLOPE(RESPIRATION, OBSERVATIONOFFSET) AS RR_SLOPE_H, 60 * REGR_SLOPE(SAO2, OBSERVATIONOFFSET) AS SPO2_SLOPE_H,
       AVG(CASE WHEN OBSERVATIONOFFSET < {lo} + 60 THEN SYSTEMICMEAN END) AS MAP_START,
       AVG(CASE WHEN OBSERVATIONOFFSET >= {hi} - 60 THEN SYSTEMICMEAN END) AS MAP_END,
       AVG(CASE WHEN OBSERVATIONOFFSET < {lo} + 60 THEN HEARTRATE END) AS HR_START,
       AVG(CASE WHEN OBSERVATIONOFFSET >= {hi} - 60 THEN HEARTRATE END) AS HR_END,
       SUM(CASE WHEN SYSTEMICMEAN < 20 OR SYSTEMICMEAN > 250 OR HEARTRATE <= 0 OR HEARTRATE > 300
                  OR RESPIRATION > 60 OR SAO2 < 50 OR SAO2 > 100 THEN 1 ELSE 0 END) AS N_ARTIFACT,
       MAX(ABS(SYSTEMICMEAN - PREV_MAP)) AS MAX_MAP_JUMP, MAX(ABS(HEARTRATE - PREV_HR)) AS MAX_HR_JUMP
FROM (SELECT v.*, LAG(SYSTEMICMEAN) OVER (ORDER BY OBSERVATIONOFFSET) AS PREV_MAP,
             LAG(HEARTRATE) OVER (ORDER BY OBSERVATIONOFFSET) AS PREV_HR
      FROM RAW_EICU.VITALPERIODIC v
      WHERE PATIENTUNITSTAYID = {nid} AND OBSERVATIONOFFSET BETWEEN {lo} AND {hi})
"""


def main():
    alerts = db.query(ALERTS)
    crit, med = [], []
    for a in alerts:
        d = drivers(a["ALERT_TEXT"])
        if not 2 <= len(d) <= 3:
            continue
        h = int(a["HOUR_INDEX"])
        lo, hi = h * 60 - 360, h * 60 + 59
        s = db.query(VITAL_STATS.format(nid=int(a["NATIVE_ID"]), lo=lo, hi=hi))[0]
        rec = {"token": a["TOKEN"], "alert_hour": h, "tier": a["TIER"], "components": a["TRIGGER_COMPONENTS"],
               "alert_text": a["ALERT_TEXT"], "drivers": d,
               **{k.lower(): (None if v is None else round(float(v), 2)) for k, v in s.items()}}
        if a["TIER"] == "CRITICAL":
            ok = (rec["n_map"] or 0) >= 60 and (rec["n_artifact"] or 0) == 0 and (rec["map_slope_h"] or 0) < 0 \
                 and (rec["hr_slope_h"] or 0) > 0 and (rec["max_map_jump"] or 99) <= 25 and (rec["max_hr_jump"] or 99) <= 30 \
                 and any(re.match(r"(MAP|SBP|HR|shock index|lactate)", x) for x in d)
            if ok:
                rec["score"] = round(-rec["map_slope_h"] * 6 + rec["hr_slope_h"] * 6, 1)
                crit.append(rec)
        else:
            ok = (rec["n_hr"] or 0) >= 60 and (rec["n_artifact"] or 0) == 0
            if ok:
                med.append(rec)
    crit.sort(key=lambda r: -r["score"])
    print(f"{len(alerts)} candidate alerts; {len(crit)} CRITICAL shock pass the story filter, {len(med)} MEDIUM\n")
    print("== CRITICAL shock (6 h before alert: MAP change, HR change from the 5-minute trend)")
    for r in crit[:8]:
        print(f"  token {r['token']} h{r['alert_hour']:>3} MAP {r['map_start']}->{r['map_end']} ({6 * r['map_slope_h']:+.1f}/6h)"
              f"  HR {r['hr_start']}->{r['hr_end']} ({6 * r['hr_slope_h']:+.1f}/6h)  n_map={r['n_map']} jumps map/hr="
              f"{r['max_map_jump']}/{r['max_hr_jump']}\n      {r['alert_text']}")
    print("\n== MEDIUM respiratory / renal")
    for r in med[:12]:
        print(f"  token {r['token']} h{r['alert_hour']:>3} {r['components']:5} RR {6 * (r['rr_slope_h'] or 0):+.1f}/6h "
              f"SpO2 {6 * (r['spo2_slope_h'] or 0):+.1f}/6h\n      {r['alert_text']}")
    if "--save" in sys.argv:
        pick = json.loads(sys.argv[sys.argv.index("--save") + 1])
        CONFIG.write_text(json.dumps(pick, indent=2, ensure_ascii=False) + "\n")
        print(f"\nsaved {CONFIG}")


if __name__ == "__main__":
    main()
