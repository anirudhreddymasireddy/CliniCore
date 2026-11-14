"""Data layer for the CareGuard app. Reads model outputs + staff feedback from CSV files or Exasol; writes only through retrain.Store.
No Dash import here, so it is unit-testable."""
from __future__ import annotations
import os, time
import numpy as np, pandas as pd
from retrain import Store, FB1, FB3, REG, THRESHOLD, FALSE_ALARM_REASONS

GROUP_LABELS = {}          # optional prettier names, filled by the app if wanted
DOMAIN_NAMES = {"sepsis": "Sepsis", "arf": "Acute respiratory failure", "aki": "Acute kidney injury", "shock": "Circulatory shock"}
CSV_FILES = {"ST1_PREDICTIONS": "stage1_predictions.csv", "ST1_ROUNDS": "stage1_rounds.csv", "ST3_RISKS": "results/stage3_risks_snapshot.csv",
             "ST3_RISK_TIMELINE": "results/stage3_risk_timeline.csv", "ST3_RISK_METRICS": "results/stage3_multirisk_metrics.csv", "ST3_VITALS": "results/stage3_vitals.csv"}


class Data:
    def __init__(self, kind="csv", base=".", state="app_state", schema="CAREGUARD_RESULTS", con=None):
        self.kind, self.base, self.schema, self.con = kind, base, schema, con
        self.store = Store(kind, state if kind == "csv" else None, schema, con)

    _cache: dict = {}
    # result tables only change when a pipeline reruns: Stage 1 predictions rarely, the ICU board at most every few minutes
    CACHE_S = {"ST1_PREDICTIONS": 600, "ST3_RISK_METRICS": 600}
    CACHE_DEFAULT = 60

    def table(self, name):
        if name in (FB1, FB3, REG):
            return self._table(name)                 # staff tables: always fresh
        if self.kind == "csv":                       # re-read a result file only when it changed on disk
            f = os.path.join(self.base, CSV_FILES[name])
            key = os.path.getmtime(f) if os.path.exists(f) else None
            hit = self._cache.get(name)
            if hit and hit[0] == key:
                return hit[1]
            df = self._table(name); self._cache[name] = (key, df)
            return df
        hit = self._cache.get(name)
        if hit and time.time() - hit[0] < self.CACHE_S.get(name, self.CACHE_DEFAULT):
            return hit[1]
        df = self._table(name); self._cache[name] = (time.time(), df)
        return df

    def _table(self, name):
        if self.kind == "csv" and name not in (FB1, FB3, REG):
            f = os.path.join(self.base, CSV_FILES[name])
            df = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()
        else:
            df = self.store.read(name)
        df.columns = [c.lower() for c in df.columns]          # Exasol and CSV spell some columns differently; use lowercase everywhere
        return df

    # ---------------- Stage 1
    def labelled_visits(self):
        fb = self.table(FB1)
        return set(fb.visit_id.astype(int)) if len(fb) else set()

    def doctor_queue(self):
        p = self.table("ST1_PREDICTIONS")
        if p.empty:
            return p
        p = p[~p.visit_id.astype(int).isin(self.labelled_visits())]
        return p[p.route == "DOCTOR_QUEUE"].sort_values("confidence").reset_index(drop=True)

    def audit_sample(self, rate=0.05):
        """Deterministic 5% of confidently-answered visits are sent to a doctor anyway (checks the model is not silently wrong)."""
        p = self.table("ST1_PREDICTIONS")
        if p.empty:
            return p
        c = p[(p.route == "PATIENT_VIEW") & ~p.visit_id.astype(int).isin(self.labelled_visits())]
        h = (c.visit_id.astype(np.int64) * 2654435761 % 10007) / 10007.0
        return c[h < rate].reset_index(drop=True)

    def patient_view(self, visit_id):
        p = self.table("ST1_PREDICTIONS")
        r = p[p.visit_id.astype(int) == int(visit_id)]
        if r.empty:
            return None
        r = r.iloc[0]
        if r.route == "PATIENT_VIEW":
            return dict(kind="answer", group=r.predicted_group, confidence=float(r.confidence), row=r)
        return dict(kind="doctor", row=r)             # option A: no model guess shown to the patient

    def label_visit(self, visit_id, doctor_label, staff_role="doctor"):
        p = self.table("ST1_PREDICTIONS"); r = p[p.visit_id.astype(int) == int(visit_id)].iloc[0]
        reason = "audit_confident" if r.route == "PATIENT_VIEW" else "low_confidence"
        self.store.add_doctor_label(int(visit_id), doctor_label, r.predicted_group, float(r.confidence), reason, staff_role, str(r.model_version).lstrip("v") or 0)

    # ---------------- Stage 3
    def icu_board(self):
        r = self.table("ST3_RISKS")
        if r.empty:
            return r
        return r.sort_values("criticality", ascending=False).reset_index(drop=True)

    def verdicts(self):
        fb = self.table(FB3)
        return fb if len(fb) else pd.DataFrame(columns=["stay_id", "hour_idx", "risk", "verdict"])

    def timeline(self, stay_id):
        t = self.table("ST3_RISK_TIMELINE")
        return t[t.stay_id.astype(int) == int(stay_id)].sort_values("hour_idx") if len(t) else t

    def give_verdict(self, stay_id, hour_idx, risk, verdict, staff_role="nurse", reason_code="", note=""):
        self.store.add_alert_verdict(int(stay_id), float(hour_idx), risk, verdict, staff_role, 0, reason_code, note)

    def false_alarm_log(self, n=10):
        fb = self.table(FB3)
        if fb.empty:
            return fb
        fb = fb[fb.verdict == "false_alarm"].tail(n).iloc[::-1]
        fb["reason"] = fb.reason_code.map(FALSE_ALARM_REASONS) if "reason_code" in fb else ""
        return fb

    # ---------------- counters / registry
    def counter(self, stage):
        fb = self.table(FB1 if stage == 1 else FB3)
        reg = self.table(REG)
        last = 0
        if len(reg):
            g = reg[reg.stage == stage]
            if len(g):
                last = int(g.consumed_through_id.max())
        if fb.empty:
            return 0
        new = fb[fb.feedback_id > last]
        return len(new) if stage == 1 else int((new.verdict == "false_alarm").sum())

    def registry(self):
        return self.table(REG)
