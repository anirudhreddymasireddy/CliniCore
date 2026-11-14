#!/usr/bin/env python3
"""
CareGuard automatic retraining.

  Stage 1 (diagnosis group): retrain when 50 new doctor labels have arrived (low-confidence cases + the 5% audit sample).
  Stage 3 (ICU early warning): retrain when 50 new FALSE ALARMS have been marked by a nurse or doctor.
Both counters are independent. Staff only click; nothing else is needed. A retrained model goes live ONLY if it passes the
safety gate on a frozen holdout; otherwise the old model stays and the attempt is logged.

How the app talks to this (append-only tables, so nothing is ever edited):
  ST1_FEEDBACK  feedback_id, ts, visit_id, model_label, model_confidence, doctor_label, reason, staff_role, model_version
  ST3_FEEDBACK  feedback_id, ts, stay_id, hour_idx, risk, verdict ('false_alarm'|'true_alarm'), reason_code, note, staff_role, model_version
                (a false alarm MUST carry a reason_code from FALSE_ALARM_REASONS; 'other' also needs a note)
  MODEL_REGISTRY  ts, stage, version_after, trigger, consumed_through_id, n_new, promoted, detail
After saving a label, the app calls Service.check(); or run `python retrain.py watch` and it checks every few seconds.

  python retrain.py demo  --dir data/samples/synth_default --live1-dir data/samples/synth_surge --live3-dir data/samples/synth_hospital_c
  python retrain.py watch --source exasol --schema CAREGUARD_A --live1-schema CAREGUARD_SURGE --live3-schema CAREGUARD_C
"""
from __future__ import annotations
import argparse, glob, json, os, shutil, sys, time, datetime as dt, warnings
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage1_pipeline as S1
from stage1_pipeline import Source, write_exasol

THRESHOLD = 50
FB1, FB3, REG = "ST1_FEEDBACK", "ST3_FEEDBACK", "MODEL_REGISTRY"
GATE_TOL = 0.02                       # a candidate may lose at most 2 points on the frozen holdout
FALSE_ALARM_REASONS = {
    "sensor_or_measurement_error": "Sensor or measurement error (probe off, bad reading)",
    "expected_for_this_patient": "Normal or expected for this patient (chronic / baseline)",
    "already_being_treated": "Already being treated, no new action needed",
    "treated_early_event_prevented": "Warning was right, early treatment prevented the event",
    "wrong_risk_type": "Patient is unwell, but not from this risk",
    "other": "Other (write what it was)"}
EXCLUDE_FROM_TRAINING = {"treated_early_event_prevented"}   # warning was correct: teaching the model "no event" here would make it too quiet (treatment paradox)
FEEDBACK_WEIGHT = 5.0                 # a staff-verified example counts 5x a historical one when retraining Stage 3
now = lambda: dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------------------ storage (csv folder or Exasol tables)
class Store:
    def __init__(self, kind="csv", where="state", schema="CAREGUARD_RESULTS", con=None):
        self.kind, self.where, self.schema, self.con = kind, where, schema, con
        if kind == "csv":
            os.makedirs(where, exist_ok=True)

    def read(self, name):
        try:
            if self.kind == "csv":
                f = os.path.join(self.where, f"{name}.csv")
                return pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()
            df = S1.fetch_df(self.con, f'SELECT * FROM "{self.schema}"."{name}"'); df.columns = [c.lower() for c in df.columns]
            # columns Exasol refused (reserved words) were stored as NAME_X by write_exasol: give them their real name back
            return df.rename(columns={c: c[:-2] for c in df.columns if c.endswith("_x") and c[:-2] not in df.columns})
        except Exception:
            return pd.DataFrame()

    def append(self, name, df):
        if df.empty:
            return
        if self.kind == "csv":
            f = os.path.join(self.where, f"{name}.csv")
            df.to_csv(f, mode="a", header=not os.path.exists(f), index=False)
        else:
            write_exasol(self.con, self.schema, name, df)

    def next_id(self, name):
        d = self.read(name)
        return int(d.feedback_id.max()) + 1 if len(d) else 1

    # ---- what the app calls when staff click
    def add_doctor_label(self, visit_id, doctor_label, model_label, model_confidence, reason, staff_role="doctor", model_version=0):
        self.append(FB1, pd.DataFrame([dict(feedback_id=self.next_id(FB1), ts=now(), visit_id=int(visit_id), model_label=str(model_label),
                    model_confidence=float(model_confidence), doctor_label=str(doctor_label), reason=reason, staff_role=staff_role, model_version=int(model_version))]))

    def add_alert_verdict(self, stay_id, hour_idx, risk, verdict, staff_role="nurse", model_version=0, reason_code="", note=""):
        assert verdict in ("false_alarm", "true_alarm")
        if verdict == "false_alarm":
            if reason_code not in FALSE_ALARM_REASONS:
                raise ValueError(f"A false alarm needs a reason. Choose one of: {list(FALSE_ALARM_REASONS)}")
            if reason_code == "other" and not str(note).strip():
                raise ValueError("Reason 'other' needs a short note saying what it was.")
        self.append(FB3, pd.DataFrame([dict(feedback_id=self.next_id(FB3), ts=now(), stay_id=int(stay_id), hour_idx=float(hour_idx), risk=risk,
                    verdict=verdict, reason_code=reason_code if verdict == "false_alarm" else "", note=str(note).strip()[:300], staff_role=staff_role, model_version=int(model_version))]))


# ------------------------------------------------------------------ trigger logic (model-free, unit-tested)
class Service:
    def __init__(self, store, retrain1=None, retrain3=None, threshold=THRESHOLD, corrections_only=False):
        self.store, self.fn, self.threshold, self.corrections_only = store, {1: retrain1, 3: retrain3}, threshold, corrections_only

    def _last(self, stage):
        r = self.store.read(REG)
        r = r[r.stage == stage] if len(r) else r
        return (int(r.consumed_through_id.max()) if len(r) else 0), (int(r.version_after.max()) if len(r) else 0)

    def pending(self, stage):
        """(count that drives the trigger, new feedback rows) since the last retraining attempt of this stage."""
        fb = self.store.read(FB1 if stage == 1 else FB3)
        if fb.empty:
            return 0, fb
        new = fb[fb.feedback_id > self._last(stage)[0]]
        if stage == 1:
            n = int((new.doctor_label != new.model_label).sum()) if self.corrections_only else len(new)
        else:
            n = int((new.verdict == "false_alarm").sum())          # only corrected FALSE alarms count; confirmed alarms still train the model
        return n, new

    def check(self):
        events = []
        for stage in (1, 3):
            n, new = self.pending(stage)
            if n >= self.threshold and self.fn[stage] is not None:
                fb_all = self.store.read(FB1 if stage == 1 else FB3)
                through = int(fb_all.feedback_id.max()); ver = self._last(stage)[1]
                res = self.fn[stage](fb_all, ver + 1)
                row = dict(ts=now(), stage=stage, version_after=ver + 1 if res["promoted"] else ver,
                           trigger=f"auto_{n}_{'doctor_labels' if stage == 1 else 'false_alarms'}", consumed_through_id=through,
                           n_new=int(n), promoted=bool(res["promoted"]), detail=json.dumps(res.get("detail", {}), default=str))
                self.store.append(REG, pd.DataFrame([row]))          # attempt logged even if rejected -> counter restarts, no retrain loop
                events.append(row)
        return events

    def watch(self, every=10):
        print(f"watching for feedback (retrain at {self.threshold} labels / false alarms), Ctrl+C to stop")
        while True:
            for e in self.check():
                print(f"[{e['ts']}] stage {e['stage']} retrained: promoted={e['promoted']} version={e['version_after']} {e['detail'][:300]}")
            time.sleep(every)


# ------------------------------------------------------------------ Stage 1 retraining
class Stage1:
    def __init__(self, A, LIVE, model_dir="models"):
        self.model_dir = model_dir; os.makedirs(model_dir, exist_ok=True)
        dA, self.live = S1.build_frame(A), S1.build_frame(LIVE)
        rng = np.random.default_rng(0); idx = rng.permutation(len(dA)); k = int(.8 * len(dA))
        self.train, self.hold = dA.iloc[idx[:k]].reset_index(drop=True), dA.iloc[idx[k:]].reset_index(drop=True)
        i2 = rng.permutation(len(self.live)); h = len(self.live) // 2
        self.pool, self.test = self.live.iloc[i2[:h]].reset_index(drop=True), self.live.iloc[i2[h:]].reset_index(drop=True)
        self.version = 0; self.m = S1.Model().fit(self.train); joblib.dump(self.m, f"{model_dir}/stage1_v0.joblib")
        self.v0_acc = S1.report(self.m, self.hold)["accuracy"]            # floor: many small steps may not add up to a big loss

    def retrain(self, fb, next_version):
        lab = fb.drop_duplicates("visit_id", keep="last")[["visit_id", "doctor_label"]]
        L = self.live.merge(lab, left_on="stay_id", right_on="visit_id"); L["dx_group"] = L.doctor_label
        cand = S1.Model().fit(pd.concat([self.train, L[self.train.columns]], ignore_index=True))
        old, new = S1.report(self.m, self.hold), S1.report(cand, self.hold)
        promoted = new["accuracy"] >= old["accuracy"] - GATE_TOL and new["accuracy"] >= self.v0_acc - 0.03
        detail = dict(labels_used=int(len(L)), holdout_A_before=old["accuracy"], holdout_A_after=new["accuracy"],
                      new_hospital_before=S1.report(self.m, self.test), new_hospital_after=S1.report(cand, self.test))
        if promoted:
            self.m, self.version = cand, next_version; joblib.dump(cand, f"{self.model_dir}/stage1_v{next_version}.joblib")
        return dict(promoted=promoted, detail=detail)


class Stage1NHAMCS:
    """Stage 1 on real NHAMCS data. Base training = 2018-2021 (80%), frozen holdout = the other 20%. Doctor labels can refer to 2022 visits
    (ST1_PREDICTIONS) or to live intakes (ST1_INTAKE). A promoted model is written to models/ and models/stage1_live.json points the app to it."""
    def __init__(self, cache="nhamcs_cache", store=None, model_dir="models"):
        self.model_dir, self.store = model_dir, store; os.makedirs(model_dir, exist_ok=True)
        d = pd.read_csv(os.path.join(cache, "nhamcs_ed.csv.gz")); d["cc"] = d.cc.fillna("")
        tr = d[d.year <= 2021].reset_index(drop=True); rng = np.random.RandomState(0); p = rng.permutation(len(tr)); k = int(.8 * len(tr))
        self.train, self.hold, self.live = tr.iloc[p[:k]].reset_index(drop=True), tr.iloc[p[k:]].reset_index(drop=True), d[d.year == 2022].reset_index(drop=True)
        f = os.path.join(model_dir, "stage1_nhamcs.joblib")
        self.m = joblib.load(f) if os.path.exists(f) else S1.Model().fit(self.train)
        live = os.path.join(model_dir, "stage1_live.json")
        self.version = json.load(open(live))["version"] if os.path.exists(live) else 0
        if self.version:
            self.m = joblib.load(json.load(open(live))["path"])
        self.v0_acc = S1.report(self.m, self.hold)["accuracy"]

    def _intakes(self):
        it = self.store.read("ST1_INTAKE") if self.store is not None else pd.DataFrame()
        if it.empty:
            return it
        it["cc"] = it.cc.fillna(""); return it

    def retrain(self, fb, next_version):
        lab = fb.drop_duplicates("visit_id", keep="last")[["visit_id", "doctor_label"]]
        pool = pd.concat([self.live, self._intakes()], ignore_index=True)
        L = pool.merge(lab, left_on="stay_id", right_on="visit_id"); L["dx_group"] = L.doctor_label
        cols = [c for c in self.train.columns if c in L.columns]
        cand = S1.Model().fit(pd.concat([self.train, L[cols]], ignore_index=True))
        test = self.live[~self.live.stay_id.isin(lab.visit_id)]
        old, new = S1.report(self.m, self.hold), S1.report(cand, self.hold)
        promoted = new["accuracy"] >= old["accuracy"] - GATE_TOL and new["accuracy"] >= self.v0_acc - 0.03
        detail = dict(labels_used=int(len(L)), holdout_before=old["accuracy"], holdout_after=new["accuracy"],
                      year2022_before=S1.report(self.m, test)["accuracy"], year2022_after=S1.report(cand, test)["accuracy"])
        if promoted:
            self.m, self.version = cand, next_version; path = f"{self.model_dir}/stage1_nhamcs_v{next_version}.joblib"; joblib.dump(cand, path)
            json.dump(dict(version=next_version, path=path), open(f"{self.model_dir}/stage1_live.json", "w"))
        return dict(promoted=promoted, detail=detail)


# ------------------------------------------------------------------ Stage 3 retraining
class Stage3:
    """ICU early-warning models that learn from staff false-alarm feedback. Data: MIMIC-shaped sources (A = training hospital,
    LIVE = the hospital on the board) or prebuilt panels (PhysioNet 2019). board_stays + publish: after a promoted retrain
    the ICU board is rebuilt with the new models, so the app shows the effect."""
    def __init__(self, A=None, LIVE=None, model_dir="models", target_rate=0.15, panels=None, board_stays=None, publish=None):
        import stage3_multirisk as M3, stage3_pipeline as P3, risks as R
        self.M3, self.P3, self.R, self.rate, self.model_dir = M3, P3, R, target_rate, model_dir
        self.board_stays, self.publish = board_stays, publish
        os.makedirs(model_dir, exist_ok=True)
        if panels is not None:
            pa, Ha, pb, Hb = panels
            pa, self.Xa = M3.build_X(pa, Ha); pb, Xb = M3.build_X(pb, Hb); self.Xb = Xb.reindex(columns=self.Xa.columns)
        else:
            pa, Ta = M3.load(A); pa, self.Xa = M3.build_X(pa, M3.history(Ta, pa.stay_id.unique()))
            pb, Tb = M3.load(LIVE); pb, Xb = M3.build_X(pb, M3.history(Tb, pb.stay_id.unique())); self.Xb = Xb.reindex(columns=self.Xa.columns)
        self.pa, self.pb = pa, pb
        self.La, self.Ma = M3.labels(pa); self.Lb, self.Mb = M3.labels(pb)
        val, tr = P3.stay_split(pa); self.va, self.tr = pa.stay_id.isin(val).values, pa.stay_id.isin(tr).values
        ids = pb.stay_id.drop_duplicates().sample(frac=1, random_state=1).values; h = len(ids) // 2
        self.pool_m, self.test_m = pb.stay_id.isin(ids[:h]).values, pb.stay_id.isin(ids[h:]).values     # staff review the pool; the test half is never touched
        self.version = 0; self.models, self.thr = {}, {}
        self.domains = [d for d in R.DOMAINS if self.La[d][self.tr & self.Ma[d].values].nunique() == 2]   # risks with events in this data
        for d in self.domains:
            self.models[d] = self._fit(self.Xa[self.tr & self.Ma[d].values], self.La[d][self.tr & self.Ma[d].values])
            self.thr[d] = self._thr(self.models[d], d)
        self._save()
        self.row_pos = {(int(s), float(h_)): i for i, (s, h_) in enumerate(zip(pb.stay_id, pb.hour_idx))}

    def _fit(self, X, y, w=None):
        import lightgbm as lgb
        pos = max(float(y.sum()), 1.0)
        return lgb.LGBMClassifier(**self.P3.PARAMS, scale_pos_weight=min((len(y) - pos) / pos, 20)).fit(X, y, sample_weight=w)

    def _thr(self, model, d):
        sub = self.pa[self.va & self.Ma[d].values][["stay_id", "hour_idx", "hour"]].copy()
        sub["risk"] = model.predict_proba(self.Xa[self.va & self.Ma[d].values])[:, 1]
        return self.P3.threshold_for_rate(sub, "risk", self.rate)

    def _save(self):
        joblib.dump(dict(models=self.models, thr=self.thr), f"{self.model_dir}/stage3_v{self.version}.joblib")

    def alert_rows(self, mask, d, models=None, thr=None):
        """Model early-warning episodes (flag = risk >= threshold while the standard-score tier is still below High)."""
        models, thr = models or self.models, thr or self.thr
        sel = mask & self.Mb[d].values
        sub = self.pb[sel][["stay_id", "hour_idx", "hour"]].copy(); sub["risk"] = models[d].predict_proba(self.Xb[sel])[:, 1]
        return self.P3.alert_times(sub, "risk", thr[d])

    def head_quality(self, model, d, which="A"):
        """AUPRC, event-stay recall and alert precision at a threshold that gives the same alert volume for every model."""
        from sklearn.metrics import average_precision_score
        if which == "A":
            df, X, y, m = self.pa, self.Xa, self.La[d], self.va & self.Ma[d].values
        else:
            df, X, y, m = self.pb, self.Xb, self.Lb[d], self.test_m & self.Mb[d].values
        sub = df[m][["stay_id", "hour_idx", "hour"]].copy(); sub["risk"] = model.predict_proba(X[m])[:, 1]; sub["y"] = y[m].values
        if sub.y.sum() == 0:
            return dict(auprc=None, recall=None, precision=None, alerts=0)
        thr = self.P3.threshold_for_rate(sub, "risk", self.rate)
        al = self.P3.alert_times(sub, "risk", thr).merge(sub[["stay_id", "hour_idx", "y"]], on=["stay_id", "hour_idx"], how="left")
        last_ev = sub[sub.y == 1].groupby("stay_id").hour_idx.max()
        caught = sum(((al.stay_id == s) & (al.hour_idx <= h)).any() for s, h in last_ev.items())
        return dict(auprc=round(float(average_precision_score(sub.y, sub.risk)), 4), recall=round(caught / len(last_ev), 3),
                    precision=round(float(al.y.mean()), 3) if len(al) else None, alerts=int(len(al)))

    def retrain(self, fb, next_version):
        R = self.R; detail, promoted_any, new_models = {}, False, dict(self.models)
        if "reason_code" in fb.columns:
            fb = fb[~fb.reason_code.isin(EXCLUDE_FROM_TRAINING)]        # still logged and counted toward the 50, but not used as "no event" examples
        for d in self.domains:
            f = fb[fb.risk == d]
            idx = [self.row_pos[(int(s), float(h))] for s, h in zip(f.stay_id, f.hour_idx) if (int(s), float(h)) in self.row_pos]
            if not idx or (f.verdict == "false_alarm").sum() == 0:
                continue
            fy = (f.verdict.values[[i for i, (s, h) in enumerate(zip(f.stay_id, f.hour_idx)) if (int(s), float(h)) in self.row_pos]] == "true_alarm").astype(int)
            base = self.tr & self.Ma[d].values
            X = pd.concat([self.Xa[base], self.Xb.iloc[idx]]); y = np.concatenate([self.La[d][base].values, fy])
            w = np.concatenate([np.ones(base.sum()), np.full(len(idx), FEEDBACK_WEIGHT)])
            cand = self._fit(X, pd.Series(y), w)
            o, n = self.head_quality(self.models[d], d, "A"), self.head_quality(cand, d, "A")
            ok = (n["auprc"] is not None) and n["auprc"] >= (o["auprc"] or 0) - GATE_TOL and (n["recall"] or 0) >= (o["recall"] or 0) - GATE_TOL
            detail[d] = dict(feedback_rows=int(len(idx)), false_alarms=int((fy == 0).sum()), passed_gate=bool(ok), holdout_A_before=o, holdout_A_after=n,
                             new_hospital_before=self.head_quality(self.models[d], d, "B"), new_hospital_after=self.head_quality(cand, d, "B"))
            if ok:
                new_models[d] = cand; promoted_any = True
        if promoted_any:
            self.models, self.version = new_models, next_version
            self.thr = {d: self._thr(self.models[d], d) for d in self.domains}; self._save()
            self.republish()
        return dict(promoted=promoted_any, detail=detail)

    def republish(self, out="results"):
        """Rebuild the ICU board with the current models (only when the board patients are part of LIVE)."""
        if not self.board_stays:
            return
        sel = self.pb.stay_id.isin(self.board_stays).values
        df = self.pb[sel].copy()
        for d in self.domains:
            df[f"risk_{d}"] = self.models[d].predict_proba(self.Xb[sel])[:, 1]
        _, snap, tl = self.M3.board(df, self.thr, out)
        self.M3.save_heads(self.models, self.thr, self.Xa.columns, f"retrained v{self.version}")
        if self.publish:
            self.publish(snap, tl)
        print(f"   ICU board republished with model v{self.version} ({len(snap)} patients)")


def physionet_stage3(a, store):
    """Stage 3 retraining on the same PhysioNet patients physionet_icu.py used (ids saved in results/stage3_physionet_ids.json)."""
    import physionet_icu as PN
    ids = json.load(open("results/stage3_physionet_ids.json"))
    ns = argparse.Namespace(source=a.source, schema=a.sepsis_schema, cache=a.sepsis_cache)
    d = PN.load_raw(ns); d = d[d.patient_id.astype(str).isin(set(ids["train"]) | set(ids["test"]))]
    print(f"Stage 3 (PhysioNet): building panels for {d.patient_id.nunique()} patients ...", flush=True)
    p, H, hosp = PN.to_panel(d); isB = p.stay_id.map(hosp).eq("B").values
    publish = None
    if store.kind == "exasol":
        def publish(snap, tl):
            self_m3 = __import__("stage3_multirisk"); self_m3.write_board_exasol(store.con, store.schema, snap, tl, "results")
    return Stage3(panels=(p[~isB].copy(), H, p[isB].copy(), H), board_stays=set(ids["board_stay_ids"]), publish=publish)


# ------------------------------------------------------------------ demo: simulated staff
def demo(a):
    shutil.rmtree("state", ignore_errors=True); shutil.rmtree("models", ignore_errors=True)
    store = Store("csv", "state"); A = Source("csv", a.dir)
    s1 = Stage1(A, Source("csv", a.live1_dir)); s3 = Stage3(A, Source("csv", a.live3_dir))
    svc = Service(store, s1.retrain, s3.retrain)
    rng = np.random.default_rng(1)
    print(f"=== Stage 1: simulated doctors label cases; retraining fires every {THRESHOLD} labels")
    queue = s1.pool.copy(); step = 10
    while len(queue) >= step and svc._last(1)[1] < 3 and len(store.read(REG)[store.read(REG).stage == 1]) < 3 if len(store.read(REG)) else True:
        pred, conf, margin, auto = S1.route(s1.m, queue)
        low = np.argsort(conf)[:step - 1]; conf_i = np.setdiff1d(np.where(auto)[0], low)
        pick = list(low) + ([int(rng.choice(conf_i))] if len(conf_i) else [])            # mostly low-confidence, plus audit of a confident answer
        for j in pick:
            r = queue.iloc[j]
            store.add_doctor_label(r.stay_id, r.dx_group, pred[j], conf[j], "low_confidence" if j in low else "audit_confident", "doctor", s1.version)
        queue = queue.drop(queue.index[pick]).reset_index(drop=True)
        n, _ = svc.pending(1)
        for e in svc.check():
            d = json.loads(e["detail"]); b, f = d["new_hospital_before"], d["new_hospital_after"]
            print(f"  {e['n_new']} doctor labels -> retrained, promoted={e['promoted']} (v{e['version_after']}): "
                  f"new-hospital accuracy {b['accuracy']} -> {f['accuracy']}, wrong answers shown to patients {b['wrong_shown_to_patient']} -> {f['wrong_shown_to_patient']}, "
                  f"old-hospital holdout {d['holdout_A_before']} -> {d['holdout_A_after']}")
        if len(store.read(REG)) and (store.read(REG).stage == 1).sum() >= 3:
            break
    print(f"=== Stage 3: simulated nurses mark alerts true/false; retraining fires every {THRESHOLD} false alarms")
    seen, nfb = set(), 0
    while (store.read(REG).stage == 3).sum() < 2 if len(store.read(REG)) else True:
        cand = []
        for d in s3.R.DOMAINS:
            al = s3.alert_rows(s3.pool_m, d)
            cand += [(d, int(r.stay_id), float(r.hour_idx)) for r in al.itertuples() if (d, int(r.stay_id), float(r.hour_idx)) not in seen]
        if not cand:
            print("  no more alerts to review in the demo pool"); break
        for d, s, h in [cand[i] for i in rng.permutation(len(cand))[:10]]:
            seen.add((d, s, h)); y = int(s3.Lb[d].values[s3.row_pos[(s, h)]])
            store.add_alert_verdict(s, h, d, "true_alarm" if y else "false_alarm", "nurse", s3.version, reason_code="" if y else "expected_for_this_patient")
        for e in svc.check():
            dd = json.loads(e["detail"]); print(f"  {e['n_new']} false alarms corrected -> retrained, promoted={e['promoted']} (v{e['version_after']})")
            for d, v in dd.items():
                b, f = v["new_hospital_before"], v["new_hospital_after"]
                print(f"    {d:6s} gate={'pass' if v['passed_gate'] else 'FAIL'}; same alert volume on unseen stays: real alerts {b['precision']} -> {f['precision']}, "
                      f"events caught {b['recall']} -> {f['recall']}")
        if len(seen) > 4000:
            break
    print("registry:"); print(store.read(REG)[["stage", "version_after", "trigger", "n_new", "promoted"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["demo", "watch", "once"])
    ap.add_argument("--source", choices=["csv", "exasol"], default="csv"); ap.add_argument("--dir"); ap.add_argument("--live1-dir"); ap.add_argument("--live3-dir")
    ap.add_argument("--schema", default="CAREGUARD_A"); ap.add_argument("--live1-schema", default="CAREGUARD_SURGE"); ap.add_argument("--live3-schema", default="CAREGUARD_C")
    ap.add_argument("--out-schema", default="CAREGUARD_RESULTS"); ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--corrections-only", action="store_true", help="Stage 1: count only labels that differ from the model's guess")
    ap.add_argument("--state", default="app_state", help="csv mode: folder the app writes staff feedback to")
    ap.add_argument("--stage3-physionet", action="store_true", help="Stage 3 learns on PhysioNet 2019 (the real-patient ICU board from physionet_icu.py)")
    ap.add_argument("--sepsis-cache", default="sepsis2019_cache"); ap.add_argument("--sepsis-schema", default="CAREGUARD_SEPSIS19")
    ap.add_argument("--stage1-nhamcs", default=None, help="folder with nhamcs_ed.csv.gz: Stage 1 retrains on real NHAMCS data (recommended)")
    a = ap.parse_args()
    if a.cmd == "demo":
        return demo(a)
    ex = a.source == "exasol"
    A = Source(a.source, a.schema if ex else a.dir)
    store = Store("exasol", schema=a.out_schema, con=A.con) if ex else Store("csv", a.state)
    s1 = Stage1NHAMCS(a.stage1_nhamcs, store) if a.stage1_nhamcs else Stage1(A, Source(a.source, a.live1_schema if ex else a.live1_dir)); s3 = physionet_stage3(a, store) if a.stage3_physionet else Stage3(A, Source(a.source, a.live3_schema if ex else a.live3_dir))
    svc = Service(store, s1.retrain, s3.retrain, corrections_only=a.corrections_only)
    if a.cmd == "once":
        print(svc.check() or "nothing to do yet:", {1: svc.pending(1)[0], 3: svc.pending(3)[0]})
    else:
        svc.watch(a.every)


if __name__ == "__main__":
    main()
