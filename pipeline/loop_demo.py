#!/usr/bin/env python3
"""
loop_demo.py - shows the doctor-in-the-loop mechanism on synthetic distribution shift (not a clinical result).

Stage 1  train on hospital A (default) -> deploy on a SURGE hospital (new wording, shifted vitals, and a NOVEL
         diagnosis class the model has never seen). Low-confidence cases go to a doctor (oracle = planted label);
         each round the doctor labels the queue head, the model is retrained, a promotion gate on a frozen
         hospital-A holdout decides whether the new model may replace the old one.
Stage 3  train hourly sepsis-risk model on hospital A -> deploy on hospital C (no arterial lines, lactate rarely ordered, GCS every
         12h, atypical sepsis: afebrile and no tachycardia). Doctors review stays the model missed / alarmed on; their
         confirmed onset times are added and the model is retrained.

Everything is computed from MIMIC-schema tables via scores.py, the same code path as real MIMIC-IV.
Numbers printed here describe the synthetic generator, not patients.
"""
from __future__ import annotations
import os, sys, json, argparse, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import synth_mimic as G
import scores as S
from load_tables import load_tables
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb


# ================================================================= Stage 1
def ed_frame(tables, truth_dir):
    tr, es, pts = tables["triage"], tables["edstays"], tables["patients"]
    d = tr.merge(es[["stay_id", "subject_id", "gender", "arrival_transport", "intime"]], on=["stay_id", "subject_id"]).merge(
        pts[["subject_id", "anchor_age"]], on="subject_id")
    n2 = S.news2_from_triage(tr).reset_index(drop=True)
    d = d.reset_index(drop=True)
    t_raw = pd.to_numeric(d.temperature, errors="coerce")
    d["temp_c"] = np.where((t_raw >= 25) & (t_raw <= 45), t_raw, (t_raw - 32) * 5 / 9)
    for c, nm in [("heartrate", "hr"), ("resprate", "rr"), ("o2sat", "spo2"), ("sbp", "sbp"), ("dbp", "dbp")]:
        d[nm] = S._clean(nm, d[c]) if nm in S.PLAUSIBLE else d[c]
    d["news2_triage"] = n2.news2.values
    d["male"] = (d.gender == "M").astype(float); d["ambulance"] = (d.arrival_transport == "AMBULANCE").astype(float)
    d["cc"] = d.chiefcomplaint.fillna("").str.lower()
    # history: number of chronic diagnoses on file for the subject (from diagnoses_icd, never from the label table)
    hist = tables.get("diagnoses_icd")
    d["n_hist_dx"] = d.subject_id.map(hist.groupby("subject_id").size()).fillna(0).values if hist is not None else 0
    lab = pd.read_csv(os.path.join(truth_dir, "truth_ed.csv.gz"))[["ed_stay_id", "dx_group"]].rename(columns={"ed_stay_id": "stay_id"})
    return d.merge(lab, on="stay_id")


NUM = ["temp_c", "hr", "rr", "spo2", "sbp", "dbp", "acuity", "news2_triage", "anchor_age", "male", "ambulance", "n_hist_dx"]


class Stage1:
    def __init__(self):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.med = None

    def _X(self, d, fit=False):
        num = d[NUM].astype(float)
        if fit:
            self.med = num.median()
            self.mu = num.fillna(self.med).mean(); self.sd = num.fillna(self.med).std().replace(0, 1)
        miss = num.isna().astype(float).values
        z = ((num.fillna(self.med) - self.mu) / self.sd).values
        txt = self.vec.fit_transform(d.cc) if fit else self.vec.transform(d.cc)
        return hstack([csr_matrix(np.hstack([z, miss])), txt]).tocsr()

    def fit(self, d):
        X = self._X(d, fit=True)
        self.clf = LogisticRegression(max_iter=600, C=2.0).fit(X, d.dx_group)
        return self

    def proba(self, d):
        return self.clf.predict_proba(self._X(d)), self.clf.classes_


def stage1_route(model, d, tau):
    p, cls = model.proba(d)
    conf = p.max(1); pred = cls[p.argmax(1)]
    srt = np.sort(p, 1); margin = srt[:, -1] - srt[:, -2]
    auto = (conf >= tau) & (margin >= .15)
    return pred, conf, auto


def stage1_report(model, d, tau):
    pred, conf, auto = stage1_route(model, d, tau)
    ok = pred == d.dx_group.values
    return dict(n=len(d), model_accuracy=round(float(ok.mean()), 3), auto_coverage=round(float(auto.mean()), 3),
                auto_accuracy=round(float(ok[auto].mean()), 3) if auto.any() else None,
                errors_shown_to_patient=int((auto & ~ok).sum()), sent_to_doctor=int((~auto).sum()),
                novel_class_sent_to_doctor=(round(float((~auto[d.dx_group.values == "novel_resp_cluster"]).mean()), 3) if (d.dx_group == "novel_resp_cluster").any() else None))


def run_stage1(dirs, tau=.80, rounds=5, per_round=None, out=None):
    A, A_test, SU = (load_tables(dirs[k]) for k in ("A", "A_test", "surge"))
    dA, dAt, dS = ed_frame(A, dirs["A"] + "/synth_truth"), ed_frame(A_test, dirs["A_test"] + "/synth_truth"), ed_frame(SU, dirs["surge"] + "/synth_truth")
    # surge: first half = live traffic queue the doctors work through, second half = frozen surge test set
    rng = np.random.default_rng(0); idx = rng.permutation(len(dS)); half = len(dS) // 2
    pool, dS_test = dS.iloc[idx[:half]].reset_index(drop=True), dS.iloc[idx[half:]].reset_index(drop=True)
    per_round = per_round or max(20, len(pool) // (rounds * 2))
    m = Stage1().fit(dA); log = []
    log.append(dict(round=0, labels_from_doctors=0, A_holdout=stage1_report(m, dAt, tau), surge_test=stage1_report(m, dS_test, tau)))
    train = dA.copy(); queue = pool.copy(); asked = 0
    for r in range(1, rounds + 1):
        _, conf, auto = stage1_route(m, queue, tau)
        order = np.argsort(conf)                      # least confident first (what a real queue would show doctors)
        take = queue.iloc[order[:per_round]]
        train = pd.concat([train, take], ignore_index=True)    # doctor label = planted truth (oracle)
        queue = queue.iloc[order[per_round:]].reset_index(drop=True)
        asked += len(take)
        cand = Stage1().fit(train)
        gate_old, gate_new = stage1_report(m, dAt, tau)["model_accuracy"], stage1_report(cand, dAt, tau)["model_accuracy"]
        promoted = gate_new >= gate_old - .01          # promotion gate: frozen hospital-A holdout may not drop > 1 point
        if promoted:
            m = cand
        log.append(dict(round=r, labels_from_doctors=asked, promoted=bool(promoted), gate_A_holdout=(gate_old, gate_new),
                        A_holdout=stage1_report(m, dAt, tau), surge_test=stage1_report(m, dS_test, tau)))
    return log


# ================================================================= Stage 3
def stage3_features(tables):
    P, _ = S.build_hourly_panel(tables)
    o = S.sepsis3_proxy(tables, P)
    P = S.add_sepsis_label(P, o)
    g = P.groupby("stay_id")
    for c in ["hr", "rr", "spo2", "map", "temp_c", "sbp"]:
        P[c + "_m6"] = g[c].transform(lambda s: s.rolling(6, min_periods=1).mean())
        P[c + "_d6"] = P[c] - g[c].shift(6)
    P["hour_idx"] = g.cumcount()
    P["age"] = P.stay_id.map(tables["icustays"].merge(tables["patients"], on="subject_id").set_index("stay_id").anchor_age)
    P = P[P.post_onset == 0].copy()
    return P, o


FEATS = None


def feat_cols(P):
    drop = {"stay_id", "hour", "label_6h", "post_onset", "news2_band", "qsofa_pos", "sirs_pos"}
    cols = [c for c in P.columns if c not in drop and not c.endswith("_raw") and pd.api.types.is_numeric_dtype(P[c])]
    # the model must not see the rules it is compared with as hard inputs? they are legitimate features: keep NEWS2/SOFA.
    return cols


def fit_lgb(P, cols):
    pos = P.label_6h.mean()
    return lgb.LGBMClassifier(n_estimators=250, learning_rate=.05, num_leaves=15, min_child_samples=30, subsample=.8, colsample_bytree=.7,
                              scale_pos_weight=min(10, (1 - pos) / max(pos, 1e-4)), verbose=-1, random_state=0).fit(P[cols], P.label_6h)


def alert_stats(P, score, thr, onset_df, label=""):
    P = P.assign(alert=(score >= thr).astype(int))
    pd_days = len(P) / 24.0
    o = onset_df.set_index("stay_id").sepsis3_onset.dropna()
    hit = 0
    for sid, t in o.items():
        w = P[(P.stay_id == sid) & (P.hour < t) & (P.hour >= t - pd.Timedelta(hours=12))]
        hit += int(len(w) > 0 and w.alert.max() > 0)
    n_sep = int(len(o))
    return dict(alerts_per_patient_day=round(float(P.alert.sum() / pd_days), 2), sepsis_stays=n_sep,
                stays_alerted_within_12h_before_onset=hit, stay_sensitivity=round(hit / max(n_sep, 1), 3))


def thr_for_rate(P, score, rate):
    s = np.sort(score)[::-1]; k = int(rate * len(P) / 24.0)
    return s[min(max(k, 1), len(s) - 1)]


def run_stage3(dirs, rate=1.0, rounds=4, stays_per_round=25):
    TA, TAt, TB = (load_tables(dirs[k]) for k in ("A", "A_test", "B"))
    PA, oA = stage3_features(TA); PAt, oAt = stage3_features(TAt); PB, oB = stage3_features(TB)
    cols = feat_cols(PA)
    m = fit_lgb(PA, cols)
    sA = m.predict_proba(PAt[cols])[:, 1]; thr = thr_for_rate(PAt, sA, rate)
    # split B stays: pool (doctors review, retrain) vs frozen test
    rng = np.random.default_rng(1); sids = PB.stay_id.unique(); rng.shuffle(sids)
    k = len(sids) // 2; pool_ids, test_ids = set(sids[:k]), set(sids[k:])
    PBp, PBt = PB[PB.stay_id.isin(pool_ids)], PB[PB.stay_id.isin(test_ids)]
    oBt = oB[oB.stay_id.isin(test_ids)]

    def evaluate(model, tag):
        s = model.predict_proba(PBt[cols])[:, 1]
        res = dict(AUPRC=round(float(average_precision_score(PBt.label_6h, s)), 4), base_rate=round(float(PBt.label_6h.mean()), 4))
        res.update(alert_stats(PBt, s, thr_for_rate(PBt, s, rate), oBt))
        return res
    rules = {}
    for nm, sc, th in [("NEWS2>=5", PBt.news2.fillna(0).values, 5), ("qSOFA>=2", PBt.qsofa.fillna(0).values, 2), ("SOFA>=4", PBt.sofa.values, 4)]:
        rules[nm] = alert_stats(PBt, sc, th, oBt); rules[nm]["AUPRC"] = round(float(average_precision_score(PBt.label_6h, sc)), 4)
    log = dict(rules_on_target_hospital=rules, A_test_model=dict(AUPRC=round(float(average_precision_score(PAt.label_6h, sA)), 4), **alert_stats(PAt, sA, thr, oAt)))
    log["rounds"] = [dict(round=0, doctor_reviewed_stays=0, target_hospital_test=evaluate(m, "B"))]
    # doctor review order: stays whose max score is highest among sepsis-labelled ones missed/uncertain first -> here: stays ranked by
    # (label disagreement x uncertainty); a doctor confirms onset time (oracle = Sepsis-3 proxy labels on observed data)
    train = PA.copy(); reviewed = []
    sp = m.predict_proba(PBp[cols])[:, 1]; pool_scores = PBp.assign(s=sp).groupby("stay_id").agg(smax=("s", "max"), pos=("label_6h", "max"))
    review_order = list(pool_scores.sort_values(["pos", "smax"], ascending=[False, True]).index)   # positives the model scored lowest first
    for r in range(1, rounds + 1):
        take = review_order[(r - 1) * stays_per_round: r * stays_per_round]
        reviewed += take
        train = pd.concat([PA, PBp[PBp.stay_id.isin(reviewed)]], ignore_index=True)
        cand = fit_lgb(train, cols)
        gate_old = average_precision_score(PAt.label_6h, m.predict_proba(PAt[cols])[:, 1]); gate_new = average_precision_score(PAt.label_6h, cand.predict_proba(PAt[cols])[:, 1])
        promoted = gate_new >= gate_old * 0.95
        if promoted:
            m = cand
        log["rounds"].append(dict(round=r, doctor_reviewed_stays=len(reviewed), promoted=bool(promoted), gate_A_AUPRC=(round(gate_old, 4), round(gate_new, 4)), target_hospital_test=evaluate(m, "B")))
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--work", default="/tmp/loop_demo"); ap.add_argument("--small", action="store_true")
    a = ap.parse_args()
    sz = dict(A=(2500, 700, 1), A_test=(1200, 300, 2), B=(1800, 700, 3), surge=(2500, 0, 4)) if not a.small else dict(A=(800, 200, 1), A_test=(400, 100, 2), B=(600, 200, 3), surge=(800, 0, 4))
    prof = dict(A="default", A_test="default", B="hospital_c", surge="surge")
    dirs = {}
    for k, (ne, ni, sd) in sz.items():
        dirs[k] = os.path.join(a.work, k)
        if not os.path.exists(dirs[k] + "/icu/icustays.csv.gz"):
            G.generate(dirs[k], n_ed=ne, n_icu_direct=ni, seed=sd, profile=prof[k])
    out = dict(stage1=run_stage1(dirs), stage3=run_stage3(dirs))
    json.dump(out, open(os.path.join(a.work, "loop_demo_results.json"), "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))
