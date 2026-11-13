#!/usr/bin/env python3
"""Stage 1 on real NHAMCS data: train 2018-2021, validate on a held-out 20% of those years, TEST on 2022 (never touched in training).
  python nhamcs_stage1.py --cache nhamcs_cache [--model logreg|mlp|both]
Writes: results/nhamcs_stage1_metrics.json, stage1_predictions.csv (2022 visits: routing, top-3 guesses, evidence), models/stage1_nhamcs.joblib."""
from __future__ import annotations
import argparse, json, os, sys, warnings
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage1_pipeline as S1


def ece(P, y, classes):
    conf, ok = P.max(1), (np.array(classes)[P.argmax(1)] == y)
    e = 0.0
    for lo in np.arange(0, 1, .1):
        m = (conf > lo) & (conf <= lo + .1)
        if m.any():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(e)


def evaluate(m, d):
    pred, conf, margin, P = m.predict(d); auto = (conf >= S1.TAU) & (margin >= S1.MARGIN); ok = pred == d.dx_group.values
    top3 = np.mean([d.dx_group.values[i] in np.array(m.classes_)[np.argsort(-P[i])[:3]] for i in range(len(d))])
    return dict(n=len(d), accuracy=round(float(ok.mean()), 3), top3_accuracy=round(float(top3), 3), auto_share=round(float(auto.mean()), 3),
                auto_accuracy=round(float(ok[auto].mean()), 3) if auto.any() else None, doctor_share=round(float((~auto).mean()), 3),
                wrong_shown_to_patient_pct=round(float((auto & ~ok).mean() * 100), 2), accuracy_when_sent_to_doctor=round(float(ok[~auto].mean()), 3) if (~auto).any() else None,
                ece=round(ece(P, d.dx_group.values, m.classes_), 3))


def policy_sweep(m, ho, te):
    """What the confidence cut-off costs: share answered automatically vs how often those answers are right (chosen on validation, reported on 2022)."""
    rows = []
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9):
        r = {"tau": tau}
        for nm, d in (("validation", ho), ("test_2022", te)):
            pred, conf, margin, _ = m.predict(d); auto = (conf >= tau) & (margin >= S1.MARGIN); ok = pred == d.dx_group.values
            r[f"auto_share_{nm}"] = round(float(auto.mean()), 3); r[f"auto_accuracy_{nm}"] = round(float(ok[auto].mean()), 3) if auto.any() else None
        rows.append(r)
    return rows


def conformal(m, ho, te, alpha=0.10):
    """Split conformal prediction sets: on exchangeable data the true group is inside the set at least 1-alpha of the time. Singleton set = confident."""
    cls = np.array(m.classes_); _, _, _, Ph = m.predict(ho); idx = {c: i for i, c in enumerate(cls)}
    s = 1 - Ph[np.arange(len(ho)), [idx[c] for c in ho.dx_group]]; n = len(s); q = np.quantile(s, min(1, np.ceil((n + 1) * (1 - alpha)) / n))
    _, _, _, P = m.predict(te); inset = P >= 1 - q; size = inset.sum(1); truth = np.array([idx[c] for c in te.dx_group]); cov = inset[np.arange(len(te)), truth]
    single = size == 1
    return dict(alpha=alpha, qhat=float(q), coverage_2022=round(float(cov.mean()), 3), mean_set_size=round(float(size.mean()), 2), singleton_share=round(float(single.mean()), 3),
                singleton_accuracy=round(float(cov[single].mean()), 3) if single.any() else None, empty_set_share=round(float((size == 0).mean()), 3))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cache", default="nhamcs_cache"); ap.add_argument("--model", choices=["logreg", "mlp", "both"], default="both")
    ap.add_argument("--exasol", action="store_true"); ap.add_argument("--out-schema", default="CAREGUARD_RESULTS")
    ap.add_argument("--split", choices=["year", "random"], default="year", help="year: train <= 2021, test 2022 (NHAMCS); random: 80/20 by visit (MIMIC-IV-ED, shifted dates)")
    a = ap.parse_args()
    d = pd.read_csv(os.path.join(a.cache, "nhamcs_ed.csv.gz")); d["cc"] = d.cc.fillna("")
    if a.split == "year":
        tr_all, te = d[d.year <= d.year.max() - 1].reset_index(drop=True), d[d.year == d.year.max()].reset_index(drop=True)
    else:
        r = np.random.RandomState(1).permutation(len(d)); k = int(.8 * len(d))
        tr_all, te = d.iloc[r[:k]].reset_index(drop=True), d.iloc[r[k:]].reset_index(drop=True)
    rng = np.random.RandomState(0); perm = rng.permutation(len(tr_all)); k = int(.8 * len(tr_all)); tr, ho = tr_all.iloc[perm[:k]], tr_all.iloc[perm[k:]]
    print(f"train {len(tr)} (2018-2021) | validation {len(ho)} | TEST 2022 {len(te)} | groups {d.dx_group.nunique()}")
    base = te.dx_group.value_counts(normalize=True).iloc[0]; res = {"majority_class_accuracy_2022": round(float(base), 3)}; models = {}
    for kind in (["logreg", "mlp"] if a.model == "both" else [a.model]):
        S1.MODEL_KIND = kind; m = S1.Model().fit(tr); models[kind] = m
        res[kind] = dict(validation=evaluate(m, ho), test_2022=evaluate(m, te)); print(kind, json.dumps(res[kind]["test_2022"]))
    best = max(models, key=lambda k: res[k]["validation"]["accuracy"] - 0.5 * res[k]["validation"]["ece"])      # pick on validation only
    res["chosen_on_validation"] = best; m = models[best]; os.makedirs("results", exist_ok=True); os.makedirs("models", exist_ok=True)
    res["policy_sweep"] = policy_sweep(m, ho, te); res["conformal"] = conformal(m, ho, te); print("policy sweep", json.dumps(res["policy_sweep"])); print("conformal", json.dumps(res["conformal"]))
    json.dump(res, open("results/nhamcs_stage1_metrics.json", "w"), indent=1); joblib.dump(m, "models/stage1_nhamcs.joblib")
    preds = S1.predictions_table(m, te, 0); preds.to_csv("stage1_predictions.csv", index=False)
    # what the live API needs: plausible ranges seen in training (unusual-case check), conformal cut-off, routing rule
    num = ["anchor_age", "temp_c", "hr", "rr", "spo2", "sbp", "dbp", "news2_triage"]
    meta = dict(model="logistic regression on triage vitals + reason-for-visit codes", trained_on="NHAMCS ED 2018-2021 adults", tested_on="NHAMCS ED 2022",
                version=0, tau=S1.TAU, margin=S1.MARGIN, conformal_qhat=res["conformal"]["qhat"], conformal_alpha=res["conformal"]["alpha"],
                ranges={c: [float(tr[c].quantile(.005)), float(tr[c].quantile(.995))] for c in num},
                known_tokens=sorted(set(" ".join(tr.cc).split())), test_2022=res[best]["test_2022"])
    json.dump(meta, open("models/stage1_nhamcs_meta.json", "w"))
    json.dump(dict(version=0, path="models/stage1_nhamcs.joblib"), open("models/stage1_live.json", "w"))
    print(f"chosen {best}; wrote stage1_predictions.csv ({len(preds)} visits; to doctor: {(preds.route == 'DOCTOR_QUEUE').sum()})")
    # per-group accuracy on 2022 (honest view of where it is weak)
    pr, cf, mg, _ = m.predict(te); g = pd.DataFrame(dict(group=te.dx_group, ok=pr == te.dx_group.values)).groupby("group").ok.agg(["mean", "size"]).round(3).sort_values("mean")
    print(g.to_string()); g.to_csv("results/nhamcs_stage1_per_group.csv")
    if a.exasol:
        con = S1.Source("exasol", a.out_schema).con; S1.write_exasol(con, a.out_schema, "ST1_PREDICTIONS", preds, replace=True); print("written to Exasol ST1_PREDICTIONS")


if __name__ == "__main__":
    main()
