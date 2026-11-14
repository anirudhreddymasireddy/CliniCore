#!/usr/bin/env python3
"""
CareGuard Stage 1 (intake prediction) + Stage 2 (confidence routing, doctor loop, gated retraining).

Reads MIMIC-ED-schema tables from Exasol (or a local CSV folder), trains a calibrated model, routes cases,
simulates doctor review rounds, retrains with a promotion gate, and writes results back to Exasol.

  # from Exasol (after load_to_exasol.py):
  python stage1_pipeline.py --source exasol --schema CAREGUARD_A --shift-schema CAREGUARD_SURGE
  # from local CSV folders (no database needed):
  python stage1_pipeline.py --source csv --dir data/samples/synth_default --shift-dir data/samples/synth_surge

Label = diagnosis group (table TRUTH_ED, column DX_GROUP). For real NHAMCS/MIMIC replace it with your ICD->category map.
The label table is read ONLY to build y and to simulate the doctor; it is never a feature.
"""
from __future__ import annotations
import argparse, glob, json, os, re, ssl, sys, warnings, datetime as dt
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scores as S
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import hstack, csr_matrix

NEEDED = ["triage", "edstays", "patients", "diagnoses_icd", "truth_ed"]
TAU, MARGIN = 0.80, 0.15
AUDIT_RATE = 0.05                   # share of confident (auto-answered) cases a doctor also checks each round
MODEL_KIND = "logreg"             # set by --model: logreg | mlp   (mlp = 2 ReLU layers 64-32, isotonic-calibrated)            # auto-answer only if confidence >= TAU and margin to 2nd >= MARGIN


# ---------------------------------------------------------------- data access
def _sslopt(dsn):
    """Local Exasol Personal uses a self-signed certificate; remote hosts (Exasol SaaS) are verified unless EXA_INSECURE=1."""
    local = dsn.split(":")[0] in ("127.0.0.1", "localhost") or os.environ.get("EXA_INSECURE") == "1"
    return {"cert_reqs": ssl.CERT_NONE} if local else {"cert_reqs": ssl.CERT_REQUIRED}


class Source:
    def __init__(self, kind, where):
        self.kind, self.where, self.con = kind, where, None
        if kind == "exasol":
            import pyexasol
            pw = os.environ.get("EXA_PASS") or open(os.path.expanduser("~/.exasol-starter-kit/credentials/personal_sys_password")).read().strip()
            dsn = os.environ.get("EXA_DSN", "127.0.0.1:8563")
            self.con = pyexasol.connect(dsn=dsn, user=os.environ.get("EXA_USER", "sys"), password=pw,
                                        encryption=True, websocket_sslopt=_sslopt(dsn))

    def read(self, table):
        if self.kind == "exasol":
            df = self.con.export_to_pandas(f'SELECT * FROM "{self.where}"."{table.upper()}"')
        else:
            f = glob.glob(os.path.join(self.where, "**", f"{table}.csv*"), recursive=True)
            if not f:
                raise FileNotFoundError(f"{table} not found under {self.where}")
            df = pd.read_csv(f[0], low_memory=False)
        df.columns = [c.lower() for c in df.columns]
        return df


def fetch_df(con, sql):
    """Small/medium query results over the normal session (fast for small tables). export_to_pandas opens an extra HTTP
    transfer per call, which costs a fixed delay even for 3 rows - right for millions of rows, wrong for app polling."""
    st = con.execute(sql)
    cols = st.columns()
    df = pd.DataFrame(st.fetchall(), columns=list(cols))
    for c, t in cols.items():
        if str(t.get("type", "")).upper() in ("DECIMAL", "DOUBLE"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def write_exasol(con, schema, table, df, replace=False):
    """Write df to schema.table.
    replace=True : result tables (rebuilt on every run) -> CREATE OR REPLACE, so reruns never duplicate rows or hit an old layout.
    replace=False: append-only tables (feedback, registry) -> table is created on first use; a different column layout raises a clear error.
    Column names Exasol refuses (for example LEVEL) are renamed to NAME_X automatically."""
    if con is None or df.empty:
        return
    from load_to_exasol import ddl
    d = df.copy(); d.columns = [c.upper() for c in d.columns]
    con.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    exists = con.execute(f"SELECT COUNT(*) FROM EXA_ALL_TABLES WHERE TABLE_SCHEMA='{schema}' AND TABLE_NAME='{table.upper()}'").fetchone()[0]
    if exists and not replace:
        cols = [r[0] for r in con.execute(f"SELECT COLUMN_NAME FROM EXA_ALL_COLUMNS WHERE COLUMN_SCHEMA='{schema}' AND COLUMN_TABLE='{table.upper()}' ORDER BY COLUMN_ORDINAL_POSITION").fetchall()]
        d = d.rename(columns=lambda c: c if c in cols or (c + "_X") not in cols else c + "_X")
        if cols != list(d.columns):
            raise RuntimeError(f"{schema}.{table} already exists with columns {cols} but the new data has {list(d.columns)}. "
                               f'Drop the old table and rerun:  DROP TABLE "{schema}"."{table.upper()}"')
    else:
        for _ in range(12):
            try:
                con.execute(f'OPEN SCHEMA "{schema}"'); con.execute(ddl(d, table, wide_text=not replace)); break
            except Exception as e:
                m = re.search(r"(\w+) not allowed as column name", str(e))
                if not m or m.group(1).upper() not in d.columns:
                    raise
                d = d.rename(columns={m.group(1).upper(): m.group(1).upper() + "_X"})
    con.import_from_pandas(d, (schema, table.upper()))


# ---------------------------------------------------------------- features
NUM = ["temp_c", "hr", "rr", "spo2", "sbp", "dbp", "acuity", "news2_triage", "anchor_age", "male", "ambulance", "n_hist_dx"]


def build_frame(src: Source) -> pd.DataFrame:
    T = {t: src.read(t) for t in NEEDED}
    tr, es, pts = T["triage"], T["edstays"], T["patients"]
    d = tr.merge(es[["stay_id", "subject_id", "gender", "arrival_transport"]], on=["stay_id", "subject_id"]).merge(pts[["subject_id", "anchor_age"]], on="subject_id")
    d = d.reset_index(drop=True)
    n2 = S.news2_from_triage(tr.set_index(tr.index)).reset_index(drop=True)
    n2 = n2.set_index(tr.stay_id.values)
    t_raw = pd.to_numeric(d.temperature, errors="coerce")
    d["temp_c"] = np.where((t_raw >= 25) & (t_raw <= 45), t_raw, (t_raw - 32) * 5 / 9)        # Celsius found in the F column
    for c, nm in [("heartrate", "hr"), ("resprate", "rr"), ("o2sat", "spo2"), ("sbp", "sbp"), ("dbp", "dbp")]:
        d[nm] = S._clean(nm, d[c])                                                          # implausible -> NaN
    d["news2_triage"] = d.stay_id.map(n2.news2)
    d["male"] = (d.gender == "M").astype(float); d["ambulance"] = (d.arrival_transport == "AMBULANCE").astype(float)
    d["cc"] = d.chiefcomplaint.fillna("").astype(str).str.lower()
    d["n_hist_dx"] = d.subject_id.map(T["diagnoses_icd"].groupby("subject_id").size()).fillna(0)   # history: chronic dx on file
    lab = T["truth_ed"][["ed_stay_id", "dx_group"]].rename(columns={"ed_stay_id": "stay_id"})
    return d.merge(lab, on="stay_id")


class Model:
    def __init__(self):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)

    def _X(self, d, fit=False):
        num = d[NUM].astype(float)
        if fit:
            self.med = num.median(); f = num.fillna(self.med); self.mu, self.sd = f.mean(), f.std().replace(0, 1)
        z = ((num.fillna(self.med) - self.mu) / self.sd).values
        txt = self.vec.fit_transform(d.cc) if fit else self.vec.transform(d.cc)
        return hstack([csr_matrix(np.hstack([z, num.isna().astype(float).values])), txt]).tocsr()

    def fit(self, d):
        X, y = self._X(d, True), d.dx_group
        if MODEL_KIND == "mlp":
            base = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-3, early_stopping=True, max_iter=300, random_state=0)
            k = int(min(3, y.value_counts().min()))
            # MLPs are over-confident -> calibrate (sigmoid, cross-validated); too few examples of a class -> train uncalibrated
            self.clf = (CalibratedClassifierCV(base, method="sigmoid", cv=k) if k >= 2 else base).fit(X, y)
        else:
            # logistic regression is trained on log-loss, so its probabilities are already reasonable; ECE is reported below
            self.clf = LogisticRegression(max_iter=600, C=2.0).fit(X, y)
        self.classes_ = self.clf.classes_
        return self

    def predict(self, d):
        p = self.clf.predict_proba(self._X(d)); cls = self.clf.classes_
        s = np.sort(p, 1); conf, margin = s[:, -1], s[:, -1] - s[:, -2]
        return cls[p.argmax(1)], conf, margin, p


def route(m, d):
    pred, conf, margin, p = m.predict(d)
    auto = (conf >= TAU) & (margin >= MARGIN)
    return pred, conf, margin, auto


def predictions_table(m, d, version):
    """One row per visit: routing + top-3 guesses + the evidence the doctor/patient pages show."""
    pred, conf, margin, auto = route(m, d)
    _, _, _, P = m.predict(d); cls = np.array(m.classes_); top = np.argsort(-P, 1)[:, :3]
    out = dict(visit_id=d.stay_id.astype(int).values, model_version=f"v{version}", predicted_group=pred, confidence=conf.round(4),
               margin=margin.round(4), route=np.where(auto, "PATIENT_VIEW", "DOCTOR_QUEUE"))
    for k in range(3):
        out[f"top{k+1}_group"] = cls[top[:, k]]; out[f"top{k+1}_prob"] = P[np.arange(len(d)), top[:, k]].round(3)
    if "cc_text" in d.columns:
        out["chief_complaint"] = d["cc_text"].values
    for c in ["cc", "anchor_age", "male", "temp_c", "hr", "rr", "spo2", "sbp", "dbp", "acuity"]:
        if c in d.columns:
            if not (c == "cc" and "cc_text" in d.columns):
                out[{"cc": "chief_complaint"}.get(c, c)] = d[c].values
    return pd.DataFrame(out)


def report(m, d):
    pred, conf, margin, auto = route(m, d); ok = pred == d.dx_group.values
    _, _, _, P = m.predict(d); conf_all = P.max(1)
    ece = float(sum((conf_all[(conf_all > lo) & (conf_all <= lo + .1)].size / len(d)) * abs(ok[(conf_all > lo) & (conf_all <= lo + .1)].mean() - conf_all[(conf_all > lo) & (conf_all <= lo + .1)].mean())
                    for lo in np.arange(0, 1, .1) if ((conf_all > lo) & (conf_all <= lo + .1)).any()))
    return dict(n=len(d), ece=round(ece, 3), accuracy=round(float(ok.mean()), 3), auto_share=round(float(auto.mean()), 3),
                auto_accuracy=round(float(ok[auto].mean()), 3) if auto.any() else None, wrong_shown_to_patient=int((auto & ~ok).sum()),
                sent_to_doctor=int((~auto).sum()))


# ---------------------------------------------------------------- run
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["exasol", "csv"], default="exasol")
    ap.add_argument("--schema", default="CAREGUARD_A"); ap.add_argument("--shift-schema", default=None)
    ap.add_argument("--dir"); ap.add_argument("--shift-dir"); ap.add_argument("--out-schema", default="CAREGUARD_RESULTS")
    ap.add_argument("--rounds", type=int, default=5); ap.add_argument("--model", choices=["logreg", "mlp"], default="logreg")
    a = ap.parse_args()
    global MODEL_KIND
    MODEL_KIND = a.model
    A = Source(a.source, a.schema if a.source == "exasol" else a.dir)
    SH = Source(a.source, a.shift_schema if a.source == "exasol" else a.shift_dir) if (a.shift_schema or a.shift_dir) else None
    dA = build_frame(A)
    rng = np.random.default_rng(0); idx = rng.permutation(len(dA)); k = int(.8 * len(dA))
    train, hold = dA.iloc[idx[:k]].reset_index(drop=True), dA.iloc[idx[k:]].reset_index(drop=True)
    os.makedirs("models", exist_ok=True)
    m = Model().fit(train); version = 0; joblib.dump(m, "models/stage1_v0.joblib")
    out_con = A.con if a.source == "exasol" else None
    rounds, feedback, preds = [], [], []
    base = dict(round=0, model_version="v0", labels_from_doctors=0, hospital_A_holdout=report(m, hold))
    if SH is None:
        rounds.append(base); print(json.dumps(base, indent=1)); live = None
    else:
        dS = build_frame(SH); i2 = rng.permutation(len(dS)); h = len(dS) // 2
        pool, test = dS.iloc[i2[:h]].reset_index(drop=True), dS.iloc[i2[h:]].reset_index(drop=True)
        base["new_hospital_test"] = report(m, test); rounds.append(base)
        print("round 0 (no doctor labels):", json.dumps(base["new_hospital_test"]))
        per = max(20, len(pool) // (a.rounds * 2)); queue = pool.copy(); cur = pd.concat([train], ignore_index=True); asked = 0
        for r in range(1, a.rounds + 1):
            pred, conf, margin, auto = route(m, queue)
            order = np.argsort(conf); low = order[:per]                                      # doctor sees least confident first
            conf_idx = np.setdiff1d(np.where(auto)[0], low)
            n_aud = int(np.ceil(AUDIT_RATE * len(conf_idx))) if len(conf_idx) else 0         # 5% random audit of CONFIDENT answers
            aud = rng.choice(conf_idx, size=n_aud, replace=False) if n_aud else np.array([], int)
            sel = np.concatenate([low, aud]).astype(int); reason = ["low_confidence"] * len(low) + ["audit_confident"] * len(aud)
            take = queue.iloc[sel]
            for j, rs, row in zip(sel, reason, take.itertuples()):
                feedback.append(dict(visit_id=int(row.stay_id), round=r, reason=rs, model_label=str(pred[j]), model_confidence=round(float(conf[j]), 4),
                                     doctor_label=str(row.dx_group), model_was_right=int(pred[j] == row.dx_group), ts=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            cur = pd.concat([cur, take], ignore_index=True); queue = queue.drop(queue.index[sel]).reset_index(drop=True); asked += len(take)
            cand = Model().fit(cur)
            old, new = report(m, hold)["accuracy"], report(cand, hold)["accuracy"]
            promoted = new >= old - 0.02                                                      # gate: old-hospital holdout may not drop > 2 points (small holdout is noisy)
            if promoted:
                version += 1; m = cand; joblib.dump(m, f"models/stage1_v{version}.joblib")
            row = dict(round=r, model_version=f"v{version}", labels_from_doctors=asked, promoted=bool(promoted), gate_old_new=[old, new],
                       hospital_A_holdout=report(m, hold), new_hospital_test=report(m, test))
            fb = pd.DataFrame(feedback); au = fb[(fb['round'] == r) & (fb.reason == 'audit_confident')]
            row['audit_cases'] = int(len(au)); row['audit_error_rate'] = round(float(1 - au.model_was_right.mean()), 3) if len(au) else None
            rounds.append(row)
            print(f"round {r}: audit_errors={row['audit_error_rate']}/{row['audit_cases']} labels={asked} promoted={promoted} new-hospital {json.dumps(row['new_hospital_test'])}")
        # final per-visit predictions on the new-hospital test set (what the app/doctor queue reads)
        preds = predictions_table(m, test, version)
    json.dump(rounds, open("stage1_rounds.json", "w"), indent=1, default=str)
    flat = pd.DataFrame([dict(round=r["round"], model_version=r["model_version"], labels_from_doctors=r["labels_from_doctors"],
                              accuracy_new_hospital=(r.get("new_hospital_test") or {}).get("accuracy"), auto_share_new_hospital=(r.get("new_hospital_test") or {}).get("auto_share"),
                              wrong_shown_new_hospital=(r.get("new_hospital_test") or {}).get("wrong_shown_to_patient"), accuracy_hospital_A=r["hospital_A_holdout"]["accuracy"]) for r in rounds])
    print(flat.to_string(index=False)); flat.to_csv("stage1_rounds.csv", index=False)
    if len(feedback):
        pd.DataFrame(feedback).to_csv("doctor_feedback.csv", index=False); preds.to_csv("stage1_predictions.csv", index=False)
    if out_con is not None:
        write_exasol(out_con, a.out_schema, "ST1_ROUNDS", flat, replace=True)
        if len(feedback):
            write_exasol(out_con, a.out_schema, "DOCTOR_FEEDBACK", pd.DataFrame(feedback), replace=True); write_exasol(out_con, a.out_schema, "ST1_PREDICTIONS", preds, replace=True)
        print(f"results written to Exasol schema {a.out_schema}")


if __name__ == "__main__":
    main()
