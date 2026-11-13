#!/usr/bin/env python3
"""Compare LightGBM, GRU-D and GRU-D+LightGBM on PhysioNet/CinC 2019 Sepsis. Same patients, same metrics for all three.
Train on hospital system A (80% train / 20% validation: early stopping + alert threshold), test on hospital system B (never touched in training).

  python sepsis_benchmark.py --source exasol            # table CAREGUARD_SEPSIS19.SEPSIS2019 (made by sepsis2019_load.py)
  python sepsis_benchmark.py --source csv               # sepsis2019_cache/sepsis2019.csv.gz
  python sepsis_benchmark.py --quick                    # 2000 patients, 2 epochs: a 5-minute trial
  python sepsis_benchmark.py --skip grud                # LightGBM only (no PyTorch needed)

Metrics (test hospital B): AUPRC and AUROC over all patient-hours; PhysioNet utility (re-implemented, normalised: 0 = never alert, 1 = ideal) at the threshold that is best
on validation; share of septic patients alerted in the useful window; false-alert hours per 24 patient-hours. 95% intervals come from resampling patients."""
from __future__ import annotations
import argparse, json, os, sys, time, warnings
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VIT = ["hr", "o2sat", "temp", "sbp", "map", "dbp", "resp", "etco2"]
LAB = ["baseexcess", "hco3", "fio2", "ph", "paco2", "sao2", "ast", "bun", "alkalinephos", "calcium", "chloride", "creatinine", "bilirubin_direct", "glucose", "lactate", "magnesium",
       "phosphate", "potassium", "bilirubin_total", "troponini", "hct", "hgb", "ptt", "wbc", "fibrinogen", "platelets"]
V = VIT + LAB                                        # 34 time-varying variables
STAT = ["age", "gender", "unit1", "unit2", "hospadmtime"]


# ------------------------------------------------------------------ data
def load(a):
    if a.source == "exasol":
        from stage1_pipeline import Source
        con = Source("exasol", a.schema).con
        d = con.export_to_pandas(f'SELECT * FROM "{a.schema}"."SEPSIS2019"')
    else:
        d = pd.read_csv(os.path.join(a.cache, "sepsis2019.csv.gz"))
    d.columns = [c.lower() for c in d.columns]
    d["patient_id"] = d.patient_id.astype(str)
    d = d.sort_values(["patient_id", "iculos"]).reset_index(drop=True)
    for c in V + STAT + ["iculos", "sepsislabel"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def split(d, a):
    rng = np.random.RandomState(a.seed)
    pid = d.drop_duplicates("patient_id")[["patient_id", "set_name"]]
    if set(pid.set_name) >= {"A", "B"}:
        A, B = pid[pid.set_name == "A"].patient_id.values, pid[pid.set_name == "B"].patient_id.values; shifted = True
    else:
        print("WARNING: no hospital-system labels (A/B) in this copy -> random 50/50 patient split. Results are NOT a hospital-shift test.")
        p = pid.patient_id.values.copy(); rng.shuffle(p); A, B = p[: len(p) // 2], p[len(p) // 2:]; shifted = False
    if a.quick:
        A, B = rng.permutation(A)[:1400], rng.permutation(B)[:600]
    A = rng.permutation(A); k = int(0.8 * len(A))
    return A[:k], A[k:], B, shifted


# ------------------------------------------------------------------ LightGBM features (hourly, per patient)
def features(d):
    g = d.groupby("patient_id", sort=False)
    X = pd.DataFrame(index=d.index)
    for c in V:
        X[c] = g[c].ffill()
        last = d.iculos.where(d[c].notna()); X[c + "_age_h"] = d.iculos - last.groupby(d.patient_id).ffill()
    for c in VIT:
        r = X.groupby(d.patient_id, sort=False)[c].rolling(6, min_periods=1)
        X[c + "_min6"] = r.min().reset_index(level=0, drop=True); X[c + "_max6"] = r.max().reset_index(level=0, drop=True); X[c + "_mean6"] = r.mean().reset_index(level=0, drop=True)
        X[c + "_d6"] = X[c] - X.groupby(d.patient_id, sort=False)[c].shift(6)
    for c in STAT + ["iculos"]:
        X[c] = d[c]
    X["shock_index"] = X.hr / X.sbp
    X["sirs"] = (X.temp.gt(38) | X.temp.lt(36)).astype(int) + X.hr.gt(90).astype(int) + X.resp.gt(20).astype(int) + (X.wbc.gt(12) | X.wbc.lt(4)).astype(int)
    X["qsofa"] = X.resp.ge(22).astype(int) + X.sbp.le(100).astype(int)
    return X.astype("float32")


def lgbm_fit(Xtr, ytr, Xva, yva, seed):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                           reg_lambda=2.0, n_estimators=600, random_state=seed, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision", callbacks=[lgb.early_stopping(40, verbose=False)])
    return m


# ------------------------------------------------------------------ GRU-D (Che et al. 2018), written out explicitly
def tensors(d, pids, stats):
    """Per patient: raw values (NaN = missing), observation mask, hours since last observation, last observed value, statics."""
    mu, sd, smu, ssd = stats
    out = []
    for pid, g in d[d.patient_id.isin(set(pids))].groupby("patient_id", sort=False):
        x = g[V].values.astype(np.float32); m = ~np.isnan(x); T = len(x)
        z = (x - mu) / sd
        delta = np.zeros_like(z); last = np.zeros_like(z); seen = np.zeros(z.shape[1], bool); lv = np.zeros(z.shape[1], np.float32); dl = np.zeros(z.shape[1], np.float32)
        for t in range(T):
            if t > 0:
                dl = dl + 1
            delta[t] = np.where(seen, dl, 0); last[t] = lv
            obs = m[t]; lv = np.where(obs, z[t], lv); dl = np.where(obs, 0, dl); seen = seen | obs
        s = ((g[STAT].iloc[0].values.astype(np.float32) - smu) / ssd); s = np.nan_to_num(s)
        out.append(dict(pid=pid, x=np.nan_to_num(z).astype(np.float32), m=m.astype(np.float32), delta=np.minimum(delta, 48) / 12.0, last=last.astype(np.float32), s=s.astype(np.float32),
                        y=g.sepsislabel.fillna(0).values.astype(np.float32)))
    return out


def norm_stats(d, pids):
    t = d[d.patient_id.isin(set(pids))]
    mu = t[V].mean().values.astype(np.float32); sd = t[V].std().replace(0, 1).fillna(1).values.astype(np.float32)
    smu = t[STAT].mean().values.astype(np.float32); ssd = t[STAT].std().replace(0, 1).fillna(1).values.astype(np.float32)
    return np.nan_to_num(mu), np.nan_to_num(sd, nan=1.0), np.nan_to_num(smu), np.nan_to_num(ssd, nan=1.0)


def batches(items, bs, shuffle, rng, cap=None):
    idx = np.argsort([len(i["y"]) for i in items])
    chunks = [idx[i:i + bs] for i in range(0, len(idx), bs)]
    if shuffle:
        rng.shuffle(chunks)
    for ch in chunks:
        yield ch, collate([items[i] for i in ch], cap)


def collate(b, cap):
    import torch
    T = max(len(i["y"]) for i in b); T = min(T, cap) if cap else T
    def pad(k, shape):
        a = np.zeros((len(b), T) + shape, np.float32)
        for j, i in enumerate(b):
            n = min(len(i["y"]), T); a[j, :n] = i[k][:n]
        return torch.from_numpy(a)
    D = len(V)
    mask = np.zeros((len(b), T), np.float32)
    for j, i in enumerate(b):
        mask[j, :min(len(i["y"]), T)] = 1
    return dict(x=pad("x", (D,)), m=pad("m", (D,)), delta=pad("delta", (D,)), last=pad("last", (D,)), y=pad("y", ()), valid=torch.from_numpy(mask),
                s=torch.from_numpy(np.stack([i["s"] for i in b])))


def make_grud(hidden, nstat):
    import torch, torch.nn as nn
    D = len(V)

    class GRUD(nn.Module):
        def __init__(s):
            super().__init__()
            s.gx = nn.Parameter(torch.zeros(D)); s.bx = nn.Parameter(torch.zeros(D))            # input decay (per variable)
            s.gh = nn.Linear(D, hidden)                                                          # hidden-state decay
            s.cell = nn.GRUCell(2 * D + nstat, hidden); s.out = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1)); s.hidden = hidden

        def forward(s, x, m, delta, last, st):
            B, T, _ = x.shape; h = x.new_zeros(B, s.hidden); outs = []
            for t in range(T):
                gam_x = torch.exp(-torch.relu(s.gx * delta[:, t] + s.bx))
                xh = m[:, t] * x[:, t] + (1 - m[:, t]) * gam_x * last[:, t]                     # empirical mean is 0 after z-scoring
                h = h * torch.exp(-torch.relu(s.gh(delta[:, t])))
                h = s.cell(torch.cat([xh, m[:, t], st], 1), h)
                outs.append(s.out(h).squeeze(1))
            return torch.stack(outs, 1)
    return GRUD()


def grud_train(tr, va, a, seed, tag):
    import torch
    torch.manual_seed(seed); rng = np.random.RandomState(seed); torch.set_num_threads(max(1, os.cpu_count() or 1))
    net = make_grud(a.hidden, len(STAT)); opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    pw = torch.tensor(float(a.pos_weight)); best, best_state = -1, None
    for ep in range(a.epochs):
        net.train(); t0 = time.time(); tot = 0
        for _, b in batches(tr, a.bs, True, rng, cap=a.train_cap):
            opt.zero_grad(); lo = net(b["x"], b["m"], b["delta"], b["last"], b["s"])
            l = torch.nn.functional.binary_cross_entropy_with_logits(lo, b["y"], pos_weight=pw, reduction="none"); l = (l * b["valid"]).sum() / b["valid"].sum()
            l.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); tot += float(l)
        pv = grud_predict(net, va, a); yv = np.concatenate([i["y"] for i in va]); ap = average_precision_score(yv, pv) if yv.sum() else 0
        print(f"  [{tag}] epoch {ep+1}/{a.epochs} loss {tot:.2f} val AUPRC {ap:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if ap > best:
            best = ap; best_state = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state); return net


def grud_predict(net, items, a):
    import torch
    net.eval(); res = [None] * len(items)
    with torch.no_grad():
        for ch, b in batches(items, 128, False, None):
            p = torch.sigmoid(net(b["x"], b["m"], b["delta"], b["last"], b["s"])).numpy()
            for j, i in enumerate(ch):
                res[i] = p[j, :len(items[i]["y"])]
    return np.concatenate(res) if len(res) else np.array([])


def put(out, rows, items, vals):
    """Write per-hour predictions `vals` (concatenated in the order of `items`) back to the rows of the big table."""
    off = 0
    for it in items:
        r = rows[it["pid"]]; out[r] = vals[off:off + len(r)]; off += len(r)


# ------------------------------------------------------------------ evaluation
def onset_table(d):
    first = d[d.sepsislabel == 1].groupby("patient_id").iculos.min() + 6                   # label starts 6 h before onset
    return d.patient_id.map(first)


def utility_parts(t_rel, pred):
    """Per-row PhysioNet utility pieces. t_rel = hours relative to onset (NaN for non-septic). Returns obs, inaction, best arrays."""
    dt_e, dt_o, dt_l, fp, fn_min = -12, -6, 3, -0.05, -2.0
    sep = ~np.isnan(t_rel); t = np.where(sep, t_rel, 0)
    tp = np.where(t < dt_e, 0, np.where(t <= dt_o, (t - dt_e) / (dt_o - dt_e), np.where(t <= dt_l, 1 - (t - dt_o) / (dt_l - dt_o), 0)))
    fn = np.where(t <= dt_o, 0, np.where(t <= dt_l, fn_min * (t - dt_o) / (dt_l - dt_o), fn_min))
    obs = np.where(sep, np.where(pred, np.where(t >= dt_e, tp, fp), np.where(t >= dt_o, fn, 0)), np.where(pred, fp, 0))
    inact = np.where(sep & (t >= dt_o), fn, 0); best = np.where(sep & (t >= dt_e), tp, 0)
    return obs, inact, best


def norm_util(obs, inact, best):
    return (obs.sum() - inact.sum()) / (best.sum() - inact.sum()) if best.sum() != inact.sum() else np.nan


def pick_threshold(score, t_rel, grid=60):
    cand = np.unique(np.quantile(score, np.linspace(0.80, 0.9995, grid))); best = (-9, cand[-1])
    for c in cand:
        u = norm_util(*utility_parts(t_rel, score >= c))
        if u > best[0]:
            best = (u, c)
    return float(best[1])


def evaluate(score, y, t_rel, pid_codes, thr):
    pred = score >= thr; o, i, b = utility_parts(t_rel, pred)
    n = pid_codes.max() + 1; po, pi, pb = (np.bincount(pid_codes, w, n) for w in (o, i, b))
    sep_p = np.bincount(pid_codes, (~np.isnan(t_rel)).astype(float), n) > 0
    win = (~np.isnan(t_rel)) & (t_rel >= -12) & (t_rel <= 3) & pred
    caught = np.bincount(pid_codes, win.astype(float), n) > 0
    fa = pred & np.isnan(t_rel)
    return dict(auprc=average_precision_score(y, score), auroc=roc_auc_score(y, score), utility=float((po.sum() - pi.sum()) / (pb.sum() - pi.sum())),
                septic_patients_alerted=float(caught[sep_p].mean()), false_alert_hours_per_24h=float(fa.sum() / len(y) * 24), threshold=float(thr)), (po, pi, pb)


def bootstrap(score, y, pid_codes, parts, reps, seed, base=None):
    """Patient-level resampling. Returns 95% intervals for AUPRC and utility (and for the difference to `base` when given)."""
    rng = np.random.RandomState(seed); n = pid_codes.max() + 1
    starts = np.r_[0, np.flatnonzero(np.diff(pid_codes)) + 1]; ends = np.r_[starts[1:], len(pid_codes)]
    po, pi, pb = parts; res = []
    for _ in range(reps):
        pick = rng.randint(0, n, n); idx = np.concatenate([np.arange(starts[p], ends[p]) for p in pick])
        u = (po[pick].sum() - pi[pick].sum()) / (pb[pick].sum() - pi[pick].sum()); ap = average_precision_score(y[idx], score[idx])
        row = [ap, u]
        if base is not None:
            bs, bparts = base; bu = (bparts[0][pick].sum() - bparts[1][pick].sum()) / (bparts[2][pick].sum() - bparts[1][pick].sum())
            row += [ap - average_precision_score(y[idx], bs[idx]), u - bu]
        res.append(row)
    r = np.array(res); q = lambda c: (float(np.percentile(r[:, c], 2.5)), float(np.percentile(r[:, c], 97.5)))
    out = dict(auprc_ci=q(0), utility_ci=q(1))
    if base is not None:
        out.update(auprc_diff_ci=q(2), utility_diff_ci=q(3))
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["exasol", "csv"], default="exasol"); ap.add_argument("--schema", default="CAREGUARD_SEPSIS19"); ap.add_argument("--cache", default="sepsis2019_cache")
    ap.add_argument("--out", default="results"); ap.add_argument("--out-schema", default="CAREGUARD_RESULTS"); ap.add_argument("--no-exasol-write", action="store_true")
    ap.add_argument("--skip", default="", help="e.g. grud to run LightGBM only"); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--hidden", type=int, default=64); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--train-cap", type=int, default=168, help="max hours per patient used while training GRU-D"); ap.add_argument("--pos-weight", type=float, default=8.0)
    ap.add_argument("--boot", type=int, default=100); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.quick:
        a.epochs, a.boot = min(a.epochs, 2), min(a.boot, 30)
    os.makedirs(a.out, exist_ok=True); os.makedirs("models", exist_ok=True); t0 = time.time()
    d = load(a); trn, val, tst, shifted = split(d, a)
    d = d[d.patient_id.isin(set(trn) | set(val) | set(tst))].reset_index(drop=True)
    print(f"[1/5] data: train {len(trn)} / validation {len(val)} (hospital A), test {len(tst)} (hospital {'B' if shifted else 'random half'}); rows {len(d)}; sepsis patients {d[d.sepsislabel==1].patient_id.nunique()}", flush=True)
    sel = lambda P: d.patient_id.isin(set(P)).values
    mtr, mva, mte = sel(trn), sel(val), sel(tst)
    X = features(d); print(f"[2/5] LightGBM features {X.shape[1]} ({time.time()-t0:.0f}s)", flush=True)
    y = d.sepsislabel.fillna(0).values; onset = onset_table(d); t_rel = (d.iculos - onset).values
    scores = {}
    lg = lgbm_fit(X[mtr], y[mtr], X[mva], y[mva], a.seed); scores["LightGBM"] = lg.predict_proba(X)[:, 1]
    print(f"[3/5] LightGBM done: {lg.best_iteration_} trees ({time.time()-t0:.0f}s)", flush=True)
    if "grud" not in a.skip:
        try:
            import torch  # noqa
        except Exception as e:
            sys.exit(f"PyTorch is needed for GRU-D ({e}). Install it (pip install torch) or run with --skip grud")
        stats = norm_stats(d, trn); T = {k: tensors(d, P, stats) for k, P in (("tr", trn), ("va", val), ("te", tst))}
        print(f"[4/5] GRU-D: training on {len(T['tr'])} patients ({a.epochs} epochs; the slowest step)", flush=True)
        full = grud_train(T["tr"], T["va"], a, a.seed, "GRU-D"); torch.save(full.state_dict(), "models/grud_sepsis19.pt")
        rows = d.groupby("patient_id", sort=False).indices; g_all = np.zeros(len(d))
        for k in ("tr", "va", "te"):
            put(g_all, rows, T[k], grud_predict(full, T[k], a))
        scores["GRU-D"] = g_all
        # hybrid: out-of-fold GRU-D scores on the training patients (2 folds), mean of the two fold models for validation/test
        rng = np.random.RandomState(a.seed + 1); perm = rng.permutation(len(T["tr"])); f1, f2 = perm[: len(perm) // 2], perm[len(perm) // 2:]
        nets = [grud_train([T["tr"][i] for i in f], T["va"], a, a.seed + 2 + j, f"fold{j+1}") for j, f in enumerate((f1, f2))]
        stack = np.zeros(len(d))
        for j, (f, other) in enumerate(((f1, f2), (f2, f1))):
            part = [T["tr"][i] for i in other]; put(stack, rows, part, grud_predict(nets[j], part, a))       # model j never saw `other`
        for k in ("va", "te"):
            put(stack, rows, T[k], np.mean([grud_predict(n, T[k], a) for n in nets], 0))
        Xh = X.copy(); Xh["grud"] = stack; gg = Xh.groupby(d.patient_id, sort=False).grud
        Xh["grud_max6"] = gg.rolling(6, min_periods=1).max().reset_index(level=0, drop=True); Xh["grud_d6"] = Xh.grud - gg.shift(6)
        lh = lgbm_fit(Xh[mtr], y[mtr], Xh[mva], y[mva], a.seed); scores["GRU-D + LightGBM"] = lh.predict_proba(Xh)[:, 1]
        print(f"      hybrid done ({time.time()-t0:.0f}s)", flush=True)
    # ---- evaluation
    print("[5/5] evaluation on validation (threshold) and test (final numbers)", flush=True)
    codes_te = pd.factorize(d.patient_id[mte])[0]; rows = []; ev = {}
    for name, sc in scores.items():
        thr = pick_threshold(sc[mva], t_rel[mva]); r, parts = evaluate(sc[mte], y[mte], t_rel[mte], codes_te, thr); ev[name] = (sc[mte], parts); r["model"] = name; rows.append(r)
    base = ev["LightGBM"]
    for r in rows:
        sc, parts = ev[r["model"]]; r.update(bootstrap(sc, y[mte], codes_te, parts, a.boot, a.seed, None if r["model"] == "LightGBM" else (base[0], base[1])))
    res = pd.DataFrame(rows)[["model", "auprc", "auprc_ci", "auroc", "utility", "utility_ci", "septic_patients_alerted", "false_alert_hours_per_24h", "threshold"] + [c for c in ("auprc_diff_ci", "utility_diff_ci") if c in rows[-1]]]
    print(res.round(3).to_string(index=False))
    res.to_csv(os.path.join(a.out, "sepsis19_benchmark.csv"), index=False)
    json.dump(dict(shifted_hospital=shifted, n_train=len(trn), n_val=len(val), n_test=len(tst), results=json.loads(res.to_json(orient="records")), seconds=round(time.time() - t0)), open(os.path.join(a.out, "sepsis19_benchmark.json"), "w"), indent=1)
    # decision: the simplest model unless another is clearly better (95% interval of the difference to LightGBM excludes 0 on BOTH AUPRC and utility)
    pick = "LightGBM"
    for r in rows[1:] if rows[0]["model"] == "LightGBM" else []:
        if r["auprc_diff_ci"][0] > 0 and r["utility_diff_ci"][0] > 0 and r["utility"] > [x for x in rows if x["model"] == pick][0]["utility"]:
            pick = r["model"]
    print(f"\nDecision rule: keep LightGBM unless another model beats it with a 95% interval above 0 on both AUPRC and utility.  ->  {pick}")
    if not a.no_exasol_write and a.source == "exasol":
        from stage1_pipeline import Source, write_exasol
        con = Source("exasol", a.schema).con; out = res.copy()
        for c in ("auprc_ci", "utility_ci", "auprc_diff_ci", "utility_diff_ci"):
            if c in out:
                out[c] = out[c].astype(str)
        write_exasol(con, a.out_schema, "SEPSIS19_BENCH", out, replace=True); print(f"written to Exasol {a.out_schema}.SEPSIS19_BENCH")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
