"""Cost, compute and latency: CliniCore's small models vs calling a large language model.

    python scripts/cost_comparison.py   -> results/cost_comparison.csv, docs/img/cost_comparison.png

Everything that is an assumption is in ASSUME and printed with the results.
Prices: official list prices per 1M tokens, checked 14 Nov 2026
  (platform.claude.com/docs/en/about-claude/pricing, developers.openai.com/api/docs/pricing).
Our latency: measured on a 2-core cloud CPU (Intel Xeon 2.1 GHz) with the saved models (see measure()).
"""
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

ASSUME = dict(
    ed_visits_per_day=150,          # one mid-size emergency department
    icu_beds=20,                    # scored every hour
    ed_tokens_in=300, ed_tokens_out=150,     # triage form + 17-group list + instructions; answer with top 3 + reason
    icu_tokens_in=800, icu_tokens_out=300,   # 24 h of hourly vitals/labs (~2,300 characters); 4 risks + score + evidence
    llm_first_token_s=0.67, llm_tokens_per_s=83,   # example public API speed (GPT-5.1, pricepertoken.com)
    open_llm_params=70e9,           # reference open-weight model size for compute (closed models do not publish sizes)
)

# USD per 1M tokens (input, output), standard rate, short context
PRICES = [
    ("Claude Fable 5.2", 8.0, 40.0), ("GPT-6 Astra", 10.0, 50.0), ("Claude Opus 5.5", 4.0, 20.0),
    ("Claude Sonnet 5.5", 2.0, 10.0), ("GPT-6 Sol", 2.0, 10.0), ("Claude Haiku 5.5", 0.10, 0.50), ("GPT-6 Luna", 0.10, 0.50),
]


def measure():
    """Time our saved models on this machine."""
    import sys
    import joblib
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(ROOT / "pipeline"))
    m = joblib.load(ROOT / "models/stage1_nhamcs.joblib")
    d = pd.read_csv(ROOT / "nhamcs_cache/nhamcs_ed.csv.gz")
    te = d[d.year == 2022].head(300).copy(); te["cc"] = te["cc"].fillna("")
    m.predict(te.head(5))
    t = time.perf_counter()
    for i in range(200):
        m.predict(te.iloc[[i]])
    ed_ms = (time.perf_counter() - t) / 200 * 1e3
    h = joblib.load(ROOT / "models/stage3_heads.joblib")
    X = pd.DataFrame(np.random.randn(1, len(h["columns"])), columns=h["columns"])
    for mod in h["models"].values():
        mod.predict_proba(X)
    t = time.perf_counter()
    for _ in range(100):
        for mod in h["models"].values():
            mod.predict_proba(X)
    icu_ms = (time.perf_counter() - t) / 100 * 1e3
    return ed_ms, icu_ms


def main():
    a = ASSUME
    ed_year, icu_year = a["ed_visits_per_day"] * 365, a["icu_beds"] * 24 * 365
    ed_ms, icu_ms = measure()
    rows = []
    for name, pin, pout in PRICES:
        ed_call = (a["ed_tokens_in"] * pin + a["ed_tokens_out"] * pout) / 1e6
        icu_call = (a["icu_tokens_in"] * pin + a["icu_tokens_out"] * pout) / 1e6
        rows.append(dict(model=name, usd_per_ed_case=ed_call, usd_per_icu_check=icu_call,
                         usd_per_year=ed_call * ed_year + icu_call * icu_year,
                         ed_latency_s=a["llm_first_token_s"] + a["ed_tokens_out"] / a["llm_tokens_per_s"],
                         icu_latency_s=a["llm_first_token_s"] + a["icu_tokens_out"] / a["llm_tokens_per_s"]))
    # CliniCore: runs on the laptop that already runs the app; CPU time per year
    cpu_s_year = ed_year * ed_ms / 1e3 + icu_year * icu_ms / 1e3
    kwh = cpu_s_year * 30 / 3.6e6          # ~30 W extra CPU power while computing
    rows.append(dict(model="CliniCore (logistic regression + LightGBM)", usd_per_ed_case=0.0, usd_per_icu_check=0.0,
                     usd_per_year=kwh * 0.40, ed_latency_s=ed_ms / 1e3, icu_latency_s=icu_ms / 1e3))

    flops = dict(
        cc_ed=2 * 18035 * 17,                                  # dense upper bound; real input is sparse
        cc_icu=4 * 300 * 14,                                   # tree comparisons, 4 risks x 300 trees x <= 14 splits
        llm_ed=2 * a["open_llm_params"] * (a["ed_tokens_in"] + a["ed_tokens_out"]),
        llm_icu=2 * a["open_llm_params"] * (a["icu_tokens_in"] + a["icu_tokens_out"]),
    )

    out = ROOT / "results" / "cost_comparison.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"workload: {ed_year:,} ED cases + {icu_year:,} ICU checks per year; assumptions: {a}")
    print(f"CliniCore measured: {ed_ms:.2f} ms per ED case, {icu_ms:.2f} ms per ICU check (4 risks); CPU time/year {cpu_s_year:.0f} s")
    for r in rows:
        print(f"  {r['model']:<45} ${r['usd_per_ed_case']:.5f}/ED  ${r['usd_per_icu_check']:.5f}/ICU  ${r['usd_per_year']:>9,.2f}/year  "
              f"{r['ed_latency_s']*1e3:>8.1f} ms ED  {r['icu_latency_s']*1e3:>8.1f} ms ICU")
    print("compute per prediction (FLOPs/operations):", {k: f"{v:.2e}" for k, v in flops.items()},
          f"ratio ED {flops['llm_ed']/flops['cc_ed']:.1e}, ICU {flops['llm_icu']/flops['cc_icu']:.1e}")
    chart(rows, flops, ed_year, icu_year)


def chart(rows, flops, ed_year, icu_year):
    BG, GRID, TXT, SUB, MUTED, OURS, OTHER = "#0b1220", "#253049", "#e6ecf5", "#9aa7bd", "#5d6b85", "#2dd4bf", "#64748b"
    plt.rcParams.update({"font.size": 12, "text.color": TXT, "xtick.color": SUB, "ytick.color": TXT})
    fig = plt.figure(figsize=(16, 8.8), dpi=110, facecolor=BG)
    fig.text(0.035, 0.94, "Cost and speed: CliniCore vs asking a large language model", fontsize=22, weight="bold")
    fig.text(0.035, 0.895, f"One hospital for a year: {ed_year:,} ED cases + {icu_year:,} hourly ICU checks (20 beds). "
             "LLM list prices, Nov 2026; token counts estimated.", fontsize=12.5, color=SUB)

    def style(ax):
        ax.set_facecolor(BG)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID); ax.tick_params(axis="y", length=0, pad=10)
        ax.grid(axis="x", color=GRID, lw=0.8, which="major"); ax.set_axisbelow(True); ax.set_xscale("log")

    order = sorted(rows, key=lambda r: -r["usd_per_year"])
    labels = [("CliniCore" if r["model"].startswith("CliniCore") else r["model"]) for r in order]
    ours = [r["model"].startswith("CliniCore") for r in order]
    ys = list(range(len(order)))[::-1]

    ax = fig.add_axes([0.16, 0.27, 0.33, 0.55]); style(ax)
    for y, r, o in zip(ys, order, ours):
        v = max(r["usd_per_year"], 0.01)
        ax.barh(y, v, height=0.56, color=OURS if o else OTHER, zorder=3, left=0.005)
        ax.text(v * 2.2, y, "< $0.01 (laptop electricity)" if o else f"${r['usd_per_year']:,.0f}", va="center", fontsize=11.5,
                weight="bold" if o else None)
    ax.set_yticks(ys); ax.set_yticklabels(labels)
    for t, o in zip(ax.get_yticklabels(), ours):
        if o: t.set_color(OURS); t.set_weight("bold")
    ax.set_xlim(0.005, 3e5); ax.set_xlabel("US dollars per year (log scale)", color=SUB)
    ax.set_xticks([0.01, 1, 100, 1e4]); ax.set_xticklabels(["$0.01", "$1", "$100", "$10,000"])
    ax.set_title("Running cost per year", loc="left", fontsize=15, weight="bold", color=TXT, pad=12, x=-0.36)

    ax = fig.add_axes([0.64, 0.27, 0.33, 0.55]); style(ax)
    lat = sorted([("CliniCore · ICU check", rows[-1]["icu_latency_s"], True), ("CliniCore · ED case", rows[-1]["ed_latency_s"], True),
                  ("LLM API · ED case", order[0]["ed_latency_s"], False), ("LLM API · ICU check", order[0]["icu_latency_s"], False)],
                 key=lambda t: -t[1])
    ys2 = list(range(len(lat)))[::-1]
    for y, (lab, s, o) in zip(ys2, lat):
        ax.barh(y, s, height=0.5, color=OURS if o else OTHER, zorder=3, left=1e-4)
        ax.text(s * 1.3, y, f"{s*1e3:.1f} ms" if s < 1 else f"≈ {s:.1f} s", va="center", fontsize=11.5, weight="bold" if o else None)
    ax.set_yticks(ys2); ax.set_yticklabels([t[0] for t in lat])
    for t, l in zip(ax.get_yticklabels(), lat):
        if l[2]: t.set_color(OURS); t.set_weight("bold")
    ax.set_xlim(1e-4, 60); ax.set_xlabel("time per prediction (log scale)", color=SUB)
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1, 10]); ax.set_xticklabels(["1 ms", "10 ms", "0.1 s", "1 s", "10 s"])
    ax.set_title("Time to answer", loc="left", fontsize=15, weight="bold", color=TXT, pad=12, x=-0.36)
    ax.text(-0.36, -0.2, f"Compute per prediction: CliniCore ≤ {flops['cc_ed']:.0e} operations; a 70-billion-parameter LLM ≈ {flops['llm_ed']:.0e} FLOPs "
            f"(~{flops['llm_ed']/flops['cc_ed']:.0e}× more).\nCliniCore measured on a 2-core CPU; LLM time = first token 0.67 s + 83 tokens/s (public API example).",
            transform=ax.transAxes, fontsize=9.8, color=SUB, va="top") if False else None
    fig.text(0.035, 0.115, f"Compute per prediction: CliniCore ≤ {flops['cc_ed']:.0e} operations; a 70-billion-parameter open LLM ≈ {flops['llm_ed']:.0e} FLOPs (about {flops['llm_ed']/flops['cc_ed']:.0e}× more) and two 80 GB GPUs to host.\nCliniCore measured on a 2-core CPU; LLM time = first token 0.67 s + 83 tokens/s (public API example).", fontsize=10.5, color=SUB, va="top")
    fig.text(0.035, 0.035, "Cost only, not accuracy: LLMs were not tested on these tasks. Prices: platform.claude.com and developers.openai.com list prices "
             "(standard, no batch or caching discounts). Script: scripts/cost_comparison.py", fontsize=9.5, color=MUTED)
    fig.savefig(ROOT / "docs/img/cost_comparison.png", facecolor=BG)
    print("wrote", ROOT / "docs/img/cost_comparison.png")


if __name__ == "__main__":
    main()
