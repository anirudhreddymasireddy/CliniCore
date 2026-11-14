"""Head-to-head on the same patients: CliniCore risk models vs bedside scores.

    python scripts/make_bedside_chart.py  -> docs/img/comparison_bedside.png

Numbers: pipeline/physionet_icu.py on the team laptop (14 Nov 2026), hospital-B
test patients, models trained on hospital A only.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parents[1] / "docs" / "img" / "comparison_bedside.png"
BG, GRID, TXT, SUB, MUTED = "#0b1220", "#253049", "#e6ecf5", "#9aa7bd", "#5d6b85"
OURS, STD, TIER = "#2dd4bf", "#94a3b8", "#475569"

# risk, horizon, model AUROC (B), standard score name, its AUROC, current rule tier AUROC
ROWS = [("Sepsis", "6 h", 0.753, "qSOFA", 0.583, 0.615),
        ("Breathing failure", "6 h", 0.840, "NEWS2", 0.788, 0.613),
        ("Kidney injury", "12 h", 0.909, None, None, 0.589),
        ("Shock", "6 h", 0.939, None, None, 0.501)]

plt.rcParams.update({"font.size": 12, "text.color": TXT, "xtick.color": SUB, "ytick.color": TXT})
fig = plt.figure(figsize=(16, 8.2), dpi=110, facecolor=BG)
fig.text(0.035, 0.935, "Same patients, head to head: CliniCore vs bedside scores", fontsize=22, weight="bold")
fig.text(0.035, 0.89, "Real ICU patients (PhysioNet 2019) · models trained on hospital A · tested on hospital B, never seen in training",
         fontsize=12.5, color=SUB)
ax = fig.add_axes([0.24, 0.2, 0.68, 0.6]); ax.set_facecolor(BG)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.tick_params(axis="y", length=0, pad=12)
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)

ys = list(range(len(ROWS)))[::-1]
for y, (risk, hz, m, sname, s, tier) in zip(ys, ROWS):
    best_base = max(v for v in (s, tier) if v is not None)
    ax.plot([best_base, m], [y, y], color=OURS, lw=3, alpha=0.35, zorder=2, solid_capstyle="round")
    ax.scatter([tier], [y], s=170, color=TIER, edgecolor=BG, lw=2, zorder=3)
    ax.text(tier, y - 0.3, f"{tier:.2f}", ha="center", va="top", fontsize=10.5, color=SUB)
    if s is not None:
        ax.scatter([s], [y], s=170, color=STD, edgecolor=BG, lw=2, zorder=3)
        ax.text(s, y + 0.27, f"{sname} {s:.2f}", ha="center", va="bottom", fontsize=10.5, color=SUB)
    ax.scatter([m], [y], s=260, color=OURS, edgecolor=BG, lw=2, zorder=4)
    ax.text(m + 0.012, y, f"{m:.2f}", va="center", fontsize=13, weight="bold")
    ax.text(0.985, y, f"+{m - best_base:.2f}", va="center", ha="right", fontsize=12, color=OURS,
            transform=ax.get_yaxis_transform())
ax.set_yticks(ys); ax.set_yticklabels([f"{r[0]}  ·  next {r[1]}" for r in ROWS], fontsize=13)
ax.axvline(0.5, color=MUTED, lw=1, ls="--"); ax.text(0.503, 3.55, "chance", color=MUTED, fontsize=10)
ax.set_xlim(0.45, 1.03); ax.set_ylim(-0.7, 3.7)
ax.set_xlabel("AUROC on hospital B (higher is better, 1 = perfect)", color=SUB)
ax.text(0.985, 1.03, "gain over best\nbedside baseline", ha="right", va="bottom", fontsize=10, color=SUB, transform=ax.transAxes)
ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=12, color=OURS, label="CliniCore model (LightGBM)"),
                   Line2D([], [], marker="o", ls="", ms=10, color=STD, label="Standard score (qSOFA / NEWS2)"),
                   Line2D([], [], marker="o", ls="", ms=10, color=TIER, label="Current bedside rule tier")],
          loc="upper left", bbox_to_anchor=(-0.33, -0.12), ncol=3, frameon=False, fontsize=11.5, labelcolor=TXT)
fig.text(0.035, 0.045, "Kidney, breathing and shock outcomes are rule-defined (reaching High within the window): PhysioNet has no vasopressors, urine output or PaO2. "
         "6,000 hospital-B patients.", fontsize=9.5, color=MUTED)
fig.savefig(OUT, facecolor=BG)
print("wrote", OUT)
