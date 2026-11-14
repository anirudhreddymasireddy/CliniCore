"""Where CliniCore stands vs published sepsis models (slide + README chart).

    python scripts/make_comparison_chart.py  -> docs/img/comparison.png

Numbers are from the cited papers and our own hospital-A -> hospital-B test.
Not head-to-head: patients, splits and sepsis definitions differ.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "docs" / "img" / "comparison.png"
BG, PANEL, GRID = "#0b1220", "#131c2e", "#253049"
TXT, SUB, MUTED = "#e6ecf5", "#9aa7bd", "#5d6b85"
OURS, OTHER = "#2dd4bf", "#64748b"

plt.rcParams.update({"font.size": 12, "text.color": TXT, "axes.labelcolor": SUB,
                     "xtick.color": SUB, "ytick.color": TXT})
fig = plt.figure(figsize=(16, 8.8), dpi=110, facecolor=BG)
fig.text(0.035, 0.94, "Where CliniCore stands on ICU sepsis prediction", fontsize=22, weight="bold")
fig.text(0.035, 0.895, "Published results vs ours. Not head-to-head: patients, data splits and sepsis definitions differ.",
         fontsize=12.5, color=SUB)


def style(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)


# ---- left: PhysioNet 2019 utility
ax = fig.add_axes([0.20, 0.25, 0.27, 0.56]); style(ax)
rows = [("Winner · hospital A (seen)", 0.433, False), ("Winner · hospital B (seen)", 0.434, False),
        ("Winner · all 3 hospitals", 0.360, False), ("6th of 78 teams · all 3", 0.332, False),
        ("CliniCore · new hospital (A → B)", 0.306, True), ("Winner · hospital C (unseen)", -0.123, False)]
ys = range(len(rows))[::-1]
for y, (lab, v, ours) in zip(ys, rows):
    ax.barh(y, v, height=0.56, color=OURS if ours else OTHER, zorder=3)
    ax.text(max(v, 0) + 0.012, y, f"{v:.3f}" if v >= 0 else f"{v:.3f}  (drops below zero)",
            va="center", ha="left", fontsize=12, color=TXT, weight="bold" if ours else None)
ax.set_yticks(list(ys)); ax.set_yticklabels([r[0] for r in rows])
for t, r in zip(ax.get_yticklabels(), rows):
    if r[2]: t.set_color(OURS); t.set_weight("bold")
ax.axvline(0, color=MUTED, lw=1)
ax.set_xlim(-0.2, 0.52); ax.set_xlabel("Challenge utility score (higher is better, max 1)")
ax.set_title("PhysioNet/CinC 2019 challenge", loc="left", fontsize=15, weight="bold", color=TXT, pad=14, x=-0.62)
ax.text(-0.62, -0.2, "CliniCore: our re-implementation of the utility score, public hospital-B data;\nchallenge: official hidden test set.",
        transform=ax.transAxes, fontsize=10, color=SUB, va="top")

# ---- right: AUROC
ax = fig.add_axes([0.70, 0.25, 0.27, 0.56]); style(ax)
rows = [("NEWS · review range", (0.59, 0.88), False), ("qSOFA · review range", (0.57, 0.79), False),
        ("Epic · vendor claim", (0.76, 0.83), False), ("Epic · independent check", 0.63, False),
        ("CliniCore · new hospital", 0.80, True)]
ys = range(len(rows))[::-1]
for y, (lab, v, ours) in zip(ys, rows):
    c = OURS if ours else OTHER
    if isinstance(v, tuple):
        ax.plot(v, [y, y], color=c, lw=9, solid_capstyle="round", zorder=3, alpha=0.9)
        ax.text(v[1] + 0.012, y, f"{v[0]:.2f}–{v[1]:.2f}", va="center", fontsize=12)
    else:
        ax.scatter([v], [y], s=230, color=c, edgecolor=BG, linewidth=2, zorder=4)
        ax.text(v + 0.018, y, f"{v:.2f}", va="center", fontsize=12, weight="bold" if ours else None)
ax.set_yticks(list(ys)); ax.set_yticklabels([r[0] for r in rows])
for t, r in zip(ax.get_yticklabels(), rows):
    if r[2]: t.set_color(OURS); t.set_weight("bold")
ax.axvline(0.5, color=MUTED, lw=1, ls="--"); ax.text(0.505, 4.42, "chance", color=MUTED, fontsize=10)
ax.set_xlim(0.48, 0.97); ax.set_ylim(-0.6, 4.6)
ax.set_xlabel("AUROC (higher is better, 1 = perfect)")
ax.set_title("Discrimination (AUROC)", loc="left", fontsize=15, weight="bold", color=TXT, pad=14, x=-0.55)
ax.text(-0.55, -0.2, "Epic: all hospitalised patients, one US health system (2021). NEWS/qSOFA: ED mortality,\nsystematic review. CliniCore: ICU sepsis within 6 h, trained hospital A, tested hospital B.",
        transform=ax.transAxes, fontsize=10, color=SUB, va="top")

fig.text(0.035, 0.035, "Sources: Reyna et al. 2020 / arXiv 2012.11013 (challenge) · SBU, CinC 2019 · Wong et al., JAMA Intern Med 2021 (Epic) · "
         "Sabir et al., Emerg Med J 2022 (NEWS/qSOFA)", fontsize=9.5, color=MUTED)
fig.savefig(OUT, facecolor=BG)
print("wrote", OUT)
