"""Architecture figures for the technical report.

    python scripts/make_report_figures.py  -> docs/img/arch_*.png, docs/img/system_overview.png

Sizes are the real ones from the code / saved models:
  Stage 1 logistic regression : 18,035 inputs -> 17 outputs (306,612 weights)
  Stage 1 MLP (tried)         : 18,035 -> 64 ReLU -> 32 ReLU -> 17 (1,156,945 weights)
  GRU-D (tried, Stage 3)      : 34 vars + 34 masks + 5 statics -> GRU 64 -> 32 ReLU -> 1 (31,109 weights)
  LightGBM (Stage 3, chosen)  : 91 features, 300 trees x 15 leaves per risk, 4 risks
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
BG, PANEL, EDGE, DIM = "#0b1220", "#131c2e", "#2a3650", "#3b4a66"
TXT, SUB = "#e6ecf5", "#9aa7bd"
TEAL, AMBER, RED, GREEN, BLUE, VIOLET = "#2dd4bf", "#f59e0b", "#f43f5e", "#22c55e", "#60a5fa", "#a78bfa"


def canvas(title, subtitle, w=16, h=8):
    fig = plt.figure(figsize=(w, h), dpi=110, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w * 10); ax.set_ylim(0, h * 10); ax.axis("off")
    ax.text(4, h * 10 - 5, title, color=TXT, fontsize=20, weight="bold", va="center")
    ax.text(4, h * 10 - 9.5, subtitle, color=SUB, fontsize=12, va="center")
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / name, facecolor=BG); plt.close(fig); print("wrote", OUT / name)


def box(ax, x, y, w, h, head, lines=(), color=TEAL, fs=12.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6", fc=PANEL, ec=color, lw=2))
    ax.text(x + 1.5, y + h - 2.6, head, color=color, fontsize=fs, weight="bold", va="center")
    for i, l in enumerate(lines):
        ax.text(x + 1.5, y + h - 6 - i * 3, l, color=TXT, fontsize=10.3, va="center")


def arrow(ax, a, b, color=SUB, lw=1.8, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>,head_length=7,head_width=4", connectionstyle=f"arc3,rad={rad}",
                                 color=color, lw=lw, shrinkA=2, shrinkB=2))


def layer(ax, x, yc, n_show, label, sub, color, gap=4.8, r=1.5, dots=True):
    """Column of neurons; returns their centres. Shows n_show circles with '...' in the middle."""
    k = n_show
    ys = [yc + (i - (k - 1) / 2) * gap for i in range(k)]
    pts = []
    for i, y in enumerate(ys):
        if dots and k >= 5 and i == k // 2:
            ax.text(x, y, "⋮", color=SUB, fontsize=16, ha="center", va="center"); continue
        ax.add_patch(Circle((x, y), r, fc=PANEL, ec=color, lw=2, zorder=3)); pts.append((x, y))
    top = max(ys) + 5
    ax.text(x, top + 2.5, label, color=color, fontsize=12.5, weight="bold", ha="center")
    for j, s in enumerate(sub.split("\n")):
        ax.text(x, min(ys) - 4.5 - j * 2.8, s, color=SUB, fontsize=10, ha="center", va="center")
    return pts


def connect(ax, A, B, color=DIM, lw=0.6):
    for a in A:
        for b in B:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, zorder=1, alpha=0.8)


# ---------------------------------------------------------------- logistic regression
def logreg():
    fig, ax = canvas("Stage 1 model (chosen): multinomial logistic regression",
                     "A neural network with no hidden layer: every input connects straight to 17 outputs, then softmax turns scores into probabilities")
    inp = layer(ax, 32, 39, 9, "Input layer", "18,035 inputs\n12 numbers + 12 missing flags\n+ 18,011 complaint words / word pairs", BLUE)
    out = layer(ax, 96, 39, 7, "Output layer", "17 diagnosis groups\nsoftmax", TEAL, gap=6)
    connect(ax, inp, out)
    box(ax, 112, 46, 44, 20, "What one output computes", ["score_g = w_g · x + b_g", "p_g = exp(score_g) / Σ exp(score_k)", "weights: 17 × 18,035 + 17 = 306,612", "L2 penalty (C = 2), lbfgs, 600 iterations"], TEAL)
    box(ax, 112, 18, 44, 24, "Inputs in detail", ["Numbers (z-scored): temp, HR, RR, SpO2,", "  SBP, DBP, triage level, early-warning", "  points, age, sex, ambulance, history count", "Missing flag per number (blank ≠ normal)", "Text: TF-IDF of complaint codes + words,", "  1- and 2-word terms seen ≥ 2 times"], BLUE)
    ax.text(4, 3.5, "Hidden layers: 0  ·  Activation: softmax at output  ·  Trained on 42,049 visits (2018-21), model picked on 10,513, tested on 11,607 from 2022",
            color=SUB, fontsize=11)
    save(fig, "arch_logreg.png")


# ---------------------------------------------------------------- MLP
def mlp():
    fig, ax = canvas("Stage 1 alternative (tried): multilayer perceptron 64 → 32",
                     "Same inputs and outputs, plus two hidden ReLU layers. 2022 test: 58.4% accuracy (LR 59.0%), top-3 79.9% (82.8%), never sure enough to answer alone")
    inp = layer(ax, 16, 38, 9, "Input", "18,035", BLUE)
    h1 = layer(ax, 48, 38, 9, "Hidden 1", "64 neurons\nReLU", VIOLET, gap=5.0)
    h2 = layer(ax, 78, 38, 7, "Hidden 2", "32 neurons\nReLU", VIOLET, gap=5.6)
    out = layer(ax, 106, 38, 7, "Output", "17 groups\nsoftmax", TEAL, gap=6)
    connect(ax, inp, h1); connect(ax, h1, h2); connect(ax, h2, out)
    box(ax, 118, 30, 38, 36, "Settings (scikit-learn)", ["layers: 18,035 → 64 → 32 → 17", "weights: 1,156,945", "  18,035×64+64 = 1,154,304", "  64×32+32 = 2,080", "  32×17+17 = 561", "Adam, L2 α = 0.001", "early stopping (10% held out)", "probabilities re-calibrated", "  (sigmoid, 3-fold)"], VIOLET)
    ax.text(4, 6, "Hidden layers: 2  ·  Regularisation: L2 weight penalty + early stopping (no dropout layer in scikit-learn's MLP)  ·  3.8× more weights than the logistic regression",
            color=SUB, fontsize=11)
    save(fig, "arch_mlp.png")


# ---------------------------------------------------------------- GRU-D
def grud():
    fig, ax = canvas("Stage 3 alternative (tried): GRU-D recurrent network for sepsis",
                     "A GRU that reads the ICU stay hour by hour and handles missing labs by decaying old values towards normal (Che et al. 2018)")
    xs = [26, 60, 94]
    labels = ["hour t-1", "hour t", "hour t+1"]
    for i, x in enumerate(xs):
        box(ax, x - 13, 33, 26, 17, f"GRU-D cell · {labels[i]}", ["hidden state: 64", "input decay γx", "hidden decay γh"], VIOLET, fs=11.5)
        box(ax, x - 13, 9, 26, 16, "Inputs this hour", ["34 values (8 vitals, 26 labs)", "34 observed? masks", "5 statics (age, sex, unit…)"], BLUE, fs=11.5)
        arrow(ax, (x, 25.5), (x, 32.5), BLUE)
        ax.add_patch(FancyBboxPatch((x - 9, 55), 18, 6, boxstyle="round,pad=0.3,rounding_size=1.2", fc=PANEL, ec=TEAL, lw=2))
        ax.text(x, 58, "64 → 32 ReLU → 1", color=TEAL, fontsize=10.5, ha="center", va="center", weight="bold")
        arrow(ax, (x, 50.5), (x, 54.6), TEAL)
        ax.text(x, 63.5, "p(sepsis in 6 h)", color=TXT, fontsize=10.5, ha="center")
        if i < 2:
            arrow(ax, (x + 13.5, 41.5), (xs[i + 1] - 13.5, 41.5), VIOLET, lw=2.4)
            ax.text((x + xs[i + 1]) / 2, 44, "h", color=VIOLET, fontsize=12, ha="center", weight="bold")
    box(ax, 116, 9, 40, 58, "Layers and sizes", ["1  input decay (per variable):", "   x̂ = m·x + (1-m)·γx·x_last", "   γx = exp(-relu(wx·Δt + bx))   68 w", "2  hidden decay:", "   h ← h·exp(-relu(Wh·Δt))  2,240 w", "3  GRU cell, input 73 → 64:", "   3 gates (update, reset, new)", "   3·(73·64 + 64·64 + 2·64) = 26,688 w", "4  head: 64 → 32 (ReLU) → 1", "   2,080 + 33 = 2,113 w", "", "Total: 31,109 weights", "Adam lr 0.002, 6 epochs, batch 128,", "weighted BCE, gradient clip 5"], VIOLET)
    ax.text(4, 5, "Recurrent layer: 1 (GRU, 64 units) + 2 dense layers  ·  Result on hospital B: AUPRC 0.045, AUROC 0.737 vs LightGBM 0.069 / 0.800, so LightGBM was chosen",
            color=SUB, fontsize=11)
    save(fig, "arch_grud.png")


# ---------------------------------------------------------------- LightGBM
def tree(ax, x, y, color, s=1.0, depth=3):
    pts = [(x, y)]
    ax.add_patch(Circle((x, y), 1.1 * s, fc=color, ec="none", zorder=3))
    lvl = [(x, y)]
    for d in range(1, depth + 1):
        nxt = []
        for (px, py) in lvl:
            for sgn in (-1, 1):
                cx, cy = px + sgn * 7 * s / (2 ** (d - 1)), py - 5 * s
                ax.plot([px, cx], [py, cy], color=color, lw=1.2, zorder=2)
                ax.add_patch(Circle((cx, cy), (0.9 if d < depth else 0.7) * s, fc=color if d < depth else PANEL, ec=color, lw=1.2, zorder=3))
                nxt.append((cx, cy))
        lvl = nxt
    return lvl


def lightgbm():
    fig, ax = canvas("Stage 3 model (chosen): LightGBM, one per risk",
                     "Gradient-boosted decision trees: 300 small trees are added up; each new tree corrects what the previous ones still get wrong")
    xs = [20, 52, 84]
    for i, x in enumerate(xs):
        tree(ax, x, 58, [BLUE, VIOLET, AMBER][i])
        ax.text(x, 64, ["tree 1", "tree 2", "tree 300"][i], color=TXT, fontsize=12, ha="center", weight="bold")
        ax.text(x, 34, ["e.g. lactate > 2.1?", "e.g. MAP falling 6 h?", "e.g. resp. rate > 24?"][i], color=SUB, fontsize=10, ha="center")
        ax.text(x, 30.5, ["+0.31", "+0.12", "−0.02"][i], color=TXT, fontsize=12, ha="center", weight="bold")
        if i == 0: ax.text(4, 30.5, "(illustration)", color=SUB, fontsize=9.5)
        if i < 2:
            ax.text(x + 16, 50, "+" if i == 0 else "+ … +", color=SUB, fontsize=18, ha="center", va="center")
    arrow(ax, (98, 50), (106, 50), TEAL, lw=2.4)
    box(ax, 107, 40, 49, 22, "Sum → probability", ["score = base + Σ leaf values of the 300 trees", "p = 1 / (1 + e^-score)", "alert if p > threshold set for", "  0.15 alerts per patient-day (hospital A)"], TEAL)
    box(ax, 4, 6, 62, 20, "Settings per risk model", ["300 trees · up to 15 leaves each · learning rate 0.05", "80% rows and 80% features sampled per tree", "L2 = 2 · rare events up-weighted (max 20×)", "91 features: latest vitals/labs + bedside scores, hours since each", "  was measured, 6 h change/min/max, current rule tiers, age, history"], AMBER, fs=12)
    box(ax, 71, 6, 85, 20, "Why trees and not a neural net here", ["Hourly ICU data is tabular and full of gaps; trees split on 'missing' natively", "Beat GRU-D on both AUPRC and utility on a new hospital (benchmark)", "Fast: all four trained in under 5 minutes on the team laptop", "Per-patient reasons: which features pushed the risk up (shown to staff in plain words)", "4 models (sepsis, breathing, kidney, shock) = 1,200 trees, ~18,000 leaves"], GREEN, fs=12)
    save(fig, "arch_lightgbm.png")


# ---------------------------------------------------------------- system overview
def overview():
    fig, ax = canvas("CliniCore: data, models and the staff app", "Everything runs on one laptop: Exasol holds data, predictions and staff feedback; FastAPI + React serve the app", 16, 9)
    box(ax, 4, 26, 34, 48, "Data", ["NHAMCS ED 2018-22", "  (CDC, public domain)", "  86,864 visits", "PhysioNet 2019 ICU", "  (CC BY 4.0)", "  40,336 patients", "MIMIC-IV-shaped samples", "  (synthetic, demo)", "Optional: real MIMIC-IV", "  or a live hospital feed", "  (same code paths)"], BLUE)
    box(ax, 50, 52, 30, 22, "Stage 1 · ED", ["logistic regression, 17 groups", "sure → shown to staff", "unsure → doctor queue"], TEAL)
    box(ax, 50, 26, 30, 20, "Stage 2 · learning loop", ["doctor labels / false alarms", "50 → retrain → safety gate", "pass → new version live"], RED)
    box(ax, 92, 52, 30, 22, "Stage 3 · ICU", ["4 LightGBM risk models", "+ bedside rule tiers", "criticality 0-100, hourly"], VIOLET)
    box(ax, 92, 26, 30, 20, "Staff app (port 8765)", ["ED intake · Doctor queue", "ICU risk · False alarm", "Notify doctor · Inbox"], GREEN)
    box(ax, 128, 26, 28, 48, "Who sees what", ["Triage nurse:", "  intake + result", "ED doctor:", "  unsure cases", "ICU nurse:", "  board, false alarms", "On-call doctor:", "  pop-up, reply", "Developers:", "  metrics, registry"], SUB)
    box(ax, 4, 3, 118, 17, "Exasol (or CSV files)", ["raw: NHAMCS_ED, SEPSIS2019  ·  predictions: ST1_PREDICTIONS, ST3_RISKS, ST3_VITALS, ST3_RISK_TIMELINE", "feedback: ST1_FEEDBACK (doctor labels), ST3_FEEDBACK (false alarms + reason)  ·  MODEL_REGISTRY  ·  notifications"], AMBER)
    arrow(ax, (21, 25.5), (21, 20.5), BLUE)
    arrow(ax, (46, 20.5), (49.5, 60), AMBER); ax.text(43.5, 40, "reads", color=SUB, fontsize=9.5, rotation=84)
    arrow(ax, (88, 20.5), (91.5, 60), AMBER); ax.text(85.5, 40, "reads", color=SUB, fontsize=9.5, rotation=84)
    arrow(ax, (65, 51.5), (65, 46.5), TEAL); arrow(ax, (107, 51.5), (107, 46.5), VIOLET)
    arrow(ax, (65, 25.5), (65, 20.5), RED); arrow(ax, (107, 25.5), (107, 20.5), GREEN)
    ax.text(108.5, 23, "staff actions", color=SUB, fontsize=9.5); ax.text(66.5, 23, "new versions", color=SUB, fontsize=9.5)
    save(fig, "system_overview.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    logreg(); mlp(); grud(); lightgbm(); overview()
