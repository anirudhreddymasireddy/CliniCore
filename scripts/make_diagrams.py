"""Animated model diagrams for slides and README.

    python scripts/make_diagrams.py        -> docs/img/model_stage{1,2,3}.gif + .png

Each GIF walks one patient through the stage step by step. The PNG is the
last frame (everything lit), for slides where GIFs don't play.
"""
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from PIL import Image

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
BG, PANEL, EDGE, DIM = "#0b1220", "#131c2e", "#2a3650", "#4b5874"
TXT, SUB = "#e6ecf5", "#9aa7bd"
TEAL, AMBER, RED, GREEN, BLUE, VIOLET = "#2dd4bf", "#f59e0b", "#f43f5e", "#22c55e", "#60a5fa", "#a78bfa"
W, H = 16, 7.6


def canvas(title, subtitle):
    fig = plt.figure(figsize=(W, H), dpi=90, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 160); ax.set_ylim(0, 76); ax.axis("off")
    ax.text(4, 71.5, title, color=TXT, fontsize=21, weight="bold", va="center")
    ax.text(4, 67.3, subtitle, color=SUB, fontsize=12, va="center")
    return fig, ax


def box(ax, x, y, w, h, head, lines=(), on=False, color=TEAL, small=False):
    ec = color if on else EDGE
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6",
                                fc=PANEL if not on else "#16233a", ec=ec, lw=2.6 if on else 1.4))
    ax.text(x + 1.4, y + h - 2.4, head, color=color if on else TXT, fontsize=12.5 if not small else 11,
            weight="bold", va="center")
    for i, l in enumerate(lines):
        ax.text(x + 1.4, y + h - 5.6 - i * 2.9, l, color=TXT if on else SUB, fontsize=10.2, va="center")


def arrow(ax, a, b, on=False, color=TEAL, rad=0.0, label=None, lpos=0.5, loff=(0, 1.6)):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>,head_length=7,head_width=4",
                                 connectionstyle=f"arc3,rad={rad}", color=color if on else DIM,
                                 lw=2.4 if on else 1.3, shrinkA=2, shrinkB=2))
    if label:
        mx, my = a[0] + (b[0] - a[0]) * lpos + loff[0], a[1] + (b[1] - a[1]) * lpos + loff[1]
        ax.text(mx, my, label, color=color if on else SUB, fontsize=10, ha="center", va="center", weight="bold")


def dot(ax, a, b, t, color):
    x, y = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
    ax.add_patch(Circle((x, y), 1.1, fc=color, ec="white", lw=1.2, zorder=10))


def bars(ax, x, y, vals, labels, on, hi=0, w=16, color=TEAL):
    for i, (v, l) in enumerate(zip(vals, labels)):
        yy = y - i * 2.7
        ax.add_patch(FancyBboxPatch((x, yy - 0.8), w, 1.6, boxstyle="round,pad=0,rounding_size=0.6", fc="#1d2940", ec="none"))
        ax.add_patch(FancyBboxPatch((x, yy - 0.8), w * v * on, 1.6, boxstyle="round,pad=0,rounding_size=0.6",
                                    fc=color if i == hi else "#3b4a66", ec="none"))
        ax.text(x - 0.8, yy, l, color=TXT if i == hi else SUB, fontsize=9.3, ha="right", va="center")
        if on > 0.95:
            ax.text(x + w + 0.8, yy, f"{v:.0%}", color=TXT if i == hi else SUB, fontsize=9.3, va="center")


def counter(ax, x, y, n, on, color=AMBER, label="labels", w=22):
    ax.add_patch(FancyBboxPatch((x, y), w, 3, boxstyle="round,pad=0,rounding_size=1.2", fc="#1d2940", ec="none"))
    if n > 2:
        ax.add_patch(FancyBboxPatch((x, y), w * min(n, 50) / 50, 3, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc=color if on else DIM, ec="none"))
    ax.text(x + w / 2, y + 1.5, f"{n} / 50 {label}", color=BG if n > 25 else TXT, fontsize=10.5,
            ha="center", va="center", weight="bold")


def render(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor=BG); plt.close(fig)
    buf.seek(0); return Image.open(buf).convert("RGB")


def save(frames, name, durations):
    pal = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
    pal[0].save(OUT / f"{name}.gif", save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True)
    frames[-1].save(OUT / f"{name}.png", optimize=True)
    print("wrote", OUT / f"{name}.gif", OUT / f"{name}.png")


# ---------------------------------------------------------------- stage 1
def stage1(step, t):
    fig, ax = canvas("Stage 1 · ED intake: which diagnosis group, and is the model sure?",
                     "Multinomial logistic regression on triage data · trained on 52k real ED visits (NHAMCS 2018-21) · tested on 11.6k visits from 2022")
    S = lambda k: step >= k
    box(ax, 4, 30, 30, 32, "1  Triage form", ["Complaint (codes or free text)", "HR · BP · RR · SpO2 · temp", "Pain, triage level, ambulance", "Age, sex", "13 chronic conditions"], S(0), BLUE)
    box(ax, 41, 30, 30, 32, "2  Features", ["Complaint -> word/code weights", "  (TF-IDF, words + word pairs)", "Vitals standardised (z-scores)", "Missing-value flags", "Robust to blanks"], S(1), BLUE)
    box(ax, 78, 30, 36, 32, "3  Logistic regression", ["score_g = w_g · x + b_g  (17 groups)", "softmax -> probability per group", "L2-regularised (C = 2)"], S(2), TEAL)
    bars(ax, 98, 43, [0.86, 0.06, 0.03, 0.02], ["Abdominal", "Gynecologic", "Urinary", "Other"], min(1, (step - 2 + t) if step >= 2 else 0), color=TEAL)
    box(ax, 121, 30, 35, 32, "4  Is the model sure?", ["top probability >= 0.80", "gap to 2nd >= 0.15", "nothing unusual:", "  vital outside training range,", "  unseen complaint, 3+ vitals missing", "+ 5% random re-check"], S(3), AMBER)
    arrow(ax, (34.5, 46), (40.5, 46), S(1)); arrow(ax, (71.5, 46), (77.5, 46), S(2)); arrow(ax, (114.5, 46), (120.5, 46), S(3))
    box(ax, 102, 4, 28, 17, "Sure  (23% of visits)", ["Group shown to staff", "83.5% correct in 2022", "+ list of plausible groups"], S(4), GREEN, small=True)
    box(ax, 133, 4, 24, 17, "Unsure", ["-> Doctor queue", "doctor decides", "(Stage 2)"], S(4), RED, small=True)
    arrow(ax, (132, 29.5), (118, 21.5), S(4), GREEN); arrow(ax, (142, 29.5), (145, 21.5), S(4), RED)
    ax.text(4, 14, "Why not a neural net?", color=TXT, fontsize=12, weight="bold")
    ax.text(4, 10.4, "An MLP (64-32 ReLU) was tried: 58.4% vs 59.0%, and never confident enough to answer.", color=SUB, fontsize=10.5)
    ax.text(4, 7.2, "Logistic regression is calibrated (ECE 0.022) and every weight can be read.", color=SUB, fontsize=10.5)
    ax.text(4, 4.0, "Accuracy 59% (top-3 83%) vs 17% for always guessing the commonest group.", color=SUB, fontsize=10.5)
    flow = [((34.5, 46), (40.5, 46)), ((71.5, 46), (77.5, 46)), ((114.5, 46), (120.5, 46)), ((132, 29.5), (118, 21.5))]
    if 1 <= step <= 4 and t < 1:
        a, b = flow[step - 1]; dot(ax, a, b, t, AMBER)
    return render(fig)


# ---------------------------------------------------------------- stage 2
def stage2(step, t, n):
    fig, ax = canvas("Stage 2 · Doctor in the loop: unsure cases teach the model",
                     "Not a separate model: it is the routing + labelling + safe-retraining loop around Stage 1 (and the same loop runs for Stage 3 false alarms)")
    S = lambda k: step >= k
    box(ax, 4, 34, 30, 26, "Doctor queue", ["New · Unsure · 5% re-check", "model's top guesses shown", "plausible-group list"], S(0), RED)
    box(ax, 41, 34, 32, 26, "Doctor decides", ["Picks the group (search all 17)", "+ free text: specific diagnosis", "patient sees it after review"], S(1), BLUE)
    box(ax, 80, 34, 32, 26, "Label stored", ["Exasol table ST1_FEEDBACK", "who · when · group · note", "audit trail"], S(2), BLUE)
    counter(ax, 85, 37, n, S(2))
    box(ax, 119, 34, 37, 26, "Retrain at 50 labels", ["refit on original + new labels", "safety gate on held-out 2022:", "  acc >= current - 2 pts", "  acc >= first model - 3 pts"], S(3), AMBER)
    arrow(ax, (34.5, 47), (40.5, 47), S(1)); arrow(ax, (73.5, 47), (79.5, 47), S(2)); arrow(ax, (112.5, 47), (118.5, 47), S(3))
    box(ax, 104, 5, 24, 18, "Gate passed", ["new version live", "app reloads it", "counter -> 0"], S(4), GREEN, small=True)
    box(ax, 132, 5, 24, 18, "Gate failed", ["old model kept", "labels kept", "for next round"], S(4), RED, small=True)
    arrow(ax, (132, 33.5), (117, 23.5), S(4), GREEN); arrow(ax, (142, 33.5), (144, 23.5), S(4), RED)
    arrow(ax, (103.5, 12), (19, 33.5), S(5), GREEN, rad=-0.25, label="better Stage 1 model -> fewer unsure cases", lpos=0.5, loff=(-4, 6))
    ax.text(4, 12, "Same loop in the ICU:", color=TXT, fontsize=12, weight="bold")
    ax.text(4, 8.6, "nurse marks a false alarm + reason -> 50 reasons -> retrain risk models", color=SUB, fontsize=10.5)
    ax.text(4, 5.4, "(alarms 'treated early, event prevented' are never used as negatives)", color=SUB, fontsize=10.5)
    flow = [((34.5, 47), (40.5, 47)), ((73.5, 47), (79.5, 47)), ((112.5, 47), (118.5, 47)), ((132, 33.5), (117, 23.5))]
    if 1 <= step <= 4 and t < 1:
        a, b = flow[step - 1]; dot(ax, a, b, t, AMBER)
    return render(fig)


# ---------------------------------------------------------------- stage 3
def trees(ax, x, y, on, color):
    for i in range(5):
        cx = x + i * 4.2
        c = color if on else DIM
        ax.plot([cx, cx - 1.2], [y, y - 2], color=c, lw=1.2); ax.plot([cx, cx + 1.2], [y, y - 2], color=c, lw=1.2)
        for px in (cx, cx - 1.2, cx + 1.2):
            ax.add_patch(Circle((px, y if px == cx else y - 2), 0.45, fc=c, ec="none"))
        if i < 4:
            ax.text(cx + 2.1, y - 1, "+", color=SUB, fontsize=9, ha="center", va="center")


def stage3(step, t, n, crit):
    fig, ax = canvas("Stage 3 · ICU: four risks every hour, one criticality score, nurses correct false alarms",
                     "LightGBM per risk (gradient-boosted trees) + rule tiers · trained on PhysioNet 2019 hospital A, tested on hospital B")
    S = lambda k: step >= k
    box(ax, 4, 22, 27, 40, "1  Every hour", ["Vitals: HR, BP, MAP,", "  RR, SpO2, temp", "Labs: lactate, creatinine,", "  WBC, platelets, ...", "Age, hours in ICU", "", "real data or live feed"], S(0), BLUE)
    box(ax, 37, 22, 27, 40, "2  Features", ["latest value", "6 h trend & variability", "worst value so far", "missing flags", "rule-score inputs", "", "same code for MIMIC-IV"], S(1), BLUE)
    heads = [("Sepsis", "next 6 h", RED), ("Breathing failure", "next 6 h", BLUE), ("Kidney injury", "next 12 h", VIOLET), ("Shock", "next 6 h", AMBER)]
    ax.text(71, 61, "3  One LightGBM model per risk", color=TEAL if S(2) else TXT, fontsize=12.5, weight="bold", va="center")
    for i, (nm, hz, c) in enumerate(heads):
        y = 51 - i * 9.4
        box(ax, 70, y, 42, 7.4, f"{nm}  ·  {hz}", [], S(2), c, small=True)
        trees(ax, 92, y + 4.6, S(2), c)
    ax.text(71, 13.5, "~300 small trees, each fixes the last one's errors;", color=SUB, fontsize=9.8)
    ax.text(71, 10.8, "alert threshold = 0.15 alerts per patient-day", color=SUB, fontsize=9.8)
    box(ax, 118, 36, 38, 26, "4  Criticality 0-100", ["each risk: tier Low/Watch/High/Critical", "  from bedside rules, raised if the", "  model expects High within its window", "score = 100 x (1 - prod(1 - severity))"], S(3), AMBER)
    if S(3):
        col = GREEN if crit < 15 else AMBER if crit < 40 else "#fb923c" if crit < 70 else RED
        ax.add_patch(Circle((150, 41), 3.6, fc=BG, ec=col, lw=4)); ax.text(150, 41, f"{crit}", color=col, fontsize=13, ha="center", va="center", weight="bold")
    box(ax, 118, 4, 18, 26, "5  Board", ["monitor wall", "ranked risks", "notify doctor", "(SBAR pop-up)"], S(4), GREEN, small=True)
    box(ax, 138, 4, 18, 26, "6  False alarm?", ["nurse picks reason", "(required)"], S(5), RED, small=True)
    counter(ax, 139.5, 13, n, S(5), RED, "", w=15)
    if S(5):
        ax.text(147, 9, "50 -> retrain + gate", color=SUB, fontsize=9.4, ha="center")
    arrow(ax, (31.5, 42), (36.5, 42), S(1)); arrow(ax, (64.5, 42), (69.5, 42), S(2)); arrow(ax, (112.5, 47), (117.5, 47), S(3))
    arrow(ax, (127, 35.5), (127, 30.5), S(4), GREEN); arrow(ax, (136.5, 17), (137.5, 17), S(5), RED)
    arrow(ax, (147, 3.6), (100, 13.5), S(6), RED, rad=-0.3, label="retrained models (only if gate passes)", lpos=0.62, loff=(-14, -3.2))
    flow = [((31.5, 42), (36.5, 42)), ((64.5, 42), (69.5, 42)), ((112.5, 47), (117.5, 47)), ((127, 35.5), (127, 30.5)), ((136.5, 17), (137.5, 17))]
    if 1 <= step <= 5 and t < 1:
        a, b = flow[step - 1]; dot(ax, a, b, t, AMBER)
    return render(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ts = [0.0, 0.33, 0.66, 1.0]
    # stage 1
    fr, du = [], []
    for s in range(5):
        for t in ts:
            fr.append(stage1(s, t)); du.append(160)
        du[-1] = 900
    fr.append(stage1(5, 1)); du.append(2600)
    save(fr, "model_stage1", du)
    # stage 2
    fr, du = [], []
    for s in range(6):
        if s == 3:
            for n in range(34, 51, 4):
                fr.append(stage2(2, 1, n)); du.append(150)
        for t in ts:
            fr.append(stage2(s, t, 50 if s >= 3 else 33)); du.append(160)
        du[-1] = 900
    fr.append(stage2(6, 1, 0)); du.append(2600)
    save(fr, "model_stage2", du)
    # stage 3
    fr, du = [], []
    crits = {0: 0, 1: 0, 2: 0, 3: 72, 4: 72, 5: 72, 6: 72}
    for s in range(7):
        if s == 3:
            for c in (12, 30, 48, 63, 72):
                fr.append(stage3(3, 1, 47, c)); du.append(150)
        if s == 6:
            for n in (48, 49, 50):
                fr.append(stage3(5, 1, n, 72)); du.append(250)
        for t in ts:
            fr.append(stage3(s, t, 50 if s >= 6 else 47, crits[s])); du.append(160)
        du[-1] = 900
    fr.append(stage3(7, 1, 0, 72)); du.append(2600)
    save(fr, "model_stage3", du)


if __name__ == "__main__":
    main()
