"""Charts for the demo from stage1_rounds.csv (written by stage1_pipeline.py).  python plot_rounds.py"""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = pd.read_csv("stage1_rounds.csv")
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].plot(d["labels_from_doctors"], d["accuracy_new_hospital"] * 100, "o-", label="new hospital")
ax[0].plot(d["labels_from_doctors"], d["accuracy_hospital_A"] * 100, "s--", label="original hospital (gate)")
ax[0].set(title="Accuracy as doctors label cases", xlabel="doctor labels", ylabel="% correct"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(d["labels_from_doctors"], d["auto_share_new_hospital"] * 100, "o-", color="tab:green")
ax[1].set(title="Share answered without a doctor", xlabel="doctor labels", ylabel="% of cases"); ax[1].grid(alpha=.3)
ax[2].bar(d["round"], d["wrong_shown_new_hospital"], color="tab:red")
ax[2].set(title="Wrong answers shown to patients", xlabel="round", ylabel="cases")
plt.tight_layout(); plt.savefig("stage1_rounds.png", dpi=150); print("saved stage1_rounds.png")
