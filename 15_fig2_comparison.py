# Portfolio figure 2: why a Gaussian Process?
import pandas as pd
import matplotlib.pyplot as plt

BLUE = "#2a78d6"
LIGHT = "#c3c2b7"
GREY = "#52514e"
INK = "#0b0b0b"

table = pd.read_csv("model_comparison.csv")
table = table.sort_values("Stress_Leg_Bottom_MPa", ascending=False)

colors = [BLUE if name == "Gaussian Process" else LIGHT for name in table["Model"]]

plt.rcParams.update({"font.size": 14, "axes.edgecolor": GREY, "xtick.color": GREY, "ytick.color": INK})
fig, ax = plt.subplots(figsize=(10, 5.5))
bars = ax.barh(table["Model"], table["Stress_Leg_Bottom_MPa"], color=colors, height=0.6)

for bar, value in zip(bars, table["Stress_Leg_Bottom_MPa"]):
    ax.text(value + 0.2, bar.get_y() + bar.get_height() / 2, f"{value:.1f} MPa",
            va="center", fontsize=14, color=INK)

ax.set_xlim(0, 13.5)
ax.set_xlabel("Average prediction error on 10 unseen brackets (MPa)  —  lower is better")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color="#e6e5e1", linewidth=0.8)
ax.set_axisbelow(True)
fig.suptitle("Why a Gaussian Process? It was 6–11× more accurate", fontsize=17, color=INK, x=0.03, ha="left")
ax.set_title("4 models trained on the same 50 Ansys runs, leg-bottom stress", fontsize=13, color=GREY, loc="left")
plt.tight_layout()
plt.savefig("fig2_model_comparison.png", dpi=200)
print("Saved fig2_model_comparison.png")