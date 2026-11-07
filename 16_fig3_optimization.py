# Portfolio figure 3: the optimization story (verified in Ansys)
import matplotlib.pyplot as plt

BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#d03b3b"
GREEN = "#0ca30c"
GREY = "#52514e"
INK = "#0b0b0b"

rounds = ["Round 1\n60 Ansys runs\nmass 62.2 g", "Round 2\n+7 adaptive runs\nmass 65.3 g"]
ml_pred = [134.5, 130.6]
ansys = [146.6, 123.5]
x = [0, 1]

plt.rcParams.update({"font.size": 14, "axes.edgecolor": GREY, "xtick.color": INK, "ytick.color": GREY})
fig, ax = plt.subplots(figsize=(10, 7))

ax.axhline(138, color=RED, linestyle="--", linewidth=2)
ax.text(1.42, 138.8, "Allowable 138 MPa", color=RED, fontsize=13, ha="right", va="bottom")

for i in x:
    ax.plot([i, i], [ml_pred[i], ansys[i]], color=GREY, linewidth=2, zorder=1)
ax.scatter(x, ml_pred, s=220, color=BLUE, edgecolor="white", linewidth=2, zorder=3, label="ML surrogate prediction")
ax.scatter(x, ansys, s=220, color=ORANGE, edgecolor="white", linewidth=2, zorder=3, label="Ansys verification")

for i in x:
    ax.text(i - 0.08, ml_pred[i], f"ML: {ml_pred[i]:.1f}", color=INK, fontsize=13, ha="right", va="center")
    ax.text(i + 0.08, ansys[i], f"Ansys: {ansys[i]:.1f}", color=INK, fontsize=13, ha="left", va="center")

ax.text(0, 151.5, "✗ NOT SAFE", color=RED, fontsize=15, ha="center", fontweight="bold")
ax.text(1, 151.5, "✓ SAFE", color=GREEN, fontsize=15, ha="center", fontweight="bold")
ax.annotate("", xy=(0.82, 132), xytext=(0.18, 132),
            arrowprops=dict(arrowstyle="->", color=GREY, linewidth=1.5))
ax.text(0.5, 133, "adaptive sampling\n(more data in the weak corner)", color=GREY, fontsize=12, ha="center", va="bottom")

ax.set_xticks(x)
ax.set_xticklabels(rounds)
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(115, 156)
ax.set_ylabel("Leg-bottom stress (MPa)")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(loc="lower left", frameon=False, fontsize=12)
fig.suptitle("Lightest safe bracket: 65.3 g, 35% lighter, verified in Ansys", fontsize=17, color=INK, x=0.03, ha="left")
ax.set_title("Leg fixed at 120 mm, load 100 N. The first optimum failed FEA verification and was fixed.",
             fontsize=12, color=GREY, loc="left")
plt.tight_layout()
plt.savefig("fig3_optimization_story.png", dpi=200)
print("Saved fig3_optimization_story.png")