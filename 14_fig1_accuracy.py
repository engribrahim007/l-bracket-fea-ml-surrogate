# Portfolio figure 1: how accurate is the surrogate?
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

BLUE = "#2a78d6"
GREY = "#52514e"
INK = "#0b0b0b"

data = pd.read_csv("bracket_dataset.csv")
features = ["Plate_Thickness", "Fillet_Radius", "Leg_Length", "Hole_Diameter", "Rib_Thickness"]
target = "Stress_Leg_Bottom_MPa"
train = data[data["Set"] == "train"]
test = data[data["Set"] == "test"]

scaler = StandardScaler().fit(train[features])
kernel = ConstantKernel(1.0) * RBF(length_scale=[1, 1, 1, 1, 1], length_scale_bounds=(0.01, 1000)) + WhiteKernel(1e-3)
model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10, random_state=0)
model.fit(scaler.transform(train[features]), train[target])
pred, std = model.predict(scaler.transform(test[features]), return_std=True)
mae = mean_absolute_error(test[target], pred)
r2 = r2_score(test[target], pred)

plt.rcParams.update({"font.size": 14, "axes.edgecolor": GREY, "axes.labelcolor": INK,
                     "xtick.color": GREY, "ytick.color": GREY})
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot([0, 140], [0, 140], color=GREY, linestyle="--", linewidth=1.5, zorder=1)
ax.text(100, 92, "perfect prediction", color=GREY, fontsize=12, rotation=45)
ax.errorbar(test[target], pred, yerr=2 * std, fmt="o", color=BLUE, markersize=10,
            markeredgecolor="white", markeredgewidth=2, ecolor=BLUE, elinewidth=2, capsize=4, zorder=3)
ax.set_xlim(0, 140)
ax.set_ylim(0, 140)
ax.set_aspect("equal")
ax.grid(color="#e6e5e1", linewidth=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("Stress calculated by Ansys FEA (MPa)")
ax.set_ylabel("Stress predicted by ML surrogate (MPa)")
fig.suptitle("ML surrogate vs Ansys: 10 brackets the model never saw", fontsize=17, color=INK, x=0.06, ha="left")
ax.set_title(f"Leg-bottom stress  |  average error {mae:.1f} MPa  |  R² = {r2:.3f}",
             fontsize=13, color=GREY, loc="left")
plt.tight_layout()
plt.savefig("fig1_surrogate_accuracy.png", dpi=200)
print("Saved fig1_surrogate_accuracy.png")