# Lesson 3b: Gaussian Process for all 4 outputs + parity plots
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv("bracket_dataset.csv")
features = ["Plate_Thickness", "Fillet_Radius", "Leg_Length", "Hole_Diameter", "Rib_Thickness"]
targets = ["Stress_Leg_Bottom_MPa", "Stress_Hole_MPa", "Stress_Fillet_MPa", "Tip_Deflection_mm"]

train = data[data["Set"] == "train"]
test = data[data["Set"] == "test"]
scaler = StandardScaler().fit(train[features])
X_train = scaler.transform(train[features])
X_test = scaler.transform(test[features])

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for ax, target in zip(axes.flat, targets):
    kernel = ConstantKernel(1.0) * RBF(length_scale=[1, 1, 1, 1, 1], length_scale_bounds=(0.01, 1000)) + WhiteKernel(1e-3)
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10, random_state=0)
    model.fit(X_train, train[target])
    pred, std = model.predict(X_test, return_std=True)

    mae = mean_absolute_error(test[target], pred)
    r2 = r2_score(test[target], pred)
    worst = (pred - test[target]).abs().max()
    print(f"{target:24s} MAE = {mae:.3f}   worst miss = {worst:.3f}   R2 = {r2:.3f}")

    ax.errorbar(test[target], pred, yerr=2 * std, fmt="o", capsize=3)
    low = min(test[target].min(), pred.min())
    high = max(test[target].max(), pred.max())
    ax.plot([low, high], [low, high], "r--")
    ax.set_xlabel("Ansys")
    ax.set_ylabel("Gaussian Process")
    ax.set_title(f"{target}\nMAE = {mae:.2f}, R2 = {r2:.3f}")

plt.tight_layout()
plt.savefig("plot_02_parity_GP.png", dpi=150)
print("Saved plot_02_parity_GP.png")