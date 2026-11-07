# Step 2: Find the lightest safe bracket with leg = 120 mm, using the GP surrogate
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("bracket_dataset.csv")
features = ["Plate_Thickness", "Fillet_Radius", "Leg_Length", "Hole_Diameter", "Rib_Thickness"]
stresses = ["Stress_Leg_Bottom_MPa", "Stress_Hole_MPa", "Stress_Fillet_MPa"]
ALLOWABLE = 138
LEG = 120

# 1. Train one GP per stress on ALL 60 brackets
scaler = StandardScaler().fit(data[features])
X_all = scaler.transform(data[features])
kernel = ConstantKernel(1.0) * RBF(length_scale=[1, 1, 1, 1, 1], length_scale_bounds=(0.01, 1000)) + WhiteKernel(1e-3)
models = {}
for target in stresses:
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10, random_state=0)
    model.fit(X_all, data[target])
    models[target] = model

# 2. Make 20,000 candidate brackets (leg fixed at 120 mm)
rng = np.random.default_rng(0)
n = 20000
cand = pd.DataFrame({
    "Plate_Thickness": rng.uniform(3, 8, n),
    "Fillet_Radius": rng.uniform(2, 15, n),
    "Leg_Length": np.full(n, LEG),
    "Hole_Diameter": rng.uniform(6, 12, n),
    "Rib_Thickness": rng.uniform(1.5, 6, n),
}).round(2)

# 3a. Predict stresses, with a safety margin (prediction + 2 x uncertainty)
X_cand = scaler.transform(cand[features])
for target in stresses:
    pred, std = models[target].predict(X_cand, return_std=True)
    cand[target] = pred
    cand[target + "_upper"] = pred + 2 * std

# 3b. Exact mass from the hand formula
def bracket_mass_kg(t, R, B, d, tr):
    k = 1 - np.pi / 4
    volume = 40 * (B*t + (60 - t)*t + k*R**2) - (np.pi/4)*d**2*t + tr*((40 - 2*t)**2/2 - k*R**2)
    return volume * 2.7e-6

cand["Mass_g"] = bracket_mass_kg(cand["Plate_Thickness"], cand["Fillet_Radius"], cand["Leg_Length"],
                                 cand["Hole_Diameter"], cand["Rib_Thickness"]) * 1000

# 4. Keep the safe ones, show the 5 lightest
cand["Worst_Upper_MPa"] = cand[[s + "_upper" for s in stresses]].max(axis=1)
safe = cand[cand["Worst_Upper_MPa"] <= ALLOWABLE]
print("Candidates tried:", len(cand), "   Safe:", len(safe))

best = safe.sort_values("Mass_g").head(5)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)
print(best[features + ["Mass_g", "Stress_Leg_Bottom_MPa", "Stress_Hole_MPa", "Worst_Upper_MPa"]].round(2).to_string(index=False))

nominal = bracket_mass_kg(5, 8, 120, 8, 4) * 1000
print("Nominal-shape bracket at leg 120 (t5, R8, d8, tr4):", round(nominal, 1), "g")