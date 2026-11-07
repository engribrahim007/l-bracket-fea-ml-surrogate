# Lesson 3: Gaussian Process model for leg-bottom stress
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("bracket_dataset.csv")
features = ["Plate_Thickness", "Fillet_Radius", "Leg_Length", "Hole_Diameter", "Rib_Thickness"]
target = "Stress_Leg_Bottom_MPa"

train = data[data["Set"] == "train"]
test = data[data["Set"] == "test"]

# Scale inputs (learned from training data only)
scaler = StandardScaler().fit(train[features])
X_train = scaler.transform(train[features])
X_test = scaler.transform(test[features])

# Build and train the Gaussian Process
kernel = ConstantKernel(1.0) * RBF(length_scale=[1, 1, 1, 1, 1], length_scale_bounds=(0.01, 1000)) + WhiteKernel(1e-3)
model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10, random_state=0)
model.fit(X_train, train[target])

# Exam on the 10 unseen brackets, with uncertainty
pred, std = model.predict(X_test, return_std=True)

mae = mean_absolute_error(test[target], pred)
rmse = mean_squared_error(test[target], pred) ** 0.5
r2 = r2_score(test[target], pred)
print("MAE  =", round(mae, 2), "MPa")
print("RMSE =", round(rmse, 2), "MPa")
print("R2   =", round(r2, 3))

result = pd.DataFrame({"Design_ID": test["Design_ID"], "Ansys_MPa": test[target].round(1),
                       "GP_MPa": pred.round(1), "Plus_Minus": (2 * std).round(1)})
result["Error_MPa"] = (result["GP_MPa"] - result["Ansys_MPa"]).round(1)
print(result.to_string(index=False))

print("Learned length scales:", model.kernel_.k1.k2.length_scale.round(2))