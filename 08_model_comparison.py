# Step 1: Compare 4 models on the same train/test split
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone

data = pd.read_csv("bracket_dataset.csv")
features = ["Plate_Thickness", "Fillet_Radius", "Leg_Length", "Hole_Diameter", "Rib_Thickness"]
targets = ["Stress_Leg_Bottom_MPa", "Stress_Hole_MPa", "Stress_Fillet_MPa", "Tip_Deflection_mm"]
train = data[data["Set"] == "train"]
test = data[data["Set"] == "test"]
scaler = StandardScaler().fit(train[features])
X_train = scaler.transform(train[features])
X_test = scaler.transform(test[features])

kernel = ConstantKernel(1.0) * RBF(length_scale=[1, 1, 1, 1, 1], length_scale_bounds=(0.01, 1000)) + WhiteKernel(1e-3)

models = {
    "Linear regression": LinearRegression(),
    "Random forest": RandomForestRegressor(n_estimators=300, random_state=0),
    "Gradient boosting": GradientBoostingRegressor(random_state=0),
    "Gaussian Process": GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10, random_state=0),
}

rows = []
for model_name, template in models.items():
    row = {"Model": model_name}
    for target in targets:
        model = clone(template)
        model.fit(X_train, train[target])
        pred = model.predict(X_test)
        row[target] = round(mean_absolute_error(test[target], pred), 3)
    rows.append(row)

table = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("Average miss (MAE) on the 10 unseen test brackets:")
print(table.to_string(index=False))
table.to_csv("model_comparison.csv", index=False)