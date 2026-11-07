# Lesson 2: First ML model - linear regression (the baseline)
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("bracket_dataset.csv")

features = ["Plate_Thickness", "Fillet_Radius", "Leg_Length", "Hole_Diameter", "Rib_Thickness"]
target = "Stress_Leg_Bottom_MPa"

# Split: 50 for learning, 10 for the exam
train = data[data["Set"] == "train"]
test = data[data["Set"] == "test"]

# Train the model on the 50 training brackets only
model = LinearRegression()
model.fit(train[features], train[target])

# Exam: predict the 10 unseen test brackets
pred = model.predict(test[features])

# Score the exam
mae = mean_absolute_error(test[target], pred)
rmse = mean_squared_error(test[target], pred) ** 0.5
r2 = r2_score(test[target], pred)
print("MAE  =", round(mae, 2), "MPa")
print("RMSE =", round(rmse, 2), "MPa")
print("R2   =", round(r2, 3))

# Show each test bracket: Ansys vs model
result = pd.DataFrame({"Design_ID": test["Design_ID"], "Ansys_MPa": test[target].round(1), "Predicted_MPa": pred.round(1)})
result["Error_MPa"] = (result["Predicted_MPa"] - result["Ansys_MPa"]).round(1)
print(result.to_string(index=False))