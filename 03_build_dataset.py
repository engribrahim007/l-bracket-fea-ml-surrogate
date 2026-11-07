# Step 5: Join DOE inputs with Ansys results into one clean dataset
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# Read both files
inputs = pd.read_csv("doe_inputs.csv")
results = pd.read_csv("doe_results.csv", comment="#")

# Remove DP 0 (the old setup point, not part of the DOE)
results = results[results["Name"] != "DP 0"]

# "DP 12" -> 12, so it matches Design_ID
results["Design_ID"] = results["Name"].str.replace("DP ", "").astype(int)

# Give the Ansys output columns clear names
results = results.rename(columns={"P8": "Mass_kg", "P9": "Stress_Fillet_MPa", "P10": "Stress_Hole_MPa",
    "P11": "Tip_Deflection_mm", "P12": "Stress_Leg_Bottom_MPa", "P16": "Mesh_Nodes", "P18": "Skewness_Max"})

# Keep only the columns we need
results = results[["Design_ID", "Mass_kg", "Stress_Fillet_MPa", "Stress_Hole_MPa",
    "Tip_Deflection_mm", "Stress_Leg_Bottom_MPa", "Mesh_Nodes", "Skewness_Max"]]

# Join inputs and results by Design_ID
data = inputs.merge(results, on="Design_ID")
print("Rows after joining:", len(data))

# Quality check 1: Ansys mass vs hand formula
data["Mass_Error_pct"] = (data["Mass_kg"] - data["Expected_Mass_kg"]) / data["Expected_Mass_kg"] * 100
print("Largest mass error (%):", round(data["Mass_Error_pct"].abs().max(), 4))

# Quality check 2: rows with high mesh skewness
print("Rows with skewness above 0.95:", list(data.loc[data["Skewness_Max"] > 0.95, "Design_ID"]))

print(data[["Design_ID", "Set", "Plate_Thickness", "Mass_kg", "Stress_Leg_Bottom_MPa", "Tip_Deflection_mm"]].head())

data.to_csv("bracket_dataset.csv", index=False)
print("Saved bracket_dataset.csv")