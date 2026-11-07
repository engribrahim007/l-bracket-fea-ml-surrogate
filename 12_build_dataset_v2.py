# Build dataset version 2: 60 DOE + DP 61 (old optimum) + 6 adaptive = 67 designs
import numpy as np
import pandas as pd
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

def bracket_mass_kg(t, R, B, d, tr):
    k = 1 - np.pi / 4
    volume = 40 * (B*t + (60 - t)*t + k*R**2) - (np.pi/4)*d**2*t + tr*((40 - 2*t)**2/2 - k*R**2)
    return volume * 2.7e-6

# Inputs: the 60 DOE designs + the 6 adaptive designs
inputs = pd.concat([pd.read_csv("doe_inputs.csv"), pd.read_csv("doe_inputs_adaptive.csv")])

# Add DP 61 (the first optimum we verified in Ansys)
dp61 = pd.DataFrame([{"Design_ID": 61, "Set": "adaptive", "Plate_Thickness": 3.16, "Fillet_Radius": 4.19,
                      "Leg_Length": 120, "Hole_Diameter": 11.92, "Rib_Thickness": 1.55}])
dp61["Expected_Mass_kg"] = round(bracket_mass_kg(3.16, 4.19, 120, 11.92, 1.55), 6)
inputs = pd.concat([inputs, dp61])

# Ansys results (DP 0 to DP 67), with clear names
results = pd.read_csv("doe_results_adaptive.csv", comment="#")
results = results[results["Name"] != "DP 0"]
results["Design_ID"] = results["Name"].str.replace("DP ", "").astype(int)
results = results.rename(columns={"P8": "Mass_kg", "P9": "Stress_Fillet_MPa", "P10": "Stress_Hole_MPa",
    "P11": "Tip_Deflection_mm", "P12": "Stress_Leg_Bottom_MPa", "P16": "Mesh_Nodes", "P18": "Skewness_Max"})
results = results[["Design_ID", "Mass_kg", "Stress_Fillet_MPa", "Stress_Hole_MPa",
    "Tip_Deflection_mm", "Stress_Leg_Bottom_MPa", "Mesh_Nodes", "Skewness_Max"]]

# Join and check
data = inputs.merge(results, on="Design_ID").sort_values("Design_ID")
data["Mass_Error_pct"] = (data["Mass_kg"] - data["Expected_Mass_kg"]) / data["Expected_Mass_kg"] * 100
print("Rows:", len(data))
print(data["Set"].value_counts())
print("Largest mass error (%):", round(data["Mass_Error_pct"].abs().max(), 4))
print(data[data["Set"] == "adaptive"][["Design_ID", "Plate_Thickness", "Leg_Length", "Mass_kg", "Stress_Leg_Bottom_MPa"]].to_string(index=False))

data.to_csv("bracket_dataset_v2.csv", index=False)
print("Saved bracket_dataset_v2.csv")