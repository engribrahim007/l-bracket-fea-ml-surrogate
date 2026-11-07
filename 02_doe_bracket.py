# Step 4: Design of Experiments (DOE) for the L-bracket
import numpy as np
import pandas as pd
from scipy.stats import qmc
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# Design variables and their ranges (mm)
names = ["Plate_Thickness", "Fillet_Radius", "Leg_Length", "Hole_Diameter", "Rib_Thickness"]
lower = [3, 2, 70, 6, 1.5]
upper = [8, 15, 120, 12, 6]

# Latin Hypercube: 50 designs for training, 10 separate designs for testing
train = qmc.scale(qmc.LatinHypercube(d=5, rng=1).random(n=50), lower, upper)
test = qmc.scale(qmc.LatinHypercube(d=5, rng=2).random(n=10), lower, upper)

# Put all 60 designs in one table
df = pd.DataFrame(np.vstack([train, test]), columns=names).round(2)
df.insert(0, "Set", ["train"] * 50 + ["test"] * 10)
df.insert(0, "Design_ID", range(1, 61))

# ---- Convert design variables into the Workbench parameters ----
df["P1_Hole_Radius"] = df["Hole_Diameter"] / 2
df["P2_Fillet_Radius"] = df["Fillet_Radius"]
df["P3_Rib_Half_Front"] = df["Rib_Thickness"] / 2
df["P4_Rib_Half_Back"] = df["Rib_Thickness"] / 2
df["P5_Leg_Length"] = df["Leg_Length"]
df["P6_Thickness_Horizontal_Leg"] = df["Plate_Thickness"]
df["P7_Thickness_Wall_Leg"] = df["Plate_Thickness"]

# ---- Mesh sizes from our locked mesh rules ----
df["P13_Mesh_Global"] = (df["Plate_Thickness"] / 1.5).round(2)
df["P14_Mesh_Fillet"] = (df["Fillet_Radius"] / 6).round(2)
df["P15_Mesh_Hole"] = (df["P1_Hole_Radius"] * 0.262).round(2)

# ---- Expected mass from the hand formula (to check Ansys later) ----
def bracket_mass_kg(t, R, B, d, tr):
    k = 1 - np.pi / 4
    volume = 40 * (B*t + (60 - t)*t + k*R**2) - (np.pi/4)*d**2*t + tr*((40 - 2*t)**2/2 - k*R**2)
    return volume * 2.7e-6

print("Check nominal design:", round(bracket_mass_kg(5, 8, 100, 8, 4), 6))

df["Expected_Mass_kg"] = bracket_mass_kg(df["Plate_Thickness"], df["Fillet_Radius"],
                                         df["Leg_Length"], df["Hole_Diameter"],
                                         df["Rib_Thickness"]).round(6)

print(df.head(5))
print(df["Expected_Mass_kg"].describe())

df.to_csv("doe_inputs.csv", index=False)
print("Saved doe_inputs.csv with", len(df), "designs")