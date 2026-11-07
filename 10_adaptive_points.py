# Adaptive sampling: 6 new designs in the thin + long-leg corner
import numpy as np
import pandas as pd
from scipy.stats import qmc
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)

names = ["Plate_Thickness", "Fillet_Radius", "Leg_Length", "Hole_Diameter", "Rib_Thickness"]
lower = [3, 2, 110, 6, 1.5]       # corner: thin plate, long leg
upper = [3.8, 15, 120, 12, 6]

new = pd.DataFrame(qmc.scale(qmc.LatinHypercube(d=5, rng=3).random(n=6), lower, upper), columns=names).round(2)
new.insert(0, "Set", "adaptive")
new.insert(0, "Design_ID", range(62, 68))

# Same conversions as the main DOE
new["P1_Hole_Radius"] = new["Hole_Diameter"] / 2
new["P2_Fillet_Radius"] = new["Fillet_Radius"]
new["P3_Rib_Half_Front"] = new["Rib_Thickness"] / 2
new["P4_Rib_Half_Back"] = new["Rib_Thickness"] / 2
new["P5_Leg_Length"] = new["Leg_Length"]
new["P6_Thickness_Horizontal_Leg"] = new["Plate_Thickness"]
new["P7_Thickness_Wall_Leg"] = new["Plate_Thickness"]
new["P13_Mesh_Global"] = (new["Plate_Thickness"] / 1.5).round(2)
new["P14_Mesh_Fillet"] = (new["Fillet_Radius"] / 6).round(2)
new["P15_Mesh_Hole"] = (new["P1_Hole_Radius"] * 0.262).round(2)

def bracket_mass_kg(t, R, B, d, tr):
    k = 1 - np.pi / 4
    volume = 40 * (B*t + (60 - t)*t + k*R**2) - (np.pi/4)*d**2*t + tr*((40 - 2*t)**2/2 - k*R**2)
    return volume * 2.7e-6

new["Expected_Mass_kg"] = bracket_mass_kg(new["Plate_Thickness"], new["Fillet_Radius"], new["Leg_Length"],
                                          new["Hole_Diameter"], new["Rib_Thickness"]).round(6)
print(new.to_string(index=False))
new.to_csv("doe_inputs_adaptive.csv", index=False)
print("Saved doe_inputs_adaptive.csv")