# Step 5b: First look at the data - stress vs each design variable
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("bracket_dataset.csv")
inputs = ["Plate_Thickness", "Fillet_Radius", "Leg_Length", "Hole_Diameter", "Rib_Thickness"]

fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)

for ax, name in zip(axes, inputs):
    ax.scatter(data[name], data["Stress_Leg_Bottom_MPa"])
    ax.axhline(138, color="red", linestyle="--")
    ax.set_xlabel(name + " (mm)")

axes[0].set_ylabel("Leg-bottom stress (MPa)")
fig.suptitle("Leg-bottom stress vs each design variable (60 Ansys runs)")
plt.tight_layout()
plt.savefig("plot_01_stress_vs_inputs.png", dpi=150)
print("Saved plot_01_stress_vs_inputs.png")