# L-Bracket Stress Predictor: a Gaussian Process surrogate of Ansys FEA
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from bracket_3d import bracket_3d

YIELD = 276            # MPa, Al 6061-T6
TRAIN_LOAD = 100       # N, the load used in every Ansys run
FEATURES = ["Plate_Thickness", "Fillet_Radius", "Leg_Length", "Hole_Diameter", "Rib_Thickness"]
TARGETS = ["Stress_Leg_Bottom_MPa", "Stress_Hole_MPa", "Stress_Fillet_MPa", "Tip_Deflection_mm"]
STRESSES = ["Stress_Leg_Bottom_MPa", "Stress_Hole_MPa", "Stress_Fillet_MPa"]

NOMINAL = {"t": 5.0, "R": 8.0, "B": 100.0, "d": 8.0, "tr": 4.0}
OPTIMIZED = {"t": 3.33, "R": 3.17, "B": 120.0, "d": 10.55, "tr": 1.56}   # verified in Ansys (DP 68)

st.set_page_config(page_title="L-Bracket Stress Predictor", layout="wide")

# Keep the page bright while it updates (no grey fading when a slider moves)
st.markdown("<style>[data-stale='true'] {opacity: 1 !important; transition: none !important;}</style>",
            unsafe_allow_html=True)


@st.cache_resource
def train_models():
    data = pd.read_csv("bracket_dataset_v2.csv")
    scaler = StandardScaler().fit(data[FEATURES])
    X = scaler.transform(data[FEATURES])
    models = {}
    for target in TARGETS:
        kernel = ConstantKernel(1.0) * RBF(length_scale=[1, 1, 1, 1, 1], length_scale_bounds=(0.01, 1000)) + WhiteKernel(1e-3)
        model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10, random_state=0)
        model.fit(X, data[target])
        models[target] = model
    return scaler, models, len(data)


def bracket_mass_g(t, R, B, d, tr):
    k = 1 - np.pi / 4
    volume = 40 * (B*t + (60 - t)*t + k*R**2) - (np.pi/4)*d**2*t + tr*((40 - 2*t)**2/2 - k*R**2)
    return volume * 2.7e-3


METAL = "#7f93a8"      # drawing colours chosen for the dark app background
RIB = "#5d7187"
EDGE = "#e6edf3"
LABEL = "#c3c2b7"
RED = "#e66767"


def draw_bracket(t, R, B, d, tr, load):
    fig, (side, top) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})

    # ---- Side view: L-profile, fillet, rib ----
    side.add_patch(Polygon([(t, t), (40 - t, t), (t, 40 - t)], closed=True, facecolor=RIB, edgecolor=EDGE, linewidth=1))
    arc = np.linspace(-np.pi / 2, -np.pi, 30)
    fillet = [(t + R + R * np.cos(a), t + R + R * np.sin(a)) for a in arc]
    outline = [(0, 0), (B, 0), (B, t)] + [(t + R, t)] + fillet + [(t, 60), (0, 60)]
    side.add_patch(Polygon(outline, closed=True, facecolor=METAL, edgecolor=EDGE, linewidth=1.5))
    hx = B - 20
    for x in (hx - d / 2, hx + d / 2):
        side.plot([x, x], [0, t], color=EDGE, linestyle="--", linewidth=1)
    side.annotate("", xy=(B - 1, t + 1), xytext=(B - 1, t + 22),
                  arrowprops=dict(arrowstyle="-|>", color=RED, linewidth=2))
    side.text(B - 3, t + 24, f"{load:.0f} N", color=RED, ha="right", fontsize=11)
    for y in range(0, 61, 6):
        side.plot([-6, 0], [y - 4, y], color=LABEL, linewidth=1)
    side.text(-8, 64, "fixed wall", color=LABEL, fontsize=10)
    side.set_title("Side view", fontsize=12, loc="left", color=LABEL)
    side.set_xlim(-12, 128)
    side.set_ylim(-8, 70)

    # ---- Top view: horizontal leg, hole, rib ----
    top.add_patch(Rectangle((0, -20), B, 40, facecolor=METAL, edgecolor=EDGE, linewidth=1.5))
    top.add_patch(Rectangle((0, -20), t, 40, facecolor=RIB, edgecolor=EDGE, linewidth=1))
    top.add_patch(Rectangle((t, -tr / 2), 40 - 2 * t, tr, facecolor=RIB, edgecolor=EDGE, linewidth=1))
    top.add_patch(Circle((hx, 0), d / 2, facecolor="#0e1117", edgecolor=EDGE, linewidth=1.5))
    top.set_title("Top view", fontsize=12, loc="left", color=LABEL)
    top.set_xlim(-8, 128)
    top.set_ylim(-28, 28)

    fig.patch.set_alpha(0)          # transparent background, blends with the app
    for ax in (side, top):
        ax.set_facecolor("none")
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout()
    return fig


@st.cache_data
def candidate_stresses(leg_length):
    """Predict the worst stress (incl. uncertainty) at 100 N for 20,000 random brackets with this leg length."""
    rng = np.random.default_rng(0)
    n = 20000
    cand = pd.DataFrame({
        "Plate_Thickness": rng.uniform(3, 8, n),
        "Fillet_Radius": rng.uniform(2, 15, n),
        "Leg_Length": np.full(n, leg_length),
        "Hole_Diameter": rng.uniform(6, 12, n),
        "Rib_Thickness": rng.uniform(1.5, 6, n),
    }).round(2)
    X = scaler.transform(cand[FEATURES])
    uppers = []
    for target in STRESSES:
        pred, std = models[target].predict(X, return_std=True)
        uppers.append(pred + 2 * std)
    cand["Worst_Upper_100N"] = np.max(uppers, axis=0)
    cand["Mass_g"] = bracket_mass_g(cand["Plate_Thickness"], cand["Fillet_Radius"], cand["Leg_Length"],
                                    cand["Hole_Diameter"], cand["Rib_Thickness"])
    return cand


def set_design(values):
    """Move all five sliders to the given design (used by the preset and optimizer buttons)."""
    for key, value in values.items():
        st.session_state[key] = value


scaler, models, n_designs = train_models()

# Slider start values (only set the first time the page opens)
for key, value in NOMINAL.items():
    st.session_state.setdefault(key, value)

# ---- Sidebar: presets, dimensions, load case ----
st.sidebar.header("Quick start")
p1, p2 = st.sidebar.columns(2)
p1.button("Nominal design", on_click=set_design, args=(NOMINAL,), width="stretch")
p2.button("Optimized design", on_click=set_design, args=(OPTIMIZED,), width="stretch")

st.sidebar.header("Bracket dimensions")
t = st.sidebar.slider("Plate thickness (mm)", 3.0, 8.0, step=0.01, key="t")
R = st.sidebar.slider("Fillet radius (mm)", 2.0, 15.0, step=0.01, key="R")
B = st.sidebar.slider("Leg length (mm)", 70.0, 120.0, step=0.1, key="B")
d = st.sidebar.slider("Hole diameter (mm)", 6.0, 12.0, step=0.01, key="d")
tr = st.sidebar.slider("Rib thickness (mm)", 1.5, 6.0, step=0.01, key="tr")
st.sidebar.caption("Sliders are limited to the ranges covered by the Ansys training data.")

st.sidebar.header("Load case")
load = st.sidebar.number_input("Tip load (N)", min_value=10.0, max_value=200.0, value=100.0, step=5.0)
sf = st.sidebar.number_input("Safety factor on yield", min_value=1.2, max_value=4.0, value=2.0, step=0.1)
allowable = YIELD / sf
scale = load / TRAIN_LOAD     # linear elastic: stress and deflection scale with load

# ---- Predictions (the GP predicts at 100 N; results are scaled to the chosen load) ----
X_new = scaler.transform(pd.DataFrame([[t, R, B, d, tr]], columns=FEATURES))
results = {}
for target in TARGETS:
    pred, std = models[target].predict(X_new, return_std=True)
    results[target] = (pred[0] * scale, 2 * std[0] * scale)

leg, leg_pm = results["Stress_Leg_Bottom_MPa"]
hole, hole_pm = results["Stress_Hole_MPa"]
fil, fil_pm = results["Stress_Fillet_MPa"]
defl, defl_pm = results["Tip_Deflection_mm"]
mass = bracket_mass_g(t, R, B, d, tr)
nominal_mass = bracket_mass_g(NOMINAL["t"], NOMINAL["R"], B, NOMINAL["d"], NOMINAL["tr"])
saving = (nominal_mass - mass) / nominal_mass * 100

# ---- Safety verdict (uses the upper end of each uncertainty band) ----
worst_name, worst_upper = max(
    [("Leg-bottom", leg + leg_pm), ("Hole", hole + hole_pm), ("Fillet", fil + fil_pm)],
    key=lambda item: item[1])
usage = worst_upper / allowable * 100
safe = worst_upper <= allowable

# ---- Header ----
st.title("L-Bracket Stress Predictor")
st.write(f"Machine-learning surrogate trained on {n_designs} Ansys FEA simulations. "
         "Aluminium 6061-T6, tip load, wall fixed, width 40 mm, wall height 60 mm.")

# ---- Layout: results on the left, rotatable 3D model on the right ----
left, right = st.columns([1.1, 1])

with left:
    st.subheader("Predicted results")
    c1, c2 = st.columns(2)
    c1.metric(f"Leg-bottom stress (± {leg_pm:.1f})", f"{leg:.1f} MPa")
    c2.metric(f"Hole stress (± {hole_pm:.1f})", f"{hole:.1f} MPa")
    c3, c4 = st.columns(2)
    c3.metric(f"Fillet stress (± {fil_pm:.1f})", f"{fil:.1f} MPa")
    c4.metric(f"Tip deflection (± {defl_pm:.2f})", f"{defl:.2f} mm")
    c5, c6 = st.columns(2)
    if abs(saving) < 0.5:
        mass_note = None
    elif saving > 0:
        mass_note = f"{saving:.0f}% lighter than nominal"
    else:
        mass_note = f"-{-saving:.0f}% heavier than nominal"
    c5.metric("Mass", f"{mass:.1f} g", mass_note)
    c6.metric(f"Allowable stress (yield ÷ {sf:.1f})", f"{allowable:.0f} MPa")

    st.subheader("Safety check")
    if safe:
        st.success(f"✅ SAFE: highest stress (incl. uncertainty) = {worst_upper:.1f} MPa "
                   f"at the {worst_name.lower()}, {usage:.0f}% of the {allowable:.0f} MPa allowable.")
    else:
        st.error(f"❌ NOT SAFE: highest stress (incl. uncertainty) = {worst_upper:.1f} MPa "
                 f"at the {worst_name.lower()}, {usage:.0f}% of the {allowable:.0f} MPa allowable.")
    st.progress(min(usage / 100, 1.0), text=f"Stress used: {usage:.0f}% of allowable")
    if worst_upper > YIELD:
        st.warning("⚠️ Predicted stress is above the 276 MPa yield strength. The linear model is no longer "
                   "valid here, so the real stress could be different. Reduce the load or strengthen the bracket.")
    if max(leg_pm, hole_pm, fil_pm) > 5 * scale:
        st.warning("⚠️ Low confidence: the model has little training data near this design. "
                   "Check it with an FEA run.")

with right:
    st.subheader("3D model")
    st.caption("Drag to rotate · scroll to zoom · right-drag to pan")
    model_color = "#7f93a8" if safe else "#d03b3b"
    st.plotly_chart(bracket_3d(t, R, B, d, tr, model_color), width="stretch")

# ---- Optimizer: lightest safe bracket for this leg length and load case ----
st.subheader("Find the lightest safe design")
st.write(f"Searches 20,000 bracket designs with the leg fixed at **{B:.1f} mm**, "
         f"for a **{load:.0f} N** load and safety factor **{sf:.1f}**.")
if st.button("🔍 Find lightest safe design", type="primary"):
    cand = candidate_stresses(round(B, 1))
    ok = cand[cand["Worst_Upper_100N"] * scale <= allowable]
    st.session_state["best_settings"] = (round(B, 1), load, sf)
    if ok.empty:
        st.session_state["best"] = None
    else:
        best = ok.sort_values("Mass_g").iloc[0]
        st.session_state["best"] = {"t": float(best["Plate_Thickness"]), "R": float(best["Fillet_Radius"]),
                                    "B": float(best["Leg_Length"]), "d": float(best["Hole_Diameter"]),
                                    "tr": float(best["Rib_Thickness"]), "mass": float(best["Mass_g"]),
                                    "stress": float(best["Worst_Upper_100N"] * scale)}

if "best" in st.session_state and st.session_state.get("best_settings") != (round(B, 1), load, sf):
    del st.session_state["best"]          # leg length, load or safety factor changed: old answer no longer valid

if "best" in st.session_state:
    best = st.session_state["best"]
    if best is None:
        st.error("No safe design exists in the trained ranges for this leg length and load case.")
    else:
        st.info(f"Lightest safe design: thickness **{best['t']:.2f}**, fillet **{best['R']:.2f}**, "
                f"leg **{best['B']:.1f}**, hole **{best['d']:.2f}**, rib **{best['tr']:.2f}** mm  →  "
                f"**{best['mass']:.1f} g**, highest stress {best['stress']:.1f} MPa (incl. uncertainty).")
        design = {k: best[k] for k in ("t", "R", "B", "d", "tr")}
        st.button("Apply this design to the sliders", on_click=set_design, args=(design,))
        st.caption("This is a surrogate prediction. Always verify the final design with an FEA run "
                   "(the 120 mm / 100 N optimum was verified in Ansys at 123.5 MPa).")

# ---- 2D drawings (to scale) ----
with st.expander("2D drawings: side and top view", expanded=True):
    fig = draw_bracket(t, R, B, d, tr, load)
    st.pyplot(fig)
    plt.close(fig)

with st.expander("About this model and its limits"):
    st.markdown(f"""
- **What it is:** a Gaussian Process surrogate trained on {n_designs} Ansys Mechanical simulations
  (60 Latin Hypercube designs + 7 adaptive designs in the thin, long-leg corner), all at a 100 N tip load.
- **Accuracy:** about 1 MPa average error on 10 unseen test designs (R² = 0.997 for leg-bottom stress).
- **Other loads:** results are scaled linearly from 100 N. This is exact for linear elastic behaviour,
  i.e. while stresses stay below the 276 MPa yield strength.
- **Allowable:** 276 MPa yield ÷ safety factor. The verdict uses prediction + uncertainty band.
- **Valid only inside the slider ranges**, for this load case (tip load, wall fixed, Al 6061-T6).
- **Not modelled:** welds, fatigue, the sharp rib-toe singularity, buckling.
- **Final designs must always be verified with FEA.**
- **Data source:** Ansys Mechanical 2026 R1 Student.
""")


# ---------------- Independent Ansys validation (added 7 Nov 2026) ----------------
with st.expander("Independent Ansys validation: 3 new cases"):
    st.markdown(
        "Three new designs (not in the training data) were solved in Ansys Mechanical "
        "at the actual load, with the same mesh rules, and compared with this app."
    )
    validation = pd.DataFrame({
        "Case": ["1", "2", "3"],
        "Design (mm)*": ["4.5 / 6 / 105 / 9 / 3", "3.5 / 5 / 115 / 10 / 2", "3.5 / 5 / 115 / 10 / 2"],
        "Load (N)": ["140", "140", "80"],
        "Leg stress, app → Ansys (MPa)": ["75.9 → 77.05", "153.5 → 153.85", "87.7 → 87.92"],
        "Error": ["1.5 %", "0.2 %", "0.2 %"],
        "Deflection, app → Ansys (mm)": ["1.12 → 1.107", "3.16 → 3.152", "1.81 → 1.801"],
        "Verdict, app / Ansys": ["SAFE / SAFE", "NOT SAFE / NOT SAFE", "SAFE / SAFE"],
    })
    st.table(validation.set_index("Case"))
    st.caption("*Design = plate thickness / fillet radius / leg length / hole diameter / rib thickness")
    st.markdown(
        "- Governing (leg-bottom) stress within **1.5 %** of Ansys in all three cases; "
        "deflection within **1 %**.\n"
        "- All three SAFE / NOT SAFE verdicts agree with Ansys.\n"
        "- In Ansys, Case 3 ÷ Case 2 = 0.571 = 80 ÷ 140, confirming the linear load scaling.\n"
        "- Known limitation: fillet stress is over-predicted by about 6 % for the thin design "
        "(on the safe side)."
    )

# ---------------- Footer: author ----------------
st.divider()
st.caption(
    "Built by **Hafiz Muhammad Ibrahim**, Mechanical Design Engineer · "
    "[LinkedIn](https://www.linkedin.com/in/engrmbrahim) · "
    "[GitHub](https://github.com/engribrahim007)"
)