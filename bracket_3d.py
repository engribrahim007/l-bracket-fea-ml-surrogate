# 3D model of the L-bracket (rotatable with the mouse), built from the 5 design values
import numpy as np
import plotly.graph_objects as go


def prism(points_2d, z0, z1, plane="xy"):
    """Extrude a fan-triangulable 2D polygon between z0 and z1. Returns vertices and triangles."""
    n = len(points_2d)
    verts, tris = [], []
    for z in (z0, z1):
        for (a, b) in points_2d:
            verts.append((a, b, z) if plane == "xy" else (a, z, b))
    for i in range(1, n - 1):
        tris.append((0, i, i + 1))
        tris.append((n, n + i + 1, n + i))
    for i in range(n):
        j = (i + 1) % n
        tris.append((i, j, n + j))
        tris.append((i, n + j, n + i))
    return verts, tris


def plate_with_hole(B, t, d, n=72):
    """Horizontal leg: x 0..B, z -20..20, thickness y 0..t, round hole at x = B-20."""
    cx, r = B - 20, d / 2
    corners = [(B, -20), (B, 20), (0, 20), (0, -20)]
    angles = sorted(set(np.linspace(0, 2 * np.pi, n, endpoint=False).tolist() +
                        [np.arctan2(z - 0, x - cx) % (2 * np.pi) for x, z in corners]))
    inner, outer = [], []
    for a in angles:
        dx, dz = np.cos(a), np.sin(a)
        inner.append((cx + r * dx, r * dz))
        s = min([(B - cx) / dx if dx > 1e-9 else np.inf, (0 - cx) / dx if dx < -1e-9 else np.inf,
                 20 / dz if dz > 1e-9 else np.inf, -20 / dz if dz < -1e-9 else np.inf])
        outer.append((cx + s * dx, s * dz))
    m = len(angles)
    verts, tris = [], []
    for y in (0, t):
        verts += [(x, y, z) for x, z in inner] + [(x, y, z) for x, z in outer]
    def idx(layer, ring, i):
        return layer * 2 * m + ring * m + (i % m)
    for i in range(m):
        for layer in (0, 1):
            a, b = idx(layer, 0, i), idx(layer, 0, i + 1)
            c, e = idx(layer, 1, i), idx(layer, 1, i + 1)
            tris += [(a, b, e), (a, e, c)]
        for ring in (0, 1):
            a, b = idx(0, ring, i), idx(0, ring, i + 1)
            c, e = idx(1, ring, i), idx(1, ring, i + 1)
            tris += [(a, b, e), (a, e, c)]
    return verts, tris


def bracket_3d(t, R, B, d, tr, color):
    parts = []
    parts.append(prism([(0, 0), (t, 0), (t, 60), (0, 60)], -20, 20))
    parts.append(plate_with_hole(B, t, d))
    arc = np.linspace(-np.pi / 2, -np.pi, 24)
    parts.append(prism([(t, t)] + [(t + R + R * np.cos(a), t + R + R * np.sin(a)) for a in arc], -20, 20))
    parts.append(prism([(t, t), (40 - t, t), (t, 40 - t)], -tr / 2, tr / 2))

    fig = go.Figure()
    for verts, tris in parts:
        v, f = np.array(verts), np.array(tris)
        fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 2], z=v[:, 1], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                color=color, flatshading=True, hoverinfo="skip",
                                lighting=dict(ambient=0.45, diffuse=0.8, specular=0.3, roughness=0.5)))
    fig.update_layout(
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode="data",
                   camera=dict(eye=dict(x=1.3, y=-1.6, z=0.9))),
        margin=dict(l=0, r=0, t=0, b=0), height=420,
        paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return fig