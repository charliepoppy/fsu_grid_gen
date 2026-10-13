#!/usr/bin/env python3
"""
FSU intake lower/upper wall construction from physics + a few fixed inputs.

    ramp 1 (theta0) -> focused isentropic turn (MoC, simple wave) -> ramp 2 (theta1)
    -> shoulder -> 9 deg facet -> isolator floor
    cowl lip -> 9 deg cowl -> isolator ceiling

solve_geometry() is the single source of truth; plotting and validation only read it.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from geometry.gasdynamics import pm_nu, pm_M, oblique_shock
from geometry.add_notch import add_notch
from geometry.geometry import  solve_geometry
from geometry.lines import intersect, ang
from geometry.plots import plot_cad_residual, plot_MoC, plot_overlay_zooms, draw_construction

from utils.style import set_style
set_style()

HERE = Path(__file__).resolve().parent

# ======================================================================
# Inputs
# ======================================================================
CFG = {
    "Minf": 3.0,
    "theta0": 6.67,                    # ramp 1 [deg]              (CAD)
    "theta1": 20.8,                    # ramp 2 [deg]              (CAD)
    "cowl_angle": 9.0,                 # cowl   [deg]               (spec)
    "facet_angle": 9.0,                # facet  [deg]               (spec)
    "leading_edge": (0.0, 0.0992),     # ramp tangent point on r = 0.1 LE radius (CAD frame)
    "cowl_lip": (140.4, 50.0),         # specified
    "focus": (140.696, 49.696),        # MoC wave focus, least-squares fit to CAD (0.43 from lip)
    "shoulder_x": 156.1754,            # CAD spline end, full precision
    "ceiling": 56.44,                  # measured from sketch (CAD-anchored calibration), +-0.1
    # "floor": 39.12,                    # measured from sketch, +-0.1
    "floor": 39.10,                    # measured from sketch, +-0.1
    "iso_length": 90.0,                # measured from facet end (ASSUMPTION - check FSU paper)

    "notch_depth": 1.0,       # vertical step at the shoulder [mm]
    "notch_angle": 10.6,      # notched floor angle [deg], from step bottom to the isolator floor
}

# References the construction is validated against
REF = {
    "H_t": 17.34,                                   # FSU isolator height
    "cad_curve_end": (156.1754, 34.8450),          # CAD, full precision
    "digitised": {                                  # hand-digitised / sketch-measured
        # "curve end": (155.7, 34.845),
        "curve end": (156.175,  34.831),
        "cowl end": (181.06, 56.44),
        "facet end": (183.19, 39.05),
    },
    "cad_curve_csv": HERE / "zz_archive" / "spline_samples2.csv",   # full-precision CAD samples
}

TOL = {"angle": 0.01, "point": 0.10, "cad": 0.05, "H": 0.05, "focus": 1e-6}


# ======================================================================
# Validation
# ======================================================================
def validate(G, cfg, ref, tol, out_path=None):
    P, W = G["pts"], G["W"]
    rows = []                                    # (group, check, value, target, diff, tol, unit)

    def add(group, name, val, target, t, unit):
        rows.append((group, name, val, target, val - target, t, unit))

    # --- angles: measured from the constructed points ---
    add("angle", "ramp 1 (LE -> turn start)",     ang(P["leading edge"], P["turn start"]), cfg["theta0"], tol["angle"], "deg")
    add("angle", "turn first segment",            ang(W[0], W[1]),                         cfg["theta0"], tol["angle"], "deg")
    add("angle", "turn last segment",             ang(W[-2], W[-1]),                       cfg["theta1"], tol["angle"], "deg")
    add("angle", "ramp 2 (turn end -> shoulder)", ang(P["turn end"], P["shoulder"]),       cfg["theta1"], tol["angle"], "deg")
    add("angle", "facet (shoulder -> facet end)", ang(P["shoulder"], P["facet end"]),      cfg["facet_angle"], tol["angle"], "deg")
    add("angle", "cowl (lip -> cowl end)",        ang(P["cowl lip"], P["cowl end"]),       cfg["cowl_angle"], tol["angle"], "deg")
    add("angle", "isolator floor",                ang(P["facet end"], P["outlet floor"]),  0.0, tol["angle"], "deg")
    add("angle", "isolator ceiling",              ang(P["cowl end"], P["outlet ceiling"]), 0.0, tol["angle"], "deg")

    # --- MoC self-consistency: every wave passes through F ---
    d = W - G["F"]
    miss = np.abs(d[:, 0] * np.sin(G["phi"]) - d[:, 1] * np.cos(G["phi"]))
    add("MoC", "max wave miss distance from focus", miss.max(), 0.0, tol["focus"], "")

    # --- constraints ---
    add("constraint", "isolator height H", G["ceiling"] - G["floor"], ref["H_t"], tol["H"], "")
    add("constraint", "shoulder vs CAD curve end (y)", P["shoulder"][1], ref["cad_curve_end"][1], tol["cad"], "")

    # --- notch (only if add_notch has run) ---
    if "lower_notched" in G:
        add("notch", "step height", P["shoulder"][1] - P["step bottom"][1], cfg["notch_depth"], 1e-9, "")
        add("notch", "step face vertical (dx)", P["step bottom"][0] - P["shoulder"][0], 0.0, 1e-12, "")
        add("notch", "notch floor angle", ang(P["step bottom"], P["notch end"]), cfg["notch_angle"], tol["angle"], "deg")
        add("notch", "notch end on isolator floor", P["notch end"][1], G["floor"], 1e-12, "")
        # the two walls must be identical upstream of the step and downstream of the notch end
        ln, lw = G["lower_notched"], G["lower"]
        xs = lw[:, 0][(lw[:, 0] < P["shoulder"][0]) | (lw[:, 0] >= P["notch end"][0])]
        add("notch", "walls identical outside notch",
            np.abs(np.interp(xs, ln[:, 0], ln[:, 1]) - np.interp(xs, lw[:, 0], lw[:, 1])).max(), 0.0, 1e-12, "")

    # --- MoC wall vs full-precision CAD curve ---
    cad = None
    if Path(ref["cad_curve_csv"]).exists():
        cad = np.loadtxt(ref["cad_curve_csv"], delimiter=",", skiprows=1)
        lw = G["lower"]
        r = cad[:, 1] - np.interp(cad[:, 0], lw[:, 0], lw[:, 1])
        add("CAD", "wall vs CAD curve, max |dy|", np.abs(r).max(), 0.0, tol["cad"], "")
        add("CAD", "wall vs CAD curve, rms dy", np.sqrt(np.mean(r**2)), 0.0, tol["cad"], "")
        G["cad_residual"] = (cad[:, 0], r)

    # --- corners vs digitised ---
    pt_rows = []
    for name, p_ref in ref["digitised"].items():
        key = "shoulder" if name == "curve end" else name
        p = P[key]
        dx, dy = p[0] - p_ref[0], p[1] - p_ref[1]
        pt_rows.append((name, p, p_ref, dx, dy, np.hypot(dx, dy)))

    # --- report ---
    lines = ["", "=" * 92, "GEOMETRY VALIDATION", "=" * 92,
             f"{'group':<11}{'check':<36}{'value':>11}{'target':>11}{'diff':>11}{'tol':>8}  ok",
             "-" * 92]
    n_fail = 0
    for g, name, val, tgt, diff, t, unit in rows:
        ok = abs(diff) <= t
        n_fail += not ok
        lines.append(f"{g:<11}{name:<36}{val:>11.4f}{tgt:>11.4f}{diff:>+11.4f}{t:>8.2g}  {'PASS' if ok else 'FAIL'}")

    lines += ["", f"{'corner':<12}{'constructed':>22}{'digitised':>22}{'dx':>9}{'dy':>9}{'|d|':>8}  ok",
              "-" * 92]
    for name, p, p_ref, dx, dy, dd in pt_rows:
        ok = dd <= tol["point"]
        n_fail += not ok
        lines.append(f"{name:<12}  ({p[0]:8.3f},{p[1]:8.3f})  ({p_ref[0]:8.3f},{p_ref[1]:8.3f})"
                     f"{dx:>+9.3f}{dy:>+9.3f}{dd:>8.3f}  {'PASS' if ok else 'FAIL'}")

    lines += ["", "derived (information)", "-" * 92]
    for name, p in P.items():
        lines.append(f"  {name:<24} ({p[0]:9.4f}, {p[1]:9.4f})")
    lines += [f"  focus offset from lip      {np.hypot(*(G['F'] - P['cowl lip'])):.4f}",
              f"  LE shock   beta {np.degrees(G['LE_shock']['beta']):.3f} deg, M2 {G['LE_shock']['M2']:.4f}",
              f"  turn       M {G['M'][0]:.4f} -> {G['M'][-1]:.4f}",
              f"  cowl shock beta {np.degrees(G['cowl_shock']['beta']):.3f} deg, M3 {G['cowl_shock']['M2']:.4f}",
              "", f"{n_fail} FAIL(s)", "=" * 92]
    report = "\n".join(lines)
    print(report)
    if out_path:
        Path(out_path).write_text(report + "\n")
    return n_fail == 0





def image_extent(img, px1, xy1, px2, xy2):
    """Map pixel centres (col, row) to physical coords from two reference points."""
    (u1, v1), (u2, v2) = px1, px2
    (X1, Y1), (X2, Y2) = xy1, xy2
    sx, sy = (X2 - X1) / (u2 - u1), (Y2 - Y1) / (v2 - v1)
    h, w = img.shape[:2]
    left, top = X1 - sx * u1, Y1 - sy * v1
    return [left, left + sx * w, top + sy * h, top], (sx, sy)



# ======================================================================
# Export (consumed by mesh.py)
# ======================================================================
def export_geometry(G, cfg, validated, out_dir):
    """
    geometry.json : inputs, named points, scalars, validation flag  (human-readable)
    walls.npz     : dense wall polylines (lower, upper) + MoC turn data
    """
    import json, hashlib, datetime
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
 
    extra = {"lower_notched": G["lower_notched"]} if "lower_notched" in G else {}
    np.savez(out_dir / "walls.npz", lower=G["lower"], upper=G["upper"],
             W=G["W"], M=G["M"], phi=G["phi"], focus=G["F"], **extra)
    walls_hash = hashlib.sha256((out_dir / "walls.npz").read_bytes()).hexdigest()[:16]
 
    doc = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "validated": bool(validated),
        "walls_npz_sha256": walls_hash,
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "points": {k: [float(v[0]), float(v[1])] for k, v in G["pts"].items()},
        "scalars": {"ceiling": float(G["ceiling"]), "floor": float(G["floor"]),
                    "x_out": float(G["x_out"]), "lip_y": float(G["pts"]["cowl lip"][1]),
                    "focus": [float(G["F"][0]), float(G["F"][1])]},
    }
    (out_dir / "geometry.json").write_text(json.dumps(doc, indent=2))
    print(f"geometry exported -> {out_dir}  (validated={validated}, walls sha {walls_hash})")
 
 

# ======================================================================
if __name__ == "__main__":
    fig_dir = HERE / "figs"
    fig_dir.mkdir(exist_ok=True)
 
    G = add_notch(solve_geometry(CFG), CFG)
    # print(G["lower_notched"].shape)        # expect (4005, 2): LE + 4000 MoC points + S, B, E, outlet
    # print(G["lower_notched"][-5:])         # last 5 rows: shoulder, step bottom, notch end, outlet
    ok = validate(G, CFG, REF, TOL, out_path=fig_dir / "validation.txt")
    export_geometry(G, CFG, ok, HERE / "geometry")
    plot_cad_residual(G, fig_dir)
    plot_MoC(G, CFG, REF, fig_dir)
 
    img_path = HERE / "data" / "Inventor.png"
    if img_path.exists():
        img = mpimg.imread(img_path)
        # pixel centres; physical points include the +0.492/+0.054 offset fitted to the CAD ramp
        extent, _ = image_extent(img, (27.5, 324.5), (0.492, 0.054), (676.5, 93.5), (140.892, 50.054))
        plot_overlay_zooms(img, extent, G, CFG, REF, fig_dir)