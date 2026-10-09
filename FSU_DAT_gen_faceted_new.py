#!/usr/bin/env python3
"""FSU faceted intake, 4-block grid generator.

Blocks:  [3][2]      upper: freestream above the cowl line
         [0][1]      lower: intake ramp -> facet -> isolator
         split at the cowl lip (x = 140.4)
Coordinates are scaled to δ* (scale_factor = H_t·IBLT / (δ99/δ*)).
Output: grids/<casename>/  (.h5, params.json, gen.log, figs/)
"""

import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from Py_to_HDF5 import arrays_to_HDF5
from utils.compute import (
    build_upper_wall,
    build_lower_wall,
    linspace_with_point,
    hermite_quintic_polynomial,
    angle_deg,
    height,
    extract_isentropic_curve,
    spanwise_stretch,
    construct_3D_blocks,
)



from utils.plots import (
    print_summary_3D,
    plot_curves,
    plot_zeta,
    plot_poly_stretch,
    plot_upper_lower,
    plot_blocks_2D,
    plot_bz_x,
    spanwise_stretch_plot,
)

from utils.plots import set_style
set_style()


CONFIG = {
    "casename": "FSU facet periodic LES",
    
    
    "scaling": {"d99_to_dstar": 1.584, "IBLT": 0.07, "H_I": 17.3,},

"geometry": {
    "points": {"cowl lip":  [140.4, 50.0],
               "cowl end":  [180.0, 56.3],
               "leading edge":  [0.0, 0.0],
               "facet end": [182.4, 39.0]},
    "y_top": 80.0, "isolator_length": 90.0, "Lz": 25.0,
},

    "grid": {"nx": 800 * 3, "ny": 100 * 5, "nz": 100 * 2},

    "stretch":  {"lower": dict(s0=0.1738253, s1=0.16738253, c0=0.0, c1=0.0),
                 "upper": dict(s0=0.16738253, s1=2.0, c0=0.0, c1=0.1),
                 "spanwise": dict(stretch_type="uniform", bz_max=2.0, a=8.0)},
}





curve_x, curve_y = extract_isentropic_curve()

def build_points(cfg):
    """All geometry points: fixed inputs + derived. Never mutates cfg."""
    g = cfg["geometry"]

    pts = dict(g["points"])                          # fixed (copied)
    pts["leading edge"]    = [curve_x[0],  curve_y[0]]
    pts["curve end"]       = [curve_x[-1], curve_y[-1]]
    x_out                  = pts["facet end"][0] + g["isolator_length"]
    pts["isolator bottom"] = [x_out, pts["facet end"][1]]
    pts["isolator top"]    = [x_out, pts["cowl end"][1]]
    pts["top left"]        = [0.0,   g["y_top"]]
    pts["top right"]       = [x_out, g["y_top"]]
    pts["top middle"]      = [pts['cowl lip'][0], g["y_top"]]


    lip_x = pts["cowl lip"][0]
    pts["wall at lip"] = [lip_x, float(np.interp(lip_x, curve_x, curve_y))]   # B0/B1 interface foot
    pts["inflow split"] = [0.0, pts["cowl lip"][1]]                            # B0/B3 interface at inflow

    return pts, curve_x, curve_y


def plot_geometry(casename, pts, curve_x, curve_y, fig_dir):
    fig, ax = plt.subplots(figsize=(10, 4))

    # walls
    ax.plot(curve_x, curve_y, "k-", lw=1.2)                       # ramp + isentropic curve
    walls = [
        ["curve end", "facet end", "isolator bottom"],             # lower: facet + floor
        ["cowl lip", "cowl end", "isolator top"],                  # upper: cowl + ceiling
    ]
    for chain in walls:
        xs, ys = zip(*(pts[n] for n in chain))
        ax.plot(xs, ys, "k-", lw=1.2)

    # domain boundaries (not walls)
    bounds = [
        ["leading edge", "top left", "top right", "isolator top"],  # inflow + top + upper outlet
        ["isolator bottom", "isolator top"],                        # isolator outlet
        ["top middle", "wall at lip"],                        # isolator outlet
    ]
    for chain in bounds:
        xs, ys = zip(*(pts[n] for n in chain))
        ax.plot(xs, ys, "--", color="0.5", lw=0.8)

    # block interface upstream of the lip
    ax.plot([0.0, pts["cowl lip"][0]], [pts["cowl lip"][1]] * 2, ":", color="tab:orange", lw=1)

    for name, (px, py) in pts.items():
        ax.plot(px, py, "o", ms=5)
        ax.annotate(name, (px, py), textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"Geometry: {casename}")
    ax.grid(alpha=0.3)
    fig.savefig(fig_dir / "geometry.png", dpi=200, bbox_inches="tight")
    plt.close(fig)





HERE = Path(__file__).resolve().parent
out_dir = HERE / "grids" / CONFIG["casename"].lower()
fig_dir = out_dir / "figs"
fig_dir.mkdir(parents=True, exist_ok=True)

pts, curve_x, curve_y = build_points(CONFIG)
plot_geometry(CONFIG["casename"], pts, curve_x, curve_y, fig_dir)





# geom["lines"] = {
#     "cowl":       ("cowl_lip", "cowl_end"),
#     "facet":      ("curve_end", "facet_end"),
#     "upper_wall": ("cowl_end", "isolator_top"),
#     "lower_wall": ("facet_end", "isolator_bottom"),
# }


# geom["angles"] = {
#     "cowl":  angle_deg(geom["pts"]["cowl_lip"], geom["pts"]["cowl_end"]),
#     "facet": angle_deg(geom["pts"]["curve_end"], geom["pts"]["facet_end"]),
# }

# geom["isolator_height"] = height(geom["pts"]["isolator_top"], geom["pts"]["isolator_bottom"])

# print("Key points")
# for name, pt in geom["pts"].items():
#     print(f"{name}: {pt}")

# print(f"\nCowl angle: {geom['angles']['cowl']:.2f} deg")
# print(f"Facet angle: {geom['angles']['facet']:.2f} deg")
# print(f"Isolator height: {geom['isolator_height']:.2f}")





# save(plot_curves(grid["x"], geom["upper_wall"], geom["lower_wall"], geom["y_top"]), "plot_curves")









# # ========================================================================



# grid["eta"] = np.linspace(0.0, 1.0, grid["ny"], endpoint=True)
# grid["zeta"] = np.linspace(0.0, 1.0, grid["nz"], endpoint=True)
# grid["iota"] = np.linspace(0.0, 1.0, grid["nx"], endpoint=True)
# grid["x"], grid["Lx"], grid["nx_lip"] = linspace_with_point(
#     0.0,
#     geom["Lx"],
#     geom["pts"]["cowl_lip"][0],
#     grid["nx"],
# )



# # sanity check
# assert np.isclose(grid["x"][grid["nx_lip"]], geom["pts"]["cowl_lip"][0], atol=1e-3), \
#     "Cowl lip is not exactly on x-grid."




# # ======================================================================
# # Wall curves
# # ======================================================================

# geom["upper_wall"] = build_upper_wall(
#     grid["x"],
#     geom["pts"]["cowl_lip"][0],
#     geom["pts"]["cowl_end"][0],
#     geom["pts"]["cowl_lip"][1],
#     geom["pts"]["cowl_end"][1]
# )

# geom["lower_wall"] = build_lower_wall(
#     grid["x"],
#     geom["curve_x"],
#     geom["curve_y"],
#     geom["pts"]["facet_start"][0],
#     geom["pts"]["facet_end"][0],
#     geom["pts"]["facet_start"][1],
#     geom["pts"]["facet_end"][1]
# )

# geom["y_top"] = np.full_like(grid["x"], 80.0)




# # Stretching / mapping
# # ======================================================================

# stretch = {
#     "lower": {
#         "s0": 0.1738253,
#         "s1": 0.16738253,
#         "c0": 0.0,
#         "c1": 0.0,
#     },
#     "upper": {
#         "s0": 0.16738253,   # inherited from lower s1
#         "s1": 2.0,
#         "c0": 0.0,          # inherited from lower c1
#         "c1": 0.1,
#     },
#     "spanwise": {
#         "stetch_type": "uniform",
#         "bz_max": 2.0,
#         "a": 8.0,
#         "i0": grid["nx_lip"] / grid["nx"],
#     }
# }

# stretch["P_lower"] = hermite_quintic_polynomial(
#     grid["eta"],
#     stretch["lower"]["s0"],
#     stretch["lower"]["s1"],
#     stretch["lower"]["c0"],
#     stretch["lower"]["c1"]
# )

# stretch["P_upper"] = hermite_quintic_polynomial(
#     grid["eta"],
#     s0=stretch["upper"]["s0"],
#     s1=stretch["upper"]["s1"],
#     c0=stretch["upper"]["c0"],
#     c1=stretch["upper"]["c1"]
# )


# stretch["z"], stretch["zeta"], stretch["iota"], stretch["bz_x"] = spanwise_stretch(
#     stretch["spanwise"]["stetch_type"],
#     case["Lz"],
#     grid["nz"],
#     nx=grid["nx"],
#     bz_max=stretch["spanwise"]["bz_max"],
#     a=stretch["spanwise"]["a"],
#     i0=stretch["spanwise"]["i0"]
# )


# print("z:", stretch["z"].shape)
# print("bz_x:", stretch["bz_x"].shape)




# # ======================================================================
# # 2D mapped planes
# # ======================================================================

# mapped = {}

# H_lower = geom["upper_wall"] - geom["lower_wall"]
# H_upper = geom["y_top"] - geom["upper_wall"]

# H_ratio = H_lower / H_upper

# mapped["lower_2d"] = (
#     geom["lower_wall"] + H_lower * stretch["P_lower"][:, None]
# )

# mapped["upper_2d"] = (
#     # geom["upper_wall"] + H_upper *  H_ratio * stretch["P_upper"][:, None]
#     geom["upper_wall"] + H_upper * stretch["P_upper"][:, None]
# )

# blocks_2d = {
#     "B0": mapped["lower_2d"][:, :grid["nx_lip"] + 1],
#     "B1": mapped["lower_2d"][:, grid["nx_lip"]:],
#     "B2": mapped["upper_2d"][:, grid["nx_lip"]:],
#     "B3": mapped["upper_2d"][:, :grid["nx_lip"] + 1],
# }

# blocks_list = [blocks_2d["B0"], blocks_2d["B1"], blocks_2d["B2"], blocks_2d["B3"]]


# # ======================================================================
# # 3D blocks
# # ======================================================================

# z_B, y_B, x_B = construct_3D_blocks(
#     stretch["z"],
#     blocks_list,
#     grid["x"],
#     grid["nx_lip"],
#     case['scale_factor'],
# )






# # --------------------------- Visualisation --------------------------- 

# def save(fig, name):
#     fig.savefig(fig_dir / f"{name}.png", dpi=200, bbox_inches="tight")
#     plt.close(fig)


# # save(plot_curves(grid["x"], geom["upper_wall"], geom["lower_wall"], geom["y_top"]), "plot_curves")
# save(plot_zeta(stretch["z"][:, -1], stretch["zeta"]), "plot_zeta")
# save(plot_bz_x(stretch["iota"], stretch["bz_x"]), "plot_bz_x")
# save(spanwise_stretch_plot(stretch["z"], stretch["iota"]), "spanwise_stretch_plot")
# save(plot_blocks_2D(blocks_list, grid["x"], grid["nx_lip"]), "blocks_2d")
# save(plot_poly_stretch(grid["eta"], stretch["P_lower"]), "plot_poly_stretch_lower")
# save(plot_poly_stretch(grid["eta"], stretch["P_upper"]), "plot_poly_stretch_upper")
# save(plot_upper_lower(grid["eta"], stretch["P_upper"], stretch["P_lower"]), "plot_upper_lower")




# print_summary_3D(x_B, y_B, z_B)





# save_HDF5 = 1
# if save_HDF5:
#     print("\n\n===Saving to HDF5===")
#     arrays_to_HDF5(case["casename"], x_B, y_B, z_B, plot_hdf5=True)









