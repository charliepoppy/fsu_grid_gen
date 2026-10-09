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




HERE = Path(__file__).resolve().parent
out_dir = HERE / "grids" / CONFIG["casename"].lower()
fig_dir = out_dir / "figs"
fig_dir.mkdir(parents=True, exist_ok=True)

pts, curve_x, curve_y = build_points(CONFIG)
plot_geometry(CONFIG["casename"], pts, curve_x, curve_y, fig_dir)




