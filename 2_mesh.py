#!/usr/bin/env python3
"""
FSU intake mesher. Step 2 of the pipeline (run ./1_geometry.py first).

Reads the solved geometry (geometry/geometry.json + walls.npz) and discretises it.
No geometry is constructed here.

Variants (--variant):
  faceted : 4 blocks              notched : 6 blocks (vertical step at the shoulder)
      [3][   2   ]                    [3][      4       ]
      [0][   1   ]                    [0][ 1 ][   2    ]
                                               [   5    ]   <- layer on the notched wall
  x : node on every corner; uniform, graded towards the step for the notched variant
  y : Vinokur two-sided per column with prescribed end spacings
  z : periodic uniform, or sidewalls (wall layer + uniform core) from the lip

Block layout lives in meshing/topology.py; everything here reads blocks / ifaces / info.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from meshing.build import build_x, build_z, wall_core
from meshing.ops_writer import write_extruded_blocks
from meshing.plots import plot_x, plot_z, plot_readback, plot_grid_zooms
from meshing.topology import TOPOLOGY
from utils.style import set_style
set_style()

HERE = Path(__file__).resolve().parent

# ======================================================================
# Mesh inputs (geometry inputs live in 1_geometry.py)
# ======================================================================
MESH = {
    "casename": "FSU_{variant}_{span}_LES",
    "geometry_dir": "geometry",
    "dx_target": 0.1026,                 # CAD units (production LES value)
    # corners that must be x nodes, in order; names match geometry.json "points"
    "x_stations": {
        "faceted": ["leading edge", "cowl lip", "shoulder", "cowl end", "facet end", "outlet floor"],
        "notched": ["leading edge", "cowl lip", "shoulder", "cowl end", "notch end", "outlet floor"],
    },
    "y_top": 80.0,                       # far-field top boundary

    # ---- wall-normal distribution (all spacings in CAD units, measured vertically) ----
    "ny_lower": 500,                     # lower wall (or notch-layer top) -> mid line (cowl / lip line)
    "ny_upper": 150,                     # mid line -> top (external flow: not studied, keep coarse)
    "dy_wall": 0.006,                    # first cell at every wall (ramp, facet, floor, cowl both sides)
    "dy_interface": 0.10,                # B0/B3 interface spacing well upstream of the lip (freestream)
    "blend_length": 20.0,                # mid-line spacing blends wall -> interface over this distance upstream of lip
    "dy_top": 0.30,                      # far-field spacing at y_top
    "dy_cowl_outer": 0.15,               # outer cowl (not studied) away from the lip
    "cowl_outer_blend": 5.0,             # outer cowl spacing relaxes dy_wall -> dy_cowl_outer over this length

    # ---- notched variant ----
    "ny_notch": 80,                      # notch layer: notched wall -> (wall + step height)
    "dy_notch_top": 0.03,                # notch-layer top / duct bottom spacing away from the step
    "notch_top_blend": 5.0,              # ... reached over this length downstream of the step
    "dx_step": 0.006,                    # x spacing at the step face
    "dx_growth": 1.05,                   # max x growth ratio away from the step

    # ---- spanwise ----
    "Lz": 25.0,                          # span, CAD units
    "nz": 50,
    "span": "periodic",                  # "periodic": uniform, endpoint excluded, centred on 0
                                         # "sidewall": walls at z = 0 and Lz from the lip; uniform upstream
    "dz_wall": 0.006,                    # sidewall first cell (sidewall mode)
    "dz_growth": 1.05,                   # max spanwise growth ratio in the sidewall layer
    "dz_blend_up": 20.0,                 # uniform -> clustered over this length upstream of the lip
    "dz_blend_down": 5.0,                # external blocks relax back to uniform over this length

    # ---- output ----
    "nhalo": 5,
    "write_h0": True,

    # ---- scaling: CAD units -> delta* (applied only when writing) ----
    "scale_factor": 17.34 * 0.07 / 1.584,   # CAD units per delta* (H_t * IBLT / (d99/d*)); NB production used H_t = 17.3
}


# ======================================================================
# Load geometry
# ======================================================================
def load_geometry(geo_dir, allow_unvalidated=False):
    geo_dir = Path(geo_dir)
    doc = json.loads((geo_dir / "geometry.json").read_text())
    npz_path = geo_dir / "walls.npz"
    sha = hashlib.sha256(npz_path.read_bytes()).hexdigest()[:16]

    if sha != doc["walls_npz_sha256"]:
        raise RuntimeError("walls.npz does not match geometry.json - re-run 1_geometry.py")
    if not doc["validated"] and not allow_unvalidated:
        raise RuntimeError("geometry failed validation in 1_geometry.py (use --allow-unvalidated to override)")

    walls = np.load(npz_path)
    geo = {
        "points": {k: np.array(v) for k, v in doc["points"].items()},
        "scalars": doc["scalars"],
        "lower": walls["lower"],
        "upper": walls["upper"],
        "lower_notched": walls["lower_notched"] if "lower_notched" in walls.files else None,
        "meta": {"generated": doc["generated"], "walls_sha": sha, "config": doc["config"]},
    }
    print(f"geometry loaded: {geo_dir}  (generated {doc['generated']}, sha {sha})")
    return geo


# ======================================================================
# Walls at the x nodes
# ======================================================================
def walls_on_x(x, geo, y_top, variant):
    """
    lower : ramp/turn/facet/floor (the faceted wall; used up to the step in the notched variant)
    notch : notched wall from the step on (step bottom -> notch end -> floor); NaN upstream
    mid   : y = lip_y upstream of the lip, cowl/ceiling downstream (top of lower blocks)
    top   : far-field boundary
    """
    lw, up = geo["lower"], geo["upper"]
    lip = geo["points"]["cowl lip"]
    W = {"lower": np.interp(x, lw[:, 0], lw[:, 1]),
         "mid": np.where(x < lip[0], lip[1], np.interp(x, up[:, 0], up[:, 1])),
         "top": np.full_like(x, y_top)}
    if variant == "notched":
        if geo["lower_notched"] is None:
            raise RuntimeError("geometry has no notched wall - re-run 1_geometry.py")
        P = geo["points"]
        # interpolate along the tail only: the step is vertical, so np.interp on the full
        # polyline would be ambiguous exactly at the step x
        tail = np.array([P["step bottom"], P["notch end"], P["outlet floor"]])
        W["notch"] = np.where(x >= P["shoulder"][0], np.interp(x, tail[:, 0], tail[:, 1]), np.nan)
    return W


# ======================================================================
# Spanwise + write
# ======================================================================
def block_z(blocks, info, m):
    """Spanwise coordinates for every block, from its role (internal / external)."""
    return {ib: build_z(X[0], info["x_lip"], m, external=ib in info["external"]) for ib, (X, Y) in blocks.items()}


def write_grid(blocks, info, m, geo, out_dir):
    Z = block_z(blocks, info, m)
    blocks3 = {ib: (X, Y, Z[ib]) for ib, (X, Y) in blocks.items()}
    path = Path(out_dir) / f"{m['casename'].lower()}.h5"
    prov = {"mesh": {k: v for k, v in m.items()}, "geometry_sha": geo["meta"]["walls_sha"],
            "geometry_generated": geo["meta"]["generated"],
            "roles": {"internal": info["internal"], "external": info["external"],
                      "walls": info["walls"], "ifaces": [list(f) for f in info["ifaces"]]}}
    write_extruded_blocks(path, blocks3, nhalo=m["nhalo"], scale=m["scale_factor"],
                          h0=m["write_h0"], provenance=prov)
    (Path(out_dir) / "params.json").write_text(json.dumps(prov, indent=2, default=str))
    return path


# ======================================================================
# Checks
# ======================================================================
def check_x(x, idx, names, geo, dx_seg, dx_target):
    P = geo["points"]
    print("\nx stations (must be exact nodes):")
    print(f"  {'station':<14}{'index':>7}{'x node':>14}{'geometry x':>14}{'error':>11}")
    worst = 0.0
    for name, i in zip(names, idx):
        err = x[i] - P[name][0]
        worst = max(worst, abs(err))
        print(f"  {name:<14}{i:>7}{x[i]:>14.6f}{P[name][0]:>14.6f}{err:>+11.1e}")
    print("\nsegments:")
    for a, b, i0, i1 in zip(names[:-1], names[1:], idx[:-1], idx[1:]):
        d = np.diff(x[i0:i1 + 1])
        print(f"  {a:>14} -> {b:<14} n={i1 - i0:5d}  dx {d.min():.5f} - {d.max():.5f}")
    dx = np.diff(x)
    g = dx[1:] / dx[:-1]
    print(f"\n  nx = {x.size},  dx {dx.min():.5f} - {dx.max():.5f},  max growth {max(g.max(), 1 / g.min()):.4f},"
          f"  worst station error {worst:.1e}")
    assert worst < 1e-9, "a corner is not on a node"
    assert np.all(dx > 0), "x not monotonic"


def check_walls(x, W, geo, variant):
    """Corner heights reproduced at their nodes; nothing crosses."""
    P = geo["points"]
    corners = [("shoulder", "lower"), ("cowl lip", "mid"), ("cowl end", "mid")]
    corners += [("facet end", "lower")] if variant == "faceted" else [("step bottom", "notch"), ("notch end", "notch")]
    print("\nwall heights at corner nodes:")
    for name, wall in corners:
        i = np.argmin(np.abs(x - P[name][0]))
        print(f"  {name:<12} y={W[wall][i]:10.5f}  geometry {P[name][1]:10.5f}  err {W[wall][i] - P[name][1]:+.1e}")
    gap = W["mid"] - W["lower"]
    print(f"  lower-block height: min {gap.min():.3f} at x={x[np.argmin(gap)]:.2f}, max {gap.max():.3f}")
    assert gap.min() > 0, "lower wall crosses the mid line"


def _edge(X, Y, side):
    """First-cell size normal to a block face."""
    return {"bottom": Y[1] - Y[0], "top": Y[-1] - Y[-2],
            "left": X[:, 1] - X[:, 0], "right": X[:, -1] - X[:, -2]}[side]


def check_blocks(blocks, ifaces, info, m):
    sf = m["scale_factor"]
    print("\nblocks (ny, nx)   first dy bottom / top  [min - max]          max y-growth")
    for ib, (X, Y) in blocks.items():
        d = np.diff(Y, axis=0)
        assert np.all(d > 0), f"B{ib}: negative cell height (folded grid)"
        g = d[1:] / d[:-1]
        print(f"  B{ib}  {Y.shape[0]:4d} x {Y.shape[1]:5d}   {d[0].min():.4f}-{d[0].max():.4f} / "
              f"{d[-1].min():.4f}-{d[-1].max():.4f}   {max(g.max(), 1 / g.min()):.4f}")

    print("\nwalls, first cell:")
    for name, (ib, side) in info["walls"].items():
        v = _edge(*blocks[ib], side)
        print(f"  {name:<30} {v.min():.5f} - {v.max():.5f}   ({v.min() / sf:.4f}-{v.max() / sf:.4f} d*)")

    print("\ninterfaces:")
    for a, b, kind in ifaces:
        (Xa, Ya), (Xb, Yb) = blocks[a], blocks[b]
        if kind == "col":
            pa, pb = (Xa[:, -1], Ya[:, -1]), (Xb[:, 0], Yb[:, 0])
            ratio = (Xb[:, 1] - Xb[:, 0]) / (Xa[:, -1] - Xa[:, -2])
        else:
            pa, pb = (Xa[-1], Ya[-1]), (Xb[0], Yb[0])
            ratio = (Yb[1] - Yb[0]) / (Ya[-1] - Ya[-2])
        err = max(np.abs(pa[0] - pb[0]).max(), np.abs(pa[1] - pb[1]).max())
        print(f"  B{a}|B{b} ({kind})  node mismatch {err:.1e}   spacing ratio across {ratio.min():.3f}-{ratio.max():.3f}")
        assert err < 1e-12, f"B{a}|B{b} interface nodes do not match"

    n2d = sum(v[0].size for v in blocks.values())
    n_int = sum(blocks[b][0].size for b in info["internal"])
    print(f"\npoints per z-plane {n2d:,} (internal {100 * n_int / n2d:.0f}%);  nz = 200 -> {n2d * 200 / 1e6:.0f}M")


def check_z(blocks, ifaces, info, m):
    if m["span"] != "sidewall":
        return
    Z = block_z(blocks, info, m)
    d = np.diff(Z[info["duct"]][:, -1])
    d0 = np.diff(Z[0][:, 0])
    _, (r, ml, dc) = wall_core(m["nz"], m["Lz"], m["dz_wall"], rmax=m["dz_growth"])
    print("\nspanwise (sidewall mode):")
    print(f"  B{info['duct']} (duct): wall dz {d[0]:.5f}, growth {r:.4f} over {ml} cells/side, core dz {d.max():.4f}"
          f"  ({d.max() / (m['Lz'] / m['nz']):.2f}x the periodic dz {m['Lz'] / m['nz']:.4f})")
    print(f"  B0 inflow column: uniform dz {d0.max():.4f}")
    for a, b, kind in ifaces:
        za, zb = (Z[a][:, -1], Z[b][:, 0]) if kind == "col" else (Z[a], Z[b])
        err = np.abs(za - zb).max()
        print(f"  z nodes B{a}|B{b}: max mismatch {err:.1e}")
        assert err < 1e-12
    assert all(np.all(np.diff(z, axis=0) > 0) for z in Z.values()), "z not monotonic"


# ======================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(TOPOLOGY), default="faceted")
    ap.add_argument("--span", choices=["periodic", "sidewall"])
    ap.add_argument("--dx", dest="dx_target", type=float, help="target dx (CAD units)")
    ap.add_argument("--dx-step", dest="dx_step", type=float, help="notched: x spacing at the step face")
    ap.add_argument("--ny-lower", dest="ny_lower", type=int)
    ap.add_argument("--ny-upper", dest="ny_upper", type=int)
    ap.add_argument("--ny-notch", dest="ny_notch", type=int)
    ap.add_argument("--dy-wall", dest="dy_wall", type=float)
    ap.add_argument("--nz", type=int)
    ap.add_argument("--dz-wall", dest="dz_wall", type=float)
    ap.add_argument("--dz-growth", dest="dz_growth", type=float)
    ap.add_argument("--write", action="store_true", help="write the OPS HDF5 grid")
    ap.add_argument("--allow-unvalidated", action="store_true")
    args = ap.parse_args()

    # one settings dict: MESH defaults, overridden by whatever was given on the command line
    m = dict(MESH)
    for k in ("span", "dx_target", "dx_step", "ny_lower", "ny_upper", "ny_notch",
              "dy_wall", "nz", "dz_wall", "dz_growth"):
        if getattr(args, k) is not None:
            m[k] = getattr(args, k)
    variant = args.variant
    m["variant"] = variant
    m["casename"] = MESH["casename"].format(variant=variant, span=m["span"])

    out_dir = HERE / "grids" / m["casename"].lower()
    fig_dir = out_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"case: {m['casename']}  ->  {out_dir}")

    geo = load_geometry(HERE / m["geometry_dir"], args.allow_unvalidated)

    # ---- x ----
    names = m["x_stations"][variant]
    stations = [geo["points"][n][0] for n in names]
    refine = {names.index("shoulder"): m["dx_step"]} if variant == "notched" else None
    x, idx, dx_seg = build_x(stations, m["dx_target"], refine, m["dx_growth"])
    check_x(x, idx, names, geo, dx_seg, m["dx_target"])

    W = walls_on_x(x, geo, m["y_top"], variant)
    check_walls(x, W, geo, variant)
    plot_x(x, idx, names, W["lower"], W["mid"], W["top"], fig_dir)

    # ---- y + blocks ----
    blocks, ifaces, info = TOPOLOGY[variant](x, W, idx, names, m)
    info["ifaces"] = ifaces
    info["x_lip"] = x[info["i_lip"]]
    if variant == "notched":
        print(f"\nnotch: step at x = {x[info['i_step']]:.4f} (node {info['i_step']}), height {info['h_step']:.4f}")
    check_blocks(blocks, ifaces, info, m)
    plot_grid_zooms(blocks, x, info, geo, fig_dir, m)

    # ---- z + write ----
    check_z(blocks, ifaces, info, m)
    plot_z(blocks, info, m, fig_dir)
    n3 = sum(v[0].size for v in blocks.values()) * m["nz"]
    print(f"\n3D: nz = {m['nz']}, total {n3 / 1e6:.1f}M points, file ~{n3 * (4 if m['write_h0'] else 3) * 8 / 1e9:.1f} GB (+halos)")

    print("\nopensbli.cpp block sizes:")
    for ib, (X, Y) in blocks.items():
        print(f"  block{ib}np0 = {X.shape[1]};  block{ib}np1 = {X.shape[0]};  block{ib}np2 = {m['nz']};")

    if args.write:
        path = write_grid(blocks, info, m, geo, out_dir)
        plot_readback(path, blocks, m, fig_dir)
    else:
        print("\n  (dry run - add --write to write the HDF5)")
    print(f"figures -> {fig_dir}")


if __name__ == "__main__":
    main()
