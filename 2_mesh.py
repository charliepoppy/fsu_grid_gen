#!/usr/bin/env python3
"""
FSU intake mesher (faceted, 4-block).

Reads the solved geometry exported by MoC.py (geometry/geometry.json + walls.npz)
and discretises it. No geometry is constructed here.

Stage 1 (this file so far):
    - piecewise-uniform x with a node on every corner
    - wall heights at the x nodes
    - checks + figures
Next: wall-normal (y) stretching, block assembly, HDF5 writer.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from meshing.plots import plot_x, plot_z, plot_readback, plot_geometry, plot_grid_zooms
from meshing.build import build_x, build_y, build_z, wall_core
# import meshing.ops_writer as ops_writer
from meshing.ops_writer import write_extruded_blocks
from meshing.topology import faceted
from utils.style import set_style
set_style()

HERE = Path(__file__).resolve().parent

# ======================================================================
# Mesh inputs (geometry inputs live in MoC.py)
# ======================================================================
MESH = {
    "casename": "FSU_facet_sidewalls_LES",
    "geometry_dir": "geometry",
    "dx_target": 0.1026,                 # CAD units (production LES value)
    # corners that must be x nodes, in order; names match geometry.json "points"
    "x_stations": ["leading edge", "cowl lip", "shoulder", "cowl end", "facet end", "outlet floor"],
    "y_top": 80.0,                       # far-field top boundary

    # ---- wall-normal distribution (all spacings in CAD units, measured vertically) ----
    "ny_lower": 500,                     # B0/B1: lower wall -> mid line (cowl / lip line)
    "ny_upper": 150,                     # B3/B2: mid line -> top (external flow: not studied, keep coarse)
    "dy_wall": 0.006,                    # first cell at every wall (ramp, facet, floor, cowl both sides)
    "dy_interface": 0.10,                # B0/B3 interface spacing well upstream of the lip (freestream)
    "blend_length": 20.0,                # mid-line spacing blends wall -> interface over this distance upstream of lip
    "dy_top": 0.30,                      # far-field spacing at y_top
    "dy_cowl_outer": 0.15,               # B2 bottom (outer cowl, not studied) away from the lip
    "cowl_outer_blend": 5.0,             # B2 bottom spacing relaxes dy_wall -> dy_cowl_outer over this length

    # ---- spanwise ----
    "Lz": 25.0,                          # span, CAD units
    "nz": 50,
    "span": "sidewall",                  # "periodic": uniform, endpoint excluded, centred on 0
                                         # "sidewall": walls at z = 0 and Lz from the lip; uniform upstream
    "dz_wall": 0.006,                    # sidewall first cell (sidewall mode)
    "dz_growth": 1.05,                   # max spanwise growth ratio in the sidewall layer
    "dz_blend_up": 20.0,                 # uniform -> clustered over this length upstream of the lip
    "dz_blend_down": 5.0,                # B2 (external) relaxes back to uniform over this length

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
        raise RuntimeError("walls.npz does not match geometry.json - re-run MoC.py")
    if not doc["validated"] and not allow_unvalidated:
        raise RuntimeError("geometry failed validation in MoC.py (use --allow-unvalidated to override)")

    walls = np.load(npz_path)
    geo = {
        "points": {k: np.array(v) for k, v in doc["points"].items()},
        "scalars": doc["scalars"],
        "lower": walls["lower"],
        "upper": walls["upper"],
        "meta": {"generated": doc["generated"], "walls_sha": sha, "config": doc["config"]},
    }
    print(f"geometry loaded: {geo_dir}  (generated {doc['generated']}, sha {sha})")
    return geo



# ======================================================================
# Walls at the x nodes
# ======================================================================
def walls_on_x(x, geo, y_top):
    """
    lower : ramp/turn/facet/floor
    mid   : horizontal line y = lip_y upstream of the lip, cowl/ceiling downstream
            (= top of lower blocks, bottom of upper blocks)
    top   : far-field boundary
    """
    lw, up = geo["lower"], geo["upper"]
    lip = geo["points"]["cowl lip"]
    y_lower = np.interp(x, lw[:, 0], lw[:, 1])
    y_mid = np.where(x < lip[0], lip[1], np.interp(x, up[:, 0], up[:, 1]))
    y_top_arr = np.full_like(x, y_top)
    return y_lower, y_mid, y_top_arr





def check_z(blocks2d, x_lip, m):
    if m["span"] != "sidewall":
        return
    Z = {ib: build_z(X[0], ib, x_lip, m) for ib, (X, Y) in blocks2d.items()}
    d1 = np.diff(Z[1][:, 0]); g = d1[1:] / d1[:-1]
    d0 = np.diff(Z[0][:, 0])
    print("\nspanwise (sidewall mode):")
    _, (r, ml, dc) = wall_core(m["nz"], m["Lz"], m["dz_wall"], rmax=m["dz_growth"])
    print(f"  B1 (internal): wall dz {d1[0]:.5f}, growth {r:.4f} over {ml} cells/side, core dz {d1.max():.4f}"
          f"  ({d1.max() / (m['Lz'] / m['nz']):.2f}x the periodic dz {m['Lz'] / m['nz']:.4f})")
    print(f"  B0 inflow column: uniform dz {d0.max():.4f}")
    for a, c, za, zc in [(0, 1, Z[0][:, -1], Z[1][:, 0]), (3, 2, Z[3][:, -1], Z[2][:, 0]),
                         (0, 3, Z[0], Z[3])]:
        err = np.abs(za - zc).max()
        print(f"  z nodes B{a}|B{c}: max mismatch {err:.1e}")
        assert err < 1e-12
    assert all(np.all(np.diff(z, axis=0) > 0) for z in Z.values()), "z not monotonic"


def write_grid(blocks2d, m, geo, out_dir, x_lip):
    blocks = {ib: (X, Y, build_z(X[0], ib, x_lip, m)) for ib, (X, Y) in blocks2d.items()}
    path = Path(out_dir) / f"{m['casename'].lower()}.h5"
    prov = {"mesh": {k: v for k, v in m.items()}, "geometry_sha": geo["meta"]["walls_sha"],
            "geometry_generated": geo["meta"]["generated"]}
    write_extruded_blocks(path, blocks, nhalo=m["nhalo"], scale=m["scale_factor"],
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
    for a, b, d, n in zip(names[:-1], names[1:], dx_seg, np.diff(idx)):
        print(f"  {a:>14} -> {b:<14} n={n:5d}  dx={d:.5f}  ({(d / dx_target - 1) * 100:+.2f}% of target)")
    jump = np.abs(np.diff(dx_seg) / dx_seg[:-1]).max() * 100
    print(f"\n  nx = {x.size},  max dx jump between segments {jump:.2f}%,  worst station error {worst:.1e}")
    assert worst < 1e-9, "a corner is not on a node"
    assert np.all(np.diff(x) > 0), "x not monotonic"
    return jump


def check_walls(x, y_lower, y_mid, geo):
    """Corner heights reproduced at their nodes; lower < mid everywhere."""
    P = geo["points"]
    print("\nwall heights at corner nodes:")
    for name, wall in [("shoulder", y_lower), ("facet end", y_lower), ("cowl lip", y_mid), ("cowl end", y_mid)]:
        i = np.argmin(np.abs(x - P[name][0]))
        print(f"  {name:<12} y={wall[i]:10.5f}  geometry {P[name][1]:10.5f}  err {wall[i] - P[name][1]:+.1e}")
    gap = y_mid - y_lower
    print(f"  lower-block height: min {gap.min():.3f} at x={x[np.argmin(gap)]:.2f}, max {gap.max():.3f}")
    assert gap.min() > 0, "lower wall crosses the mid line"


def check_y(x, Y_low, Y_up, blocks, geo, i_lip, m):
    sf = m["scale_factor"]
    lw = geo["lower"]
    th = np.arctan(np.gradient(np.interp(x, lw[:, 0], lw[:, 1]), x))     # lower-wall angle

    dl = np.diff(Y_low, axis=0); du = np.diff(Y_up, axis=0)
    gl = dl[1:] / dl[:-1]; gu = du[1:] / du[:-1]

    print("\ny distribution:")
    print(f"  ny lower/upper = {Y_low.shape[0]}/{Y_up.shape[0]}  (total {Y_low.shape[0] + Y_up.shape[0] - 1} along a column)")
    rows = [
        ("lower wall, first dy",            dl[0]),
        ("lower wall, wall-normal dn",       dl[0] * np.cos(th)),
        ("cowl inner, first dy (x>=lip)",    dl[-1, i_lip:]),
        ("cowl outer, first dy (x>=lip)",    du[0, i_lip:]),
        ("B0/B3 interface, lower side",      dl[-1, :i_lip + 1]),
        ("B0/B3 interface, upper side",      du[0, :i_lip + 1]),
    ]
    for name, v in rows:
        print(f"  {name:<34} min {v.min():.5f}  max {v.max():.5f}   ({v.min() / sf:.4f}-{v.max() / sf:.4f} d*)")
    ratio = du[0, :i_lip + 1] / dl[-1, :i_lip + 1]
    print(f"  interface spacing ratio upper/lower: {ratio.min():.4f} - {ratio.max():.4f}  (1 = continuous)")
    print(f"  max growth ratio in y: lower {max(gl.max(), 1 / gl.min()):.4f}, upper {max(gu.max(), 1 / gu.min()):.4f}")
    print(f"  max lower-wall angle {np.degrees(np.abs(th).max()):.1f} deg -> min wall cell angle {90 - np.degrees(np.abs(th).max()):.1f} deg")
    dx = np.diff(x)
    print(f"  wall cell aspect dx/dy: {dx.min() / m['dy_wall']:.1f} - {dx.max() / m['dy_wall']:.1f}")

    # shared nodes across every interface
    b = blocks
    pairs = [(0, 1, (slice(None), -1), (slice(None), 0)),     # B0|B1 at lip column
             (3, 2, (slice(None), -1), (slice(None), 0)),     # B3|B2 at lip column
             (0, 3, (-1, slice(None)), (0, slice(None))),     # B0 top = B3 bottom
             (1, 2, (-1, slice(None)), (0, slice(None)))]     # B1 top = B2 bottom (cowl)
    print("  interfaces (max node mismatch):", end="")
    for a, c, sa, sc in pairs:
        err = max(np.abs(b[a][0][sa] - b[c][0][sc]).max(), np.abs(b[a][1][sa] - b[c][1][sc]).max())
        print(f"  B{a}|B{c} {err:.1e}", end="")
        assert err < 1e-12, f"B{a}|B{c} interface nodes do not match"
    print()
    assert np.all(dl > 0) and np.all(du > 0), "negative cell height (folded grid)"



# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=MESH["dx_target"], help="target dx (CAD units)")
    ap.add_argument("--allow-unvalidated", action="store_true")
    ap.add_argument("--ny-lower", dest="ny_lower", type=int)
    ap.add_argument("--ny-upper", dest="ny_upper", type=int)
    ap.add_argument("--dy-wall", dest="dy_wall", type=float)
    ap.add_argument("--nz", type=int)
    ap.add_argument("--span", choices=["periodic", "sidewall"])
    ap.add_argument("--write", action="store_true", help="write the OPS HDF5 grid")
    ap.add_argument("--dz-wall", dest="dz_wall", type=float)
    ap.add_argument("--dz-growth", dest="dz_growth", type=float)
    args = ap.parse_args()

    out_dir = HERE / "grids" / MESH["casename"].lower()
    fig_dir = out_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    geo = load_geometry(HERE / MESH["geometry_dir"], args.allow_unvalidated)

    names = MESH["x_stations"]
    stations = [geo["points"][n][0] for n in names]
    x, idx, dx_seg = build_x(stations, args.dx)
    check_x(x, idx, names, geo, dx_seg, args.dx)

    y_lower, y_mid, y_top = walls_on_x(x, geo, MESH["y_top"])
    check_walls(x, y_lower, y_mid, geo)

    i_lip = idx[names.index("cowl lip")]
    print(f"\nblock split at i_lip = {i_lip}: B0/B3 nx = {i_lip + 1}, B1/B2 nx = {x.size - i_lip}")

    plot_x(x, idx, names, y_lower, y_mid, y_top, fig_dir)

    # ---- stage 2: wall-normal distribution + 2D blocks ----
    m = dict(MESH)
    for k in ("ny_lower", "ny_upper", "dy_wall"):
        if getattr(args, k) is not None:
            m[k] = getattr(args, k)
    Y_low, Y_up, dmid = build_y(x, y_lower, y_mid, y_top, x[i_lip], m)
    blocks = faceted(x, Y_low, Y_up, i_lip)
    check_y(x, Y_low, Y_up, blocks, geo, i_lip, m)
    print("  2D block shapes (ny, nx):", {b: v[0].shape for b, v in blocks.items()})
    n2d = sum(v[0].size for v in blocks.values())
    n_int = sum(blocks[b][0].size for b in (0, 1))
    print(f"  points per z-plane: {n2d:,}  (lower/internal {n_int:,} = {100 * n_int / n2d:.0f}%)"
          f"  -> x nz=200: {n2d * 200 / 1e6:.0f}M")
    plot_grid_zooms(blocks, x, dmid, i_lip, geo, fig_dir, m)

    # ---- stage 3: extrude + write ----
    # if args.nz is not None:
    #     m["nz"] = args.nz
    # if args.span is not None:
    #     m["span"] = args.span
    for k in ("nz", "span", "dz_wall", "dz_growth"):
        if getattr(args, k) is not None:
            m[k] = getattr(args, k)
    check_z(blocks, x[i_lip], m)
    plot_z(blocks, x[i_lip], m, fig_dir)
    n3 = sum(v[0].size for v in blocks.values()) * m["nz"]
    print(f"\n3D: nz = {m['nz']}, total {n3 / 1e6:.1f}M points, file ~{n3 * (4 if m['write_h0'] else 3) * 8 / 1e9:.1f} GB (+halos)")
    if args.write:
        path = write_grid(blocks, m, geo, out_dir, x[i_lip])
        plot_readback(path, blocks, m, fig_dir)
    else:
        print("  (dry run - add --write to write the HDF5)")
    print(f"figures -> {fig_dir}")


if __name__ == "__main__":
    main()