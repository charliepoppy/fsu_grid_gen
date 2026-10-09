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

try:
    from utils.plots import set_style
    set_style()
except ImportError:
    pass

HERE = Path(__file__).resolve().parent

# ======================================================================
# Mesh inputs (geometry inputs live in MoC.py)
# ======================================================================
MESH = {
    "casename": "FSU_facet_periodic_LES",
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
    "span": "periodic",                  # "periodic": uniform, endpoint excluded, centred on 0
                                         # "sidewall": walls at z = 0 and Lz from the lip; uniform upstream
    "dz_wall": 0.006,                    # sidewall first cell (sidewall mode)
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
# x distribution
# ======================================================================
def build_x(stations, dx_target):
    """Piecewise-uniform x with a node exactly on every station. Returns x, station indices, per-segment dx."""
    xs = np.asarray(stations, float)
    if np.any(np.diff(xs) <= 0):
        raise ValueError(f"stations must be strictly increasing: {xs}")
    n = np.maximum(1, np.round(np.diff(xs) / dx_target).astype(int))
    segs = [np.linspace(a, b, k + 1)[:-1] for a, b, k in zip(xs[:-1], xs[1:], n)]
    x = np.r_[np.concatenate(segs), xs[-1]]
    idx = np.r_[0, np.cumsum(n)]
    return x, idx, np.diff(xs) / n


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


# ======================================================================
# Wall-normal (y) distribution
# ======================================================================
def vinokur(n, d0, d1):
    """
    Two-sided stretching on [0, 1] with n points and end spacings d0, d1
    (fractions of the column height). Vinokur (1983), J. Comput. Phys. 50.
    """
    from scipy.optimize import brentq
    xi = np.linspace(0.0, 1.0, n)
    A = np.sqrt(d1 / d0)
    B = 1.0 / ((n - 1) * np.sqrt(d0 * d1))
    if abs(B - 1.0) < 1e-12:
        u = xi
    elif B > 1.0:
        dy = brentq(lambda d: np.sinh(d) / d - B, 1e-12, 200.0)
        u = 0.5 * (1.0 + np.tanh(dy * (xi - 0.5)) / np.tanh(0.5 * dy))
    else:
        dy = brentq(lambda d: np.sin(d) / d - B, 1e-12, np.pi - 1e-9)
        u = 0.5 * (1.0 + np.tan(dy * (xi - 0.5)) / np.tan(0.5 * dy))
    return u / (A + (1.0 - A) * u)


def dy_mid(x, x_lip, m):
    """
    Spacing along the mid line (cowl downstream of lip, B0/B3 interface upstream).
    = dy_wall for x >= x_lip; smooth (C1) blend to dy_interface over blend_length upstream.
    """
    s = np.clip((x_lip - x) / m["blend_length"], 0.0, 1.0)
    S = s * s * (3.0 - 2.0 * s)                       # smoothstep
    return m["dy_wall"] + (m["dy_interface"] - m["dy_wall"]) * S


def build_columns(y_bot, y_top, n, d_bot, d_top):
    """(n, nx) array of y: each column stretched between y_bot and y_top with end spacings d_bot, d_top."""
    H = y_top - y_bot
    Y = np.empty((n, y_bot.size))
    for i in range(y_bot.size):
        Y[:, i] = y_bot[i] + H[i] * vinokur(n, d_bot[i] / H[i], d_top[i] / H[i])
    return Y


def dy_upper_bottom(x, x_lip, dmid, m):
    """
    Bottom spacing of the upper blocks.
      x <  lip : = dmid (B0/B3 is a flow interface -> must match the lower side)
      x >= lip : outer cowl surface, not studied. Starts at dy_wall (lip column is shared
                 with B3) and relaxes to dy_cowl_outer over cowl_outer_blend.
    """
    s = np.clip((x - x_lip) / m["cowl_outer_blend"], 0.0, 1.0)
    S = s * s * (3.0 - 2.0 * s)
    outer = m["dy_wall"] + (m["dy_cowl_outer"] - m["dy_wall"]) * S
    return np.where(x < x_lip, dmid, outer)


def build_y(x, y_lower, y_mid, y_top, x_lip, m):
    dmid = dy_mid(x, x_lip, m)
    dup = dy_upper_bottom(x, x_lip, dmid, m)
    Y_low = build_columns(y_lower, y_mid, m["ny_lower"], np.full_like(x, m["dy_wall"]), dmid)
    Y_up = build_columns(y_mid, y_top, m["ny_upper"], dup, np.full_like(x, m["dy_top"]))
    return Y_low, Y_up, dmid


def assemble_blocks_2d(x, Y_low, Y_up, i_lip):
    """4-block faceted topology. Each block: (X, Y) arrays of shape (ny, nx_block)."""
    X_low = np.broadcast_to(x, Y_low.shape)
    X_up = np.broadcast_to(x, Y_up.shape)
    sl0, sl1 = slice(0, i_lip + 1), slice(i_lip, None)
    return {
        0: (X_low[:, sl0].copy(), Y_low[:, sl0].copy()),   # lower, upstream of lip
        1: (X_low[:, sl1].copy(), Y_low[:, sl1].copy()),   # lower, intake/isolator
        2: (X_up[:, sl1].copy(), Y_up[:, sl1].copy()),     # upper, above cowl
        3: (X_up[:, sl0].copy(), Y_up[:, sl0].copy()),     # upper, upstream of lip
    }


# ======================================================================
# Spanwise extrusion + write
# ======================================================================
def _smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def sidewall_weight(x, x_lip, upper, m):
    """
    0 -> uniform span, 1 -> sidewall-clustered span.
      x <  lip : ramps 0 -> 1 over dz_blend_up approaching the lip (all blocks; B3 must equal B0)
      x >= lip : lower blocks (B1, between the sidewalls) stay 1;
                 upper blocks (B2, external, no sidewalls) relax 1 -> 0 over dz_blend_down
    """
    w = 1.0 - _smoothstep((x_lip - x) / m["dz_blend_up"])
    if upper:
        w = np.where(x >= x_lip, 1.0 - _smoothstep((x - x_lip) / m["dz_blend_down"]), w)
    return np.where(x >= x_lip, w if upper else 1.0, w)


def wall_core(n, L, d0, rmax=1.05):
    """
    n points on [0, L]: geometric layer from each wall (first cell d0, growth r <= rmax)
    then a uniform core whose spacing equals the last layer cell (spacing is continuous).
    Chooses the layer depth that minimises the core spacing. Returns z, (r, m, d_core).
    """
    from scipy.optimize import brentq
    cells, best = n - 1, None
    for m_ in range(1, cells // 2):
        nc = cells - 2 * m_
        f = lambda r: d0 * r**m_ - (L - 2.0 * d0 * (r**m_ - 1.0) / (r - 1.0)) / nc
        try:
            r = brentq(f, 1.0 + 1e-9, 2.0)
        except ValueError:
            continue
        if r <= rmax and (best is None or d0 * r**m_ < best[2]):
            best = (r, m_, d0 * r**m_)
    if best is None:
        raise ValueError(f"no wall+core distribution with n={n}, L={L}, d0={d0}, growth<={rmax}")
    r, m_, dc = best
    layer = d0 * r ** np.arange(m_)
    d = np.r_[layer, np.full(cells - 2 * m_, dc), layer[::-1]]
    z = np.r_[0.0, np.cumsum(d)]
    return z * (L / z[-1]), best


def build_z(xb, ib, x_lip, m):
    """(nz, nx_block) spanwise coordinates for block ib with x-coordinates xb."""
    nz, Lz = m["nz"], m["Lz"]
    if m["span"] == "periodic":
        z1 = 0.5 * Lz * np.linspace(-1.0, 1.0, nz, endpoint=False)
        return np.broadcast_to(z1[:, None], (nz, xb.size))
    if m["span"] != "sidewall":
        raise ValueError(f"unknown span type {m['span']!r}")
    # sidewall: nodes on both z = 0 and z = Lz (walls in B1), same nz in every block
    z_uni = np.linspace(0.0, Lz, nz)
    z_wall, _ = wall_core(nz, Lz, m["dz_wall"])
    w = sidewall_weight(xb, x_lip, upper=ib in (2, 3), m=m)
    return z_uni[:, None] * (1.0 - w[None, :]) + z_wall[:, None] * w[None, :]


def check_z(blocks2d, x_lip, m):
    if m["span"] != "sidewall":
        return
    Z = {ib: build_z(X[0], ib, x_lip, m) for ib, (X, Y) in blocks2d.items()}
    d1 = np.diff(Z[1][:, 0]); g = d1[1:] / d1[:-1]
    d0 = np.diff(Z[0][:, 0])
    print("\nspanwise (sidewall mode):")
    _, (r, ml, dc) = wall_core(m["nz"], m["Lz"], m["dz_wall"])
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
    from ops_writer import write_extruded_blocks
    blocks = {ib: (X, Y, build_z(X[0], ib, x_lip, m)) for ib, (X, Y) in blocks2d.items()}
    path = Path(out_dir) / f"{m['casename'].lower()}.h5"
    prov = {"mesh": {k: v for k, v in m.items()}, "geometry_sha": geo["meta"]["walls_sha"],
            "geometry_generated": geo["meta"]["generated"]}
    write_extruded_blocks(path, blocks, nhalo=m["nhalo"], scale=m["scale_factor"],
                          h0=m["write_h0"], provenance=prov)
    (Path(out_dir) / "params.json").write_text(json.dumps(prov, indent=2, default=str))
    return path


def plot_z(blocks2d, x_lip, m, fig_dir, stride_x=20, stride_z=4):
    """Plan view (x-z) of the spanwise distribution for lower (B0+B1) and upper (B3+B2) blocks."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True)
    for ax, ids, title in [(axes[0], (0, 1), "lower blocks B0 | B1 (internal duct between sidewalls)"),
                           (axes[1], (3, 2), "upper blocks B3 | B2 (external, above cowl)")]:
        for ib in ids:
            xb = blocks2d[ib][0][0]
            Z = build_z(xb, ib, x_lip, m)
            c = "tab:blue" if ib in (0, 3) else "k"
            for k in range(0, Z.shape[0], stride_z):
                ax.plot(xb, Z[k], color=c, lw=0.25)
            for i in range(0, xb.size, stride_x):
                ax.plot([xb[i]] * Z.shape[0], Z[:, i], color=c, lw=0.25)
        ax.axvline(x_lip, color="tab:red", lw=0.6, ls="--")
        ax.set_ylabel(r"$z$"); ax.set_title(title + f"  (every {stride_z}th z-line)", fontsize=8)
    axes[1].set_xlabel(r"$x$")
    fig.savefig(fig_dir / "mesh_z.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_readback(path, blocks2d, m, fig_dir):
    """Read k=0 back from the file and overlay on the in-memory grid (scaled)."""
    from ops_writer import read_plane
    fig, ax = plt.subplots(figsize=(10, 3.5))
    worst = 0.0
    for ib, (X, Y) in blocks2d.items():
        xr, yr = read_plane(path, ib, 0, m["nhalo"])
        worst = max(worst, np.abs(xr - X / m["scale_factor"]).max(), np.abs(yr - Y / m["scale_factor"]).max())
        for j in range(0, xr.shape[0], 10):
            ax.plot(xr[j], yr[j], lw=0.2, color="k")
        for i in range(0, xr.shape[1], 20):
            ax.plot(xr[:, i], yr[:, i], lw=0.2, color="k")
    ax.set_aspect("equal"); ax.set_title(f"readback k=0 from {Path(path).name}  (max diff vs memory {worst:.1e})", fontsize=9)
    fig.savefig(fig_dir / "readback_k0.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  readback k=0: max difference vs in-memory grid {worst:.1e}")
    assert worst < 1e-12


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
# Figures
# ======================================================================
def plot_grid_zooms(blocks, x, dmid, i_lip, geo, fig_dir, m):
    P = geo["points"]
    views = {
        "full (every 20th line)": dict(xlim=(-5, x[-1] + 5), ylim=(-5, 85), stride=20),
        "cowl lip": dict(xlim=(P["cowl lip"][0] - 3, P["cowl lip"][0] + 3), ylim=(48.5, 51.5), stride=1),
        "shoulder (lower wall)": dict(xlim=(P["shoulder"][0] - 1.5, P["shoulder"][0] + 1.5),
                                      ylim=(P["shoulder"][1] - 0.4, P["shoulder"][1] + 0.6), stride=1),
        "isolator, both walls (every 10th j)": dict(xlim=(200, 203), ylim=(38.9, 56.7), stride=1, jstride=10),
    }
    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 0.6])
    axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    colors = {0: "tab:blue", 1: "k", 2: "tab:red", 3: "tab:green"}
    for ax, (title, v) in zip(axes, list(views.items())[:3]):
        s = v["stride"]; js = v.get("jstride", s)
        for ib, (X, Y) in blocks.items():
            mask = (X[0] >= v["xlim"][0] - 1) & (X[0] <= v["xlim"][1] + 1)
            Xs, Ys = X[:, mask], Y[:, mask]
            for j in range(0, Xs.shape[0], js):
                ax.plot(Xs[j, ::1], Ys[j, ::1], color=colors[ib], lw=0.25)
            for i in range(0, Xs.shape[1], s):
                ax.plot(Xs[::1, i], Ys[::1, i], color=colors[ib], lw=0.25)
        ax.set_xlim(*v["xlim"]); ax.set_ylim(*v["ylim"]); ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)

    ax = fig.add_subplot(gs[2, :])
    ax.semilogy(x, np.full_like(x, m["dy_wall"]), "k-", lw=0.8, label="lower wall")
    ax.semilogy(x, dmid, "tab:red", lw=0.8, label="mid line (cowl / interface)")
    ax.axvline(x[i_lip], color="0.5", lw=0.5, ls="--")
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"first $\Delta y$"); ax.legend(fontsize=7)
    fig.savefig(fig_dir / "mesh_y.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
def plot_x(x, idx, names, y_lower, y_mid, y_top, fig_dir, stride=40):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.5), gridspec_kw={"height_ratios": [2, 1]}, sharex=True)

    a1.plot(x, y_lower, "k-", lw=0.8)
    a1.plot(x, y_mid, "k-", lw=0.8)
    a1.plot(x, y_top, "k-", lw=0.8)
    for i in range(0, x.size, stride):
        a1.plot([x[i]] * 2, [y_lower[i], y_top[i]], color="0.75", lw=0.3, zorder=0)
    for name, i in zip(names, idx):
        a1.axvline(x[i], color="tab:blue", lw=0.5, ls="--")
        a1.text(x[i], 82, name, rotation=90, fontsize=6, va="bottom", ha="center", color="tab:blue")
    a1.set_aspect("equal"); a1.set_ylabel(r"$y$"); a1.set_ylim(-5, 95)
    a1.set_title(f"x distribution: nx = {x.size} (every {stride}th line)", fontsize=9)

    a2.plot(0.5 * (x[1:] + x[:-1]), np.diff(x), "k.", ms=1)
    a2.set_ylabel(r"$\Delta x$"); a2.set_xlabel(r"$x$")
    fig.savefig(fig_dir / "mesh_x.png", bbox_inches="tight")
    plt.close(fig)


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
    blocks = assemble_blocks_2d(x, Y_low, Y_up, i_lip)
    check_y(x, Y_low, Y_up, blocks, geo, i_lip, m)
    print("  2D block shapes (ny, nx):", {b: v[0].shape for b, v in blocks.items()})
    n2d = sum(v[0].size for v in blocks.values())
    n_int = sum(blocks[b][0].size for b in (0, 1))
    print(f"  points per z-plane: {n2d:,}  (lower/internal {n_int:,} = {100 * n_int / n2d:.0f}%)"
          f"  -> x nz=200: {n2d * 200 / 1e6:.0f}M")
    plot_grid_zooms(blocks, x, dmid, i_lip, geo, fig_dir, m)

    # ---- stage 3: extrude + write ----
    if args.nz is not None:
        m["nz"] = args.nz
    if args.span is not None:
        m["span"] = args.span
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