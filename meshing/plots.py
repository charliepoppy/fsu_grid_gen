import matplotlib.pyplot as plt
import numpy as np
from meshing.build import build_x, build_y, build_z
from meshing.ops_writer import read_plane
from pathlib import Path

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
