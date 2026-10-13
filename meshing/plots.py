from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from meshing.build import build_z
from meshing.ops_writer import read_plane

BLOCK_COLORS = ["tab:blue", "k", "tab:red", "tab:green", "tab:purple", "tab:orange"]



def plot_z(blocks, info, m, fig_dir, stride_x=20, stride_z=4):
    """Plan view (x-z) of the spanwise distribution, internal (duct) blocks and external blocks."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True)
    x_lip = info["x_lip"]
    for ax, ids, external, title in [(axes[0], info["internal"], False, "internal blocks (duct, between sidewalls)"),
                                     (axes[1], info["external"], True, "external blocks (above the cowl)")]:
        for ib in ids:
            xb = blocks[ib][0][0]
            Z = build_z(xb, x_lip, m, external=external)
            c = BLOCK_COLORS[ib % len(BLOCK_COLORS)]
            for k in range(0, Z.shape[0], stride_z):
                ax.plot(xb, Z[k], color=c, lw=0.25)
            for i in range(0, xb.size, stride_x):
                ax.plot([xb[i]] * Z.shape[0], Z[:, i], color=c, lw=0.25)
        ax.axvline(x_lip, color="tab:red", lw=0.6, ls="--")
        ax.set_ylabel(r"$z$")
        ax.set_title(f"{title}: B{', B'.join(map(str, ids))}  (every {stride_z}th z-line)", fontsize=8)
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
def plot_grid_zooms(blocks, x, info, geo, fig_dir, m):
    P = geo["points"]
    S = P["shoulder"]
    views = {
        "full (every 20th line)": dict(xlim=(-5, x[-1] + 5), ylim=(-5, 85), stride=20),
        "cowl lip": dict(xlim=(P["cowl lip"][0] - 3, P["cowl lip"][0] + 3), ylim=(48.5, 51.5), stride=1),
        "shoulder / step": dict(xlim=(S[0] - 1.5, S[0] + 1.5), ylim=(S[1] - 1.3, S[1] + 0.6), stride=1),
    }
    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 0.6])
    axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for ax, (title, v) in zip(axes, views.items()):
        s = v["stride"]
        for ib, (X, Y) in blocks.items():
            c = BLOCK_COLORS[ib % len(BLOCK_COLORS)]
            mask = (X[0] >= v["xlim"][0] - 1) & (X[0] <= v["xlim"][1] + 1)
            Xs, Ys = X[:, mask], Y[:, mask]
            for j in range(0, Xs.shape[0], s):
                ax.plot(Xs[j], Ys[j], color=c, lw=0.25)
            for i in range(0, Xs.shape[1], s):
                ax.plot(Xs[:, i], Ys[:, i], color=c, lw=0.25)
        ax.set_xlim(*v["xlim"]); ax.set_ylim(*v["ylim"]); ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
    for ib in blocks:
        axes[0].plot([], [], color=BLOCK_COLORS[ib % len(BLOCK_COLORS)], label=f"B{ib}")
    axes[0].legend(fontsize=7, ncol=len(blocks) + 1, loc="upper left")

    ax = fig.add_subplot(gs[2, :])
    for (name, d), c in zip(info["spacing"].items(), ["k", "tab:red", "tab:orange"]):
        ax.semilogy(x, d, color=c, lw=0.8, label=name)
    ax.axvline(x[info["i_lip"]], color="0.5", lw=0.5, ls="--")
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
