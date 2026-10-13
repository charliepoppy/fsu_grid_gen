import matplotlib.pyplot as plt
import numpy as np
from geometry.lines import line

# ======================================================================
# Plotting
# ======================================================================
CONSTRUCT = dict(color="0.6", lw=0.5, zorder=1)
WALL = dict(color="k", lw=0.7, zorder=3)
NOTCH = dict(color="k", lw=0.7, zorder=3)
DIGITISED = dict(marker="s", ls="", ms=1, mfc="tab:blue", mec="none", zorder=5)
KNOWN = dict(marker="o", ls="", ms=1, mfc="tab:orange", mec="none", zorder=5)
CORNER = dict(marker="o", ls="", ms=1, color="k", mec="none", zorder=4)
LABEL_OFFSET = {"curve end": (5, -10), "cowl end": (5, 8), "facet end": (5, -10), "cowl lip": (5, 8)}
DPI = 600



def draw_construction(ax, G, cfg, ref):
    P, W, lip = G["pts"], G["W"], G["pts"]["cowl lip"]

    # construction (grey)
    line(ax, P["leading edge"], cfg["theta0"], 110, **CONSTRUCT)
    line(ax, P["turn end"], cfg["theta1"], 170, **CONSTRUCT)
    line(ax, lip, cfg["cowl_angle"], 200, **CONSTRUCT)
    line(ax, P["shoulder"], cfg["facet_angle"], 200, **CONSTRUCT)
    for i in range(0, len(W), 200):
        ax.plot([W[i, 0], G["F"][0]], [W[i, 1], G["F"][1]], **CONSTRUCT)
    ax.axhline(0.0, **CONSTRUCT); ax.axvline(0.0, **CONSTRUCT)

    # walls
    ax.plot(G["lower"][:, 0], G["lower"][:, 1], **WALL, label="wall (construction)")
    ax.plot(G["upper"][:, 0], G["upper"][:, 1], **WALL)
    ax.plot([G["x_out"]] * 2, [G["floor"], G["ceiling"]], **CONSTRUCT)
    for name in ("turn start", "turn end", "shoulder", "facet end", "cowl end"):
        ax.plot(*P[name], **CORNER)
    ax.plot(*G["F"], "o", color="r", ms=1, mew=0, zorder=4)


    # notched variant (only if add_notch has run)
    if "lower_notched" in G:
        ln = G["lower_notched"]
        ax.plot(ln[:, 0], ln[:, 1], **NOTCH, label="notched wall")
        line(ax, P["step bottom"], cfg["notch_angle"], 200, **CONSTRUCT)   # the 10.6° construction line
        for name in ("step bottom", "notch end"):
            ax.plot(*P[name], **CORNER)


    # shocks
    LS, CS = G["LE_shock"], G["cowl_shock"]
    line(ax, P["leading edge"], np.degrees(LS["beta"]), lip[0], color="tab:red", lw=0.8,
         label=rf"LE shock $\beta$ = {np.degrees(LS['beta']):.2f}°, $M_2$ = {LS['M2']:.3f}")
    imp = P["cowl shock impingement"]
    ax.plot([lip[0], imp[0]], [lip[1], imp[1]], color="tab:red", lw=0.8, ls="--",
            label=rf"cowl shock $\beta$ = {np.degrees(CS['beta']):.1f}°, $M_3$ = {CS['M2']:.3f}")
    ax.plot(*imp, "o", color="tab:red", ms=2.5, mec="none", zorder=5)

    # reference points
    for name, p in ref["digitised"].items():
        ax.plot(*p, **DIGITISED)
        ax.annotate(name, p, textcoords="offset points", xytext=LABEL_OFFSET.get(name, (5, -10)), fontsize=8)
    ax.plot([], [], **DIGITISED, label="digitised")
    for name in ("leading edge", "cowl lip"):
        ax.plot(*P[name], **KNOWN)
        ax.annotate(name, P[name], textcoords="offset points", xytext=LABEL_OFFSET.get(name, (5, -10)), fontsize=8)
    ax.plot([], [], **KNOWN, label="known")

    # isolator height
    xo = G["x_out"]
    ax.annotate("", xy=(xo, G["ceiling"]), xytext=(xo, G["floor"]),
                arrowprops=dict(arrowstyle="<->", color="0.3", lw=0.8, shrinkA=0, shrinkB=0))
    ax.text(xo + 2, 0.5 * (G["ceiling"] + G["floor"]), rf"$H$ = {G['ceiling'] - G['floor']:.2f}",
            va="center", ha="left", fontsize=8, color="0.3")


    # ramp angles
    ax.text(0.35 * (P["turn start"][0] + P["turn end"][0]), 0.18 * (P["turn start"][1]),
            rf"$\theta_0$ = {cfg['theta0']:.2f}°", fontsize=8, color="0.3", rotation=0,
            va="bottom", ha="center")

    ax.text(0.5 * (P["turn end"][0] + P["shoulder"][0]), 0.25 * (P["turn end"][1] + P["shoulder"][1]),
            rf"$\theta_1$ = {cfg['theta1']:.2f}°", fontsize=8, color="0.3", rotation=cfg["theta1"],
            va="bottom", ha="center")



def plot_MoC(G, cfg, ref, fig_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    draw_construction(ax, G, cfg, ref)
    ax.set_aspect("equal")
    ax.set_xlim(-10, G["x_out"] + 25); ax.set_ylim(-10, 70)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")
    ax.set_title(rf"Intake construction, $M_\infty$ = {cfg['Minf']}")
    ax.legend(loc="upper left", fontsize=7)
    fig.savefig(fig_dir / "MoC.png", dpi=DPI)
    plt.close(fig)


def plot_overlay_zooms(img, extent, G, cfg, ref, fig_dir):
    zooms = {
        "full": dict(xlim=(-10, G["x_out"] + 25), ylim=(-10, 70)),
        "ramp / spline join": dict(xlim=(100, 128), ylim=(12, 28)),
        "shoulder / throat": dict(xlim=(135, 192), ylim=(30, 60)),
    }
    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for ax, (title, lim) in zip(axes, zooms.items()):
        ax.imshow(img, extent=extent, origin="upper", alpha=0.35, zorder=0)
        draw_construction(ax, G, cfg, ref)
        ax.set_xlim(*lim["xlim"]); ax.set_ylim(*lim["ylim"])
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
    axes[0].legend(loc="upper left", fontsize=6)
    for ax in axes[1:]:
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        axes[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="tab:blue", lw=0.8, zorder=6))
    fig.savefig(fig_dir / "overlay_zoom.png", dpi=DPI)
    plt.close(fig)


def plot_cad_residual(G, fig_dir):
    if "cad_residual" not in G:
        return
    x, r = G["cad_residual"]
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.plot(x, r, "k-", lw=0.8)
    ax.axhline(0, color="0.6", lw=0.5)
    for name in ("turn start", "turn end", "shoulder"):
        ax.axvline(G["pts"][name][0], color="tab:blue", lw=0.5, ls="--")
        ax.text(G["pts"][name][0], ax.get_ylim()[1], f" {name}", fontsize=7, va="top", color="tab:blue")
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y_{CAD} - y_{MoC}$")
    ax.set_title("Lower wall: CAD minus construction")
    fig.savefig(fig_dir / "validation_cad_residual.png", dpi=DPI,    bbox_inches="tight")
    plt.close(fig)
