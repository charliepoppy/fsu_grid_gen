import matplotlib.pyplot as plt
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


