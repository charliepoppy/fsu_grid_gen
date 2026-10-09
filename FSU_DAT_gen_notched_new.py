#!/usr/bin/env python3
from header import *

"""
FSU intake grid generator — notched variant

Block layout:
    [3][ 4  ]
    [0][1][2]
          [5]   ← notch block (below B1 aft / B2)

All geometry defined in 2D (ny, nx), then extruded spanwise to 3D (nz, ny, nx).
"""

# ======================================================================
# Case inputs
# ======================================================================

case = {
    "casename":       "3d_notched_sidewalls",
    "d99_to_dstar":   1.584,
    "IBLT":           0.07,
    "H_t":            17.3,
    "Lz":             25.0,
    "isolator_length": 90.0,
}
case["scale_factor"] = case["H_t"] * case["IBLT"] / case["d99_to_dstar"]

# ======================================================================
# Geometry
# ======================================================================

curve_x, curve_y = extract_isentropic_curve()
Lx = curve_x[-1] + case["isolator_length"]

pts = {
    "cowl_lip":        (140.4, 50.0),
    "curve_end":       (curve_x[-1], curve_y[-1]),
    "facet_start":     (curve_x[-1], curve_y[-1]),
    "cowl_end":        (180.0, 56.3),
    "facet_end":       (182.4, 39.0),
    "isolator_top":    (Lx, 56.3),
    "isolator_bottom": (Lx, 39.0),
}

print("Key points")
for name, pt in pts.items():
    print(f"  {name}: {pt}")
print(f"  Cowl angle:       {angle_deg(pts['cowl_lip'],   pts['cowl_end']):.2f} deg")
print(f"  Facet angle:      {angle_deg(pts['curve_end'],  pts['facet_end']):.2f} deg")
print(f"  Isolator height:  {height(pts['isolator_top'],  pts['isolator_bottom']):.2f}")

# ======================================================================
# Grid parameters
# ======================================================================

NX, NY, NZ = 800, 100, 100
NY_NOTCH    = 30
NOTCH_DEPTH = 1.0

eta  = np.linspace(0.0, 1.0, NY)
x, Lx_grid, nx_lip = linspace_with_point(0.0, Lx, pts["cowl_lip"][0], NX)

assert np.isclose(x[nx_lip], pts["cowl_lip"][0], atol=1e-3), \
    "Cowl lip not exactly on x-grid."

# Notch entry = curve_end x, located on global x array
i_notch_global = int(np.argmin(np.abs(x - pts["curve_end"][0])))
print(f"\nNotch entry: target x={pts['curve_end'][0]:.4f}, "
      f"grid x={x[i_notch_global]:.4f}, "
      f"Δx={x[i_notch_global] - pts['curve_end'][0]:.4f}")

# ======================================================================
# Stretching functions (1D eta → physical)
# ======================================================================

P_lower = hermite_quintic_polynomial(eta, s0=0.1738253,  s1=0.16738253, c0=0.0, c1=0.0)
P_upper = hermite_quintic_polynomial(eta, s0=0.16738253, s1=2.0,        c0=0.0, c1=0.1)



z, zeta, iota, bz_x = spanwise_stretch(
    case["Lz"], NZ, nx=NX,
    bz_max=2.0, a=8.0,
    i0=nx_lip / NX,
    endpoint=True     # for sidewalls
)

# ======================================================================
# Wall curves (1D, defined on global x)
# ======================================================================

upper_wall = build_upper_wall(
    x,
    pts["cowl_lip"][0],  pts["cowl_end"][0],
    pts["cowl_lip"][1],  pts["cowl_end"][1],
)
lower_wall = build_lower_wall(
    x, curve_x, curve_y,
    pts["facet_start"][0], pts["facet_end"][0],
    pts["facet_start"][1], pts["facet_end"][1],
)
y_top = np.full_like(x, 80.0)

# ======================================================================
# 2D mapped planes  (ny, nx)
# ======================================================================

H_lower = upper_wall - lower_wall
H_upper = y_top - upper_wall

lower_2d = lower_wall[None, :] + H_lower[None, :] * P_lower[:, None]   # (ny, nx)
upper_2d = upper_wall[None, :] + H_upper[None, :] * P_upper[:, None]   # (ny, nx)

# Shared x array tiled to (ny, nx)
X_2d = np.tile(x[None, :], (NY, 1))

# ======================================================================
# 2D block definitions — all slicing happens here, before any 3D work
# ======================================================================

# i_notch_B1: index within the B1 x-slice (B1 starts at nx_lip)
# B1 x runs from x[nx_lip] to x[-1], so notch index within B1:
i_notch_B1 = i_notch_global - nx_lip

assert i_notch_B1 > 0, \
    f"Notch index ({i_notch_global}) must be after cowl lip ({nx_lip})"

def make_notch_2d(lower_2d_B1, x_B1):
    """
    Build the notch block 2D grid below the B1 lower wall from notch onwards.
    lower_2d_B1 : (ny, nx_B1) — the B1 2D plane
    x_B1        : (nx_B1,)    — x coords of B1
    Returns X5, Y5 each (NY_NOTCH, nx_notch)
    """
    # Bottom wall of B1 (j=0) from notch entry onwards
    y_top5 = lower_2d_B1[0, i_notch_B1:]   # (nx_notch,)
    y_bot5 = y_top5 - NOTCH_DEPTH

    eta5 = np.linspace(0.0, 1.0, NY_NOTCH)[:, None]   # (NY_NOTCH, 1)

    Y5 = y_bot5[None, :] + eta5 * (y_top5 - y_bot5)[None, :]  # (NY_NOTCH, nx_notch)
    X5 = np.tile(x_B1[i_notch_B1:][None, :], (NY_NOTCH, 1))   # (NY_NOTCH, nx_notch)
    return X5, Y5

# B1/B2 x-columns (B1 goes up to and including notch entry; B2 from notch onwards)
x_B1 = x[nx_lip:]                 # full B1 x range before split

X5_2d, Y5_2d = make_notch_2d(lower_2d[:, nx_lip:], x_B1)

blocks_2d = {
    0: (X_2d[:, :nx_lip + 1],          lower_2d[:, :nx_lip + 1]),
    1: (X_2d[:, nx_lip:nx_lip + i_notch_B1 + 1],  lower_2d[:, nx_lip:nx_lip + i_notch_B1 + 1]),
    2: (X_2d[:, nx_lip + i_notch_B1:], lower_2d[:, nx_lip + i_notch_B1:]),
    3: (X_2d[:, :nx_lip + 1],          upper_2d[:, :nx_lip + 1]),
    4: (X_2d[:, nx_lip:],              upper_2d[:, nx_lip:]),
    5: (X5_2d,                         Y5_2d),
}

print("\n2D block shapes (ny, nx):")
for b, (Xb, Yb) in blocks_2d.items():
    print(f"  B{b}: {Xb.shape}")
# ======================================================================
# 3D extrusion — spanwise (nz, ny, nx)
# ======================================================================

def construct_3D_blocks(z, x, blocks_2d, nx_lip, i_notch_global, scale_factor):
    """
    Extrude 2D blocks (ny, nx) to 3D (nz, ny, nx).
    z-coordinate uses the x-appropriate column of the spanwise stretch array.

    Block x-ranges for z-slice selection:
        B0, B3  : x[:nx_lip+1]
        B1      : x[nx_lip : nx_lip+i_notch_B1+1]   (pre-notch)
        B2, B4  : x[nx_lip+i_notch_B1:]              (post-notch / upper aft)
        B5      : same x-range as B2 (notch pocket)
    """
    i_notch_B1 = i_notch_global - nx_lip

    # z column index ranges per block, matching their x extent
    z_slices = {
        0: slice(None,              nx_lip + 1),
        1: slice(nx_lip,            nx_lip + i_notch_B1 + 1),
        2: slice(nx_lip + i_notch_B1, None),
        3: slice(None,              nx_lip + 1),
        4: slice(nx_lip,            None),
        5: slice(nx_lip + i_notch_B1, None),
    }

    x_B, y_B, z_B = {}, {}, {}

    for ib, (X2d, Y2d) in blocks_2d.items():
        zb = z[:, z_slices[ib]]          # (nz, nx_block)
        ny, nx_block = X2d.shape
        nz = zb.shape[0]

        assert zb.shape[1] == nx_block, (
            f"Block {ib}: z slice width {zb.shape[1]} != X2d width {nx_block}"
        )

        x_B[ib] = np.broadcast_to(X2d[None, :, :], (nz, ny, nx_block)).copy() / scale_factor
        y_B[ib] = np.broadcast_to(Y2d[None, :, :], (nz, ny, nx_block)).copy() / scale_factor
        z_B[ib] = np.broadcast_to(zb[:, None, :],  (nz, ny, nx_block)).copy() / scale_factor

    return x_B, y_B, z_B


x_B, y_B, z_B = construct_3D_blocks(
    z, x, blocks_2d, nx_lip, i_notch_global, case["scale_factor"]
)

print("\n3D block shapes (nz, ny, nx):")
for b in x_B:
    print(f"  B{b}: {x_B[b].shape}")






# ======================================================================
# Visualisation
# ======================================================================


if True:
    spanwise_stretch_plot(z, iota)
    plot_poly_stretch(eta, P_lower)
    plot_upper_lower(eta, P_upper, P_lower)
    plot_stretched_grid(x_B, y_B,stride=1)
    plt.show()




print_summary_3D(x_B, y_B, z_B)

# ======================================================================
# Save
# ======================================================================

save_HDF5 = 0
if save_HDF5:
    print("\n\n=== Saving to HDF5 ===")
    arrays_to_HDF5(case["casename"], x_B, y_B, z_B, plot_hdf5=True)