
# %%

from header import *


"""
LAPCAT grid generation script:
-------------------------------
 - Polynomial y-stretching using Hermite quintic curves
 - Lower intake wall constructed from smoothed ramp segments
 - Upper and freestream walls built with C2 continuity
 - Outputs structured 2D/3D grid blocks for OpenSBLI
"""





casename = "case_3_3d"
d99_to_dstar = 1.584
IBLT = 0.07  # d99 to H_t
ndim = 3



theta0 = np.radians(6.7)
theta1 = np.radians(20.8)
H_t = 17.3  # 
scale_factor = H_t * IBLT / d99_to_dstar

intake_LE_x = -10.0
cowl_lip = (140.4, 50.0) 
Lx = 250 




# ----------------------------------------------------------------------
# Improved θ(s) with 3 controllable modes
# ----------------------------------------------------------------------
# smooth endpoint basis functions (zero & zero-derivative at s=0,1)

def main():







    # --------------------------- Grid parameters --------------------------- #
    nx0, ny0 = 450, 110
    Lz, nz = 75.0, 100
    # 1D coordinates
    eta = np.linspace(0.0, 1.0, ny0)
    x, nx = build_uniform_x(nx0, x0=intake_LE_x, x1=cowl_lip[0], Lx=Lx)
    dx = float(np.diff(x)[0])  # uniform spacing for smoothing kernel


    # --------------------------- Lower wall mapping --------------------------- #
    s0, s1, c0, c1 = 0.1738253, 0.16738253, 0.0, 0.0
    P_lower = hermite_quintic_C2(eta, s0, s1, c0, c1) # stretching


    locs       = [intake_LE_x, 0, 156.4, 184, 310]
    angles_deg = [0.0, 6.7, 20.8, 9.0, 0.0]
    y_lower = build_wall_new(x, locs, angles_deg,) # wall shape






    # --------------------------- Upper wall mapping --------------------------- #

    y_top = np.full_like(x, 80.0)
    locs, angles_deg = [140.1, 184], [9.0, 0.0]
    y_upper = build_wall_smoothed(x, cowl_lip[1], locs, angles_deg, dx, x_ref=cowl_lip[0], y_ref=cowl_lip[1], sharp_at=[cowl_lip[0]])

    # Lower block (bottom half of domain)
    S_lower = np.asarray(P_lower)[:, None]
    X_lower = np.tile(x[None, :], (S_lower.shape[0], 1))
    Y_lower = y_lower[None, :] + S_lower * (y_upper - y_lower)[None, :]

    # --------------------------- Upper block (top half of domain) --------------------------- #
    s_inherit, c_inherit = s1, c1
    s_top, c_top = 2.0, 0.1
    eta_upper = eta
    P_upper = hermite_quintic_C2(eta_upper, s0=s_inherit, s1=s_top,
                                  c0=c_inherit, c1=c_top)

    print(f"P_upper shape: {P_upper.shape}, eta_upper shape: {eta_upper.shape}")
    print(f"P_upper range: {P_upper.min():.3f} to {P_upper.max():.3f}")

    # Ensure geometric continuity at block join
    H_L = y_upper - y_lower
    H_U = y_top - y_upper
    eps = 1e-12
    if np.any(H_U <= eps):
        raise ValueError("Upper gap H_U too small/nonpositive at some columns.")

    ratio = np.clip(H_L / H_U, 1e-3, 1e3)
    s0U = s1 * ratio
    c0U = c1 * ratio**2

    # Column-wise Hermite mapping for C¹ continuity
    S_upper = np.empty((ny0, x.size))
    for i in range(x.size):
        S_upper[:, i] = hermite_quintic_C2(
            eta,
            s0=s0U[i], s1=2.0,
            c0=c0U[i], c1=0.1
        )

    X_upper = np.tile(x[None, :], (ny0, 1))
    Y_upper = y_upper[None, :] + S_upper * (y_top - y_upper)[None, :]

    # --------------------------- Assemble full domain --------------------------- #
    X_global = np.concatenate((X_lower[:-1, :], X_upper), axis=0)
    Y_global = np.concatenate((Y_lower[:-1, :], Y_upper), axis=0)

    x_B, y_B = block_slice(X_global, Y_global, nx0, ny0)

    # Check C0 and C1 continuity
    c0_err = np.max(np.abs(Y_lower[-1, :] - y_upper))
    assert c0_err < 1e-12, f"C0 mismatch: {c0_err:.3e}"
    c0_err2 = np.max(np.abs(Y_upper[0, :] - y_upper))
    assert c0_err2 < 1e-12, f"Upper join mismatch: {c0_err2:.3e}"

    dyL = Y_lower[-1, :] - Y_lower[-2, :]
    dyU = Y_upper[1, :] - Y_upper[0, :]
    c1_err = np.max(np.abs(np.abs(dyL) - np.abs(dyU)))
    print(f"C1 max |Δ(dy)| = {c1_err:.3e}")
    print("length:", x_B[1][0,-1])

    for i in range(len(x_B)):
        x_B[i] = x_B[i][:-1, :-1] / scale_factor
        y_B[i] = y_B[i][:-1, :-1] /  scale_factor


    if ndim == 2:
        print_summary(x_B, y_B)


    # --------------------------- Visualisation --------------------------- #
    if 1:
        plot_stretched_grid(x_B, y_B)
        labels = [ 'Upper', 'Top', 'lower']
        # plot_wall_functions(x,  y_upper, y_top, y_lower, labels=labels, H_t=scale_factor)
        plt.show()

    # --------------------------- Extrude to 3D --------------------------- #
    if ndim == 3:
        z = np.linspace(0.0, Lz, nz)
        z_B = {}
        for i in range(len(x_B)):
            x2d, y2d = x_B[i], y_B[i]
            x3d = np.tile(x2d[None, :, :], (nz, 1, 1))
            y3d = np.tile(y2d[None, :, :], (nz, 1, 1))
            z3d = z[:, None, None] * np.ones_like(x3d)
            x_B[i], y_B[i], z_B[i] = x3d, y3d, z3d

        print_summary_3D(x_B, y_B, z_B)

    # --------------------------- Save .dat files --------------------------- #

    if 1:
        for i in range(len(x_B)):
            fname = f"{casename}_B{i}.dat"
            if ndim == 2:
                x, y = x_B[i].T, y_B[i].T
                save_block_to_dat_2D(x, y, fname)
            if ndim == 3:
                x, y, z = x_B[i].T, y_B[i].T, z_B[i].T
                save_block_to_dat_3D(x, y, z, fname)



# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

# %%
