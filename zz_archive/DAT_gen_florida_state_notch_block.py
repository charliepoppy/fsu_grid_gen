
# %%

from header import *


"""
LAPCAT grid generation script:
-------------------------------
 - Polynomial y-stretching using Hermite quintic curves
 - Lower intake wall constructed from smoothed ramp segments
 - Upper and freestream walls built with C2 continuity
 - Outputs structured 2D/3D grid blocks for OpenSBLI


 let's clean up the logic and add the same optimisation from BFS
"""





casename = "3D_notched"
d99_to_dstar = 1.584
IBLT = 0.07  # d99 to H_t
ndim = 3



H_t = 17.3  # 
scale_factor = H_t * IBLT / d99_to_dstar
# scale_factor = 1
intake_LE_x = -10.0
cowl_lip = (140.4, 50.0) 

def build_block(x, y_bot, y_top, S):
    """
    Build a structured block between y_bot(x) and y_top(x) using stretch S(η,x).
    """
    X = np.tile(x[None, :], (S.shape[0], 1))
    Y = y_bot[None, :] + S * (y_top - y_bot)[None, :]
    return X, Y



def main():



    # --------------------------- Grid parameters --------------------------- #
    nx0, ny0 = 450, 110
    Lz, nz = 25.0, 31

    eta = np.linspace(0.0, 1.0, ny0)
    x, _ = build_uniform_x(nx0, x0=intake_LE_x, x1=cowl_lip[0], Lx=250)
    dx = float(np.diff(x)[0]) 


    # --------------------------- Walls --------------------------- #


    locs_lower       = [intake_LE_x, 0, 156.4, 184, 310]
    angles_lower = [0.0, 6.7, 20.8, 17.0, 0.0]
    y_lower = build_wall_new(x, locs_lower, angles_lower,) # wall shape

    # Upper/top wall
    y_top = np.full_like(x, 80.0)
    locs_up, angles_up = [140.1, 184], [9.0, 0.0]
    y_upper = build_wall_smoothed(x, cowl_lip[1], locs_up, angles_up, dx, x_ref=cowl_lip[0], y_ref=cowl_lip[1], sharp_at=[cowl_lip[0]])




    # --------------------------- Lower block mapping --------------------------- #

    s0, s1, c0, c1 = 0.1738253, 0.16738253, 0.0, 0.0
    P_lower = hermite_quintic_C2(eta, s0, s1, c0, c1) # stretching
    S_lower = np.asarray(P_lower)[:, None]
    X_lower, Y_lower = build_block(x, y_lower, y_upper, S_lower)



    # --------------------------- Upper block --------------------------- #

    H_L = y_upper - y_lower
    H_U = y_top - y_upper

    eps = 1e-12
    if np.any(H_U <= eps):
        raise ValueError("Upper gap H_U too small")

    ratio = np.clip(H_L / H_U, 1e-3, 1e3)

    s0U = s1 * ratio
    c0U = c1 * ratio**2

    S_upper = np.empty((ny0, x.size))
    for i in range(x.size):
        S_upper[:, i] = hermite_quintic_C2(
            eta,
            s0=s0U[i], s1=2.0,
            c0=c0U[i], c1=0.1
        )

    X_upper, Y_upper = build_block(x, y_upper, y_top, S_upper)



    # --------------------------- Assemble global domain --------------------------- #
    X_global = np.vstack((X_lower[:-1], X_upper))
    Y_global = np.vstack((Y_lower[:-1], Y_upper))

    x_B, y_B = block_slice(X_global, Y_global, nx0, ny0)





    # --------------------------- Notch block construction --------------------------- #


    locs_notch   = [156.4, 184,]
    angles_notch = [18.5, 0.0, ]
    notch_depth = 1.0

    y_notched = build_notched_wall(x, locs_notch, angles_notch, y_start=35.0 - notch_depth) # wall shape


    # Locate notch entry
    i_notch = np.where(np.isclose(x_B[1][0,:], 156.4, rtol=1e-3))[0][0]

    P_lower = hermite_quintic_C2(eta, s0, s1, c0, c1) # stretching




    # --- Block 5: linear, uniform-stretched notch block ---
    ny4 = 30
    # 1. Top boundary from Block 1 at notch location
    y_top5 = y_B[1][0, i_notch:]

    # 2. Bottom boundary = notch depth below the top
    y_bot5 = y_top5 - notch_depth

    # 3. Vertical resolution (as you had before)
    ny5 = ny4  # or set explicitly

    # 4. Uniform eta
    eta5 = np.linspace(0.0, 1.0, ny5)[:, None]   # shape (ny5,1)

    # 5. Pure linear interpolation between top and bottom
    Y5 = y_bot5[None, :] + eta5 * (y_top5 - y_bot5)[None, :]

    # 6. X copies from Block 1 slice
    X5 = np.tile(x_B[1][0, i_notch:][None, :], (ny5, 1))

    # 7. Save block
    x_B[5] = X5
    y_B[5] = Y5



    x_B[4] = x_B[2]
    y_B[4] = y_B[2]


    # slice up B1
    x_B[2] = x_B[1][:, i_notch:]
    y_B[2] = y_B[1][:, i_notch:]
    x_B[1] = x_B[1][:, :i_notch+1]
    y_B[1] = y_B[1][:, :i_notch+1]


    # --------------------------- Continuity checks --------------------------- #
    assert np.allclose(Y_lower[-1], y_upper, atol=1e-12)
    assert np.allclose(Y_upper[0], y_upper, atol=1e-12)

    dyL = Y_lower[-1] - Y_lower[-2]
    dyU = Y_upper[1] - Y_upper[0]
    print(f"C1 mismatch: {np.max(np.abs(np.abs(dyL) - np.abs(dyU))):.3e}")

    # --------------------------- Trim halos and scale --------------------------- #

    for i in range(len(x_B)):
        x_B[i] = x_B[i][:-1, :-1] / scale_factor
        y_B[i] = y_B[i][:-1, :-1] /  scale_factor




    if ndim == 2:
        print_summary(x_B, y_B)



    # --------------------------- Visualisation --------------------------- #
    if 1:
        plot_stretched_grid(x_B, y_B, stride=1,)
        labels = [ 'Upper', 'Top', 'lower', 'notched']
        plot_wall_functions(x,  y_upper, y_top, y_lower, y_notched, labels=labels, H_t=scale_factor)
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

            x3d = x3d[:-1, :, :]
            y3d = y3d[:-1, :, :]
            z3d = z3d[:-1, :, :]


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
