from header import *


"""

    using CAD extracted points ets create the FSU intake

    points are digitised from image.
    we know for certain:
    y0 = 0
    θ0 = 6.7 degrees
    θ1 = 20.8 degrees

    
    isolator length 90
    cowl and facet angle 9 degs
    cowl lip (140.4, 50)

    
"""






# ======================================================================
# Case inputs
# ======================================================================


ndim = 3

# pull in csv and plot actual isentropic
HERE = r"G:\work\OpenSBLIEnvnew\Charlie_apps\6_Florida_State_Intake\3_Grid_Gen"
csv_path = r"\isentropic_curve.csv"

data = np.genfromtxt(HERE + csv_path, delimiter=',', skip_header=1)




curve_x = data[:, 0]
curve_y = data[:, 1]





case = {
    "casename": "3d_notched_sidewalls" ,
    "d99_to_dstar": 1.584,
    "IBLT": 0.07,
    "H_t": 17.3,
    "Lz": 25.0,
    "isolator_length": 90.0,
}
case["scale_factor"] = case["H_t"] * case["IBLT"] / case["d99_to_dstar"]




dydx = np.gradient(curve_y, curve_x)
theta = np.degrees(np.arctan(dydx))





# isolator
iso_length = 90
iso_end = curve_x[-1] + iso_length
iso_height = 56.3
lower_height = 39

points = {

    'cowl_lip':        (140.4, 50.0),
    'cowl_end':        (180, iso_height),
    'curve_end':       (curve_x[-1], curve_y[-1]),
    'facet_end':       (182.4, lower_height),
    'isolator_top':    (iso_end, iso_height),
    'isolator_bottom': (iso_end, lower_height),

}



# cowl line:
cowl0 = points['cowl_lip']
cowl1 = points['cowl_end']

# facet line
facet0 = points['curve_end']
facet1 = points['facet_end']

# upper wall
upper0 = points['cowl_end']
upper1 = points['isolator_top']

# lower wall
lower0 = points['facet_end']
lower1 = points['isolator_bottom']





# test points

print("Key points")
for k in points.keys():
    print(k + ": ", points[k])



cowl_grad = (cowl1[1] - cowl0[1]) / (cowl1[0] - cowl0[0])
print(f"cowl angle: {np.degrees(np.arctan(cowl_grad)):.2f}")
facet_grad = (facet1[1] - facet0[1]) / (facet1[0] - facet0[0])
print(f"facet angle: {np.degrees(np.arctan(facet_grad)):.2f}")
isolator_height = upper1[1] - lower1[1]
print(f"Isolator height: {isolator_height:.2f}")
intake_length_actual = lower1[0]
print(f"intake_length_actual: {intake_length_actual}")
cowl_lip = cowl0[0]





def calc_nx_global(nx_lip):

    """
        takes number of points up to cowl, extrapolates nx_global,
          adjusts intake length to ensure fit
    """

    # cowl_lip = 140.4
    dx = cowl_lip / (nx_lip - 1)
    print(f"dx: {dx}")

    nx_global = round(intake_length_actual / dx)
    print(f"nx_global: {nx_global}")

    intake_length_fit = nx_global * dx
    print(f"intake_length_fit: {intake_length_fit}")

    return nx_global, intake_length_fit





# --------------------------- Grid parameters --------------------------- #
nx_lip, ny_lip = 450, 100
# 1D coordinates
eta = np.linspace(0.0, 1.0, ny_lip)
Lz, nz = 25.0, 100



x, nx_global = build_uniform_x(nx_lip, x1=cowl_lip, Lx=intake_length_actual)
intake_length_fit = x[-1]






def build_y_lower_on(x):
    nx0 = 1000
    nx1 = 500

    # isentropic curve (raw)
    x0 = np.linspace(curve_x[0], curve_x[-1], nx0)
    y0 = np.interp(x0, curve_x, curve_y)

    # facet (curve_end -> facet1)
    x0_end, y0_end = x0[-1], y0[-1]
    x1_end, y1_end = facet1  # ensure facet1 is defined as (x,y)
    m1 = (y1_end - y0_end) / (x1_end - x0_end)
    x1 = np.linspace(x0_end, x1_end, nx1)
    y1 = y0_end + m1 * (x1 - x0_end)

    # lower isolator floor (horizontal to intake_length_fit)
    x2 = np.linspace(x1_end, intake_length_fit, nx1)
    y2 = np.full_like(x2, y1_end)

    x_raw = np.concatenate([x0, x1[1:], x2[1:]])
    y_raw = np.concatenate([y0, y1[1:], y2[1:]])

    return np.interp(x, x_raw, y_raw)





def build_y_notched(x, notch_depth):
    nx0 = 1000
    nx1 = 500

    # isentropic curve (raw)
    x0 = np.linspace(curve_x[0], curve_x[-1], nx0)
    y0 = np.interp(x0, curve_x, curve_y)

    # facet (curve_end -> facet1)
    x0_end, y0_end = x0[-1], y0[-1] - notch_depth
    x1_end, y1_end = facet1  # ensure facet1 is defined as (x,y)
    m1 = (y1_end - y0_end) / (x1_end - x0_end)
    x1 = np.linspace(x0_end, x1_end, nx1)
    y1 = y0_end + m1 * (x1 - x0_end)

    # lower isolator floor (horizontal to intake_length_fit)
    x2 = np.linspace(x1_end, intake_length_fit, nx1)
    y2 = np.full_like(x2, y1_end)

    x_raw = np.concatenate([x0, x1[0:], x2[1:]])
    y_raw = np.concatenate([y0, y1[0:], y2[1:]])

    return np.interp(x, x_raw, y_raw)








def build_y_upper_on(x):
    nxA = 200

    # freestream: x in [0, cowl_lip.x], y = cowl_lip.y
    x0 = np.linspace(0.0, cowl0[0], nxA)
    y0 = np.full_like(x0, cowl0[1])

    # cowl: cowl_lip -> cowl_end
    x1s, y1s = cowl0
    x1e, y1e = cowl1
    m1 = (y1e - y1s) / (x1e - x1s)
    x1 = np.linspace(x1s, x1e, nxA)
    y1 = y1s + m1 * (x1 - x1s)

    # upper isolator: horizontal to intake_length_fit at y = iso_height
    x2 = np.linspace(x1e, intake_length_fit, nxA)
    y2 = np.full_like(x2, y1e)

    x_raw = np.concatenate([x0, x1[1:], x2[1:]])
    y_raw = np.concatenate([y0, y1[1:], y2[1:]])

    return np.interp(x, x_raw, y_raw)



x = np.linspace(0.0, intake_length_fit, nx_global)
y_upper = build_y_upper_on(x)
y_lower = build_y_lower_on(x)
y_top   = np.full_like(x, 80.0)

notch_depth = 1.0
y_notched = build_y_notched(x, notch_depth)

x_notch = facet0[0]          # where the notch starts in x
i_notch = int(np.argmin(np.abs(x - x_notch)))
print("x_notch target:", x_notch, "grid x:", x[i_notch], "Δx:", x[i_notch]-x_notch)


y_notch_top = y_notched.copy()
y_notch_top[i_notch:] += notch_depth



plt.figure(figsize=(12,6))
plt.plot(x, y_top, 'k--', lw=1.5, label="Top wall")
plt.plot(x, y_upper, 'k', lw=2, label="upper wall")
plt.plot(x, y_lower, 'r', lw=2, label="Faceted wall")
plt.plot(x, y_notched, 'b', lw=1.5, label="Notched wall")
plt.plot(x, y_notch_top, 'g', lw=1.5)
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


y_lower = y_notch_top


# --------------------------- Lower wall mapping --------------------------- #
s0, s1, c0, c1 = 0.1738253, 0.16738253, 0.0, 0.0
P_lower = hermite_quintic_C2(eta, s0, s1, c0, c1) # stretching


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
S_upper = np.empty((ny_lip, x.size))
for i in range(x.size):
    S_upper[:, i] = hermite_quintic_C2(
        eta,
        s0=s0U[i], s1=2.0,
        c0=c0U[i], c1=0.1
    )

X_upper = np.tile(x[None, :], (ny_lip, 1))
Y_upper = y_upper[None, :] + S_upper * (y_top - y_upper)[None, :]

# --------------------------- Assemble full domain --------------------------- #
X_global = np.concatenate((X_lower[:-1, :], X_upper), axis=0)
Y_global = np.concatenate((Y_lower[:-1, :], Y_upper), axis=0)

x_B, y_B = block_slice(X_global, Y_global, nx_lip, ny_lip)

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





def build_notch_block():

    # Locate notch entry
    i_notch = np.where(np.isclose(x_B[1][0,:], 156.4, rtol=1e-3))[0][0]

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


    return X5, Y5


x_B[5], y_B[5] = build_notch_block()

x_B[4] = x_B[2]
y_B[4] = y_B[2]


# Locate notch entry
i_notch = np.where(np.isclose(x_B[1][0,:], 156.4, rtol=1e-3))[0][0]
# slice up B1
x_B[2] = x_B[1][:, i_notch:]
y_B[2] = y_B[1][:, i_notch:]
x_B[1] = x_B[1][:, :i_notch+1]
y_B[1] = y_B[1][:, :i_notch+1]




# --------------------------- Visualisation --------------------------- #
if 1:
    plot_stretched_grid(x_B, y_B)
    labels = [ 'Upper', 'Top', 'lower']
    # plot_wall_functions(x,  y_upper, y_top, y_lower, labels=labels, H_t=scale_factor)
    plt.show()



if ndim == 2:
    for i in range(len(x_B)):
        x_B[i] = x_B[i][:-1, :-1] / scale_factor
        y_B[i] = y_B[i][:-1, :-1] /  scale_factor
    print_summary(x_B, y_B)

if ndim == 3:
    z = np.linspace(0.0, Lz, nz)
    z_B = {}
    for i in range(len(x_B)):
        x2d, y2d = x_B[i], y_B[i]
        x3d = np.tile(x2d[None, :, :], (nz, 1, 1))
        y3d = np.tile(y2d[None, :, :], (nz, 1, 1))
        z3d = z[:, None, None] * np.ones_like(x3d) 
        x_B[i], y_B[i], z_B[i] = x3d, y3d, z3d

        x_B[i] = x_B[i][:-1, :-1, :-1] / scale_factor
        y_B[i] = y_B[i][:-1, :-1, :-1] / scale_factor
        z_B[i] = z_B[i][:-1, :-1, :-1] / scale_factor
        
    
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


