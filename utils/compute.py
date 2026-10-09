import numpy as np
from numpy import zeros, tan, radians, sinh, tanh, sqrt, arctan
try:
    from scipy.integrate import cumulative_trapezoid
except ImportError:
    from scipy.integrate import cumtrapz as cumulative_trapezoid

def build_uniform_x(nx_0, x0=0.0, x1=401.1, Lx=551.4):
    """make a uniform linspace up to end of intake, ensuring a point is at 0 and 401.1"""
    actual_Lx = Lx
    dx = (x1 - x0) / nx_0
    nx_1 = int(round((actual_Lx - x1) / dx))
    Lx_1_new_end = nx_1 * dx + x1
    nx  = nx_0 + nx_1 + 1
    x = np.linspace(x0, Lx_1_new_end, nx)

    return x, nx


def calc_nx_global(nx_lip, cowl_lip, intake_length_actual):

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



def linspace_with_point(x0, x1, x_lip, nx):
    """
    creates a linspace ensuring point on step by varying length
    overall length may change slightly to ensure x_lip
    """

    L_guess = x1 - x0
    a = (x_lip - x0) / L_guess

    # nearest intended node location
    k = int(round((nx - 1) * a))
    k = max(1, min(k, nx - 2))   # keep inside domain

    dx = (x_lip - x0) / k
    x1_new = x0 + (nx - 1) * dx
    x = x0 + np.arange(nx) * dx

    return x, x1_new, k


def build_lower_wall(x, curve_x, curve_y, x0, x1, y0, y1):
    """
    Map any x array to the corresponding lower-wall y values.

    Lower wall consists of:
      1. isentropic curve for x < x0
      2. straight facet from (x0, y0) to (x1, y1) for x0 <= x < x1
      3. horizontal lower isolator at y = y1 for x >= x1

    """

    m = (y1 - y0) / (x1 - x0)
    
    y = np.empty_like(x, dtype=float)
    # Region 1: isentropic curve
    mask0 = x < x0
    if np.any(mask0):
        y[mask0] = np.interp(x[mask0], curve_x, curve_y)

    # Region 2: straight facet
    mask1 = (x >= x0) & (x < x1)
    if np.any(mask1):
        y[mask1] = y0 + m * (x[mask1] - x0)

    # Region 3: horizontal isolator floor
    mask2 = x >= x1
    if np.any(mask2):
        y[mask2] = y1

    return y






def build_upper_wall(x, x0, x1, y0, y1,):
    """
        provides the y coordinate of the upper wall for each x point:

                cowl_lip.y,                                      x<cowl_lip.x
        y(x) =  m(x-cowl_lip.x) + cowl_lip.y,    isolator_start.x>=x>=cowl_lip.x
                isolator_start.y,                                   x>=isolator_start.x

                m = (isolator_start.y - cowl_lip.y) / (isolator_start.x - cowl_lip.x)
    """


    y = np.empty_like(x, dtype=float)

    m = (y1 - y0) / (x1 - x0)



    m = (y1 - y0) / (x1 - x0)

    y = np.empty_like(x, dtype=float)

    # Region 1: freestream upstream of cowl lip
    mask0 = x < x0
    y[mask0] = y0

    # Region 2: sloping cowl section
    mask1 = (x >= x0) & (x < x1)
    y[mask1] = y0 + m * (x[mask1] - x0)

    # Region 3: horizontal isolator / downstream
    mask2 = x >= x1
    y[mask2] = y1

    return y



def spanwise_stretch(stretch_type, Lz, nz, nx=None,  bz_max=2.0, a=8.0, i0=0.5,):

    """ 
        symmetric tanh stretching across z with bz as a function of x 
        returns nz, nx array of z coords

        bz_max = max value of bz
        a = steepness
        i0 = inflection location

    """


    if stretch_type == "uniform":
        zeta = np.linspace(-1.0, 1.0, nz, endpoint=False)
        iota = np.linspace(0.0, 1.0, nx)
        # z = 0.5 * Lz * zeta[:, None]
        bz_x = np.zeros(nx)
        z = np.broadcast_to(
            (0.5 * Lz * zeta)[:, None],
            (nz, nx)
        )
        return z, zeta, iota, bz_x

    zeta = np.linspace(-1.0, 1.0, nz, endpoint=True)
    iota = np.linspace(0.0, 1.0, nx)

    # streamwise variation of stretching strength
    raw = np.tanh(a*(iota - i0))
    bz_x = bz_max * (raw - raw[0]) / (raw[-1] - raw[0])

    # build ratio safely
    ratio = np.empty((nz, nx))
    small = np.abs(bz_x) < 1e-8

    ratio[:, small] = zeta[:, None]

    ratio[:, ~small] = (
        np.tanh(bz_x[None, ~small] * zeta[:, None]) /
        np.tanh(bz_x[None, ~small])
    )

    z = 0.5 * Lz * (1 + ratio)

    return z, zeta, iota, bz_x




def construct_3D_blocks(z, Blocks, x, nx_lip, scalefactor=1.0):
    """
        construct 3D blocks using 2d xy planes and xz planes
    """

    z_B, y_B, x_B = {}, {}, {}

    x_blocks = {
        0: x[:nx_lip+1],
        1: x[nx_lip:],
        2: x[nx_lip:],
        3: x[:nx_lip+1],
    }

    z_blocks = {
        0: z[:, :nx_lip+1],
        1: z[:, nx_lip:],
        2: z[:, nx_lip:],
        3: z[:, :nx_lip+1],
    }

    for ib, y2d in enumerate(Blocks):
        xb = x_blocks[ib]          # (nx_block,)
        zb = z_blocks[ib]          # (nz, nx_block)

        ny, nx_block = y2d.shape
        nz = zb.shape[0]



        # x: repeat along z and y
        x3d = np.broadcast_to(xb[None, None, :], (nz, ny, nx_block))

        # y: repeat along z
        y3d = np.broadcast_to(y2d[None, :, :], (nz, ny, nx_block))

        # z: repeat along y
        z3d = np.broadcast_to(zb[:, None, :], (nz, ny, nx_block))

        print("using scalefactor:", scalefactor)

        x_B[ib] = x3d.copy() / scalefactor
        y_B[ib] = y3d.copy() / scalefactor
        z_B[ib] = z3d.copy() / scalefactor

    return z_B, y_B, x_B


def extract_isentropic_curve():
    
    from pathlib import Path

    """Load isentropic curve data (works on Windows and Linux)."""

    base_dir = Path(__file__).resolve().parent
    file_path = base_dir  / "isentropic_curve.csv"

    data = np.genfromtxt(file_path, delimiter=",", skip_header=1)

    curve_x = data[:, 0]
    curve_y = data[:, 1]

    return curve_x, curve_y


# ----------------------------
# Geometry helpers
# ----------------------------
def gradient(p0, p1):
    return (p1[1] - p0[1]) / (p1[0] - p0[0])

def angle_deg(p0, p1):
    return np.degrees(np.arctan(gradient(p0, p1)))

def length(p0, p1):
    return np.hypot(p1[0] - p0[0], p1[1] - p0[1])

def height(top, bottom):
    return top[1] - bottom[1]





def hermite_quintic_polynomial(eta, s0=0.0, s1=0.0, c0=0.0, c1=0.0):
    """
    C^2 Hermite quintic on [0,1] with endpoint data:

        hermite allows tuning of endpoint gradient and second derivative
    


        P(0)  = 0
        P(1)  = 1
        P'(0) = s0
        P'(1) = s1
        P''(0)= c0
        P''(1)= c1


        returns y vector prior to upper lower wall mappings

        

    """


    t  = np.clip(np.asarray(eta, float), 0.0, 1.0)

    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t

    H00 = 1.0 - 10.0*t3 + 15.0*t4 - 6.0*t5
    H10 = t - 6.0*t3 + 8.0*t4 - 3.0*t5
    H20 = 0.5 * (t2 - 3.0*t3 + 3.0*t4 - t5)

    H01 = 10.0*t3 - 15.0*t4 + 6.0*t5
    H11 = -4.0*t3 + 7.0*t4 - 3.0*t5
    H21 = 0.5 * (t3 - 2.0*t4 + t5)

    y = H01 + s0*H10 + c0*H20 + s1*H11 + c1*H21

    return y



def gaussian_kernel(dx, sigma_cells, nsigma):
    """Gaussian kernel for smoothing wall.
        sigmacells : width of smoothing
        nsigma : how far tail extends
    """
    if sigma_cells <= 0:
        return np.array([1.0])           # no smoothing
    sigma = sigma_cells * dx
    half = int(np.ceil(nsigma*sigma/dx))
    s = np.arange(-half, half+1)*dx
    k = np.exp(-0.5*(s/sigma)**2); k /= k.sum()
    return k


def build_wall_smoothed(x, y0, locs, angles_deg, dx, sigma_cells=1, nsigma=3,
                        x_ref=None, y_ref=None, pad_mode='edge', sharp_at='none', tol=1e-9):
    """
    C∞ wall in x: build a piecewise-constant slope field from segment angles,
    smooth it with a Gaussian, then integrate (trapezoid) and anchor at y0. 
    If (x_ref, y_ref) is given, enforce y(x_ref)=y_ref
    """
    # piecewise-constant slope from segment angles
    # piecewise-constant slopes: intervals (-inf,loc0),(loc0,loc1),...,(loc_{n-1},+inf)
    edges  = np.r_[-1e9, np.array(locs, float), 1e9]
    slopes = np.r_[0.0, np.tan(np.radians(angles_deg))]  # len = len(locs)+1
    g = np.zeros_like(x)
    
    for s0, s1, m in zip(edges[:-1], edges[1:], slopes):
        mask = (x >= s0) & (x < s1)
        g[mask] = m

    
    # Gaussian smoothing with edge padding
    k = gaussian_kernel(dx, sigma_cells=sigma_cells, nsigma=nsigma)
    k = k / k.sum()
    pad = len(k) // 2

    # Decide which locs are "sharp"
    if sharp_at == 'all':
        sharp_mask = np.ones_like(locs, dtype=bool)
    elif sharp_at == 'none':
        sharp_mask = np.zeros_like(locs, dtype=bool)
    else:
        sharp_list = np.atleast_1d(sharp_at).astype(float)
        sharp_mask = np.array([np.any(np.isclose(L, sharp_list, atol=tol)) for L in locs], dtype=bool)

    # Build cut indices: only at sharp locs (so we *don’t* smooth across them)
    idxs = [np.searchsorted(x, L) for L, keep_sharp in zip(locs, sharp_mask) if keep_sharp]
    cuts = [0] + idxs + [len(x)]

    # Smooth piecewise without crossing sharp corners
    if len(cuts) == 2:
        # no sharp cuts -> single convolution over entire array
        g_s = np.convolve(np.pad(g, pad, mode=pad_mode), k, mode='valid')
    else:
        g_s = np.empty_like(g)
        for a, b in zip(cuts[:-1], cuts[1:]):
            chunk = g[a:b]
            if chunk.size == 0:
                continue
            chunk_s = np.convolve(np.pad(chunk, pad, mode=pad_mode), k, mode='valid')
            g_s[a:b] = chunk_s

    # integrate slope to get y(x)
    I = np.r_[0.0, np.cumsum(0.5*(g_s[1:] + g_s[:-1]) * np.diff(x))]

    # single-point anchor (no distortion of slopes/curvatures)
    if (x_ref is not None) and (y_ref is not None):
        I_ref = np.interp(x_ref, x, I)
        y = y_ref + (I - I_ref)
    else:
        y = y0 + I

    return y



def simple_wall(x, locs, angles_deg):
    """
    Construct a simple ramped wall profile with optional light Gaussian smoothing.

    """
    # --- 1. Build piecewise-constant slope field --- #
    edges  = np.r_[-1e9, np.array(locs, float), 1e9]
    slopes = np.r_[0.0, np.tan(np.radians(angles_deg))]   # len = len(locs)+1
    g = np.zeros_like(x)

    for s0, s1, m in zip(edges[:-1], edges[1:], slopes):
        mask = (x >= s0) & (x < s1)
        g[mask] = m

    # --- 3. Integrate slope to get y(x) --- #
    y= np.r_[0.0, np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(x))]

    return y




def build_isentropic(x_start=227.7, x_end=483.0, y_start=26.0, thetas=[6.5, 14.2]):
    """

    """

    def prandtl_meyer(M, gama=1.4):
        A = sqrt((((gama - 1) / (gama + 1)) * (M**2 - 1)))
        B = sqrt(M**2 - 1)
        nu = sqrt((gama + 1) / (gama - 1)) * arctan(A) - arctan(B)
        return nu

    def inverse_prandtl_meyer(nu_target, gamma=1.4, tol=1e-8):
        # Newton iteration to find M from ν
        M = 2.0
        for _ in range(100):
            f = prandtl_meyer(M, gamma) - nu_target
            df = (
                sqrt(M**2 - 1)
                / (1 + 0.5 * (gamma - 1) * M**2)
                / M
            )
            M_new = M - f / df
            if abs(M_new - M) < tol:
                break
            M = M_new
        return M

    # lets make an isentropic curve given inflow angle, outflow angle, length
    gama = 1.4
    theta_1 = radians(thetas[0])
    theta_2 = radians(thetas[1])
    M1 = 4.93 # 6.6 deg turn of mach 6 flow
    L = x_end - x_start
    nx = 200

    # ---- Compute theta, ν, M distributions ----
    theta = np.linspace(theta_1, theta_2, nx)
    nu1 = prandtl_meyer(M1, gama)
    nu = nu1 + (theta - theta_1)
    M = np.array([inverse_prandtl_meyer(nui, gama) for nui in nu])


    # ---- Build isentropic wall curve ----
    x = np.linspace(x_start, x_end, nx)
    y = np.zeros_like(x)
    for i in range(1, nx):
        dx = x[i] - x[i - 1]
        dy = np.tan(theta[i]) * dx
        y[i] = y[i - 1] + dy
    y += y_start

    # ---- Get coordinates of the isentropic wall curve ----
    # Local turning defines wall slope: dy/dx = tan(theta)
    y = np.zeros_like(x)
    for i in range(1, nx):
        dy = np.tan(theta[i]) * (x[i] - x[i - 1])
        y[i] = y[i - 1] + dy


    y = y + y_start


    return y, x



def build_wall_compound(x, x_locs, angles_deg, M1=4.93,):
    """
    Build a full lower-wall geometry consisting of:
        ramp → isentropic curve → ramp → ramp
    """
    y = np.zeros_like(x)

    # --- 1. First ramp --- #
    x1, x2 = x_locs[0], x_locs[1]
    m1 = np.tan(np.radians(angles_deg[0]))
    i1, i2 = np.searchsorted(x, [x1, x2])
    y[i1:i2] = m1 * (x[i1:i2] - x1)
    y_join_1 = y[i2 - 1]

    # --- 2. Isentropic curved section --- #
    x3, x4 = x_locs[1], x_locs[2]
    i3, i4 = np.searchsorted(x, [x3, x4])
    theta_1 = np.radians(angles_deg[1])
    theta_2 = np.radians(angles_deg[2])
    theta = np.linspace(theta_1, theta_2, i4 - i3)

    y[i3] = y_join_1
    for j in range(i3 + 1, i4):
        dy = np.tan(theta[j - i3]) * (x[j] - x[j - 1])
        y[j] = y[j - 1] + dy
    y_join_2 = y[i4 - 1]

    # --- 3. Third ramp --- #
    x5, x6 = x_locs[2], x_locs[3]
    i5, i6 = np.searchsorted(x, [x5, x6])
    m3 = np.tan(np.radians(angles_deg[3]))
    y[i5:i6] = y_join_2 + m3 * (x[i5:i6] - x5)
    y_join_3 = y[i6 - 1]



    # --- 4. Fourth ramp --- #
    x5, x6 = x_locs[3], x_locs[4]
    i5, i6 = np.searchsorted(x, [x5, x6])
    m3 = np.tan(np.radians(angles_deg[4]))
    y[i5:i6] = y_join_3 + m3 * (x[i5:i6] - x5)
    y_join_4 = y[i6 - 1]

    return y


def build_wall_new(x, x_locs, angles_deg):
    """
    Build lower-wall geometry consisting of:
        ramp → isentropic curve → ramp → ramp
        
    x_locs = [x1, x2, x3, x4, x5]
    angles_deg = [ramp1_angle, -, -, ramp3_angle, ramp4_angle]
    """

    y = np.zeros_like(x)

    # --------------------------
    # 1. First ramp  (x1 → x2)
    # --------------------------
    x1, x2 = x_locs[0], x_locs[1]
    i1, i2 = np.searchsorted(x, [x1, x2])
    m1 = np.tan(np.radians(angles_deg[0]))

    y[i1:i2] = m1 * (x[i1:i2] - x1)
    y_join_1 = y[i2-1]

    # --------------------------
    # 2. Isentropic section (x2 → x3)
    # --------------------------
    x3, x4 = x_locs[1], x_locs[2]
    i3, i4 = np.searchsorted(x, [x3, x4])
    
    # local coordinate in [0,1]
    s_local = (x[i3:i4] - x3) / (x4 - x3)

    # endpoint angles
    theta0 = np.radians(6.7)
    theta1 = np.radians(20.8)

    # smooth basis
    def C1(s): return s**2 * (1 - s)**2
    def C2(s): return s**3 * (1 - s)**2
    def C3(s): return s**2 * (1 - s)**3

    # your optimized parameters
    a, b, c = -0.4869003331181572, 1.4903021210056506, -1.8097491346735368

    def theta_full(s):
        H = 3*s**2 - 2*s**3
        return theta0 + (theta1 - theta0)*H + a*C1(s) + b*C2(s) + c*C3(s)

    dydx = np.tan(theta_full(s_local))

    # integrate locally over this segment
    y_iso_local = cumulative_trapezoid(dydx, x[i3:i4], initial=0.0)

    # shift so it joins ramp 1 perfectly
    y[i3:i4] = y_join_1 + y_iso_local
    y_join_2 = y[i4-1]   # this fixes your missing variable

    # --------------------------
    # 3. Third ramp  (x3 → x4)
    # --------------------------
    x5, x6 = x_locs[2], x_locs[3]
    i5, i6 = np.searchsorted(x, [x5, x6])
    m3 = np.tan(np.radians(angles_deg[3]))

    y[i5:i6] = y_join_2 + m3*(x[i5:i6] - x5)
    y_join_3 = y[i6-1]

    # --------------------------
    # 4. Fourth ramp  (x4 → x5)
    # --------------------------
    x7, x8 = x_locs[3], x_locs[4]
    i7, i8 = np.searchsorted(x, [x7, x8])
    m4 = np.tan(np.radians(angles_deg[4]))

    y[i7:i8] = y_join_3 + m4*(x[i7:i8] - x7)

    return y


def build_wall_CAD(x, x_locs, angles_deg):
    """
    Build lower-wall geometry consisting of:
    cad curve, facet, flat
    """




    y = np.zeros_like(x)





    return y



def build_notched_wall(x, x_locs, angles_deg, y_start):
    """
    Build lower-wall geometry:
        ramp1 (x1→x2), ramp2 (x2→x3), ramp3 (x3→x4)

    x_locs = [x1, x2, x3, x4]
    angles_deg = [θ1, θ2, θ3]
    """
    y = np.zeros_like(x) + y_start

    # Unpack
    x1, x2  = x_locs
    th1, th2 = angles_deg

    # Precompute slopes
    m1 = np.tan(np.radians(th1))
    m2 = np.tan(np.radians(th2))

    # Indices
    i1, i2 = np.searchsorted(x, [x1, x2 ])

    # --- Ramp 1 ---
    y[i1:i2] = y_start + m1 * (x[i1:i2] - x1)
    y_join_1 = y[i2-1]

    # --- Ramp 2 ---
    y[i2:] = y_join_1 + m2 * (x[i2:] - x2)
    # y_join_2 = y[i3-1]

    # # --- Ramp 3 ---
    # y[i3:i4] = y_join_2 + m3 * (x[i3:i4] - x3)

    return y





def block_slice(X, Y, nx0, ny0):
    """
    Return 4 rectangular blocks split at column nx0 and row ny0.
    The split row/column are INCLUDED in both adjacent blocks
    so interfaces share identical nodes.
    """
    ny, nx = X.shape
    x_B, y_B = {}, {}

    # bottom-left: j=0..ny0, i=0..nx0
    x_B[0] = X[0:ny0+1, 0:nx0+1].copy()
    y_B[0] = Y[0:ny0+1, 0:nx0+1].copy()

    # bottom-right: j=0..ny0, i=nx0..nx-1
    x_B[1] = X[0:ny0+1, nx0:nx].copy()
    y_B[1] = Y[0:ny0+1, nx0:nx].copy()

    # top-right: j=ny0..ny-1, i=nx0..nx-1
    x_B[2] = X[ny0:ny, nx0:nx].copy()
    y_B[2] = Y[ny0:ny, nx0:nx].copy()
    # print(y_B[2][-1, :])

    # top-left: j=ny0..ny-1, i=0..nx0
    x_B[3] = X[ny0:ny, 0:nx0+1].copy()
    y_B[3] = Y[ny0:ny, 0:nx0+1].copy()

    return x_B, y_B



def check_c0_continuity(x_B, y_B, tol=1e-12):
    """ ensure that grids match at interfaces"""
    print("Checking c0 continuity...")
    def check(a, b, dx, dy, tol):
        max_dx = np.max(np.abs(dx))
        max_dy = np.max(np.abs(dy))
        print(f" Block {a} - {b}")
        if max_dx < tol and max_dy < tol:
            print("PASS")
        else:
            print(f"FAIL: max Δx={max_dx:.3e}, max Δy={max_dy:.3e}")



    a, b = 0, 3
    dx = x_B[a][-1, :] - x_B[b][0, :]
    dy = y_B[a][-1, :] - y_B[b][0, :]
    check(a, b, dx, dy, tol)


    a, b = 0, 1
    dx = x_B[a][:, -1] - x_B[b][:, 0]
    dy = y_B[a][:, -1] - y_B[b][:, 0]
    check(a, b, dx, dy, tol)



    a, b = 1, 2
    dx = x_B[a][-1, :] - x_B[b][0, :]
    dy = y_B[a][-1, :] - y_B[b][0, :]
    check(a, b, dx, dy, tol)    


    a, b = 2, 3
    dx = x_B[a][:, 0] - x_B[b][:, -1]
    dy = y_B[a][:, 0] - y_B[b][:, -1]
    check(a, b, dx, dy, tol)
    return
