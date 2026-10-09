import numpy as np
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


def dy_mid(x, x_lip, m):
    """
    Spacing along the mid line (cowl downstream of lip, B0/B3 interface upstream).
    = dy_wall for x >= x_lip; smooth (C1) blend to dy_interface over blend_length upstream.
    """
    s = np.clip((x_lip - x) / m["blend_length"], 0.0, 1.0)
    S = s * s * (3.0 - 2.0 * s)                       # smoothstep
    return m["dy_wall"] + (m["dy_interface"] - m["dy_wall"]) * S


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





def build_columns(y_bot, y_top, n, d_bot, d_top):
    """(n, nx) array of y: each column stretched between y_bot and y_top with end spacings d_bot, d_top."""
    H = y_top - y_bot
    Y = np.empty((n, y_bot.size))
    for i in range(y_bot.size):
        Y[:, i] = y_bot[i] + H[i] * vinokur(n, d_bot[i] / H[i], d_top[i] / H[i])
    return Y



def build_y(x, y_lower, y_mid, y_top, x_lip, m):
    dmid = dy_mid(x, x_lip, m)
    dup = dy_upper_bottom(x, x_lip, dmid, m)
    Y_low = build_columns(y_lower, y_mid, m["ny_lower"], np.full_like(x, m["dy_wall"]), dmid)
    Y_up = build_columns(y_mid, y_top, m["ny_upper"], dup, np.full_like(x, m["dy_top"]))
    return Y_low, Y_up, dmid

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
