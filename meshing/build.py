import numpy as np
# ======================================================================
# x distribution
# ======================================================================
def _graded(L, d_fine, dx, gmax):
    """
    Cell sizes covering length L: geometric from d_fine (growth gmax) until the next cell
    would reach dx, then uniform cells (<= dx) for the remainder. Fine end first.
    """
    layer = []
    d = d_fine
    while d < dx and sum(layer) + d < L:
        layer.append(d)
        d *= gmax
    rem = L - sum(layer)
    n_u = max(1, int(np.ceil(rem / dx - 1e-9)))
    return np.r_[layer, np.full(n_u, rem / n_u)]


def build_x(stations, dx_target, refine=None, gmax=1.05):
    """
    x nodes with a node exactly on every station. Returns x, station indices, per-segment max dx.

    refine : {station index: dx at that station} - segments touching a refined station are
             graded geometrically (growth <= gmax) from that dx up to dx_target.
             Segments without a refined end are uniform (exactly as before).
    """
    xs = np.asarray(stations, float)
    if np.any(np.diff(xs) <= 0):
        raise ValueError(f"stations must be strictly increasing: {xs}")
    refine = refine or {}
    segs, n, dmax = [], [], []
    for k, (a, b) in enumerate(zip(xs[:-1], xs[1:])):
        fa, fb = refine.get(k), refine.get(k + 1)
        if fa is None and fb is None:                       # uniform segment (unchanged behaviour)
            nk = max(1, int(round((b - a) / dx_target)))
            segs.append(np.linspace(a, b, nk + 1)[:-1])
            dmax.append((b - a) / nk)
        else:
            if fa is not None and fb is not None:           # fine at both ends: grade to the middle
                half = _graded(0.5 * (b - a), min(fa, fb), dx_target, gmax)
                d = np.r_[half, half[::-1]]
            elif fa is not None:
                d = _graded(b - a, fa, dx_target, gmax)
            else:
                d = _graded(b - a, fb, dx_target, gmax)[::-1]
            pts = a + np.r_[0.0, np.cumsum(d)[:-1]]
            segs.append(pts)
            nk = d.size
            dmax.append(d.max())
        n.append(segs[-1].size)
    x = np.r_[np.concatenate(segs), xs[-1]]
    idx = np.r_[0, np.cumsum(n)]
    x[idx] = xs                                             # stations exact (no round-off from cumsum)
    return x, idx, np.array(dmax)


def wall_core(n, L, d0, rmax=1.05):
    """
    n points on [0, L]: geometric layer from each wall (first cell d0, growth r <= rmax)
    then a uniform core whose spacing equals the last layer cell (spacing is continuous).
    Chooses the layer depth that minimises the core spacing. Returns z, (r, m, d_core).
    """
    from scipy.optimize import brentq
    cells, best = n - 1, None
    for m_ in range(1, cells // 2):
        nc = cells - 2 * m_
        f = lambda r: d0 * r**m_ - (L - 2.0 * d0 * (r**m_ - 1.0) / (r - 1.0)) / nc
        try:
            r = brentq(f, 1.0 + 1e-9, 2.0)
        except ValueError:
            continue
        if r <= rmax and (best is None or d0 * r**m_ < best[2]):
            best = (r, m_, d0 * r**m_)
    if best is None:
        raise ValueError(f"no wall+core distribution with n={n}, L={L}, d0={d0}, growth<={rmax}")
    r, m_, dc = best
    layer = d0 * r ** np.arange(m_)
    d = np.r_[layer, np.full(cells - 2 * m_, dc), layer[::-1]]
    z = np.r_[0.0, np.cumsum(d)]
    return z * (L / z[-1]), best



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

# ======================================================================
# Spanwise extrusion + write
# ======================================================================
def _smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)



def sidewall_weight(x, x_lip, upper, m):
    """
    0 -> uniform span, 1 -> sidewall-clustered span.
      x <  lip : ramps 0 -> 1 over dz_blend_up approaching the lip (all blocks; B3 must equal B0)
      x >= lip : lower blocks (B1, between the sidewalls) stay 1;
                 upper blocks (B2, external, no sidewalls) relax 1 -> 0 over dz_blend_down
    """
    w = 1.0 - _smoothstep((x_lip - x) / m["dz_blend_up"])
    if upper:
        w = np.where(x >= x_lip, 1.0 - _smoothstep((x - x_lip) / m["dz_blend_down"]), w)
    return np.where(x >= x_lip, w if upper else 1.0, w)


def build_z(xb, x_lip, m, external=False):
    """
    (nz, nx_block) spanwise coordinates for a block with x-coordinates xb.
    external : True for blocks outside the duct (above the cowl) - the sidewall
               clustering relaxes back to uniform downstream of the lip.
    """
    nz, Lz = m["nz"], m["Lz"]
    if m["span"] == "periodic":
        z1 = 0.5 * Lz * np.linspace(-1.0, 1.0, nz, endpoint=False)
        return np.broadcast_to(z1[:, None], (nz, xb.size))
    if m["span"] != "sidewall":
        raise ValueError(f"unknown span type {m['span']!r}")
    # sidewall: nodes on both z = 0 and z = Lz (walls on the internal blocks), same nz in every block
    z_uni = np.linspace(0.0, Lz, nz)
    z_wall, _ = wall_core(nz, Lz, m["dz_wall"], rmax=m["dz_growth"])
    w = sidewall_weight(xb, x_lip, upper=external, m=m)
    return z_uni[:, None] * (1.0 - w[None, :]) + z_wall[:, None] * w[None, :]
