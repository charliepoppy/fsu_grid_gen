"""
Block topologies. Each builder takes the same inputs and returns the same three things:

    blocks : {ib: (X, Y)}           2D block planes, shape (ny, nx_block)
    ifaces : [(a, b, kind), ...]    flow interfaces; "col" = a's last column is b's first column,
                                    "row" = a's top row is b's bottom row
    info   : roles and bookkeeping  internal / external blocks, walls, split indices, spacing laws

Nothing outside this file needs to know how many blocks there are.
"""
import numpy as np

from meshing.build import build_columns, build_y, dy_mid, dy_upper_bottom, _smoothstep


def _cut(x, Y, s):
    """(X, Y) for the columns s of a strip Y."""
    return np.broadcast_to(x[s], Y[:, s].shape).copy(), Y[:, s].copy()


# ======================================================================
# Faceted: 4 blocks
# ======================================================================
def faceted(x, W, idx, names, m):
    """
        [3][   2   ]     3/2 : above the lip line / cowl (external)
        [0][   1   ]     0/1 : ramp -> facet -> isolator (internal)
    split at the cowl lip column.
    """
    i_lip = idx[names.index("cowl lip")]
    Y_low, Y_up, dmid = build_y(x, W["lower"], W["mid"], W["top"], x[i_lip], m)

    a, b = slice(0, i_lip + 1), slice(i_lip, None)
    blocks = {0: _cut(x, Y_low, a), 1: _cut(x, Y_low, b),
              2: _cut(x, Y_up, b), 3: _cut(x, Y_up, a)}
    ifaces = [(0, 1, "col"),        # lip column, lower
              (3, 2, "col"),        # lip column, upper
              (0, 3, "row")]        # lip line upstream of the lip
    info = dict(
        i_lip=i_lip, internal=[0, 1], external=[2, 3], duct=1,
        walls={"lower wall (to lip)": (0, "bottom"), "ramp/facet/floor": (1, "bottom"),
               "cowl inner": (1, "top"), "cowl outer": (2, "bottom")},
        spacing={"lower wall": np.full_like(x, m["dy_wall"]), "mid line (cowl / interface)": dmid},
    )
    return blocks, ifaces, info


# ======================================================================
# Notched: 6 blocks
# ======================================================================
def notched(x, W, idx, names, m):
    """
        [3][        4         ]     3/4 : external
        [0][ 1 ][      2      ]     0/1/2 : duct; B1 ends at the step column
                 [      5      ]     5 : notch layer, notched wall -> (notched wall + step height)
    The B5/B2 line is the old wall level carried downstream (notched wall + h). It is a flow
    interface: fine at the step lip (shear layer), relaxing to dy_notch_top downstream.
    """
    i_lip = idx[names.index("cowl lip")]
    i_st = idx[names.index("shoulder")]
    x_lip, x_st = x[i_lip], x[i_st]
    down = x >= x_st
    h = W["lower"][i_st] - W["notch"][i_st]                              # step height

    # spacing on the B5/B2 line: dy_wall at the step lip, blending to dy_notch_top
    d_top5 = m["dy_wall"] + (m["dy_notch_top"] - m["dy_wall"]) * _smoothstep((x - x_st) / m["notch_top_blend"])

    # main lower strip: faceted wall up to the step, (notched wall + h) from the step on
    y_bot = np.where(down, np.nan_to_num(W["notch"]) + h, W["lower"])
    d_bot = np.where(down, d_top5, m["dy_wall"])
    dmid = dy_mid(x, x_lip, m)
    dup = dy_upper_bottom(x, x_lip, dmid, m)
    Y_low = build_columns(y_bot, W["mid"], m["ny_lower"], d_bot, dmid)
    Y_up = build_columns(W["mid"], W["top"], m["ny_upper"], dup, np.full_like(x, m["dy_top"]))

    # notch layer: only from the step column on
    s = slice(i_st, None)
    Y5 = build_columns(W["notch"][s], y_bot[s], m["ny_notch"], np.full(x[s].size, m["dy_wall"]), d_top5[s])

    blocks = {0: _cut(x, Y_low, slice(0, i_lip + 1)),
              1: _cut(x, Y_low, slice(i_lip, i_st + 1)),
              2: _cut(x, Y_low, s),
              3: _cut(x, Y_up, slice(0, i_lip + 1)),
              4: _cut(x, Y_up, slice(i_lip, None)),
              5: (np.broadcast_to(x[s], Y5.shape).copy(), Y5)}
    ifaces = [(0, 1, "col"),        # lip column, lower
              (1, 2, "col"),        # step column, above the step top
              (3, 4, "col"),        # lip column, upper
              (0, 3, "row"),        # lip line upstream of the lip
              (5, 2, "row")]        # notch layer top / duct bottom (old wall level)
    info = dict(
        i_lip=i_lip, i_step=i_st, h_step=h, internal=[0, 1, 2, 5], external=[3, 4], duct=2,
        walls={"lower wall (to step)": (1, "bottom"), "notch floor / isolator floor": (5, "bottom"),
               "step face (x)": (5, "left"), "cowl inner": (2, "top"), "cowl outer": (4, "bottom")},
        spacing={"lower wall": np.full_like(x, m["dy_wall"]), "mid line (cowl / interface)": dmid,
                 "notch layer top (B5/B2)": np.where(down, d_top5, np.nan)},
    )
    return blocks, ifaces, info


TOPOLOGY = {"faceted": faceted, "notched": notched}
