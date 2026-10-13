# topology.py
import numpy as np
def faceted(x, Y_low, Y_up, i_lip):
    """4-block faceted topology: split both strips at the cowl lip column."""
    X_low = np.broadcast_to(x, Y_low.shape)
    X_up = np.broadcast_to(x, Y_up.shape)
    sl0, sl1 = slice(0, i_lip + 1), slice(i_lip, None)
    blocks = {
        0: (X_low[:, sl0].copy(), Y_low[:, sl0].copy()),   # lower, upstream of lip
        1: (X_low[:, sl1].copy(), Y_low[:, sl1].copy()),   # lower, intake/isolator
        2: (X_up[:, sl1].copy(), Y_up[:, sl1].copy()),     # upper, above cowl
        3: (X_up[:, sl0].copy(), Y_up[:, sl0].copy()),     # upper, upstream of lip
    }
    ifaces = [(0, 1, "col"),        # lip column, lower
              (3, 2, "col"),        # lip column, upper
              (0, 3, "row")]        # lip line upstream (flow passes through)
    info = {"i_lip": i_lip,
            "internal": [0, 1],     # duct: gets sidewalls
            "external": [2, 3]}     # outside the cowl: span relaxes to uniform
    return blocks, ifaces, info


def notched():
    """6-block faceted topology: split both strips at the cowl lip and facet lip columns."""
    X_low = np.broadcast_to(x, Y_low.shape)
    X_up = np.broadcast_to(x, Y_up.shape)
    sl0, sl1, sl2 = slice(0, i_lip + 1), slice(i_lip, i_facet), slice(i_facet, None)
    blocks = {
        0: (X_low[:, sl0].copy(), Y_low[:, sl0].copy()),   # lower, upstream of lip
        1: (X_low[:, sl1].copy(), Y_low[:, sl1].copy()),   # lower, intake/isolator
        2: (X_up[:, sl1].copy(), Y_up[:, sl1].copy()),     # upper, above cowl
        3: (X_up[:, sl0].copy(), Y_up[:, sl0].copy()),     # upper, upstream of lip
        4: (X_low[:, sl2].copy(), Y_low[:, sl2].copy()),   # lower, downstream of facet
        5: (X_up[:, sl2].copy(), Y_up[:, sl2].copy()),     # upper, downstream of facet
    }
    ifaces = [(0, 1, "col"),        # lip column, lower
              (3, 2, "col"),        # lip column, upper
              (0, 3, "row")]        # lip line upstream (flow passes through)
    info = {"i_lip": i_lip,
            "internal": [0, 1],     # duct: gets sidewalls
            "external": [2, 3]}     # outside the cowl: span relaxes to uniform
    return blocks, ifaces, info
    