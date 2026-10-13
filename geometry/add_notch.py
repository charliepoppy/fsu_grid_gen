import numpy as np


def add_notch(G, cfg):
    """
    Notched variant of the solved faceted geometry G.

    Vertical step of notch_depth at the shoulder, then a straight floor at notch_angle
    until it meets the isolator floor. Everything else (ramp, turn, shoulder, cowl, ceiling,
    floor, outlet) is shared with the faceted geometry by construction.
    """
    P = G["pts"]
    LE, W, S = P["leading edge"], G["W"], P["shoulder"]
    x_out, floor = G["x_out"], G["floor"]

    d, a = cfg["notch_depth"], np.radians(cfg["notch_angle"])
    B = np.array([S[0], S[1] - d])                                   # step bottom
    E = np.array([B[0] + (floor - B[1]) / np.tan(a), floor])         # notch end (on the isolator floor)

    G = dict(G)                                                      # don't mutate the caller's dict
    G["pts"] = {**P, "step bottom": B, "notch end": E}
    G["lower_notched"] = np.vstack([LE, W, S, B, E, [x_out, floor]])
    return G
