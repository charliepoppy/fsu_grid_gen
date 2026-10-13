# add the notch to the geometry prior to meshing

# takes the solved faceted geometry G and adds three things:

# step bottom: the shoulder, moved down by notch_depth
# notch end: where a line from the step bottom at notch_angle meets the isolator floor
# lower_notched: the full lower-wall polyline with the step in it
import numpy as np
def add_notch(G, cfg):
    """Vertical step of notch_depth at the shoulder, floor at notch_angle back up to the isolator floor."""
    x_out, floor = G["x_out"], G["floor"]
    P = G["pts"]
    S = P["shoulder"]
    d, a = cfg["notch_depth"], np.radians(cfg["notch_angle"])
    B = np.array([S[0], S[1] - d])                       # straight down by d
    E = np.array([B[0] + (floor - B[1]) / np.tan(a), floor])                                # notch end
    LE, W = P["leading edge"], G["W"]
    lower_notched = np.vstack([LE, W, S, B, E, [x_out, floor]])
    G = dict(G)                               # don't mutate the caller's dict
    G["pts"] = {**P, "step bottom": B, "notch end": E}
    G["lower_notched"] = lower_notched
    return G