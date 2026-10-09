import numpy as np
from geometry.gasdynamics import pm_nu, pm_M, oblique_shock
from geometry.moc import focused_isentropic_wall
from geometry.lines import intersect, ang




# ======================================================================
# Geometry
# ======================================================================


def solve_geometry(cfg):
    """Everything derived from CFG. Returns dict G."""
    LE, lip, F = map(np.asarray, (cfg["leading_edge"], cfg["cowl_lip"], cfg["focus"]))
    th1 = np.radians(cfg["theta1"])
    tc, tf = np.tan(np.radians(cfg["cowl_angle"])), np.tan(np.radians(cfg["facet_angle"]))

    LE_shock = oblique_shock(cfg["Minf"], np.radians(cfg["theta0"]))
    W, phi, M = focused_isentropic_wall(cfg["Minf"], cfg["theta0"], cfg["theta1"], F, LE)

    xs = cfg["shoulder_x"]
    shoulder = np.array([xs, W[-1, 1] + np.tan(th1) * (xs - W[-1, 0])])
    ceiling, floor = cfg["ceiling"], cfg["floor"]
    cowl_end = np.array([lip[0] + (ceiling - lip[1]) / tc, ceiling])
    facet_end = np.array([shoulder[0] + (floor - shoulder[1]) / tf, floor])
    x_out = facet_end[0] + cfg["iso_length"]

    cowl_shock = oblique_shock(M[-1], th1 - np.radians(cfg["cowl_angle"]))
    cs_ang = cfg["theta1"] - np.degrees(cowl_shock["beta"])
    impinge = intersect(lip, np.radians(cs_ang), shoulder, np.radians(cfg["facet_angle"]))

    lower = np.vstack([LE, W, shoulder, facet_end, [x_out, floor]])
    upper = np.vstack([lip, cowl_end, [x_out, ceiling]])

    pts = {"leading edge": LE, "turn start": W[0], "turn end": W[-1], "shoulder": shoulder,
           "facet end": facet_end, "cowl lip": lip, "cowl end": cowl_end,
           "outlet floor": np.array([x_out, floor]), "outlet ceiling": np.array([x_out, ceiling]),
           "cowl shock impingement": impinge}

    return dict(W=W, phi=phi, M=M, F=F, LE_shock=LE_shock, cowl_shock=cowl_shock, cs_ang=cs_ang,
                ceiling=ceiling, floor=floor, x_out=x_out, lower=lower, upper=upper, pts=pts)

