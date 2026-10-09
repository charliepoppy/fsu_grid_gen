from geometry.gasdynamics import pm_nu, pm_M, oblique_shock
from geometry.lines import intersect
import numpy as np

def focused_isentropic_wall(Minf, th0_deg, th1_deg, F, LE, n=4000):
    """Wall turning th0 -> th1 whose left-running Mach waves all pass through F."""
    th0, th1 = np.radians(th0_deg), np.radians(th1_deg)
    M2 = oblique_shock(Minf, th0)["M2"]
    nu2 = pm_nu(M2)
    th = np.linspace(th0, th1, n)
    M = np.array([pm_M(nu2 - (t - th0)) for t in th])     # compression: nu falls
    phi = th + np.arcsin(1 / M)                              # wave angle = theta + mu
    W = [intersect(LE, th0, F, phi[0])]
    for i in range(1, n):
        W.append(intersect(W[-1], 0.5 * (th[i - 1] + th[i]), F, phi[i]))
    return np.array(W), phi, M
