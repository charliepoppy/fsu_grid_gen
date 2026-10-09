import numpy as np
from scipy.optimize import brentq, minimize_scalar
# ======================================================================
# Gas dynamics
# ======================================================================
def pm_nu(M, g=1.4):
    """Prandtl-Meyer function nu(M), radians."""
    return (np.sqrt((g + 1) / (g - 1)) * np.arctan(np.sqrt((g - 1) / (g + 1) * (M**2 - 1)))
            - np.arctan(np.sqrt(M**2 - 1)))


def pm_M(nu, g=1.4):
    """Inverse Prandtl-Meyer: M from nu."""
    return brentq(lambda M: pm_nu(M, g) - nu, 1 + 1e-12, 50)


def theta_from_beta(M, beta, g=1.4):
    return np.arctan(2 / np.tan(beta) * (M**2 * np.sin(beta)**2 - 1)
                     / (M**2 * (g + np.cos(2 * beta)) + 2))


def beta_from_theta(M, theta, g=1.4, strong=False):
    """Shock angle for deflection theta (radians). Weak branch by default."""
    mu = np.arcsin(1 / M)
    res = minimize_scalar(lambda b: -theta_from_beta(M, b, g),
                          bounds=(mu, np.pi / 2), method="bounded")
    beta_max, theta_max = res.x, -res.fun
    if theta > theta_max:
        raise ValueError(f"detached: theta {np.degrees(theta):.2f} > theta_max {np.degrees(theta_max):.2f} deg")
    f = lambda b: theta_from_beta(M, b, g) - theta
    lo, hi = (beta_max, np.pi / 2) if strong else (mu + 1e-12, beta_max)
    return brentq(f, lo, hi)


def oblique_shock(M1, theta, g=1.4):
    """Post-shock state across a weak oblique shock. theta in radians."""
    beta = beta_from_theta(M1, theta, g)
    Mn1 = M1 * np.sin(beta)
    p = 1 + 2 * g / (g + 1) * (Mn1**2 - 1)
    r = (g + 1) * Mn1**2 / ((g - 1) * Mn1**2 + 2)
    Mn2 = np.sqrt((1 + 0.5 * (g - 1) * Mn1**2) / (g * Mn1**2 - 0.5 * (g - 1)))
    return {"beta": beta, "M2": Mn2 / np.sin(beta - theta), "p2_p1": p, "rho2_rho1": r, "T2_T1": p / r}

