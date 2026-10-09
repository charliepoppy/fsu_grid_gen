from numpy import sqrt, arctan, radians, degrees, sin, tan
import numpy as np
import matplotlib.pyplot as plt

def prandtl_meyer(M, gama=1.4):
    A = sqrt(((gama - 1) / (gama + 1)) * (M**2 - 1))
    B = sqrt(M**2 - 1)
    return sqrt((gama + 1) / (gama - 1)) * np.arctan(A) - np.arctan(B)

def inverse_prandtl_meyer(nu_target, gamma=1.4, tol=1e-8):
    M = 2.0
    for _ in range(100):
        f = prandtl_meyer(M, gamma) - nu_target
        df = sqrt(M**2 - 1) / (1 + 0.5*(gamma - 1)*M**2) / M
        M_new = M - f/df
        if abs(M_new - M) < tol:
            break
        M = M_new
    return M


# ---------------------------------------------------------------------
# Oblique shock angle solver (Newton iteration)
# ---------------------------------------------------------------------
def oblique_shock_beta(M, theta_deg, gamma=1.4, tol=1e-8):
    theta = radians(theta_deg)
    # Initial guess (weak shock)
    beta = theta + np.radians(10.0)

    for _ in range(100):
        f = np.tan(theta) - 2*np.tan(beta)*((M**2*np.sin(beta)**2 - 1) /
            (M**2*(gamma + np.cos(2*beta)) + 2))
        # derivative df/dbeta
        dfdB = -2*((M**2*np.sin(beta)**2 - 1)/(M**2*(gamma + np.cos(2*beta)) + 2)) * (1/np.cos(beta)**2) \
                - 2*np.tan(beta)*(
                    (2*M**2*np.sin(beta)*np.cos(beta)*(M**2*(gamma + np.cos(2*beta)) + 2)
                    - (M**2*np.sin(beta)**2 - 1)*(-2*M**2*np.sin(2*beta)))
                    / (M**2*(gamma + np.cos(2*beta)) + 2)**2
                )
        beta_new = beta - f/dfdB
        if abs(beta_new - beta) < tol:
            break
        beta = beta_new
    return degrees(beta)


# ---------------------------------------------------------------------
# Domain plotting
# ---------------------------------------------------------------------

# downstream Mach behind an oblique shock
def downstream_mach(M1, beta_deg, theta_deg, gamma=1.4):
    beta, theta = radians(beta_deg), radians(theta_deg)
    Mn1 = M1 * sin(beta)
    Mn2 = sqrt((1 + 0.5*(gamma-1)*Mn1**2) / (gamma*Mn1**2 - 0.5*(gamma-1)))
    M2 = Mn2 / sin(beta - theta)
    return M2

# ---------------------------------------------------------------------
def plot_multi_ramp(M1=3.0, theta_list=[6.7, 14.2, 20.8], L=100):
    """
    Plot N ramps starting from x=0 with given deflection angles.

    Parameters
    ----------
    M1 : float
        Upstream Mach number (for later shock calculations)
    theta_list : list of floats
        List of ramp angles (deg), sequential turning
    L : float
        Total length of ramp region
    """

    fig, ax = plt.subplots(figsize=(8,4))
    n = len(theta_list)
    ramp_edges = np.linspace(0, L, n+1)
    M_local = M1
    x_prev, y_prev = 0.0, 0.0

    for i, theta in enumerate(theta_list):
        x0, x1 = ramp_edges[i], ramp_edges[i+1]
        x = np.linspace(x0, x1, 200)
        # local slope angle (in radians)
        slope = radians(theta)
        # offset by the previous ramp's end height
        y = y_prev + tan(slope) * (x - x0)
        ax.plot(x, y, lw=2, label=f'Ramp {i+1} ({theta:.1f}°)')
        # update end point for continuity
        x_prev, y_prev = x1, y[-1]


        # compute and plot oblique shock
        beta = oblique_shock_beta(M_local, theta)
        shock_x = np.linspace(x1, x1+150, 200)
        shock_y = y[-1] + tan(radians(beta))*(shock_x - x1)
        ax.plot(shock_x, shock_y, 'r--', lw=1.2,
                label=f'Shock {i+1} β={beta:.1f}°')

        # update Mach for next ramp
        M_local = downstream_mach(M_local, beta, theta)
        x_prev, y_prev = x1, y[-1]

    

    # ------------------------------------------------------------
    # Plot formatting
    # ------------------------------------------------------------
    plt.plot(140.4, 50, '.', lw=1.2, label='Cowl lip')
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(0, L + 200)
    ax.set_ylim(0, y_prev + 80)
    ax.legend()
    ax.set_title(f"Multi-Ramp Compression, M₁={M1:.1f}")
    plt.tight_layout()
    plt.show()


def main():

    plot_multi_ramp()








if __name__ == "__main__":
    main()
