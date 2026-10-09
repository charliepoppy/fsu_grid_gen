from numpy import sqrt, arctan, radians, degrees
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

def build_isentropic(x_start=0.0, x_end=150, y_start=0, thetas=[6.7, 20.8], M1=2.83, gama=1.4):
    theta = np.radians(np.linspace(thetas[0], thetas[1], 200))
    nu1 = prandtl_meyer(M1, gama)
    nu  = nu1 - (theta - theta[0])
    M   = np.array([inverse_prandtl_meyer(nu_i, gama) for nu_i in nu])

    x = np.linspace(x_start, x_end, len(theta))
    y = np.zeros_like(x)
    for i in range(1, len(x)):
        dx = x[i] - x[i-1]
        dy = 0.5*(np.tan(theta[i]) + np.tan(theta[i-1])) * dx
        y[i] = y[i-1] + dy
    y += y_start

    return x, y, theta, M


def plot_isentropic_with_waves(x, y, theta, M, n_lines=40, cmap_name='viridis'):
    """
    Plot isentropic wall and compression-wave family coloured by local Mach.
    """
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(x, y, 'k', lw=2, label='Isentropic wall')

    # Modern colormap API
    cmap = plt.colormaps.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=np.min(M), vmax=np.max(M))

    idx = np.linspace(0, len(x)-1, n_lines, dtype=int)

    for i in idx:
        xi, yi = x[i], y[i]
        mu = np.arcsin(1/M[i])
        wave_angle = theta[i] + mu
        xw = np.linspace(xi, x[-1]+100, 200)
        yw = yi + np.tan(wave_angle)*(xw - xi)
        ax.plot(xw, yw, color=cmap(norm(M[i])), lw=0.8)

    # Plot cowl lip or target point
    ax.plot(140.4, 50.0, 'ko', ms=4, label='Cowl lip')
    # plot line -75 to 0 on x axis
    ax.plot([0, -75.7], [0, 0], 'k--', lw=1.5, label='Flatplate')

    # Plot cowl gradient passing through cowl lip
    cowl_x = 140.4
    cowl_y = 50
    cowl_grad = np.tan(np.radians(9))
    x = np.linspace(cowl_x, 180.0, 200)
    y = cowl_grad*(x - cowl_x) + cowl_y
    ax.plot(x, y, 'k--', lw=1.5)

    # Lower wall (17.43 orthogonally below)
    offset = 17.43
    angle = np.radians(9)
    x_lower0 = cowl_x - offset*np.sin(angle)
    y_lower0 = cowl_y - offset*np.cos(angle)
    y_lower = cowl_grad*(x - x_lower0) + y_lower0
    ax.plot(x, y_lower, 'b--', lw=1.5, label='Lower wall (17.43 ⟂ offset)')



    # plot initial oblique shock
    shock_grad = np.tan(np.radians(25))
    x = np.linspace(0, 200, 200)
    y = shock_grad*(x - 0) + 0
    ax.plot(x, y, 'r--', lw=1.5, label='Initial oblique shock')


    # Properly attach colourbar to this Axes
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # required for colorbar creation
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Local Mach number')

    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f'Isentropic ramp and Mach-wave family for flow turning from {degrees(theta[0]):.2f}° to {degrees(theta[-1]):.2f}°')
    fig.tight_layout()
    plt.show()


def main():
    x, y, theta, M = build_isentropic()
    plot_isentropic_with_waves(x, y, theta, M)

if __name__ == "__main__":
    main()
