

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm

import matplotlib as mpl

def set_style():
    mpl.rcParams.update({
        # figure / saving
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.facecolor": "white",

        # fonts: Computer Modern maths so axis labels match LaTeX in the thesis
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,

        # axes and ticks
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.titlelocation": "left",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,

        # lines / markers / legend
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "legend.frameon": False,
    })



def plot_bz_x(iota, bz_x):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(iota, bz_x)
    ax.set_xlabel("x/Lx"); ax.set_ylabel("bz");
    ax.set_title("Streamwise variation of spanwise stretching")
    return fig



def spanwise_stretch_plot(z, iota,
                           stride_x=6,
                           stride_z=6,
                           split_x=0.6,
                           save=False):

    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=300)

    z_norm = z / z[-1:, :]
    x_plot = np.tile(iota, (z.shape[0], 1))

    # split location index
    split_idx = np.argmin(np.abs(iota - split_x))

    # colours
    c_left  = "tab:blue"
    c_right = "tab:red"

    # -----------------------------
    # Horizontal lines (thicker)
    # -----------------------------
    for j in range(0, z.shape[0], stride_z):

        ax.plot(
            x_plot[j, :split_idx+1],
            z_norm[j, :split_idx+1],
            color=c_left,
            linewidth=0.45,
            alpha=0.8
        )

        ax.plot(
            x_plot[j, split_idx:],
            z_norm[j, split_idx:],
            color=c_right,
            linewidth=0.45,
            alpha=0.8
        )

    # -----------------------------
    # Vertical lines (thinner)
    # -----------------------------
    for i in range(0, split_idx+1, stride_x):

        ax.plot(
            x_plot[:, i],
            z_norm[:, i],
            color=c_left,
            linewidth=0.18,
            alpha=0.45
        )

    for i in range(split_idx, z.shape[1], stride_x):

        ax.plot(
            x_plot[:, i],
            z_norm[:, i],
            color=c_right,
            linewidth=0.18,
            alpha=0.45
        )

    # block interface
    ax.axvline(split_x,
               color='k',
               linewidth=1.0,
               linestyle='--',
               alpha=0.8)

    # boundaries
    boundary_kwargs = dict(color="black", linewidth=0.9)

    ax.plot(x_plot[0, :],  z_norm[0, :],  **boundary_kwargs)
    ax.plot(x_plot[-1, :], z_norm[-1, :], **boundary_kwargs)
    ax.plot(x_plot[:, 0],  z_norm[:, 0],  **boundary_kwargs)
    ax.plot(x_plot[:, -1], z_norm[:, -1], **boundary_kwargs)

    ax.set_xlabel(r"$x/L_x$")
    ax.set_ylabel(r"$z/L_z$")

    ax.set_xlim(iota.min(), iota.max())
    ax.set_ylim(0.0, 1.0)

    # block annotations
    ax.text(
        0.30, 0.96,
        "External periodic flow",
        transform=ax.transAxes,
        ha='center',
        va='top',
        fontsize=11,
        color=c_left,
        bbox=dict(
            facecolor='white',
            edgecolor='none',
            alpha=0.85,
            pad=2.5
        )
    )

    ax.text(
        0.77, 0.96,
        "Sidewall-confined flow",
        transform=ax.transAxes,
        ha='center',
        va='top',
        fontsize=11,
        color=c_right,
        bbox=dict(
            facecolor='white',
            edgecolor='none',
            alpha=0.85,
            pad=2.5
        )
    )

    ax.set_aspect("auto", adjustable="box")

    ax.set_title("Spanwise grid distribution", pad=10)

    ax.tick_params(direction="in",
                   top=True,
                   right=True,
                   width=1.0)

    fig.tight_layout()

    if save:
        fig.savefig("spanwise_grid_distribution.pdf",
                    bbox_inches="tight")
        fig.savefig("spanwise_grid_distribution.png",
                    dpi=600,
                    bbox_inches="tight")

    return fig



def plot_upper_lower(eta, P_upper, P_lower):
    rcparam_func()
    fig, ax = plt.subplots(figsize=(6,4))
    
    by = 5.0
    ref_hyp = np.sinh(by * eta) / np.sinh(by)
    ax.plot(eta, ref_hyp, 'k--', lw=1, label=f'Reference hyperbolic stretch (by={by})')

    ax.plot(eta+1, P_upper+1, lw=2, label='C² spline - upper blocks')
    ax.plot(eta, P_lower, lw=2, label='C² spline  - lower blocks')
    ax.set_xlabel(r'$\eta$'); ax.set_ylabel(r'$P(\eta)$'); ax.legend()
    # ax.set_title('upper/lower block Polynomial stretching functions')
    ax.grid(True, alpha=0.3)
    return fig


def plot_curves(x, y_upper, y_lower, y_top):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x, y_upper, label='Upper wall')
    ax.plot(x, y_lower, label='Lower wall')
    ax.plot(x, y_top, label='Top wall')
    ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$y$'); ax.legend()
    ax.set_title('Upper/lower wall curves')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    return fig

def plot_poly_stretch(eta, P_lower):
    rcparam_func()

    fig, ax = plt.subplots(figsize=(6,4))

    by = 5.0
    ref_hyp = np.sinh(by * eta) / np.sinh(by)
    ax.plot(eta, ref_hyp, 'k--', lw=1, label=f'Reference hyperbolic stretch (by={by})')

    ax.plot(eta, P_lower, lw=2, label='Hermite C² quintic polynomial - lower blocks')

    ax.set_xlabel(r'$\eta$'); ax.set_ylabel(r'$P(\eta)$'); ax.legend()
    ax.set_title('lower block Polynomial stretching function')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    return fig


# plot stretching functions

def plot_stretching_functions(eta, P_lower, P_upper, eta_upper, Ksi_lower, Ksi_upper, Y_lower, Y_upper, x, y_lower, y_upper, y_top):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(eta_upper+1, P_upper+1, lw=2, label='C² spline - upper blocks')
    ax.plot(eta, P_lower, lw=2, label='Hermite C² quintic polynomial - lower blocks')

    by = 5.0
    ref_hyp = np.sinh(by * eta) / np.sinh(by)
    ax.plot(eta, ref_hyp, 'k--', lw=1, label=f'Reference hyperbolic stretch (by={by})')


    ax.set_xlabel(r'$\eta$'); ax.set_ylabel(r'$P(\eta)$'); ax.legend()
    ax.set_title('upper/lower block Polynomial stretching functions')
    ax.grid(True, alpha=0.3)
    return fig


# plot of total comp space
def plot_total_comp_space(Ksi_lower, Ksi_upper, Y_lower, Y_upper, x, y_lower, y_upper, y_top):
    fig, ax = plt.subplots(figsize=(6,4))
    plt.scatter(Ksi_upper, Y_upper+1, s=1, alpha=0.5, marker='.', label='Upper blocks')
    plt.scatter(Ksi_lower, Y_lower, s=1, alpha=0.5, marker='.', label='Lower blocks')
    plt.xlabel(r'$\xi$'); plt.ylabel(r'$\eta$'); plt.legend()
    plt.title('Stretched upper/lower computational domains')
    plt.grid(True, alpha=0.3)
    return fig


# 3. plot the wall functions
def plot_wall_functions(x, *walls, labels=None, colors=None, lw=2, H_t=1):
    fig, ax = plt.subplots(figsize=(7, 4))


    # background_path = r"G:\work\OpenSBLIEnvnew\Charlie_apps\6_Florida_State_Intake\3_Grid_Gen\khobragade.png"
    # background_path = background_path.replace("G:\\", "/mnt/g/").replace("\\", "/")
    # img = plt.imread(background_path)

    # # img = plt.imread(background_path)
    # image_x0 = -18.5    # left boundary of PNG in your coordinate system
    # image_x1 = 220      # right boundary of PNG
    # image_y0 = -19.3    # bottom of PNG
    # image_y1 = 71      
    # ax.imshow(
    #     img,
    #     extent=np.array([image_x0, image_x1, image_y0, image_y1])/H_t,
    #     alpha=0.85
    # )





    ax.set_aspect('equal', adjustable='box')

    n = len(walls)
    if labels is None:
        labels = [f'Wall {i}' for i in range(n)]

    for i, y in enumerate(walls):
        color = None if colors is None else colors[i % len(colors)]
        ax.plot(x/H_t, y/H_t, lw=lw, color=color, label=labels[i])

    plt.axvline(0.0, color='k', ls='--', alpha=0.4, label='Isentropic start (0.0)')
    plt.axvline(156.4, color='k', ls='--', alpha=0.4, label='Isentropic end (156.4)')
    plt.axvline(140.4, color='k', ls='--', alpha=0.4, label='Cowl lip (140.4)')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Wall Functions')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    return fig


def plot_blocks_2D(Blocks, x, nx_lip):

    fig, ax = plt.subplots(figsize=(6,4))


    colours = ['b', 'g', 'r', 'c',]
    for ib, blk in enumerate(Blocks):


        c = colours[ib]


        if ib in [0, 3]:
            xb = x[:nx_lip+1]
        else:
            xb = x[nx_lip:]

        for j in range(blk.shape[0]):
            ax.plot(xb, blk[j, :], c, linewidth=0.5)

        # vertical grid lines
        for i in range(blk.shape[1]):
            ax.plot(xb[i]*np.ones(blk.shape[0]), blk[:, i], c, linewidth=0.5)


    ax.set_xlabel('x'); ax.set_ylabel('y');
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Generated grid")


    return fig


# 4. plot the stretched grid between lower upper walls
def plot_stretched_grid(x_B, y_B,stride=1):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.set_aspect('equal')
    legend_handles = []


    keys = sorted(x_B.keys())  #     ensure consistent ordering
    colours = plt.cm.tab10(np.linspace(0, 1, len(keys)))

    

    for b, c in zip(keys, colours):
        X = x_B[b]
        Y = y_B[b]
        ny, nx = X.shape
        #  vertical
        for i in range(0, nx, stride): 
            ax.plot(X[:, i], Y[:, i], color=c, lw=0.4) 
        
        #  horizontals
        for j in range(0, ny, stride): 
            ax.plot(X[j, :], Y[j, :], color=c, lw=0.4) 

        # one proxy handle per block for the legend
        legend_handles.append(Line2D([0], [0], color=c, lw=1.5, label=f'Block {b}'))

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Florida State multiblock grid')
    ax.grid(True, alpha=0.3)

    # tidy legend (outside, single row if many blocks)
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., frameon=True)

    plt.tight_layout()
    return fig


    
def plot_dx(X_global, Y_global, eps=1e-14):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.set_aspect('equal')

    X = X_global
    Y = Y_global
    ny, nx = X.shape

    dx = np.diff(X, axis=1)  # (ny, nx-1)

    # cell-centered coords for dx (average between columns)
    Xc = 0.5 * (X[:, :-1] + X[:, 1:])
    Yc = 0.5 * (Y[:, :-1] + Y[:, 1:])

    # Clip tiny/negative values to NaN so they don’t show up
    dx_safe = np.where(dx > eps, dx, np.nan)

    cf = ax.contourf(Xc, Yc, np.log10(dx_safe), 50, cmap='viridis')
    fig.colorbar(cf, ax=ax, label='log10(Δx)')


    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.tight_layout()
    return fig


def plot_dy(X_global, Y_global, eps=1e-14):
    """
    Plot log10(Δy) ignoring values below eps (machine precision cutoff).
    """
    fig, ax = plt.subplots(figsize=(6,4))
    ax.set_aspect('equal')

    dy = np.diff(Y_global, axis=0)  # differences in y-direction

    # Cell-centred coords
    Xc = 0.5 * (X_global[:-1, :] + X_global[1:, :])
    Yc = 0.5 * (Y_global[:-1, :] + Y_global[1:, :])

    dy_safe = np.where(dy > eps, dy, np.nan)

    cf = ax.contourf(Xc, Yc, np.log10(dy_safe), 50, cmap='viridis')
    fig.colorbar(cf, ax=ax, label='log10(Δy)')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.tight_layout()
    return fig


def print_summary(nx0, ny0, x, X_global, x_B, y_B, s0, s1, c0, c1):
    print("\n=== Grid Gen Summary ===")
    print(f"Requested nx, ny up to intake lip: {nx0}, {ny0}")

    glob_ny, glob_nx = X_global.shape
    print(f"Global grid shape (nx, ny):        {glob_nx}, {glob_ny}")

    desired_Lx = 551.4
    actual_Lx  = x[-1]
    print(f"Desired Lx vs actual:              {desired_Lx:.4f} vs {actual_Lx:.4f}")

    print("\nLower polynomial parameters:")
    print(f"  s0, s1: {s0}, {s1}")
    print(f"  c0, c1: {c0}, {c1}")

    print("\n=== Grid Block Summary ===")
    for i in range(len(x_B)):
        print(f"\nBlock {i}:")
        x, y = x_B[i], y_B[i]
        ny, nx = x.shape

        print(f"  Grid size (nx, ny):              {nx:4d}, {ny:4d}")
        print(f"  x-range: {x.min():10.4f}, {x.max():10.4f}")
        print(f"  y-range: {y.min():10.4f}, {y.max():10.4f}")

        dxmin, dxmax = np.diff(x, axis=1).min(), np.diff(x, axis=1).max()
        dymin, dymax = np.diff(y, axis=0).min(), np.diff(y, axis=0).max()

        print(f"  Δx min/max: {dxmin:8.4f} / {dxmax:10.4f}")
        print(f"  Δy min/max: {dymin:8.4f} / {dymax:10.4f}")

    print("\n==========================\n")
    return


def plot_grid_points(x_B, y_B, pt_size=2, stride=[1,1]):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.set_aspect('equal')
    for i in range(len(x_B)):
        x, y = x_B[i][::stride[0], ::stride[1]], y_B[i][::stride[0], ::stride[1]]
        ax.scatter(x, y, s=pt_size, marker='o', label=f'Block {i}', alpha=0.6)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Grid points')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig



def plot_stretched_grid(x_B, z_B,stride=1):


    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(6,4))
    stride = 1
    ax.set_aspect('equal')
    legend_handles = []
    keys = [0, 1]
    colours = plt.cm.tab10(np.linspace(0, 1, len(keys)))

    for b, c in zip(keys, colours):
        X = x_B[b][:, 0, :]
        Z = z_B[b][:, 0, :]
        nz,  nx = X.shape
        #  vertical
        for i in range(0, nx, stride): 
            ax.plot(X[:, i], Z[:, i], color=c, lw=0.4) 

        #  horizontals
        for j in range(0, nz, stride): 
            ax.plot(X[j, :], Z[j, :], color=c, lw=0.4) 

        # one proxy handle per block for the legend
        legend_handles.append(Line2D([0], [0], color=c, lw=1.5, label=f'Block {b}'))


    plt.figure(figsize=(12,6))
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_title('Top down grid at y = 0')
    plt.tight_layout()


    return fig






def plot_zeta(z, zeta):
    # plot z-
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.plot(z, zeta, 'k-o', ms=8, )
    ax.set_xlabel("z")
    ax.set_ylabel("zeta")
    ax.set_title("z-stretching")
    # plt.show()
    return fig



# 4. plot the stretched grid between lower upper walls
def plot_stretched_grid(x_B, y_B,stride=1, x_notch=None, y_notch=None,):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.set_aspect('equal')
    legend_handles = []


    keys = sorted(x_B.keys())  #     ensure consistent ordering
    colours = plt.cm.tab10(np.linspace(0, 1, len(keys)))

    for b, c in zip(keys, colours):
        X = x_B[b]
        Y = y_B[b]
        nz, ny, nx = X.shape
        #  vertical
        for i in range(0, nx, stride): 
            ax.plot(X[0, :, i], Y[0, :, i], color=c, lw=0.4) 
        
        #  horizontals
        for j in range(0, ny, stride): 
            ax.plot(X[0, j, :], Y[0, j, :], color=c, lw=0.4) 

        # one proxy handle per block for the legend
        legend_handles.append(Line2D([0], [0], color=c, lw=1.5, label=f'Block {b}'))




    if x_notch is not None:
        plt.plot(x_notch, y_notch, 'k--', lw=1.5, label='Notch')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Floida State multiblock grid')
    ax.grid(True, alpha=0.3)

    # tidy legend (outside, single row if many blocks)
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., frameon=True)

    plt.tight_layout()
    return fig



def print_summary(x_B, y_B):
    print("\n=== Grid Gen Summary ===")
    for i in range(len(x_B)):
        print(f"\nBlock {i}:")
        x, y = x_B[i], y_B[i]
        ny, nx = x.shape

        print(f"  Grid size (nx, ny):              {nx:4d}, {ny:4d}")
        print(f"  x-range: {x.min():10.4f}, {x.max():10.4f}")
        print(f"  y-range: {y.min():10.4f}, {y.max():10.4f}")

        dxmin, dxmax = np.diff(x, axis=1).min(), np.diff(x, axis=1).max()
        dymin, dymax = np.diff(y, axis=0).min(), np.diff(y, axis=0).max()

        print(f"  Δx min/max: {dxmin:8.4f} / {dxmax:10.4f}")
        print(f"  Δy min/max: {dymin:8.4f} / {dymax:10.4f}")

    print("\n==========================\n")
    return




def print_summary_3D(x_B, y_B, z_B):
    print("\n=== Grid Gen Summary ===")

    grid_size = 0
    for i in range(len(x_B)):
        print(f"\nBlock {i}:")
        x, y, z = x_B[i], y_B[i], z_B[i]
        nz, ny, nx = x.shape

        grid_size += nx*ny*nz

        print(f"  Grid size (nx, ny, nz):              {nx:4d}, {ny:4d}, {nz:4d}")
        print(f"  x-range:  {x.min():11.4f}, {x.max():10.4f}")
        print(f"  y-range:  {y.min():11.4f}, {y.max():10.4f}")
        print(f"  z-range:  {z.min():11.4f}, {z.max():10.4f}")

        dxmin, dxmax = np.diff(x, axis=2).min(), np.diff(x, axis=2).max()
        dymin, dymax = np.diff(y, axis=1).min(), np.diff(y, axis=1).max()
        dzmin, dzmax = np.diff(z, axis=0).min(), np.diff(z, axis=0).max()

        print(f"  Δx min/max: {dxmin:8.4f} / {dxmax:5.4f}")
        print(f"  Δy min/max: {dymin:8.4f} / {dymax:5.4f}")
        print(f"  Δz min/max: {dzmin:8.4f} / {dzmax:5.4f}")


    print(f"\nTotal grid size: {grid_size:,}")
    print("\n==========================\n")
    return




