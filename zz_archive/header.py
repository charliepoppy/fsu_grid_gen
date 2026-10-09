# header.py (no sys.path manipulation)
import numpy as np
from numpy import zeros, tan, radians, sinh, tanh, allclose
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from Py_to_HDF5 import arrays_to_HDF5

from utils.compute import (
    build_uniform_x,
    # hermite_quintic_C2,
    gaussian_kernel,
    build_wall_smoothed,
    build_isentropic,
    block_slice,
    check_c0_continuity,
    simple_wall,
    # build_wall_with_isentropic,
    # build_wall_ramp_curve_ramp,
    build_wall_compound,
    build_wall_new,
    build_notched_wall,
    build_upper_wall,
    build_lower_wall,
    linspace_with_point,
    hermite_quintic_polynomial,

    gradient,
    angle_deg,
    length,
    height,
    extract_isentropic_curve,

    spanwise_stretch,
    construct_3D_blocks,




)

from utils.plots import (
    plot_stretching_functions,
    plot_wall_functions,
    plot_stretched_grid,
    plot_total_comp_space,
    plot_dx,
    plot_dy,
    print_summary,
    plot_grid_points,
    print_summary,
    print_summary_3D,
    plot_stretched_grid,
    plot_curves,
    plot_zeta,
    plot_poly_stretch,
    plot_upper_lower,
    plot_blocks_2D,
    plot_bz_x,
    spanwise_stretch_plot,



)


from utils.save_DAT import save_block_to_dat_2D, save_block_to_dat_3D

__all__ = [
    "build_uniform_x",
    # "hermite_quintic_C2",
    "gaussian_kernel",
    "build_wall_smoothed",
    "block_slice",
    "plot_stretching_functions",
    "plot_wall_functions",
    "plot_stretched_grid",
    "plot_total_comp_space",
    "plot_dx",
    "plot_dy",
    "print_summary",
    "np",
    "plt",
    "check_c0_continuity",
    "save_block_to_dat_2D",
    "save_block_to_dat_3D",
    'plot_grid_points',
    'print_summary',
    'print_summary_3D',
    'build_isentropic',
    'simple_wall',
    # 'build_wall_with_isentropic',
    # 'build_wall_ramp_curve_ramp',
    'build_wall_compound',
    'plot_stretched_grid',
    'build_wall_new',
    'build_notched_wall',
    'build_upper_wall',
    'plot_curves',
    'plot_zeta',
    'build_lower_wall',
    'linspace_with_point',
    'hermite_quintic_polynomial',

    'gradient',
    'angle_deg',
    'length',
    'height',

    'plot_poly_stretch',
    'plot_upper_lower',
    'plot_blocks_2D',
    'plot_bz_x',
    'spanwise_stretch_plot',
    'extract_isentropic_curve',

    'spanwise_stretch',
    'construct_3D_blocks',
    'arrays_to_HDF5',
    
    ]
