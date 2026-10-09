import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def set_style():
    plt.rcParams.update({
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



