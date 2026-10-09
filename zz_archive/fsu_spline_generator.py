import ezdxf
from ezdxf.math import BSpline
import matplotlib.pyplot as plt
import numpy as np


HERE = r"G:\work\OpenSBLIEnvnew\Charlie_apps\6_Florida_State_Intake\3_Grid_Gen"
csv_path = r"\isentropic_curve.csv"


# read the csv file
data = np.genfromtxt(HERE + csv_path, delimiter=',')
joined_u = data


x = joined_u[:, 0]
y = joined_u[:, 4]





cowl = (140.4, 50.0)
curve_end = (115.3 , 19.3)



# --- Target angles (deg) ---
theta_0 = 6.7
theta_1 = 20.8
m0 = np.tan(np.radians(theta_0))
m1 = np.tan(np.radians(theta_1))

# --- Line 0: through (0,0) with slope m0, spanning X-range you care about ---
L = 20.0
x0 = np.linspace(0.0, L, 200)
y0 = m0 * x0

# --- Line 1: anchored at the START of the spline in the shifted frame ---
x_s0, y_s0 = x[0], y[0]   # start point after shift/flip
x1 = np.linspace(x_s0 + L/2 + 40, x_s0 - L/2 , 200)  # go "left" from start point
y1 = y_s0 + m1 * (x1 - x_s0)

plt.figure()
plt.title("FSU CAD Curve (shifted to end at origin)")
plt.plot(x, y, label="CAD spline (sampled)")



plt.plot(cowl[0], cowl[1], "ro", label="Cowl lip")
plt.plot(curve_end[0], curve_end[1], "ro", label="Curve end")


plt.axis("equal")
plt.legend()
plt.grid()
plt.show()