import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares

from scipy.optimize import minimize_scalar

from matplotlib.offsetbox import AnnotationBbox, OffsetImage

"""
Compare two reconstruction methods for the intake curve:
1. Constrained polynomial fit
2. Smooth θ(s)-based isentropic-style curve

points are digitised from image.
we know for certain:
y0 = 0
θ0 = 6.7 degrees
θ1 = 20.8 degrees
L we'll assume is 155

isolator length 90
cowl and facet angle 9 degs
cowl lip (140.4, 50)

"""


# ============================================================
#  Digitised data
L = 155
theta0 = np.radians(6.7)
theta1 = np.radians(20.8)
intake_LE_x = -75.7



# ----------------------------------------------------------------------
# Improved θ(s) with 3 controllable modes
# ----------------------------------------------------------------------


# smooth endpoint basis functions (zero & zero-derivative at s=0,1)
def C1(s): return s**2 * (1 - s)**2
def C2(s): return s**3 * (1 - s)**2
def C3(s): return s**2 * (1 - s)**3

def theta_full(s, params):
    a, b, c = params
    H = 3*s**2 - 2*s**3
    return theta0 + (theta1 - theta0)*H + a*C1(s) + b*C2(s) + c*C3(s)

def generate_curve_params(params, N=2000):
    x = np.linspace(0, L, N)
    s = x / L
    dydx = np.tan(theta_full(s, params))
    y = cumulative_trapezoid(dydx, x, initial=0.0)
    return x, y



a_opt, b_opt, c_opt = -0.4869003331181572, 1.4903021210056506, -1.8097491346735368
x_iso, y_iso = generate_curve_params((a_opt, b_opt, c_opt))
curve_end = (x_iso[-1], y_iso[-1])




# -----------------------------------------------------------------------
# Improved θ(s) with 3 controllable modes
# ----------------------------------------------------------------------

m1 = np.tan(np.radians(9.0))
m2 = np.tan(np.radians(10.6))
m3 = np.tan(np.radians(20.8))


# cowl end
cowl_lip = (140.4, 50.0)
cowl_end_x = 179.5
cowl_end_y = cowl_lip[1] + m1 * (cowl_end_x - cowl_lip[0])
cowl_end = (cowl_end_x, cowl_end_y)

 
# facet
ramp_1_end = 183
ramp_1_L = ramp_1_end - x_iso[-1]
ramp_1_x = curve_end[0] + ramp_1_L
ramp_1_y = curve_end[1] + m2 * (ramp_1_x - curve_end[0])
ramp_1 = (ramp_1_x, ramp_1_y)


# lower isolator
ramp_2_x = ramp_1_x + 90
ramp_2_y = ramp_1_y
ramp_2 = (ramp_2_x, ramp_2_y)


# upper isolator
upper_wall_x = cowl_end_x + 90
upper_wall_y = cowl_end_y
upper_wall = (upper_wall_x, upper_wall_y)



background_path = r"G:\work\OpenSBLIEnvnew\Charlie_apps\6_Florida_State_Intake\3_Grid_Gen\florida_notch.png"
img = plt.imread(background_path)

fig, ax = plt.subplots(figsize=(12, 6))

image_x0 = 135    # left boundary of PNG in your coordinate system
image_x1 = 220      # right boundary of PNG
image_y0 = 30    # bottom of PNG
image_y1 = 68    

ax.imshow(
    img,
    extent=[image_x0, image_x1, image_y0, image_y1],
    alpha=0.85
)



# plt.figure(figsize=(12,6))
plt.plot([cowl_lip[0], cowl_end[0]],
         [cowl_lip[1], cowl_end[1]], 'r-x', ms=8, label="Cowl")




plt.plot([curve_end[0], ramp_1[0]],
         [curve_end[1], ramp_1[1]], 'k-x', ms=8, label="Facet")

plt.plot(x_iso, y_iso, label=f"Isentropic θ(s) (eps={1:.3f})", lw=2)

plt.plot([ramp_2[0], ramp_1[0]], 
         [ramp_2[1], ramp_1[1]], 'k-o', ms=8, label="Isolator")


plt.plot([upper_wall[0], cowl_end[0]],
         [upper_wall[1], cowl_end[1]], 'k-o', ms=8, label="Upper wall")


# is ramp
l1_x = np.linspace(100, 200, 100)
l1_y = np.tan(np.radians(20.8)) * (l1_x -115) + 38.4
plt.plot(l1_x, l1_y, 'k--', lw=1.5, label='L1')



l2_x = np.linspace(100, 200, 100)
l2_y = np.tan(np.radians(9)) * (l2_x -183) + 40

plt.plot(l2_x, l2_y, 'k--', lw=1.5, label='L2')




plt.xlabel("$x$")
plt.ylabel("$y$")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.tight_layout()
plt.show()



















