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
# ============================================================
data = np.array([
    [0.000000000,    0.000000000],
    [8.78238806,     1.205189373],
    [17.14656716,    2.388823221],
    [39.03283369,    5.058666306],
    [53.67014713,    6.463855679],
    [60.30573007,    7.196900445],
    [68.58626228,    8.112156103],
    [77.64746481,   9.31733261],
    [88.24208317,   12.06563304],
    [100.3631712,   14.34741597],
    [112.6376098,   18.96999865],
    [129.2265608,   25.29333796],
    [136.0921631,   27.88590509],
    [142.0516312,   29.93044874],
    [148.3596173,   32.36844465],
    [155.0000000,   35.00000000],
])
x_data, y_data = data[:,0], data[:,1]


L = x_data.max()
theta0 = np.radians(6.7)
theta1 = np.radians(20.8)
intake_LE_x = -75.7




# ----------------------------------------------------------------------
# Improved θ(s) with 3 controllable modes
# ----------------------------------------------------------------------

theta0 = np.radians(6.7)
theta1 = np.radians(20.8)

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

def residuals(params):
    x, y = generate_curve_params(params)
    y_i = np.interp(x_data, x, y)
    return y_i - y_data

# initial guess
params0 = np.array([0.0, 0.0, 0.0])

# bounds — limit crazy curvature
bounds = ([-3, -3, -3], [3, 3, 3])

res = least_squares(residuals, params0, bounds=bounds)
a_opt, b_opt, c_opt = res.x

print("Optimal parameters:", a_opt, b_opt, c_opt)
print("RMS error:", np.sqrt(np.mean(res.fun**2)))

# final curve
x_iso, y_iso = generate_curve_params((a_opt, b_opt, c_opt))
curve_end = (x_iso[-1], y_iso[-1])





# pull in csv and plot actual isentropic


HERE = r"G:\work\OpenSBLIEnvnew\Charlie_apps\6_Florida_State_Intake\3_Grid_Gen"
csv_path = r"\isentropic_curve.csv"


# read the csv file
data = np.genfromtxt(HERE + csv_path, delimiter=',')
joined_u = data

x_true = joined_u[:, 0]
y_true = joined_u[:, 4]












# -----------------------------------------------------------------------
# Improved θ(s) with 3 controllable modes
# ----------------------------------------------------------------------

m1 = np.tan(np.radians(9.0))


# cowl end
cowl_lip = (140.4, 50.0)
cowl_end_x = 179.5
cowl_end_y = cowl_lip[1] + m1 * (cowl_end_x - cowl_lip[0])
cowl_end = (cowl_end_x, cowl_end_y)


ramp_1_end = 183
ramp_1_L = ramp_1_end - x_iso[-1]
ramp_1_x = curve_end[0] + ramp_1_L
ramp_1_y = curve_end[1] + m1 * (ramp_1_x - curve_end[0])
ramp_1 = (ramp_1_x, ramp_1_y)


# ramp 2
ramp_2_x = ramp_1_x + 90
ramp_2_y = ramp_1_y
ramp_2 = (ramp_2_x, ramp_2_y)


# upper wall
upper_wall_x = cowl_end_x + 90
upper_wall_y = cowl_end_y
upper_wall = (upper_wall_x, upper_wall_y)



# ============================================================
#  Plot everything
# ============================================================

background_path = r"G:\work\OpenSBLIEnvnew\Charlie_apps\6_Florida_State_Intake\3_Grid_Gen\khobragade.png"
img = plt.imread(background_path)

fig, ax = plt.subplots(figsize=(12, 6))


if 0:
    
    image_x0 = -18.5    # left boundary of PNG in your coordinate system
    image_x1 = 220      # right boundary of PNG
    image_y0 = -19.3    # bottom of PNG
    image_y1 = 71
    ax.imshow(
        img,
        extent=[image_x0, image_x1, image_y0, image_y1],
        alpha=0.85
    )



# plt.figure(figsize=(12,6))
# plt.scatter(x_data, y_data, label="Reverse engineered isentropic curve", marker='.', s=1, alpha=0.5, color='k')
plt.plot(x_iso, y_iso, label=f"Reverse engineered isentrope: θ(s) (eps={1:.3f})", lw=2, color='k')
plt.plot(x_true, y_true, 'b', lw=2, label="Actual CAD isentropic curve")

# plt.plot([intake_LE_x, 0], [0, 0], 'k-o', ms=8, label="Flatplate")
plt.plot([cowl_lip[0], cowl_end[0]],
         [cowl_lip[1], cowl_end[1]], 'r-x', ms=8, label="Cowl")
plt.plot([curve_end[0], ramp_1[0]],
         [curve_end[1], ramp_1[1]], 'k-x', ms=8, label="Facet")



plt.plot([ramp_2[0], ramp_1[0]], 
         [ramp_2[1], ramp_1[1]], 'k-o', ms=8, label="Isolator")


plt.plot([upper_wall[0], cowl_end[0]],
         [upper_wall[1], cowl_end[1]], 'k-o', ms=8, label="Upper wall")


         


plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.tight_layout()
plt.show()
