import ezdxf
from ezdxf.math import BSpline



HERE = r"G:\work\OpenSBLIEnvnew\Charlie_apps\6_Florida_State_Intake\3_Grid_Gen"
dxf_path = r"\FSU_curve.dxf"
csv_path = r"\spline_samples.csv"

doc = ezdxf.readfile(HERE + dxf_path)
msp = doc.modelspace()

splines = list(msp.query("SPLINE"))
if not splines:
    raise RuntimeError("No SPLINE entities found. Check your DXF export options.")

s = splines[0]  # if multiple, you can filter by layer or handle
print("Degree:", s.dxf.degree)
print("Num control points:", len(s.control_points))
print("Num fit points:", len(s.fit_points))

# Build a math B-spline representation (works for standard DXF splines)
bs = BSpline(
    control_points=list(s.control_points),
    order=s.dxf.degree + 1,
    knots=list(s.knots),
    weights=list(s.weights) if s.weights else None,
)

# Sample N points uniformly in parameter t in [0, 1]
N = 500
pts = [bs.point(t) for t in [i/(N-1) for i in range(N)]]

# Extract x,y (ignore z for 2D sketch)
xy = [(p.x, p.y) for p in pts]

# Save to CSV (simple, no pandas)
with open(HERE + csv_path, "w") as f:
    f.write("x,y\n")
    for x, y in xy:
        f.write(f"{x},{y}\n")

print("Wrote spline_samples.csv with", len(xy), "points")
