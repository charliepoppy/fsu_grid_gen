import numpy as np

def intersect(p, a, q, b):
    """Point where line (p, angle a) meets line (q, angle b). Angles in radians."""
    A = np.array([[np.cos(a), -np.cos(b)], [np.sin(a), -np.sin(b)]])
    s = np.linalg.solve(A, np.subtract(q, p))[0]
    return np.asarray(p, float) + s * np.array([np.cos(a), np.sin(a)])


def ang(p, q):
    """Angle of segment p->q in degrees."""
    return np.degrees(np.arctan2(q[1] - p[1], q[0] - p[0]))

def line(ax, p, ang_deg, x1, **kw):
    xs = np.array([p[0], x1])
    ax.plot(xs, p[1] + np.tan(np.radians(ang_deg)) * (xs - p[0]), **kw)

