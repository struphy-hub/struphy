from numpy import empty
import numpy as np
from matplotlib import pyplot as plt

# ==========================================================
def find_span( knots, degree, x ):
    # Knot index at left/right boundary
    low  = degree
    high = 0
    high = len(knots)-1-degree

    # Check if point is exactly on left/right boundary, or outside domain
    if x <= knots[low ]: returnVal = low
    elif x >= knots[high]: returnVal = high-1
    else:
        # Perform binary search
        span = (low+high)//2
        while x < knots[span] or x >= knots[span+1]:
            if x < knots[span]:
                high = span
            else:
                low  = span
            span = (low+high)//2
        returnVal = span

    return returnVal

# ==========================================================
def all_bsplines( knots, degree, x, span ):
    left   = empty( degree  , dtype=float )
    right  = empty( degree  , dtype=float )
    values = empty( degree+1, dtype=float )

    values[0] = 1.0
    for j in range(0,degree):
        left [j] = x - knots[span-j]
        right[j] = knots[span+1+j] - x
        saved    = 0.0
        for r in range(0,j+1):
            temp      = values[r] / (right[r] + left[j-r])
            values[r] = saved + right[r] * temp
            saved     = left[j-r] * temp
        values[j+1] = saved

    return values

# ==========================================================
def point_on_bspline_curve(knots, P, x):
    degree = len(knots) - len(P) - 1
    d = P.shape[-1]

    span = find_span( knots, degree, x )
    b    = all_bsplines( knots, degree, x, span )

    c = np.zeros(d)
    for k in range(0, degree+1):
        c[:] += b[k]*P[span-degree+k,:]
    return c

# ==========================================================
def point_on_bspline_curve_2D(knots1, knots2, P, p1, p2, x, y):
    #d = P.shape[-1]

    span_x = find_span( knots1, p1, x )
    span_y = find_span( knots2, p2, y )
    b_x    = all_bsplines( knots1, p1, x, span_x )
    b_y    = all_bsplines( knots2, p2, y, span_y )

    c = 0
    for kx in range(0, p1+1):
        for ky in range(0, p2+1):
            c += b_x[kx]*b_y[ky]*P[span_x-p1+kx,span_y-p2+ky]
    return c

# ==========================================================
def plot_field_1d(knots, degree, u, nx=101, color='b'):
    n = len(knots) - degree - 1

    xmin = knots[degree]
    xmax = knots[-degree-1]

    xs = np.linspace(xmin, xmax, nx)

    P = np.zeros((len(u), 1))
    P[:,0] = u[:]
    Q = np.zeros((nx, 1))
    for i,x in enumerate(xs):
        Q[i,:] = point_on_bspline_curve(knots, P, x)
    plt.plot(xs, Q[:,0], '-'+color)

