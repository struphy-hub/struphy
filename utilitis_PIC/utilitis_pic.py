from pyccel.decorators import types
from pyccel.decorators import pure
from pyccel.decorators import external_call


#==============================================================================
@external_call
@pure
@types('double[:]','double[:]','double[:]')
def cross(a, b, r):
    r[0] = a[1]*b[2] - a[2]*b[1]
    r[1] = a[2]*b[0] - a[0]*b[2]
    r[2] = a[0]*b[1] - a[1]*b[0]


#==============================================================================
@pure
@types('double[:]','int','double')
def find_span(knots, degree, x):
    """
    Determine the knot span index at location x, given the B-Splines' knot sequence and polynomial degree.

    For a degree p, the knot span index i identifies the indices [i-p:i] of all p+1 non-zero basis functions at a
    given location x.

    Parameters
    ----------
    knots : array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    x : float
        Location of interest.

    Returns
    -------
    span : int
        Knot span index.

    """
    # Knot index at left/right boundary
    low  = degree
    high = 0
    high = len(knots) - 1 - degree

    # Check if point is exactly on left/right boundary, or outside domain
    if x <= knots[low ]: returnVal = low
    elif x >= knots[high]: returnVal = high - 1
    else:
        # Perform binary search
        span = (low + high)//2
        while x < knots[span] or x >= knots[span + 1]:
            if x < knots[span]:
                high = span
            else:
                low  = span
            span = (low + high)//2
        returnVal = span

    return returnVal




#==============================================================================
@types('double[:]','int','double','int','double[:]','double[:]','double[:]')
def basis_funs(knots, degree, x, span, left, right, values):
    """
    Compute the non-vanishing B-splines at location x, given the knot sequence, polynomial degree and knot span.

    Parameters
    ----------
    knots : array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    x : float
        Evaluation point.

    span : int
        Knot span index.

    Results
    -------
    values : numpy.ndarray
        Values of p+1 non-vanishing B-Splines at location x.

    """
    # to avoid degree being intent(inout)
    # TODO improve
    p = degree

#    from numpy      import empty
#    left   = empty( p  , dtype=float )
#    right  = empty( p  , dtype=float )
    left[:] = 0.
    right[:] = 0.

    values[0] = 1.0
    for j in range(0, p):
        left [j] = x - knots[span - j]
        right[j] = knots[span + 1 + j] - x
        saved    = 0.0
        for r in range(0, j + 1):
            temp      = values[r]/(right[r] + left[j - r])
            values[r] = saved + right[r]*temp
            saved     = left[j - r]*temp
        values[j + 1] = saved