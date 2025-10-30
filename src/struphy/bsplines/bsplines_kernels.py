# coding: utf-8

"""
Basic functions for point-wise B-spline evaluation
"""

from typing import Final

from numpy import empty, zeros
from pyccel.decorators import pure, stack_array


@pure
def scaling(t_d: "Final[float[:]]", p_d: "int", span_d: "int", values: "float[:]"):
    """
    Scaling coefficients for M-splines.

    Parameters
    ----------
    knots : array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    span : int
        Knot span index.

    values : array[float]
        Output: scaling vector with elements (p + 1)/(t[i + p + 1] - t[i])
    """

    for il in range(p_d + 1):
        i = span_d - il
        values[p_d - il] *= (p_d + 1) / (t_d[i + p_d + 1] - t_d[i])


@pure
def find_span(t: "Final[float[:]]", p: "int", eta: "float") -> "int":
    """
    Computes the knot span index i for which the B-splines i-p until i are non-vanishing at point eta.

    Parameters:
    -----------
        t : array
            knot sequence

        p : integer
            degree of the basis splines

        eta : float
            Evaluation point

    Returns:
    --------
        span-index
    """

    # Knot index at left/right boundary
    low = p
    high = len(t) - 1 - p

    # Check if point is exactly on left/right boundary, or outside domain
    if eta <= t[low]:
        returnVal = low
    elif eta >= t[high]:
        returnVal = high - 1
    else:
        # Perform binary search
        span = (low + high) // 2

        while eta < t[span] or eta >= t[span + 1]:
            if eta < t[span]:
                high = span
            else:
                low = span
            span = (low + high) // 2

        returnVal = span

    return returnVal


@pure
def basis_funs(
    t: "Final[float[:]]",
    p: "int",
    eta: "float",
    span: "int",
    left: "float[:]",
    right: "float[:]",
    values: "float[:]",
):
    """
    Parameters
    ----------
    t : array[float]
        Knots sequence.

    p : int
        Polynomial degree of B-splines.

    eta : float
        Evaluation point.

    span : int
        Knot span index.

    left : array[float]
        Internal array of size p.

    right : array[float]
        Internal array of size p.

    values : array[float]
        Output: values of p + 1 non-vanishing B-Splines at location eta.
    """

    left[:] = 0.0
    right[:] = 0.0

    values[0] = 1.0

    for j in range(p):
        left[j] = eta - t[span - j]
        right[j] = t[span + 1 + j] - eta
        saved = 0.0
        for r in range(j + 1):
            temp = values[r] / (right[r] + left[j - r])
            values[r] = saved + right[r] * temp
            saved = left[j - r] * temp
        values[j + 1] = saved


@pure
def basis_funs_all(
    t: "float[:]",
    p: "int",
    eta: "float",
    span: "int",
    left: "float[:]",
    right: "float[:]",
    values: "float[:,:]",
    diff: "float[:]",
):
    """
    Parameters
    ----------
    t : array[float]
        Knots sequence.

    p : int
        Polynomial degree of B-splines.

    eta : float
        Evaluation point.

    span : int
        Knot span index.

    left : array[float]
        Internal array of size p.

    right : array[float]
        Internal array of size p.

    values : array[float]
        Outout: values of (p + 1, p + 1) non-vanishing B-Splines at location eta.

    diff : array[float]
        Output: scaling array (p) for M-splines.
    """

    left[:] = 0.0
    right[:] = 0.0

    values[:, :] = 0.0
    values[0, 0] = 1.0

    for j in range(p):
        left[j] = eta - t[span - j]
        right[j] = t[span + 1 + j] - eta
        saved = 0.0
        for r in range(j + 1):
            diff[r] = 1.0 / (right[r] + left[j - r])
            temp = values[j, r] * diff[r]
            values[j + 1, r] = saved + right[r] * temp
            saved = left[j - r] * temp
        values[j + 1, j + 1] = saved

    diff[:] = diff * p


@pure
def basis_funs_all_ders(
    knots: "Final[float[:]]",
    degree: int,
    eta: float,
    span: int,
    left: "float[:]",
    right: "float[:]",
    n: int,
    ders: "float[:, :]",
):
    """
    Copied from gvec_to_python, where it is tested.

    Evaluate value and n derivatives at eta of all basis functions with
    support in interval [x_{span-1}, x_{span}].

    ders[i,j] = (d/deta)^i B_k(eta) with k=(span-degree+j),
                for 0 <= i <= n and 0 <= j <= degree+1.

    Parameters
    ----------
    knots : array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    eta : float
        Evaluation point.

    span : int
        Knot span index.

    n : int
        Max derivative of interest.

    Results
    -------
    ders : numpy.ndarray (n+1,degree+1)
        2D array of n+1 (from 0-th to n-th) derivatives at eta of all (degree+1)
        non-vanishing basis functions in given span.

    Notes
    -----
    The original Algorithm A2.3 in The NURBS Book [1] is here improved:
        - 'left' and 'right' arrays are 1 element shorter;
        - inverse of knot differences are saved to avoid unnecessary divisions;
        - innermost loops are replaced with vector operations on slices.
    """
    # left = empty(degree)
    # right = empty(degree)
    ndu = empty((degree + 1, degree + 1))
    a = empty((2, degree + 1))
    # ders = zeros((n+1, degree+1))  # output array

    # Number of derivatives that need to be effectively computed
    # Derivatives higher than degree are = 0.
    ne = min(n, degree)

    # Compute nonzero basis functions and knot differences for splines
    # up to degree, which are needed to compute derivatives.
    # Store values in 2D temporary array 'ndu' (square matrix).
    ndu[0, 0] = 1.0
    for j in range(0, degree):
        left[j] = eta - knots[span - j]
        right[j] = knots[span + 1 + j] - eta
        saved = 0.0
        for r in range(0, j + 1):
            # compute inverse of knot differences and save them into lower triangular part of ndu
            ndu[j + 1, r] = 1.0 / (right[r] + left[j - r])
            # compute basis functions and save them into upper triangular part of ndu
            temp = ndu[r, j] * ndu[j + 1, r]
            ndu[r, j + 1] = saved + right[r] * temp
            saved = left[j - r] * temp
        ndu[j + 1, j + 1] = saved

    # Compute derivatives in 2D output array 'ders'
    ders[0, :] = ndu[:, degree]
    for r in range(0, degree + 1):
        s1 = 0
        s2 = 1
        a[0, 0] = 1.0
        for k in range(1, ne + 1):
            d = 0.0
            rk = r - k
            pk = degree - k
            if r >= k:
                a[s2, 0] = a[s1, 0] * ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
            j1 = 1 if (rk > -1) else -rk
            j2 = k - 1 if (r - 1 <= pk) else degree - r
            a[s2, j1 : j2 + 1] = (a[s1, j1 : j2 + 1] - a[s1, j1 - 1 : j2]) * ndu[pk + 1, rk + j1 : rk + j2 + 1]
            for l in range(j2 + 1 - j1):
                d += a[s2, j1 + l] * ndu[rk + j1 + l, pk]
            # d += dot(a[s2, j1:j2+1], ndu[rk+j1:rk+j2+1, pk])
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] * ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]
            ders[k, r] = d
            j = s1
            s1 = s2
            s2 = j

    # Multiply derivatives by correct factors
    r = degree
    for k in range(1, ne + 1):
        ders[k, :] = ders[k, :] * r
        r = r * (degree - k)


@pure
@stack_array("left", "right")
def b_splines_slim(tn: "float[:]", pn: "int", eta: "float", span: "int", values: "float[:]"):
    """
    Computes the values of pn + 1 non-vanishing B-splines at position eta.

    Parameters
    ----------
    tn : array
        Knots sequence.

    pn : int
        Polynomial degree of B-splines.

    eta : float
        Evaluation point.

    span : int
        Knot span index.

    values : array[float]
        Outout: values of pn + 1 non-vanishing B-Splines at location eta.
    """

    # Initialize variables left and right used for computing the values
    left = empty(pn, dtype=float)
    right = empty(pn, dtype=float)
    left[:] = 0.0
    right[:] = 0.0

    values[0] = 1.0

    for j in range(pn):
        left[j] = eta - tn[span - j]
        right[j] = tn[span + 1 + j] - eta
        saved = 0.0
        for r in range(j + 1):
            temp = values[r] / (right[r] + left[j - r])
            values[r] = saved + right[r] * temp
            saved = left[j - r] * temp
        values[j + 1] = saved


@pure
@stack_array("left", "right", "b_values")
def d_splines_slim(tn: "float[:]", pn: "int", eta: "float", span: "int", values: "float[:]"):
    """
    Computes the values of pn non-vanishing D-splines at position eta.

    Parameters
    ----------
    tn : array
        Knot sequence of B-splines.

    pn : int
        Polynomial degree of B-splines.

    eta : float
        Evaluation point.

    span : int
        Knot span index.

    Returns
    -------
    values : array
        Values of p non-vanishing D-Splines at location eta.
    """

    # compute D-spline degree
    pd = pn - 1

    # make sure the arrays we are writing to are empty
    values[:] = 0.0

    # Initialize variables left and right used for computing the B-splines up to degree p-1
    left = empty(pn - 1, dtype=float)
    right = empty(pn - 1, dtype=float)
    left[:] = 0.0
    right[:] = 0.0

    # Compute B-splines up to degree p-1
    b_values = empty(pn, dtype=float)
    b_values[:] = 0.0

    b_values[0] = 1.0

    for j in range(pd):
        left[j] = eta - tn[span - j]
        right[j] = tn[span + 1 + j] - eta
        saved = 0.0
        for r in range(j + 1):
            temp = b_values[r] / (right[r] + left[j - r])
            b_values[r] = saved + right[r] * temp
            saved = left[j - r] * temp
        b_values[j + 1] = saved

    # compute D-splines values by scaling
    for il in range(pd + 1):
        values[pd - il] = pn / (tn[span - il + pn] - tn[span - il]) * b_values[pd - il]


@pure
@stack_array("left", "right")
def b_d_splines_slim(tn: "float[:]", pn: "int", eta: "float", span: "int", bn: "float[:]", bd: "float[:]"):
    """
    One function to compute the values of non-vanishing B-splines and D-splines.

    Parameters
    ----------
        tn : array
            Knot sequence of B-splines.

        pn : int
            Polynomial degree of B-splines.

        span : integer
            Knot span index i -> [i-p,i] basis functions are non-vanishing.

        eta : float
            Evaluation point.

        bn : array[float]
            Output: pn + 1 non-vanishing B-splines at eta

        bd : array[float]
            Output: pn non-vanishing D-splines at eta
    """

    # compute D-spline degree
    pd = pn - 1

    # make sure the arrays we are writing to are empty
    bn[:] = 0.0
    bd[:] = 0.0

    # Initialize variables left and right used for computing the value
    left = zeros(pn, dtype=float)
    right = zeros(pn, dtype=float)

    bn[0] = 1.0

    for j in range(pn):
        left[j] = eta - tn[span - j]
        right[j] = tn[span + 1 + j] - eta
        saved = 0.0

        if j == pn - 1:
            # compute D-splines values by scaling B-splines of degree pn-1
            for il in range(pd + 1):
                bd[pd - il] = pn / (tn[span - il + pn] - tn[span - il]) * bn[pd - il]

        for r in range(j + 1):
            temp = bn[r] / (right[r] + left[j - r])
            bn[r] = saved + right[r] * temp
            saved = left[j - r] * temp

        bn[j + 1] = saved


@pure
@stack_array("left", "right", "diff")
def b_der_splines_slim(tn: "float[:]", pn: "int", eta: "float", span: "int", bn: "float[:]", der: "float[:]"):
    """
    Parameters
    ----------
    tn : array_like
        Knots sequence of B-splines.

    pn : int
        Polynomial degree of B-splines.

    eta : float
        Evaluation point.

    span : int
        Knot span index.

    bn : array[float]
        Out: pn + 1 values of B-splines at eta.

    der : array[float]
        Out: pn + 1 values of derivatives of B-splines at eta.
    """

    left = zeros(pn, dtype=float)
    right = zeros(pn, dtype=float)
    diff = zeros(pn, dtype=float)

    values = zeros((pn + 1, pn + 1), dtype=float)
    values[0, 0] = 1.0

    for j in range(pn):
        left[j] = eta - tn[span - j]
        right[j] = tn[span + 1 + j] - eta
        saved = 0.0
        for r in range(j + 1):
            diff[r] = 1.0 / (right[r] + left[j - r])
            temp = values[j, r] * diff[r]
            values[j + 1, r] = saved + right[r] * temp
            saved = left[j - r] * temp
        values[j + 1, j + 1] = saved

    diff[:] = diff * pn

    # compute derivatives
    # j = 0
    saved = values[pn - 1, 0] * diff[0]
    der[0] = -saved

    # j = 1, ... , pn
    for j in range(1, pn):
        temp = saved
        saved = values[pn - 1, j] * diff[j]
        der[j] = temp - saved

    # j = pn
    bn[:] = values[pn, :]
    der[pn] = saved


@pure
def basis_funs_and_der(
    t: "float[:]",
    p: "int",
    eta: "float",
    span: "int",
    left: "float[:]",
    right: "float[:]",
    values: "float[:,:]",
    diff: "float[:]",
    der: "float[:]",
):
    """
    Parameters
    ----------
    t : array_like
        Knots sequence.

    p : int
        Polynomial degree of B-splines.

    eta : float
        Evaluation point.

    span : int
        Knot span index.

    left : array_like
        p left values

    right : array_like
        p right values

    values_all : array_like

    Returns
    -------
    values : numpy.ndarray
        Values of (2, p + 1) non-vanishing B-Splines and derivatives at location eta.
    """

    left[:] = 0.0
    right[:] = 0.0

    values[:, :] = 0.0
    values[0, 0] = 1.0

    for j in range(p):
        left[j] = eta - t[span - j]
        right[j] = t[span + 1 + j] - eta
        saved = 0.0
        for r in range(j + 1):
            diff[r] = 1.0 / (right[r] + left[j - r])
            temp = values[j, r] * diff[r]
            values[j + 1, r] = saved + right[r] * temp
            saved = left[j - r] * temp
        values[j + 1, j + 1] = saved

    diff[:] = diff * p

    # compute derivatives
    # j = 0
    saved = values[p - 1, 0] * diff[0]
    der[0] = -saved

    # j = 1, ... , p
    for j in range(1, p):
        temp = saved
        saved = values[p - 1, j] * diff[j]
        der[j] = temp - saved

    # j = p
    der[p] = saved


@pure
@stack_array("values_b")
def basis_funs_1st_der(
    t: "Final[float[:]]",
    p: "int",
    eta: "float",
    span: "int",
    left: "float[:]",
    right: "float[:]",
    values: "float[:]",
):
    """
    Parameters
    ----------
    t : array_like
        Knots sequence.

    p : int
        Polynomial degree of B-splines.

    eta : float
        Evaluation point.

    span : int
        Knot span index.

    Returns
    -------
    values : numpy.ndarray
        Derivatives of p + 1 non-vanishing B-Splines at location eta.
    """

    # Compute nonzero basis functions and knot differences for splines up to degree p - 1
    values_b = empty(p + 1, dtype=float)
    basis_funs(t, p - 1, eta, span, left, right, values_b)

    # Compute derivatives at x using formula based on difference of splines of degree p - 1
    # -------
    # j = 0
    saved = p * values_b[0] / (t[span + 1] - t[span + 1 - p])
    values[0] = -saved

    # j = 1, ... , p - 1
    for j in range(1, p):
        temp = saved
        saved = p * values_b[j] / (t[span + j + 1] - t[span + j + 1 - p])
        values[j] = temp - saved

    # j = degree
    values[p] = saved


@pure
@stack_array("values_b")
def b_spl_1st_der_slim(t: "float[:]", p: "int", eta: "float", span: "int", values: "float[:]"):
    """
    Parameters
    ----------
    t : array_like
        Knots sequence.

    p : int
        Polynomial degree of B-splines.

    eta : float
        Evaluation point.

    span : int
        Knot span index.

    Returns
    -------
    values : array
        Derivatives of p + 1 non-vanishing B-Splines at location eta.
    """

    # Compute nonzero basis functions and knot differences for splines up to degree p - 1
    values_b = empty(p + 1, dtype=float)
    b_splines_slim(t, p - 1, eta, span, values_b)

    # Compute derivatives at x using formula based on difference of splines of degree p - 1
    # -------
    # j = 0
    saved = p * values_b[0] / (t[span + 1] - t[span + 1 - p])
    values[0] = -saved

    # j = 1, ... , p - 1
    for j in range(1, p):
        temp = saved
        saved = p * values_b[j] / (t[span + j + 1] - t[span + j + 1 - p])
        values[j] = temp - saved

    # j = degree
    values[p] = saved


@pure
def piecewise(p: "int", delta: "float", eta: "float") -> "float":
    r"""
    evaluate a hat function (B-spline) centered at eta0 (the center of the support) at eta1, i.e.

    .. math::
        1.0 / delta * S((eta1 - eta0)/ delta)

    where S is the B-spline of degree p, delta is the cell size of a uniform mesh.
    Here we use the expression of the B-spline in each cell directly. For the moment, p = 0, 1, or 2.

    Parameters
    ----------
    p : int
        degree of the hat function (B-spline)

    delta : float
        the cell size of a uniform mesh.

    eta : float
        eta = eta1 - eta0

    Returns:
    --------
        value of the hat function
    """
    if abs(eta) > delta * (p + 1) * 0.5:
        return 0.0
    else:
        if p == 0:
            return 1.0
        elif p == 1:
            return 1 - abs(eta / delta)
        elif p == 2:
            if eta >= 0.5 * delta:
                temp = eta / delta + 1.5
                return 0.5 * (3 - temp) ** 2.0
            elif eta < -0.5 * delta:
                temp = eta / delta + 1.5
                return 0.5 * temp**2.0
            else:
                temp = eta / delta + 1.5
                return 0.5 * (-3.0 + 6.0 * temp - 2 * temp**2.0)
        else:
            return -1.0


@pure
def piecewise_der(p: "int", delta: "float", eta: "float") -> "float":
    r"""
    evaluate the derivative of a hat function (B-spline) centered at eta0 (the center of the support) at eta1, i.e.
    .. math::
        1.0 / delta^2 * S'((eta1 - eta0)/ delta)

    where S is the B-spline of degree p, delta is the cell size of a uniform mesh.
    Here we use the expression of the derivative of the B-spline in each cell directly. For the moment, p = 0, 1, or 2.

    Parameters
    ----------
    p : int
        degree of the hat function (B-spline)

    delta : float
        the cell size of a uniform mesh.

    eta : float
        eta = eta1 - eta0

    Returns:
    --------
        value of the derivative of the hat function
    """
    if abs(eta) > delta * (p + 1) * 0.5:
        return 0.0
    else:
        if p == 0:
            return 0.0
        elif p == 1:
            if eta < 0:
                return 1 / delta
            else:
                return -1 / delta
        elif p == 2:
            if eta >= 0.5 * delta:
                temp = eta / delta + 1.5
                return -(3 - temp) / delta
            elif eta < -0.5 * delta:
                temp = eta / delta + 1.5
                return temp / delta
            else:
                temp = eta / delta + 1.5
                return 0.5 * (6.0 - 4.0 * temp) / delta
        else:
            return -1.0


@pure
@stack_array("values_stored", "values_temp", "w")
def convolution(p: "int", grids: "Final[float[:]]", eta: "float") -> "float":
    r"""
    evaluate a hat function (B-spline) at eta, i.e.

    .. math::
        1.0 / delta * S(eta/ delta)

    where S is the B-spline of degree p.
    Here we use the definition by convolution of the B-splines.

    Parameters
    ----------
    p : int
        degree of the hat function (B-spline)

    grids : float array
        p + 2 points used in the definition of B-splines.

    eta : float
        evluation point

    Returns:
    --------
        value of the hat function at eta
    """
    if eta < grids[p + 1] and eta >= grids[0]:
        value_stored = zeros(p + 1, dtype=float)
        value_temp = zeros(p + 1, dtype=float)
        w = zeros(p + 1, dtype=float)
        # 0 degree B-spline evluation
        # index of the cell where eta is located
        ie = int((eta - grids[0]) / (grids[1] - grids[0]))
        # value (is 1) in this cell
        value_stored[ie] = 1.0
        result = value_stored[ie]

        for loop in range(p):
            for ii in range(p - loop + 1):
                w[ii] = (eta - grids[ii]) / (grids[ii + loop + 1] - grids[ii])

            for ii in range(p - loop):
                value_temp[ii] = w[ii] * value_stored[ii] + (1 - w[ii + 1]) * value_stored[ii + 1]

            value_stored[:] = value_temp

        result = value_stored[0]
    else:
        result = 0.0
    return result


@pure
@stack_array("values_stored", "values_temp", "w")
def convolution_der(p: "int", grids: "Final[float[:]]", eta: "float") -> "float":
    r"""
    evaluate the derivative of a hat function (B-spline) at eta, i.e.

    .. math::
        1.0 / delta^2 * S'(eta/ delta)

    where S is the B-spline of degree p.
    Here we use the definition by convolution of the B-splines.

    Parameters
    ----------
    p : int
        degree of the hat function (B-spline)

    grids : float array
        p + 2 points used in the definition of B-splines.

    eta : float
        evluation point

    Returns:
    --------
        value of the derivative of the hat function at eta
    """
    if eta < grids[p + 1] and eta >= grids[0]:
        value_stored = zeros(p + 1, dtype=float)
        value_temp = zeros(p + 1, dtype=float)
        w = zeros(p + 1, dtype=float)
        # 0 degree B-spline evluation
        # index of the cell where eta is located
        ie = int((eta - grids[0]) / (grids[1] - grids[0]))
        # value (is 1) in this cell
        value_stored[ie] = 1.0
        result = 0.0

        for loop in range(1, p):
            for ii in range(p - loop + 2):
                w[ii] = (eta - grids[ii]) / (grids[ii + loop] - grids[ii])

            for ii in range(p - loop + 1):
                value_temp[ii] = w[ii] * value_stored[ii] + (1 - w[ii + 1]) * value_stored[ii + 1]

            value_stored[:] = value_temp

        result = p / (grids[p] - grids[0]) * value_stored[0] - p / (grids[p + 1] - grids[1]) * value_stored[1]
    else:
        result = 0.0
    return result
