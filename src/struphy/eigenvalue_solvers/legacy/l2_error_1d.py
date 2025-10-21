# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Modules to compute L2-errors in 1d.
"""

import scipy.sparse as spa

from struphy.utils.arrays import xp as np


# ======= error in V0 ====================
def l2_error_V0(spline_space, mapping, coeff, fun):
    """
    Computes the 1d L2 - error (N) of the given B-spline space of degree p with coefficients coeff with the function fun.

    Parameters
    ----------
    spline_space : Spline_space_1d
        a 1d B-spline space

    mapping : callable
        derivative of mapping df/dxi

    coeff : array_like
        coefficients of the spline space

    fun : callable
        function for which the error shall be computed
    """

    p = spline_space.p  # spline degrees
    Nel = spline_space.Nel  # number of elements
    NbaseN = spline_space.NbaseN  # total number of basis functions (N)

    n_quad = spline_space.n_quad  # number of quadrature points per element
    pts = spline_space.pts  # global quadrature points in format (element, local quad_point)
    wts = spline_space.wts  # global quadrature weights in format (element, local weight)

    basisN = spline_space.basisN  # evaluated basis functions at quadrature points

    # evaluation of mapping at quadrature points
    mat_map = mapping(pts)

    # evaluation of function at quadrature points
    mat_f = fun(pts)

    # assembly
    error = np.zeros(Nel, dtype=float)

    for ie in range(Nel):
        for q in range(n_quad):
            bi = 0.0

            for il in range(p + 1):
                bi += coeff[(ie + il) % NbaseN] * basisN[ie, il, 0, q]

            error[ie] += wts[ie, q] * (bi - mat_f[ie, q]) ** 2

    return np.sqrt(error.sum())


# ======= error in V1 ====================
def l2_error_V1(spline_space, mapping, coeff, fun):
    """
    Computes the 1d L2 - error (D) of the given B-spline space of degree p with coefficients coeff with the function fun.

    Parameters
    ----------
    spline_space : Spline_space_1d
        a 1d B-spline space

    mapping : callable
        derivative of mapping df/dxi

    coeff : array_like
        coefficients of the spline space

    fun : callable
        function for which the error shall be computed
    """

    p = spline_space.p  # spline degrees
    Nel = spline_space.Nel  # number of elements
    NbaseD = spline_space.NbaseD  # total number of basis functions (N)

    n_quad = spline_space.n_quad  # number of quadrature points per element
    pts = spline_space.pts  # global quadrature points in format (element, local quad_point)
    wts = spline_space.wts  # global quadrature weights in format (element, local weight)

    basisD = spline_space.basisD  # evaluated basis functions at quadrature points

    # evaluation of mapping at quadrature points
    mat_map = 1 / mapping(pts)

    # evaluation of function at quadrature points
    mat_f = fun(pts)

    # assembly
    error = np.zeros(Nel, dtype=float)

    for ie in range(Nel):
        for q in range(n_quad):
            bi = 0.0

            for il in range(p):
                bi += coeff[(ie + il) % NbaseD] * basisD[ie, il, 0, q]

            error[ie] += wts[ie, q] * (bi - mat_f[ie, q]) ** 2

    return np.sqrt(error.sum())
