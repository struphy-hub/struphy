# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Modules to compute inner products in 1d.
"""

import cunumpy as xp
import scipy.sparse as spa


# ======= inner product in V0 ====================
def inner_prod_V0(spline_space, fun, mapping=None):
    """
    Computes the 1d inner product (N) of the given B-spline space of degree p with the function fun.

    Parameters
    ----------
    spline_space : spline_space_1d
        a 1d B-spline space

    mapping : callable
        derivative of mapping df/dxi

    fun : callable
        function for which the inner product with every basis function in V0 shall be computed
    """

    p = spline_space.p  # spline degrees
    Nel = spline_space.Nel  # number of elements
    NbaseN = spline_space.NbaseN  # total number of basis functions (N)

    n_quad = spline_space.n_quad  # number of quadrature points per element
    pts = spline_space.pts  # global quadrature points in format (element, local quad_point)
    wts = spline_space.wts  # global quadrature weights in format (element, local weight)

    basisN = spline_space.basisN  # evaluated basis functions at quadrature points

    # evaluation of mapping at quadrature points
    if mapping is None:
        mat_map = xp.ones(pts.shape, dtype=float)
    else:
        mat_map = mapping(pts.flatten()).reshape(pts.shape)

    # evaluation of function at quadrature points
    mat_f = fun(pts.flatten()).reshape(pts.shape)

    # assembly
    F = xp.zeros(NbaseN, dtype=float)

    for ie in range(Nel):
        for il in range(p + 1):
            value = 0.0

            for q in range(n_quad):
                value += wts[ie, q] * basisN[ie, il, 0, q] * mat_f[ie, q] * mat_map[ie, q]

            F[(ie + il) % NbaseN] += value

    return F


# ======= inner product in V1 ====================
def inner_prod_V1(spline_space, fun, mapping=None):
    """
    Computes the 1d inner product (D) of the given B-spline space of degree p with the function fun.

    Parameters
    ----------
    spline_space : spline_space_1d
        a 1d B-spline space

    mapping : callable
        derivative of mapping df/dxi

    fun : callable
        function for which the inner product with every basis function in V1 shall be computed
    """

    p = spline_space.p  # spline degrees
    Nel = spline_space.Nel  # number of elements
    NbaseD = spline_space.NbaseD  # total number of basis functions (N)

    n_quad = spline_space.n_quad  # number of quadrature points per element
    pts = spline_space.pts  # global quadrature points in format (element, local quad_point)
    wts = spline_space.wts  # global quadrature weights in format (element, local weight)

    basisD = spline_space.basisD  # evaluated basis functions at quadrature points

    # evaluation of mapping at quadrature points
    if mapping is None:
        mat_map = xp.ones(pts.shape, dtype=float)
    else:
        mat_map = 1 / mapping(pts.flatten()).reshape(pts.shape)

    # evaluation of function at quadrature points
    mat_f = fun(pts.flatten()).reshape(pts.shape)

    # assembly
    F = xp.zeros(NbaseD, dtype=float)

    for ie in range(Nel):
        for il in range(p):
            value = 0.0

            for q in range(n_quad):
                value += wts[ie, q] * basisD[ie, il, 0, q] * mat_f[ie, q] * mat_map[ie, q]

            F[(ie + il) % NbaseD] += value

    return F
