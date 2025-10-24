# coding: utf-8
#
# Copyright 2020 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Acccelerated functions for point-wise evaluation of tensor product B-splines.

S(eta1) = sum_i [ c_i * B_i(eta1) ] with c_i in R.

Possible combinations for tensor product (B):
* (N)
* (D)
* (dN/deta)
"""

from typing import Final

from numpy import empty, zeros
from pyccel.decorators import pure, stack_array

import struphy.bsplines.bsplines_kernels as bsplines_kernels


# =============================================================================
@pure
def evaluation_kernel_1d(p1: int, basis1: "Final[float[:]]", ind1: "Final[int[:]]", coeff: "Final[float[:]]") -> float:
    """
    Summing non-zero contributions.

    Parameters
    ----------
        p1 : int
            Degree of the univariate spline.

        basis1 : array[float]
            The p+1 values of non-zero basis splines at one point (eta1,) from 'basis_funs' of shape.

        ind1 : array[int]
            Global indices of non-vanishing splines in the element of the considered point.

        coeff : array[float]
            The spline coefficients c_i.

    Returns
    -------
        spline_value : float
            Value of spline at point (eta1,).
    """

    spline_value = 0.0

    for il1 in range(p1 + 1):
        i1 = ind1[il1]

        spline_value += coeff[i1] * basis1[il1]

    return spline_value


# =============================================================================
@pure
@stack_array("tmp1", "tmp2")
def evaluate(
    kind1: int,
    t1: "Final[float[:]]",
    p1: int,
    ind1: "Final[int[:,:]]",
    coeff: "Final[float[:]]",
    eta1: float,
) -> float:
    """
    Point-wise evaluation of a spline.

    Parameters
    ----------
        kind : int
            Kind of spline to evaluate.
                * 0 : N
                * 1 : D
                * 2 : dN/deta
                * 3 : ddN/deta^2

        t1 : array[float]
            Knot vector of univariate spline.

        p1 : int
            Degree of univariate spline.

        ind1 : array[int]
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).

        coeff : array[float]
            The spline coefficients c_i.

        eta1 : float
            Point of evaluation.

    Returns
    -------
        spline_value: float
            Value of spline at point (eta1,).
    """

    # find knot span indices
    span1 = bsplines_kernels.find_span(t1, p1, eta1)

    # evaluate non-vanishing basis functions
    b1 = empty(p1 + 1, dtype=float)
    bl1 = empty(p1, dtype=float)
    br1 = empty(p1, dtype=float)
    tmp1 = zeros(p1 + 1, dtype=int)

    if kind1 == 1:
        bsplines_kernels.basis_funs(t1, p1, eta1, span1, bl1, br1, b1)
    elif kind1 == 2:
        bsplines_kernels.basis_funs(t1, p1, eta1, span1, bl1, br1, b1)
        bsplines_kernels.scaling(t1, p1, span1, b1)
    elif kind1 == 3:
        bsplines_kernels.basis_funs_1st_der(t1, p1, eta1, span1, bl1, br1, b1)
    elif kind1 == 4:
        tmp2 = zeros((3, p1 + 1), dtype=float)
        bsplines_kernels.basis_funs_all_ders(t1, p1, eta1, span1, bl1, br1, 2, tmp2)
        b1[:] = tmp2[2, :]

    # sum up non-vanishing contributions
    tmp1[:] = ind1[span1 - p1, :]
    spline_value = evaluation_kernel_1d(p1, b1, tmp1, coeff)

    return spline_value


# =============================================================================
@pure
def evaluate_vector(
    t1: "Final[float[:]]",
    p1: int,
    ind1: "Final[int[:,:]]",
    coeff: "Final[float[:]]",
    eta1: "float[:]",
    spline_values: "float[:]",
    kind: int,
):
    """
    Vector evaluation of a uni-variate spline.

    Parameters
    ----------
        t1 : array[float]
            Knot vector of univariate spline.

        p1 : int
            Degree of univariate spline.

        ind1 : array[int]
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).

        coeff : array[float]
            The spline coefficients c_i.

        eta1 : array[float]
            Points of evaluation in a 1d array.

        spline_values : array[float]
            Splines evaluated at points S_ij = S(eta1_i, eta2_j).

        kind : int
            Kind of spline to evaluate.
                * 0 : N
                * 1 : D
                * 2 : dN/deta
                * 3 : ddN/deta^2
    """

    for i1 in range(len(eta1)):
        if kind == 0:
            spline_values[i1] = evaluate(1, t1, p1, ind1, coeff, eta1[i1])
        elif kind == 1:
            spline_values[i1] = evaluate(2, t1, p1, ind1, coeff, eta1[i1])
        elif kind == 2:
            spline_values[i1] = evaluate(3, t1, p1, ind1, coeff, eta1[i1])
        elif kind == 3:
            spline_values[i1] = evaluate(4, t1, p1, ind1, coeff, eta1[i1])
