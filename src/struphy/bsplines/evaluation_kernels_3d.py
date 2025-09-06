# coding: utf-8

"""
Acccelerated functions for point-wise evaluation of tensor product B-splines.

S(eta1, eta2, eta3) = sum_ijk [ c_ijk * B_i(eta1) * B_j(eta2) * B_k(eta3) ] with c_ijk in R.

Possible combinations for tensor product (BBB):
* (NNN)
* (DNN)
* (NDN)
* (NND)
* (NDD)
* (DND)
* (DDN)
* (DDD)
* (dN/deta N N)
* (N dN/deta N)
* (N N dN/deta)
"""

from numpy import empty, shape, zeros
from pyccel.decorators import stack_array

import struphy.bsplines.bsplines_kernels as bsplines_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments

#################################
### Single process evaluation ###
#################################


def evaluation_kernel_3d(
    p1: int,
    p2: int,
    p3: int,
    basis1: "float[:]",
    basis2: "float[:]",
    basis3: "float[:]",
    ind1: "int[:]",
    ind2: "int[:]",
    ind3: "int[:]",
    coeff: "float[:,:,:]",
) -> float:
    """
    Summing non-zero contributions of a spline function (serial, needs global arrays).

    Parameters
    ----------
        p1, p2, p3 : int
            Degrees of the univariate splines in each direction.

        basis1, basis2, basis3 : array[float]
            The p + 1 values of non-zero basis splines at one point (eta1, eta2, eta3) in each direction.

        ind1, ind2, ind3 : array[int]
            Global indices of non-vanishing splines in the element of the considered point.

        coeff : array[float]
            The spline coefficients c_ijk.

    Returns
    -------
        spline_value : float
            Value of tensor-product spline at point (eta1, eta2, eta3).
    """

    spline_value = 0.0

    for il1 in range(p1 + 1):
        i1 = ind1[il1]
        for il2 in range(p2 + 1):
            i2 = ind2[il2]
            for il3 in range(p3 + 1):
                i3 = ind3[il3]

                spline_value += coeff[i1, i2, i3] * basis1[il1] * basis2[il2] * basis3[il3]

    return spline_value


@stack_array("b1", "b2", "b3", "bl1", "bl2", "bl3", "br1", "br2", "br3", "tmp1", "tmp2", "tmp3")
def evaluate_3d(
    kind1: int,
    kind2: int,
    kind3: int,
    t1: "float[:]",
    t2: "float[:]",
    t3: "float[:]",
    p1: int,
    p2: int,
    p3: int,
    ind1: "int[:,:]",
    ind2: "int[:,:]",
    ind3: "int[:,:]",
    coeff: "float[:,:,:]",
    eta1: float,
    eta2: float,
    eta3: float,
) -> float:
    """
    Point-wise evaluation of a tensor-product spline (serial, needs global arrays).

    Parameters
    ----------
        kind1, kind2, kind3 : int
            Kind of univariate spline. 1 for B-spline, 2 for M-spline and 3 for derivative of B-spline.

        t1, t2, t3 : array[float]
            Knot vectors of univariate splines.

        p1, p2, p3 : int
            Degrees of univariate splines.

        ind1, ind2, ind3 : array[int]
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).

        coeff : array[float]
            The spline coefficients c_ijk.

        eta1, eta2, eta3 : float
            Point of evaluation.

    Returns
    -------
        spline_value: float
            Value of tensor-product spline at point (eta1, eta2, eta3).
    """

    # find knot span indices
    span1 = bsplines_kernels.find_span(t1, p1, eta1)
    span2 = bsplines_kernels.find_span(t2, p2, eta2)
    span3 = bsplines_kernels.find_span(t3, p3, eta3)

    # evaluate non-vanishing basis functions
    b1 = empty(p1 + 1, dtype=float)
    b2 = empty(p2 + 1, dtype=float)
    b3 = empty(p3 + 1, dtype=float)

    bl1 = empty(p1, dtype=float)
    bl2 = empty(p2, dtype=float)
    bl3 = empty(p3, dtype=float)

    br1 = empty(p1, dtype=float)
    br2 = empty(p2, dtype=float)
    br3 = empty(p3, dtype=float)

    tmp1 = zeros(p1 + 1, dtype=int)
    tmp2 = zeros(p2 + 1, dtype=int)
    tmp3 = zeros(p3 + 1, dtype=int)

    # 1st direction
    if kind1 == 1:
        bsplines_kernels.basis_funs(t1, p1, eta1, span1, bl1, br1, b1)
    elif kind1 == 2:
        bsplines_kernels.basis_funs(t1, p1, eta1, span1, bl1, br1, b1)
        bsplines_kernels.scaling(t1, p1, span1, b1)
    elif kind1 == 3:
        bsplines_kernels.basis_funs_1st_der(t1, p1, eta1, span1, bl1, br1, b1)

    # 2nd direction
    if kind2 == 1:
        bsplines_kernels.basis_funs(t2, p2, eta2, span2, bl2, br2, b2)
    elif kind2 == 2:
        bsplines_kernels.basis_funs(t2, p2, eta2, span2, bl2, br2, b2)
        bsplines_kernels.scaling(t2, p2, span2, b2)
    elif kind2 == 3:
        bsplines_kernels.basis_funs_1st_der(t2, p2, eta2, span2, bl2, br2, b2)

    # 3rd direction
    if kind3 == 1:
        bsplines_kernels.basis_funs(t3, p3, eta3, span3, bl3, br3, b3)
    elif kind3 == 2:
        bsplines_kernels.basis_funs(t3, p3, eta3, span3, bl3, br3, b3)
        bsplines_kernels.scaling(t3, p3, span3, b3)
    elif kind3 == 3:
        bsplines_kernels.basis_funs_1st_der(t3, p3, eta3, span3, bl3, br3, b3)

    # sum up non-vanishing contributions
    tmp1[:] = ind1[span1 - p1, :]
    tmp2[:] = ind2[span2 - p2, :]
    tmp3[:] = ind3[span3 - p3, :]
    spline_value = evaluation_kernel_3d(p1, p2, p3, b1, b2, b3, tmp1, tmp2, tmp3, coeff)

    return spline_value


def evaluate_tensor_product(
    t1: "float[:]",
    t2: "float[:]",
    t3: "float[:]",
    p1: int,
    p2: int,
    p3: int,
    ind1: "int[:,:]",
    ind2: "int[:,:]",
    ind3: "int[:,:]",
    coeff: "float[:,:,:]",
    eta1: "float[:]",
    eta2: "float[:]",
    eta3: "float[:]",
    spline_values: "float[:,:,:]",
    kind: int,
):
    """
    Tensor-product evaluation of a tensor-product spline (serial, needs global arrays).

    Parameters
    ----------
        t1, t2, t3 : array[float]
            Knot vectors of univariate splines.

        p1, p2, p3 : int
            Degrees of univariate splines.

        ind1, ind2, ind3 : array[int]
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).

        coeff : array[float]
            The spline coefficients c_ijk.

        eta1, eta2, eta3 : array[float]
            Points of evaluation in 1d arrays.

        spline_values : array[float]
            Splines evaluated at points S_ijk = S(eta1_i, eta2_j, eta3_k).

        kind : int
            Kind of spline to evaluate.
                * 0  : NNN
                * 11 : DNN
                * 12 : NDN
                * 13 : NND
                * 21 : NDD
                * 22 : DND
                * 23 : DDN
                * 3  : DDD
                * 41 : dN/deta N N
                * 42 : N dN/deta N
                * 43 : N N dN/deta
    """

    for i1 in range(len(eta1)):
        for i2 in range(len(eta2)):
            for i3 in range(len(eta3)):
                if kind == 0:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        1, 1, 1, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1], eta2[i2], eta3[i3]
                    )
                elif kind == 11:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        2, 1, 1, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1], eta2[i2], eta3[i3]
                    )
                elif kind == 12:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        1, 2, 1, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1], eta2[i2], eta3[i3]
                    )
                elif kind == 13:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        1, 1, 2, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1], eta2[i2], eta3[i3]
                    )
                elif kind == 21:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        1, 2, 2, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1], eta2[i2], eta3[i3]
                    )
                elif kind == 22:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        2, 1, 2, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1], eta2[i2], eta3[i3]
                    )
                elif kind == 23:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        2, 2, 1, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1], eta2[i2], eta3[i3]
                    )
                elif kind == 3:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        2, 2, 2, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1], eta2[i2], eta3[i3]
                    )
                elif kind == 41:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        3, 1, 1, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1], eta2[i2], eta3[i3]
                    )
                elif kind == 42:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        1, 3, 1, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1], eta2[i2], eta3[i3]
                    )
                elif kind == 43:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        1, 1, 3, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1], eta2[i2], eta3[i3]
                    )


def evaluate_matrix(
    t1: "float[:]",
    t2: "float[:]",
    t3: "float[:]",
    p1: int,
    p2: int,
    p3: int,
    ind1: "int[:,:]",
    ind2: "int[:,:]",
    ind3: "int[:,:]",
    coeff: "float[:,:,:]",
    eta1: "float[:,:,:]",
    eta2: "float[:,:,:]",
    eta3: "float[:,:,:]",
    spline_values: "float[:,:,:]",
    kind: int,
):
    """
    General evaluation of a tensor-product spline (serial, needs global arrays).

    Parameters
    ----------
        t1, t2, t3 : array[float]
            Knot vectors of univariate splines.

        p1, p2, p3 : int
            Degrees of univariate splines.

        ind1, ind2, ind3 : array[int]
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).

        coeff : array[float]
            The spline coefficients c_ijk.

        eta1, eta2, eta3 : array[float]
            Points of evaluation in 3d arrays such that shape(eta1) == shape(eta2) == shape(eta3).

        spline_values : array[float]
            Splines evaluated at points S_ijk = S(eta1_i, eta2_j, eta3_k).

        kind : int
            Kind of spline to evaluate.
                * 0  : NNN
                * 11 : DNN
                * 12 : NDN
                * 13 : NND
                * 21 : NDD
                * 22 : DND
                * 23 : DDN
                * 3  : DDD
                * 41 : dN/deta N N
                * 42 : N dN/deta N
                * 43 : N N dN/deta
    """

    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                if kind == 0:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        1,
                        1,
                        1,
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        ind1,
                        ind2,
                        ind3,
                        coeff,
                        eta1[i1, i2, i3],
                        eta2[i1, i2, i3],
                        eta3[i1, i2, i3],
                    )
                elif kind == 11:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        2,
                        1,
                        1,
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        ind1,
                        ind2,
                        ind3,
                        coeff,
                        eta1[i1, i2, i3],
                        eta2[i1, i2, i3],
                        eta3[i1, i2, i3],
                    )
                elif kind == 12:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        1,
                        2,
                        1,
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        ind1,
                        ind2,
                        ind3,
                        coeff,
                        eta1[i1, i2, i3],
                        eta2[i1, i2, i3],
                        eta3[i1, i2, i3],
                    )
                elif kind == 13:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        1,
                        1,
                        2,
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        ind1,
                        ind2,
                        ind3,
                        coeff,
                        eta1[i1, i2, i3],
                        eta2[i1, i2, i3],
                        eta3[i1, i2, i3],
                    )
                elif kind == 21:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        1,
                        2,
                        2,
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        ind1,
                        ind2,
                        ind3,
                        coeff,
                        eta1[i1, i2, i3],
                        eta2[i1, i2, i3],
                        eta3[i1, i2, i3],
                    )
                elif kind == 22:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        2,
                        1,
                        2,
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        ind1,
                        ind2,
                        ind3,
                        coeff,
                        eta1[i1, i2, i3],
                        eta2[i1, i2, i3],
                        eta3[i1, i2, i3],
                    )
                elif kind == 23:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        2,
                        2,
                        1,
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        ind1,
                        ind2,
                        ind3,
                        coeff,
                        eta1[i1, i2, i3],
                        eta2[i1, i2, i3],
                        eta3[i1, i2, i3],
                    )
                elif kind == 3:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        2,
                        2,
                        2,
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        ind1,
                        ind2,
                        ind3,
                        coeff,
                        eta1[i1, i2, i3],
                        eta2[i1, i2, i3],
                        eta3[i1, i2, i3],
                    )
                elif kind == 41:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        3,
                        1,
                        1,
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        ind1,
                        ind2,
                        ind3,
                        coeff,
                        eta1[i1, i2, i3],
                        eta2[i1, i2, i3],
                        eta3[i1, i2, i3],
                    )
                elif kind == 42:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        1,
                        3,
                        1,
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        ind1,
                        ind2,
                        ind3,
                        coeff,
                        eta1[i1, i2, i3],
                        eta2[i1, i2, i3],
                        eta3[i1, i2, i3],
                    )
                elif kind == 43:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        1,
                        1,
                        3,
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        ind1,
                        ind2,
                        ind3,
                        coeff,
                        eta1[i1, i2, i3],
                        eta2[i1, i2, i3],
                        eta3[i1, i2, i3],
                    )


def evaluate_sparse(
    t1: "float[:]",
    t2: "float[:]",
    t3: "float[:]",
    p1: int,
    p2: int,
    p3: int,
    ind1: "int[:,:]",
    ind2: "int[:,:]",
    ind3: "int[:,:]",
    coeff: "float[:,:,:]",
    eta1: "float[:,:,:]",
    eta2: "float[:,:,:]",
    eta3: "float[:,:,:]",
    spline_values: "float[:,:,:]",
    kind: int,
):
    """
    Evaluation of a tensor-product spline using sparse meshgrids (serial, needs global arrays).

    Parameters
    ----------
        t1, t2, t3 : array[float]
            Knot vectors of univariate splines.

        p1, p2, p3 : int
            Degrees of univariate splines.

        ind1, ind2, ind3 : array[int]
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).

        coeff : array[float]
            The spline coefficients c_ijk.

        eta1, eta2, eta3 : array[float]
            Points of evaluation in 3d arrays such that shape(eta1) = (:,1,1), shape(eta2) = (1,:,1), shape(eta3) = (1,1,:).

        spline_values : array[float]
            Splines evaluated at points S_ijk = S(eta1_i, eta2_j, eta3_k).

        kind : int
            Kind of spline to evaluate.
                * 0  : NNN
                * 11 : DNN
                * 12 : NDN
                * 13 : NND
                * 21 : NDD
                * 22 : DND
                * 23 : DDN
                * 3  : DDD
                * 41 : dN/deta N N
                * 42 : N dN/deta N
                * 43 : N N dN/deta
    """

    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                if kind == 0:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        1,
                        1,
                        1,
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        ind1,
                        ind2,
                        ind3,
                        coeff,
                        eta1[i1, 0, 0],
                        eta2[0, i2, 0],
                        eta3[0, 0, i3],
                    )
                elif kind == 11:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        2,
                        1,
                        1,
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        ind1,
                        ind2,
                        ind3,
                        coeff,
                        eta1[i1, 0, 0],
                        eta2[0, i2, 0],
                        eta3[0, 0, i3],
                    )
                elif kind == 12:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        1,
                        2,
                        1,
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        ind1,
                        ind2,
                        ind3,
                        coeff,
                        eta1[i1, 0, 0],
                        eta2[0, i2, 0],
                        eta3[0, 0, i3],
                    )
                elif kind == 13:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        1,
                        1,
                        2,
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        ind1,
                        ind2,
                        ind3,
                        coeff,
                        eta1[i1, 0, 0],
                        eta2[0, i2, 0],
                        eta3[0, 0, i3],
                    )
                elif kind == 21:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        1,
                        2,
                        2,
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        ind1,
                        ind2,
                        ind3,
                        coeff,
                        eta1[i1, 0, 0],
                        eta2[0, i2, 0],
                        eta3[0, 0, i3],
                    )
                elif kind == 22:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        2,
                        1,
                        2,
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        ind1,
                        ind2,
                        ind3,
                        coeff,
                        eta1[i1, 0, 0],
                        eta2[0, i2, 0],
                        eta3[0, 0, i3],
                    )
                elif kind == 23:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        2,
                        2,
                        1,
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        ind1,
                        ind2,
                        ind3,
                        coeff,
                        eta1[i1, 0, 0],
                        eta2[0, i2, 0],
                        eta3[0, 0, i3],
                    )
                elif kind == 3:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        2,
                        2,
                        2,
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        ind1,
                        ind2,
                        ind3,
                        coeff,
                        eta1[i1, 0, 0],
                        eta2[0, i2, 0],
                        eta3[0, 0, i3],
                    )
                elif kind == 41:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        3,
                        1,
                        1,
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        ind1,
                        ind2,
                        ind3,
                        coeff,
                        eta1[i1, 0, 0],
                        eta2[0, i2, 0],
                        eta3[0, 0, i3],
                    )
                elif kind == 42:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        1,
                        3,
                        1,
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        ind1,
                        ind2,
                        ind3,
                        coeff,
                        eta1[i1, 0, 0],
                        eta2[0, i2, 0],
                        eta3[0, 0, i3],
                    )
                elif kind == 43:
                    spline_values[i1, i2, i3] = evaluate_3d(
                        1,
                        1,
                        3,
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        ind1,
                        ind2,
                        ind3,
                        coeff,
                        eta1[i1, 0, 0],
                        eta2[0, i2, 0],
                        eta3[0, 0, i3],
                    )


##############################
### Distributed evaluation ###
##############################


def eval_spline_mpi_kernel(
    p1: "int",
    p2: "int",
    p3: "int",
    basis1: "float[:]",
    basis2: "float[:]",
    basis3: "float[:]",
    span1: "int",
    span2: "int",
    span3: "int",
    _data: "float[:,:,:]",
    starts: "int[:]",
) -> float:
    """
    Summing non-zero contributions of a spline function with distributed memory (domain decomposition).

    Parameters
    ----------
        p1, p2, p3 : int
            Degrees of the univariate splines in each direction.

        basis1, basis2, basis3 : array[float]
            The p + 1 values of non-zero basis splines at one point (eta1, eta2, eta3) in each direction.

        span1, span2, span3: int
            Knot span index in each direction.

        _data : array[float]
            The spline coefficients c_ijk of the current process, ie. the _data attribute of a StencilVector.

        starts : array[int]
            Starting indices of current process.

    Returns
    -------
        spline_value : float
            Value of tensor-product spline at point (eta1, eta2, eta3).
    """

    spline_value = 0.0

    for il1 in range(p1 + 1):
        i1 = span1 + il1 - starts[0]
        for il2 in range(p2 + 1):
            i2 = span2 + il2 - starts[1]
            for il3 in range(p3 + 1):
                i3 = span3 + il3 - starts[2]

                spline_value += _data[i1, i2, i3] * basis1[il1] * basis2[il2] * basis3[il3]

    return spline_value


def eval_spline_derivative_mpi_kernel(
    p1: "int",
    p2: "int",
    p3: "int",
    basis1: "float[:]",
    basis2: "float[:]",
    basis3: "float[:]",
    span1: "int",
    span2: "int",
    span3: "int",
    _data: "float[:,:,:]",
    starts: "int[:]",
    direction: "int",
) -> float:
    """
    Kernel for the derivative of a spline in one direction (distributed, domain decomp.).

    Parameters
    ----------
        p1, p2, p3 : int
            Degrees of the univariate splines in each direction.

        basis1, basis2, basis3 : array[float]
            The p + 1 values of non-zero basis splines at one point (eta1, eta2, eta3) in each direction.

        span1, span2, span3: int
            Knot span index in each direction.

        _data : array[float]
            The spline coefficients c_ijk in current process, ie. the _data attribute of a StencilVector.

        starts : array[int]
            Starting indices of current process.

    Returns
    -------
        spline_value : float
            Derivative in one direction of tensor-product spline at point (eta1, eta2, eta3).
    """

    spline_value = 0.0

    if direction == int(1):
        for il1 in range(p1 + 1):
            i1 = span1 + il1 - starts[0]
            for il2 in range(p2 + 1):
                i2 = span2 + il2 - starts[1]
                for il3 in range(p3 + 1):
                    i3 = span3 + il3 - starts[2]

                    spline_value += (
                        (_data[i1 + 1, i2, i3] - _data[i1, i2, i3]) * basis1[il1] * basis2[il2] * basis3[il3]
                    )

    if direction == int(2):
        for il1 in range(p1 + 1):
            i1 = span1 + il1 - starts[0]
            for il2 in range(p2 + 1):
                i2 = span2 + il2 - starts[1]
                for il3 in range(p3 + 1):
                    i3 = span3 + il3 - starts[2]

                    spline_value += (
                        (_data[i1, i2 + 1, i3] - _data[i1, i2, i3]) * basis1[il1] * basis2[il2] * basis3[il3]
                    )

    if direction == int(3):
        for il1 in range(p1 + 1):
            i1 = span1 + il1 - starts[0]
            for il2 in range(p2 + 1):
                i2 = span2 + il2 - starts[1]
                for il3 in range(p3 + 1):
                    i3 = span3 + il3 - starts[2]

                    spline_value += (
                        (_data[i1, i2, i3 + 1] - _data[i1, i2, i3]) * basis1[il1] * basis2[il2] * basis3[il3]
                    )

    return spline_value


@stack_array("bn1", "bn2", "bn3", "bd1", "bd2", "bd3")
def eval_spline_mpi(
    eta1: float,
    eta2: float,
    eta3: float,
    _data: "float[:,:,:]",
    kind: "int[:]",
    pn: "int[:]",
    tn1: "float[:]",
    tn2: "float[:]",
    tn3: "float[:]",
    starts: "int[:]",
) -> float:
    """
    Point-wise evaluation of a tensor-product spline, distributed.

    Parameters
    ----------
        eta1, eta2, eta3 : float
            Evaluation point in [0, 1]^3.

        _data : array[float]
            The spline coefficients c_ijk.

        kind : array[int]
            Kind of 1d basis in each direction: 0 = N-spline, 1 = D-spline.

        pn : array[int]
            Spline degrees of V0 in each direction.

        tn1, tn2, tn3 : array[float]
            Knot vectors of V0 in each direction.

        starts : array[float]
            Start indices of splines on current process.

    Returns
    -------
        value : float
            Value of tensor-product spline at point (eta_1, eta_2, eta_3).
    """

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # get spline values at eta
    span1 = bsplines_kernels.find_span(tn1, pn[0], eta1)
    span2 = bsplines_kernels.find_span(tn2, pn[1], eta2)
    span3 = bsplines_kernels.find_span(tn3, pn[2], eta3)

    bsplines_kernels.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
    bsplines_kernels.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
    bsplines_kernels.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

    if kind[0] == 0:
        b1 = bn1
    else:
        b1 = bd1

    if kind[1] == 0:
        b2 = bn2
    else:
        b2 = bd2

    if kind[2] == 0:
        b3 = bn3
    else:
        b3 = bd3

    value = eval_spline_mpi_kernel(
        pn[0] - kind[0], pn[1] - kind[1], pn[2] - kind[2], b1, b2, b3, span1, span2, span3, _data, starts
    )

    return value


def eval_spline_mpi_tensor_product(
    eta1: "float[:]",
    eta2: "float[:]",
    eta3: "float[:]",
    _data: "float[:,:,:]",
    kind: "int[:]",
    pn: "int[:]",
    tn1: "float[:]",
    tn2: "float[:]",
    tn3: "float[:]",
    starts: "int[:]",
    values: "float[:,:,:]",
):
    """
    Tensor-product evaluation of a tensor-product spline, distributed.

    Parameters
    ----------
        eta1, eta2, eta3 : array[float]
            Evaluation points as 1d arrays; points not on local process domain must be flagged as -1.
            The out array is global and contains the spline values as S_ijk = S(eta1[i], eta2[j], eta3[k]).

        _data : array[float]
            The spline coefficients c_ijk.

        kind : array[int]
            Kind of 1d basis in each direction: 0 = N-spline, 1 = D-spline.

        pn : array[int]
            Spline degrees of V0 in each direction.

        tn1, tn2, tn3 : array[float]
            Knot vectors of V0 in each direction.

        starts : array[float]
            Start indices of splines on current process.

        values : array[float]
            Return array for spline values S_ijk = S(eta1[i], eta2[j], eta3[k]).
    """

    for i in range(len(eta1)):
        if eta1[i] == -1.0:
            continue  # point not in process domain
        for j in range(len(eta2)):
            if eta2[j] == -1.0:
                continue  # point not in process domain
            for k in range(len(eta3)):
                if eta3[k] == -1.0:
                    continue  # point not in process domain

                values[i, j, k] = eval_spline_mpi(eta1[i], eta2[j], eta3[k], _data, kind, pn, tn1, tn2, tn3, starts)


@stack_array("bn1", "bn2", "bn3", "bd1", "bd2", "bd3")
def eval_spline_mpi_tensor_product_fast(
    eta1: "float[:]",
    eta2: "float[:]",
    eta3: "float[:]",
    _data: "float[:,:,:]",
    kind: "int[:]",
    pn: "int[:]",
    tn1: "float[:]",
    tn2: "float[:]",
    tn3: "float[:]",
    starts: "int[:]",
    values: "float[:,:,:]",
):
    """
    Tensor-product evaluation of a tensor-product spline, distributed,
    and optimized by computing spans and spline values only once.

    Parameters
    ----------
        eta1, eta2, eta3 : array[float]
            Evaluation points as 1d arrays; points not on local process domain must be flagged as -1.
            Spline values are obtained as S_ijk = S(eta1[i], eta2[j], eta3[k]).

        _data : array[float]
            The spline coefficients c_ijk.

        kind : array[int]
            Kind of 1d basis in each direction: 0 = N-spline, 1 = D-spline.

        pn : array[int]
            Spline degrees of V0 in each direction.

        tn1, tn2, tn3 : array[float]
            Knot vectors of V0 in each direction.

        starts : array[float]
            Start indices of splines on current process.

        values : array[float]
            Return array for spline values S_ijk = S(eta1[i], eta2[j], eta3[k]).
    """
    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    for i in range(len(eta1)):
        if eta1[i] == -1.0:
            continue  # point not in process domain

        span1 = bsplines_kernels.find_span(tn1, pn[0], eta1[i])
        bsplines_kernels.b_d_splines_slim(tn1, pn[0], eta1[i], span1, bn1, bd1)
        if kind[0] == 0:
            b1 = bn1
        else:
            b1 = bd1

        for j in range(len(eta2)):
            if eta2[j] == -1.0:
                continue  # point not in process domain

            span2 = bsplines_kernels.find_span(tn2, pn[1], eta2[j])
            bsplines_kernels.b_d_splines_slim(tn2, pn[1], eta2[j], span2, bn2, bd2)
            if kind[1] == 0:
                b2 = bn2
            else:
                b2 = bd2

            for k in range(len(eta3)):
                if eta3[k] == -1.0:
                    continue  # point not in process domain

                span3 = bsplines_kernels.find_span(tn3, pn[2], eta3[k])
                bsplines_kernels.b_d_splines_slim(tn3, pn[2], eta3[k], span3, bn3, bd3)
                if kind[2] == 0:
                    b3 = bn3
                else:
                    b3 = bd3

                values[i, j, k] = eval_spline_mpi_kernel(
                    pn[0] - kind[0], pn[1] - kind[1], pn[2] - kind[2], b1, b2, b3, span1, span2, span3, _data, starts
                )


@stack_array("b1", "b2", "b3")
def eval_spline_mpi_tensor_product_fixed(
    span1s: "int[:]",
    span2s: "int[:]",
    span3s: "int[:]",
    b1s: "float[:,:]",
    b2s: "float[:,:]",
    b3s: "float[:,:]",
    _data: "float[:,:,:]",
    kind: "int[:]",
    pn: "int[:]",
    starts: "int[:]",
    values: "float[:,:,:]",
):
    """
    Tensor-product evaluation of a tensor-product spline, distributed,
    and optimized for a fixed grid (spans and spline values have been pre-evaluated).

    Parameters
    ----------
        span1s, span2s, span3s : array[int]
            Knot span indices.

        b1s, b2s, b3s : array[float]
            Values of p+1 non-zero basis functions at evaluation points.

        _data : array[float]
            The spline coefficients c_ijk.

        kind : array[int]
            Kind of 1d basis in each direction: 0 = N-spline, 1 = D-spline.

        pn : array[int]
            Spline degrees of V0 in each direction.

        starts : array[float]
            Start indices of splines on current process.

        values : array[float]
            Return array for spline values S_ijk where ijk are the flattened indices corresponding to spans.
    """

    # allocate spline values
    b1 = empty(pn[0] - kind[0] + 1, dtype=float)
    b2 = empty(pn[1] - kind[1] + 1, dtype=float)
    b3 = empty(pn[2] - kind[2] + 1, dtype=float)

    ni = span1s.size
    nj = span2s.size
    nk = span3s.size

    for i in range(ni):
        span1 = span1s[i]
        b1[:] = b1s[i, :]
        for j in range(nj):
            span2 = span2s[j]
            b2[:] = b2s[j, :]
            for k in range(nk):
                span3 = span3s[k]
                b3[:] = b3s[k, :]

                values[i, j, k] = eval_spline_mpi_kernel(
                    pn[0] - kind[0], pn[1] - kind[1], pn[2] - kind[2], b1, b2, b3, span1, span2, span3, _data, starts
                )


def eval_spline_mpi_matrix(
    eta1: "float[:,:,:]",
    eta2: "float[:,:,:]",
    eta3: "float[:,:,:]",
    _data: "float[:,:,:]",
    kind: "int[:]",
    pn: "int[:]",
    tn1: "float[:]",
    tn2: "float[:]",
    tn3: "float[:]",
    starts: "int[:]",
    values: "float[:,:,:]",
):
    """
    3d array evaluation of a tensor-product spline, distributed.

    Parameters
    ----------
        eta1, eta2, eta3 : array[float]
            Evaluation points as 3d arrays; points not on local process domain must be flagged as -1.
            Spline values are obtained as S_ijk = S(eta1[i,j,k], eta2[i,j,k], eta3[i,j,k]).

        _data : array[float]
            The spline coefficients c_ijk.

        kind : array[int]
            Kind of 1d basis in each direction: 0 = N-spline, 1 = D-spline.

        pn : array[int]
            Spline degrees of V0 in each direction.

        tn1, tn2, tn3 : array[float]
            Knot vectors of V0 in each direction.

        starts : array[float]
            Start indices of splines on current process.

        values : array[float]
            Return array for spline values S_ijk = S(eta1[i,j,k], eta2[i,j,k], eta3[i,j,k]).
    """

    shp = shape(eta1)

    for i in range(shp[0]):
        for j in range(shp[1]):
            for k in range(shp[2]):
                if eta1[i, j, k] == -1.0:
                    continue  # point not in process domain
                if eta2[i, j, k] == -1.0:
                    continue  # point not in process domain
                if eta3[i, j, k] == -1.0:
                    continue  # point not in process domain

                values[i, j, k] = eval_spline_mpi(
                    eta1[i, j, k], eta2[i, j, k], eta3[i, j, k], _data, kind, pn, tn1, tn2, tn3, starts
                )


def eval_spline_mpi_sparse_meshgrid(
    eta1: "float[:,:,:]",
    eta2: "float[:,:,:]",
    eta3: "float[:,:,:]",
    _data: "float[:,:,:]",
    kind: "int[:]",
    pn: "int[:]",
    tn1: "float[:]",
    tn2: "float[:]",
    tn3: "float[:]",
    starts: "int[:]",
    values: "float[:,:,:]",
):
    """
    Sparse meshgrid evaluation of a tensor-product spline, distributed.

    Parameters
    ----------
        eta1, eta2, eta3 : array[float]
            Evaluation points as 3d arrays obtained from sparse meshgrid; points not on local process domain must be flagged as -1.
            Spline values are obtained as S_ijk = S(eta1[i,0,0], eta2[0,j,0], eta3[0,0,k]).

        _data : array[float]
            The spline coefficients c_ijk.

        kind : array[int]
            Kind of 1d basis in each direction: 0 = N-spline, 1 = D-spline.

        pn : array[int]
            Spline degrees of V0 in each direction.

        tn1, tn2, tn3 : array[float]
            Knot vectors of V0 in each direction.

        starts : array[float]
            Start indices of splines on current process.

        values : array[float]
            Return array for spline values S_ijk = S(eta1[i,0,0], eta2[0,j,0], eta3[0,0,k]).
    """

    from numpy import size

    n1 = size(eta1)
    n2 = size(eta2)
    n3 = size(eta3)

    for i in range(n1):
        if eta1[i, 0, 0] == -1.0:
            continue  # point not in process domain
        for j in range(n2):
            if eta2[0, j, 0] == -1.0:
                continue  # point not in process domain
            for k in range(n3):
                if eta3[0, 0, k] == -1.0:
                    continue  # point not in process domain

                values[i, j, k] = eval_spline_mpi(
                    eta1[i, 0, 0], eta2[0, j, 0], eta3[0, 0, k], _data, kind, pn, tn1, tn2, tn3, starts
                )


def eval_spline_mpi_markers(
    markers: "float[:,:]",
    _data: "float[:,:,:]",
    kind: "int[:]",
    pn: "int[:]",
    tn1: "float[:]",
    tn2: "float[:]",
    tn3: "float[:]",
    starts: "int[:]",
    values: "float[:]",
):
    """
    Flat (marker) evaluation of a tensor-product spline, distributed.

    Parameters
    ----------
        markers : array[float]
            Marker coordinates in format [Np, 3]; markers not on local process domain must be flagged as -1.
            Spline values are obtained as S_p = S(*markers[p, :]).

        _data : array[float]
            The spline coefficients c_ijk.

        kind : array[int]
            Kind of 1d basis in each direction: 0 = N-spline, 1 = D-spline.

        pn : array[int]
            Spline degrees of V0 in each direction.

        tn1, tn2, tn3 : array[float]
            Knot vectors of V0 in each direction.

        starts : array[float]
            Start indices of splines on current process.

        values : array[float]
            Return 1D array for spline values S_p = S(*markers[p, :]).
    """

    Np = shape(markers)[0]

    for ip in range(Np):
        if markers[ip, 0] == -1:
            continue  # point not in process domain

        values[ip] = eval_spline_mpi(
            markers[ip, 0], markers[ip, 1], markers[ip, 2], _data, kind, pn, tn1, tn2, tn3, starts
        )


###########################################
### Distributed Derham field evaluation ###
###########################################


def get_spans(eta1: float, eta2: float, eta3: float, args_derham: "DerhamArguments"):
    """Compute the knot span index,
    the N-spline values (in bn) and the D-spline values (in bd)
    at (eta1, eta2, eta3)."""

    # find spans
    span1 = bsplines_kernels.find_span(args_derham.tn1, args_derham.pn[0], eta1)
    span2 = bsplines_kernels.find_span(args_derham.tn2, args_derham.pn[1], eta2)
    span3 = bsplines_kernels.find_span(args_derham.tn3, args_derham.pn[2], eta3)

    # get spline values at eta
    bsplines_kernels.b_d_splines_slim(
        args_derham.tn1, args_derham.pn[0], eta1, int(span1), args_derham.bn1, args_derham.bd1
    )
    bsplines_kernels.b_d_splines_slim(
        args_derham.tn2, args_derham.pn[1], eta2, int(span2), args_derham.bn2, args_derham.bd2
    )
    bsplines_kernels.b_d_splines_slim(
        args_derham.tn3, args_derham.pn[2], eta3, int(span3), args_derham.bn3, args_derham.bd3
    )

    return span1, span2, span3


def eval_0form_spline_mpi(
    span1: int, span2: int, span3: int, args_derham: "DerhamArguments", form_coeffs: "float[:,:,:]"
) -> float:
    """Single-point evaluation of Derham 0-form spline defined by form_coeffs,
    given N-spline values (in bn) and knot span indices span."""

    out = eval_spline_mpi_kernel(
        args_derham.pn[0],
        args_derham.pn[1],
        args_derham.pn[2],
        args_derham.bn1,
        args_derham.bn2,
        args_derham.bn3,
        span1,
        span2,
        span3,
        form_coeffs,
        args_derham.starts,
    )
    return out


def eval_1form_spline_mpi(
    span1: int,
    span2: int,
    span3: int,
    args_derham: "DerhamArguments",
    form_coeffs_1: "float[:,:,:]",
    form_coeffs_2: "float[:,:,:]",
    form_coeffs_3: "float[:,:,:]",
    out: "float[:]",
):
    """Single-point evaluation of Derham 1-form spline defined by form_coeffs,
    given N-spline values (in bn), D-spline values (in bd)
    and knot span indices span."""

    out[0] = eval_spline_mpi_kernel(
        args_derham.pn[0] - 1,
        args_derham.pn[1],
        args_derham.pn[2],
        args_derham.bd1,
        args_derham.bn2,
        args_derham.bn3,
        span1,
        span2,
        span3,
        form_coeffs_1,
        args_derham.starts,
    )

    out[1] = eval_spline_mpi_kernel(
        args_derham.pn[0],
        args_derham.pn[1] - 1,
        args_derham.pn[2],
        args_derham.bn1,
        args_derham.bd2,
        args_derham.bn3,
        span1,
        span2,
        span3,
        form_coeffs_2,
        args_derham.starts,
    )

    out[2] = eval_spline_mpi_kernel(
        args_derham.pn[0],
        args_derham.pn[1],
        args_derham.pn[2] - 1,
        args_derham.bn1,
        args_derham.bn2,
        args_derham.bd3,
        span1,
        span2,
        span3,
        form_coeffs_3,
        args_derham.starts,
    )


def eval_2form_spline_mpi(
    span1: int,
    span2: int,
    span3: int,
    args_derham: "DerhamArguments",
    form_coeffs_1: "float[:,:,:]",
    form_coeffs_2: "float[:,:,:]",
    form_coeffs_3: "float[:,:,:]",
    out: "float[:]",
):
    """Single-point evaluation of Derham 2-form spline defined by form_coeffs,
    given N-spline values (in bn), D-spline values (in bd)
    and knot span indices span."""

    out[0] = eval_spline_mpi_kernel(
        args_derham.pn[0],
        args_derham.pn[1] - 1,
        args_derham.pn[2] - 1,
        args_derham.bn1,
        args_derham.bd2,
        args_derham.bd3,
        span1,
        span2,
        span3,
        form_coeffs_1,
        args_derham.starts,
    )

    out[1] = eval_spline_mpi_kernel(
        args_derham.pn[0] - 1,
        args_derham.pn[1],
        args_derham.pn[2] - 1,
        args_derham.bd1,
        args_derham.bn2,
        args_derham.bd3,
        span1,
        span2,
        span3,
        form_coeffs_2,
        args_derham.starts,
    )

    out[2] = eval_spline_mpi_kernel(
        args_derham.pn[0] - 1,
        args_derham.pn[1] - 1,
        args_derham.pn[2],
        args_derham.bd1,
        args_derham.bd2,
        args_derham.bn3,
        span1,
        span2,
        span3,
        form_coeffs_3,
        args_derham.starts,
    )


def eval_3form_spline_mpi(
    span1: int, span2: int, span3: int, args_derham: "DerhamArguments", form_coeffs: "float[:,:,:]"
) -> float:
    """Single-point evaluation of Derham 0-form spline defined by form_coeffs,
    given D-spline values (in bd) and knot span indices span."""

    out = eval_spline_mpi_kernel(
        args_derham.pn[0] - 1,
        args_derham.pn[1] - 1,
        args_derham.pn[2] - 1,
        args_derham.bd1,
        args_derham.bd2,
        args_derham.bd3,
        span1,
        span2,
        span3,
        form_coeffs,
        args_derham.starts,
    )
    return out


def eval_vectorfield_spline_mpi(
    span1: int,
    span2: int,
    span3: int,
    args_derham: "DerhamArguments",
    form_coeffs_1: "float[:,:,:]",
    form_coeffs_2: "float[:,:,:]",
    form_coeffs_3: "float[:,:,:]",
    out: "float[:]",
):
    """Single-point evaluation of vector-field spline (H^1)^3 defined by form_coeffs,
    given N-spline values (in bn) and knot span indices span."""

    out[0] = eval_spline_mpi_kernel(
        args_derham.pn[0],
        args_derham.pn[1],
        args_derham.pn[2],
        args_derham.bn1,
        args_derham.bn2,
        args_derham.bn3,
        span1,
        span2,
        span3,
        form_coeffs_1,
        args_derham.starts,
    )

    out[1] = eval_spline_mpi_kernel(
        args_derham.pn[0],
        args_derham.pn[1],
        args_derham.pn[2],
        args_derham.bn1,
        args_derham.bn2,
        args_derham.bn3,
        span1,
        span2,
        span3,
        form_coeffs_2,
        args_derham.starts,
    )

    out[2] = eval_spline_mpi_kernel(
        args_derham.pn[0],
        args_derham.pn[1],
        args_derham.pn[2],
        args_derham.bn1,
        args_derham.bn2,
        args_derham.bn3,
        span1,
        span2,
        span3,
        form_coeffs_3,
        args_derham.starts,
    )
