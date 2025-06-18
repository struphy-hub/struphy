# coding: utf-8
#
# Copyright 2020 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Acccelerated functions for point-wise evaluation of tensor product B-splines.

S(eta1, eta2) = sum_ij [ c_ij * B_i(eta1) * B_j(eta2) ] with c_ij in R.

Possible combinations for tensor product (BB):
* (NN)
* (DN)
* (ND)
* (DD)
* (dN/deta N)
* (N dN/deta)
"""
from pyccel.decorators import pure, stack_array

from numpy import empty, zeros

import struphy.bsplines.bsplines_kernels as bsplines_kernels


@pure
def evaluation_kernel_2d(p1: int, p2: int, basis1: 'float[:]', basis2: 'float[:]', ind1: 'int[:]', ind2: 'int[:]', coeff: 'float[:,:]') -> float:
    """
    Summing non-zero contributions of a spline function.

    Parameters
    ----------
        p1, p2 : int                 
            Degrees of the univariate splines in each direction.

        basis1, basis2 : array[float]           
            The p + 1 values of non-zero basis splines at one point (eta1, eta2) in each direction.

        ind1, ind2 : array[int]                 
            Global indices of non-vanishing splines in the element of the considered point.

        coeff : array[float]
            The spline coefficients c_ij. 

    Returns
    -------
        spline_value : float
            Value of tensor-product spline at point (eta1, eta2).
    """

    spline_value = 0.

    for il1 in range(p1 + 1):
        i1 = ind1[il1]
        for il2 in range(p2 + 1):
            i2 = ind2[il2]

            spline_value += coeff[i1, i2] * basis1[il1] * basis2[il2]

    return spline_value


@pure
@stack_array('tmp1', 'tmp2', 'tmp3', 'tmp4')
def evaluate_2d(kind1: int, kind2: int, t1: 'float[:]', t2: 'float[:]', p1: int, p2: int, ind1: 'int[:,:]', ind2: 'int[:,:]', coeff: 'float[:,:]', eta1: float, eta2: float) -> float:
    """
    Point-wise evaluation of a tensor-product spline. 

    Parameters
    ----------
        kind1, kind2 : int
            Kind of univariate spline. 1 for B-spline, 2 for M-spline and 3 for derivative of B-spline.

        t1, t2 : array[float]
            Knot vectors of univariate splines.

        p1, p2 : int                 
            Degrees of univariate splines.

        ind1, ind2 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).

        coeff : array[float]
            The spline coefficients c_ij. 

        eta1, eta2 : float              
            Point of evaluation.

    Returns
    -------
        spline_value: float
            Value of tensor-product spline at point (eta1, eta2).
    """

    # find knot span indices
    span1 = bsplines_kernels.find_span(t1, p1, eta1)
    span2 = bsplines_kernels.find_span(t2, p2, eta2)

    # evaluate non-vanishing basis functions
    b1 = empty(p1 + 1, dtype=float)
    b2 = empty(p2 + 1, dtype=float)

    bl1 = empty(p1, dtype=float)
    bl2 = empty(p2, dtype=float)

    br1 = empty(p1, dtype=float)
    br2 = empty(p2, dtype=float)

    tmp1 = zeros(p1 + 1, dtype=int)
    tmp2 = zeros(p2 + 1, dtype=int)

    # 1st direction
    if kind1 == 1:
        bsplines_kernels.basis_funs(t1, p1, eta1, span1, bl1, br1, b1)
    elif kind1 == 2:
        bsplines_kernels.basis_funs(t1, p1, eta1, span1, bl1, br1, b1)
        bsplines_kernels.scaling(t1, p1, span1, b1)
    elif kind1 == 3:
        bsplines_kernels.basis_funs_1st_der(t1, p1, eta1, span1, bl1, br1, b1)
    elif kind1 == 4:
        tmp3 = zeros((3, p1 + 1), dtype=float)
        bsplines_kernels.basis_funs_all_ders(
            t1, p1, eta1, span1, bl1, br1, 2, tmp3)
        b1[:] = tmp3[2, :]

    # 2nd direction
    if kind2 == 1:
        bsplines_kernels.basis_funs(t2, p2, eta2, span2, bl2, br2, b2)
    elif kind2 == 2:
        bsplines_kernels.basis_funs(t2, p2, eta2, span2, bl2, br2, b2)
        bsplines_kernels.scaling(t2, p2, span2, b2)
    elif kind2 == 3:
        bsplines_kernels.basis_funs_1st_der(t2, p2, eta2, span2, bl2, br2, b2)
    elif kind2 == 4:
        tmp4 = zeros((3, p2 + 1), dtype=float)
        bsplines_kernels.basis_funs_all_ders(
            t2, p2, eta2, span2, bl2, br2, 2, tmp4)
        b2[:] = tmp4[2, :]

    # sum up non-vanishing contributions
    tmp1[:] = ind1[span1 - p1, :]
    tmp2[:] = ind2[span2 - p2, :]
    spline_value = evaluation_kernel_2d(p1, p2, b1, b2, tmp1, tmp2, coeff)

    return spline_value


@pure
def evaluate_tensor_product_2d(t1: 'float[:]', t2: 'float[:]', p1: int, p2: int, ind1: 'int[:,:]', ind2: 'int[:,:]', coeff: 'float[:,:]', eta1: 'float[:]', eta2: 'float[:]', spline_values: 'float[:,:]', kind: int):
    """
    Tensor-product evaluation of a tensor-product spline. 

    Parameters
    ----------
        t1, t2 : array[float]
            Knot vectors of univariate splines.

        p1, p2 : int                 
            Degrees of univariate splines.

        ind1, ind2 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).

        coeff : array[float]
            The spline coefficients c_ij. 

        eta1, eta2 : array[float]              
            Points of evaluation in 1d arrays.

        spline_values : array[float]
            Splines evaluated at points S_ij = S(eta1_i, eta2_j).

        kind : int
            Kind of spline to evaluate.
                * 0  : NN
                * 11 : DN
                * 12 : ND
                * 2  : DD
                * 31 : dN/deta N
                * 32 : N dN/deta
                * 41 : ddN/deta^2 N
                * 42 : n ddN/deta^3
                * 43 : dN/deta dN/deta
    """

    for i1 in range(len(eta1)):
        for i2 in range(len(eta2)):

            if kind == 0:
                spline_values[i1, i2] = evaluate_2d(
                    1, 1, t1, t2, p1, p2, ind1, ind2, coeff, eta1[i1], eta2[i2])
            elif kind == 11:
                spline_values[i1, i2] = evaluate_2d(
                    2, 1, t1, t2, p1, p2, ind1, ind2, coeff, eta1[i1], eta2[i2])
            elif kind == 12:
                spline_values[i1, i2] = evaluate_2d(
                    1, 2, t1, t2, p1, p2, ind1, ind2, coeff, eta1[i1], eta2[i2])
            elif kind == 2:
                spline_values[i1, i2] = evaluate_2d(
                    2, 2, t1, t2, p1, p2, ind1, ind2, coeff, eta1[i1], eta2[i2])
            elif kind == 31:
                spline_values[i1, i2] = evaluate_2d(
                    3, 1, t1, t2, p1, p2, ind1, ind2, coeff, eta1[i1], eta2[i2])
            elif kind == 32:
                spline_values[i1, i2] = evaluate_2d(
                    1, 3, t1, t2, p1, p2, ind1, ind2, coeff, eta1[i1], eta2[i2])
            elif kind == 41:
                spline_values[i1, i2] = evaluate_2d(
                    4, 1, t1, t2, p1, p2, ind1, ind2, coeff, eta1[i1], eta2[i2])
            elif kind == 42:
                spline_values[i1, i2] = evaluate_2d(
                    1, 4, t1, t2, p1, p2, ind1, ind2, coeff, eta1[i1], eta2[i2])
            elif kind == 43:
                spline_values[i1, i2] = evaluate_2d(
                    3, 3, t1, t2, p1, p2, ind1, ind2, coeff, eta1[i1], eta2[i2])


@pure
def evaluate_matrix_2d(t1: 'float[:]', t2: 'float[:]', p1: int, p2: int, ind1: 'int[:,:]', ind2: 'int[:,:]', coeff: 'float[:,:]', eta1: 'float[:,:]', eta2: 'float[:,:]', spline_values: 'float[:,:]', kind: int):
    """
    General evaluation of a tensor-product spline. 

    Parameters
    ----------
        t1, t2 : array[float]
            Knot vectors of univariate splines.

        p1, p2 : int                 
            Degrees of univariate splines.

        ind1, ind2 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).

        coeff : array[float]
            The spline coefficients c_ij. 

        eta1, eta2 : array[float]              
            Points of evaluation in 3d arrays such that shape(eta1) == shape(eta2).

        spline_values : array[float]
            Splines evaluated at points S_ij = S(eta1_i, eta2_j).

        kind : int
            Kind of spline to evaluate.
                * 0  : NN
                * 11 : DN
                * 12 : ND
                * 2  : DD
                * 31 : dN/deta N
                * 32 : N dN/deta
                * 41 : ddN/deta^2 N
                * 42 : n ddN/deta^3
                * 43 : dN/deta dN/deta
    """

    from numpy import shape

    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]

    for i1 in range(n1):
        for i2 in range(n2):

            if kind == 0:
                spline_values[i1, i2] = evaluate_2d(
                    1, 1, t1, t2, p1, p2, ind1, ind2, coeff, eta1[i1, i2], eta2[i1, i2])
            elif kind == 11:
                spline_values[i1, i2] = evaluate_2d(
                    2, 1, t1, t2, p1, p2, ind1, ind2, coeff, eta1[i1, i2], eta2[i1, i2])
            elif kind == 12:
                spline_values[i1, i2] = evaluate_2d(
                    1, 2, t1, t2, p1, p2, ind1, ind2, coeff, eta1[i1, i2], eta2[i1, i2])
            elif kind == 2:
                spline_values[i1, i2] = evaluate_2d(
                    2, 2, t1, t2, p1, p2, ind1, ind2, coeff, eta1[i1, i2], eta2[i1, i2])
            elif kind == 31:
                spline_values[i1, i2] = evaluate_2d(
                    3, 1, t1, t2, p1, p2, ind1, ind2, coeff, eta1[i1, i2], eta2[i1, i2])
            elif kind == 32:
                spline_values[i1, i2] = evaluate_2d(
                    1, 3, t1, t2, p1, p2, ind1, ind2, coeff, eta1[i1, i2], eta2[i1, i2])
            elif kind == 41:
                spline_values[i1, i2] = evaluate_2d(
                    4, 1, t1, t2, p1, p2, ind1, ind2, coeff, eta1[i1, i2], eta2[i1, i2])
            elif kind == 42:
                spline_values[i1, i2] = evaluate_2d(
                    1, 4, t1, t2, p1, p2, ind1, ind2, coeff, eta1[i1, i2], eta2[i1, i2])
            elif kind == 43:
                spline_values[i1, i2] = evaluate_2d(
                    3, 3, t1, t2, p1, p2, ind1, ind2, coeff, eta1[i1, i2], eta2[i1, i2])


@pure
def evaluate_sparse_2d(t1: 'float[:]', t2: 'float[:]', p1: int, p2: int, ind1: 'int[:,:]', ind2: 'int[:,:]', coeff: 'float[:,:]', eta1: 'float[:,:]', eta2: 'float[:,:]', spline_values: 'float[:,:]', kind: int):
    """
    Evaluation of a tensor-product spline using sparse meshgrids. 

    Parameters
    ----------
        t1, t2 : array[float]
            Knot vectors of univariate splines.

        p1, p2 : int                 
            Degrees of univariate splines.

        ind1, ind2 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).

        coeff : array[float]
            The spline coefficients c_ij. 

        eta1, eta2 : array[float]              
            Points of evaluation in 3d arrays such that shape(eta1) = (:,1), shape(eta2) = (1,:).

        spline_values : array[float]
            Splines evaluated at points S_ij = S(eta1_i, eta2_j).

        kind : int
            Kind of spline to evaluate.
                * 0  : NN
                * 11 : DN
                * 12 : ND
                * 2  : DD
                * 31 : dN/deta N
                * 32 : N dN/deta
                * 41 : ddN/deta^2 N
                * 42 : n ddN/deta^3
                * 43 : dN/deta dN/deta
    """

    from numpy import shape

    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]

    for i1 in range(n1):
        for i2 in range(n2):

            if kind == 0:
                spline_values[i1, i2] = evaluate_2d(
                    1, 1, t1, t2, p1, p2, ind1, ind2, coeff, eta1[i1, 0], eta2[0, i2])
            elif kind == 11:
                spline_values[i1, i2] = evaluate_2d(
                    2, 1, t1, t2, p1, p2, ind1, ind2, coeff, eta1[i1, 0], eta2[0, i2])
            elif kind == 12:
                spline_values[i1, i2] = evaluate_2d(
                    1, 2, t1, t2, p1, p2, ind1, ind2, coeff, eta1[i1, 0], eta2[0, i2])
            elif kind == 2:
                spline_values[i1, i2] = evaluate_2d(
                    2, 2, t1, t2, p1, p2, ind1, ind2, coeff, eta1[i1, 0], eta2[0, i2])
            elif kind == 31:
                spline_values[i1, i2] = evaluate_2d(
                    3, 1, t1, t2, p1, p2, ind1, ind2, coeff, eta1[i1, 0], eta2[0, i2])
            elif kind == 32:
                spline_values[i1, i2] = evaluate_2d(
                    1, 3, t1, t2, p1, p2, ind1, ind2, coeff, eta1[i1, 0], eta2[0, i2])
            elif kind == 41:
                spline_values[i1, i2] = evaluate_2d(
                    4, 1, t1, t2, p1, p2, ind1, ind2, coeff, eta1[i1, 0], eta2[0, i2])
            elif kind == 42:
                spline_values[i1, i2] = evaluate_2d(
                    1, 4, t1, t2, p1, p2, ind1, ind2, coeff, eta1[i1, 0], eta2[0, i2])
            elif kind == 43:
                spline_values[i1, i2] = evaluate_2d(
                    3, 3, t1, t2, p1, p2, ind1, ind2, coeff, eta1[i1, 0], eta2[0, i2])


@pure
def eval_spline_mpi_2d(p1: 'int', p2: 'int', basis1: 'float[:]', basis2: 'float[:]', span1: 'int', span2: 'int', coeff: 'float[:,:]', starts: 'int[:]', pn: 'int[:]') -> float:
    """
    Evaluate a spline function on the current process.

    Parameters
    ----------
        p1, p2 : int                 
            Degrees of the univariate splines in each direction.

        basis1, basis2 : array[float]           
            The p + 1 values of non-zero basis splines at one point (eta1, eta2) in each direction.

        span1, span2: int
            Particle's element index in each direction.

        coeff : array[float]
            The spline coefficients c_ij of the current process, ie. the _data attribute of a StencilVector.  

        starts : array[int]
            Starting indices of current process.

        pn : array[int]
            B-spline degrees in each direction (=paddings).

    Returns
    -------
        spline_value : float
            Value of tensor-product spline at point (eta1, eta2).
    """

    spline_value = 0.

    for il1 in range(p1 + 1):
        i1 = span1 + il1 - starts[0]  # span1 = ie1 + pn[0]
        for il2 in range(p2 + 1):
            i2 = span2 + il2 - starts[1]

            spline_value += coeff[i1, i2] * basis1[il1] * basis2[il2]

    return spline_value
