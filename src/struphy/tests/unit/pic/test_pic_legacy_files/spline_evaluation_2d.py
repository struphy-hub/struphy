# coding: utf-8


"""
Acccelerated functions for point-wise evaluation of tensor product B-splines.

S(eta1, eta2) = sum_ij c_ij * B_i(eta1) * B_j(eta2)     with c_ij in R.

Possible combinations for tensor product (BB):
(NN)
(dN/deta N)
(N dN/deta)
(DN)
(ND)
(DD)
"""

from numpy import empty

import struphy.bsplines.bsplines_kernels as bsp


# =============================================================================
def evaluation_kernel_2d(
    p1: "int",
    p2: "int",
    basis1: "float[:]",
    basis2: "float[:]",
    span1: "int",
    span2: "int",
    nbase1: "int",
    nbase2: "int",
    coeff: "float[:,:]",
):
    """Summing non-zero contributions.

    Parameters:
    -----------
        p1, p2:         int             spline degrees
        basis1, basis2: double[:]       pn+1 values of non-zero basis splines at one point eta_n from 'basis_funs' (n=1,2)
        span1, span2:   int             knot span indices from 'find_span'
        nbase1, nbase2: int             dimensions of spline spaces
        coeff:          double[:, :]    spline coefficients c_ij

    Returns:
    --------
    value: float
        Value of B-spline at point (eta1, eta2).
    """

    value = 0.0

    for il1 in range(p1 + 1):
        i1 = (span1 - il1) % nbase1
        for il2 in range(p2 + 1):
            i2 = (span2 - il2) % nbase2

            value += coeff[i1, i2] * basis1[p1 - il1] * basis2[p2 - il2]

    return value


# =============================================================================
def evaluate_n_n(
    tn1: "float[:]",
    tn2: "float[:]",
    pn1: "int",
    pn2: "int",
    nbase_n1: "int",
    nbase_n2: "int",
    coeff: "float[:,:]",
    eta1: "float",
    eta2: "float",
):
    """Point-wise evaluation of (NN)-tensor-product spline.

    Parameters:
    -----------
        tn1, tn2:           double[:]       knot vectors
        pn1, pn2:           int             spline degrees
        nbase_n1, nbase_n2: int             dimensions of univariate spline spaces
        coeff:              double[:, :]    spline coefficients c_ij
        eta1, eta2:         double          point of evaluation

    Returns:
    --------
        value: float
            Value of (NN)-tensor-product spline at point (eta1, eta2).
    """

    # find knot span indices
    span_n1 = bsp.find_span(tn1, pn1, eta1)
    span_n2 = bsp.find_span(tn2, pn2, eta2)

    # evaluate non-vanishing basis functions
    bn1 = empty(pn1 + 1, dtype=float)
    bn2 = empty(pn2 + 1, dtype=float)

    bl1 = empty(pn1, dtype=float)
    bl2 = empty(pn2, dtype=float)

    br1 = empty(pn1, dtype=float)
    br2 = empty(pn2, dtype=float)

    bsp.basis_funs(tn1, pn1, eta1, span_n1, bl1, br1, bn1)
    bsp.basis_funs(tn2, pn2, eta2, span_n2, bl2, br2, bn2)

    # sum up non-vanishing contributions
    value = evaluation_kernel_2d(pn1, pn2, bn1, bn2, span_n1, span_n2, nbase_n1, nbase_n2, coeff)

    return value


# =============================================================================
def evaluate_diffn_n(
    tn1: "float[:]",
    tn2: "float[:]",
    pn1: "int",
    pn2: "int",
    nbase_n1: "int",
    nbase_n2: "int",
    coeff: "float[:,:]",
    eta1: "float",
    eta2: "float",
):
    """Point-wise evaluation of (dN/deta N)-tensor-product spline.

    Parameters:
    -----------
        tn1, tn2:           double[:]       knot vectors
        pn1, pn2:           int             spline degrees
        nbase_n1, nbase_n2: int             dimensions of spline spaces
        coeff:              double[:, :]    spline coefficients c_ij
        eta1, eta2:         double          point of evaluation

    Returns:
    --------
        value: float
            Value of (dN/deta N)-tensor-product spline at point (eta1, eta2).
    """

    # find knot span indices
    span_n1 = bsp.find_span(tn1, pn1, eta1)
    span_n2 = bsp.find_span(tn2, pn2, eta2)

    # evaluate non-vanishing basis functions
    bn1 = empty(pn1 + 1, dtype=float)
    bn2 = empty(pn2 + 1, dtype=float)

    bl1 = empty(pn1, dtype=float)
    bl2 = empty(pn2, dtype=float)

    br1 = empty(pn1, dtype=float)
    br2 = empty(pn2, dtype=float)

    bsp.basis_funs_1st_der(tn1, pn1, eta1, span_n1, bl1, br1, bn1)
    bsp.basis_funs(tn2, pn2, eta2, span_n2, bl2, br2, bn2)

    # sum up non-vanishing contributions
    value = evaluation_kernel_2d(pn1, pn2, bn1, bn2, span_n1, span_n2, nbase_n1, nbase_n2, coeff)

    return value


# =============================================================================
def evaluate_n_diffn(
    tn1: "float[:]",
    tn2: "float[:]",
    pn1: "int",
    pn2: "int",
    nbase_n1: "int",
    nbase_n2: "int",
    coeff: "float[:,:]",
    eta1: "float",
    eta2: "float",
):
    """Point-wise evaluation of (N dN/deta)-tensor-product spline.

    Parameters:
    -----------
        tn1, tn2:           double[:]       knot vectors
        pn1, pn2:           int             spline degrees
        nbase_n1, nbase_n2: int             dimensions of spline spaces
        coeff:              double[:, :]    spline coefficients c_ij
        eta1, eta2:         double          point of evaluation

    Returns:
    --------
        value: float
            Value of (N dN/deta)-tensor-product spline at point (eta1, eta2).
    """

    # find knot span indices
    span_n1 = bsp.find_span(tn1, pn1, eta1)
    span_n2 = bsp.find_span(tn2, pn2, eta2)

    # evaluate non-vanishing basis functions
    bn1 = empty(pn1 + 1, dtype=float)
    bn2 = empty(pn2 + 1, dtype=float)

    bl1 = empty(pn1, dtype=float)
    bl2 = empty(pn2, dtype=float)

    br1 = empty(pn1, dtype=float)
    br2 = empty(pn2, dtype=float)

    bsp.basis_funs(tn1, pn1, eta1, span_n1, bl1, br1, bn1)
    bsp.basis_funs_1st_der(tn2, pn2, eta2, span_n2, bl2, br2, bn2)

    # sum up non-vanishing contributions
    value = evaluation_kernel_2d(pn1, pn2, bn1, bn2, span_n1, span_n2, nbase_n1, nbase_n2, coeff)

    return value


# =============================================================================
def evaluate_d_n(
    td1: "float[:]",
    tn2: "float[:]",
    pd1: "int",
    pn2: "int",
    nbase_d1: "int",
    nbase_n2: "int",
    coeff: "float[:,:]",
    eta1: "float",
    eta2: "float",
):
    """Point-wise evaluation of (DN)-tensor-product spline.

    Parameters:
    -----------
        td1, tn2:           double[:]       knot vectors
        pd1, pn2:           int             spline degrees
        nbase_d1, nbase_n2: int             dimensions of spline spaces
        coeff:              double[:, :]    spline coefficients c_ij
        eta1, eta2:         double          point of evaluation

    Returns:
    --------
        value: float
            Value of (DN)-tensor-product spline at point (eta1, eta2).
    """

    # find knot span indices
    span_d1 = bsp.find_span(td1, pd1, eta1)
    span_n2 = bsp.find_span(tn2, pn2, eta2)

    # evaluate non-vanishing basis functions
    bd1 = empty(pd1 + 1, dtype=float)
    bn2 = empty(pn2 + 1, dtype=float)

    bl1 = empty(pd1, dtype=float)
    bl2 = empty(pn2, dtype=float)

    br1 = empty(pd1, dtype=float)
    br2 = empty(pn2, dtype=float)

    bsp.basis_funs(td1, pd1, eta1, span_d1, bl1, br1, bd1)
    bsp.basis_funs(tn2, pn2, eta2, span_n2, bl2, br2, bn2)

    bsp.scaling(td1, pd1, span_d1, bd1)

    # sum up non-vanishing contributions
    value = evaluation_kernel_2d(pd1, pn2, bd1, bn2, span_d1, span_n2, nbase_d1, nbase_n2, coeff)

    return value


# =============================================================================
def evaluate_n_d(
    tn1: "float[:]",
    td2: "float[:]",
    pn1: "int",
    pd2: "int",
    nbase_n1: "int",
    nbase_d2: "int",
    coeff: "float[:,:]",
    eta1: "float",
    eta2: "float",
):
    """Point-wise evaluation of (ND)-tensor-product spline.

    Parameters:
    -----------
        tn1, td2:           double[:]       knot vectors
        pn1, pd2:           int             spline degrees
        nbase_n1, nbase_d2: int             dimensions of spline spaces
        coeff:              double[:, :]    spline coefficients c_ij
        eta1, eta2:         double          point of evaluation

    Returns:
    --------
        value: float
            Value of (ND)-tensor-product spline at point (eta1, eta2).
    """

    # find knot span indices
    span_n1 = bsp.find_span(tn1, pn1, eta1)
    span_d2 = bsp.find_span(td2, pd2, eta2)

    # evaluate non-vanishing basis functions
    bn1 = empty(pn1 + 1, dtype=float)
    bd2 = empty(pd2 + 1, dtype=float)

    bl1 = empty(pn1, dtype=float)
    bl2 = empty(pd2, dtype=float)

    br1 = empty(pn1, dtype=float)
    br2 = empty(pd2, dtype=float)

    bsp.basis_funs(tn1, pn1, eta1, span_n1, bl1, br1, bn1)
    bsp.basis_funs(td2, pd2, eta2, span_d2, bl2, br2, bd2)

    bsp.scaling(td2, pd2, span_d2, bd2)

    # sum up non-vanishing contributions
    value = evaluation_kernel_2d(pn1, pd2, bn1, bd2, span_n1, span_d2, nbase_n1, nbase_d2, coeff)

    return value


# =============================================================================
def evaluate_d_d(
    td1: "float[:]",
    td2: "float[:]",
    pd1: "int",
    pd2: "int",
    nbase_d1: "int",
    nbase_d2: "int",
    coeff: "float[:,:]",
    eta1: "float",
    eta2: "float",
):
    """Point-wise evaluation of (DD)-tensor-product spline.

    Parameters:
    -----------
        td1, td2:           double[:]       knot vectors
        pd1, pd2:           int             spline degrees
        nbase_d1, nbase_d2: int             dimensions of spline spaces
        coeff:              double[:, :]    spline coefficients c_ij
        eta1, eta2:         double          point of evaluation

    Returns:
    --------
        value: float
            Value of (DD)-tensor-product spline at point (eta1, eta2).
    """

    # find knot span indices
    span_d1 = bsp.find_span(td1, pd1, eta1)
    span_d2 = bsp.find_span(td2, pd2, eta2)

    # evaluate non-vanishing basis functions
    bd1 = empty(pd1 + 1, dtype=float)
    bd2 = empty(pd2 + 1, dtype=float)

    bl1 = empty(pd1, dtype=float)
    bl2 = empty(pd2, dtype=float)

    br1 = empty(pd1, dtype=float)
    br2 = empty(pd2, dtype=float)

    bsp.basis_funs(td1, pd1, eta1, span_d1, bl1, br1, bd1)
    bsp.basis_funs(td2, pd2, eta2, span_d2, bl2, br2, bd2)

    bsp.scaling(td1, pd1, span_d1, bd1)
    bsp.scaling(td2, pd2, span_d2, bd2)

    # sum up non-vanishing contributions
    value = evaluation_kernel_2d(pd1, pd2, bd1, bd2, span_d1, span_d2, nbase_d1, nbase_d2, coeff)

    return value


# =============================================================================
def evaluate_tensor_product(
    t1: "float[:]",
    t2: "float[:]",
    p1: "int",
    p2: "int",
    nbase_1: "int",
    nbase_2: "int",
    coeff: "float[:,:]",
    eta1: "float[:]",
    eta2: "float[:]",
    values: "float[:,:]",
    kind: "int",
):
    """Tensor product evaluation (meshgrid) of tensor product splines (2d).

    Parameters:
    -----------
        t1, t2:             double[:]       knot vectors
        p1, p2:             int             spline degrees
        nbase_1, nbase_2:   int             dimensions of univariate spline spaces
        coeff:              double[:, :]    spline coefficients c_ij
        eta1, eta2:         double[:]       1d arrays of points of evaluation in respective direction
        kind:               int             which tensor product spline, 0: (NN), 11: (DN), 12: (ND), 2: (DD)

    Returns:
    --------
        values:             double[:, :]    values of spline at points from xp.meshgrid(eta1, eta2, indexing='ij').
    """

    for i1 in range(len(eta1)):
        for i2 in range(len(eta2)):
            # V0 - space
            if kind == 0:
                values[i1, i2] = evaluate_n_n(t1, t2, p1, p2, nbase_1, nbase_2, coeff, eta1[i1], eta2[i2])

            # V1 - space
            elif kind == 11:
                values[i1, i2] = evaluate_d_n(t1, t2, p1, p2, nbase_1, nbase_2, coeff, eta1[i1], eta2[i2])
            elif kind == 12:
                values[i1, i2] = evaluate_n_d(t1, t2, p1, p2, nbase_1, nbase_2, coeff, eta1[i1], eta2[i2])

            # V2 - space
            elif kind == 2:
                values[i1, i2] = evaluate_d_d(t1, t2, p1, p2, nbase_1, nbase_2, coeff, eta1[i1], eta2[i2])


# =============================================================================
def evaluate_matrix(
    t1: "float[:]",
    t2: "float[:]",
    p1: "int",
    p2: "int",
    nbase_1: "int",
    nbase_2: "int",
    coeff: "float[:,:]",
    eta1: "float[:,:]",
    eta2: "float[:,:]",
    n1: "int",
    n2: "int",
    values: "float[:,:]",
    kind: "int",
):
    """Matrix evaluation of tensor product splines (2d).

    Parameters:
    -----------
        t1, t2:             double[:]       knot vectors
        p1, p2:             int             spline degrees
        nbase_1, nbase_2:   int             dimensions of univariate spline spaces
        coeff:              double[:, :]    spline coefficients c_ij
        eta1, eta2:         double[:, :]    points of evaluation
        n1, n2:             int             eta1.shape = (n1, n2)
        kind:               int             which tensor product spline, 0: (NN), 11: (DN), 12: (ND), 2: (DD)

    Returns:
    --------
        values:             double[:, :]    values of spline at points (eta1, eta2).
    """

    for i1 in range(n1):
        for i2 in range(n2):
            # V0 - space
            if kind == 0:
                values[i1, i2] = evaluate_n_n(t1, t2, p1, p2, nbase_1, nbase_2, coeff, eta1[i1, i2], eta2[i1, i2])

            # V1 - space
            elif kind == 11:
                values[i1, i2] = evaluate_d_n(t1, t2, p1, p2, nbase_1, nbase_2, coeff, eta1[i1, i2], eta2[i1, i2])
            elif kind == 12:
                values[i1, i2] = evaluate_n_d(t1, t2, p1, p2, nbase_1, nbase_2, coeff, eta1[i1, i2], eta2[i1, i2])

            # V3 - space
            elif kind == 2:
                values[i1, i2] = evaluate_d_d(t1, t2, p1, p2, nbase_1, nbase_2, coeff, eta1[i1, i2], eta2[i1, i2])
