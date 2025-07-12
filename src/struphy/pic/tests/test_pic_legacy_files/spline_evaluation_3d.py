# coding: utf-8


"""
Acccelerated functions for point-wise evaluation of tensor product B-splines.

S(eta1, eta2, eta3) = sum_ijk c_ijk * B_i(eta1) * B_j(eta2) * B_k(eta3)     with c_ijk in R.

Possible combinations for tensor product (BBB):
(NNN)
(dN/deta NN)
(N dN/deta N)
(NN dN/deta)
(DNN)
(NDN)
(NND)
(NDD)
(DND)
(DDN)
(DDD)
"""

from numpy import empty

import struphy.bsplines.bsplines_kernels as bsp


# =============================================================================
def evaluation_kernel_3d(
    p1: "int",
    p2: "int",
    p3: "int",
    basis1: "float[:]",
    basis2: "float[:]",
    basis3: "float[:]",
    span1: "int",
    span2: "int",
    span3: "int",
    nbase1: "int",
    nbase2: "int",
    nbase3: "int",
    coeff: "float[:,:,:]",
):
    """Summing non-zero contributions.

    Parameters:
    -----------
        p1, p2, p3:                 int                 spline degrees
        basis1, basis2, basis3:     double[:]           pn+1 values of non-zero basis splines at one point eta_n from 'basis_funs' (n=1,2,3)
        span1, span2, span3:        int                 knot span indices from 'find_span'
        nbase1, nbase2, nbase3:     int                 dimensions of univariate spline spaces
        coeff:                      double[:, :, :]     spline coefficients c_ijk

    Returns:
    --------
    value: float
        Value of B-spline at point (eta1, eta2, eta3).
    """

    value = 0.0

    for il1 in range(p1 + 1):
        i1 = (span1 - il1) % nbase1
        for il2 in range(p2 + 1):
            i2 = (span2 - il2) % nbase2
            for il3 in range(p3 + 1):
                i3 = (span3 - il3) % nbase3

                value += coeff[i1, i2, i3] * basis1[p1 - il1] * basis2[p2 - il2] * basis3[p3 - il3]

    return value


# =============================================================================
def evaluate_n_n_n(
    tn1: "float[:]",
    tn2: "float[:]",
    tn3: "float[:]",
    pn1: "int",
    pn2: "int",
    pn3: "int",
    nbase_n1: "int",
    nbase_n2: "int",
    nbase_n3: "int",
    coeff: "float[:,:,:]",
    eta1: "float",
    eta2: "float",
    eta3: "float",
):
    """Point-wise evaluation of (NNN)-tensor-product spline.

    Parameters:
    -----------
        tn1, tn2, tn3:                  double[:]           knot vectors
        pn1, pn2, pn3:                  int                 spline degrees
        nbase_n1, nbase_n2, nbase_n3:   int                 dimensions of univariate spline spaces
        coeff:                          double[:, :, :]     spline coefficients c_ijk
        eta1, eta2, eta3:               double              point of evaluation

    Returns:
    --------
        value: float
            Value of (NNN)-tensor-product spline at point (eta1, eta2, eta3).
    """

    # find knot span indices
    span_n1 = bsp.find_span(tn1, pn1, eta1)
    span_n2 = bsp.find_span(tn2, pn2, eta2)
    span_n3 = bsp.find_span(tn3, pn3, eta3)

    # evaluate non-vanishing basis functions
    bn1 = empty(pn1 + 1, dtype=float)
    bn2 = empty(pn2 + 1, dtype=float)
    bn3 = empty(pn3 + 1, dtype=float)

    bl1 = empty(pn1, dtype=float)
    bl2 = empty(pn2, dtype=float)
    bl3 = empty(pn3, dtype=float)

    br1 = empty(pn1, dtype=float)
    br2 = empty(pn2, dtype=float)
    br3 = empty(pn3, dtype=float)

    bsp.basis_funs(tn1, pn1, eta1, span_n1, bl1, br1, bn1)
    bsp.basis_funs(tn2, pn2, eta2, span_n2, bl2, br2, bn2)
    bsp.basis_funs(tn3, pn3, eta3, span_n3, bl3, br3, bn3)

    # sum up non-vanishing contributions
    value = evaluation_kernel_3d(
        pn1, pn2, pn3, bn1, bn2, bn3, span_n1, span_n2, span_n3, nbase_n1, nbase_n2, nbase_n3, coeff
    )

    return value


# =============================================================================
def evaluate_diffn_n_n(
    tn1: "float[:]",
    tn2: "float[:]",
    tn3: "float[:]",
    pn1: "int",
    pn2: "int",
    pn3: "int",
    nbase_n1: "int",
    nbase_n2: "int",
    nbase_n3: "int",
    coeff: "float[:,:,:]",
    eta1: "float",
    eta2: "float",
    eta3: "float",
):
    """Point-wise evaluation of (dN/deta NN)-tensor-product spline.

    Parameters:
    -----------
        tn1, tn2, tn3:                  double[:]           knot vectors
        pn1, pn2, pn3:                  int                 spline degrees
        nbase_n1, nbase_n2, nbase_n3:   int                 dimensions of univariate spline spaces
        coeff:                          double[:, :, :]     spline coefficients c_ijk
        eta1, eta2, eta3:               double              point of evaluation

    Returns:
    --------
        value: float
            Value of (dN/deta NN)-tensor-product spline at point (eta1, eta2, eta3).
    """

    # find knot span indices
    span_n1 = bsp.find_span(tn1, pn1, eta1)
    span_n2 = bsp.find_span(tn2, pn2, eta2)
    span_n3 = bsp.find_span(tn3, pn3, eta3)

    # evaluate non-vanishing basis functions
    bn1 = empty(pn1 + 1, dtype=float)
    bn2 = empty(pn2 + 1, dtype=float)
    bn3 = empty(pn3 + 1, dtype=float)

    bl1 = empty(pn1, dtype=float)
    bl2 = empty(pn2, dtype=float)
    bl3 = empty(pn3, dtype=float)

    br1 = empty(pn1, dtype=float)
    br2 = empty(pn2, dtype=float)
    br3 = empty(pn3, dtype=float)

    bsp.basis_funs_1st_der(tn1, pn1, eta1, span_n1, bl1, br1, bn1)
    bsp.basis_funs(tn2, pn2, eta2, span_n2, bl2, br2, bn2)
    bsp.basis_funs(tn3, pn3, eta3, span_n3, bl3, br3, bn3)

    # sum up non-vanishing contributions
    value = evaluation_kernel_3d(
        pn1, pn2, pn3, bn1, bn2, bn3, span_n1, span_n2, span_n3, nbase_n1, nbase_n2, nbase_n3, coeff
    )

    return value


# =============================================================================
def evaluate_n_diffn_n(
    tn1: "float[:]",
    tn2: "float[:]",
    tn3: "float[:]",
    pn1: "int",
    pn2: "int",
    pn3: "int",
    nbase_n1: "int",
    nbase_n2: "int",
    nbase_n3: "int",
    coeff: "float[:,:,:]",
    eta1: "float",
    eta2: "float",
    eta3: "float",
):
    """Point-wise evaluation of (N dN/deta N)-tensor-product spline.

    Parameters:
    -----------
        tn1, tn2, tn3:                  double[:]           knot vectors
        pn1, pn2, pn3:                  int                 spline degrees
        nbase_n1, nbase_n2, nbase_n3:   int                 dimensions of univariate spline spaces
        coeff:                          double[:, :, :]     spline coefficients c_ijk
        eta1, eta2, eta3:               double              point of evaluation

    Returns:
    --------
        value: float
            Value of (N dN/deta N)-tensor-product spline at point (eta1, eta2, eta3).
    """

    # find knot span indices
    span_n1 = bsp.find_span(tn1, pn1, eta1)
    span_n2 = bsp.find_span(tn2, pn2, eta2)
    span_n3 = bsp.find_span(tn3, pn3, eta3)

    # evaluate non-vanishing basis functions
    bn1 = empty(pn1 + 1, dtype=float)
    bn2 = empty(pn2 + 1, dtype=float)
    bn3 = empty(pn3 + 1, dtype=float)

    bl1 = empty(pn1, dtype=float)
    bl2 = empty(pn2, dtype=float)
    bl3 = empty(pn3, dtype=float)

    br1 = empty(pn1, dtype=float)
    br2 = empty(pn2, dtype=float)
    br3 = empty(pn3, dtype=float)

    bsp.basis_funs(tn1, pn1, eta1, span_n1, bl1, br1, bn1)
    bsp.basis_funs_1st_der(tn2, pn2, eta2, span_n2, bl2, br2, bn2)
    bsp.basis_funs(tn3, pn3, eta3, span_n3, bl3, br3, bn3)

    # sum up non-vanishing contributions
    value = evaluation_kernel_3d(
        pn1, pn2, pn3, bn1, bn2, bn3, span_n1, span_n2, span_n3, nbase_n1, nbase_n2, nbase_n3, coeff
    )

    return value


# =============================================================================
def evaluate_n_n_diffn(
    tn1: "float[:]",
    tn2: "float[:]",
    tn3: "float[:]",
    pn1: "int",
    pn2: "int",
    pn3: "int",
    nbase_n1: "int",
    nbase_n2: "int",
    nbase_n3: "int",
    coeff: "float[:,:,:]",
    eta1: "float",
    eta2: "float",
    eta3: "float",
):
    """Point-wise evaluation of (NN dN/deta)-tensor-product spline.

    Parameters:
    -----------
        tn1, tn2, tn3:                  double[:]           knot vectors
        pn1, pn2, pn3:                  int                 spline degrees
        nbase_n1, nbase_n2, nbase_n3:   int                 dimensions of univariate spline spaces
        coeff:                          double[:, :, :]     spline coefficients c_ijk
        eta1, eta2, eta3:               double              point of evaluation

    Returns:
    --------
        value: float
            Value of (NN dN/deta)-tensor-product spline at point (eta1, eta2, eta3).
    """

    # find knot span indices
    span_n1 = bsp.find_span(tn1, pn1, eta1)
    span_n2 = bsp.find_span(tn2, pn2, eta2)
    span_n3 = bsp.find_span(tn3, pn3, eta3)

    # evaluate non-vanishing basis functions
    bn1 = empty(pn1 + 1, dtype=float)
    bn2 = empty(pn2 + 1, dtype=float)
    bn3 = empty(pn3 + 1, dtype=float)

    bl1 = empty(pn1, dtype=float)
    bl2 = empty(pn2, dtype=float)
    bl3 = empty(pn3, dtype=float)

    br1 = empty(pn1, dtype=float)
    br2 = empty(pn2, dtype=float)
    br3 = empty(pn3, dtype=float)

    bsp.basis_funs(tn1, pn1, eta1, span_n1, bl1, br1, bn1)
    bsp.basis_funs(tn2, pn2, eta2, span_n2, bl2, br2, bn2)
    bsp.basis_funs_1st_der(tn3, pn3, eta3, span_n3, bl3, br3, bn3)

    # sum up non-vanishing contributions
    value = evaluation_kernel_3d(
        pn1, pn2, pn3, bn1, bn2, bn3, span_n1, span_n2, span_n3, nbase_n1, nbase_n2, nbase_n3, coeff
    )

    return value


# =============================================================================
def evaluate_d_n_n(
    td1: "float[:]",
    tn2: "float[:]",
    tn3: "float[:]",
    pd1: "int",
    pn2: "int",
    pn3: "int",
    nbase_d1: "int",
    nbase_n2: "int",
    nbase_n3: "int",
    coeff: "float[:,:,:]",
    eta1: "float",
    eta2: "float",
    eta3: "float",
):
    """Point-wise evaluation of (DNN)-tensor-product spline.

    Parameters:
    -----------
        td1, tn2, tn3:                  double[:]           knot vectors
        pd1, pn2, pn3:                  int                 spline degrees
        nbase_d1, nbase_n2, nbase_n3:   int                 dimensions of univariate spline spaces
        coeff:                          double[:, :, :]     spline coefficients c_ijk
        eta1, eta2, eta3:               double              point of evaluation

    Returns:
    --------
        value: float
            Value of (DNN)-tensor-product spline at point (eta1, eta2, eta3).
    """

    # find knot span indices
    span_d1 = bsp.find_span(td1, pd1, eta1)
    span_n2 = bsp.find_span(tn2, pn2, eta2)
    span_n3 = bsp.find_span(tn3, pn3, eta3)

    # evaluate non-vanishing basis functions
    bd1 = empty(pd1 + 1, dtype=float)
    bn2 = empty(pn2 + 1, dtype=float)
    bn3 = empty(pn3 + 1, dtype=float)

    bl1 = empty(pd1, dtype=float)
    bl2 = empty(pn2, dtype=float)
    bl3 = empty(pn3, dtype=float)

    br1 = empty(pd1, dtype=float)
    br2 = empty(pn2, dtype=float)
    br3 = empty(pn3, dtype=float)

    bsp.basis_funs(td1, pd1, eta1, span_d1, bl1, br1, bd1)
    bsp.basis_funs(tn2, pn2, eta2, span_n2, bl2, br2, bn2)
    bsp.basis_funs(tn3, pn3, eta3, span_n3, bl3, br3, bn3)

    bsp.scaling(td1, pd1, span_d1, bd1)

    # sum up non-vanishing contributions
    value = evaluation_kernel_3d(
        pd1, pn2, pn3, bd1, bn2, bn3, span_d1, span_n2, span_n3, nbase_d1, nbase_n2, nbase_n3, coeff
    )

    return value


# =============================================================================
def evaluate_n_d_n(
    tn1: "float[:]",
    td2: "float[:]",
    tn3: "float[:]",
    pn1: "int",
    pd2: "int",
    pn3: "int",
    nbase_n1: "int",
    nbase_d2: "int",
    nbase_n3: "int",
    coeff: "float[:,:,:]",
    eta1: "float",
    eta2: "float",
    eta3: "float",
):
    """Point-wise evaluation of (NDN)-tensor-product spline.

    Parameters:
    -----------
        tn1, td2, tn3:                  double[:]           knot vectors
        pn1, pd2, pn3:                  int                 spline degrees
        nbase_n1, nbase_d2, nbase_n3:   int                 dimensions of univariate spline spaces
        coeff:                          double[:, :, :]     spline coefficients c_ijk
        eta1, eta2, eta3:               double              point of evaluation

    Returns:
    --------
        value: float
            Value of (NDN)-tensor-product spline at point (eta1, eta2, eta3).
    """

    # find knot span indices
    span_n1 = bsp.find_span(tn1, pn1, eta1)
    span_d2 = bsp.find_span(td2, pd2, eta2)
    span_n3 = bsp.find_span(tn3, pn3, eta3)

    # evaluate non-vanishing basis functions
    bn1 = empty(pn1 + 1, dtype=float)
    bd2 = empty(pd2 + 1, dtype=float)
    bn3 = empty(pn3 + 1, dtype=float)

    bl1 = empty(pn1, dtype=float)
    bl2 = empty(pd2, dtype=float)
    bl3 = empty(pn3, dtype=float)

    br1 = empty(pn1, dtype=float)
    br2 = empty(pd2, dtype=float)
    br3 = empty(pn3, dtype=float)

    bsp.basis_funs(tn1, pn1, eta1, span_n1, bl1, br1, bn1)
    bsp.basis_funs(td2, pd2, eta2, span_d2, bl2, br2, bd2)
    bsp.basis_funs(tn3, pn3, eta3, span_n3, bl3, br3, bn3)

    bsp.scaling(td2, pd2, span_d2, bd2)

    # sum up non-vanishing contributions
    value = evaluation_kernel_3d(
        pn1, pd2, pn3, bn1, bd2, bn3, span_n1, span_d2, span_n3, nbase_n1, nbase_d2, nbase_n3, coeff
    )

    return value


# =============================================================================
def evaluate_n_n_d(
    tn1: "float[:]",
    tn2: "float[:]",
    td3: "float[:]",
    pn1: "int",
    pn2: "int",
    pd3: "int",
    nbase_n1: "int",
    nbase_n2: "int",
    nbase_d3: "int",
    coeff: "float[:,:,:]",
    eta1: "float",
    eta2: "float",
    eta3: "float",
):
    """Point-wise evaluation of (NND)-tensor-product spline.

    Parameters:
    -----------
        tn1, tn2, td3:                  double[:]           knot vectors
        pn1, pn2, pd3:                  int                 spline degrees
        nbase_n1, nbase_n2, nbase_d3:   int                 dimensions of univariate spline spaces
        coeff:                          double[:, :, :]     spline coefficients c_ijk
        eta1, eta2, eta3:               double              point of evaluation

    Returns:
    --------
        value: float
            Value of (NND)-tensor-product spline at point (eta1, eta2, eta3).
    """

    # find knot span indices
    span_n1 = bsp.find_span(tn1, pn1, eta1)
    span_n2 = bsp.find_span(tn2, pn2, eta2)
    span_d3 = bsp.find_span(td3, pd3, eta3)

    # evaluate non-vanishing basis functions
    bn1 = empty(pn1 + 1, dtype=float)
    bn2 = empty(pn2 + 1, dtype=float)
    bd3 = empty(pd3 + 1, dtype=float)

    bl1 = empty(pn1, dtype=float)
    bl2 = empty(pn2, dtype=float)
    bl3 = empty(pd3, dtype=float)

    br1 = empty(pn1, dtype=float)
    br2 = empty(pn2, dtype=float)
    br3 = empty(pd3, dtype=float)

    bsp.basis_funs(tn1, pn1, eta1, span_n1, bl1, br1, bn1)
    bsp.basis_funs(tn2, pn2, eta2, span_n2, bl2, br2, bn2)
    bsp.basis_funs(td3, pd3, eta3, span_d3, bl3, br3, bd3)

    bsp.scaling(td3, pd3, span_d3, bd3)

    # sum up non-vanishing contributions
    value = evaluation_kernel_3d(
        pn1, pn2, pd3, bn1, bn2, bd3, span_n1, span_n2, span_d3, nbase_n1, nbase_n2, nbase_d3, coeff
    )

    return value


# =============================================================================
def evaluate_n_d_d(
    tn1: "float[:]",
    td2: "float[:]",
    td3: "float[:]",
    pn1: "int",
    pd2: "int",
    pd3: "int",
    nbase_n1: "int",
    nbase_d2: "int",
    nbase_d3: "int",
    coeff: "float[:,:,:]",
    eta1: "float",
    eta2: "float",
    eta3: "float",
):
    """Point-wise evaluation of (NDD)-tensor-product spline.

    Parameters:
    -----------
        tn1, td2, td3:                  double[:]           knot vectors
        pn1, pd2, pd3:                  int                 spline degrees
        nbase_n1, nbase_d2, nbase_d3:   int                 dimensions of univariate spline spaces
        coeff:                          double[:, :, :]     spline coefficients c_ijk
        eta1, eta2, eta3:               double              point of evaluation

    Returns:
    --------
        value: float
            Value of (NDD)-tensor-product spline at point (eta1, eta2, eta3).
    """

    # find knot span indices
    span_n1 = bsp.find_span(tn1, pn1, eta1)
    span_d2 = bsp.find_span(td2, pd2, eta2)
    span_d3 = bsp.find_span(td3, pd3, eta3)

    # evaluate non-vanishing basis functions
    bn1 = empty(pn1 + 1, dtype=float)
    bd2 = empty(pd2 + 1, dtype=float)
    bd3 = empty(pd3 + 1, dtype=float)

    bl1 = empty(pn1, dtype=float)
    bl2 = empty(pd2, dtype=float)
    bl3 = empty(pd3, dtype=float)

    br1 = empty(pn1, dtype=float)
    br2 = empty(pd2, dtype=float)
    br3 = empty(pd3, dtype=float)

    bsp.basis_funs(tn1, pn1, eta1, span_n1, bl1, br1, bn1)
    bsp.basis_funs(td2, pd2, eta2, span_d2, bl2, br2, bd2)
    bsp.basis_funs(td3, pd3, eta3, span_d3, bl3, br3, bd3)

    bsp.scaling(td2, pd2, span_d2, bd2)
    bsp.scaling(td3, pd3, span_d3, bd3)

    # sum up non-vanishing contributions
    value = evaluation_kernel_3d(
        pn1, pd2, pd3, bn1, bd2, bd3, span_n1, span_d2, span_d3, nbase_n1, nbase_d2, nbase_d3, coeff
    )

    return value


# =============================================================================
def evaluate_d_n_d(
    td1: "float[:]",
    tn2: "float[:]",
    td3: "float[:]",
    pd1: "int",
    pn2: "int",
    pd3: "int",
    nbase_d1: "int",
    nbase_n2: "int",
    nbase_d3: "int",
    coeff: "float[:,:,:]",
    eta1: "float",
    eta2: "float",
    eta3: "float",
):
    """Point-wise evaluation of (DND)-tensor-product spline.

    Parameters:
    -----------
        td1, tn2, td3:                  double[:]           knot vectors
        pd1, pn2, pd3:                  int                 spline degrees
        nbase_d1, nbase_n2, nbase_d3:   int                 dimensions of univariate spline spaces
        coeff:                          double[:, :, :]     spline coefficients c_ijk
        eta1, eta2, eta3:               double              point of evaluation

    Returns:
    --------
        value: float
            Value of (DND)-tensor-product spline at point (eta1, eta2, eta3).
    """

    # find knot span indices
    span_d1 = bsp.find_span(td1, pd1, eta1)
    span_n2 = bsp.find_span(tn2, pn2, eta2)
    span_d3 = bsp.find_span(td3, pd3, eta3)

    # evaluate non-vanishing basis functions
    bd1 = empty(pd1 + 1, dtype=float)
    bn2 = empty(pn2 + 1, dtype=float)
    bd3 = empty(pd3 + 1, dtype=float)

    bl1 = empty(pd1, dtype=float)
    bl2 = empty(pn2, dtype=float)
    bl3 = empty(pd3, dtype=float)

    br1 = empty(pd1, dtype=float)
    br2 = empty(pn2, dtype=float)
    br3 = empty(pd3, dtype=float)

    bsp.basis_funs(td1, pd1, eta1, span_d1, bl1, br1, bd1)
    bsp.basis_funs(tn2, pn2, eta2, span_n2, bl2, br2, bn2)
    bsp.basis_funs(td3, pd3, eta3, span_d3, bl3, br3, bd3)

    bsp.scaling(td1, pd1, span_d1, bd1)
    bsp.scaling(td3, pd3, span_d3, bd3)

    # sum up non-vanishing contributions
    value = evaluation_kernel_3d(
        pd1, pn2, pd3, bd1, bn2, bd3, span_d1, span_n2, span_d3, nbase_d1, nbase_n2, nbase_d3, coeff
    )

    return value


# =============================================================================
def evaluate_d_d_n(
    td1: "float[:]",
    td2: "float[:]",
    tn3: "float[:]",
    pd1: "int",
    pd2: "int",
    pn3: "int",
    nbase_d1: "int",
    nbase_d2: "int",
    nbase_n3: "int",
    coeff: "float[:,:,:]",
    eta1: "float",
    eta2: "float",
    eta3: "float",
):
    """Point-wise evaluation of (DDN)-tensor-product spline.

    Parameters:
    -----------
        td1, td2, tn3:                  double[:]           knot vectors
        pd1, pd2, pn3:                  int                 spline degrees
        nbase_d1, nbase_d2, nbase_n3:   int                 dimensions of univariate spline spaces
        coeff:                          double[:, :, :]     spline coefficients c_ijk
        eta1, eta2, eta3:               double              point of evaluation

    Returns:
    --------
        value: float
            Value of (DDN)-tensor-product spline at point (eta1, eta2, eta3).
    """

    # find knot span indices
    span_d1 = bsp.find_span(td1, pd1, eta1)
    span_d2 = bsp.find_span(td2, pd2, eta2)
    span_n3 = bsp.find_span(tn3, pn3, eta3)

    # evaluate non-vanishing basis functions
    bd1 = empty(pd1 + 1, dtype=float)
    bd2 = empty(pd2 + 1, dtype=float)
    bn3 = empty(pn3 + 1, dtype=float)

    bl1 = empty(pd1, dtype=float)
    bl2 = empty(pd2, dtype=float)
    bl3 = empty(pn3, dtype=float)

    br1 = empty(pd1, dtype=float)
    br2 = empty(pd2, dtype=float)
    br3 = empty(pn3, dtype=float)

    bsp.basis_funs(td1, pd1, eta1, span_d1, bl1, br1, bd1)
    bsp.basis_funs(td2, pd2, eta2, span_d2, bl2, br2, bd2)
    bsp.basis_funs(tn3, pn3, eta3, span_n3, bl3, br3, bn3)

    bsp.scaling(td1, pd1, span_d1, bd1)
    bsp.scaling(td2, pd2, span_d2, bd2)

    # sum up non-vanishing contributions
    value = evaluation_kernel_3d(
        pd1, pd2, pn3, bd1, bd2, bn3, span_d1, span_d2, span_n3, nbase_d1, nbase_d2, nbase_n3, coeff
    )

    return value


# =============================================================================
def evaluate_d_d_d(
    td1: "float[:]",
    td2: "float[:]",
    td3: "float[:]",
    pd1: "int",
    pd2: "int",
    pd3: "int",
    nbase_d1: "int",
    nbase_d2: "int",
    nbase_d3: "int",
    coeff: "float[:,:,:]",
    eta1: "float",
    eta2: "float",
    eta3: "float",
):
    """Point-wise evaluation of (DDD)-tensor-product spline.

    Parameters:
    -----------
        td1, td2, td3:                  double[:]           knot vectors
        pd1, pd2, pd3:                  int                 spline degrees
        nbase_d1, nbase_d2, nbase_d3:   int                 dimensions of univariate spline spaces
        coeff:                          double[:, :, :]     spline coefficients c_ijk
        eta1, eta2, eta3:               double              point of evaluation

    Returns:
    --------
        value: float
            Value of (DDD)-tensor-product spline at point (eta1, eta2, eta3).
    """

    # find knot span indices
    span_d1 = bsp.find_span(td1, pd1, eta1)
    span_d2 = bsp.find_span(td2, pd2, eta2)
    span_d3 = bsp.find_span(td3, pd3, eta3)

    # evaluate non-vanishing basis functions
    bd1 = empty(pd1 + 1, dtype=float)
    bd2 = empty(pd2 + 1, dtype=float)
    bd3 = empty(pd3 + 1, dtype=float)

    bl1 = empty(pd1, dtype=float)
    bl2 = empty(pd2, dtype=float)
    bl3 = empty(pd3, dtype=float)

    br1 = empty(pd1, dtype=float)
    br2 = empty(pd2, dtype=float)
    br3 = empty(pd3, dtype=float)

    bsp.basis_funs(td1, pd1, eta1, span_d1, bl1, br1, bd1)
    bsp.basis_funs(td2, pd2, eta2, span_d2, bl2, br2, bd2)
    bsp.basis_funs(td3, pd3, eta3, span_d3, bl3, br3, bd3)

    bsp.scaling(td1, pd1, span_d1, bd1)
    bsp.scaling(td2, pd2, span_d2, bd2)
    bsp.scaling(td3, pd3, span_d3, bd3)

    # sum up non-vanishing contributions
    value = evaluation_kernel_3d(
        pd1, pd2, pd3, bd1, bd2, bd3, span_d1, span_d2, span_d3, nbase_d1, nbase_d2, nbase_d3, coeff
    )

    return value


# =============================================================================
def evaluate_tensor_product(
    t1: "float[:]",
    t2: "float[:]",
    t3: "float[:]",
    p1: "int",
    p2: "int",
    p3: "int",
    nbase_1: "int",
    nbase_2: "int",
    nbase_3: "int",
    coeff: "float[:,:,:]",
    eta1: "float[:]",
    eta2: "float[:]",
    eta3: "float[:]",
    values: "float[:,:,:]",
    kind: "int",
):
    """Tensor product evaluation (meshgrid) of tensor product splines (3d).

    Parameters:
    -----------
        t1, t2, t3:                 double[:]           knot vectors
        p1, p2, p3:                 int                 spline degrees
        nbase_1, nbase_2, nbase_3:  int                 dimensions of univariate spline spaces
        coeff:                      double[:, :, :]     spline coefficients c_ijk
        eta1, eta2, eta3:           double[:]           1d arrays of points of evaluation in respective direction
        kind:                       int                 which tensor product spline,
                                                        0: (NNN), 11: (DNN), 12: (NDN), 13: (NND),
                                                        21: (NDD), 22: (DND), 23: (DDN), 3: (DDD)

    Returns:
    --------
        values:                     double[:, :, :]     values of spline at points from
                                                        np.meshgrid(eta1, eta2, eta3, indexing='ij').
    """

    for i1 in range(len(eta1)):
        for i2 in range(len(eta2)):
            for i3 in range(len(eta3)):
                # V0 - space
                if kind == 0:
                    values[i1, i2, i3] = evaluate_n_n_n(
                        t1, t2, t3, p1, p2, p3, nbase_1, nbase_2, nbase_3, coeff, eta1[i1], eta2[i2], eta3[i3]
                    )

                # V1 - space
                elif kind == 11:
                    values[i1, i2, i3] = evaluate_d_n_n(
                        t1, t2, t3, p1, p2, p3, nbase_1, nbase_2, nbase_3, coeff, eta1[i1], eta2[i2], eta3[i3]
                    )
                elif kind == 12:
                    values[i1, i2, i3] = evaluate_n_d_n(
                        t1, t2, t3, p1, p2, p3, nbase_1, nbase_2, nbase_3, coeff, eta1[i1], eta2[i2], eta3[i3]
                    )
                elif kind == 13:
                    values[i1, i2, i3] = evaluate_n_n_d(
                        t1, t2, t3, p1, p2, p3, nbase_1, nbase_2, nbase_3, coeff, eta1[i1], eta2[i2], eta3[i3]
                    )

                # V2 - space
                elif kind == 21:
                    values[i1, i2, i3] = evaluate_n_d_d(
                        t1, t2, t3, p1, p2, p3, nbase_1, nbase_2, nbase_3, coeff, eta1[i1], eta2[i2], eta3[i3]
                    )
                elif kind == 22:
                    values[i1, i2, i3] = evaluate_d_n_d(
                        t1, t2, t3, p1, p2, p3, nbase_1, nbase_2, nbase_3, coeff, eta1[i1], eta2[i2], eta3[i3]
                    )
                elif kind == 23:
                    values[i1, i2, i3] = evaluate_d_d_n(
                        t1, t2, t3, p1, p2, p3, nbase_1, nbase_2, nbase_3, coeff, eta1[i1], eta2[i2], eta3[i3]
                    )

                # V3 - space
                elif kind == 3:
                    values[i1, i2, i3] = evaluate_d_d_d(
                        t1, t2, t3, p1, p2, p3, nbase_1, nbase_2, nbase_3, coeff, eta1[i1], eta2[i2], eta3[i3]
                    )


# =============================================================================
def evaluate_matrix(
    t1: "float[:]",
    t2: "float[:]",
    t3: "float[:]",
    p1: "int",
    p2: "int",
    p3: "int",
    nbase_1: "int",
    nbase_2: "int",
    nbase_3: "int",
    coeff: "float[:,:,:]",
    eta1: "float[:,:,:]",
    eta2: "float[:,:,:]",
    eta3: "float[:,:,:]",
    n1: "int",
    n2: "int",
    n3: "int",
    values: "float[:,:,:]",
    kind: "int",
):
    """Matrix evaluation of tensor product splines (3d).

    Parameters:
    -----------
        t1, t2, t3:                 double[:]           knot vectors
        p1, p2, p3:                 int                 spline degrees
        nbase_1, nbase_2, nbase_3:  int                 dimensions of univariate spline spaces
        coeff:                      double[:, :, :]     spline coefficients c_ijk
        eta1, eta2, eta3:           double[:, :, :]     points of evaluation
        n1, n2, n3:                 int                 eta1.shape = (n1, n2, n3)
        kind:                       int                 which tensor product spline,
                                                        0: (NNN), 11: (DNN), 12: (NDN), 13: (NND),
                                                        21: (NDD), 22: (DND), 23: (DDN), 3: (DDD)

    Returns:
    --------
        values:                     double[:, :, :]     values of spline at points (eta1, eta2, eta3).
    """

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                # V0 - space
                if kind == 0:
                    values[i1, i2, i3] = evaluate_n_n_n(
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        nbase_1,
                        nbase_2,
                        nbase_3,
                        coeff,
                        eta1[i1, i2, i3],
                        eta2[i1, i2, i3],
                        eta3[i1, i2, i3],
                    )

                # V1 - space
                elif kind == 11:
                    values[i1, i2, i3] = evaluate_d_n_n(
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        nbase_1,
                        nbase_2,
                        nbase_3,
                        coeff,
                        eta1[i1, i2, i3],
                        eta2[i1, i2, i3],
                        eta3[i1, i2, i3],
                    )
                elif kind == 12:
                    values[i1, i2, i3] = evaluate_n_d_n(
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        nbase_1,
                        nbase_2,
                        nbase_3,
                        coeff,
                        eta1[i1, i2, i3],
                        eta2[i1, i2, i3],
                        eta3[i1, i2, i3],
                    )
                elif kind == 13:
                    values[i1, i2, i3] = evaluate_n_n_d(
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        nbase_1,
                        nbase_2,
                        nbase_3,
                        coeff,
                        eta1[i1, i2, i3],
                        eta2[i1, i2, i3],
                        eta3[i1, i2, i3],
                    )

                # V2 - space
                elif kind == 21:
                    values[i1, i2, i3] = evaluate_n_d_d(
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        nbase_1,
                        nbase_2,
                        nbase_3,
                        coeff,
                        eta1[i1, i2, i3],
                        eta2[i1, i2, i3],
                        eta3[i1, i2, i3],
                    )
                elif kind == 22:
                    values[i1, i2, i3] = evaluate_d_n_d(
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        nbase_1,
                        nbase_2,
                        nbase_3,
                        coeff,
                        eta1[i1, i2, i3],
                        eta2[i1, i2, i3],
                        eta3[i1, i2, i3],
                    )
                elif kind == 23:
                    values[i1, i2, i3] = evaluate_d_d_n(
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        nbase_1,
                        nbase_2,
                        nbase_3,
                        coeff,
                        eta1[i1, i2, i3],
                        eta2[i1, i2, i3],
                        eta3[i1, i2, i3],
                    )

                # V3 - space
                elif kind == 3:
                    values[i1, i2, i3] = evaluate_d_d_d(
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        nbase_1,
                        nbase_2,
                        nbase_3,
                        coeff,
                        eta1[i1, i2, i3],
                        eta2[i1, i2, i3],
                        eta3[i1, i2, i3],
                    )


# =============================================================================
def evaluate_sparse(
    t1: "float[:]",
    t2: "float[:]",
    t3: "float[:]",
    p1: "int",
    p2: "int",
    p3: "int",
    nbase_1: "int",
    nbase_2: "int",
    nbase_3: "int",
    coeff: "float[:,:,:]",
    eta1: "float[:,:,:]",
    eta2: "float[:,:,:]",
    eta3: "float[:,:,:]",
    n1: "int",
    n2: "int",
    n3: "int",
    values: "float[:,:,:]",
    kind: "int",
):
    """Evaluation of tensor product splines (3d) at point sets obtained from sparse meshgrid.

    Sparse meshgrid output has shape (n1, 1, 1), (1, n2, 1) and (1, 1, n3)

    Parameters:
    -----------
        t1, t2, t3:                 double[:]           knot vectors
        p1, p2, p3:                 int                 spline degrees
        nbase_1, nbase_2, nbase_3:  int                 dimensions of univariate spline spaces
        coeff:                      double[:, :, :]     spline coefficients c_ijk
        eta1, eta2, eta3:           double[:, :, :]     points of evaluation
        n1, n2, n3:                 int                 n1 = eta1.shape[0], n2 = eta2.shape[1], n3 = eta3.shape[2]
        kind:                       int                 which tensor product spline,
                                                        0: (NNN), 11: (DNN), 12: (NDN), 13: (NND),
                                                        21: (NDD), 22: (DND), 23: (DDN), 3: (DDD)

    Returns:
    --------
        values:                     double[:, :, :]     values of spline at points (eta1, eta2, eta3).
    """

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                # V0 - space
                if kind == 0:
                    values[i1, i2, i3] = evaluate_n_n_n(
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        nbase_1,
                        nbase_2,
                        nbase_3,
                        coeff,
                        eta1[i1, 0, 0],
                        eta2[0, i2, 0],
                        eta3[0, 0, i3],
                    )

                # V1 - space
                elif kind == 11:
                    values[i1, i2, i3] = evaluate_d_n_n(
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        nbase_1,
                        nbase_2,
                        nbase_3,
                        coeff,
                        eta1[i1, 0, 0],
                        eta2[0, i2, 0],
                        eta3[0, 0, i3],
                    )
                elif kind == 12:
                    values[i1, i2, i3] = evaluate_n_d_n(
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        nbase_1,
                        nbase_2,
                        nbase_3,
                        coeff,
                        eta1[i1, 0, 0],
                        eta2[0, i2, 0],
                        eta3[0, 0, i3],
                    )
                elif kind == 13:
                    values[i1, i2, i3] = evaluate_n_n_d(
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        nbase_1,
                        nbase_2,
                        nbase_3,
                        coeff,
                        eta1[i1, 0, 0],
                        eta2[0, i2, 0],
                        eta3[0, 0, i3],
                    )

                # V2 - space
                elif kind == 21:
                    values[i1, i2, i3] = evaluate_n_d_d(
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        nbase_1,
                        nbase_2,
                        nbase_3,
                        coeff,
                        eta1[i1, 0, 0],
                        eta2[0, i2, 0],
                        eta3[0, 0, i3],
                    )
                elif kind == 22:
                    values[i1, i2, i3] = evaluate_d_n_d(
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        nbase_1,
                        nbase_2,
                        nbase_3,
                        coeff,
                        eta1[i1, 0, 0],
                        eta2[0, i2, 0],
                        eta3[0, 0, i3],
                    )
                elif kind == 23:
                    values[i1, i2, i3] = evaluate_d_d_n(
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        nbase_1,
                        nbase_2,
                        nbase_3,
                        coeff,
                        eta1[i1, 0, 0],
                        eta2[0, i2, 0],
                        eta3[0, 0, i3],
                    )

                # V3 - space
                elif kind == 3:
                    values[i1, i2, i3] = evaluate_d_d_d(
                        t1,
                        t2,
                        t3,
                        p1,
                        p2,
                        p3,
                        nbase_1,
                        nbase_2,
                        nbase_3,
                        coeff,
                        eta1[i1, 0, 0],
                        eta2[0, i2, 0],
                        eta3[0, 0, i3],
                    )
