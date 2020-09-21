# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Basic functions for point-wise evaluation of a 2d B-spline basis.
"""

from pyccel.decorators import types
from numpy import empty

import hylife.utilitis_FEEC.bsplines_kernels as bsp


# =============================================================================
@types('int','int','int','double[:]','double[:]','double[:]','int','int','int','int','int','int','double[:,:,:]')
def evaluation_kernel(p1, p2, p3, basis1, basis2, basis3, span1, span2, span3, nbase1, nbase2, nbase3, coeff):
    
    value = 0.
    
    for il1 in range(p1 + 1):
        i1 = (span1 - il1)%nbase1
        for il2 in range(p2 + 1):
            i2 = (span2 - il2)%nbase2
            for il3 in range(p3 + 1):
                i3 = (span3 - il3)%nbase3
                
                value += coeff[i1, i2, i3] * basis1[p1 - il1] * basis2[p2 - il2] * basis3[p3 - il3]
        
    return value


# =============================================================================
@types('double[:]','double[:]','double[:]','int','int','int','int','int','int','double[:,:,:]','double','double','double')
def evaluate_n_n_n(tn1, tn2, tn3, pn1, pn2, pn3, nbase_n1, nbase_n2, nbase_n3, coeff, eta1, eta2, eta3):

    # find knot span indices
    span_n1 = bsp.find_span(tn1, pn1, eta1)
    span_n2 = bsp.find_span(tn2, pn2, eta2)
    span_n3 = bsp.find_span(tn3, pn3, eta3)

    # evaluate non-vanishing basis functions
    bn1     = empty(pn1 + 1, dtype=float)
    bn2     = empty(pn2 + 1, dtype=float)
    bn3     = empty(pn3 + 1, dtype=float)
    
    bl1     = empty(pn1    , dtype=float)
    bl2     = empty(pn2    , dtype=float)
    bl3     = empty(pn3    , dtype=float)
    
    br1     = empty(pn1    , dtype=float)
    br2     = empty(pn2    , dtype=float)
    br3     = empty(pn3    , dtype=float)

    bsp.basis_funs(tn1, pn1, eta1, span_n1, bl1, br1, bn1)
    bsp.basis_funs(tn2, pn2, eta2, span_n2, bl2, br2, bn2)
    bsp.basis_funs(tn3, pn3, eta3, span_n3, bl3, br3, bn3)

    # sum up non-vanishing contributions
    value = evaluation_kernel(pn1, pn2, pn3, bn1, bn2, bn3, span_n1, span_n2, span_n3, nbase_n1, nbase_n2, nbase_n3, coeff)

    return value


# =============================================================================
@types('double[:]','double[:]','double[:]','int','int','int','int','int','int','double[:,:,:]','double','double','double')
def evaluate_diffn_n_n(tn1, tn2, tn3, pn1, pn2, pn3, nbase_n1, nbase_n2, nbase_n3, coeff, eta1, eta2, eta3):

    # find knot span indices
    span_n1 = bsp.find_span(tn1, pn1, eta1)
    span_n2 = bsp.find_span(tn2, pn2, eta2)
    span_n3 = bsp.find_span(tn3, pn3, eta3)

    # evaluate non-vanishing basis functions
    bn1     = empty(pn1 + 1, dtype=float)
    bn2     = empty(pn2 + 1, dtype=float)
    bn3     = empty(pn3 + 1, dtype=float)
    
    bl1     = empty(pn1    , dtype=float)
    bl2     = empty(pn2    , dtype=float)
    bl3     = empty(pn3    , dtype=float)
    
    br1     = empty(pn1    , dtype=float)
    br2     = empty(pn2    , dtype=float)
    br3     = empty(pn3    , dtype=float)

    bsp.basis_funs_1st_der(tn1, pn1, eta1, span_n1, bl1, br1, bn1)
    bsp.basis_funs(tn2, pn2, eta2, span_n2, bl2, br2, bn2)
    bsp.basis_funs(tn3, pn3, eta3, span_n3, bl3, br3, bn3)

    # sum up non-vanishing contributions
    value = evaluation_kernel(pn1, pn2, pn3, bn1, bn2, bn3, span_n1, span_n2, span_n3, nbase_n1, nbase_n2, nbase_n3, coeff)

    return value


# =============================================================================
@types('double[:]','double[:]','double[:]','int','int','int','int','int','int','double[:,:,:]','double','double','double')
def evaluate_n_diffn_n(tn1, tn2, tn3, pn1, pn2, pn3, nbase_n1, nbase_n2, nbase_n3, coeff, eta1, eta2, eta3):

    # find knot span indices
    span_n1 = bsp.find_span(tn1, pn1, eta1)
    span_n2 = bsp.find_span(tn2, pn2, eta2)
    span_n3 = bsp.find_span(tn3, pn3, eta3)

    # evaluate non-vanishing basis functions
    bn1     = empty(pn1 + 1, dtype=float)
    bn2     = empty(pn2 + 1, dtype=float)
    bn3     = empty(pn3 + 1, dtype=float)
    
    bl1     = empty(pn1    , dtype=float)
    bl2     = empty(pn2    , dtype=float)
    bl3     = empty(pn3    , dtype=float)
    
    br1     = empty(pn1    , dtype=float)
    br2     = empty(pn2    , dtype=float)
    br3     = empty(pn3    , dtype=float)

    bsp.basis_funs(tn1, pn1, eta1, span_n1, bl1, br1, bn1)
    bsp.basis_funs_1st_der(tn2, pn2, eta2, span_n2, bl2, br2, bn2)
    bsp.basis_funs(tn3, pn3, eta3, span_n3, bl3, br3, bn3)

    # sum up non-vanishing contributions
    value = evaluation_kernel(pn1, pn2, pn3, bn1, bn2, bn3, span_n1, span_n2, span_n3, nbase_n1, nbase_n2, nbase_n3, coeff)

    return value


# =============================================================================
@types('double[:]','double[:]','double[:]','int','int','int','int','int','int','double[:,:,:]','double','double','double')
def evaluate_n_n_diffn(tn1, tn2, tn3, pn1, pn2, pn3, nbase_n1, nbase_n2, nbase_n3, coeff, eta1, eta2, eta3):

    # find knot span indices
    span_n1 = bsp.find_span(tn1, pn1, eta1)
    span_n2 = bsp.find_span(tn2, pn2, eta2)
    span_n3 = bsp.find_span(tn3, pn3, eta3)

    # evaluate non-vanishing basis functions
    bn1     = empty(pn1 + 1, dtype=float)
    bn2     = empty(pn2 + 1, dtype=float)
    bn3     = empty(pn3 + 1, dtype=float)
    
    bl1     = empty(pn1    , dtype=float)
    bl2     = empty(pn2    , dtype=float)
    bl3     = empty(pn3    , dtype=float)
    
    br1     = empty(pn1    , dtype=float)
    br2     = empty(pn2    , dtype=float)
    br3     = empty(pn3    , dtype=float)

    bsp.basis_funs(tn1, pn1, eta1, span_n1, bl1, br1, bn1)
    bsp.basis_funs(tn2, pn2, eta2, span_n2, bl2, br2, bn2)
    bsp.basis_funs_1st_der(tn3, pn3, eta3, span_n3, bl3, br3, bn3)

    # sum up non-vanishing contributions
    value = evaluation_kernel(pn1, pn2, pn3, bn1, bn2, bn3, span_n1, span_n2, span_n3, nbase_n1, nbase_n2, nbase_n3, coeff)

    return value


# =============================================================================
@types('double[:]','double[:]','double[:]','int','int','int','int','int','int','double[:,:,:]','double','double','double')
def evaluate_d_n_n(td1, tn2, tn3, pd1, pn2, pn3, nbase_d1, nbase_n2, nbase_n3, coeff, eta1, eta2, eta3):

    # find knot span indices
    span_d1 = bsp.find_span(td1, pd1, eta1)
    span_n2 = bsp.find_span(tn2, pn2, eta2)
    span_n3 = bsp.find_span(tn3, pn3, eta3)

    # evaluate non-vanishing basis functions
    bd1     = empty(pd1 + 1, dtype=float)
    bn2     = empty(pn2 + 1, dtype=float)
    bn3     = empty(pn3 + 1, dtype=float)
    
    bl1     = empty(pd1    , dtype=float)
    bl2     = empty(pn2    , dtype=float)
    bl3     = empty(pn3    , dtype=float)
    
    br1     = empty(pd1    , dtype=float)
    br2     = empty(pn2    , dtype=float)
    br3     = empty(pn3    , dtype=float)

    bsp.basis_funs(td1, pd1, eta1, span_d1, bl1, br1, bd1)
    bsp.basis_funs(tn2, pn2, eta2, span_n2, bl2, br2, bn2)
    bsp.basis_funs(tn3, pn3, eta3, span_n3, bl3, br3, bn3)
    
    bsp.scaling(td1, pd1, span_d1, bd1)

    # sum up non-vanishing contributions
    value = evaluation_kernel(pd1, pn2, pn3, bd1, bn2, bn3, span_d1, span_n2, span_n3, nbase_d1, nbase_n2, nbase_n3, coeff)

    return value


# =============================================================================
@types('double[:]','double[:]','double[:]','int','int','int','int','int','int','double[:,:,:]','double','double','double')
def evaluate_n_d_n(tn1, td2, tn3, pn1, pd2, pn3, nbase_n1, nbase_d2, nbase_n3, coeff, eta1, eta2, eta3):

    # find knot span indices
    span_n1 = bsp.find_span(tn1, pn1, eta1)
    span_d2 = bsp.find_span(td2, pd2, eta2)
    span_n3 = bsp.find_span(tn3, pn3, eta3)

    # evaluate non-vanishing basis functions
    bn1     = empty(pn1 + 1, dtype=float)
    bd2     = empty(pd2 + 1, dtype=float)
    bn3     = empty(pn3 + 1, dtype=float)
    
    bl1     = empty(pn1    , dtype=float)
    bl2     = empty(pd2    , dtype=float)
    bl3     = empty(pn3    , dtype=float)
    
    br1     = empty(pn1    , dtype=float)
    br2     = empty(pd2    , dtype=float)
    br3     = empty(pn3    , dtype=float)

    bsp.basis_funs(tn1, pn1, eta1, span_n1, bl1, br1, bn1)
    bsp.basis_funs(td2, pd2, eta2, span_d2, bl2, br2, bd2)
    bsp.basis_funs(tn3, pn3, eta3, span_n3, bl3, br3, bn3)
    
    bsp.scaling(td2, pd2, span_d2, bd2)

    # sum up non-vanishing contributions
    value = evaluation_kernel(pn1, pd2, pn3, bn1, bd2, bn3, span_n1, span_d2, span_n3, nbase_n1, nbase_d2, nbase_n3, coeff)

    return value


# =============================================================================
@types('double[:]','double[:]','double[:]','int','int','int','int','int','int','double[:,:,:]','double','double','double')
def evaluate_n_n_d(tn1, tn2, td3, pn1, pn2, pd3, nbase_n1, nbase_n2, nbase_d3, coeff, eta1, eta2, eta3):

    # find knot span indices
    span_n1 = bsp.find_span(tn1, pn1, eta1)
    span_n2 = bsp.find_span(tn2, pn2, eta2)
    span_d3 = bsp.find_span(td3, pd3, eta3)

    # evaluate non-vanishing basis functions
    bn1     = empty(pn1 + 1, dtype=float)
    bn2     = empty(pn2 + 1, dtype=float)
    bd3     = empty(pd3 + 1, dtype=float)
    
    bl1     = empty(pn1    , dtype=float)
    bl2     = empty(pn2    , dtype=float)
    bl3     = empty(pd3    , dtype=float)
    
    br1     = empty(pn1    , dtype=float)
    br2     = empty(pn2    , dtype=float)
    br3     = empty(pd3    , dtype=float)

    bsp.basis_funs(tn1, pn1, eta1, span_n1, bl1, br1, bn1)
    bsp.basis_funs(tn2, pn2, eta2, span_n2, bl2, br2, bn2)
    bsp.basis_funs(td3, pd3, eta3, span_d3, bl3, br3, bd3)
    
    bsp.scaling(td3, pd3, span_d3, bd3)

    # sum up non-vanishing contributions
    value = evaluation_kernel(pn1, pn2, pd3, bn1, bn2, bd3, span_n1, span_n2, span_d3, nbase_n1, nbase_n2, nbase_d3, coeff)

    return value


# =============================================================================
@types('double[:]','double[:]','double[:]','int','int','int','int','int','int','double[:,:,:]','double','double','double')
def evaluate_n_d_d(tn1, td2, td3, pn1, pd2, pd3, nbase_n1, nbase_d2, nbase_d3, coeff, eta1, eta2, eta3):

    # find knot span indices
    span_n1 = bsp.find_span(tn1, pn1, eta1)
    span_d2 = bsp.find_span(td2, pd2, eta2)
    span_d3 = bsp.find_span(td3, pd3, eta3)

    # evaluate non-vanishing basis functions
    bn1     = empty(pn1 + 1, dtype=float)
    bd2     = empty(pd2 + 1, dtype=float)
    bd3     = empty(pd3 + 1, dtype=float)
    
    bl1     = empty(pn1    , dtype=float)
    bl2     = empty(pd2    , dtype=float)
    bl3     = empty(pd3    , dtype=float)
    
    br1     = empty(pn1    , dtype=float)
    br2     = empty(pd2    , dtype=float)
    br3     = empty(pd3    , dtype=float)

    bsp.basis_funs(tn1, pn1, eta1, span_n1, bl1, br1, bn1)
    bsp.basis_funs(td2, pd2, eta2, span_d2, bl2, br2, bd2)
    bsp.basis_funs(td3, pd3, eta3, span_d3, bl3, br3, bd3)
    
    bsp.scaling(td2, pd2, span_d2, bd2)
    bsp.scaling(td3, pd3, span_d3, bd3)

    # sum up non-vanishing contributions
    value = evaluation_kernel(pn1, pd2, pd3, bn1, bd2, bd3, span_n1, span_d2, span_d3, nbase_n1, nbase_d2, nbase_d3, coeff)

    return value



# =============================================================================
@types('double[:]','double[:]','double[:]','int','int','int','int','int','int','double[:,:,:]','double','double','double')
def evaluate_d_n_d(td1, tn2, td3, pd1, pn2, pd3, nbase_d1, nbase_n2, nbase_d3, coeff, eta1, eta2, eta3):

    # find knot span indices
    span_d1 = bsp.find_span(td1, pd1, eta1)
    span_n2 = bsp.find_span(tn2, pn2, eta2)
    span_d3 = bsp.find_span(td3, pd3, eta3)

    # evaluate non-vanishing basis functions
    bd1     = empty(pd1 + 1, dtype=float)
    bn2     = empty(pn2 + 1, dtype=float)
    bd3     = empty(pd3 + 1, dtype=float)
    
    bl1     = empty(pd1    , dtype=float)
    bl2     = empty(pn2    , dtype=float)
    bl3     = empty(pd3    , dtype=float)
    
    br1     = empty(pd1    , dtype=float)
    br2     = empty(pn2    , dtype=float)
    br3     = empty(pd3    , dtype=float)

    bsp.basis_funs(td1, pd1, eta1, span_d1, bl1, br1, bd1)
    bsp.basis_funs(tn2, pn2, eta2, span_n2, bl2, br2, bn2)
    bsp.basis_funs(td3, pd3, eta3, span_d3, bl3, br3, bd3)
    
    bsp.scaling(td1, pd1, span_d1, bd1)
    bsp.scaling(td3, pd3, span_d3, bd3)

    # sum up non-vanishing contributions
    value = evaluation_kernel(pd1, pn2, pd3, bd1, bn2, bd3, span_d1, span_n2, span_d3, nbase_d1, nbase_n2, nbase_d3, coeff)

    return value


# =============================================================================
@types('double[:]','double[:]','double[:]','int','int','int','int','int','int','double[:,:,:]','double','double','double')
def evaluate_d_d_n(td1, td2, tn3, pd1, pd2, pn3, nbase_d1, nbase_d2, nbase_n3, coeff, eta1, eta2, eta3):

    # find knot span indices
    span_d1 = bsp.find_span(td1, pd1, eta1)
    span_d2 = bsp.find_span(td2, pd2, eta2)
    span_n3 = bsp.find_span(tn3, pn3, eta3)

    # evaluate non-vanishing basis functions
    bd1     = empty(pd1 + 1, dtype=float)
    bd2     = empty(pd2 + 1, dtype=float)
    bn3     = empty(pn3 + 1, dtype=float)
    
    bl1     = empty(pd1    , dtype=float)
    bl2     = empty(pd2    , dtype=float)
    bl3     = empty(pn3    , dtype=float)
    
    br1     = empty(pd1    , dtype=float)
    br2     = empty(pd2    , dtype=float)
    br3     = empty(pn3    , dtype=float)

    bsp.basis_funs(td1, pd1, eta1, span_d1, bl1, br1, bd1)
    bsp.basis_funs(td2, pd2, eta2, span_d2, bl2, br2, bd2)
    bsp.basis_funs(tn3, pn3, eta3, span_n3, bl3, br3, bn3)
    
    bsp.scaling(td1, pd1, span_d1, bd1)
    bsp.scaling(td2, pd2, span_d2, bd2)

    # sum up non-vanishing contributions
    value = evaluation_kernel(pd1, pd2, pn3, bd1, bd2, bn3, span_d1, span_d2, span_n3, nbase_d1, nbase_d2, nbase_n3, coeff)

    return value


# =============================================================================
@types('double[:]','double[:]','double[:]','int','int','int','int','int','int','double[:,:,:]','double','double','double')
def evaluate_d_d_d(td1, td2, td3, pd1, pd2, pd3, nbase_d1, nbase_d2, nbase_d3, coeff, eta1, eta2, eta3):

    # find knot span indices
    span_d1 = bsp.find_span(td1, pd1, eta1)
    span_d2 = bsp.find_span(td2, pd2, eta2)
    span_d3 = bsp.find_span(td3, pd3, eta3)

    # evaluate non-vanishing basis functions
    bd1     = empty(pd1 + 1, dtype=float)
    bd2     = empty(pd2 + 1, dtype=float)
    bd3     = empty(pd3 + 1, dtype=float)
    
    bl1     = empty(pd1    , dtype=float)
    bl2     = empty(pd2    , dtype=float)
    bl3     = empty(pd3    , dtype=float)
    
    br1     = empty(pd1    , dtype=float)
    br2     = empty(pd2    , dtype=float)
    br3     = empty(pd3    , dtype=float)

    bsp.basis_funs(td1, pd1, eta1, span_d1, bl1, br1, bd1)
    bsp.basis_funs(td2, pd2, eta2, span_d2, bl2, br2, bd2)
    bsp.basis_funs(td3, pd3, eta3, span_d3, bl3, br3, bd3)
    
    bsp.scaling(td1, pd1, span_d1, bd1)
    bsp.scaling(td2, pd2, span_d2, bd2)
    bsp.scaling(td3, pd3, span_d3, bd3)

    # sum up non-vanishing contributions
    value = evaluation_kernel(pd1, pd2, pd3, bd1, bd2, bd3, span_d1, span_d2, span_d3, nbase_d1, nbase_d2, nbase_d3, coeff)

    return value


# =============================================================================
@types('double[:]','double[:]','double[:]','int','int','int','int','int','int','double[:,:,:]','double[:]','double[:]','double[:]','double[:,:,:]','int')
def evaluate_tensor_product(t1, t2, t3, p1, p2, p3, nbase_1, nbase_2, nbase_3, coeff, eta1, eta2, eta3, values, kind):
    
    for i1 in range(len(eta1)):
        for i2 in range(len(eta2)):
            for i3 in range(len(eta3)):
                
                # V0 - space
                if   kind == 0:
                    values[i1, i2, i3] = evaluate_n_n_n(t1, t2, t3, p1, p2, p3, nbase_1, nbase_2, nbase_3, coeff, eta1[i1], eta2[i2], eta3[i3])
                
                # V1 - space
                elif kind == 11:
                    values[i1, i2, i3] = evaluate_d_n_n(t1, t2, t3, p1, p2, p3, nbase_1, nbase_2, nbase_3, coeff, eta1[i1], eta2[i2], eta3[i3])
                elif kind == 12:
                    values[i1, i2, i3] = evaluate_n_d_n(t1, t2, t3, p1, p2, p3, nbase_1, nbase_2, nbase_3, coeff, eta1[i1], eta2[i2], eta3[i3])
                elif kind == 13:
                    values[i1, i2, i3] = evaluate_n_n_d(t1, t2, t3, p1, p2, p3, nbase_1, nbase_2, nbase_3, coeff, eta1[i1], eta2[i2], eta3[i3])
                    
                # V2 - space
                elif kind == 21:
                    values[i1, i2, i3] = evaluate_n_d_d(t1, t2, t3, p1, p2, p3, nbase_1, nbase_2, nbase_3, coeff, eta1[i1], eta2[i2], eta3[i3])
                elif kind == 22:
                    values[i1, i2, i3] = evaluate_d_n_d(t1, t2, t3, p1, p2, p3, nbase_1, nbase_2, nbase_3, coeff, eta1[i1], eta2[i2], eta3[i3])
                elif kind == 23:
                    values[i1, i2, i3] = evaluate_d_d_n(t1, t2, t3, p1, p2, p3, nbase_1, nbase_2, nbase_3, coeff, eta1[i1], eta2[i2], eta3[i3])
                    
                # V3 - space
                elif kind == 3:
                    values[i1, i2, i3] = evaluate_d_d_d(t1, t2, t3, p1, p2, p3, nbase_1, nbase_2, nbase_3, coeff, eta1[i1], eta2[i2], eta3[i3])