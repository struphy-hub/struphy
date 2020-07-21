# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Basic functions for point-wise evaluation of a 2d B-spline basis.
"""

from pyccel.decorators import types
import numpy as np

import hylife.utilitis_FEEC.bsplines_kernels as bsp


# =============================================================================
@types('int','int','double[:]','double[:]','int','int','int','int','double[:,:]')
def evaluation_kernel(p1, p2, basis1, basis2, span1, span2, nbase1, nbase2, coeff):
    
    value = 0.
    
    for il1 in range(p1 + 1):
        i1 = (span1 - il1)%nbase1
        for il2 in range(p2 + 1):
            i2 = (span2 - il2)%nbase2
                
            value += coeff[i1, i2] * basis1[p1 - il1] * basis2[p2 - il2]
        
    return value


# =============================================================================
@types('double[:]','double[:]','int','int','int','int','double[:,:]','double','double')
def evaluate_n_n(tn1, tn2, pn1, pn2, nbase_n1, nbase_n2, coeff, eta1, eta2):

    # find knot span indices
    span_n1 = bsp.find_span(tn1, pn1, eta1)
    span_n2 = bsp.find_span(tn2, pn2, eta2)

    # evaluate non-vanishing basis functions
    bn1   = np.empty(pn1 + 1, dtype=float)
    bn2   = np.empty(pn2 + 1, dtype=float)

    bsp.basis_funs(tn1, pn1, eta1, span_n1, bn1)
    bsp.basis_funs(tn2, pn2, eta2, span_n2, bn2)

    # sum up non-vanishing contributions
    value = evaluation_kernel(pn1, pn2, bn1, bn2, span_n1, span_n2, nbase_n1, nbase_n2, coeff)

    return value


# =============================================================================
@types('double[:]','double[:]','int','int','int','int','double[:,:]','double','double')
def evaluate_d_n(td1, tn2, pd1, pn2, nbase_d1, nbase_n2, coeff, eta1, eta2):

    # find knot span indices
    span_d1 = bsp.find_span(td1, pd1, eta1)
    span_n2 = bsp.find_span(tn2, pn2, eta2)

    # evaluate non-vanishing basis functions
    bd1   = np.empty(pd1 + 1, dtype=float)
    bn2   = np.empty(pn2 + 1, dtype=float)

    bsp.basis_funs(td1, pd1, eta1, span_d1, bd1)
    bsp.basis_funs(tn2, pn2, eta2, span_n2, bn2)
    
    bsp.scaling(td1, pd1, span_d1, bd1)

    # sum up non-vanishing contributions
    value = evaluation_kernel(pd1, pn2, bd1, bn2, span_d1, span_n2, nbase_d1, nbase_n2, coeff)

    return value


# =============================================================================
@types('double[:]','double[:]','int','int','int','int','double[:,:]','double','double')
def evaluate_n_d(tn1, td2, pn1, pd2, nbase_n1, nbase_d2, coeff, eta1, eta2):

    # find knot span indices
    span_n1 = bsp.find_span(tn1, pn1, eta1)
    span_d2 = bsp.find_span(td2, pd2, eta2)

    # evaluate non-vanishing basis functions
    bn1   = np.empty(pn1 + 1, dtype=float)
    bd2   = np.empty(pd2 + 1, dtype=float)

    bsp.basis_funs(tn1, pn1, eta1, span_n1, bn1)
    bsp.basis_funs(td2, pd2, eta2, span_d2, bd2)
    
    bsp.scaling(td2, pd2, span_d2, bd2)

    # sum up non-vanishing contributions
    value = evaluation_kernel(pn1, pd2, bn1, bd2, span_n1, span_d2, nbase_n1, nbase_d2, coeff)

    return value

# =============================================================================
@types('double[:]','double[:]','int','int','int','int','double[:,:]','double','double')
def evaluate_d_d(td1, td2, pd1, pd2, nbase_d1, nbase_d2, coeff, eta1, eta2):

    # find knot span indices
    span_d1 = bsp.find_span(td1, pd1, eta1)
    span_d2 = bsp.find_span(td2, pd2, eta2)

    # evaluate non-vanishing basis functions
    bd1   = np.empty(pd1 + 1, dtype=float)
    bd2   = np.empty(pd2 + 1, dtype=float)

    bsp.basis_funs(td1, pd1, eta1, span_d1, bd1)
    bsp.basis_funs(td2, pd2, eta2, span_d2, bd2)
    
    bsp.scaling(td1, pd1, span_d1, bd1)
    bsp.scaling(td2, pd2, span_d2, bd2)

    # sum up non-vanishing contributions
    value = evaluation_kernel(pd1, pd2, bd1, bd2, span_d1, span_d2, nbase_d1, nbase_d2, coeff)

    return value


# =============================================================================
@types('double[:]','double[:]','int','int','int','int','double[:,:]','double','double')
def evaluate_diffn_n(tn1, tn2, pn1, pn2, nbase_n1, nbase_n2, coeff, eta1, eta2):

    # find knot span indices
    span_n1 = bsp.find_span(tn1, pn1, eta1)
    span_n2 = bsp.find_span(tn2, pn2, eta2)

    # evaluate non-vanishing basis functions
    bn1   = np.empty(pn1 + 1, dtype=float)
    bn2   = np.empty(pn2 + 1, dtype=float)

    bsp.basis_funs_1st_der(tn1, pn1, eta1, span_n1, bn1)
    bsp.basis_funs(tn2, pn2, eta2, span_n2, bn2)

    # sum up non-vanishing contributions
    value = evaluation_kernel(pn1, pn2, bn1, bn2, span_n1, span_n2, nbase_n1, nbase_n2, coeff)

    return value


# =============================================================================
@types('double[:]','double[:]','int','int','int','int','double[:,:]','double','double')
def evaluate_n_diffn(tn1, tn2, pn1, pn2, nbase_n1, nbase_n2, coeff, eta1, eta2):

    # find knot span indices
    span_n1 = bsp.find_span(tn1, pn1, eta1)
    span_n2 = bsp.find_span(tn2, pn2, eta2)

    # evaluate non-vanishing basis functions
    bn1   = np.empty(pn1 + 1, dtype=float)
    bn2   = np.empty(pn2 + 1, dtype=float)

    bsp.basis_funs(tn1, pn1, eta1, span_n1, bn1)
    bsp.basis_funs_1st_der(tn2, pn2, eta2, span_n2, bn2)

    # sum up non-vanishing contributions
    value = evaluation_kernel(pn1, pn2, bn1, bn2, span_n1, span_n2, nbase_n1, nbase_n2, coeff)

    return value