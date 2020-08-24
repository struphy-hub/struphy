# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Basic functions for point-wise evaluation of a 1d B-spline basis.
"""

from pyccel.decorators import types
from numpy import empty

import hylife.utilitis_FEEC.bsplines_kernels as bsp


# ========================================================
@types('int','double[:]','int','int','double[:]')
def evaluation_kernel(p, basis, span, nbase, coeff):
    
    value = 0.
    
    for il in range(p + 1):
        i = (span - il)%nbase
        value += coeff[i] * basis[p - il]
        
    return value


# ========================================================
@types('double[:]','int','int','double[:]','double')
def evaluate_n(tn, pn, nbase_n, coeff, eta):

    # find knot span index
    span_n = bsp.find_span(tn, pn, eta)

    # evaluate non-vanishing basis functions
    bn     = empty(pn + 1, dtype=float)
    bl     = empty(pn    , dtype=float)
    br     = empty(pn    , dtype=float)
    
    bsp.basis_funs(tn, pn, eta, span_n, bl, br, bn)

    # sum up non-vanishing contributions
    value  = evaluation_kernel(pn, bn, span_n, nbase_n, coeff)

    return value


# ========================================================
@types('double[:]','int','int','double[:]','double')
def evaluate_d(td, pd, nbase_d, coeff, eta):

    # find knot span index
    span_d = bsp.find_span(td, pd, eta)

    # evaluate non-vanishing basis functions
    bd     = empty(pd + 1, dtype=float)
    bl     = empty(pd    , dtype=float)
    br     = empty(pd    , dtype=float)
    
    bsp.basis_funs(td, pd, eta, span_d, bl, br, bd)
    bsp.scaling(td, pd, span_d, bd)

    # sum up non-vanishing contributions
    value  = evaluation_kernel(pd, bd, span_d, nbase_d, coeff)

    return value


# ========================================================
@types('double[:]','int','int','double[:]','double')
def evaluate_diffn(tn, pn, nbase_n, coeff, eta):

    # find knot span index
    span_n = bsp.find_span(tn, pn, eta)

    # evaluate non-vanishing basis functions
    bn     = empty(pn + 1, dtype=float)
    bl     = empty(pn    , dtype=float)
    br     = empty(pn    , dtype=float)
    bsp.basis_funs_1st_der(tn, pn, eta, span_n, bl, br, bn)

    # sum up non-vanishing contributions
    value  = evaluation_kernel(pn, bn, span_n, nbase_n, coeff)

    return value