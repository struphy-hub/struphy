# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Acccelerated functions for point-wise evaluation of univariate B-splines.

S(eta) = sum_i c_i * B_i(eta)      with c_i in R.

where B can be N, D or dN/deta. 
"""

from numpy import empty

import struphy.feec.bsplines_kernels as bsp


# ========================================================
def evaluation_kernel_1d(p : 'int', basis : 'double[:]', span : 'int', nbase : 'int', coeff : 'double[:]') -> 'double':
    '''Summing non-zero contributions.

    Parameters:
    -----------
        p:      int         spline degree
        basis:  double[:]   p+1 values of non-zero basis splines at one point eta from 'basis_funs'
        span:   int         knot span index from 'find_span'
        nbase:  int         dimension of spline space 
        coeff:  double[:]   spline coefficients c_i

    Returns:
    --------
        value: float
            Value of B-spline at point eta.
    '''
    
    value = 0.
    
    for il in range(p + 1):
        i = (span - il)%nbase
        value += coeff[i] * basis[p - il]
        
    return value


# ========================================================
def eval_kernel_1d(p : 'int', basis : 'double[:]', span : 'int', ind : 'int[:]', coeff : 'double[:]') -> 'double':
    '''Summing non-zero contributions.

    Parameters:
    -----------
        p:      int         spline degree
        basis:  double[:]   p+1 values of non-zero basis splines at one point eta from 'basis_funs'
        span:   int         knot span index from 'find_span'
        ind:    int[:]      contains the global indices of non-vanishing basis functions
        coeff:  double[:]   spline coefficients c_i

    Returns:
    --------
        value: float
            Value of B-spline at point eta.
    '''
    
    value = 0.
    
    for il in range(p + 1):
        i = ind[il]
        value += coeff[i] * basis[il]
        
    return value


# ========================================================
def evaluate_n(tn : 'double[:]', pn : 'int', nbase_n : 'int', coeff : 'double[:]', eta : 'double') -> 'double':
    '''Point-wise evaluation of N-spline. 

    Parameters:
    -----------
        tn:         double[:]   knot vector
        pn:         int         spline degree
        nbase_n:    int         dimension of spline space 
        coeff:      double[:]   spline coefficients
        eta:        double      point of evaluation

    Returns:
    --------
        value: float
            Value of N-spline at point eta.
    '''

    # find knot span index
    span_n = bsp.find_span(tn, pn, eta)

    # evaluate non-vanishing basis functions
    bn     = empty(pn + 1, dtype=float)
    bl     = empty(pn    , dtype=float)
    br     = empty(pn    , dtype=float)
    
    bsp.basis_funs(tn, pn, eta, span_n, bl, br, bn) # Why do you give back bl, br?

    # sum up non-vanishing contributions
    value  = evaluation_kernel_1d(pn, bn, span_n, nbase_n, coeff)

    return value


# ========================================================
def evaluate_d(td : 'double[:]', pd : 'int', nbase_d : 'int', coeff : 'double[:]', eta : 'double') -> 'double':
    '''Point-wise evaluation of D-spline.

    Parameters:
    -----------
        td:         double[:]   knot vector
        pd:         int         spline degree
        nbase_d:    int         dimension of spline space 
        coeff:      double[:]   spline coefficients
        eta:        double      point of evaluation

    Returns:
    --------
        value: float
            Value of D-spline at point eta.
    '''

    # find knot span index
    span_d = bsp.find_span(td, pd, eta)

    # evaluate non-vanishing basis functions
    bd     = empty(pd + 1, dtype=float)
    bl     = empty(pd    , dtype=float)
    br     = empty(pd    , dtype=float)
    
    bsp.basis_funs(td, pd, eta, span_d, bl, br, bd)
    bsp.scaling(td, pd, span_d, bd)

    # sum up non-vanishing contributions
    value  = evaluation_kernel_1d(pd, bd, span_d, nbase_d, coeff)

    return value


# ========================================================
def evaluate_diffn(tn : 'double[:]', pn : 'int', nbase_n : 'int', coeff : 'double[:]', eta : 'double') -> 'double':
    '''Point-wise evaluation of derivative of N-spline. 

    Parameters:
    -----------
        tn:         double[:]   knot vector
        pn:         int         spline degree
        nbase_n:    int         dimension of spline space 
        coeff:      double[:]   spline coefficients
        eta:        double      point of evaluation

    Returns:
    --------
        value: float
            Value of dS/deta at point eta.
    '''

    # find knot span index
    span_n = bsp.find_span(tn, pn, eta)

    # evaluate non-vanishing basis functions
    bn     = empty(pn + 1, dtype=float)
    bl     = empty(pn    , dtype=float)
    br     = empty(pn    , dtype=float)
    bsp.basis_funs_1st_der(tn, pn, eta, span_n, bl, br, bn)

    # sum up non-vanishing contributions
    value  = evaluation_kernel_1d(pn, bn, span_n, nbase_n, coeff)

    return value


# ========================================================
def evaluate_vector(t : 'double[:]', p : 'int', nbase : 'int', coeff : 'double[:]', eta : 'double[:]', values : 'double[:]', kind : 'int'):
    '''Vector evaluation of N-spline or D-spline. 

    Parameters:
    -----------
        tn:         double[:]   knot vector
        pn:         int         spline degree
        nbase:      int         dimension of spline space 
        coeff:      double[:]   spline coefficients
        eta:        double[:]   1d array of points of evaluation
        kind:       int         which spline: 0: (N), 1: (D) 

    Returns:
    --------
        values:     double[:]   value of spline at points eta
    '''
    
    for i in range(len(eta)):
        # V0 - space
        if   kind == 0:
            values[i] = evaluate_n(t, p, nbase, coeff, eta[i])
        # V1 - space
        elif kind == 1:
            values[i] = evaluate_d(t, p, nbase, coeff, eta[i])
        # derivative of V0 - space
        elif kind == 2:
            values[i] = evaluate_diffn(t, p, nbase, coeff, eta[i])