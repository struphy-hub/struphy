# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Basic functions for point-wise B-spline evaluation
"""

from pyccel.decorators import types

from numpy import empty, zeros

# ==============================================================================
@types('double[:]','int','int','double[:]')
def scaling(T_d, p_d, span_d, values):
    """
    Scales local B-spline values to M-spline values
    
    Parameters
    ----------
    knots : array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.
        
    span : int
        Knot span index.
        
    Returns
    -------
    x : array_like
        Scaling vector with elements (p + 1)/(t[i + p + 1] - t[i])
    """
    
    for il in range(p_d + 1):
        i = span_d - il
        values[p_d - il] *= (p_d + 1)/(T_d[i + p_d + 1] - T_d[i])


# ==============================================================================
@types('double[:]','int','double')
def find_span(T, p, eta):
    
    # Knot index at left/right boundary
    low  = p
    high = 0
    high = len(T) - 1 - p

    # Check if point is exactly on left/right boundary, or outside domain
    if eta <= T[low]: 
        returnVal = low
    elif eta >= T[high]: 
        returnVal = high - 1
    else:
        
        # Perform binary search
        span = (low + high)//2
        
        while eta < T[span] or eta >= T[span + 1]:
            
            if eta < T[span]:
                high = span
            else:
                low  = span
            span = (low + high)//2
        
        returnVal = span

    return returnVal


# =============================================================================
@types('double[:]','int','double','int','double[:]')
def basis_funs(T, p, eta, span, values):
    """
    Parameters
    ----------
    T : array_like
        Knots sequence.

    p : int
        Polynomial degree of B-splines.

    eta : double
        Evaluation point.

    span : int
        Knot span index.

    Returns
    -------
    values : numpy.ndarray
        Values of p + 1 non-vanishing B-Splines at location eta.
    """
    
    left      = empty(p, dtype=float)
    right     = empty(p, dtype=float)
    
    values[:] = 0.
    values[0] = 1.
    
    for j in range(p):
        left[j]  = eta - T[span - j]
        right[j] = T[span + 1 + j] - eta
        saved    = 0.
        for r in range(j + 1):
            temp      = values[r]/(right[r] + left[j - r])
            values[r] = saved + right[r] * temp
            saved     = left[j - r] * temp
        values[j + 1] = saved
        
        
# =============================================================================
@types('double[:]','int','double','int','double[:,:]','double[:]')
def basis_funs_all(T, p, eta, span, values, diff):
    """
    Parameters
    ----------
    T : array_like
        Knots sequence.

    p : int
        Polynomial degree of B-splines.

    eta : double
        Evaluation point.

    span : int
        Knot span index.

    Returns
    -------
    values : numpy.ndarray
        Values of (p + 1, p + 1) non-vanishing B-Splines at location eta.
        
    diff : np.ndarray
        Scaling array (p) for M-splines.
    """
    
    left         = empty(p, dtype=float)
    right        = empty(p, dtype=float)
    
    values[:, :] = 0.
    values[0, 0] = 1.
    
    for j in range(p):
        left[j]  = eta - T[span - j]
        right[j] = T[span + 1 + j] - eta
        saved    = 0.
        for r in range(j + 1):
            diff[r] = 1. / (right[r] + left[j - r])
            temp = values[j, r] * diff[r]
            values[j + 1, r] = saved + right[r] * temp
            saved     = left[j - r] * temp
        values[j + 1, j + 1] = saved
        
    diff[:] = diff*p
        
               
# =============================================================================
@types('double[:]','int','double','int','double[:,:]','double[:]')
def basis_funs_and_der(T, p, eta, span, values):
    """
    Parameters
    ----------
    T : array_like
        Knots sequence.

    p : int
        Polynomial degree of B-splines.

    eta : double
        Evaluation point.

    span : int
        Knot span index.

    Returns
    -------
    values : numpy.ndarray
        Values of (2, p + 1) non-vanishing B-Splines and derivatives at location eta.
    """
    
    left       = empty(p, dtype=float)
    right      = empty(p, dtype=float)
    
    vals       = zeros((p + 1, p + 1), dtype=float)
    vals[0, 0] = 1.
    
    diff       = empty(p, dtype=float)
    
    for j in range(p):
        left[j]  = eta - T[span - j]
        right[j] = T[span + 1 + j] - eta
        saved    = 0.
        for r in range(j + 1):
            diff[r] = 1. / (right[r] + left[j - r])
            temp = vals[j, r] * diff[r]
            vals[j + 1, r] = saved + right[r] * temp
            saved     = left[j - r] * temp
        vals[j + 1, j + 1] = saved
        
    diff[:] = diff[:]*p
    
    
    values[:, :] = 0.
    values[0, :] = vals[p, :] 
    
    # compute derivatives
    # j = 0
    saved = vals[p - 1, 0]*diff[0]
    values[1, 0] = -saved
    
    # j = 1, ..., p
    for j in range(1, p):
        temp  = saved
        saved = vals[p - 1, j]*diff[j]
        values[1, j] = temp - saved
            
    # j = p
    values[1, p] = saved
            
            
# ==============================================================================
@types('double[:]','int','double','int','double[:]')
def basis_funs_1st_der(T, p, eta, span, values):
    """
    Parameters
    ----------
    T : array_like
        Knots sequence.

    p : int
        Polynomial degree of B-splines.

    eta : double
        Evaluation point.

    span : int
        Knot span index.

    Returns
    -------
    values : numpy.ndarray
        Derivatives of p + 1 non-vanishing B-Splines at location eta.
    """
    
    # Compute nonzero basis functions and knot differences for splines up to degree p - 1
    values_b = empty(p + 1, dtype=float)
    basis_funs(T, p - 1, eta, span, values_b)

    # Compute derivatives at x using formula based on difference of splines of degree p - 1
    # -------
    # j = 0
    saved   = p * values_b[0] / (T[span + 1] - T[span + 1 - p])
    values[0] = -saved
    
    # j = 1, ... , p - 1
    for j in range(1, p):
        temp    = saved
        saved   = p * values_b[j] / (T[span + j + 1] - T[span + j + 1 - p])
        values[j] = temp - saved
    
    # j = degree
    values[p] = saved