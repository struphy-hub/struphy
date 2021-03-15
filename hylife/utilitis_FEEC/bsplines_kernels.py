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
def scaling(t_d, p_d, span_d, values):
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
        values[p_d - il] *= (p_d + 1)/(t_d[i + p_d + 1] - t_d[i])


# ==============================================================================
@types('double[:]','int','double')
def find_span(t, p, eta):
    
    # Knot index at left/right boundary
    low  = p
    high = 0
    high = len(t) - 1 - p

    # Check if point is exactly on left/right boundary, or outside domain
    if eta <= t[low]: 
        returnVal = low
    elif eta >= t[high]: 
        returnVal = high - 1
    else:
        
        # Perform binary search
        span = (low + high)//2
        
        while eta < t[span] or eta >= t[span + 1]:
            
            if eta < t[span]:
                high = span
            else:
                low  = span
            span = (low + high)//2
        
        returnVal = span

    return returnVal


# =============================================================================
@types('double[:]','int','double','int','double[:]','double[:]','double[:]')
def basis_funs(t, p, eta, span, left, right, values):
    """
    Parameters
    ----------
    t : array_like
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
    
    left[:]   = 0.
    right[:]  = 0.
   
    values[0] = 1.
    
    for j in range(p):
        left[j]  = eta - t[span - j]
        right[j] = t[span + 1 + j] - eta
        saved    = 0.
        for r in range(j + 1):
            temp      = values[r]/(right[r] + left[j - r])
            values[r] = saved + right[r] * temp
            saved     = left[j - r] * temp
        values[j + 1] = saved
        
        
# =============================================================================
@types('double[:]','int','double','int','double[:]','double[:]','double[:,:]','double[:]')
def basis_funs_all(t, p, eta, span, left, right, values, diff):
    """
    Parameters
    ----------
    t : array_like
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
    
    left[:]      = 0.
    right[:]     = 0.
    
    values[:, :] = 0.
    values[0, 0] = 1.
    
    for j in range(p):
        left[j]  = eta - t[span - j]
        right[j] = t[span + 1 + j] - eta
        saved    = 0.
        for r in range(j + 1):
            diff[r] = 1. / (right[r] + left[j - r])
            temp = values[j, r] * diff[r]
            values[j + 1, r] = saved + right[r] * temp
            saved     = left[j - r] * temp
        values[j + 1, j + 1] = saved
        
    diff[:] = diff*p

        
               
# =============================================================================
@types('double[:]','int','double','int','double[:]','double[:]','double[:,:]','double[:]','double[:]')
def basis_funs_and_der(t, p, eta, span, left, right, values, diff, der):
    """
    Parameters
    ----------
    t : array_like
        Knots sequence.

    p : int
        Polynomial degree of B-splines.

    eta : double
        Evaluation point.

    span : int
        Knot span index.
        
    left : array_like
        p left values
        
    right : array_like
        p right values
        
    values_all : array_like

    Returns
    -------
    values : numpy.ndarray
        Values of (2, p + 1) non-vanishing B-Splines and derivatives at location eta.
    """
    
    left[:]      = 0.
    right[:]     = 0.
    
    values[:, :] = 0.
    values[0, 0] = 1.
    
    for j in range(p):
        left[j]  = eta - t[span - j]
        right[j] = t[span + 1 + j] - eta
        saved    = 0.
        for r in range(j + 1):
            diff[r] = 1. / (right[r] + left[j - r])
            temp = values[j, r] * diff[r]
            values[j + 1, r] = saved + right[r] * temp
            saved     = left[j - r] * temp
        values[j + 1, j + 1] = saved
        
    diff[:] = diff*p
    
    # compute derivatives
    # j = 0
    saved  = values[p - 1, 0]*diff[0]
    der[0] = -saved
    
    # j = 1, ... , p
    for j in range(1, p):
        temp   = saved
        saved  = values[p - 1, j]*diff[j]
        der[j] = temp - saved
            
    # j = p
    der[p] = saved
            
            
# ==============================================================================
@types('double[:]','int','double','int','double[:]','double[:]','double[:]')
def basis_funs_1st_der(t, p, eta, span, left, right, values):
    """
    Parameters
    ----------
    t : array_like
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
    basis_funs(t, p - 1, eta, span, left, right, values_b)

    # Compute derivatives at x using formula based on difference of splines of degree p - 1
    # -------
    # j = 0
    saved   = p * values_b[0] / (t[span + 1] - t[span + 1 - p])
    values[0] = -saved
    
    # j = 1, ... , p - 1
    for j in range(1, p):
        temp    = saved
        saved   = p * values_b[j] / (t[span + j + 1] - t[span + j + 1 - p])
        values[j] = temp - saved
    
    # j = degree
    values[p] = saved