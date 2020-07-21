# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Basic functions for point-wise B-spline evaluation
"""

from pyccel.decorators import types

from numpy import empty

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
    values[0] = 1.0
    
    for j in range(p):
        left[j]  = eta - T[span - j]
        right[j] = T[span + 1 + j] - eta
        saved    = 0.
        for r in range(j + 1):
            temp      = values[r]/(right[r] + left[j - r])
            values[r] = saved + right[r] * temp
            saved     = left[j - r] * temp
        values[j + 1] = saved
        
        
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
    
    
# ==============================================================================
@types('double[:]','int','double','int','int','double[:,:]')
def basis_funs_all_ders(T, p, eta, span, n, values):
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

    n : int
        Max derivative of interest (maximum equal to p).

    Results
    -------
    values : numpy.ndarray (n + 1, p + 1)
        2D array of n + 1 (from 0-th to n-th) derivatives at eta of all (p + 1) non-vanishing basis functions in given span.
    """
    
    left  = empty( p            , dtype=float)
    right = empty( p            , dtype=float)
    ndu   = empty((p + 1, p + 1), dtype=float)
    a     = empty((2, p + 1)    , dtype=float)
    

    # Compute nonzero basis functions and knot differences for splines up to degree, which are needed to compute derivatives.
    # Store values in 2D temporary array 'ndu' (square matrix).
    ndu[0, 0] = 1.
    
    for j in range(p):
        
        left [j] = eta - T[span-j]
        right[j] = T[span + 1 + j] - eta
        saved    = 0.
        
        for r in range(j + 1):
            
            # compute inverse of knot differences and save them into lower triangular part of ndu
            ndu[j + 1, r] = 1. / (right[r] + left[j - r])
            
            # compute basis functions and save them into upper triangular part of ndu
            temp          = ndu[r, j] * ndu[j + 1, r]
            ndu[r, j + 1] = saved + right[r] * temp
            saved         = left[j - r] * temp
            
        ndu[j + 1, j + 1] = saved

    
    # Compute derivatives in 2D output array 'values'
    values[0, :] = ndu[:, p]
    
    for r in range(p + 1):
        s1 = 0
        s2 = 1
        a[0, 0] = 1.
        
        for k in range(1, n + 1):
            d  = 0.
            rk = r - k
            pk = p - k
            if r >= k:
                a[s2, 0] = a[s1, 0] * ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
                
            j1 = 1     if (rk  > -1)    else -rk
            j2 = k - 1 if (r - 1 <= pk) else p - r
            
            for ii in range(j1, j2 + 1):
                a[s2, ii] = (a[s1, ii] - a[s1, ii - 1]) * ndu[pk + 1, rk + ii]
                d += a[s2, ii] * ndu[rk + ii, pk]
            
            if r <= pk:
                a[s2, k] = - a[s1, k - 1] * ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]
                
            values[k, r] = d
            j  = s1
            s1 = s2
            s2 = j

    # Multiply derivatives by correct factors
    r = p
    for k in range(1, n + 1):
        values[k, :] = values[k, :] * r
        r = r * (p - k)