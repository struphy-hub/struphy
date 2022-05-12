# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Basic functions for point-wise B-spline evaluation
"""

from numpy import empty, zeros

# ==============================================================================
def scaling(t_d : 'double[:]', p_d : 'int', span_d : 'int', values : 'double[:]'):
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
def find_span(t : 'double[:]', p : 'int', eta : 'double') -> 'int':
    """
    Computes the span, i.e. the index i for which the B-splines i-p until i are non-vanishing at this point

    Parameters:
    -----------
        t : array
            knot sequence
        
        p : integer
            degree of the basis splines
        
        eta : double
            Evaluation point
    
    Returns:
    --------
        span-index 
    """
    
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
def basis_funs(t : 'double[:]', p : 'int', eta : 'double', span : 'int', left : 'double[:]', right : 'double[:]', values : 'double[:]'):
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
def basis_funs_all(t : 'double[:]', p : 'int', eta : 'double', span : 'int', left : 'double[:]', right : 'double[:]', values : 'double[:,:]', diff : 'double[:]'):
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
def b_splines_slim(t : 'double[:]', pn : 'int', eta : 'double', span : 'int', values : 'double[:]'):
    """
    Computes the values of pn+1 non-vanishing B-splines at position eta

    Parameters
    ----------
    t : array
        Knots sequence.

    pn : int
        Polynomial degree of B-splines.

    eta : double
        Evaluation point.

    span : int
        Knot span index.

    Returns
    -------
    values : array
        Values of p + 1 non-vanishing B-Splines at location eta.
    """
    
    # Initialize variables left and right used for computing the values
    left  = empty( pn, dtype=float )
    right = empty( pn, dtype=float )
    left[:]   = 0.
    right[:]  = 0.
   
    values[0] = 1.
    
    for j in range(pn):
        left[j]  = eta - t[span - j]
        right[j] = t[span + 1 + j] - eta
        saved    = 0.
        for r in range(j + 1):
            temp      = values[r]/(right[r] + left[j - r])
            values[r] = saved + right[r] * temp
            saved     = left[j - r] * temp
        values[j + 1] = saved

    del(left)
    del(right)


# =============================================================================
def d_splines_slim(t : 'double[:]', pn : 'int', eta : 'double', span : 'int', values : 'double[:]'):
    """
    Computes the values of pn non-vanishing D-splines at position eta

    Parameters
    ----------
    t : array
        Knot sequence

    pn : int
        Polynomial degree of B-splines.

    eta : double
        Evaluation point.

    span : int
        Knot span index.

    Returns
    -------
    values : array
        Values of p non-vanishing D-Splines at location eta.
    """
    
    # compute D-spline degree
    pd = pn - 1

    # make sure the arrays we are writing to are empty
    values[:] = 0.

    # Initialize variables left and right used for computing the B-splines up to degree p-1
    left     = empty( pn - 1, dtype=float )
    right    = empty( pn - 1, dtype=float )
    left[:]  = 0.
    right[:] = 0.

    # Compute B-splines up to degree p-1
    b_values    = empty( pn, dtype=float )
    b_values[:] = 0.

    b_values[0] = 1.

    for j in range(pd):
        left[j]  = eta - t[span - j]
        right[j] = t[span + 1 + j] - eta
        saved    = 0.
        for r in range(j + 1):
            temp        = b_values[r] / (right[r] + left[j - r])
            b_values[r] = saved + right[r] * temp
            saved       = left[j - r] * temp
        b_values[j + 1] = saved

    # compute D-splines values by scaling
    for il in range(pd + 1):
        values[pd - il] = pn/(t[span - il + pn] - t[span - il]) * b_values[pd - il]

    del(left)
    del(right)
    del(b_values)


# =============================================================================
def b_d_splines_slim(t : 'double[:]', pn : 'int', eta : 'double', span : 'int', bn : 'double[:]', bd : 'double[:]'):
    """
    One function to compute the values of non-vanishing B-splines and D-splines

    Arguments : 
        t : array
            len 2*p+1, contains the knot vectors
        
        pn : int
            Polynomial degree of the B-spline in this direction

        span : integer
            index for non-vanishing basis functions; index i -> [i-p,i] basis functions are non-vanishing

        eta : array
            contains the position
        
        bn : array
            len np+1, here the values for the non-vanishing B-splines will be written

        bd : array
            len np, here the values for the non-vanishing D-splines will be written
    """

    # compute D-spline degree
    pd = pn - 1

    # make sure the arrays we are writing to are empty
    bn[:] = 0.
    bd[:] = 0.

    # Initialize variables left and right used for computing the value
    left     = empty( pn, dtype=float )
    right    = empty( pn, dtype=float )
    left[:]  = 0.
    right[:] = 0.

    bn[:] = 0.

    bn[0] = 1.

    for j in range(pn):
        left[j]  = eta - t[span - j]
        right[j] = t[span + 1 + j] - eta
        saved    = 0.

        if j == pn-1:
            # compute D-splines values by scaling B-splines of degree pn-1
            for il in range(pd + 1):
                bd[pd - il] = pn/(t[span - il + pn] - t[span - il]) * bn[pd - il]
        
        for r in range(j + 1):
            temp    = bn[r]/(right[r] + left[j - r])
            bn[r]   = saved + right[r] * temp
            saved   = left[j - r] * temp

        bn[j + 1] = saved

    del(left)
    del(right)


# =============================================================================
def basis_funs_and_der(t : 'double[:]', p : 'int', eta : 'double', span : 'int', left : 'double[:]', right : 'double[:]', values : 'double[:,:]', diff : 'double[:]', der : 'double[:]'):
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
def basis_funs_1st_der(t : 'double[:]', p : 'int', eta : 'double', span : 'int', left : 'double[:]', right : 'double[:]', values : 'double[:]'):
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


# ==============================================================================
def b_spl_1st_der_slim(t : 'double[:]', p : 'int', eta : 'double', span : 'int', values : 'double[:]'):
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
    values : array
        Derivatives of p + 1 non-vanishing B-Splines at location eta.
    """
    
    # Compute nonzero basis functions and knot differences for splines up to degree p - 1
    values_b = empty(p + 1, dtype=float)
    b_splines_slim(t, p - 1, eta, span, values_b)

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


#========================================================================================
def piecewise(p : 'int', delta : 'double', eta : 'double') -> 'double':
    # definition of B-splines defined piecewisely
    # eta is eta_j - eta_k
    if abs(eta) > delta * (p+1)*0.5:
        return 0.0 
    else:
        if p == 0:
            return 1.0
        elif p == 1:
            return 1 - abs(eta/delta)
        elif p == 2:
            if eta >= 0.5*delta:
                temp = eta/delta + 1.5
                return 0.5 * (3 - temp)**2.0
            elif eta < -0.5*delta:
                temp = eta/delta + 1.5
                return 0.5 * temp**2.0
            else:
                temp = eta/delta + 1.5
                return 0.5*(-3.0 + 6.0 * temp - 2 * temp**2.0)
        else:
            print('higher degree B-splines has not been implemented')


    ierr = 0


#========================================================================================
def piecewise_der(p : 'int', delta : 'double', eta : 'double') -> 'double':
    # definition of B-splines defined piecewisely
    # eta is eta_j - eta_k
    if abs(eta) > delta * (p+1)*0.5:
        return 0.0 
    else:
        if p == 0:
            return 0.0
        elif p == 1:
            if eta < 0:
                return 1 / delta
            else:
                return -1 / delta
        elif p == 2:
            if eta >= 0.5 * delta:
                temp = eta / delta + 1.5
                return  - (3 - temp)/delta
            elif eta < -0.5 * delta:
                temp = eta / delta + 1.5
                return  temp / delta
            else:
                temp = eta / delta + 1.5
                return 0.5 * (6.0 - 4.0 * temp) / delta
        else:
            print('higher degree B-splines has not been implemented')


    ierr = 0


#========================================================================================
def convolution(p : 'int', grids : 'double[:]', eta : 'double')  -> 'double':
    # convolution is the function which could give us the B-spline values at evaluatioin point eta
    # p is the degree of the shape function
    # 'grids' is the knots in the support of shape function centered at particle position
    if eta < grids[p+1] and eta >= grids[0]:
        value_stored  = zeros(p + 1, dtype = float)
        value_temp    = zeros(p + 1, dtype = float)
        w             = zeros(p + 1, dtype = float)
        # 0 degree B-spline evluation
        ie           = int( (eta - grids[0]) / (grids[1] - grids[0])) # index of the cell where eta is located
        value_stored[ie] = 1.0                           # value (is 1) in this cell
        result           = value_stored[ie]

        for loop in range(p):

            for ii in range(p - loop + 1):
                w[ii] = (eta - grids[ii]) / (grids[ii + loop + 1] - grids[ii])

            for ii in range(p - loop):
                value_temp[ii] = w[ii] * value_stored[ii] + (1 - w[ii+1]) * value_stored[ii+1]

            value_stored[:] = value_temp

        result = value_stored[0]
    else:
        result = 0.0
    return result

    ierr = 0


#========================================================================================
def convolution_der(p : 'int', grids : 'double[:]', eta : 'double')  -> 'double':
    # convolution is the function which could give us the B-spline values at evaluatioin point eta
    # p is the degree of the shape function
    # 'grids' is the knots in the support of shape function centered at particle position, length is p + 1
    if eta < grids[p+1] and eta >= grids[0]:
        value_stored = zeros(p + 1, dtype = float)
        value_temp   = zeros(p + 1, dtype = float)
        w            = zeros(p + 1, dtype = float)
        # 0 degree B-spline evluation
        ie           = int( (eta - grids[0]) / (grids[1] - grids[0])) # index of the cell where eta is located
        value_stored[ie] = 1.0                           # value (is 1) in this cell
        result           = 0.0

        for loop in range(1, p):

            for ii in range(p - loop + 2):
                w[ii] = (eta - grids[ii]) / (grids[ii + loop] - grids[ii])

            for ii in range(p - loop + 1):
                value_temp[ii] = w[ii] * value_stored[ii] + (1 - w[ii+1]) * value_stored[ii+1]

            value_stored[:] = value_temp


        result = p / (grids[p] - grids[0]) * value_stored[0] - p / (grids[p + 1] - grids[1]) * value_stored[1]
    else:
        result = 0.0
    return result

    ierr = 0
