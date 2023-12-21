# coding: utf-8

"""
Basic functions for point-wise B-spline evaluation
"""
from pyccel.decorators import pure, stack_array

from numpy import empty, zeros


#========================================================================================
@pure
def piecewise(p: 'int', delta: 'float', eta: 'float') -> 'float':
    r"""
    evaluate a hat function (B-spline) centered at eta0 (the center of the support) at eta1, i.e. 
    
    .. math::
        1.0 / delta * S((eta1 - eta0)/ delta)

    where S is the B-spline of degree p, delta is the cell size of a uniform mesh.
    Here we use the expression of the B-spline in each cell directly. For the moment, p = 0, 1, or 2.

    Parameters
    ----------
    p : int
        degree of the hat function (B-spline)

    delta : float
        the cell size of a uniform mesh.

    eta : float
        eta = eta1 - eta0

    Returns:
    --------
        value of the hat function
    """
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
            return -1.0


#========================================================================================
@pure
def piecewise_der(p: 'int', delta: 'float', eta: 'float') -> 'float':
    r"""
    evaluate the derivative of a hat function (B-spline) centered at eta0 (the center of the support) at eta1, i.e. 
    .. math::
        1.0 / delta^2 * S'((eta1 - eta0)/ delta)

    where S is the B-spline of degree p, delta is the cell size of a uniform mesh.
    Here we use the expression of the derivative of the B-spline in each cell directly. For the moment, p = 0, 1, or 2.

    Parameters
    ----------
    p : int
        degree of the hat function (B-spline)

    delta : float
        the cell size of a uniform mesh.

    eta : float
        eta = eta1 - eta0

    Returns:
    --------
        value of the derivative of the hat function
    """
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
            return -1.0






#========================================================================================
@pure
def convolution(p: 'int', grids: 'float[:]', eta: 'float') -> 'float':
    r"""
    evaluate a hat function (B-spline) at eta, i.e. 
    
    .. math::
        1.0 / delta * S(eta/ delta)

    where S is the B-spline of degree p.
    Here we use the definition by convolution of the B-splines.

    Parameters
    ----------
    p : int
        degree of the hat function (B-spline)

    grids : float array
        p + 2 points used in the definition of B-splines.

    eta : float
        evluation point

    Returns:
    --------
        value of the hat function at eta
    """
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



#========================================================================================
@pure
def convolution_der(p: 'int', grids: 'float[:]', eta: 'float') -> 'float':
    r"""
    evaluate the derivative of a hat function (B-spline) at eta, i.e. 
    
    .. math::
        1.0 / delta^2 * S'(eta/ delta)

    where S is the B-spline of degree p.
    Here we use the definition by convolution of the B-splines.

    Parameters
    ----------
    p : int
        degree of the hat function (B-spline)

    grids : float array
        p + 2 points used in the definition of B-splines.

    eta : float
        evluation point

    Returns:
    --------
        value of the derivative of the hat function at eta
    """
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
