# coding: utf-8
#
# Copyright 2020 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Acccelerated functions for point-wise evaluation of tensor product B-splines.

S(eta1, eta2, eta3) = sum_ijk [ c_ijk * B_i(eta1) * B_j(eta2) * B_k(eta3) ] with c_ijk in R.

Possible combinations for tensor product (BBB):
* (NNN)
* (DNN)
* (NDN)
* (NND)
* (NDD)
* (DND)
* (DDN)
* (DDD)
* (dN/deta N N)
* (N dN/deta N)
* (N N dN/deta)
"""

from numpy import empty

import struphy.feec.bsplines_kernels as bsp


# =============================================================================
def evaluation_kernel_3d(p1 : int, p2 : int, p3 : int, basis1 : 'float[:]', basis2 : 'float[:]', basis3 : 'float[:]', ind1 : 'int[:]', ind2 : 'int[:]', ind3 : 'int[:]', coeff : 'float[:,:,:]') -> float:
    """
    Summing non-zero contributions.

    Parameters
    ----------
        p1, p2, p3 : int                 
            Degrees of the univariate splines.
            
        basis1, basis2, basis3 : array[float]           
            The p+1 values of non-zero basis splines at one point (eta1, eta2, eta3) from 'basis_funs' of shape.
            
        ind1, ind2, ind3 : array[int]                 
            Global indices of non-vanishing splines in the element of the considered point.
            
        coeff : array[float]
            The spline coefficients c_ijk. 

    Returns
    -------
        spline_value : float
            Value of tensor-product spline at point (eta1, eta2, eta3).
    """
    
    spline_value = 0.
    
    for il1 in range(p1 + 1):
        i1 = ind1[il1]
        for il2 in range(p2 + 1):
            i2 = ind2[il2]
            for il3 in range(p3 + 1):
                i3 = ind3[il3]
                
                spline_value += coeff[i1, i2, i3] * basis1[il1] * basis2[il2] * basis3[il3]
        
    return spline_value


# =============================================================================
def evaluate(kind1 : int, kind2 : int, kind3 : int, t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p1 : int, p2 : int, p3 : int, ind1 : 'int[:,:]', ind2 : 'int[:,:]', ind3 : 'int[:,:]', coeff : 'float[:,:,:]', eta1 : float, eta2 : float, eta3 : float) -> float:
    """
    Point-wise evaluation of a tensor-product spline. 

    Parameters
    ----------
        kind1, kind2, kind3 : int
            Kind of univariate spline. 1 for B-spline, 2 for M-spline and 3 for derivative of B-spline.
    
        t1, t2, t3 : array[float]
            Knot vectors of univariate splines.
            
        p1, p2, p3 : int                 
            Degrees of univariate splines.
            
        ind1, ind2, ind3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
            
        coeff : array[float]
            The spline coefficients c_ijk. 
            
        eta1, eta2, eta3 : float              
            Point of evaluation.

    Returns
    -------
        spline_value: float
            Value of tensor-product spline at point (eta1, eta2, eta3).
    """

    # find knot span indices
    span1 = bsp.find_span(t1, p1, eta1)
    span2 = bsp.find_span(t2, p2, eta2)
    span3 = bsp.find_span(t3, p3, eta3)

    # evaluate non-vanishing basis functions
    b1 = empty(p1 + 1, dtype=float)
    b2 = empty(p2 + 1, dtype=float)
    b3 = empty(p3 + 1, dtype=float)
    
    bl1 = empty(p1, dtype=float)
    bl2 = empty(p2, dtype=float)
    bl3 = empty(p3, dtype=float)
    
    br1 = empty(p1, dtype=float)
    br2 = empty(p2, dtype=float)
    br3 = empty(p3, dtype=float)

    # 1st direction
    if   kind1 == 1:
        bsp.basis_funs(t1, p1, eta1, span1, bl1, br1, b1)
    elif kind1 == 2:
        bsp.basis_funs(t1, p1, eta1, span1, bl1, br1, b1)
        bsp.scaling(t1, p1, span1, b1)
    elif kind1 == 3:
        bsp.basis_funs_1st_der(t1, p1, eta1, span1, bl1, br1, b1)
    
    # 2nd direction
    if   kind2 == 1:
        bsp.basis_funs(t2, p2, eta2, span2, bl2, br2, b2)
    elif kind2 == 2:
        bsp.basis_funs(t2, p2, eta2, span2, bl2, br2, b2)
        bsp.scaling(t2, p2, span2, b2)   
    elif kind2 == 3:
        bsp.basis_funs_1st_der(t2, p2, eta2, span2, bl2, br2, b2)
        
    # 3rd direction
    if   kind3 == 1:
        bsp.basis_funs(t3, p3, eta3, span3, bl3, br3, b3)
    elif kind3 == 2:
        bsp.basis_funs(t3, p3, eta3, span3, bl3, br3, b3)
        bsp.scaling(t3, p3, span3, b3)    
    elif kind3 == 3:
        bsp.basis_funs_1st_der(t3, p3, eta3, span3, bl3, br3, b3)
    
    # sum up non-vanishing contributions
    spline_value = evaluation_kernel_3d(p1, p2, p3, b1, b2, b3, ind1[span1 - p1, :], ind2[span2 - p2, :], ind3[span3 - p3, :], coeff)

    return spline_value


# =============================================================================
def evaluate_tensor_product(t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p1 : int, p2 : int, p3 : int, ind1 : 'int[:,:]', ind2 : 'int[:,:]', ind3 : 'int[:,:]', coeff : 'float[:,:,:]', eta1 : 'float[:]', eta2 : 'float[:]', eta3 : 'float[:]', spline_values : 'float[:,:,:]', kind : int):
    """
    Tensor-product evaluation of a tensor-product spline. 

    Parameters
    ----------
        t1, t2, t3 : array[float]
            Knot vectors of univariate splines.
            
        p1, p2, p3 : int                 
            Degrees of univariate splines.
            
        ind1, ind2, ind3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
            
        coeff : array[float]
            The spline coefficients c_ijk. 
            
        eta1, eta2, eta3 : array[float]              
            Points of evaluation in 1d arrays.
            
        spline_values : array[float]
            Splines evaluated at points S_ijk = S(eta1_i, eta2_j, eta3_k).
            
        kind : int
            Kind of spline to evaluate.
                * 0  : NNN
                * 11 : DNN
                * 12 : NDN
                * 13 : NND
                * 21 : NDD
                * 22 : DND
                * 23 : DDN
                * 3  : DDD
                * 41 : dN/deta N N
                * 42 : N dN/deta N
                * 43 : N N dN/deta
    """
    
    for i1 in range(len(eta1)):
        for i2 in range(len(eta2)):
            for i3 in range(len(eta3)):
                
                if   kind == 0:
                    spline_values[i1, i2, i3] = evaluate(1, 1, 1, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1], eta2[i2], eta3[i3])
                elif kind == 11:
                    spline_values[i1, i2, i3] = evaluate(2, 1, 1, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1], eta2[i2], eta3[i3])
                elif kind == 12:
                    spline_values[i1, i2, i3] = evaluate(1, 2, 1, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1], eta2[i2], eta3[i3])
                elif kind == 13:
                    spline_values[i1, i2, i3] = evaluate(1, 1, 2, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1], eta2[i2], eta3[i3])
                elif kind == 21:
                    spline_values[i1, i2, i3] = evaluate(1, 2, 2, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1], eta2[i2], eta3[i3])
                elif kind == 22:
                    spline_values[i1, i2, i3] = evaluate(2, 1, 2, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1], eta2[i2], eta3[i3])
                elif kind == 23:
                    spline_values[i1, i2, i3] = evaluate(2, 2, 1, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1], eta2[i2], eta3[i3])
                elif kind == 3:
                    spline_values[i1, i2, i3] = evaluate(2, 2, 2, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1], eta2[i2], eta3[i3])           
                elif kind == 41:
                    spline_values[i1, i2, i3] = evaluate(3, 1, 1, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1], eta2[i2], eta3[i3])
                elif kind == 42:
                    spline_values[i1, i2, i3] = evaluate(1, 3, 1, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1], eta2[i2], eta3[i3])
                elif kind == 43:
                    spline_values[i1, i2, i3] = evaluate(1, 1, 3, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1], eta2[i2], eta3[i3])
                        
                    
# =============================================================================
def evaluate_matrix(t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p1 : int,  p2 : int, p3 : int, ind1 : 'int[:,:]', ind2 : 'int[:,:]', ind3 : 'int[:,:]', coeff : 'float[:,:,:]', eta1 : 'float[:,:,:]', eta2 : 'float[:,:,:]', eta3 : 'float[:,:,:]', spline_values : 'float[:,:,:]', kind : int):
    """
    General evaluation of a tensor-product spline. 

    Parameters
    ----------
        t1, t2, t3 : array[float]
            Knot vectors of univariate splines.
            
        p1, p2, p3 : int                 
            Degrees of univariate splines.
            
        ind1, ind2, ind3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
            
        coeff : array[float]
            The spline coefficients c_ijk. 
            
        eta1, eta2, eta3 : array[float]              
            Points of evaluation in 3d arrays such that shape(eta1) == shape(eta2) == shape(eta3).
            
        spline_values : array[float]
            Splines evaluated at points S_ijk = S(eta1_i, eta2_j, eta3_k).
            
        kind : int
            Kind of spline to evaluate.
                * 0  : NNN
                * 11 : DNN
                * 12 : NDN
                * 13 : NND
                * 21 : NDD
                * 22 : DND
                * 23 : DDN
                * 3  : DDD
                * 41 : dN/deta N N
                * 42 : N dN/deta N
                * 43 : N N dN/deta
    """
    
    from numpy import shape
    
    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]
    
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                
                if   kind == 0:
                    spline_values[i1, i2, i3] = evaluate(1, 1, 1, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3])
                elif kind == 11:
                    spline_values[i1, i2, i3] = evaluate(2, 1, 1, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3])
                elif kind == 12:
                    spline_values[i1, i2, i3] = evaluate(1, 2, 1, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3])
                elif kind == 13:
                    spline_values[i1, i2, i3] = evaluate(1, 1, 2, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3])
                elif kind == 21:
                    spline_values[i1, i2, i3] = evaluate(1, 2, 2, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3])
                elif kind == 22:
                    spline_values[i1, i2, i3] = evaluate(2, 1, 2, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3])
                elif kind == 23:
                    spline_values[i1, i2, i3] = evaluate(2, 2, 1, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3])
                elif kind == 3:
                    spline_values[i1, i2, i3] = evaluate(2, 2, 2, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3])           
                elif kind == 41:
                    spline_values[i1, i2, i3] = evaluate(3, 1, 1, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3])
                elif kind == 42:
                    spline_values[i1, i2, i3] = evaluate(1, 3, 1, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3])
                elif kind == 43:
                    spline_values[i1, i2, i3] = evaluate(1, 1, 3, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3])


# =============================================================================
def evaluate_sparse(t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p1 : int, p2 : int, p3 : int, ind1 : 'int[:,:]', ind2 : 'int[:,:]', ind3 : 'int[:,:]', coeff : 'float[:,:,:]', eta1 : 'float[:,:,:]', eta2 : 'float[:,:,:]', eta3 : 'float[:,:,:]', spline_values : 'float[:,:,:]', kind : int):
    """
    Evaluation of a tensor-product spline using sparse meshgrids. 

    Parameters
    ----------
        t1, t2, t3 : array[float]
            Knot vectors of univariate splines.
            
        p1, p2, p3 : int                 
            Degrees of univariate splines.
            
        ind1, ind2, ind3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
            
        coeff : array[float]
            The spline coefficients c_ijk. 
            
        eta1, eta2, eta3 : array[float]              
            Points of evaluation in 3d arrays such that shape(eta1) = (:,1,1), shape(eta2) = (1,:,1), shape(eta3) = (1,1,:).
            
        spline_values : array[float]
            Splines evaluated at points S_ijk = S(eta1_i, eta2_j, eta3_k).
            
        kind : int
            Kind of spline to evaluate.
                * 0  : NNN
                * 11 : DNN
                * 12 : NDN
                * 13 : NND
                * 21 : NDD
                * 22 : DND
                * 23 : DDN
                * 3  : DDD
                * 41 : dN/deta N N
                * 42 : N dN/deta N
                * 43 : N N dN/deta
    """
    
    from numpy import shape
    
    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]
    
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                
                if   kind == 0:
                    spline_values[i1, i2, i3] = evaluate(1, 1, 1, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1, 0, 0], eta2[0, i2, 0], eta3[0, 0, i3])
                elif kind == 11:
                    spline_values[i1, i2, i3] = evaluate(2, 1, 1, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1, 0, 0], eta2[0, i2, 0], eta3[0, 0, i3])
                elif kind == 12:
                    spline_values[i1, i2, i3] = evaluate(1, 2, 1, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1, 0, 0], eta2[0, i2, 0], eta3[0, 0, i3])
                elif kind == 13:
                    spline_values[i1, i2, i3] = evaluate(1, 1, 2, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1, 0, 0], eta2[0, i2, 0], eta3[0, 0, i3])
                elif kind == 21:
                    spline_values[i1, i2, i3] = evaluate(1, 2, 2, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1, 0, 0], eta2[0, i2, 0], eta3[0, 0, i3])
                elif kind == 22:
                    spline_values[i1, i2, i3] = evaluate(2, 1, 2, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1, 0, 0], eta2[0, i2, 0], eta3[0, 0, i3])
                elif kind == 23:
                    spline_values[i1, i2, i3] = evaluate(2, 2, 1, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1, 0, 0], eta2[0, i2, 0], eta3[0, 0, i3])
                elif kind == 3:
                    spline_values[i1, i2, i3] = evaluate(2, 2, 2, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1, 0, 0], eta2[0, i2, 0], eta3[0, 0, i3])           
                elif kind == 41:
                    spline_values[i1, i2, i3] = evaluate(3, 1, 1, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1, 0, 0], eta2[0, i2, 0], eta3[0, 0, i3])
                elif kind == 42:
                    spline_values[i1, i2, i3] = evaluate(1, 3, 1, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1, 0, 0], eta2[0, i2, 0], eta3[0, 0, i3])
                elif kind == 43:
                    spline_values[i1, i2, i3] = evaluate(1, 1, 3, t1, t2, t3, p1, p2, p3, ind1, ind2, ind3, coeff, eta1[i1, 0, 0], eta2[0, i2, 0], eta3[0, 0, i3])
                
