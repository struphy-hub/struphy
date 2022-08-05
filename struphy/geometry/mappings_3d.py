# coding: utf-8
#
# Copyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Module containing accelerated (pyccelized) functions for evaluation of metric coefficients corresponding to 3d mappings x_i = F(eta_1, eta_2, eta_3):

- f_i, f            : mapping                 f_i,
- df_ij, df         : Jacobian matrix         df_i/deta_j,
- det_df            : Jacobian determinant    det(df),
- df_inv_ij, df_inv : inverse Jacobian matrix (df_i/deta_j)^(-1),
- g_ij, g           : metric tensor           df^T * df,
- g_inv, g_inv_ij   : inverse metric tensor   df^(-1) * df^(-T),

The following mappings are implemented:

- kind_map = 0  : 3d spline mapping with control points cx, cy, cz
- kind_map = 1  : 2d spline mapping with control points cx, cy: F_pol = (eta_1, eta_2) --> (R, y), straight  in 3rd direction
- kind_map = 2  : 2d spline mapping with control points cx, cy: F_pol = (eta_1, eta_2) --> (R, y), curvature in 3rd direction

- kind_map = 10 : cuboid,             params_map = [l1, r1, l2, r2, l3, r3].
- kind_map = 11 : hollow cylinder,    params_map = [a1, a2, R0].
- kind_map = 12 : colella,            params_map = [Lx, Ly, alpha, Lz].
- kind_map = 13 : orthogonal,         params_map = [Lx, Ly, alpha, Lz].
- kind_map = 14 : hollow torus,       params_map = [a1, a2, R0].
- kind_map = 15 : ellipse,            params_map = [x0, y0, z0, rx, ry, Lz].
- kind_map = 16 : rotated ellipse,    params_map = [x0, y0, z0, r1, r2, Lz, th].
- kind_map = 17 : shafranov shift,    params_map = [x0, y0, z0, rx, ry, Lz, delta].
- kind_map = 18 : shafranov sqrt,     params_map = [x0, y0, z0, rx, ry, Lz, delta].
- kind_map = 19 : shafranov D-shaped, params_map = [x0, y0, z0, R0, Lz, delta_x, delta_y, delta_gs, epsilon_gs, kappa_gs].
"""

from numpy import shape, empty, zeros
from numpy import sin, cos, pi, sqrt, arctan2, arcsin

import struphy.feec.bsplines_kernels as bsp

import struphy.feec.basics.spline_evaluation_2d as eva_2d
import struphy.feec.basics.spline_evaluation_3d as eva_3d

from struphy.linear_algebra.core import det


def f_i(eta1 : float, eta2 : float, eta3 : float, component : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float:
    """
    Point-wise evaluation of Cartesian coordinate x_i = f_i(eta1, eta2, eta3), i=1,2,3. 

    Parameters
    ----------
        eta1, eta2, eta3 : float              
            Logical coordinates in [0, 1].
        
        component : int                 
            Cartesian coordinate (1 : x, 2 : y, 3 : z) to evaluate.
            
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params_map : array[float]
            Parameters for the mapping in a 1d array.
        
        tn1, tn2, tn3 : array[float]          
            Knot vectors of univariate splines.
        
        pn : array[int]
            Degrees of univariate splines [pn1, pn2, pn3].
        
        ind_n1, ind_n2, ind_n3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (f_1, f_2, f_3) in case of a IGA mapping.

    Returns
    -------
        value : float
            Cartesian coordinate x_i = f_i(eta1, eta2, eta3).
    """

    value = 0.

    # --------------- 3d spline --------------------
    if kind_map == 0:

        if   component == 1:
            value = eva_3d.evaluate(1, 1, 1, tn1, tn2, tn3, pn[0], pn[1], pn[2], ind_n1, ind_n2, ind_n3, cx, eta1, eta2, eta3)

        elif component == 2:
            value = eva_3d.evaluate(1, 1, 1, tn1, tn2, tn3, pn[0], pn[1], pn[2], ind_n1, ind_n2, ind_n3, cy, eta1, eta2, eta3)

        elif component == 3:
            value = eva_3d.evaluate(1, 1, 1, tn1, tn2, tn3, pn[0], pn[1], pn[2], ind_n1, ind_n2, ind_n3, cz, eta1, eta2, eta3)

    # ---- 2d spline (straight in 3rd direction) ---
    elif kind_map == 1:

        lz = 2*pi*cx[0, 0, 0]

        if   component == 1:
            value = eva_2d.evaluate(1, 1, tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cx[:, :, 0], eta1, eta2)

            if eta1 == 0. and cx[0, 0, 0] == cx[0, 1, 0]:
                value = cx[0, 0, 0]

        elif component == 2:
            value = eva_2d.evaluate(1, 1, tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cy[:, :, 0], eta1, eta2)

            if eta1 == 0. and cy[0, 0, 0] == cy[0, 1, 0]:
                value = cy[0, 0, 0]

        elif component == 3:
            value = lz * eta3

    # ---- 2d spline (curvature in 3rd direction) ---
    elif kind_map == 2:

        if   component == 1:
            value = eva_2d.evaluate(1, 1, tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cx[:, :, 0], eta1, eta2) * cos(2*pi*eta3)

            if eta1 == 0. and cx[0, 0, 0] == cx[0, 1, 0]:
                value = cx[0, 0, 0]*cos(2*pi*eta3)

        elif component == 2:
            value = eva_2d.evaluate(1, 1, tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cy[:, :, 0], eta1, eta2)

            if eta1 == 0. and cy[0, 0, 0] == cy[0, 1, 0]:
                value = cy[0, 0, 0]

        elif component == 3:
            value = eva_2d.evaluate(1, 1, tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cx[:, :, 0], eta1, eta2) * sin(2*pi*eta3)

            if eta1 == 0. and cx[0, 0, 0] == cx[0, 1, 0]:
                value = cx[0, 0, 0]*sin(2*pi*eta3)

    # -------------- cuboid -------------------------
    elif kind_map == 10:

        b1 = params_map[0]
        e1 = params_map[1]
        b2 = params_map[2]
        e2 = params_map[3]
        b3 = params_map[4]
        e3 = params_map[5]

        # value =  begin + (end - begin) * eta
        if   component == 1:
            value = b1 + (e1 - b1) * eta1
        elif component == 2:
            value = b2 + (e2 - b2) * eta2
        elif component == 3:
            value = b3 + (e3 - b3) * eta3

    # --------- hollow cylinder ---------------------
    elif kind_map == 11:

        a1 = params_map[0]
        a2 = params_map[1]
        r0 = params_map[2]
        lz = params_map[3]

        da = a2 - a1

        if   component == 1:
            value = (a1 + eta1 * da) * cos(2*pi*eta2) + r0
        elif component == 2:
            value = (a1 + eta1 * da) * sin(2*pi*eta2)
        elif component == 3:
            value = lz * eta3

    # ------------ colella --------------------------
    elif kind_map == 12:

        lx    = params_map[0]
        ly    = params_map[1]
        alpha = params_map[2]
        lz    = params_map[3]

        if   component == 1:
            value = lx * (eta1 + alpha * sin(2*pi*eta1) * sin(2*pi*eta2))
        elif component == 2:
            value = ly * (eta2 + alpha * sin(2*pi*eta1) * sin(2*pi*eta2))
        elif component == 3:
            value = lz * eta3

    # ----------- orthogonal ------------------------
    elif kind_map == 13:

        lx    = params_map[0]
        ly    = params_map[1]
        alpha = params_map[2]
        lz    = params_map[3]

        if   component == 1:
            value = lx * (eta1 + alpha * sin(2*pi*eta1))
        elif component == 2:
            value = ly * (eta2 + alpha * sin(2*pi*eta2))
        elif component == 3:
            value = lz * eta3

    # --------- hollow torus ------------------------
    elif kind_map == 14:

        a1 = params_map[0]
        a2 = params_map[1]
        r0 = params_map[2]

        da = a2 - a1

        if   component == 1:
            value = ((a1 + eta1 * da) * cos(2*pi*eta2) + r0) * cos(2*pi*eta3)
        elif component == 2:
            value =  (a1 + eta1 * da) * sin(2*pi*eta2)
        elif component == 3:
            value = ((a1 + eta1 * da) * cos(2*pi*eta2) + r0) * sin(2*pi*eta3)

    # ------------- ellipse -------------------------
    elif kind_map == 15:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        lz = params_map[5]

        if   component == 1:
            value = x0 + (eta1 * rx) * cos(2*pi*eta2)
        elif component == 2:
            value = y0 + (eta1 * ry) * sin(2*pi*eta2)
        elif component == 3:
            value = z0 + (eta3 * lz)

    # --------- rotated ellipse ---------------------
    elif kind_map == 16:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        r1 = params_map[3]
        r2 = params_map[4]
        lz = params_map[5]
        th = params_map[6] # Domain: [0,1)

        if   component == 1:
            value = x0 + (eta1 * r1) * cos(2*pi*th) * cos(2*pi*eta2) - (eta1 * r2) * sin(2*pi*th) * sin(2*pi*eta2)
        elif component == 2:
            value = y0 + (eta1 * r1) * sin(2*pi*th) * cos(2*pi*eta2) + (eta1 * r2) * cos(2*pi*th) * sin(2*pi*eta2)
        elif component == 3:
            value = z0 + (eta3 * lz)

    # --------- ellipse with power ---------------------
    elif kind_map == 17:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        lz = params_map[5]
        s  = params_map[6]

        if   component == 1:
            value = x0 + (eta1**s) * rx * cos(2*pi*eta2)
        elif component == 2:
            value = y0 + (eta1**s) * ry * sin(2*pi*eta2)
        elif component == 3:
            value = z0 + (eta3 * lz)

    # --------- shafranov shift ---------------------
    elif kind_map == 18:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        lz = params_map[5]
        de = params_map[6] # Domain: [0,0.1]

        if   component == 1:
            value = x0 + (eta1 * rx) * cos(2*pi*eta2) + (1-eta1**2) * rx * de
        elif component == 2:
            value = y0 + (eta1 * ry) * sin(2*pi*eta2)
        elif component == 3:
            value = z0 + (eta3 * lz)

    # --------- shafranov sqrt ---------------------
    elif kind_map == 19:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        lz = params_map[5]
        de = params_map[6] # Domain: [0,0.1]

        if   component == 1:
            value = x0 + (eta1 * rx) * cos(2*pi*eta2) + (1-sqrt(eta1)) * rx * de
        elif component == 2:
            value = y0 + (eta1 * ry) * sin(2*pi*eta2)
        elif component == 3:
            value = z0 + (eta3 * lz)

    # --------- shafranov D-shaped ---------------------
    elif kind_map == 20:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        r0 = params_map[3]
        lz = params_map[4]
        dx = params_map[5] # Grad-Shafranov shift along x-axis.
        dy = params_map[6] # Grad-Shafranov shift along y-axis.
        dg = params_map[7] # Delta = sin(alpha): Triangularity, shift of high point.
        eg = params_map[8] # Epsilon: Inverse aspect ratio a/r0.
        kg = params_map[9] # Kappa: Ellipticity (elongation).

        if   component == 1:
            value = x0 + r0 * (1 + (1 - eta1**2) * dx + eg *      eta1 * cos(2*pi*eta2 + arcsin(dg)*eta1*sin(2*pi*eta2)))
        elif component == 2:
            value = y0 + r0 * (    (1 - eta1**2) * dy + eg * kg * eta1 * sin(2*pi*eta2))
        elif component == 3:
            value = z0 + (eta3 * lz)

    # --------- shafranov D-shaped with eta3 dependence ---------------------
    elif kind_map == 21:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        r0 = params_map[3]
        lz = params_map[4]
        dx = params_map[5]  # Grad-Shafranov shift along x-axis.
        dy = params_map[6]  # Grad-Shafranov shift along y-axis.
        dg = params_map[7]  # Delta = sin(alpha): Triangularity, shift of high point.
        eg = params_map[8]  # Epsilon: Inverse aspect ratio a/r0.
        kg = params_map[9]  # Kappa: Ellipticity (elongation).
        xi = params_map[10] # Xi: Strength of dependence on eta3.

        if   component == 1:
            value = x0 + r0 * (1 + xi * cos(2*pi*eta3)) * (1 + (1 - eta1**2) * dx + eg *      eta1 * cos(2*pi*eta2 + arcsin(dg)*eta1*sin(2*pi*eta2)))
        elif component == 2:
            value = y0 + r0 * (1 - xi * cos(2*pi*eta3)) * (    (1 - eta1**2) * dy + eg * kg * eta1 * sin(2*pi*eta2))
        elif component == 3:
            value = z0 + (eta3 * lz)

    return value


def f(eta1 : float, eta2 : float, eta3 : float, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', f_out : 'float[:]'):
    """
    Point-wise evaluation of all Cartesian coordinates x_i = f_i(eta1, eta2, eta3), i=1,2,3. 

    Parameters
    ----------
        eta1, eta2, eta3 : float              
            Logical coordinates in [0, 1].
            
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params_map : array[float]
            Parameters for the mapping in a 1d array.
        
        tn1, tn2, tn3 : array[float]          
            Knot vectors of univariate splines.
        
        pn : array[int]
            Degrees of univariate splines [pn1, pn2, pn3].
        
        ind_n1, ind_n2, ind_n3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (f_1, f_2, f_3) in case of a IGA mapping.
            
        f_out : array[float]    
            Evaluated Cartesian coordinates x, y, z = f(eta1, eta2, eta3). 
    """
    
    f_out[0] = f_i(eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    f_out[1] = f_i(eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    f_out[2] = f_i(eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    
    
def f_inv_i(x : float, y : float, z : float, component : int, kind_map : int, params_map : 'float[:]') -> float:
    """
    Point-wise evaluation of inverse mapping eta_i = f^(-1)_i(x, y, z), i=1,2,3. Only possible for analytical mappings.
    
    Parameters
    ----------
        x, y, z : float              
            Cartesian coordinates.
            
        component : int
            Logical coordinate to evaluate (1 : eta_1, 2 : eta_2, 3 : eta_3).
            
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params_map : array[float]
            Parameters for the mapping in a 1d array.
        
    Returns
    -------
        value : float
            The logical coordinate in [0, 1].
    """

    value = 0.

    # -------------- cuboid -------------------------
    if kind_map == 10:

        b1 = params_map[0]
        e1 = params_map[1]
        b2 = params_map[2]
        e2 = params_map[3]
        b3 = params_map[4]
        e3 = params_map[5]

        # value =  begin + (end - begin) * eta
        if   component == 1:
            value = (x - b1)/(e1 - b1)
        elif component == 2:
            value = (y - b2)/(e2 - b2)
        elif component == 3:
            value = (z - b3)/(e3 - b3)

    # --------- hollow cylinder ---------------------
    elif kind_map == 11:

        a1 = params_map[0]
        a2 = params_map[1]
        r0 = params_map[2]
        lz = params_map[3]

        da = a2 - a1

        if   component == 1:
            value = (sqrt((x - r0)**2 + y**2) - a1)/da
        elif component == 2:
            value = (arctan2(y, x - r0)/(2*pi))%1.0
        elif component == 3:
            value = z / lz

    # --------- hollow torus ------------------------
    elif kind_map == 14:

        a1 = params_map[0]
        a2 = params_map[1]
        r0 = params_map[2]

        da = a2 - a1

        if   component == 1:
            value = (sqrt((sqrt(x**2 + z**2) - r0)**2 + y**2) - a1)/da
        elif component == 2:
            value = (arctan2(y, sqrt(x**2 + z**2) - r0)/(2*pi))%1.0
        elif component == 3:
            value = (arctan2(z, x)/(2*pi))%1.0

    # --------------- ellipse -----------------------
    elif kind_map == 15:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        lz = params_map[5]

        if   component == 1:
            eta2  = (arctan2((y - y0) * rx, (x - x0) * ry) / (2 * pi)) % 1.0
            # value = sqrt(((x - x0)**2 + (y - y0)**2) / (rx**2 * cos(2*pi*eta2)**2 + ry**2 * sin(2*pi*eta2)**2))
            value = (x - x0) / (rx * cos(2*pi*eta2)) # Equivalent.
        elif component == 2:
            value = (arctan2((y - y0) * rx, (x - x0) * ry) / (2 * pi)) % 1.0
        elif component == 3:
            value = (z - z0) / lz

    # ----------- rotated ellipse ---------------------
    elif kind_map == 16:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        r1 = params_map[3]
        r2 = params_map[4]
        lz = params_map[5]
        th = params_map[6] # Domain: [0,1)

        if   component == 1:
            temp1 =  cos(2*pi*th) * (x - x0) + sin(2*pi*th) * (y - y0)
            temp2 = -sin(2*pi*th) * (x - x0) + cos(2*pi*th) * (y - y0)
            eta2  = (arctan2(temp2 * r1, temp1 * r2) / (2 * pi)) % 1.0
            value = temp1 / (r1 * cos(2*pi*eta2))
        elif component == 2:
            temp1 =  cos(2*pi*th) * (x - x0) + sin(2*pi*th) * (y - y0)
            temp2 = -sin(2*pi*th) * (x - x0) + cos(2*pi*th) * (y - y0)
            value = (arctan2(temp2 * r1, temp1 * r2) / (2 * pi)) % 1.0
        elif component == 3:
            value = (z - z0) / lz

    # ----------- ellipse with power ---------------------
    elif kind_map == 17:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        lz = params_map[5]
        s  = params_map[6]

        if   component == 1:
            value = 0. # Not implemented.
        elif component == 2:
            value = 0. # Not implemented.
        elif component == 3:
            value = (z - z0) / lz

    # ----------- shafranov shift ---------------------
    elif kind_map == 18:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        lz = params_map[5]
        de = params_map[6] # Domain: [0,0.1]

        if   component == 1:
            value = 0. # Not implemented.
        elif component == 2:
            value = 0. # Not implemented.
        elif component == 3:
            value = (z - z0) / lz

    # ----------- shafranov sqrt ---------------------
    elif kind_map == 19:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        lz = params_map[5]
        de = params_map[6] # Domain: [0,0.1]

        if   component == 1:
            value = 0. # Not implemented.
        elif component == 2:
            value = 0. # Not implemented.
        elif component == 3:
            value = (z - z0) / lz

    # ----------- shafranov D-shaped ---------------------
    elif kind_map == 20:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        r0 = params_map[3]
        Lz = params_map[4]
        dx = params_map[5] # Grad-Shafranov shift along x-axis.
        dy = params_map[6] # Grad-Shafranov shift along y-axis.
        dg = params_map[7] # Delta = sin(alpha): Triangularity, shift of high point.
        eg = params_map[8] # Epsilon: Inverse aspect ratio a/r0.
        kg = params_map[9] # Kappa: Ellipticity (elongation).

        if   component == 1:
            value = 0. # Not implemented.
        elif component == 2:
            value = 0. # Not implemented.
        elif component == 3:
            value = (z - z0) / lz

    # ----------- shafranov D-shaped with eta3 dependence ---------------------
    elif kind_map == 21:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        r0 = params_map[3]
        Lz = params_map[4]
        dx = params_map[5]  # Grad-Shafranov shift along x-axis.
        dy = params_map[6]  # Grad-Shafranov shift along y-axis.
        dg = params_map[7]  # Delta = sin(alpha): Triangularity, shift of high point.
        eg = params_map[8]  # Epsilon: Inverse aspect ratio a/r0.
        kg = params_map[9]  # Kappa: Ellipticity (elongation).
        xi = params_map[10] # Xi: Strength of dependence on eta3.

        if   component == 1:
            value = 0. # Not implemented.
        elif component == 2:
            value = 0. # Not implemented.
        elif component == 3:
            value = (z - z0) / lz

    return value


def f_inv(x : float, y : float, z : float, kind_map : int, params_map : 'float[:]', f_inv_out : 'float[:]'):
    """
    Point-wise evaluation of all inverse mapping components eta_i = f^(-1)_i(x, y, z), i=1,2,3. Only possible for analytical mappings.
    
    Parameters
    ----------
        x, y, z : float              
            Cartesian coordinates.
            
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params_map : array[float]
            Parameters for the mapping in a 1d array.
        
        f_inv_out : array[float]
            Evaluated logical coordinates eta_1, eta_2, eta_3] = f^(-1)_i(x, y, z).
    """
    
    f_inv_out[0] = f_inv_i(x, y, z, 1, kind_map, params_map)
    f_inv_out[1] = f_inv_i(x, y, z, 2, kind_map, params_map)
    f_inv_out[2] = f_inv_i(x, y, z, 3, kind_map, params_map)
    
    
def df_ij(eta1 : float, eta2 : float, eta3 : float, component : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float:
    """
    Point-wise evaluation of ij-th component of the Jacobian matrix df_ij = df_i/deta_j (i,j=1,2,3). 

    Parameters
    ----------
        eta1, eta2, eta3 : float              
            Logical coordinates in [0, 1].
        
        component : int                 
            Component of Jacobian matrix (11, 12, ..., 32, 33) to evaluate.
            
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params_map : array[float]
            Parameters for the mapping in a 1d array.
        
        tn1, tn2, tn3 : array[float]          
            Knot vectors of univariate splines.
        
        pn : array[int]
            Degrees of univariate splines [pn1, pn2, pn3].
        
        ind_n1, ind_n2, ind_n3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (f_1, f_2, f_3) in case of a IGA mapping.

    Returns
    --------
        value : float
            The ij-th component of the Jacobian matrix.
    """

    value = 0.

    # ----------- 3d spline ------------------------
    if kind_map == 0:
        
        if   component == 11:
            value = eva_3d.evaluate(3, 1, 1, tn1, tn2, tn3, pn[0], pn[1], pn[2], ind_n1, ind_n2, ind_n3, cx, eta1, eta2, eta3)
        elif component == 12:
            value = eva_3d.evaluate(1, 3, 1, tn1, tn2, tn3, pn[0], pn[1], pn[2], ind_n1, ind_n2, ind_n3, cx, eta1, eta2, eta3)
        elif component == 13:
            value = eva_3d.evaluate(1, 1, 3, tn1, tn2, tn3, pn[0], pn[1], pn[2], ind_n1, ind_n2, ind_n3, cx, eta1, eta2, eta3)
        elif component == 21:
            value = eva_3d.evaluate(3, 1, 1, tn1, tn2, tn3, pn[0], pn[1], pn[2], ind_n1, ind_n2, ind_n3, cy, eta1, eta2, eta3)
        elif component == 22:
            value = eva_3d.evaluate(1, 3, 1, tn1, tn2, tn3, pn[0], pn[1], pn[2], ind_n1, ind_n2, ind_n3, cy, eta1, eta2, eta3)
        elif component == 23:
            value = eva_3d.evaluate(1, 1, 3, tn1, tn2, tn3, pn[0], pn[1], pn[2], ind_n1, ind_n2, ind_n3, cy, eta1, eta2, eta3)
        elif component == 31:
            value = eva_3d.evaluate(3, 1, 1, tn1, tn2, tn3, pn[0], pn[1], pn[2], ind_n1, ind_n2, ind_n3, cz, eta1, eta2, eta3)
        elif component == 32:
            value = eva_3d.evaluate(1, 3, 1, tn1, tn2, tn3, pn[0], pn[1], pn[2], ind_n1, ind_n2, ind_n3, cz, eta1, eta2, eta3)
        elif component == 33:
            value = eva_3d.evaluate(1, 1, 3, tn1, tn2, tn3, pn[0], pn[1], pn[2], ind_n1, ind_n2, ind_n3, cz, eta1, eta2, eta3)
               
    # ----- 2d spline (straight in 3rd direction) ---
    elif kind_map == 1:
        
        lz = 2*pi*cx[0, 0, 0]
        
        if   component == 11:
            value = eva_2d.evaluate(3, 1, tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cx[:, :, 0], eta1, eta2)
        elif component == 12:
            value = eva_2d.evaluate(1, 3, tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cx[:, :, 0], eta1, eta2)
            
            if eta1 == 0. and cx[0, 0, 0] == cx[0, 1, 0]:
                value = 0.
            
        elif component == 13:
            value = 0.
        elif component == 21:
            value = eva_2d.evaluate(3, 1, tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cy[:, :, 0], eta1, eta2)
        elif component == 22:
            value = eva_2d.evaluate(1, 3, tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cy[:, :, 0], eta1, eta2)
            
            if eta1 == 0. and cy[0, 0, 0] == cy[0, 1, 0]:
                value = 0.
            
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.
        elif component == 33:
            value = lz
    
    # ---- 2d spline (curvature in 3rd direction) ---
    elif kind_map == 2:
        
        if   component == 11:
            value = eva_2d.evaluate(3, 1, tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cx[:, :, 0], eta1, eta2) * cos(2*pi*eta3)
        elif component == 12:
            value = eva_2d.evaluate(1, 3, tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cx[:, :, 0], eta1, eta2) * cos(2*pi*eta3)
            
            if eta1 == 0. and cx[0, 0, 0] == cx[0, 1, 0]:
                value = 0.
            
        elif component == 13:
            value = eva_2d.evaluate(1, 1, tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cx[:, :, 0], eta1, eta2) * sin(2*pi*eta3) * (-2*pi)
        elif component == 21:
            value = eva_2d.evaluate(3, 1, tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cy[:, :, 0], eta1, eta2)
        elif component == 22:
            value = eva_2d.evaluate(1, 3, tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cy[:, :, 0], eta1, eta2)
            
            if eta1 == 0. and cy[0, 0, 0] == cy[0, 1, 0]:
                value = 0.
            
        elif component == 23:
            value = 0.
        elif component == 31:
            value = eva_2d.evaluate(3, 1, tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cx[:, :, 0], eta1, eta2) * sin(2*pi*eta3)
        elif component == 32:
            value = eva_2d.evaluate(1, 3, tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cx[:, :, 0], eta1, eta2) * sin(2*pi*eta3)
            
            if eta1 == 0. and cx[0, 0, 0] == cx[0, 1, 0]:
                value = 0.
            
        elif component == 33:
            value = eva_2d.evaluate(1, 1, tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cx[:, :, 0], eta1, eta2) * cos(2*pi*eta3) * 2*pi
    
    # ---------------- cuboid -----------------------
    elif kind_map == 10:
         
        b1 = params_map[0]
        e1 = params_map[1]
        b2 = params_map[2]
        e2 = params_map[3]
        b3 = params_map[4]
        e3 = params_map[5]
        
        if   component == 11:
            value = e1 - b1
        elif component == 12:
            value = 0.
        elif component == 13:
            value = 0.
        elif component == 21:
            value = 0.
        elif component == 22:
            value = e2 - b2
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.
        elif component == 33:
            value = e3 - b3

    # ------------ hollow cylinder -------------------
    elif kind_map == 11:

        a1 = params_map[0]
        a2 = params_map[1]
        r0 = params_map[2]
        lz = params_map[3]

        da = a2 - a1

        if   component == 11:
            value = da * cos(2*pi*eta2)
        elif component == 12:
            value = -2*pi * (a1 + eta1 * da) * sin(2*pi*eta2)
        elif component == 13:
            value = 0.
        elif component == 21:
            value = da * sin(2*pi*eta2)
        elif component == 22:
            value = 2*pi * (a1 + eta1 * da) * cos(2*pi*eta2)
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.
        elif component == 33:
            value = lz

    # ---------------- colella -----------------------
    elif kind_map == 12:

        lx    = params_map[0]
        ly    = params_map[1]
        alpha = params_map[2]
        lz    = params_map[3]

        if   component == 11:
            value = lx * (1 + alpha * cos(2*pi*eta1) * sin(2*pi*eta2) * 2*pi)
        elif component == 12:
            value = lx * alpha * sin(2*pi*eta1) * cos(2*pi*eta2) * 2*pi
        elif component == 13:
            value = 0.
        elif component == 21:
            value = ly * alpha * cos(2*pi*eta1) * sin(2*pi*eta2) * 2*pi
        elif component == 22:
            value = ly * (1 + alpha * sin(2*pi*eta1) * cos(2*pi*eta2) * 2*pi)
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.    
        elif component == 33:
            value = lz

    # ------------------ orthogonal -------------------
    elif kind_map == 13:

        lx    = params_map[0]
        ly    = params_map[1]
        alpha = params_map[2]
        lz    = params_map[3]

        if   component == 11:
            value = lx * (1 + alpha * cos(2*pi*eta1) * 2*pi)
        elif component == 12:
            value = 0.
        elif component == 13:
            value = 0.
        elif component == 21:
            value = 0.
        elif component == 22:
            value = ly * (1 + alpha * cos(2*pi*eta2) * 2*pi)
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.    
        elif component == 33:
            value = lz

    # -------------- hollow torus ----------------------
    elif kind_map == 14:

        a1 = params_map[0]
        a2 = params_map[1]
        r0 = params_map[2]

        da = a2 - a1

        if   component == 11:
            value = da * cos(2*pi*eta2) * cos(2*pi*eta3)
        elif component == 12:
            value = -2*pi * (a1 + eta1 * da) * sin(2*pi*eta2) * cos(2*pi*eta3)
        elif component == 13:
            value = -2*pi * ((a1 + eta1 * da) * cos(2*pi*eta2) + r0) * sin(2*pi*eta3)
        elif component == 21:
            value = da * sin(2*pi*eta2)
        elif component == 22:
            value = (a1 + eta1 * da) * cos(2*pi*eta2) * 2*pi
        elif component == 23:
            value = 0.
        elif component == 31:
            value = da * cos(2*pi*eta2) * sin(2*pi*eta3)
        elif component == 32:
            value = -2*pi * (a1 + eta1 * da) * sin(2*pi*eta2) * sin(2*pi*eta3)
        elif component == 33:
            value = ((a1 + eta1 * da) * cos(2*pi*eta2) + r0) * cos(2*pi*eta3) * 2*pi

    # ----------------- ellipse -------------------------
    elif kind_map == 15:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        lz = params_map[5]

        if   component == 11:
            value = rx * cos(2*pi*eta2)
        elif component == 12:
            value = -2*pi * (eta1 * rx) * sin(2*pi*eta2)
        elif component == 13:
            value = 0.
        elif component == 21:
            value = ry * sin(2*pi*eta2)
        elif component == 22:
            value =  2*pi * (eta1 * ry) * cos(2*pi*eta2)
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.
        elif component == 33:
            value = lz

    # -------------- rotated ellipse ---------------------
    elif kind_map == 16:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        r1 = params_map[3]
        r2 = params_map[4]
        lz = params_map[5]
        th = params_map[6] # Domain: [0,1)

        if   component == 11:
            value = r1 * cos(2*pi*th) * cos(2*pi*eta2) - r2 * sin(2*pi*th) * sin(2*pi*eta2)
        elif component == 12:
            value = -2*pi * (eta1 * r1) * cos(2*pi*th) * sin(2*pi*eta2) - 2*pi * (eta1 * r2) * sin(2*pi*th) * cos(2*pi*eta2)
        elif component == 13:
            value = 0.
        elif component == 21:
            value = r1 * sin(2*pi*th) * cos(2*pi*eta2) + r2 * cos(2*pi*th) * sin(2*pi*eta2)
        elif component == 22:
            value = -2*pi * (eta1 * r1) * sin(2*pi*th) * sin(2*pi*eta2) + 2*pi * (eta1 * r2) * cos(2*pi*th) * cos(2*pi*eta2)
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.
        elif component == 33:
            value = lz

    # -------------- ellipse with power ---------------------
    elif kind_map == 17:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        lz = params_map[5]
        s  = params_map[6]

        if   component == 11:
            value = (eta1**(s-1)) * rx * cos(2*pi*eta2)
        elif component == 12:
            value = -2*pi * (eta1**s) * rx * sin(2*pi*eta2)
        elif component == 13:
            value = 0.
        elif component == 21:
            value = (eta1**(s-1)) * ry * sin(2*pi*eta2)
        elif component == 22:
            value =  2*pi * (eta1**s) * ry * cos(2*pi*eta2)
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.
        elif component == 33:
            value = lz

    # -------------- shafranov shift ---------------------
    elif kind_map == 18:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        lz = params_map[5]
        de = params_map[6] # Domain: [0,0.1]

        if   component == 11:
            value = rx * cos(2*pi*eta2) - 2 * eta1 * rx * de
        elif component == 12:
            value = -2*pi * (eta1 * rx) * sin(2*pi*eta2)
        elif component == 13:
            value = 0.
        elif component == 21:
            value = ry * sin(2*pi*eta2)
        elif component == 22:
            value =  2*pi * (eta1 * ry) * cos(2*pi*eta2)
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.
        elif component == 33:
            value = lz

    # -------------- shafranov sqrt ---------------------
    elif kind_map == 19:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        lz = params_map[5]
        de = params_map[6] # Domain: [0,0.1]

        if   component == 11:
            value = rx * cos(2*pi*eta2) - 0.5 / sqrt(eta1) * rx * de
        elif component == 12:
            value = -2*pi * (eta1 * rx) * sin(2*pi*eta2)
        elif component == 13:
            value = 0.
        elif component == 21:
            value = ry * sin(2*pi*eta2)
        elif component == 22:
            value =  2*pi * (eta1 * ry) * cos(2*pi*eta2)
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.
        elif component == 33:
            value = lz

    # -------------- shafranov shift, D-shaped ---------------------
    elif kind_map == 20:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        r0 = params_map[3]
        lz = params_map[4]
        dx = params_map[5] # Grad-Shafranov shift along x-axis.
        dy = params_map[6] # Grad-Shafranov shift along y-axis.
        dg = params_map[7] # Delta = sin(alpha): Triangularity, shift of high point.
        eg = params_map[8] # Epsilon: Inverse aspect ratio a/R0.
        kg = params_map[9] # Kappa: Ellipticity (elongation).

        if   component == 11:
            value = r0 * (- 2 * dx * eta1 - eg * eta1 * sin(2*pi*eta2) * arcsin(dg) * sin(eta1 * sin(2*pi*eta2) * arcsin(dg) + 2*pi*eta2) + eg * cos(eta1 * sin(2*pi*eta2) * arcsin(dg) + 2*pi*eta2))
        elif component == 12:
            value = - r0 * eg * eta1 * (2*pi*eta1 * cos(2*pi*eta2) * arcsin(dg) + 2*pi) * sin(eta1 * sin(2*pi*eta2) * arcsin(dg) + 2*pi*eta2)
        elif component == 13:
            value = 0.
        elif component == 21:
            value = r0 * (- 2 * dy * eta1 + eg * kg * sin(2*pi*eta2))
        elif component == 22:
            value = 2 * pi * r0 * eg * eta1 * kg * cos(2*pi*eta2)
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.
        elif component == 33:
            value = lz

    # -------------- shafranov D-shaped with eta3 dependence ---------------------
    elif kind_map == 21:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        r0 = params_map[3]
        lz = params_map[4]
        dx = params_map[5]  # Grad-Shafranov shift along x-axis.
        dy = params_map[6]  # Grad-Shafranov shift along y-axis.
        dg = params_map[7]  # Delta = sin(alpha): Triangularity, shift of high point.
        eg = params_map[8]  # Epsilon: Inverse aspect ratio a/R0.
        kg = params_map[9]  # Kappa: Ellipticity (elongation).
        xi = params_map[10] # Xi: Strength of dependence on eta3.

        if   component == 11:
            value = r0 * (1 + xi * cos(2*pi*eta3)) * (- 2 * dx * eta1 - eg * eta1 * sin(2*pi*eta2) * arcsin(dg) * sin(eta1 * sin(2*pi*eta2) * arcsin(dg) + 2*pi*eta2) + eg * cos(eta1 * sin(2*pi*eta2) * arcsin(dg) + 2*pi*eta2))
        elif component == 12:
            value = - r0 * eg * eta1 * (1 + xi * cos(2*pi*eta3)) * (2*pi*eta1 * cos(2*pi*eta2) * arcsin(dg) + 2*pi) * sin(eta1 * sin(2*pi*eta2) * arcsin(dg) + 2*pi*eta2)
        elif component == 13:
            value = - 2 * pi * r0 * xi * (dx * (1 - eta1**2) + eg * eta1 * cos(eta1 * sin(2*pi*eta2) * arcsin(dg) + 2*pi*eta2) + 1) * sin(2*pi*eta3)
        elif component == 21:
            value = r0 * (- 2 * dy * eta1 + eg * kg * sin(2*pi*eta2)) * (1 - xi * cos(2*pi*eta3))
        elif component == 22:
            value = 2 * pi * r0 * eg * eta1 * kg * (1 - xi * cos(2*pi*eta3)) * cos(2*pi*eta2)
        elif component == 23:
            value = 2 * pi * r0 * xi * (dy * (1 - eta1**2) + eg * eta1 * kg * sin(2*pi*eta2)) * sin(2*pi*eta3)
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.
        elif component == 33:
            value = lz

    return value


def df(eta1 : float, eta2 : float, eta3 : float, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', df_out : 'float[:,:]'):
    """
    Point-wise evaluation of all components of the Jacobian matrix df_ij = df_i/deta_j (i,j=1,2,3). 

    Parameters
    ----------
        eta1, eta2, eta3 : float              
            Logical coordinates in [0, 1].
            
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params_map : array[float]
            Parameters for the mapping in a 1d array.
        
        tn1, tn2, tn3 : array[float]          
            Knot vectors of univariate splines.
        
        pn : array[int]
            Degrees of univariate splines [pn1, pn2, pn3].
        
        ind_n1, ind_n2, ind_n3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (f_1, f_2, f_3) in case of a IGA mapping.

        df_out : array[float]
            Evaluated Jacobian matrix.
    """
    
    df_out[0, 0] = df_ij(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    df_out[0, 1] = df_ij(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    df_out[0, 2] = df_ij(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    
    df_out[1, 0] = df_ij(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    df_out[1, 1] = df_ij(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    df_out[1, 2] = df_ij(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    
    df_out[2, 0] = df_ij(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    df_out[2, 1] = df_ij(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    df_out[2, 2] = df_ij(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    
    
def det_df(eta1 : float, eta2 : float, eta3 : float, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float:
    """
    Point-wise evaluation of the Jacobian determinant det(df) = df/deta1.dot(df/deta2 x df/deta3). 
    
    Parameters
    ----------
        eta1, eta2, eta3 : float              
            Logical coordinates in [0, 1].
            
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params_map : array[float]
            Parameters for the mapping in a 1d array.
        
        tn1, tn2, tn3 : array[float]          
            Knot vectors of univariate splines.
        
        pn : array[int]
            Degrees of univariate splines [pn1, pn2, pn3].
        
        ind_n1, ind_n2, ind_n3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (f_1, f_2, f_3) in case of a IGA mapping.

    Returns:
    --------
        detdf : float
            Jacobian determinant det(df)(eta1, eta2, eta3).
    """
    
    df_mat = empty((3, 3), dtype=float)
    
    df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz, df_mat)
    
    detdf = det(df_mat)
            
    return detdf


def df_inv_ij(eta1 : float, eta2 : float, eta3 : float, component : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float:
    """
    Point-wise evaluation of ij-th component of the inverse Jacobian matrix df^(-1)_ij (i,j=1,2,3). 
    
    The 3 x 3 inverse is computed directly from df, using the cross product of the columns of df:

                            | [ (df/deta2) x (df/deta3) ]^T |
    (df)^(-1) = 1/det_df *  | [ (df/deta3) x (df/deta1) ]^T |
                            | [ (df/deta1) x (df/deta2) ]^T |
    
    Parameters
    ----------
        eta1, eta2, eta3 : float              
            Logical coordinates in [0, 1].
        
        component : int                 
            Component of inverse Jacobian matrix (11, 12, ..., 32, 33) to evaluate.
            
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params_map : array[float]
            Parameters for the mapping in a 1d array.
        
        tn1, tn2, tn3 : array[float]          
            Knot vectors of univariate splines.
        
        pn : array[int]
            Degrees of univariate splines [pn1, pn2, pn3].
        
        ind_n1, ind_n2, ind_n3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (f_1, f_2, f_3) in case of a IGA mapping.

    Returns
    -------
        value : float
            The ij-th component of the inverse Jacobian matrix.
    """
    
    value = 0.
    
    df_mat = empty((3, 3), dtype=float)
    
    df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz, df_mat)
    
    detdf = det(df_mat)

    if   component == 11:
        value = (df_mat[1, 1]*df_mat[2, 2] - df_mat[2, 1]*df_mat[1, 2])/detdf
    elif component == 12:
        value = (df_mat[2, 1]*df_mat[0, 2] - df_mat[0, 1]*df_mat[2, 2])/detdf
    elif component == 13:
        value = (df_mat[0, 1]*df_mat[1, 2] - df_mat[1, 1]*df_mat[0, 2])/detdf
    elif component == 21:
        value = (df_mat[1, 2]*df_mat[2, 0] - df_mat[2, 2]*df_mat[1, 0])/detdf
    elif component == 22:
        value = (df_mat[2, 2]*df_mat[0, 0] - df_mat[0, 2]*df_mat[2, 0])/detdf
    elif component == 23:
        value = (df_mat[0, 2]*df_mat[1, 0] - df_mat[1, 2]*df_mat[0, 0])/detdf
    elif component == 31:
        value = (df_mat[1, 0]*df_mat[2, 1] - df_mat[2, 0]*df_mat[1, 1])/detdf
    elif component == 32:
        value = (df_mat[2, 0]*df_mat[0, 1] - df_mat[0, 0]*df_mat[2, 1])/detdf
    elif component == 33:
        value = (df_mat[0, 0]*df_mat[1, 1] - df_mat[1, 0]*df_mat[0, 1])/detdf
            
    return value


def df_inv(eta1 : float, eta2 : float, eta3 : float, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', df_inv_out : 'float[:,:]'):
    """
    Point-wise evaluation of all components of the inverse Jacobian matrix df^(-1)_ij (i,j=1,2,3). 
    
    The 3 x 3 inverse is computed directly from df, using the cross product of the columns of df:

                            | [ (df/deta2) x (df/deta3) ]^T |
    (df)^(-1) = 1/det_df *  | [ (df/deta3) x (df/deta1) ]^T |
                            | [ (df/deta1) x (df/deta2) ]^T |
    
    Parameters
    ----------
        eta1, eta2, eta3 : float              
            Logical coordinates in [0, 1].
            
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params_map : array[float]
            Parameters for the mapping in a 1d array.
        
        tn1, tn2, tn3 : array[float]          
            Knot vectors of univariate splines.
        
        pn : array[int]
            Degrees of univariate splines [pn1, pn2, pn3].
        
        ind_n1, ind_n2, ind_n3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (f_1, f_2, f_3) in case of a IGA mapping.

        df_inv_out : array[float]
            Evaluated inverse Jacobian matrix.
    """
    
    df_inv_out[0, 0] = df_inv_ij(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    df_inv_out[0, 1] = df_inv_ij(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    df_inv_out[0, 2] = df_inv_ij(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    
    df_inv_out[1, 0] = df_inv_ij(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    df_inv_out[1, 1] = df_inv_ij(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    df_inv_out[1, 2] = df_inv_ij(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    
    df_inv_out[2, 0] = df_inv_ij(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    df_inv_out[2, 1] = df_inv_ij(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    df_inv_out[2, 2] = df_inv_ij(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    
    
def g_ij(eta1 : float, eta2 : float, eta3 : float, component : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float:
    """
    Point-wise evaluation of ij-th component of metric tensor g_ij = sum_k [ (df^T)_ik (df)_kj ] (i,j,k=1,2,3). 
    
    Parameters
    ----------
        eta1, eta2, eta3 : float              
            Logical coordinates in [0, 1].
        
        component : int                 
            Component of metric (11, 12, ..., 32, 33) to evaluate.
            
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params_map : array[float]
            Parameters for the mapping in a 1d array.
        
        tn1, tn2, tn3 : array[float]          
            Knot vectors of univariate splines.
        
        pn : array[int]
            Degrees of univariate splines [pn1, pn2, pn3].
        
        ind_n1, ind_n2, ind_n3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (f_1, f_2, f_3) in case of a IGA mapping.

    Returns
    --------
        value : float
            The ij-th component of the metric tensor.
    """
    
    value = 0.

    if   component == 11:
        df_11 = df_ij(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_21 = df_ij(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_31 = df_ij(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        value = df_11*df_11 + df_21*df_21 + df_31*df_31
        
    elif component == 22:                                              
        df_12 = df_ij(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_22 = df_ij(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_32 = df_ij(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        value = df_12*df_12 + df_22*df_22 + df_32*df_32
                 
    elif component == 33:                                              
        df_13 = df_ij(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_23 = df_ij(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_33 = df_ij(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        value = df_13*df_13 + df_23*df_23 + df_33*df_33
                 
    elif component == 12 or component == 21:
        df_11 = df_ij(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_21 = df_ij(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_31 = df_ij(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_12 = df_ij(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_22 = df_ij(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_32 = df_ij(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        value = df_11*df_12 + df_21*df_22 + df_31*df_32
                 
    elif component == 13 or component == 31:
        df_11 = df_ij(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_21 = df_ij(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_31 = df_ij(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_13 = df_ij(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_23 = df_ij(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_33 = df_ij(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        value = df_11*df_13 + df_21*df_23 + df_31*df_33
                 
    elif component == 23 or component == 32:  
        df_12 = df_ij(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_22 = df_ij(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_32 = df_ij(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_13 = df_ij(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_23 = df_ij(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_33 = df_ij(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        value = df_12*df_13 + df_22*df_23 + df_32*df_33
               
    return value


def g(eta1 : float, eta2 : float, eta3 : float, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', g_out : 'float[:,:]'):
    """
    Point-wise evaluation of all components of the metric tensor g_ij = sum_k [ (df^T)_ik (df)_kj ] (i,j,k=1,2,3). 
    
    Parameters
    ----------
        eta1, eta2, eta3 : float              
            Logical coordinates in [0, 1].
            
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params_map : array[float]
            Parameters for the mapping in a 1d array.
        
        tn1, tn2, tn3 : array[float]          
            Knot vectors of univariate splines.
        
        pn : array[int]
            Degrees of univariate splines [pn1, pn2, pn3].
        
        ind_n1, ind_n2, ind_n3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (f_1, f_2, f_3) in case of a IGA mapping.

        g_out : float
            Evaluated components of the metric tensor.
    """
    
    g_out[0, 0] = g_ij(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    g_out[0, 1] = g_ij(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    g_out[0, 2] = g_ij(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    
    g_out[1, 0] = g_ij(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    g_out[1, 1] = g_ij(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    g_out[1, 2] = g_ij(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    
    g_out[2, 0] = g_ij(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    g_out[2, 1] = g_ij(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    g_out[2, 2] = g_ij(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    
    
def g_inv_ij(eta1 : float, eta2 : float, eta3 : float, component : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float:
    """
    Point-wise evaluation of ij-th component of the inverse metric tensor g^(-1)_ij = sum_k [ (df^-1)_ik (df^-T)_kj ] (i,j,k=1,2,3). 
    
    Parameters
    ----------
        eta1, eta2, eta3 : float              
            Logical coordinates in [0, 1].
        
        component : int                 
            Component of inverse metric tensor (11, 12, ..., 32, 33) to evaluate.
            
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params_map : array[float]
            Parameters for the mapping in a 1d array.
        
        tn1, tn2, tn3 : array[float]          
            Knot vectors of univariate splines.
        
        pn : array[int]
            Degrees of univariate splines [pn1, pn2, pn3].
        
        ind_n1, ind_n2, ind_n3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (f_1, f_2, f_3) in case of a IGA mapping.

    Returns
    -------
        value : float
            The ij-th component of the inverse metric tensor.
    """
    
    value = 0.
    
    df_mat = empty((3, 3), dtype=float)
    
    df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz, df_mat)
    
    detdf = det(df_mat)
    
    df_inv_11 = (df_mat[1, 1]*df_mat[2, 2] - df_mat[2, 1]*df_mat[1, 2])/detdf
    df_inv_12 = (df_mat[2, 1]*df_mat[0, 2] - df_mat[0, 1]*df_mat[2, 2])/detdf
    df_inv_13 = (df_mat[0, 1]*df_mat[1, 2] - df_mat[1, 1]*df_mat[0, 2])/detdf
    
    df_inv_21 = (df_mat[1, 2]*df_mat[2, 0] - df_mat[2, 2]*df_mat[1, 0])/detdf
    df_inv_22 = (df_mat[2, 2]*df_mat[0, 0] - df_mat[0, 2]*df_mat[2, 0])/detdf
    df_inv_23 = (df_mat[0, 2]*df_mat[1, 0] - df_mat[1, 2]*df_mat[0, 0])/detdf
    
    df_inv_31 = (df_mat[1, 0]*df_mat[2, 1] - df_mat[2, 0]*df_mat[1, 1])/detdf
    df_inv_32 = (df_mat[2, 0]*df_mat[0, 1] - df_mat[0, 0]*df_mat[2, 1])/detdf
    df_inv_33 = (df_mat[0, 0]*df_mat[1, 1] - df_mat[1, 0]*df_mat[0, 1])/detdf
    
    if   component == 11:
        value = df_inv_11*df_inv_11 + df_inv_12*df_inv_12 + df_inv_13*df_inv_13
                  
    elif component == 22:                                              
        value = df_inv_21*df_inv_21 + df_inv_22*df_inv_22 + df_inv_23*df_inv_23
                  
    elif component == 33:                                              
        value = df_inv_31*df_inv_31 + df_inv_32*df_inv_32 + df_inv_33*df_inv_33
                  
    elif component == 12 or component == 21:
        value = df_inv_11*df_inv_21 + df_inv_12*df_inv_22 + df_inv_13*df_inv_23
                  
    elif component == 13 or component == 31:
        value = df_inv_11*df_inv_31 + df_inv_12*df_inv_32 + df_inv_13*df_inv_33
                  
    elif component == 23 or component == 32:  
        value = df_inv_21*df_inv_31 + df_inv_22*df_inv_32 + df_inv_23*df_inv_33
    
    return value


def g_inv(eta1 : float, eta2 : float, eta3 : float, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', g_inv_out : 'float[:,:]'):
    """
    Point-wise evaluation of all components of the inverse metric tensor g^(-1)_ij = sum_k [ (df^-1)_ik (df^-T)_kj ] (i,j,k=1,2,3). 
    
    Parameters
    ----------
        eta1, eta2, eta3 : float              
            Logical coordinates in [0, 1].
        
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params_map : array[float]
            Parameters for the mapping in a 1d array.
        
        tn1, tn2, tn3 : array[float]          
            Knot vectors of univariate splines.
        
        pn : array[int]
            Degrees of univariate splines [pn1, pn2, pn3].
        
        ind_n1, ind_n2, ind_n3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (f_1, f_2, f_3) in case of a IGA mapping.
            
        g_inv_out : array[float]
            Evaluated inverse metric tensor.
    """
    
    g_inv_out[0, 0] = g_inv_ij(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    g_inv_out[0, 1] = g_inv_ij(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    g_inv_out[0, 2] = g_inv_ij(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    
    g_inv_out[1, 0] = g_inv_ij(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    g_inv_out[1, 1] = g_inv_ij(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    g_inv_out[1, 2] = g_inv_ij(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    
    g_inv_out[2, 0] = g_inv_ij(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    g_inv_out[2, 1] = g_inv_ij(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    g_inv_out[2, 2] = g_inv_ij(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    
    
def mappings_all(eta1 : float, eta2 : float, eta3 : float, kind_fun : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float:
    """
    Point-wise evaluation of
        - f_i       : mapping x_i = f_i(eta1, eta2, eta3),
        - df_ij     : Jacobian matrix df_i/deta_j,
        - det_df    : Jacobian determinant det(df),
        - df_inv_ij : inverse Jacobian matrix (df_i/deta_j)^(-1),
        - g_ij      : metric tensor df^T * df,
        - g_inv_ij  : inverse metric tensor df^(-1) * df^(-T).
    
    Parameters
    ----------
        eta1, eta2, eta3 : float              
            Logical coordinates in [0, 1].
            
        kind_fun : int
            Which metric coefficient to evaluate
        
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params_map : array[float]
            Parameters for the mapping in a 1d array.
        
        tn1, tn2, tn3 : array[float]          
            Knot vectors of univariate splines.
        
        pn : array[int]
            Degrees of univariate splines [pn1, pn2, pn3].
        
        ind_n1, ind_n2, ind_n3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (f_1, f_2, f_3) in case of a IGA mapping.

    Returns:
    --------
        value : float
            Point value of metric coefficient at (eta1, eta2, eta3).
    """
    
    value = 0.
    
    # mapping f
    if   kind_fun == 1:
        value = f_i(eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 2:
        value = f_i(eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 3:
        value = f_i(eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    
    # Jacobian matrix df
    elif kind_fun == 11:
        value = df_ij(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 12:
        value = df_ij(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 13:
        value = df_ij(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 14:
        value = df_ij(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 15:
        value = df_ij(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 16:
        value = df_ij(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 17:
        value = df_ij(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 18:
        value = df_ij(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 19:
        value = df_ij(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        
    # Jacobian determinant det_df
    elif kind_fun == 4:
        value = det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        
    # inverse Jacobian matrix df_inv
    elif kind_fun == 21:
        value = df_inv_ij(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 22:
        value = df_inv_ij(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 23:
        value = df_inv_ij(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 24:
        value = df_inv_ij(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 25:
        value = df_inv_ij(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 26:
        value = df_inv_ij(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 27:
        value = df_inv_ij(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 28:
        value = df_inv_ij(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 29:
        value = df_inv_ij(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        
    # metric tensor g
    elif kind_fun == 31:
        value = g_ij(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 32:
        value = g_ij(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 33:
        value = g_ij(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 34:
        value = g_ij(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 35:
        value = g_ij(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 36:
        value = g_ij(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 37:
        value = g_ij(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)     
    elif kind_fun == 38:
        value = g_ij(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 39:
        value = g_ij(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    
    # metric tensor g_inv
    elif kind_fun == 41:
        value = g_inv_ij(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 42:
        value = g_inv_ij(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 43:
        value = g_inv_ij(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)  
    elif kind_fun == 44:
        value = g_inv_ij(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 45:
        value = g_inv_ij(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 46:
        value = g_inv_ij(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 47:
        value = g_inv_ij(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 48:
        value = g_inv_ij(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 49:
        value = g_inv_ij(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    
    return value

   
def kernel_evaluate(eta1 : 'float[:,:,:]', eta2 : 'float[:,:,:]', eta3 : 'float[:,:,:]', kind_fun : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', mat_f : 'float[:,:,:]'):
    """
    Matrix-wise evaluation of
        - f_i       : mapping x_i = f_i(eta1, eta2, eta3),
        - df_ij     : Jacobian matrix df_i/deta_j,
        - det_df    : Jacobian determinant det(df),
        - df_inv_ij : inverse Jacobian matrix (df_i/deta_j)^(-1),
        - g_ij      : metric tensor df^T * df,
        - g_inv_ij  : inverse metric tensor df^(-1) * df^(-T).

    Parameters
    ----------
        eta1, eta2, eta3 : array[float]              
            Logical coordinatess in 3d arrays with shape(eta1) == shape(eta2) == shape(eta3).
            
        kind_fun : int
            Which metric coefficient to evaluate
        
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params_map : array[float]
            Parameters for the mapping in a 1d array.
        
        tn1, tn2, tn3 : array[float]          
            Knot vectors of univariate splines.
        
        pn : array[int]
            Degrees of univariate splines [pn1, pn2, pn3].
        
        ind_n1, ind_n2, ind_n3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (f_1, f_2, f_3) in case of a IGA mapping.
            
        mat_f : array[float]
            matrix-valued mapping/metric coefficient evaluated at (eta1, eta2, eta3).
    """

    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                mat_f[i1, i2, i3] = mappings_all(eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

     
def kernel_evaluate_sparse(eta1 : 'float[:,:,:]', eta2 : 'float[:,:,:]', eta3 : 'float[:,:,:]', kind_fun : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', mat_f : 'float[:,:,:]'):
    """
    Same as kernel_evaluate, but for sparse meshgrids.
    
    Parameters
    ----------
        eta1, eta2, eta3 : array[float]              
            Logical coordinatess in 3d arrays with shape(eta1) = (:,1,1), shape(eta2) = (1,:,1), shape(eta3) = (1,1,:).
            
        kind_fun : int
            Which metric coefficient to evaluate
        
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params_map : array[float]
            Parameters for the mapping in a 1d array.
        
        tn1, tn2, tn3 : array[float]          
            Knot vectors of univariate splines.
        
        pn : array[int]
            Degrees of univariate splines [pn1, pn2, pn3].
        
        ind_n1, ind_n2, ind_n3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (f_1, f_2, f_3) in case of a IGA mapping.
            
        mat_f : array[float]
            matrix-valued mapping/metric coefficient evaluated at (eta1, eta2, eta3).
    """

    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                mat_f[i1, i2, i3] = mappings_all(eta1[i1, 0, 0], eta2[0, i2, 0], eta3[0, 0, i3], kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

                     
def kernel_evaluate_flat(eta1 : 'float[:]', eta2 : 'float[:]', eta3 : 'float[:]', kind_fun : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', mat_f : 'float[:]'):
    """
    Same as kernel_evaluate, but for flat evaluation.
    
    Parameters
    ----------
        eta1, eta2, eta3 : array[float]              
            Logical coordinatess in 1d arrays with len(eta1) == len(eta2) == len(eta3).
            
        kind_fun : int
            Which metric coefficient to evaluate
        
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params_map : array[float]
            Parameters for the mapping in a 1d array.
        
        tn1, tn2, tn3 : array[float]          
            Knot vectors of univariate splines.
        
        pn : array[int]
            Degrees of univariate splines [pn1, pn2, pn3].
        
        ind_n1, ind_n2, ind_n3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (f_1, f_2, f_3) in case of a IGA mapping.
            
        mat_f : array[float]
            matrix-valued mapping/metric coefficient evaluated at (eta1, eta2, eta3).

    Returns
    -------
        mat_f:  np.array
            1d array [f(x1, y1, z1) f(x2, y2, z2) etc.]
    """

    for i in range(len(eta1)):
        mat_f[i] = mappings_all(eta1[i], eta2[i], eta3[i], kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)


def f_df_pic(kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', span_n1 : int, span_n2 : int, span_n3 : int, ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', f_out : 'float[:]', df_out : 'float[:,:]', f_or_df : 'int'):
    """
    Fast evaluation of mapping F and/or Jacobian matrix DF because of avoiding memory allocation and multiple computations. Especially well suited for PIC routines.
    
    Parameters
    ----------
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params_map : array[float]
            Parameters for the mapping in a 1d array.
            
        tn1, tn2, tn3 : array[float]          
            Knot vectors of univariate splines.
        
        pn : array[int]
            Degrees of univariate splines [pn1, pn2, pn3].
            
        span1, span2, span3 : int
            Knot span indices at considered point.
            
        ind_n1, ind_n2, ind_n3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
            
        cx, cy, cz : array[float]     
            Control points of (f_1, f_2, f_3) in case of a IGA mapping.
            
        eta1, eta2, eta3 : float
            Logical coordinates (evaluation point).
            
        f_out : array[float]
            Empty buffer for the three evaluated Cartesian components.
            
        df_out : array[float]
            Empty buffer for the nine evaluated Jacobian matrix components.
            
        f_or_df : int
            Wether to evaluate mapping only (0), Jacobian matrix only (1) or both (2).
    """
    
    # ------------------- 3d spline mapping ------------------------------
    if kind_map == 0:

        bn1 = zeros(pn[0] + 1, dtype=float)
        bn2 = zeros(pn[1] + 1, dtype=float)
        bn3 = zeros(pn[2] + 1, dtype=float)

        der1 = zeros(pn[0] + 1, dtype=float)
        der2 = zeros(pn[1] + 1, dtype=float)
        der3 = zeros(pn[2] + 1, dtype=float)
        
        # evaluate non-vanishing basis functions and its derivatives
        bsp.b_der_splines_slim(tn1, pn[0], eta1, span_n1, bn1, der1)
        bsp.b_der_splines_slim(tn2, pn[1], eta2, span_n2, bn2, der2)
        bsp.b_der_splines_slim(tn3, pn[2], eta3, span_n3, bn3, der3)
        
        # evaluate mapping
        if f_or_df == 0 or f_or_df == 2:
            
            f_out[0] = eva_3d.evaluation_kernel_3d(pn[0], pn[1], pn[2], bn1, bn2, bn3, ind_n1[span_n1 - pn[0], :], ind_n2[span_n2 - pn[1], :], ind_n3[span_n3 - pn[2], :], cx)
            f_out[1] = eva_3d.evaluation_kernel_3d(pn[0], pn[1], pn[2], bn1, bn2, bn3, ind_n1[span_n1 - pn[0], :], ind_n2[span_n2 - pn[1], :], ind_n3[span_n3 - pn[2], :], cy)
            f_out[2] = eva_3d.evaluation_kernel_3d(pn[0], pn[1], pn[2], bn1, bn2, bn3, ind_n1[span_n1 - pn[0], :], ind_n2[span_n2 - pn[1], :], ind_n3[span_n3 - pn[2], :], cz)
        
        # evaluate Jacobian matrix
        if f_or_df == 1 or f_or_df == 2:
            
            # sum-up non-vanishing contributions (line 1: df_11, df_12 and df_13)
            df_out[0, 0] = eva_3d.evaluation_kernel_3d(pn[0], pn[1], pn[2], der1, bn2, bn3, ind_n1[span_n1 - pn[0], :], ind_n2[span_n2 - pn[1], :], ind_n3[span_n3 - pn[2], :], cx)
            df_out[0, 1] = eva_3d.evaluation_kernel_3d(pn[0], pn[1], pn[2], bn1, der2, bn3, ind_n1[span_n1 - pn[0], :], ind_n2[span_n2 - pn[1], :], ind_n3[span_n3 - pn[2], :], cx)
            df_out[0, 2] = eva_3d.evaluation_kernel_3d(pn[0], pn[1], pn[2], bn1, bn2, der3, ind_n1[span_n1 - pn[0], :], ind_n2[span_n2 - pn[1], :], ind_n3[span_n3 - pn[2], :], cx)

            # sum-up non-vanishing contributions (line 2: df_21, df_22 and df_23)
            df_out[1, 0] = eva_3d.evaluation_kernel_3d(pn[0], pn[1], pn[2], der1, bn2, bn3, ind_n1[span_n1 - pn[0], :], ind_n2[span_n2 - pn[1], :], ind_n3[span_n3 - pn[2], :], cy)
            df_out[1, 1] = eva_3d.evaluation_kernel_3d(pn[0], pn[1], pn[2], bn1, der2, bn3, ind_n1[span_n1 - pn[0], :], ind_n2[span_n2 - pn[1], :], ind_n3[span_n3 - pn[2], :], cy)
            df_out[1, 2] = eva_3d.evaluation_kernel_3d(pn[0], pn[1], pn[2], bn1, bn2, der3, ind_n1[span_n1 - pn[0], :], ind_n2[span_n2 - pn[1], :], ind_n3[span_n3 - pn[2], :], cy)

            # sum-up non-vanishing contributions (line 3: df_31, df_32 and df_33)
            df_out[2, 0] = eva_3d.evaluation_kernel_3d(pn[0], pn[1], pn[2], der1, bn2, bn3, ind_n1[span_n1 - pn[0], :], ind_n2[span_n2 - pn[1], :], ind_n3[span_n3 - pn[2], :], cz)
            df_out[2, 1] = eva_3d.evaluation_kernel_3d(pn[0], pn[1], pn[2], bn1, der2, bn3, ind_n1[span_n1 - pn[0], :], ind_n2[span_n2 - pn[1], :], ind_n3[span_n3 - pn[2], :], cz)
            df_out[2, 2] = eva_3d.evaluation_kernel_3d(pn[0], pn[1], pn[2], bn1, bn2, der3, ind_n1[span_n1 - pn[0], :], ind_n2[span_n2 - pn[1], :], ind_n3[span_n3 - pn[2], :], cz)  
            
           
    # --------------------- 2d spline (straight) --------------------------
    elif kind_map == 1:
        
        lz = 2*pi*cx[0, 0, 0]
        
        bn1 = zeros(pn[0] + 1, dtype=float)
        bn2 = zeros(pn[1] + 1, dtype=float)

        der1 = zeros(pn[0] + 1, dtype=float)
        der2 = zeros(pn[1] + 1, dtype=float)

        # evaluate non-vanishing basis functions and its derivatives
        bsp.b_der_splines_slim(tn1, pn[0], eta1, span_n1, bn1, der1)
        bsp.b_der_splines_slim(tn2, pn[1], eta2, span_n2, bn2, der2)
        
        # evaluate mapping
        if f_or_df == 0 or f_or_df == 2:
            
            f_out[0] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], bn1, bn2, ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cx[:, :, 0])
            f_out[1] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], bn1, bn2, ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cy[:, :, 0])
            f_out[2] = lz * eta3
        
        # evaluate Jacobian matrix
        if f_or_df == 1 or f_or_df == 2:

            # sum-up non-vanishing contributions (line 1: df_11, df_12 and df_13)
            df_out[0, 0] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], der1, bn2, ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cx[:, :, 0])
            df_out[0, 1] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], bn1, der2, ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cx[:, :, 0])
            df_out[0, 2] = 0.

            # sum-up non-vanishing contributions (line 2: df_21, df_22 and df_23)
            df_out[1, 0] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], der1, bn2, ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cy[:, :, 0])
            df_out[1, 1] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], bn1, der2, ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cy[:, :, 0])
            df_out[1, 2] = 0.

            # sum-up non-vanishing contributions (line 3: df_31, df_32 and df_33)
            df_out[2, 0] = 0.
            df_out[2, 1] = 0.
            df_out[2, 2] = lz
        
        
    # --------------------- 2d spline (toroidal) ---------------------------
    elif kind_map == 2:

        bn1 = zeros(pn[0] + 1, dtype=float)
        bn2 = zeros(pn[1] + 1, dtype=float)

        der1 = zeros(pn[0] + 1, dtype=float)
        der2 = zeros(pn[1] + 1, dtype=float)
        
        # evaluate non-vanishing basis functions and its derivatives
        bsp.b_der_splines_slim(tn1, pn[0], eta1, span_n1, bn1, der1)
        bsp.b_der_splines_slim(tn2, pn[1], eta2, span_n2, bn2, der2)
        
        # evaluate mapping
        if f_or_df == 0 or f_or_df == 2:
            
            f_out[0] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], bn1, bn2, ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cx[:, :, 0]) * cos(2*pi*eta3)
            f_out[1] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], bn1, bn2, ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cy[:, :, 0])
            f_out[2] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], bn1, bn2, ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cx[:, :, 0]) * sin(2*pi*eta3)
        
        # evaluate Jacobian matrix
        if f_or_df == 1 or f_or_df == 2:

            # sum-up non-vanishing contributions (line 1: df_11, df_12 and df_13)
            df_out[0, 0] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], der1, bn2, ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cx[:, :, 0]) * cos(2*pi*eta3)
            df_out[0, 1] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], bn1, der2, ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cx[:, :, 0]) * cos(2*pi*eta3)
            df_out[0, 2] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], bn1, bn2, ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cx[:, :, 0]) * sin(2*pi*eta3) * (-2*pi)

            # sum-up non-vanishing contributions (line 2: df_21, df_22 and df_23)
            df_out[1, 0] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], der1, bn2, ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cy[:, :, 0])
            df_out[1, 1] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], bn1, der2, ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cy[:, :, 0])
            df_out[1, 2] = 0.

            # sum-up non-vanishing contributions (line 3: df_31, df_32 and df_33)
            df_out[2, 0] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], der1, bn2, ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cx[:, :, 0]) * sin(2*pi*eta3)
            df_out[2, 1] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], bn1, der2, ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cx[:, :, 0]) * sin(2*pi*eta3)
            df_out[2, 2] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], bn1, bn2, ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cx[:, :, 0]) * cos(2*pi*eta3) * 2*pi
        
        
    # -------------------------- analytical -------------------------------
    else:
        
        # evaluate mapping
        if f_or_df == 0 or f_or_df == 2:
            
            f(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz, f_out)
        
        # evaluate Jacobian matrix
        if f_or_df == 1 or f_or_df == 2:
        
            df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz, df_out)


def f_df_pic_legacy(kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', span_n1 : int, span_n2 : int, span_n3 : int, ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', l1 : 'float[:]', l2 : 'float[:]', l3 : 'float[:]', r1 : 'float[:]', r2 : 'float[:]', r3 : 'float[:]', b1 : 'float[:,:]', b2 : 'float[:,:]', b3 : 'float[:,:]', d1 : 'float[:]', d2 : 'float[:]', d3 : 'float[:]', der1 : 'float[:]', der2 : 'float[:]', der3 : 'float[:]', eta1 : float, eta2 : float, eta3 : float, f_out : 'float[:]', df_out : 'float[:,:]', f_or_df : int):
    """
    Fast evaluation of mapping and/or Jacobian matrix because of avoiding memory allocation and multiple computations. Especially well suited for PIC routines.
    
    Parameters
    ----------
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params_map : array[float]
            Parameters for the mapping in a 1d array.
            
        tn1, tn2, tn3 : array[float]          
            Knot vectors of univariate splines.
        
        pn : array[int]
            Degrees of univariate splines [pn1, pn2, pn3].
            
        span1, span2, span3 : int
            Knot span indices at considered point.
            
        ind_n1, ind_n2, ind_n3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
            
        cx, cy, cz : array[float]     
            Control points of (f_1, f_2, f_3) in case of a IGA mapping.
            
        l1, l2, l3 : array[float]
            Empty buffer for spline evaluation, 1d array of length pn.
            
        r1, r2, r3 : array[float]
            Empty buffer for spline evaluation, 1d array of length pn.
            
        b1, b2, b3 : array[float]
            Empty buffer for pn + 1 non-vanishing splines at considered point.
            
        d1, d2, d3 : array[float]
            Empty buffer for pn + 1 scaling values for M-splines.
            
        der1, der2, der3 : array[float]
            Empty buffer for derivatives of pn + 1 non-vanishing splines at considered point.
            
        eta1, eta2, eta3 : float
            Logical coordinates (evaluation point).
            
        f_out : array[float]
            Empty buffer for the three evaluated Cartesian components.
            
        df_out : array[float]
            Empty buffer for the nine evaluated Jacobian matrix components.
            
        f_or_df : int
            Wether to evaluate mapping only (0), Jacobian matrix only (1) or both (2).
    """
    
    # ------------------- 3d spline mapping ------------------------------
    if kind_map == 0:
        
        # evaluate non-vanishing basis functions and its derivatives
        bsp.basis_funs_and_der(tn1, pn[0], eta1, span_n1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(tn2, pn[1], eta2, span_n2, l2, r2, b2, d2, der2)
        bsp.basis_funs_and_der(tn3, pn[2], eta3, span_n3, l3, r3, b3, d3, der3)
        
        # evaluate mapping
        if f_or_df == 0 or f_or_df == 2:
            
            f_out[0] = eva_3d.evaluation_kernel_3d(pn[0], pn[1], pn[2], b1[pn[0]], b2[pn[1]], b3[pn[2]], ind_n1[span_n1 - pn[0], :], ind_n2[span_n2 - pn[1], :], ind_n3[span_n3 - pn[2], :], cx)
            f_out[1] = eva_3d.evaluation_kernel_3d(pn[0], pn[1], pn[2], b1[pn[0]], b2[pn[1]], b3[pn[2]], ind_n1[span_n1 - pn[0], :], ind_n2[span_n2 - pn[1], :], ind_n3[span_n3 - pn[2], :], cy)
            f_out[2] = eva_3d.evaluation_kernel_3d(pn[0], pn[1], pn[2], b1[pn[0]], b2[pn[1]], b3[pn[2]], ind_n1[span_n1 - pn[0], :], ind_n2[span_n2 - pn[1], :], ind_n3[span_n3 - pn[2], :], cz)
        
        # evaluate Jacobian matrix
        if f_or_df == 1 or f_or_df == 2:
            
            # sum-up non-vanishing contributions (line 1: df_11, df_12 and df_13)
            df_out[0, 0] = eva_3d.evaluation_kernel_3d(pn[0], pn[1], pn[2], der1, b2[pn[1]], b3[pn[2]], ind_n1[span_n1 - pn[0], :], ind_n2[span_n2 - pn[1], :], ind_n3[span_n3 - pn[2], :], cx)
            df_out[0, 1] = eva_3d.evaluation_kernel_3d(pn[0], pn[1], pn[2], b1[pn[0]], der2, b3[pn[2]], ind_n1[span_n1 - pn[0], :], ind_n2[span_n2 - pn[1], :], ind_n3[span_n3 - pn[2], :], cx)
            df_out[0, 2] = eva_3d.evaluation_kernel_3d(pn[0], pn[1], pn[2], b1[pn[0]], b2[pn[1]], der3, ind_n1[span_n1 - pn[0], :], ind_n2[span_n2 - pn[1], :], ind_n3[span_n3 - pn[2], :], cx)

            # sum-up non-vanishing contributions (line 2: df_21, df_22 and df_23)
            df_out[1, 0] = eva_3d.evaluation_kernel_3d(pn[0], pn[1], pn[2], der1, b2[pn[1]], b3[pn[2]], ind_n1[span_n1 - pn[0], :], ind_n2[span_n2 - pn[1], :], ind_n3[span_n3 - pn[2], :], cy)
            df_out[1, 1] = eva_3d.evaluation_kernel_3d(pn[0], pn[1], pn[2], b1[pn[0]], der2, b3[pn[2]], ind_n1[span_n1 - pn[0], :], ind_n2[span_n2 - pn[1], :], ind_n3[span_n3 - pn[2], :], cy)
            df_out[1, 2] = eva_3d.evaluation_kernel_3d(pn[0], pn[1], pn[2], b1[pn[0]], b2[pn[1]], der3, ind_n1[span_n1 - pn[0], :], ind_n2[span_n2 - pn[1], :], ind_n3[span_n3 - pn[2], :], cy)

            # sum-up non-vanishing contributions (line 3: df_31, df_32 and df_33)
            df_out[2, 0] = eva_3d.evaluation_kernel_3d(pn[0], pn[1], pn[2], der1, b2[pn[1]], b3[pn[2]], ind_n1[span_n1 - pn[0], :], ind_n2[span_n2 - pn[1], :], ind_n3[span_n3 - pn[2], :], cz)
            df_out[2, 1] = eva_3d.evaluation_kernel_3d(pn[0], pn[1], pn[2], b1[pn[0]], der2, b3[pn[2]], ind_n1[span_n1 - pn[0], :], ind_n2[span_n2 - pn[1], :], ind_n3[span_n3 - pn[2], :], cz)
            df_out[2, 2] = eva_3d.evaluation_kernel_3d(pn[0], pn[1], pn[2], b1[pn[0]], b2[pn[1]], der3, ind_n1[span_n1 - pn[0], :], ind_n2[span_n2 - pn[1], :], ind_n3[span_n3 - pn[2], :], cz)  
            
           
    # --------------------- 2d spline (straight) --------------------------
    elif kind_map == 1:
        
        lz = 2*pi*cx[0, 0, 0]
        
        # evaluate non-vanishing basis functions and its derivatives
        bsp.basis_funs_and_der(tn1, pn[0], eta1, span_n1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(tn2, pn[1], eta2, span_n2, l2, r2, b2, d2, der2)
        
        # evaluate mapping
        if f_or_df == 0 or f_or_df == 2:
            
            f_out[0] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], b1[pn[0]], b2[pn[1]], ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cx[:, :, 0])
            f_out[1] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], b1[pn[0]], b2[pn[1]], ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cy[:, :, 0])
            f_out[2] = lz * eta3
        
        # evaluate Jacobian matrix
        if f_or_df == 1 or f_or_df == 2:

            # sum-up non-vanishing contributions (line 1: df_11, df_12 and df_13)
            df_out[0, 0] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], der1, b2[pn[1]], ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cx[:, :, 0])
            df_out[0, 1] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], b1[pn[0]], der2, ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cx[:, :, 0])
            df_out[0, 2] = 0.

            # sum-up non-vanishing contributions (line 2: df_21, df_22 and df_23)
            df_out[1, 0] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], der1, b2[pn[1]], ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cy[:, :, 0])
            df_out[1, 1] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], b1[pn[0]], der2, ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cy[:, :, 0])
            df_out[1, 2] = 0.

            # sum-up non-vanishing contributions (line 3: df_31, df_32 and df_33)
            df_out[2, 0] = 0.
            df_out[2, 1] = 0.
            df_out[2, 2] = lz
        
        
    # --------------------- 2d spline (toroidal) ---------------------------
    elif kind_map == 2:
        
        # evaluate non-vanishing basis functions and its derivatives
        bsp.basis_funs_and_der(tn1, pn[0], eta1, span_n1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(tn2, pn[1], eta2, span_n2, l2, r2, b2, d2, der2)
        
        # evaluate mapping
        if f_or_df == 0 or f_or_df == 2:
            
            f_out[0] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], b1[pn[0]], b2[pn[1]], ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cx[:, :, 0]) * cos(2*pi*eta3)
            f_out[1] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], b1[pn[0]], b2[pn[1]], ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cy[:, :, 0])
            f_out[2] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], b1[pn[0]], b2[pn[1]], ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cx[:, :, 0]) * sin(2*pi*eta3)
        
        # evaluate Jacobian matrix
        if f_or_df == 1 or f_or_df == 2:

            # sum-up non-vanishing contributions (line 1: df_11, df_12 and df_13)
            df_out[0, 0] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], der1, b2[pn[1]], ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cx[:, :, 0]) * cos(2*pi*eta3)
            df_out[0, 1] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], b1[pn[0]], der2, ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cx[:, :, 0]) * cos(2*pi*eta3)
            df_out[0, 2] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], b1[pn[0]], b2[pn[1]], ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cx[:, :, 0]) * sin(2*pi*eta3) * (-2*pi)

            # sum-up non-vanishing contributions (line 2: df_21, df_22 and df_23)
            df_out[1, 0] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], der1, b2[pn[1]], ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cy[:, :, 0])
            df_out[1, 1] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], b1[pn[0]], der2, ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cy[:, :, 0])
            df_out[1, 2] = 0.

            # sum-up non-vanishing contributions (line 3: df_31, df_32 and df_33)
            df_out[2, 0] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], der1, b2[pn[1]], ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cx[:, :, 0]) * sin(2*pi*eta3)
            df_out[2, 1] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], b1[pn[0]], der2, ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cx[:, :, 0]) * sin(2*pi*eta3)
            df_out[2, 2] = eva_2d.evaluation_kernel_2d(pn[0], pn[1], b1[pn[0]], b2[pn[1]], ind_n1[span_n1 - pn[0]], ind_n2[span_n2 - pn[1]], cx[:, :, 0]) * cos(2*pi*eta3) * 2*pi
        
        
    # -------------------------- analytical -------------------------------
    else:
        
        # evaluate mapping
        if f_or_df == 0 or f_or_df == 2:
            
            f(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz, f_out)
        
        # evaluate Jacobian matrix
        if f_or_df == 1 or f_or_df == 2:
        
            df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz, df_out)


def loop_legacy(kind_map: 'int', params_map: 'float[:]',
                t1: 'float[:]', t2: 'float[:]', t3: 'float[:]', p: 'int[:]', 
                ind1: 'int[:,:]', ind2: 'int[:,:]', ind3: 'int[:,:]',
                cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                eta1s: 'float[:]', eta2s: 'float[:]', eta3s: 'float[:]',
                f_out: 'float[:,:]', df_out: 'float[:,:,:]'):

    l1 = zeros(p[0], dtype=float)
    l2 = zeros(p[1], dtype=float)
    l3 = zeros(p[2], dtype=float)

    r1 = zeros(p[0], dtype=float)
    r2 = zeros(p[1], dtype=float)
    r3 = zeros(p[2], dtype=float)

    b1 = zeros((p[0] + 1, p[0] + 1), dtype=float)
    b2 = zeros((p[1] + 1, p[1] + 1), dtype=float)
    b3 = zeros((p[2] + 1, p[2] + 1), dtype=float)

    d1 = zeros(p[0], dtype=float)
    d2 = zeros(p[1], dtype=float)
    d3 = zeros(p[2], dtype=float)

    der1 = zeros(p[0] + 1, dtype=float)
    der2 = zeros(p[1] + 1, dtype=float)
    der3 = zeros(p[2] + 1, dtype=float)

    for n in range(len(eta1s)):

        # spans (i.e. index for non-vanishing basis functions)
        span1 = bsp.find_span(t1, p[0], eta1s[n])
        span2 = bsp.find_span(t2, p[1], eta2s[n])
        span3 = bsp.find_span(t3, p[2], eta3s[n])

        f_df_pic_legacy(kind_map, params_map,
                 t1, t2, t3, p, span1, span2, span3,
                 ind1, ind2, ind3, cx, cy, cz,
                 l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, der1, der2, der3, eta1s[n], eta2s[n], eta3s[n], f_out[n, :], df_out[n, :, :], 2)


def loop_slim(kind_map: 'int', params_map: 'float[:]',
              t1: 'float[:]', t2: 'float[:]', t3: 'float[:]', p: 'int[:]', 
              ind1: 'int[:,:]', ind2: 'int[:,:]', ind3: 'int[:,:]',
              cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
              eta1s: 'float[:]', eta2s: 'float[:]', eta3s: 'float[:]',
              f_out: 'float[:,:]', df_out: 'float[:,:,:]'):

    for n in range(len(eta1s)):

        # spans (i.e. index for non-vanishing basis functions)
        span1 = bsp.find_span(t1, p[0], eta1s[n])
        span2 = bsp.find_span(t2, p[1], eta2s[n])
        span3 = bsp.find_span(t3, p[2], eta3s[n])

        f_df_pic(kind_map, params_map,
                      t1, t2, t3,
                      p, span1, span2, span3,
                      ind1, ind2, ind3, cx, cy, cz,
                      eta1s[n], eta2s[n], eta3s[n], f_out[n, :], df_out[n, :, :], 2)


def loop_f(kind_map : int, params_map : 'float[:]', 
            tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', 
            ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', 
            cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', 
            eta1s: 'float[:]', eta2s: 'float[:]', eta3s: 'float[:]',
            f_out : 'float[:,:]'):

    for n in range(len(eta1s)):

        f(eta1s[n], eta2s[n], eta3s[n],
            kind_map, params_map,
            tn1, tn2, tn3, pn, 
            ind_n1, ind_n2, ind_n3, 
            cx, cy, cz,
            f_out[n, :])


def loop_df(kind_map : int, params_map : 'float[:]', 
            tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', 
            ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', 
            cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', 
            eta1s: 'float[:]', eta2s: 'float[:]', eta3s: 'float[:]',
            mat_out : 'float[:,:,:]'):

    for n in range(len(eta1s)):

        df(eta1s[n], eta2s[n], eta3s[n],
            kind_map, params_map,
            tn1, tn2, tn3, pn, 
            ind_n1, ind_n2, ind_n3, 
            cx, cy, cz,
            mat_out[n, :, :])


def loop_detdf(kind_map : int, params_map : 'float[:]', 
            tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', 
            ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', 
            cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', 
            eta1s: 'float[:]', eta2s: 'float[:]', eta3s: 'float[:]',
            f_out : 'float[:]'):

    for n in range(len(eta1s)):

        f_out[n] = det_df(eta1s[n], eta2s[n], eta3s[n],
                            kind_map, params_map,
                            tn1, tn2, tn3, pn, 
                            ind_n1, ind_n2, ind_n3, 
                            cx, cy, cz)


def loop_dfinv(kind_map : int, params_map : 'float[:]', 
            tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', 
            ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', 
            cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', 
            eta1s: 'float[:]', eta2s: 'float[:]', eta3s: 'float[:]',
            mat_out : 'float[:,:,:]'):

    for n in range(len(eta1s)):

        df_inv(eta1s[n], eta2s[n], eta3s[n],
                kind_map, params_map,
                tn1, tn2, tn3, pn, 
                ind_n1, ind_n2, ind_n3, 
                cx, cy, cz,
                mat_out[n, :, :])


def loop_g(kind_map : int, params_map : 'float[:]', 
            tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', 
            ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', 
            cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', 
            eta1s: 'float[:]', eta2s: 'float[:]', eta3s: 'float[:]',
            mat_out : 'float[:,:,:]'):

    for n in range(len(eta1s)):

        g(eta1s[n], eta2s[n], eta3s[n],
                kind_map, params_map,
                tn1, tn2, tn3, pn, 
                ind_n1, ind_n2, ind_n3, 
                cx, cy, cz,
                mat_out[n, :, :])


def loop_ginv(kind_map : int, params_map : 'float[:]', 
            tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', 
            ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', 
            cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', 
            eta1s: 'float[:]', eta2s: 'float[:]', eta3s: 'float[:]',
            mat_out : 'float[:,:,:]'):

    for n in range(len(eta1s)):

        g_inv(eta1s[n], eta2s[n], eta3s[n],
                kind_map, params_map,
                tn1, tn2, tn3, pn, 
                ind_n1, ind_n2, ind_n3, 
                cx, cy, cz,
                mat_out[n, :, :])