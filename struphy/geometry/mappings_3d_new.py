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

from struphy.linear_algebra.core import det, matrix_matrix, transpose, matrix_inv


def f(eta1 : float, eta2 : float, eta3 : float, # evaluation point
      kind_map : int, params_map : 'float[:]', # mapping parameters
      t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p : 'int[:]', # spline mapping knots and degrees
      ind1 : 'int[:,:]', ind2 : 'int[:,:]', ind3 : 'int[:,:]', # spline index arrays
      cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', # control points (numpy array, cloned to each process)
      f_out: 'float[:]'): # output array
    """
    Point-wise evaluation of (x, y, z) = F(eta1, eta2, eta3). 

    Parameters
    ----------
        eta1, eta2, eta3 : float              
            Logical coordinates in [0, 1].
            
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params_map : array[float]
            Parameters for the mapping in a 1d array.
        
        t1, t2, t3 : array[float]          
            Knot vectors of univariate splines.
        
        p : array[int]
            Degrees of univariate splines.
        
        ind1, ind2, ind3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (F_x, F_y, F_z) in case of a IGA mapping.

        f_out : array[float]
            Output: (x, y, z) = F(eta1, eta2, eta3).
    """

    # --------------- 3d spline --------------------
    if kind_map == 0:

        # mapping spans
        span1 = bsp.find_span(t1, p[0], eta1)
        span2 = bsp.find_span(t2, p[1], eta2)
        span3 = bsp.find_span(t3, p[2], eta3)

        # p + 1 non-zero mapping splines 
        b1 = zeros(p[0] + 1, dtype=float)
        b2 = zeros(p[1] + 1, dtype=float)
        b3 = zeros(p[2] + 1, dtype=float)

        bsp.b_splines_slim(t1, p[0], eta1, span1, b1)
        bsp.b_splines_slim(t2, p[1], eta2, span2, b2)
        bsp.b_splines_slim(t3, p[2], eta3, span3, b3)

        # Evaluate spline mapping
        f_out[0] = eva_3d.evaluation_kernel_3d(p[0], p[1], p[2], b1, b2, b3, ind1[span1 - p[0], :], ind2[span2 - p[1], :], ind3[span3 - p[2], :], cx)
        f_out[1] = eva_3d.evaluation_kernel_3d(p[0], p[1], p[2], b1, b2, b3, ind1[span1 - p[0], :], ind2[span2 - p[1], :], ind3[span3 - p[2], :], cy)
        f_out[2] = eva_3d.evaluation_kernel_3d(p[0], p[1], p[2], b1, b2, b3, ind1[span1 - p[0], :], ind2[span2 - p[1], :], ind3[span3 - p[2], :], cz)

    # ---- 2d spline (straight in 3rd direction) ---
    elif kind_map == 1:

        # The length in the 3rd direction is tied to the value x_0 = F_x(0, 0, 0), which mimics a minor radius
        lz = 2*pi*cx[0, 0, 0]

        # mapping spans
        span1 = bsp.find_span(t1, p[0], eta1)
        span2 = bsp.find_span(t2, p[1], eta2)

        # p + 1 non-zero mapping splines 
        b1 = zeros(p[0] + 1, dtype=float)
        b2 = zeros(p[1] + 1, dtype=float)

        bsp.b_splines_slim(t1, p[0], eta1, span1, b1)
        bsp.b_splines_slim(t2, p[1], eta2, span2, b2)

        # Evaluate mapping
        f_out[0] = eva_2d.evaluation_kernel_2d(p[0], p[1], b1, b2, ind1[span1 - p[0], :], ind2[span2 - p[1], :], cx[:, :, 0])
        f_out[1] = eva_2d.evaluation_kernel_2d(p[0], p[1], b1, b2, ind1[span1 - p[0], :], ind2[span2 - p[1], :], cy[:, :, 0])
        f_out[2] = lz * eta3

        # TODO: explanation
        if eta1 == 0. and cx[0, 0, 0] == cx[0, 1, 0]:
            f_out[0] = cx[0, 0, 0]

        if eta1 == 0. and cy[0, 0, 0] == cy[0, 1, 0]:
            f_out[1] = cy[0, 0, 0]

    # ---- 2d spline (curvature in 3rd direction) ---
    elif kind_map == 2:

        # mapping spans
        span1 = bsp.find_span(t1, p[0], eta1)
        span2 = bsp.find_span(t2, p[1], eta2)

        # p + 1 non-zero mapping splines 
        b1 = zeros(p[0] + 1, dtype=float)
        b2 = zeros(p[1] + 1, dtype=float)

        bsp.b_splines_slim(t1, p[0], eta1, span1, b1)
        bsp.b_splines_slim(t2, p[1], eta2, span2, b2)

        # Evaluate mapping
        f_out[0] = eva_2d.evaluation_kernel_2d(p[0], p[1], b1, b2, ind1[span1 - p[0], :], ind2[span2 - p[1], :], cx[:, :, 0]) * cos(2*pi*eta3)
        f_out[1] = eva_2d.evaluation_kernel_2d(p[0], p[1], b1, b2, ind1[span1 - p[0], :], ind2[span2 - p[1], :], cy[:, :, 0])
        f_out[2] = eva_2d.evaluation_kernel_2d(p[0], p[1], b1, b2, ind1[span1 - p[0], :], ind2[span2 - p[1], :], cx[:, :, 0]) * sin(2*pi*eta3)

        # TODO: explanation
        if eta1 == 0. and cx[0, 0, 0] == cx[0, 1, 0]:
            f_out[0] = cx[0, 0, 0]*cos(2*pi*eta3)

        if eta1 == 0. and cy[0, 0, 0] == cy[0, 1, 0]:
            f_out[1] = cy[0, 0, 0]

        if eta1 == 0. and cx[0, 0, 0] == cx[0, 1, 0]:
            f_out[2] = cx[0, 0, 0]*sin(2*pi*eta3)

    # -------------- cuboid -------------------------
    elif kind_map == 10:

        l1 = params_map[0]
        r1 = params_map[1]
        l2 = params_map[2]
        r2 = params_map[3]
        l3 = params_map[4]
        r3 = params_map[5]

        # value =  begin + (end - begin) * eta
        f_out[0] = l1 + (r1 - l1) * eta1
        f_out[1] = l2 + (r2 - l2) * eta2
        f_out[2] = l3 + (r3 - l3) * eta3

    # --------- hollow cylinder ---------------------
    elif kind_map == 11:

        a1 = params_map[0]
        a2 = params_map[1]
        r0 = params_map[2]

        da = a2 - a1

        f_out[0] = (a1 + eta1 * da) * cos(2*pi*eta2) + r0
        f_out[1] = (a1 + eta1 * da) * sin(2*pi*eta2)
        f_out[2] = 2*pi*r0 * eta3

    # ------------ colella --------------------------
    elif kind_map == 12:

        lx    = params_map[0]
        ly    = params_map[1]
        alpha = params_map[2]
        lz    = params_map[3]

        f_out[0] = lx * (eta1 + alpha * sin(2*pi*eta1) * sin(2*pi*eta2))
        f_out[1] = ly * (eta2 + alpha * sin(2*pi*eta1) * sin(2*pi*eta2))
        f_out[2] = lz * eta3

    # ----------- orthogonal ------------------------
    elif kind_map == 13:

        lx    = params_map[0]
        ly    = params_map[1]
        alpha = params_map[2]
        lz    = params_map[3]

        f_out[0] = lx * (eta1 + alpha * sin(2*pi*eta1))
        f_out[1] = ly * (eta2 + alpha * sin(2*pi*eta2))
        f_out[2] = lz * eta3

    # --------- hollow torus ------------------------
    elif kind_map == 14:

        a1 = params_map[0]
        a2 = params_map[1]
        r0 = params_map[2]

        da = a2 - a1

        f_out[0] = ((a1 + eta1 * da) * cos(2*pi*eta2) + r0) * cos(2*pi*eta3)
        f_out[1] =  (a1 + eta1 * da) * sin(2*pi*eta2)
        f_out[2] = ((a1 + eta1 * da) * cos(2*pi*eta2) + r0) * sin(2*pi*eta3)

    # ------------- ellipse -------------------------
    elif kind_map == 15:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        lz = params_map[5]

        f_out[0] = x0 + (eta1 * rx) * cos(2*pi*eta2)
        f_out[1] = y0 + (eta1 * ry) * sin(2*pi*eta2)
        f_out[2] = z0 + (eta3 * lz)

    # --------- rotated ellipse ---------------------
    elif kind_map == 16:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        r1 = params_map[3]
        r2 = params_map[4]
        lz = params_map[5]
        th = params_map[6] # Domain: [0,1)

        f_out[0] = x0 + (eta1 * r1) * cos(2*pi*th) * cos(2*pi*eta2) - (eta1 * r2) * sin(2*pi*th) * sin(2*pi*eta2)
        f_out[1] = y0 + (eta1 * r1) * sin(2*pi*th) * cos(2*pi*eta2) + (eta1 * r2) * cos(2*pi*th) * sin(2*pi*eta2)
        f_out[2] = z0 + (eta3 * lz)

    # --------- shafranov shift ---------------------
    elif kind_map == 17:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        lz = params_map[5]
        de = params_map[6] # Domain: [0,0.1]

        f_out[0] = x0 + (eta1 * rx) * cos(2*pi*eta2) + (1-eta1**2) * rx * de
        f_out[1] = y0 + (eta1 * ry) * sin(2*pi*eta2)
        f_out[2] = z0 + (eta3 * lz)

    # --------- shafranov sqrt ---------------------
    elif kind_map == 18:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        lz = params_map[5]
        de = params_map[6] # Domain: [0,0.1]

        f_out[0] = x0 + (eta1 * rx) * cos(2*pi*eta2) + (1-sqrt(eta1)) * rx * de
        f_out[1] = y0 + (eta1 * ry) * sin(2*pi*eta2)
        f_out[2] = z0 + (eta3 * lz)

    # --------- shafranov D-shaped ---------------------
    elif kind_map == 19:

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

        f_out[0] = x0 + r0 * (1 + (1 - eta1**2) * dx + eg *      eta1 * cos(2*pi*eta2 + arcsin(dg)*eta1*sin(2*pi*eta2)))
        f_out[1] = y0 + r0 * (    (1 - eta1**2) * dy + eg * kg * eta1 * sin(2*pi*eta2))
        f_out[2] = z0 + (eta3 * lz)
    
 
def df(eta1 : float, eta2 : float, eta3 : float, # evaluation point
      kind_map : int, params_map : 'float[:]', # mapping parameters
      t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p : 'int[:]', # spline mapping knots and degrees
      ind1 : 'int[:,:]', ind2 : 'int[:,:]', ind3 : 'int[:,:]', # spline index arrays
      cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', # control points (numpy array, cloned to each process)
      df_out: 'float[:,:]'): # output array
    """
    Point-wise evaluation of the Jacobian matrix DF = (dF_i/deta_j)_(i,j=1,2,3). 

    Parameters
    ----------
        eta1, eta2, eta3 : float              
            Logical coordinates in [0, 1].
            
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params_map : array[float]
            Parameters for the mapping in a 1d array.
        
        t1, t2, t3 : array[float]          
            Knot vectors of univariate splines.
        
        p : array[int]
            Degrees of univariate splines.
        
        ind1, ind2, ind3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (F_x, F_y, F_z) in case of a IGA mapping.

        df_out : array[float]
            Output: DF(eta1, eta2, eta3).
    """

    value = 0.

    # ----------- 3d spline ------------------------
    if kind_map == 0:

        # mapping spans
        span1 = bsp.find_span(t1, p[0], eta1)
        span2 = bsp.find_span(t2, p[1], eta2)
        span3 = bsp.find_span(t3, p[2], eta3)

        # non-zero splines of mapping, and derivatives
        b1 = zeros(p[0] + 1, dtype=float)
        b2 = zeros(p[1] + 1, dtype=float)
        b3 = zeros(p[2] + 1, dtype=float)

        der1 = zeros(p[0] + 1, dtype=float)
        der2 = zeros(p[1] + 1, dtype=float)
        der3 = zeros(p[2] + 1, dtype=float)

        bsp.b_der_splines_slim(t1, p[0], eta1, span1, b1, der1)
        bsp.b_der_splines_slim(t2, p[1], eta2, span2, b2, der2)
        bsp.b_der_splines_slim(t3, p[2], eta3, span3, b3, der3)
        
        # Evaluation of Jacobian
        df_out[0, 0] = eva_3d.evaluation_kernel_3d(p[0], p[1], p[2], der1, b2, b3, ind1[span1 - p[0], :], ind2[span2 - p[1], :], ind3[span3 - p[2], :], cx)
        df_out[0, 1] = eva_3d.evaluation_kernel_3d(p[0], p[1], p[2], b1, der2, b3, ind1[span1 - p[0], :], ind2[span2 - p[1], :], ind3[span3 - p[2], :], cx)
        df_out[0, 2] = eva_3d.evaluation_kernel_3d(p[0], p[1], p[2], b1, b2, der3, ind1[span1 - p[0], :], ind2[span2 - p[1], :], ind3[span3 - p[2], :], cx)
        df_out[1, 0] = eva_3d.evaluation_kernel_3d(p[0], p[1], p[2], der1, b2, b3, ind1[span1 - p[0], :], ind2[span2 - p[1], :], ind3[span3 - p[2], :], cy)
        df_out[1, 1] = eva_3d.evaluation_kernel_3d(p[0], p[1], p[2], b1, der2, b3, ind1[span1 - p[0], :], ind2[span2 - p[1], :], ind3[span3 - p[2], :], cy)
        df_out[1, 2] = eva_3d.evaluation_kernel_3d(p[0], p[1], p[2], b1, b2, der3, ind1[span1 - p[0], :], ind2[span2 - p[1], :], ind3[span3 - p[2], :], cy)
        df_out[2, 0] = eva_3d.evaluation_kernel_3d(p[0], p[1], p[2], der1, b2, b3, ind1[span1 - p[0], :], ind2[span2 - p[1], :], ind3[span3 - p[2], :], cz)
        df_out[2, 1] = eva_3d.evaluation_kernel_3d(p[0], p[1], p[2], b1, der2, b3, ind1[span1 - p[0], :], ind2[span2 - p[1], :], ind3[span3 - p[2], :], cz)
        df_out[2, 2] = eva_3d.evaluation_kernel_3d(p[0], p[1], p[2], b1, b2, der3, ind1[span1 - p[0], :], ind2[span2 - p[1], :], ind3[span3 - p[2], :], cz)
               
    # ----- 2d spline (straight in 3rd direction) ---
    elif kind_map == 1:
        
        # The length in the 3rd direction is tied to the value x_0 = F_x(0, 0, 0), which mimics a minor radius
        lz = 2*pi*cx[0, 0, 0]

        # mapping spans
        span1 = bsp.find_span(t1, p[0], eta1)
        span2 = bsp.find_span(t2, p[1], eta2)

        # non-zero splines of mapping, and derivatives
        b1 = zeros(p[0] + 1, dtype=float)
        b2 = zeros(p[1] + 1, dtype=float)

        der1 = zeros(p[0] + 1, dtype=float)
        der2 = zeros(p[1] + 1, dtype=float)

        bsp.b_der_splines_slim(t1, p[0], eta1, span1, b1, der1)
        bsp.b_der_splines_slim(t2, p[1], eta2, span2, b2, der2)
        
        # Evaluation of Jacobian
        df_out[0, 0] = eva_2d.evaluation_kernel_2d(p[0], p[1], der1, b2, ind1[span1 - p[0], :], ind2[span2 - p[1], :], cx[:, :, 0])
        df_out[0, 1] = eva_2d.evaluation_kernel_2d(p[0], p[1], b1, der2, ind1[span1 - p[0], :], ind2[span2 - p[1], :], cx[:, :, 0])
        df_out[0, 2] = 0.
        df_out[1, 0] = eva_2d.evaluation_kernel_2d(p[0], p[1], der1, b2, ind1[span1 - p[0], :], ind2[span2 - p[1], :], cy[:, :, 0])
        df_out[1, 1] = eva_2d.evaluation_kernel_2d(p[0], p[1], b1, der2, ind1[span1 - p[0], :], ind2[span2 - p[1], :], cy[:, :, 0])
        df_out[1, 2] = 0.
        df_out[2, 0] = 0.
        df_out[2, 1] = 0.
        df_out[2, 2] = lz

        # TODO: explanation
        if eta1 == 0. and cx[0, 0, 0] == cx[0, 1, 0]:
            df_out[0, 1] = 0.

        if eta1 == 0. and cy[0, 0, 0] == cy[0, 1, 0]:
            df_out[1, 1] = 0.
    
    # ---- 2d spline (curvature in 3rd direction) ---
    elif kind_map == 2:

        # mapping spans
        span1 = bsp.find_span(t1, p[0], eta1)
        span2 = bsp.find_span(t2, p[1], eta2)

        # non-zero splines of mapping, and derivatives
        b1 = zeros(p[0] + 1, dtype=float)
        b2 = zeros(p[1] + 1, dtype=float)

        der1 = zeros(p[0] + 1, dtype=float)
        der2 = zeros(p[1] + 1, dtype=float)

        bsp.b_der_splines_slim(t1, p[0], eta1, span1, b1, der1)
        bsp.b_der_splines_slim(t2, p[1], eta2, span2, b2, der2)
        
        df_out[0, 0] = eva_2d.evaluation_kernel_2d(p[0], p[1], der1, b2, ind1[span1 - p[0], :], ind2[span2 - p[1], :], cx[:, :, 0]) * cos(2*pi*eta3)
        df_out[0, 1] = eva_2d.evaluation_kernel_2d(p[0], p[1], b1, der2, ind1[span1 - p[0], :], ind2[span2 - p[1], :], cx[:, :, 0]) * cos(2*pi*eta3)
        df_out[0, 2] = eva_2d.evaluation_kernel_2d(p[0], p[1], b1, b2, ind1[span1 - p[0], :], ind2[span2 - p[1], :], cx[:, :, 0]) * sin(2*pi*eta3) * (-2*pi)
        df_out[1, 0] = eva_2d.evaluation_kernel_2d(p[0], p[1], der1, b2, ind1[span1 - p[0], :], ind2[span2 - p[1], :], cy[:, :, 0])
        df_out[1, 1] = eva_2d.evaluation_kernel_2d(p[0], p[1], b1, der2, ind1[span1 - p[0], :], ind2[span2 - p[1], :], cy[:, :, 0])
        df_out[1, 2] = 0.
        df_out[2, 0] = eva_2d.evaluation_kernel_2d(p[0], p[1], der1, b2, ind1[span1 - p[0], :], ind2[span2 - p[1], :], cx[:, :, 0]) * sin(2*pi*eta3)
        df_out[2, 1] = eva_2d.evaluation_kernel_2d(p[0], p[1], b1, der2, ind1[span1 - p[0], :], ind2[span2 - p[1], :], cx[:, :, 0]) * sin(2*pi*eta3)
        df_out[2, 2] = eva_2d.evaluation_kernel_2d(p[0], p[1], b1, b2, ind1[span1 - p[0], :], ind2[span2 - p[1], :], cx[:, :, 0]) * cos(2*pi*eta3) * 2*pi

        # TODO: explanation
        if eta1 == 0. and cx[0, 0, 0] == cx[0, 1, 0]:
                df_out[0, 1] = 0.

        if eta1 == 0. and cy[0, 0, 0] == cy[0, 1, 0]:
                df_out[1, 1] = 0.

        if eta1 == 0. and cx[0, 0, 0] == cx[0, 1, 0]:
                df_out[2, 1] = 0.
    
    # ---------------- cuboid -----------------------
    elif kind_map == 10:
         
        l1 = params_map[0]
        r1 = params_map[1]
        l2 = params_map[2]
        r2 = params_map[3]
        l3 = params_map[4]
        r3 = params_map[5]
        
        df_out[0, 0] = r1 - l1
        df_out[0, 1] = 0.
        df_out[0, 2] = 0.
        df_out[1, 0] = 0.
        df_out[1, 1] = r2 - l2
        df_out[1, 2] = 0.
        df_out[2, 0] = 0.
        df_out[2, 1] = 0.
        df_out[2, 2] = r3 - l3
            
    # ------------ hollow cylinder -------------------
    elif kind_map == 11:

        a1 = params_map[0]
        a2 = params_map[1]
        r0 = params_map[2]

        da = a2 - a1

        df_out[0, 0] = da * cos(2*pi*eta2)
        df_out[0, 1] = -2*pi * (a1 + eta1 * da) * sin(2*pi*eta2)
        df_out[0, 2] = 0.
        df_out[1, 0] = da * sin(2*pi*eta2)
        df_out[1, 1] = 2*pi * (a1 + eta1 * da) * cos(2*pi*eta2)
        df_out[1, 2] = 0.
        df_out[2, 0] = 0.
        df_out[2, 1] = 0.
        df_out[2, 2] = 2*pi*r0

    # ---------------- colella -----------------------
    elif kind_map == 12:

        lx    = params_map[0]
        ly    = params_map[1]
        alpha = params_map[2]
        lz    = params_map[3]

        df_out[0, 0] = lx * (1 + alpha * cos(2*pi*eta1) * sin(2*pi*eta2) * 2*pi)
        df_out[0, 1] = lx * alpha * sin(2*pi*eta1) * cos(2*pi*eta2) * 2*pi
        df_out[0, 2] = 0.
        df_out[1, 0] = ly * alpha * cos(2*pi*eta1) * sin(2*pi*eta2) * 2*pi
        df_out[1, 1] = ly * (1 + alpha * sin(2*pi*eta1) * cos(2*pi*eta2) * 2*pi)
        df_out[1, 2] = 0.
        df_out[2, 0] = 0.
        df_out[2, 1] = 0.    
        df_out[2, 2] = lz

    # ------------------ orthogonal -------------------
    elif kind_map == 13:

        lx    = params_map[0]
        ly    = params_map[1]
        alpha = params_map[2]
        lz    = params_map[3]

        df_out[0, 0] = lx * (1 + alpha * cos(2*pi*eta1) * 2*pi)
        df_out[0, 1] = 0.
        df_out[0, 2] = 0.
        df_out[1, 0] = 0.
        df_out[1, 1] = ly * (1 + alpha * cos(2*pi*eta2) * 2*pi)
        df_out[1, 2] = 0.
        df_out[2, 0] = 0.
        df_out[2, 1] = 0.    
        df_out[2, 2] = lz

    # -------------- hollow torus ----------------------
    elif kind_map == 14:

        a1 = params_map[0]
        a2 = params_map[1]
        r0 = params_map[2]

        da = a2 - a1

        df_out[0, 0] = da * cos(2*pi*eta2) * cos(2*pi*eta3)
        df_out[0, 1] = -2*pi * (a1 + eta1 * da) * sin(2*pi*eta2) * cos(2*pi*eta3)
        df_out[0, 2] = -2*pi * ((a1 + eta1 * da) * cos(2*pi*eta2) + r0) * sin(2*pi*eta3)
        df_out[1, 0] = da * sin(2*pi*eta2)
        df_out[1, 1] = (a1 + eta1 * da) * cos(2*pi*eta2) * 2*pi
        df_out[1, 2] = 0.
        df_out[2, 0] = da * cos(2*pi*eta2) * sin(2*pi*eta3)
        df_out[2, 1] = -2*pi * (a1 + eta1 * da) * sin(2*pi*eta2) * sin(2*pi*eta3)
        df_out[2, 2] = ((a1 + eta1 * da) * cos(2*pi*eta2) + r0) * cos(2*pi*eta3) * 2*pi

    # ----------------- ellipse -------------------------
    elif kind_map == 15:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        lz = params_map[5]

        df_out[0, 0] = rx * cos(2*pi*eta2)
        df_out[0, 1] = -2*pi * (eta1 * rx) * sin(2*pi*eta2)
        df_out[0, 2] = 0.
        df_out[1, 0] = ry * sin(2*pi*eta2)
        df_out[1, 1] =  2*pi * (eta1 * ry) * cos(2*pi*eta2)
        df_out[1, 2] = 0.
        df_out[2, 0] = 0.
        df_out[2, 1] = 0.
        df_out[2, 2] = lz

    # -------------- rotated ellipse ---------------------
    elif kind_map == 16:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        r1 = params_map[3]
        r2 = params_map[4]
        lz = params_map[5]
        th = params_map[6] # Domain: [0,1)

        df_out[0, 0] = r1 * cos(2*pi*th) * cos(2*pi*eta2) - r2 * sin(2*pi*th) * sin(2*pi*eta2)
        df_out[0, 1] = -2*pi * (eta1 * r1) * cos(2*pi*th) * sin(2*pi*eta2) - 2*pi * (eta1 * r2) * sin(2*pi*th) * cos(2*pi*eta2)
        df_out[0, 2] = 0.
        df_out[1, 0] = r1 * sin(2*pi*th) * cos(2*pi*eta2) + r2 * cos(2*pi*th) * sin(2*pi*eta2)
        df_out[1, 1] = -2*pi * (eta1 * r1) * sin(2*pi*th) * sin(2*pi*eta2) + 2*pi * (eta1 * r2) * cos(2*pi*th) * cos(2*pi*eta2)
        df_out[1, 2] = 0.
        df_out[2, 0] = 0.
        df_out[2, 1] = 0.
        df_out[2, 2] = lz

    # --------------- shafranov shift ---------------------
    elif kind_map == 17:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        lz = params_map[5]
        de = params_map[6] # Domain: [0,0.1]

        df_out[0, 0] = rx * cos(2*pi*eta2) - 2 * eta1 * rx * de
        df_out[0, 1] = -2*pi * (eta1 * rx) * sin(2*pi*eta2)
        df_out[0, 2] = 0.
        df_out[1, 0] = ry * sin(2*pi*eta2)
        df_out[1, 1] =  2*pi * (eta1 * ry) * cos(2*pi*eta2)
        df_out[1, 2] = 0.
        df_out[2, 0] = 0.
        df_out[2, 1] = 0.
        df_out[2, 2] = lz

    # ----------------- shafranov sqrt ---------------------
    elif kind_map == 18:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        lz = params_map[5]
        de = params_map[6] # Domain: [0,0.1]

        df_out[0, 0] = rx * cos(2*pi*eta2) - 0.5 / sqrt(eta1) * rx * de
        df_out[0, 1] = -2*pi * (eta1 * rx) * sin(2*pi*eta2)
        df_out[0, 2] = 0.
        df_out[1, 0] = ry * sin(2*pi*eta2)
        df_out[1, 1] =  2*pi * (eta1 * ry) * cos(2*pi*eta2)
        df_out[1, 2] = 0.
        df_out[2, 0] = 0.
        df_out[2, 1] = 0.
        df_out[2, 2] = lz

    # --------------- shafranov D-shaped ---------------------
    elif kind_map == 19:

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

        df_out[0, 0] = r0 * (- 2 * dx * eta1 - eg * eta1 * sin(2*pi*eta2) * arcsin(dg) * sin(eta1 * sin(2*pi*eta2) * arcsin(dg) + 2*pi*eta2) + eg * cos(eta1 * sin(2*pi*eta2) * arcsin(dg) + 2*pi*eta2))
        df_out[0, 1] = - r0 * eg * eta1 * (2*pi*eta1 * cos(2*pi*eta2) * arcsin(dg) + 2*pi) * sin(eta1 * sin(2*pi*eta2) * arcsin(dg) + 2*pi*eta2)
        df_out[0, 2] = 0.
        df_out[1, 0] = r0 * (- 2 * dy * eta1 + eg * kg * sin(2*pi*eta2))
        df_out[1, 1] = 2 * pi * r0 * eg * eta1 * kg * cos(2*pi*eta2)
        df_out[1, 2] = 0.
        df_out[2, 0] = 0.
        df_out[2, 1] = 0.
        df_out[2, 2] = lz

    return value

  
def det_df(eta1 : float, eta2 : float, eta3 : float, # evaluation point
      kind_map : int, params_map : 'float[:]', # mapping parameters
      t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p : 'int[:]', # spline mapping knots and degrees
      ind1 : 'int[:,:]', ind2 : 'int[:,:]', ind3 : 'int[:,:]', # spline index arrays
      cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float: # control points (numpy array, cloned to each process)
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
        
        t1, t2, t3 : array[float]          
            Knot vectors of univariate splines.
        
        p : array[int]
            Degrees of univariate splines.
        
        ind1, ind2, ind3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (F_x, F_y, F_z) in case of a IGA mapping.

    Returns:
    --------
        detdf : float
            Jacobian determinant det(df)(eta1, eta2, eta3).
    """
    
    df_mat = empty((3, 3), dtype=float)
    df(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, df_mat)
    
    detdf = det(df_mat)
            
    return detdf


def df_inv(eta1 : float, eta2 : float, eta3 : float, # evaluation point
      kind_map : int, params_map : 'float[:]', # mapping parameters
      t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p : 'int[:]', # spline mapping knots and degrees
      ind1 : 'int[:,:]', ind2 : 'int[:,:]', ind3 : 'int[:,:]', # spline index arrays
      cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', # control points (numpy array, cloned to each process)
      dfinv_out: 'float[:,:]'): # output array
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
            
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params_map : array[float]
            Parameters for the mapping in a 1d array.
        
        t1, t2, t3 : array[float]          
            Knot vectors of univariate splines.
        
        p : array[int]
            Degrees of univariate splines.
        
        ind1, ind2, ind3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (F_x, F_y, F_z) in case of a IGA mapping.

        dfinv_out : array[float]
            Output: the inverse Jacobian matrix at (eta1, eta2, eta3).
    """
    
    df_mat = empty((3, 3), dtype=float)
    df(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, df_mat)
    
    matrix_inv(df_mat, dfinv_out)
    
    
def g(eta1 : float, eta2 : float, eta3 : float, # evaluation point
      kind_map : int, params_map : 'float[:]', # mapping parameters
      t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p : 'int[:]', # spline mapping knots and degrees
      ind1 : 'int[:,:]', ind2 : 'int[:,:]', ind3 : 'int[:,:]', # spline index arrays
      cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', # control points (numpy array, cloned to each process)
      g_out: 'float[:,:]'): # output array
    """
    Point-wise evaluation of the metric tensor g = df^T * df. 
    
    Parameters
    ----------
        eta1, eta2, eta3 : float              
            Logical coordinates in [0, 1].
            
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params_map : array[float]
            Parameters for the mapping in a 1d array.
        
        t1, t2, t3 : array[float]          
            Knot vectors of univariate splines.
        
        p : array[int]
            Degrees of univariate splines.
        
        ind1, ind2, ind3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (F_x, F_y, F_z) in case of a IGA mapping.

        g_out : array[float]
            Output: g(eta1, eta2, eta3).
    """
    
    df_mat = empty((3, 3), dtype=float)
    df(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, df_mat)

    df_t = empty((3, 3), dtype=float)
    transpose(df_mat, df_t)

    matrix_matrix(df_t, df_mat, g_out)


def g_inv(eta1 : float, eta2 : float, eta3 : float, # evaluation point
      kind_map : int, params_map : 'float[:]', # mapping parameters
      t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p : 'int[:]', # spline mapping knots and degrees
      ind1 : 'int[:,:]', ind2 : 'int[:,:]', ind3 : 'int[:,:]', # spline index arrays
      cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', # control points (numpy array, cloned to each process)
      ginv_out: 'float[:,:]'): # output array
    """
    Point-wise evaluation of the inverse metric tensor g^(-1) = df^(-1) * df^(-T). 
    
    Parameters
    ----------
        eta1, eta2, eta3 : float              
            Logical coordinates in [0, 1].
            
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params_map : array[float]
            Parameters for the mapping in a 1d array.
        
        t1, t2, t3 : array[float]          
            Knot vectors of univariate splines.
        
        p : array[int]
            Degrees of univariate splines.
        
        ind1, ind2, ind3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (F_x, F_y, F_z) in case of a IGA mapping.

        ginv_out : array[float]
            Output: g^(-1)(eta1, eta2, eta3).
    """
    
    g_mat = empty((3, 3), dtype=float)
    g(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, g_mat)

    matrix_inv(g_mat, ginv_out)
    
    
def mappings_all(eta1 : float, eta2 : float, eta3 : float, # evaluation point
            kind_fun : int, # metric coefficient key
            kind_map : int, params_map : 'float[:]', # mapping parameters 
            t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p : 'int[:]', # spline mapping knots and degrees
            ind1 : 'int[:,:]', ind2 : 'int[:,:]', ind3 : 'int[:,:]', # spline index arrays
            cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float: # control points (numpy array, cloned to each process)
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
        
        t1, t2, t3 : array[float]          
            Knot vectors of univariate splines.
        
        p : array[int]
            Degrees of univariate splines.
        
        ind1, ind2, ind3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (F_x, F_y, F_z) in case of a IGA mapping.

    Returns:
    --------
        value : float
            Point value of metric coefficient at (eta1, eta2, eta3).
    """
    
    value = 0.
    
    # mapping f
    if   kind_fun == 1:
        f_vec = empty(3, dtype=float)
        f(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, f_vec)
        value = f_vec[0]
    elif kind_fun == 2:
        f_vec = empty(3, dtype=float)
        f(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, f_vec)
        value = f_vec[1]
    elif kind_fun == 3:
        f_vec = empty(3, dtype=float)
        f(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, f_vec)
        value = f_vec[2]
    
    # Jacobian matrix df
    elif kind_fun == 11:
        df_mat = empty((3, 3), dtype=float)
        df(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, df_mat)
        value = df_mat[0, 0]
    elif kind_fun == 12:
        df_mat = empty((3, 3), dtype=float)
        df(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, df_mat)
        value = df_mat[0, 1]
    elif kind_fun == 13:
        df_mat = empty((3, 3), dtype=float)
        df(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, df_mat)
        value = df_mat[0, 2]
    elif kind_fun == 14:
        df_mat = empty((3, 3), dtype=float)
        df(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, df_mat)
        value = df_mat[1, 0]
    elif kind_fun == 15:
        df_mat = empty((3, 3), dtype=float)
        df(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, df_mat)
        value = df_mat[1, 1]
    elif kind_fun == 16:
        df_mat = empty((3, 3), dtype=float)
        df(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, df_mat)
        value = df_mat[1, 2]
    elif kind_fun == 17:
        df_mat = empty((3, 3), dtype=float)
        df(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, df_mat)
        value = df_mat[2, 0]
    elif kind_fun == 18:
        df_mat = empty((3, 3), dtype=float)
        df(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, df_mat)
        value = df_mat[2, 1]
    elif kind_fun == 19:
        df_mat = empty((3, 3), dtype=float)
        df(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, df_mat)
        value = df_mat[2, 2]
        
    # Jacobian determinant det_df
    elif kind_fun == 4:
        value = det_df(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz)
        
    # inverse Jacobian matrix df_inv
    elif kind_fun == 21:
        dfinv_mat = empty((3, 3), dtype=float)
        df_inv(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, dfinv_mat)
        value = dfinv_mat[0, 0]
    elif kind_fun == 22:
        dfinv_mat = empty((3, 3), dtype=float)
        df_inv(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, dfinv_mat)
        value = dfinv_mat[0, 1]
    elif kind_fun == 23:
        dfinv_mat = empty((3, 3), dtype=float)
        df_inv(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, dfinv_mat)
        value = dfinv_mat[0, 2]
    elif kind_fun == 24:
        dfinv_mat = empty((3, 3), dtype=float)
        df_inv(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, dfinv_mat)
        value = dfinv_mat[1, 0]
    elif kind_fun == 25:
        dfinv_mat = empty((3, 3), dtype=float)
        df_inv(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, dfinv_mat)
        value = dfinv_mat[1, 1]
    elif kind_fun == 26:
        dfinv_mat = empty((3, 3), dtype=float)
        df_inv(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, dfinv_mat)
        value = dfinv_mat[1, 2]
    elif kind_fun == 27:
        dfinv_mat = empty((3, 3), dtype=float)
        df_inv(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, dfinv_mat)
        value = dfinv_mat[2, 0]
    elif kind_fun == 28:
        dfinv_mat = empty((3, 3), dtype=float)
        df_inv(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, dfinv_mat)
        value = dfinv_mat[2, 1]
    elif kind_fun == 29:
        dfinv_mat = empty((3, 3), dtype=float)
        df_inv(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, dfinv_mat)
        value = dfinv_mat[2, 2]
        
    # metric tensor g
    elif kind_fun == 31:
        g_mat = empty((3, 3), dtype=float)
        g(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, g_mat)
        value = g_mat[0, 0]
    elif kind_fun == 32:
        g_mat = empty((3, 3), dtype=float)
        g(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, g_mat)
        value = g_mat[0, 1]
    elif kind_fun == 33:
        g_mat = empty((3, 3), dtype=float)
        g(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, g_mat)
        value = g_mat[0, 2]
    elif kind_fun == 34:
        g_mat = empty((3, 3), dtype=float)
        g(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, g_mat)
        value = g_mat[1, 0]
    elif kind_fun == 35:
        g_mat = empty((3, 3), dtype=float)
        g(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, g_mat)
        value = g_mat[1, 1]
    elif kind_fun == 36:
        g_mat = empty((3, 3), dtype=float)
        g(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, g_mat)
        value = g_mat[1, 2]
    elif kind_fun == 37:
        g_mat = empty((3, 3), dtype=float)
        g(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, g_mat)
        value = g_mat[2, 0]     
    elif kind_fun == 38:
        g_mat = empty((3, 3), dtype=float)
        g(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, g_mat)
        value = g_mat[2, 1]
    elif kind_fun == 39:
        g_mat = empty((3, 3), dtype=float)
        g(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, g_mat)
        value = g_mat[2, 2]
    
    # metric tensor g_inv
    elif kind_fun == 41:
        ginv_mat = empty((3, 3), dtype=float)
        g_inv(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, ginv_mat)
        value = ginv_mat[0, 0]
    elif kind_fun == 42:
        ginv_mat = empty((3, 3), dtype=float)
        g_inv(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, ginv_mat)
        value = ginv_mat[0, 1]
    elif kind_fun == 43:
        ginv_mat = empty((3, 3), dtype=float)
        g_inv(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, ginv_mat)
        value = ginv_mat[0, 2]  
    elif kind_fun == 44:
        ginv_mat = empty((3, 3), dtype=float)
        g_inv(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, ginv_mat)
        value = ginv_mat[1, 0]
    elif kind_fun == 45:
        ginv_mat = empty((3, 3), dtype=float)
        g_inv(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, ginv_mat)
        value = ginv_mat[1, 1]
    elif kind_fun == 46:
        ginv_mat = empty((3, 3), dtype=float)
        g_inv(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, ginv_mat)
        value = ginv_mat[1, 2]
    elif kind_fun == 47:
        ginv_mat = empty((3, 3), dtype=float)
        g_inv(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, ginv_mat)
        value = ginv_mat[2, 0]
    elif kind_fun == 48:
        ginv_mat = empty((3, 3), dtype=float)
        g_inv(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, ginv_mat)
        value = ginv_mat[2, 1]
    elif kind_fun == 49:
        ginv_mat = empty((3, 3), dtype=float)
        g_inv(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, ginv_mat)
        value = ginv_mat[2, 2]
    
    return value

   
def kernel_evaluate(eta1 : 'float[:,:,:]', eta2 : 'float[:,:,:]', eta3 : 'float[:,:,:]', kind_fun : int, kind_map : int, params_map : 'float[:]', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p : 'int[:]', ind1 : 'int[:,:]', ind2 : 'int[:,:]', ind3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', mat_f : 'float[:,:,:]'):
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
        
        t1, t2, t3 : array[float]          
            Knot vectors of univariate splines.
        
        p : array[int]
            Degrees of univariate splines.
        
        ind1, ind2, ind3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (F_x, F_y, F_z) in case of a IGA mapping.
            
        mat_f : array[float]
            matrix-valued mapping/metric coefficient evaluated at (eta1, eta2, eta3).
    """

    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                mat_f[i1, i2, i3] = mappings_all(eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_fun, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz)

     
def kernel_evaluate_sparse(eta1 : 'float[:,:,:]', eta2 : 'float[:,:,:]', eta3 : 'float[:,:,:]', kind_fun : int, kind_map : int, params_map : 'float[:]', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p : 'int[:]', ind1 : 'int[:,:]', ind2 : 'int[:,:]', ind3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', mat_f : 'float[:,:,:]'):
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
        
        t1, t2, t3 : array[float]          
            Knot vectors of univariate splines.
        
        p : array[int]
            Degrees of univariate splines.
        
        ind1, ind2, ind3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (F_x, F_y, F_z) in case of a IGA mapping.
            
        mat_f : array[float]
            matrix-valued mapping/metric coefficient evaluated at (eta1, eta2, eta3).
    """

    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                mat_f[i1, i2, i3] = mappings_all(eta1[i1, 0, 0], eta2[0, i2, 0], eta3[0, 0, i3], kind_fun, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz)

                     
def kernel_evaluate_flat(eta1 : 'float[:]', eta2 : 'float[:]', eta3 : 'float[:]', kind_fun : int, kind_map : int, params_map : 'float[:]', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p : 'int[:]', ind1 : 'int[:,:]', ind2 : 'int[:,:]', ind3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', mat_f : 'float[:]'):
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
        
        t1, t2, t3 : array[float]          
            Knot vectors of univariate splines.
        
        p : array[int]
            Degrees of univariate splines.
        
        ind1, ind2, ind3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (F_x, F_y, F_z) in case of a IGA mapping.
            
        mat_f : array[float]
            matrix-valued mapping/metric coefficient evaluated at (eta1, eta2, eta3).

    Returns
    -------
        mat_f:  np.array
            1d array [f(x1, y1, z1) f(x2, y2, z2) etc.]
    """

    for i in range(len(eta1)):
        mat_f[i] = mappings_all(eta1[i], eta2[i], eta3[i], kind_fun, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz)


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


def loop_f_and_df(kind_map : int, params_map : 'float[:]', 
            tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', 
            ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', 
            cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', 
            eta1s: 'float[:]', eta2s: 'float[:]', eta3s: 'float[:]',
            f_out : 'float[:,:]', mat_out : 'float[:,:,:]'):

    for n in range(len(eta1s)):

        f(eta1s[n], eta2s[n], eta3s[n],
            kind_map, params_map,
            tn1, tn2, tn3, pn, 
            ind_n1, ind_n2, ind_n3, 
            cx, cy, cz,
            f_out[n, :])

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