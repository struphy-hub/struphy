# coding: utf-8
#
# Copyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)

"""Module containing accelerated (pyccelized) functions for evaluation of metric coefficients corresponding to 3d mappings x_i = F(eta_1, eta_2, eta_3):

- f      : mapping,                 f_i
- df     : Jacobian matrix,         df_i/deta_j
- det_df : Jacobian determinant,    det(df)
- df_inv : inverse Jacobian matrix, (df_i/deta_j)^(-1)
- g      : metric tensor,           df^T * df 
- g_inv  : inverse metric tensor,   df^(-1) * df^(-T)

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

from numpy import shape
from numpy import sin, cos, pi, empty, sqrt, arctan2, arcsin


import struphy.feec.basics.spline_evaluation_2d as eva_2d
import struphy.feec.basics.spline_evaluation_3d as eva_3d


# =======================================================================
def f(eta1 : 'float', eta2 : 'float', eta3 : 'float', component : 'int', kind_map : 'int', params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> 'float':
    """Point-wise evaluation of Cartesian coordinate x_i = f_i(eta1, eta2, eta3), i=1,2,3. 

    Parameters:
    -----------
        eta1, eta2, eta3:       float              logical coordinates in [0, 1]
        component:              int                 Cartesian coordinate (1: x, 2: y, 3: z)
        kind_map:               int                 kind of mapping (see module docstring)
        params_map:             float[:]           parameters for the mapping
        tn1, tn2, tn3:          float[:]           knot vectors for mapping
        pn:                     int[:]              spline degrees for mapping
        nbase_n:                int[:]              dimensions of univariate spline spaces for mapping 
        cx, cy, cz:             float[:, :, :]     control points of (f_1, f_2, f_3)

    Returns:
    --------
        value:  float
            Cartesian coordinate x_i = f_i(eta1, eta2, eta3)
    """

    value = 0.

    # =========== 3d spline ========================
    if kind_map == 0:

        if   component == 1:
            value = eva_3d.evaluate_n_n_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cx, eta1, eta2, eta3)

        elif component == 2:
            value = eva_3d.evaluate_n_n_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cy, eta1, eta2, eta3)

        elif component == 3:
            value = eva_3d.evaluate_n_n_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cz, eta1, eta2, eta3)

    # ==== 2d spline (straight in 3rd direction) ===
    elif kind_map == 1:

        Lz = 2*pi*cx[0, 0, 0]

        if   component == 1:
            value = eva_2d.evaluate_n_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx[:, :, 0], eta1, eta2)

            if eta1 == 0. and cx[0, 0, 0] == cx[0, 1, 0]:
                value = cx[0, 0, 0]

        elif component == 2:
            value = eva_2d.evaluate_n_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cy[:, :, 0], eta1, eta2)

            if eta1 == 0. and cy[0, 0, 0] == cy[0, 1, 0]:
                value = cy[0, 0, 0]

        elif component == 3:
            value = Lz * eta3

    # ==== 2d spline (curvature in 3rd direction) ===
    elif kind_map == 2:

        if   component == 1:
            value = eva_2d.evaluate_n_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx[:, :, 0], eta1, eta2) * cos(2*pi*eta3)

            if eta1 == 0. and cx[0, 0, 0] == cx[0, 1, 0]:
                value = cx[0, 0, 0]*cos(2*pi*eta3)

        elif component == 2:
            value = eva_2d.evaluate_n_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cy[:, :, 0], eta1, eta2)

            if eta1 == 0. and cy[0, 0, 0] == cy[0, 1, 0]:
                value = cy[0, 0, 0]

        elif component == 3:
            value = eva_2d.evaluate_n_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx[:, :, 0], eta1, eta2) * sin(2*pi*eta3)

            if eta1 == 0. and cx[0, 0, 0] == cx[0, 1, 0]:
                value = cx[0, 0, 0]*sin(2*pi*eta3)

    # ============== cuboid =========================
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

    # ========= hollow cylinder =====================
    elif kind_map == 11:

        a1 = params_map[0]
        a2 = params_map[1]
        r0 = params_map[2]

        da = a2 - a1

        if   component == 1:
            value = (a1 + eta1 * da) * cos(2*pi*eta2) + r0
        elif component == 2:
            value = (a1 + eta1 * da) * sin(2*pi*eta2)
        elif component == 3:
            value = 2*pi*r0 * eta3

    # ============ colella ==========================
    elif kind_map == 12:

        Lx    = params_map[0]
        Ly    = params_map[1]
        alpha = params_map[2]
        Lz    = params_map[3]

        if   component == 1:
            value = Lx * (eta1 + alpha * sin(2*pi*eta1) * sin(2*pi*eta2))
        elif component == 2:
            value = Ly * (eta2 + alpha * sin(2*pi*eta1) * sin(2*pi*eta2))
        elif component == 3:
            value = Lz * eta3

    # =========== orthogonal ========================
    elif kind_map == 13:

        Lx    = params_map[0]
        Ly    = params_map[1]
        alpha = params_map[2]
        Lz    = params_map[3]

        if   component == 1:
            value = Lx * (eta1 + alpha * sin(2*pi*eta1))
        elif component == 2:
            value = Ly * (eta2 + alpha * sin(2*pi*eta2))
        elif component == 3:
            value = Lz * eta3

    # ========= hollow torus ========================
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

    # ========= ellipse =====================
    elif kind_map == 15:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        Lz = params_map[5]

        if   component == 1:
            value = x0 + (eta1 * rx) * cos(2*pi*eta2)
        elif component == 2:
            value = y0 + (eta1 * ry) * sin(2*pi*eta2)
        elif component == 3:
            value = z0 + (eta3 * Lz)

    # ========= rotated ellipse =====================
    elif kind_map == 16:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        r1 = params_map[3]
        r2 = params_map[4]
        Lz = params_map[5]
        th = params_map[6] # Domain: [0,1)

        if   component == 1:
            value = x0 + (eta1 * r1) * cos(2*pi*th) * cos(2*pi*eta2) - (eta1 * r2) * sin(2*pi*th) * sin(2*pi*eta2)
        elif component == 2:
            value = y0 + (eta1 * r1) * sin(2*pi*th) * cos(2*pi*eta2) + (eta1 * r2) * cos(2*pi*th) * sin(2*pi*eta2)
        elif component == 3:
            value = z0 + (eta3 * Lz)

    # ========= shafranov shift =====================
    elif kind_map == 17:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        Lz = params_map[5]
        de = params_map[6] # Domain: [0,0.1]

        if   component == 1:
            value = x0 + (eta1 * rx) * cos(2*pi*eta2) + (1-eta1**2) * rx * de
        elif component == 2:
            value = y0 + (eta1 * ry) * sin(2*pi*eta2)
        elif component == 3:
            value = z0 + (eta3 * Lz)

    # ========= shafranov sqrt =====================
    elif kind_map == 18:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        Lz = params_map[5]
        de = params_map[6] # Domain: [0,0.1]

        if   component == 1:
            value = x0 + (eta1 * rx) * cos(2*pi*eta2) + (1-sqrt(eta1)) * rx * de
        elif component == 2:
            value = y0 + (eta1 * ry) * sin(2*pi*eta2)
        elif component == 3:
            value = z0 + (eta3 * Lz)

    # ========= shafranov D-shaped =====================
    elif kind_map == 19:

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
            value = x0 + r0 * (1 + (1 - eta1**2) * dx + eg *      eta1 * cos(2*pi*eta2 + arcsin(dg)*eta1*sin(2*pi*eta2)))
        elif component == 2:
            value = y0 + r0 * (    (1 - eta1**2) * dy + eg * kg * eta1 * sin(2*pi*eta2))
        elif component == 3:
            value = z0 + (eta3 * Lz)

    return value


# =======================================================================
def f_vec(eta1 : 'float', eta2 : 'float', eta3 : 'float', kind_map : 'int', params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:]', ind_n2 : 'int[:]', ind_n3 : 'int[:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', vec_out : 'float[:]'):
    """
    Point-wise evaluation of Cartesian coordinate x_i = f_i(eta1, eta2, eta3), i=1,2,3. 

    Parameters:
    -----------
        eta1, eta2, eta3:           float              logical coordinates in [0, 1]
        kind_map:                   int                 kind of mapping (see module docstring)
        params_map:                 float[:]           parameters for the mapping
        tn1, tn2, tn3:              float[:]           knot vectors for mapping
        pn:                         int[:]              spline degrees for mapping
        ind_n1, ind_n2, ind_n3      int[:]              contains the global indices of non-vanishing B-splines
        cx, cy, cz:                 float[:, :, :]     control points of (f_1, f_2, f_3)
        vec_out:                    float[:]           Mapping vector will be written here

    Returns:
    --------
        value:  float
            Cartesian coordinate x_i = f_i(eta1, eta2, eta3)
    """

    # make sure that the output vector is empty
    vec_out[:] = 0.

    # =========== 3d spline ========================
    if kind_map == 0:

        vec_out[0] = eva_3d.eval_n_n_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], ind_n1, ind_n2, ind_n3, cx, eta1, eta2, eta3)
        vec_out[1] = eva_3d.eval_n_n_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], ind_n1, ind_n2, ind_n3, cy, eta1, eta2, eta3)
        vec_out[2] = eva_3d.eval_n_n_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], ind_n1, ind_n2, ind_n3, cz, eta1, eta2, eta3)

    # ==== 2d spline (straight in 3rd direction) ===
    elif kind_map == 1:

        Lz = 2*pi*cx[0, 0, 0]

        if eta1 == 0. and cx[0, 0, 0] == cx[0, 1, 0]:
            vec_out[0] = cx[0, 0, 0]
        else:
            vec_out[0] = eva_2d.eval_n_n(tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cx[:, :, 0], eta1, eta2)


        if eta1 == 0. and cy[0, 0, 0] == cy[0, 1, 0]:
            vec_out[1] = cy[0, 0, 0]
        else:
            vec_out[1] = eva_2d.eval_n_n(tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cy[:, :, 0], eta1, eta2)

        vec_out[2] = Lz * eta3

    # ==== 2d spline (curvature in 3rd direction) ===
    elif kind_map == 2:


        if eta1 == 0. and cx[0, 0, 0] == cx[0, 1, 0]:
            vec_out[0] = cx[0, 0, 0]*cos(2*pi*eta3)
        else:
            vec_out[0] = eva_2d.eval_n_n(tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cx[:, :, 0], eta1, eta2) * cos(2*pi*eta3)


        if eta1 == 0. and cy[0, 0, 0] == cy[0, 1, 0]:
            vec_out[1] = cy[0, 0, 0]
        else:
            vec_out[1] = eva_2d.eval_n_n(tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cy[:, :, 0], eta1, eta2)

        if eta1 == 0. and cx[0, 0, 0] == cx[0, 1, 0]:
            vec_out[2] = cx[0, 0, 0]*sin(2*pi*eta3)
        else:
            vec_out[2] = eva_2d.eval_n_n(tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cx[:, :, 0], eta1, eta2) * sin(2*pi*eta3)

    # ============== cuboid =========================
    elif kind_map == 10:

        b1 = params_map[0]
        e1 = params_map[1]
        b2 = params_map[2]
        e2 = params_map[3]
        b3 = params_map[4]
        e3 = params_map[5]

        # value =  begin + (end - begin) * eta
        vec_out[0] = b1 + (e1 - b1) * eta1
        vec_out[1] = b2 + (e2 - b2) * eta2
        vec_out[2] = b3 + (e3 - b3) * eta3

    # ========= hollow cylinder =====================
    elif kind_map == 11:

        a1 = params_map[0]
        a2 = params_map[1]
        r0 = params_map[2]

        da = a2 - a1

        vec_out[0] = (a1 + eta1 * da) * cos(2*pi*eta2) + r0
        vec_out[1] = (a1 + eta1 * da) * sin(2*pi*eta2)
        vec_out[2] = 2*pi*r0 * eta3

    # ============ colella ==========================
    elif kind_map == 12:

        Lx    = params_map[0]
        Ly    = params_map[1]
        alpha = params_map[2]
        Lz    = params_map[3]

        vec_out[0] = Lx * (eta1 + alpha * sin(2*pi*eta1) * sin(2*pi*eta2))
        vec_out[1] = Ly * (eta2 + alpha * sin(2*pi*eta1) * sin(2*pi*eta2))
        vec_out[2] = Lz * eta3

    # =========== orthogonal ========================
    elif kind_map == 13:

        Lx    = params_map[0]
        Ly    = params_map[1]
        alpha = params_map[2]
        Lz    = params_map[3]

        vec_out[0] = Lx * (eta1 + alpha * sin(2*pi*eta1))
        vec_out[1] = Ly * (eta2 + alpha * sin(2*pi*eta2))
        vec_out[2] = Lz * eta3

    # ========= hollow torus ========================
    elif kind_map == 14:

        a1 = params_map[0]
        a2 = params_map[1]
        r0 = params_map[2]

        da = a2 - a1

        vec_out[0] = ((a1 + eta1 * da) * cos(2*pi*eta2) + r0) * cos(2*pi*eta3)
        vec_out[1] =  (a1 + eta1 * da) * sin(2*pi*eta2)
        vec_out[2] = ((a1 + eta1 * da) * cos(2*pi*eta2) + r0) * sin(2*pi*eta3)

    # ========= ellipse =====================
    elif kind_map == 15:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        Lz = params_map[5]

        vec_out[0] = x0 + (eta1 * rx) * cos(2*pi*eta2)
        vec_out[1] = y0 + (eta1 * ry) * sin(2*pi*eta2)
        vec_out[2] = z0 + (eta3 * Lz)

    # ========= rotated ellipse =====================
    elif kind_map == 16:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        r1 = params_map[3]
        r2 = params_map[4]
        Lz = params_map[5]
        th = params_map[6] # Domain: [0,1)

        vec_out[0] = x0 + (eta1 * r1) * cos(2*pi*th) * cos(2*pi*eta2) - (eta1 * r2) * sin(2*pi*th) * sin(2*pi*eta2)
        vec_out[1] = y0 + (eta1 * r1) * sin(2*pi*th) * cos(2*pi*eta2) + (eta1 * r2) * cos(2*pi*th) * sin(2*pi*eta2)
        vec_out[2] = z0 + (eta3 * Lz)

    # ========= soloviev approx =====================
    elif kind_map == 17:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        Lz = params_map[5]
        de = params_map[6] # Domain: [0,0.1]

        vec_out[0] = x0 + (eta1 * rx) * cos(2*pi*eta2) + (1-eta1**2) * rx * de
        vec_out[1] = y0 + (eta1 * ry) * sin(2*pi*eta2)
        vec_out[2] = z0 + (eta3 * Lz)

    # ========= soloviev sqrt =====================
    elif kind_map == 18:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        Lz = params_map[5]
        de = params_map[6] # Domain: [0,0.1]

        vec_out[0] = x0 + (eta1 * rx) * cos(2*pi*eta2) + (1-sqrt(eta1)) * rx * de
        vec_out[1] = y0 + (eta1 * ry) * sin(2*pi*eta2)
        vec_out[2] = z0 + (eta3 * Lz)

    # ========= soloviev cf =====================
    elif kind_map == 19:

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

        vec_out[0] = x0 + r0 * (1 + (1 - eta1**2) * dx + eg *      eta1 * cos(2*pi*eta2 + arcsin(dg)*eta1*sin(2*pi*eta2)))
        vec_out[1] = y0 + r0 * (    (1 - eta1**2) * dy + eg * kg * eta1 * sin(2*pi*eta2))
        vec_out[2] = z0 + (eta3 * Lz)


# =======================================================================
def f_vec_ana(eta1 : 'float', eta2 : 'float', eta3 : 'float', kind_map : 'int', params_map : 'float[:]', vec_out : 'float[:]'):
    """
    Point-wise evaluation of Cartesian coordinate x_i = f_i(eta1, eta2, eta3), i=1,2,3; only for analytical mappings. 

    Parameters:
    -----------
        eta1, eta2, eta3:           float              logical coordinates in [0, 1]
        kind_map:                   int                 kind of mapping (see module docstring)
        params_map:                 float[:]           parameters for the mapping
        vec_out:                    float[:]           Mapping vector will be written here

    Returns:
    --------
        value:  float
            Cartesian coordinate x_i = f_i(eta1, eta2, eta3)
    """

    # make sure that the output vector is empty
    vec_out[:] = 0.

    # ============== cuboid =========================
    if kind_map == 10:

        b1 = params_map[0]
        e1 = params_map[1]
        b2 = params_map[2]
        e2 = params_map[3]
        b3 = params_map[4]
        e3 = params_map[5]

        # value =  begin + (end - begin) * eta
        vec_out[0] = b1 + (e1 - b1) * eta1
        vec_out[1] = b2 + (e2 - b2) * eta2
        vec_out[2] = b3 + (e3 - b3) * eta3

    # ========= hollow cylinder =====================
    elif kind_map == 11:

        a1 = params_map[0]
        a2 = params_map[1]
        r0 = params_map[2]

        da = a2 - a1

        vec_out[0] = (a1 + eta1 * da) * cos(2*pi*eta2) + r0
        vec_out[1] = (a1 + eta1 * da) * sin(2*pi*eta2)
        vec_out[2] = 2*pi*r0 * eta3

    # ============ colella ==========================
    elif kind_map == 12:

        Lx    = params_map[0]
        Ly    = params_map[1]
        alpha = params_map[2]
        Lz    = params_map[3]

        vec_out[0] = Lx * (eta1 + alpha * sin(2*pi*eta1) * sin(2*pi*eta2))
        vec_out[1] = Ly * (eta2 + alpha * sin(2*pi*eta1) * sin(2*pi*eta2))
        vec_out[2] = Lz * eta3

    # =========== orthogonal ========================
    elif kind_map == 13:

        Lx    = params_map[0]
        Ly    = params_map[1]
        alpha = params_map[2]
        Lz    = params_map[3]

        vec_out[0] = Lx * (eta1 + alpha * sin(2*pi*eta1))
        vec_out[1] = Ly * (eta2 + alpha * sin(2*pi*eta2))
        vec_out[2] = Lz * eta3

    # ========= hollow torus ========================
    elif kind_map == 14:

        a1 = params_map[0]
        a2 = params_map[1]
        r0 = params_map[2]

        da = a2 - a1

        vec_out[0] = ((a1 + eta1 * da) * cos(2*pi*eta2) + r0) * cos(2*pi*eta3)
        vec_out[1] =  (a1 + eta1 * da) * sin(2*pi*eta2)
        vec_out[2] = ((a1 + eta1 * da) * cos(2*pi*eta2) + r0) * sin(2*pi*eta3)

    # ========= ellipse =====================
    elif kind_map == 15:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        Lz = params_map[5]

        vec_out[0] = x0 + (eta1 * rx) * cos(2*pi*eta2)
        vec_out[1] = y0 + (eta1 * ry) * sin(2*pi*eta2)
        vec_out[2] = z0 + (eta3 * Lz)

    # ========= rotated ellipse =====================
    elif kind_map == 16:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        r1 = params_map[3]
        r2 = params_map[4]
        Lz = params_map[5]
        th = params_map[6] # Domain: [0,1)

        vec_out[0] = x0 + (eta1 * r1) * cos(2*pi*th) * cos(2*pi*eta2) - (eta1 * r2) * sin(2*pi*th) * sin(2*pi*eta2)
        vec_out[1] = y0 + (eta1 * r1) * sin(2*pi*th) * cos(2*pi*eta2) + (eta1 * r2) * cos(2*pi*th) * sin(2*pi*eta2)
        vec_out[2] = z0 + (eta3 * Lz)

    # ========= soloviev approx =====================
    elif kind_map == 17:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        Lz = params_map[5]
        de = params_map[6] # Domain: [0,0.1]

        vec_out[0] = x0 + (eta1 * rx) * cos(2*pi*eta2) + (1-eta1**2) * rx * de
        vec_out[1] = y0 + (eta1 * ry) * sin(2*pi*eta2)
        vec_out[2] = z0 + (eta3 * Lz)

    # ========= soloviev sqrt =====================
    elif kind_map == 18:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        Lz = params_map[5]
        de = params_map[6] # Domain: [0,0.1]

        vec_out[0] = x0 + (eta1 * rx) * cos(2*pi*eta2) + (1-sqrt(eta1)) * rx * de
        vec_out[1] = y0 + (eta1 * ry) * sin(2*pi*eta2)
        vec_out[2] = z0 + (eta3 * Lz)

    # ========= soloviev cf =====================
    elif kind_map == 19:

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

        vec_out[0] = x0 + r0 * (1 + (1 - eta1**2) * dx + eg *      eta1 * cos(2*pi*eta2 + arcsin(dg)*eta1*sin(2*pi*eta2)))
        vec_out[1] = y0 + r0 * (    (1 - eta1**2) * dy + eg * kg * eta1 * sin(2*pi*eta2))
        vec_out[2] = z0 + (eta3 * Lz)


# =======================================================================
def f_inv(x : 'float', y : 'float', z : 'float', component : 'int', kind_map : 'int', params_map : 'float[:]') -> 'float':
    """
    Point-wise evaluation of inverse mapping eta_i = f^(-1)_i(x, y, z), i=1,2,3. Only possible for analytical mappings.
    
    Parameters:
    -----------
        x, y, z:       float        Cartesian coordinates
        component:     int           Logical coordinate (1: eta1, 2: eta2, 3: eta3)
        kind_map:      int           kind of mapping (see module docstring)
        params_map:    float[:]     parameters for the mapping

    Returns:
    --------
        value:  float
            Logical coordinate eta_i = f^(-1)_i(eta1, eta2, eta3) in [0, 1]
    """

    value = 0.

    # ============== cuboid =========================
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

    # ========= hollow cylinder =====================
    elif kind_map == 11:

        a1 = params_map[0]
        a2 = params_map[1]
        r0 = params_map[2]

        da = a2 - a1

        if   component == 1:
            value = (sqrt((x - r0)**2 + y**2) - a1)/da
        elif component == 2:
            value = (arctan2(y, x - r0)/(2*pi))%1.0
        elif component == 3:
            value = z/(2*pi*r0)

    # ========= hollow torus ========================
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

    # ========= ellipse =====================
    elif kind_map == 15:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        Lz = params_map[5]

        if   component == 1:
            eta2  = (arctan2((y - y0) * rx, (x - x0) * ry) / (2 * pi)) % 1.0
            # value = sqrt(((x - x0)**2 + (y - y0)**2) / (rx**2 * cos(2*pi*eta2)**2 + ry**2 * sin(2*pi*eta2)**2))
            value = (x - x0) / (rx * cos(2*pi*eta2)) # Equivalent.
        elif component == 2:
            value = (arctan2((y - y0) * rx, (x - x0) * ry) / (2 * pi)) % 1.0
        elif component == 3:
            value = (z - z0) / Lz

    # ========= rotated ellipse =====================
    elif kind_map == 16:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        r1 = params_map[3]
        r2 = params_map[4]
        Lz = params_map[5]
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
            value = (z - z0) / Lz

    # ========= shafranov shift =====================
    elif kind_map == 17:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        Lz = params_map[5]
        de = params_map[6] # Domain: [0,0.1]

        if   component == 1:
            # eta1**2 * (rx * de) + eta1 * (-rx * cos(2*pi*eta2)) + (x - x0) - (rx * de) = 0
            # eta1 = (rx * cos(2*pi*eta2) \pm sqrt(rx**2 * cos(2*pi*eta2)**2 - 4 * (rx * de) * ((x - x0) - (rx * de)))) / (2 * rx * de)
            #      = (cos(2*pi*eta2) \pm sqrt(cos(2*pi*eta2)**2 - 4 * de * ((x - x0) / rx - de))) / (2 * de)
            # eta1 = (y - y0) / (ry * sin(2*pi*eta2))
            value = 0. # TODO: Not implemented. Multiple solutions.
        elif component == 2:
            value = 0. # TODO: Not implemented. Multiple solutions.
        elif component == 3:
            value = (z - z0) / Lz

    # ========= shafranov sqrt =====================
    elif kind_map == 18:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        Lz = params_map[5]
        de = params_map[6] # Domain: [0,0.1]

        # Not implemented.

    # ========= shafranov D-shaped =====================
    elif kind_map == 19:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        r0 = params_map[3]
        Lz = params_map[4]
        dx = params_map[5] # Grad-Shafranov shift along x-axis.
        dy = params_map[6] # Grad-Shafranov shift along y-axis.
        dg = params_map[7] # Delta = sin(alpha): Triangularity, shift of high point.
        eg = params_map[8] # Epsilon: Inverse aspect ratio a/R0.
        kg = params_map[9] # Kappa: Ellipticity (elongation).

        # Not implemented.

    return value


# =======================================================================
def f_inv_vec(x : 'float', y : 'float', z : 'float', kind_map : 'int', params_map : 'float[:]', vec_out : 'float[:]'):
    """
    Point-wise evaluation of inverse mapping eta_i = f^(-1)_i(x, y, z), i=1,2,3. Only possible for analytical mappings.
    
    Parameters:
    -----------
        x, y, z:        float          Cartesian coordinates
        kind_map:       int             kind of mapping (see module docstring)
        params_map:     float[:]       parameters for the mapping
        vec_out:        float[:]       vector in which the result will be written
    """

    # make sure that the vector is empty
    vec_out[:] = 0.

    # ============== cuboid =========================
    if kind_map == 10:

        b1 = params_map[0]
        e1 = params_map[1]
        b2 = params_map[2]
        e2 = params_map[3]
        b3 = params_map[4]
        e3 = params_map[5]

        # value =  begin + (end - begin) * eta
        vec_out[0] = (x - b1)/(e1 - b1)
        vec_out[1] = (y - b2)/(e2 - b2)
        vec_out[2] = (z - b3)/(e3 - b3)

    # ========= hollow cylinder =====================
    elif kind_map == 11:

        a1 = params_map[0]
        a2 = params_map[1]
        r0 = params_map[2]

        da = a2 - a1

        vec_out[0] = (sqrt((x - r0)**2 + y**2) - a1)/da
        vec_out[1] = (arctan2(y, x - r0)/(2*pi))%1.0
        vec_out[2] = z/(2*pi*r0)

    # ========= hollow torus ========================
    elif kind_map == 14:

        a1 = params_map[0]
        a2 = params_map[1]
        r0 = params_map[2]

        da = a2 - a1

        vec_out[0] = (sqrt((sqrt(x**2 + z**2) - r0)**2 + y**2) - a1)/da
        vec_out[1] = (arctan2(y, sqrt(x**2 + z**2) - r0)/(2*pi))%1.0
        vec_out[2] = (arctan2(z, x)/(2*pi))%1.0

    # ========= ellipse =====================
    elif kind_map == 15:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        Lz = params_map[5]

        # vec_out[0] = sqrt(((x - x0)**2 + (y - y0)**2) / (rx**2 * cos(2*pi*eta2)**2 + ry**2 * sin(2*pi*eta2)**2))
        eta2  = (arctan2((y - y0) * rx, (x - x0) * ry) / (2 * pi)) % 1.0
        vec_out[0] = (x - x0) / (rx * cos(2*pi*eta2)) # Equivalent.
        vec_out[1] = (arctan2((y - y0) * rx, (x - x0) * ry) / (2 * pi)) % 1.0
        vec_out[2] = (z - z0) / Lz

    # ========= rotated ellipse =====================
    elif kind_map == 16:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        r1 = params_map[3]
        r2 = params_map[4]
        Lz = params_map[5]
        th = params_map[6] # Domain: [0,1)

        temp1 =  cos(2*pi*th) * (x - x0) + sin(2*pi*th) * (y - y0)
        temp2 = -sin(2*pi*th) * (x - x0) + cos(2*pi*th) * (y - y0)
        eta2  = (arctan2(temp2 * r1, temp1 * r2) / (2 * pi)) % 1.0
        vec_out[0] = temp1 / (r1 * cos(2*pi*eta2))

        temp1 =  cos(2*pi*th) * (x - x0) + sin(2*pi*th) * (y - y0)
        temp2 = -sin(2*pi*th) * (x - x0) + cos(2*pi*th) * (y - y0)
        vec_out[1] = (arctan2(temp2 * r1, temp1 * r2) / (2 * pi)) % 1.0

        vec_out[2] = (z - z0) / Lz

    # ========= soloviev approx =====================
    elif kind_map == 17:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        Lz = params_map[5]
        de = params_map[6] # Domain: [0,0.1]

        # eta1**2 * (rx * de) + eta1 * (-rx * cos(2*pi*eta2)) + (x - x0) - (rx * de) = 0
        # eta1 = (rx * cos(2*pi*eta2) \pm sqrt(rx**2 * cos(2*pi*eta2)**2 - 4 * (rx * de) * ((x - x0) - (rx * de)))) / (2 * rx * de)
        #      = (cos(2*pi*eta2) \pm sqrt(cos(2*pi*eta2)**2 - 4 * de * ((x - x0) / rx - de))) / (2 * de)
        # eta1 = (y - y0) / (ry * sin(2*pi*eta2))
        vec_out[0] = 0. # TODO: Not implemented. Multiple solutions.
        vec_out[1] = 0. # TODO: Not implemented. Multiple solutions.
        vec_out[2] = (z - z0) / Lz

    # ========= soloviev sqrt =====================
    elif kind_map == 18:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        Lz = params_map[5]
        de = params_map[6] # Domain: [0,0.1]

        # Not implemented.

    # ========= soloviev cf =====================
    elif kind_map == 19:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        r0 = params_map[3]
        Lz = params_map[4]
        dx = params_map[5] # Grad-Shafranov shift along x-axis.
        dy = params_map[6] # Grad-Shafranov shift along y-axis.
        dg = params_map[7] # Delta = sin(alpha): Triangularity, shift of high point.
        eg = params_map[8] # Epsilon: Inverse aspect ratio a/R0.
        kg = params_map[9] # Kappa: Ellipticity (elongation).

        # Not implemented.

    ierr = 0.


# =======================================================================
def df(eta1 : 'float', eta2 : 'float', eta3 : 'float', component : 'int', kind_map : 'int', params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> 'float':
    """Point-wise evaluation of ij-th component of the Jacobian matrix df_ij = df_i/deta_j (i,j=1,2,3). 

    Parameters:
    -----------
        eta1, eta2, eta3:       float              logical coordinates in [0, 1]
        component:              int                 11 : (df1/deta1), 12 : (df1/deta2), 13 : (df1/deta3)
                                                    21 : (df2/deta1), 22 : (df2/deta2), 23 : (df2/deta3)
                                                    31 : (df3/deta1), 32 : (df3/deta2), 33 : (df3/deta3)
        kind_map:               int                 kind of mapping (see module docstring)
        params_map:             float[:]           parameters for the mapping
        tn1, tn2, tn3:          float[:]           knot vectors for mapping
        pn:                     int[:]              spline degrees for mapping
        nbase_n:                int[:]              dimensions of univariate spline spaces for mapping 
        cx, cy, cz:             float[:, :, :]     control points of (f_1, f_2, f_3)

    Returns:
    --------
        value:  float
            point value df_ij(eta1, eta2, eta3)
    """

    value = 0.

    # =========== 3d spline ========================
    if kind_map == 0:
        
        if   component == 11:
            value = eva_3d.evaluate_diffn_n_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cx, eta1, eta2, eta3)
        elif component == 12:
            value = eva_3d.evaluate_n_diffn_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cx, eta1, eta2, eta3)
        elif component == 13:
            value = eva_3d.evaluate_n_n_diffn(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cx, eta1, eta2, eta3)
        elif component == 21:
            value = eva_3d.evaluate_diffn_n_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cy, eta1, eta2, eta3)
        elif component == 22:
            value = eva_3d.evaluate_n_diffn_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cy, eta1, eta2, eta3)
        elif component == 23:
            value = eva_3d.evaluate_n_n_diffn(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cy, eta1, eta2, eta3)
        elif component == 31:
            value = eva_3d.evaluate_diffn_n_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cz, eta1, eta2, eta3)
        elif component == 32:
            value = eva_3d.evaluate_n_diffn_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cz, eta1, eta2, eta3)
        elif component == 33:
            value = eva_3d.evaluate_n_n_diffn(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cz, eta1, eta2, eta3)
               
    # ==== 2d spline (straight in 3rd direction) ===
    elif kind_map == 1:
        
        Lz = 2*pi*cx[0, 0, 0]
        
        if   component == 11:
            value = eva_2d.evaluate_diffn_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx[:, :, 0], eta1, eta2)
        elif component == 12:
            value = eva_2d.evaluate_n_diffn(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx[:, :, 0], eta1, eta2)
            
            if eta1 == 0. and cx[0, 0, 0] == cx[0, 1, 0]:
                value = 0.
            
        elif component == 13:
            value = 0.
        elif component == 21:
            value = eva_2d.evaluate_diffn_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cy[:, :, 0], eta1, eta2)
        elif component == 22:
            value = eva_2d.evaluate_n_diffn(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cy[:, :, 0], eta1, eta2)
            
            if eta1 == 0. and cy[0, 0, 0] == cy[0, 1, 0]:
                value = 0.
            
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.
        elif component == 33:
            value = Lz
    
    # ==== 2d spline (curvature in 3rd direction) ===
    elif kind_map == 2:
        
        if   component == 11:
            value = eva_2d.evaluate_diffn_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx[:, :, 0], eta1, eta2) * cos(2*pi*eta3)
        elif component == 12:
            value = eva_2d.evaluate_n_diffn(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx[:, :, 0], eta1, eta2) * cos(2*pi*eta3)
            
            if eta1 == 0. and cx[0, 0, 0] == cx[0, 1, 0]:
                value = 0.
            
        elif component == 13:
            value = eva_2d.evaluate_n_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx[:, :, 0], eta1, eta2) * sin(2*pi*eta3) * (-2*pi)
        elif component == 21:
            value = eva_2d.evaluate_diffn_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cy[:, :, 0], eta1, eta2)
        elif component == 22:
            value = eva_2d.evaluate_n_diffn(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cy[:, :, 0], eta1, eta2)
            
            if eta1 == 0. and cy[0, 0, 0] == cy[0, 1, 0]:
                value = 0.
            
        elif component == 23:
            value = 0.
        elif component == 31:
            value = eva_2d.evaluate_diffn_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx[:, :, 0], eta1, eta2) * sin(2*pi*eta3)
        elif component == 32:
            value = eva_2d.evaluate_n_diffn(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx[:, :, 0], eta1, eta2) * sin(2*pi*eta3)
            
            if eta1 == 0. and cx[0, 0, 0] == cx[0, 1, 0]:
                value = 0.
            
        elif component == 33:
            value = eva_2d.evaluate_n_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx[:, :, 0], eta1, eta2) * cos(2*pi*eta3) * 2*pi
    
    # ============== cuboid ===================
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
            
    # ======== hollow cylinder =================
    elif kind_map == 11:

        a1 = params_map[0]
        a2 = params_map[1]
        r0 = params_map[2]

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
            value = 2*pi*r0

    # ============ colella =================
    elif kind_map == 12:

        Lx    = params_map[0]
        Ly    = params_map[1]
        alpha = params_map[2]
        Lz    = params_map[3]

        if   component == 11:
            value = Lx * (1 + alpha * cos(2*pi*eta1) * sin(2*pi*eta2) * 2*pi)
        elif component == 12:
            value = Lx * alpha * sin(2*pi*eta1) * cos(2*pi*eta2) * 2*pi
        elif component == 13:
            value = 0.
        elif component == 21:
            value = Ly * alpha * cos(2*pi*eta1) * sin(2*pi*eta2) * 2*pi
        elif component == 22:
            value = Ly * (1 + alpha * sin(2*pi*eta1) * cos(2*pi*eta2) * 2*pi)
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.    
        elif component == 33:
            value = Lz

    # =========== orthogonal ================
    elif kind_map == 13:

        Lx    = params_map[0]
        Ly    = params_map[1]
        alpha = params_map[2]
        Lz    = params_map[3]

        if   component == 11:
            value = Lx * (1 + alpha * cos(2*pi*eta1) * 2*pi)
        elif component == 12:
            value = 0.
        elif component == 13:
            value = 0.
        elif component == 21:
            value = 0.
        elif component == 22:
            value = Ly * (1 + alpha * cos(2*pi*eta2) * 2*pi)
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.    
        elif component == 33:
            value = Lz

    # ========= hollow torus ==================
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

    # ========= ellipse =====================
    elif kind_map == 15:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        Lz = params_map[5]

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
            value = Lz

    # ========= rotated ellipse =====================
    elif kind_map == 16:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        r1 = params_map[3]
        r2 = params_map[4]
        Lz = params_map[5]
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
            value = Lz

    # ========= shafranov shift =====================
    elif kind_map == 17:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        Lz = params_map[5]
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
            value = Lz

    # ========= shafranov sqrt =====================
    elif kind_map == 18:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        Lz = params_map[5]
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
            value = Lz

    # ========= shafranov D-shaped =====================
    elif kind_map == 19:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        r0 = params_map[3]
        Lz = params_map[4]
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
            value = Lz

    return value


# =======================================================================
def df_mat(eta1 : 'float', eta2 : 'float', eta3 : 'float', kind_map : 'int', params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:]', ind_n2 : 'int[:]', ind_n3 : 'int[:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', mat_out : 'float[:,:]'):
    """Point-wise evaluation of ij-th component of the Jacobian matrix df_ij = df_i/deta_j (i,j=1,2,3). 

    Parameters:
    -----------
        eta1, eta2, eta3:           float              logical coordinates in [0, 1]
        kind_map:                   int                 kind of mapping (see module docstring)
        params_map:                 float[:]           parameters for the mapping
        tn1, tn2, tn3:              float[:]           knot vectors for mapping
        pn:                         int[:]              spline degrees for mapping
        ind_n1, ind_n2, ind_n3      int[:]              contains the global indices of non-vanishing B-splines
        cx, cy, cz:                 float[:, :, :]     control points of (f_1, f_2, f_3)
        mat_out:                    float[:,:]         matrix in which the resulting Jacobian matrix will be written
    """

    # make sure that the matrix is empty
    mat_out[:,:] = 0.

    # =========== 3d spline ========================
    if kind_map == 0:
        
        mat_out[0,0] = eva_3d.eval_diffn_n_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], ind_n1, ind_n2, ind_n3, cx, eta1, eta2, eta3)
        mat_out[0,1] = eva_3d.eval_n_diffn_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], ind_n1, ind_n2, ind_n3, cx, eta1, eta2, eta3)
        mat_out[0,2] = eva_3d.eval_n_n_diffn(tn1, tn2, tn3, pn[0], pn[1], pn[2], ind_n1, ind_n2, ind_n3, cx, eta1, eta2, eta3)
        mat_out[1,0] = eva_3d.eval_diffn_n_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], ind_n1, ind_n2, ind_n3, cy, eta1, eta2, eta3)
        mat_out[1,1] = eva_3d.eval_n_diffn_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], ind_n1, ind_n2, ind_n3, cy, eta1, eta2, eta3)
        mat_out[1,2] = eva_3d.eval_n_n_diffn(tn1, tn2, tn3, pn[0], pn[1], pn[2], ind_n1, ind_n2, ind_n3, cy, eta1, eta2, eta3)
        mat_out[2,0] = eva_3d.eval_diffn_n_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], ind_n1, ind_n2, ind_n3, cz, eta1, eta2, eta3)
        mat_out[2,1] = eva_3d.eval_n_diffn_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], ind_n1, ind_n2, ind_n3, cz, eta1, eta2, eta3)
        mat_out[2,2] = eva_3d.eval_n_n_diffn(tn1, tn2, tn3, pn[0], pn[1], pn[2], ind_n1, ind_n2, ind_n3, cz, eta1, eta2, eta3)
               
    # ==== 2d spline (straight in 3rd direction) ===
    elif kind_map == 1:
        
        Lz = 2*pi*cx[0, 0, 0]
        
        mat_out[0,0] = eva_2d.eval_diffn_n(tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cx[:, :, 0], eta1, eta2)
            
        if eta1 == 0. and cx[0, 0, 0] == cx[0, 1, 0]:
            mat_out[0,1] = 0.
        else:
            mat_out[0,1] = eva_2d.eval_n_diffn(tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cx[:, :, 0], eta1, eta2)

        mat_out[0,2] = 0.
        mat_out[1,0] = eva_2d.eval_diffn_n(tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cy[:, :, 0], eta1, eta2)
        
        if eta1 == 0. and cy[0, 0, 0] == cy[0, 1, 0]:
            mat_out[1,1] = 0.
        else:
            mat_out[1,1]  = eva_2d.eval_n_diffn(tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cy[:, :, 0], eta1, eta2)
            
        mat_out[1,2] = 0.
        mat_out[2,0] = 0.
        mat_out[2,1] = 0.
        mat_out[2,2] = Lz
    
    # ==== 2d spline (curvature in 3rd direction) ===
    elif kind_map == 2:
        
        mat_out[0,0] = eva_2d.eval_diffn_n(tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cx[:, :, 0], eta1, eta2) * cos(2*pi*eta3)

        if eta1 == 0. and cx[0, 0, 0] == cx[0, 1, 0]:
            mat_out[0,1] = 0.
        else:
            mat_out[0,1] = eva_2d.eval_n_diffn(tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cx[:, :, 0], eta1, eta2) * cos(2*pi*eta3)
            
        mat_out[0,2] = eva_2d.eval_n_n(tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cx[:, :, 0], eta1, eta2) * sin(2*pi*eta3) * (-2*pi)
        mat_out[1,0] = eva_2d.eval_diffn_n(tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cy[:, :, 0], eta1, eta2)
            
        if eta1 == 0. and cy[0, 0, 0] == cy[0, 1, 0]:
            mat_out[1,1] = 0.
        else:
            mat_out[1,1] = eva_2d.eval_n_diffn(tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cy[:, :, 0], eta1, eta2)
            
        mat_out[1,2] = 0.
        mat_out[2,0] = eva_2d.eval_diffn_n(tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cx[:, :, 0], eta1, eta2) * sin(2*pi*eta3)

        if eta1 == 0. and cx[0, 0, 0] == cx[0, 1, 0]:
            mat_out[2,1] = 0.
        else:
            mat_out[2,1] = eva_2d.eval_n_diffn(tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cx[:, :, 0], eta1, eta2) * sin(2*pi*eta3)
            
        mat_out[2,2] = eva_2d.eval_n_n(tn1, tn2, pn[0], pn[1], ind_n1, ind_n2, cx[:, :, 0], eta1, eta2) * cos(2*pi*eta3) * 2*pi
    
    
    # ============== cuboid ===================
    if kind_map == 10:
         
        b1 = params_map[0]
        e1 = params_map[1]
        b2 = params_map[2]
        e2 = params_map[3]
        b3 = params_map[4]
        e3 = params_map[5]
        
        mat_out[0,0] = e1 - b1
        mat_out[0,1] = 0.
        mat_out[0,2] = 0.
        mat_out[1,0] = 0.
        mat_out[1,1] = e2 - b2
        mat_out[1,2] = 0.
        mat_out[2,0] = 0.
        mat_out[2,1] = 0.
        mat_out[2,2] = e3 - b3
            
    # ======== hollow cylinder =================
    elif kind_map == 11:

        a1 = params_map[0]
        a2 = params_map[1]
        r0 = params_map[2]

        da = a2 - a1

        mat_out[0,0] = da * cos(2*pi*eta2)
        mat_out[0,1] = -2*pi * (a1 + eta1 * da) * sin(2*pi*eta2)
        mat_out[0,2] = 0.
        mat_out[1,0] = da * sin(2*pi*eta2)
        mat_out[1,1] = 2*pi * (a1 + eta1 * da) * cos(2*pi*eta2)
        mat_out[1,2] = 0.
        mat_out[2,0] = 0.
        mat_out[2,1] = 0.
        mat_out[2,2] = 2*pi*r0

    # ============ colella =================
    elif kind_map == 12:

        Lx    = params_map[0]
        Ly    = params_map[1]
        alpha = params_map[2]
        Lz    = params_map[3]

        mat_out[0,0] = Lx * (1 + alpha * cos(2*pi*eta1) * sin(2*pi*eta2) * 2*pi)
        mat_out[0,1] = Lx * alpha * sin(2*pi*eta1) * cos(2*pi*eta2) * 2*pi
        mat_out[0,2] = 0.
        mat_out[1,0] = Ly * alpha * cos(2*pi*eta1) * sin(2*pi*eta2) * 2*pi
        mat_out[1,1] = Ly * (1 + alpha * sin(2*pi*eta1) * cos(2*pi*eta2) * 2*pi)
        mat_out[1,2] = 0.
        mat_out[2,0] = 0.
        mat_out[2,1] = 0.    
        mat_out[2,2] = Lz

    # =========== orthogonal ================
    elif kind_map == 13:

        Lx    = params_map[0]
        Ly    = params_map[1]
        alpha = params_map[2]
        Lz    = params_map[3]

        mat_out[0,0] = Lx * (1 + alpha * cos(2*pi*eta1) * 2*pi)
        mat_out[0,1] = 0.
        mat_out[0,2] = 0.
        mat_out[1,0] = 0.
        mat_out[1,1] = Ly * (1 + alpha * cos(2*pi*eta2) * 2*pi)
        mat_out[1,2] = 0.
        mat_out[2,0] = 0.
        mat_out[2,1] = 0.    
        mat_out[2,2] = Lz

    # ========= hollow torus ==================
    elif kind_map == 14:

        a1 = params_map[0]
        a2 = params_map[1]
        r0 = params_map[2]

        da = a2 - a1

        mat_out[0,0] = da * cos(2*pi*eta2) * cos(2*pi*eta3)
        mat_out[0,1] = -2*pi * (a1 + eta1 * da) * sin(2*pi*eta2) * cos(2*pi*eta3)
        mat_out[0,2] = -2*pi * ((a1 + eta1 * da) * cos(2*pi*eta2) + r0) * sin(2*pi*eta3)
        mat_out[1,0] = da * sin(2*pi*eta2)
        mat_out[1,1] = (a1 + eta1 * da) * cos(2*pi*eta2) * 2*pi
        mat_out[1,2] = 0.
        mat_out[2,0] = da * cos(2*pi*eta2) * sin(2*pi*eta3)
        mat_out[2,1] = -2*pi * (a1 + eta1 * da) * sin(2*pi*eta2) * sin(2*pi*eta3)
        mat_out[2,2] = ((a1 + eta1 * da) * cos(2*pi*eta2) + r0) * cos(2*pi*eta3) * 2*pi

    # ========= ellipse =====================
    elif kind_map == 15:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        Lz = params_map[5]

        mat_out[0,0] = rx * cos(2*pi*eta2)
        mat_out[0,1] = -2*pi * (eta1 * rx) * sin(2*pi*eta2)
        mat_out[0,2] = 0.
        mat_out[1,0] = ry * sin(2*pi*eta2)
        mat_out[1,1] =  2*pi * (eta1 * ry) * cos(2*pi*eta2)
        mat_out[1,2] = 0.
        mat_out[2,0] = 0.
        mat_out[2,1] = 0.
        mat_out[2,2] = Lz

    # ========= rotated ellipse =====================
    elif kind_map == 16:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        r1 = params_map[3]
        r2 = params_map[4]
        Lz = params_map[5]
        th = params_map[6] # Domain: [0,1)

        mat_out[0,0] = r1 * cos(2*pi*th) * cos(2*pi*eta2) - r2 * sin(2*pi*th) * sin(2*pi*eta2)
        mat_out[0,1] = -2*pi * (eta1 * r1) * cos(2*pi*th) * sin(2*pi*eta2) - 2*pi * (eta1 * r2) * sin(2*pi*th) * cos(2*pi*eta2)
        mat_out[0,2] = 0.
        mat_out[1,0] = r1 * sin(2*pi*th) * cos(2*pi*eta2) + r2 * cos(2*pi*th) * sin(2*pi*eta2)
        mat_out[1,1] = -2*pi * (eta1 * r1) * sin(2*pi*th) * sin(2*pi*eta2) + 2*pi * (eta1 * r2) * cos(2*pi*th) * cos(2*pi*eta2)
        mat_out[1,2] = 0.
        mat_out[2,0] = 0.
        mat_out[2,1] = 0.
        mat_out[2,2] = Lz

    # ========= soloviev approx =====================
    elif kind_map == 17:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        Lz = params_map[5]
        de = params_map[6] # Domain: [0,0.1]

        mat_out[0,0] = rx * cos(2*pi*eta2) - 2 * eta1 * rx * de
        mat_out[0,1] = -2*pi * (eta1 * rx) * sin(2*pi*eta2)
        mat_out[0,2] = 0.
        mat_out[1,0] = ry * sin(2*pi*eta2)
        mat_out[1,1] =  2*pi * (eta1 * ry) * cos(2*pi*eta2)
        mat_out[1,2] = 0.
        mat_out[2,0] = 0.
        mat_out[2,1] = 0.
        mat_out[2,2] = Lz

    # ========= soloviev sqrt =====================
    elif kind_map == 18:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        Lz = params_map[5]
        de = params_map[6] # Domain: [0,0.1]

        mat_out[0,0] = rx * cos(2*pi*eta2) - 0.5 / sqrt(eta1) * rx * de
        mat_out[0,1] = -2*pi * (eta1 * rx) * sin(2*pi*eta2)
        mat_out[0,2] = 0.
        mat_out[1,0] = ry * sin(2*pi*eta2)
        mat_out[1,1] =  2*pi * (eta1 * ry) * cos(2*pi*eta2)
        mat_out[1,2] = 0.
        mat_out[2,0] = 0.
        mat_out[2,1] = 0.
        mat_out[2,2] = Lz

    # ========= soloviev cf =====================
    elif kind_map == 19:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        r0 = params_map[3]
        Lz = params_map[4]
        dx = params_map[5] # Grad-Shafranov shift along x-axis.
        dy = params_map[6] # Grad-Shafranov shift along y-axis.
        dg = params_map[7] # Delta = sin(alpha): Triangularity, shift of high point.
        eg = params_map[8] # Epsilon: Inverse aspect ratio a/R0.
        kg = params_map[9] # Kappa: Ellipticity (elongation).

        mat_out[0,0] = r0 * (- 2 * dx * eta1 - eg * eta1 * sin(2*pi*eta2) * arcsin(dg) * sin(eta1 * sin(2*pi*eta2) * arcsin(dg) + 2*pi*eta2) + eg * cos(eta1 * sin(2*pi*eta2) * arcsin(dg) + 2*pi*eta2))
        mat_out[0,1] = - r0 * eg * eta1 * (2*pi*eta1 * cos(2*pi*eta2) * arcsin(dg) + 2*pi) * sin(eta1 * sin(2*pi*eta2) * arcsin(dg) + 2*pi*eta2)
        mat_out[0,2] = 0.
        mat_out[1,0] = r0 * (- 2 * dy * eta1 + eg * kg * sin(2*pi*eta2))
        mat_out[1,1] = 2 * pi * r0 * eg * eta1 * kg * cos(2*pi*eta2)
        mat_out[1,2] = 0.
        mat_out[2,0] = 0.
        mat_out[2,1] = 0.
        mat_out[2,2] = Lz
    
    # ========= invalid mapping =================
    else:
        print('Invalid mapping given !!')
        
        
# =======================================================================
def df_ana(eta1 : 'float', eta2 : 'float', eta3 : 'float', component : 'int', kind_map : 'int', params_map : 'float[:]') -> 'float':
    """
    Point-wise evaluation of ij-th component of the Jacobian matrix df_ij = df_i/deta_j (i,j=1,2,3). Only for analytical mappings, not spline mappings.

    Parameters:
    -----------
        eta1, eta2, eta3:       float              logical coordinates in [0, 1]
        component:              int                 11 : (df1/deta1), 12 : (df1/deta2), 13 : (df1/deta3)
                                                    21 : (df2/deta1), 22 : (df2/deta2), 23 : (df2/deta3)
                                                    31 : (df3/deta1), 32 : (df3/deta2), 33 : (df3/deta3)
        kind_map:               int                 kind of mapping (see module docstring)
        params_map:             float[:]           parameters for the mapping

    Returns:
    --------
        value:  float
            point value df_ij(eta1, eta2, eta3)
    """

    value = 0.

    # ============== cuboid ===================
    if kind_map == 10:
         
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
            
    # ======== hollow cylinder =================
    elif kind_map == 11:

        a1 = params_map[0]
        a2 = params_map[1]
        r0 = params_map[2]

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
            value = 2*pi*r0

    # ============ colella =================
    elif kind_map == 12:

        Lx    = params_map[0]
        Ly    = params_map[1]
        alpha = params_map[2]
        Lz    = params_map[3]

        if   component == 11:
            value = Lx * (1 + alpha * cos(2*pi*eta1) * sin(2*pi*eta2) * 2*pi)
        elif component == 12:
            value = Lx * alpha * sin(2*pi*eta1) * cos(2*pi*eta2) * 2*pi
        elif component == 13:
            value = 0.
        elif component == 21:
            value = Ly * alpha * cos(2*pi*eta1) * sin(2*pi*eta2) * 2*pi
        elif component == 22:
            value = Ly * (1 + alpha * sin(2*pi*eta1) * cos(2*pi*eta2) * 2*pi)
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.    
        elif component == 33:
            value = Lz

    # =========== orthogonal ================
    elif kind_map == 13:

        Lx    = params_map[0]
        Ly    = params_map[1]
        alpha = params_map[2]
        Lz    = params_map[3]

        if   component == 11:
            value = Lx * (1 + alpha * cos(2*pi*eta1) * 2*pi)
        elif component == 12:
            value = 0.
        elif component == 13:
            value = 0.
        elif component == 21:
            value = 0.
        elif component == 22:
            value = Ly * (1 + alpha * cos(2*pi*eta2) * 2*pi)
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.    
        elif component == 33:
            value = Lz

    # ========= hollow torus ==================
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

    # ========= ellipse =====================
    elif kind_map == 15:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        Lz = params_map[5]

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
            value = Lz

    # ========= rotated ellipse =====================
    elif kind_map == 16:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        r1 = params_map[3]
        r2 = params_map[4]
        Lz = params_map[5]
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
            value = Lz

    # ========= soloviev approx =====================
    elif kind_map == 17:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        Lz = params_map[5]
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
            value = Lz

    # ========= soloviev sqrt =====================
    elif kind_map == 18:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        Lz = params_map[5]
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
            value = Lz

    # ========= soloviev cf =====================
    elif kind_map == 19:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        r0 = params_map[3]
        Lz = params_map[4]
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
            value = Lz

    return value


# =======================================================================
def df_ana_mat(eta1 : 'float', eta2 : 'float', eta3 : 'float', kind_map : 'int', params_map : 'float[:]', mat_out : 'float[:,:]'):
    """
    Point-wise evaluation of ij-th component of the Jacobian matrix df_ij = df_i/deta_j (i,j=1,2,3). Only for analytical mappings, not spline mappings.

    Parameters:
    -----------
        eta1, eta2, eta3:       float              logical coordinates in [0, 1]
        component:              int                 11 : (df1/deta1), 12 : (df1/deta2), 13 : (df1/deta3)
                                                    21 : (df2/deta1), 22 : (df2/deta2), 23 : (df2/deta3)
                                                    31 : (df3/deta1), 32 : (df3/deta2), 33 : (df3/deta3)
        kind_map:               int                 kind of mapping (see module docstring)
        params_map:             float[:]           parameters for the mapping
        mat_out:                float[:,:]         matrix in which the resulting Jacobian matrix will be written
    """

    # make sure that the matrix is empty
    mat_out[:,:] = 0.

    # ============== cuboid ===================
    if kind_map == 10:
         
        b1 = params_map[0]
        e1 = params_map[1]
        b2 = params_map[2]
        e2 = params_map[3]
        b3 = params_map[4]
        e3 = params_map[5]
        
        mat_out[0,0] = e1 - b1
        mat_out[0,1] = 0.
        mat_out[0,2] = 0.
        mat_out[1,0] = 0.
        mat_out[1,1] = e2 - b2
        mat_out[1,2] = 0.
        mat_out[2,0] = 0.
        mat_out[2,1] = 0.
        mat_out[2,2] = e3 - b3
            
    # ======== hollow cylinder =================
    elif kind_map == 11:

        a1 = params_map[0]
        a2 = params_map[1]
        r0 = params_map[2]

        da = a2 - a1

        mat_out[0,0] = da * cos(2*pi*eta2)
        mat_out[0,1] = -2*pi * (a1 + eta1 * da) * sin(2*pi*eta2)
        mat_out[0,2] = 0.
        mat_out[1,0] = da * sin(2*pi*eta2)
        mat_out[1,1] = 2*pi * (a1 + eta1 * da) * cos(2*pi*eta2)
        mat_out[1,2] = 0.
        mat_out[2,0] = 0.
        mat_out[2,1] = 0.
        mat_out[2,2] = 2*pi*r0

    # ============ colella =================
    elif kind_map == 12:

        Lx    = params_map[0]
        Ly    = params_map[1]
        alpha = params_map[2]
        Lz    = params_map[3]

        mat_out[0,0] = Lx * (1 + alpha * cos(2*pi*eta1) * sin(2*pi*eta2) * 2*pi)
        mat_out[0,1] = Lx * alpha * sin(2*pi*eta1) * cos(2*pi*eta2) * 2*pi
        mat_out[0,2] = 0.
        mat_out[1,0] = Ly * alpha * cos(2*pi*eta1) * sin(2*pi*eta2) * 2*pi
        mat_out[1,1] = Ly * (1 + alpha * sin(2*pi*eta1) * cos(2*pi*eta2) * 2*pi)
        mat_out[1,2] = 0.
        mat_out[2,0] = 0.
        mat_out[2,1] = 0.    
        mat_out[2,2] = Lz

    # =========== orthogonal ================
    elif kind_map == 13:

        Lx    = params_map[0]
        Ly    = params_map[1]
        alpha = params_map[2]
        Lz    = params_map[3]

        mat_out[0,0] = Lx * (1 + alpha * cos(2*pi*eta1) * 2*pi)
        mat_out[0,1] = 0.
        mat_out[0,2] = 0.
        mat_out[1,0] = 0.
        mat_out[1,1] = Ly * (1 + alpha * cos(2*pi*eta2) * 2*pi)
        mat_out[1,2] = 0.
        mat_out[2,0] = 0.
        mat_out[2,1] = 0.    
        mat_out[2,2] = Lz

    # ========= hollow torus ==================
    elif kind_map == 14:

        a1 = params_map[0]
        a2 = params_map[1]
        r0 = params_map[2]

        da = a2 - a1

        mat_out[0,0] = da * cos(2*pi*eta2) * cos(2*pi*eta3)
        mat_out[0,1] = -2*pi * (a1 + eta1 * da) * sin(2*pi*eta2) * cos(2*pi*eta3)
        mat_out[0,2] = -2*pi * ((a1 + eta1 * da) * cos(2*pi*eta2) + r0) * sin(2*pi*eta3)
        mat_out[1,0] = da * sin(2*pi*eta2)
        mat_out[1,1] = (a1 + eta1 * da) * cos(2*pi*eta2) * 2*pi
        mat_out[1,2] = 0.
        mat_out[2,0] = da * cos(2*pi*eta2) * sin(2*pi*eta3)
        mat_out[2,1] = -2*pi * (a1 + eta1 * da) * sin(2*pi*eta2) * sin(2*pi*eta3)
        mat_out[2,2] = ((a1 + eta1 * da) * cos(2*pi*eta2) + r0) * cos(2*pi*eta3) * 2*pi

    # ========= ellipse =====================
    elif kind_map == 15:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        Lz = params_map[5]

        mat_out[0,0] = rx * cos(2*pi*eta2)
        mat_out[0,1] = -2*pi * (eta1 * rx) * sin(2*pi*eta2)
        mat_out[0,2] = 0.
        mat_out[1,0] = ry * sin(2*pi*eta2)
        mat_out[1,1] =  2*pi * (eta1 * ry) * cos(2*pi*eta2)
        mat_out[1,2] = 0.
        mat_out[2,0] = 0.
        mat_out[2,1] = 0.
        mat_out[2,2] = Lz

    # ========= rotated ellipse =====================
    elif kind_map == 16:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        r1 = params_map[3]
        r2 = params_map[4]
        Lz = params_map[5]
        th = params_map[6] # Domain: [0,1)

        mat_out[0,0] = r1 * cos(2*pi*th) * cos(2*pi*eta2) - r2 * sin(2*pi*th) * sin(2*pi*eta2)
        mat_out[0,1] = -2*pi * (eta1 * r1) * cos(2*pi*th) * sin(2*pi*eta2) - 2*pi * (eta1 * r2) * sin(2*pi*th) * cos(2*pi*eta2)
        mat_out[0,2] = 0.
        mat_out[1,0] = r1 * sin(2*pi*th) * cos(2*pi*eta2) + r2 * cos(2*pi*th) * sin(2*pi*eta2)
        mat_out[1,1] = -2*pi * (eta1 * r1) * sin(2*pi*th) * sin(2*pi*eta2) + 2*pi * (eta1 * r2) * cos(2*pi*th) * cos(2*pi*eta2)
        mat_out[1,2] = 0.
        mat_out[2,0] = 0.
        mat_out[2,1] = 0.
        mat_out[2,2] = Lz

    # ========= soloviev approx =====================
    elif kind_map == 17:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        Lz = params_map[5]
        de = params_map[6] # Domain: [0,0.1]

        mat_out[0,0] = rx * cos(2*pi*eta2) - 2 * eta1 * rx * de
        mat_out[0,1] = -2*pi * (eta1 * rx) * sin(2*pi*eta2)
        mat_out[0,2] = 0.
        mat_out[1,0] = ry * sin(2*pi*eta2)
        mat_out[1,1] =  2*pi * (eta1 * ry) * cos(2*pi*eta2)
        mat_out[1,2] = 0.
        mat_out[2,0] = 0.
        mat_out[2,1] = 0.
        mat_out[2,2] = Lz

    # ========= soloviev sqrt =====================
    elif kind_map == 18:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        rx = params_map[3]
        ry = params_map[4]
        Lz = params_map[5]
        de = params_map[6] # Domain: [0,0.1]

        mat_out[0,0] = rx * cos(2*pi*eta2) - 0.5 / sqrt(eta1) * rx * de
        mat_out[0,1] = -2*pi * (eta1 * rx) * sin(2*pi*eta2)
        mat_out[0,2] = 0.
        mat_out[1,0] = ry * sin(2*pi*eta2)
        mat_out[1,1] =  2*pi * (eta1 * ry) * cos(2*pi*eta2)
        mat_out[1,2] = 0.
        mat_out[2,0] = 0.
        mat_out[2,1] = 0.
        mat_out[2,2] = Lz

    # ========= soloviev cf =====================
    elif kind_map == 19:

        x0 = params_map[0]
        y0 = params_map[1]
        z0 = params_map[2]
        r0 = params_map[3]
        Lz = params_map[4]
        dx = params_map[5] # Grad-Shafranov shift along x-axis.
        dy = params_map[6] # Grad-Shafranov shift along y-axis.
        dg = params_map[7] # Delta = sin(alpha): Triangularity, shift of high point.
        eg = params_map[8] # Epsilon: Inverse aspect ratio a/R0.
        kg = params_map[9] # Kappa: Ellipticity (elongation).

        mat_out[0,0] = r0 * (- 2 * dx * eta1 - eg * eta1 * sin(2*pi*eta2) * arcsin(dg) * sin(eta1 * sin(2*pi*eta2) * arcsin(dg) + 2*pi*eta2) + eg * cos(eta1 * sin(2*pi*eta2) * arcsin(dg) + 2*pi*eta2))
        mat_out[0,1] = - r0 * eg * eta1 * (2*pi*eta1 * cos(2*pi*eta2) * arcsin(dg) + 2*pi) * sin(eta1 * sin(2*pi*eta2) * arcsin(dg) + 2*pi*eta2)
        mat_out[0,2] = 0.
        mat_out[1,0] = r0 * (- 2 * dy * eta1 + eg * kg * sin(2*pi*eta2))
        mat_out[1,1] = 2 * pi * r0 * eg * eta1 * kg * cos(2*pi*eta2)
        mat_out[1,2] = 0.
        mat_out[2,0] = 0.
        mat_out[2,1] = 0.
        mat_out[2,2] = Lz
    
    # ========= invalid mapping =================
    else:
        print('Invalid mapping given !!')


# =======================================================================
def det_df(eta1 : 'float', eta2 : 'float', eta3 : 'float', kind_map : 'int', params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> 'float':
    """Point-wise evaluation of Jacobian determinant det(df) = df/deta1.(df/deta2 x df/deta3). 
    
    Parameters:
    -----------
        eta1, eta2, eta3:       float              logical coordinates in [0, 1]
        kind_map:               int                 kind of mapping (see module docstring)
        params_map:             float[:]           parameters for the mapping
        tn1, tn2, tn3:          float[:]           knot vectors for mapping
        pn:                     int[:]              spline degrees for mapping
        nbase_n:                int[:]              dimensions of univariate spline spaces for mapping 
        cx, cy, cz:             float[:, :, :]     control points of (f_1, f_2, f_3)

    Returns:
    --------
        value:  float
            point value of Jacobian determinant det(df)(eta1, eta2, eta3)
    """
    
    value = 0.
    
    df_11 = df(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_12 = df(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_13 = df(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    df_21 = df(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_22 = df(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_23 = df(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    df_31 = df(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_32 = df(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_33 = df(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    value = df_11*(df_22*df_33 - df_32*df_23) + df_21*(df_32*df_13 - df_12*df_33) + df_31*(df_12*df_23 - df_22*df_13)
            
    return value


# =======================================================================
def det_df_mat(eta1 : 'float', eta2 : 'float', eta3 : 'float', kind_map : 'int', params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:]', ind_n2 : 'int[:]', ind_n3 : 'int[:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> 'float':
    """
    Point-wise evaluation of Jacobian determinant det(df) = df/deta1.(df/deta2 x df/deta3). 
    
    Parameters:
    -----------
        eta1, eta2, eta3:       float              logical coordinates in [0, 1]
        kind_map:               int                 kind of mapping (see module docstring)
        params_map:             float[:]           parameters for the mapping
        tn1, tn2, tn3:          float[:]           knot vectors for mapping
        pn:                     int[:]              spline degrees for mapping
        nbase_n:                int[:]              dimensions of univariate spline spaces for mapping 
        cx, cy, cz:             float[:, :, :]     control points of (f_1, f_2, f_3)

    Returns:
    --------
        value:  float
            point value of Jacobian determinant det(df)(eta1, eta2, eta3)
    """
    
    value = 0.
    mat_out = empty( (3,3), dtype=float )
    
    df_mat(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz, mat_out)
    
    value = mat_out[0,0]*( mat_out[1,1]*mat_out[2,2] - mat_out[2,1]*mat_out[1,2] ) + mat_out[1,0]*( mat_out[2,1]*mat_out[0,2] - mat_out[0,1]*mat_out[2,2] ) + mat_out[2,0]*( mat_out[0,1]*mat_out[1,2] - mat_out[1,1]*mat_out[0,2] )
            
    return value


# =======================================================================
def df_inv(eta1 : 'float', eta2 : 'float', eta3 : 'float', component : 'int', kind_map : 'int', params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> 'float':
    """Point-wise evaluation of ij-th component of the inverse Jacobian matrix df^(-1)_ij (i,j=1,2,3). 
    
    The 3 x 3 inverse is computed directly from df, using the cross product of the columns of df:

                            | [ (df/deta2) x (df/deta3) ]^T |
    (df)^(-1) = 1/det_df *  | [ (df/deta3) x (df/deta1) ]^T |
                            | [ (df/deta1) x (df/deta2) ]^T |
    
    Parameters:
    -----------
        eta1, eta2, eta3:       float              logical coordinates in [0, 1]
        component:              int                 index ij (11, 12, 13, 21, 22, 23, 31, 32, 33)
        kind_map:               int                 kind of mapping (see module docstring)
        params_map:             float[:]           parameters for the mapping
        tn1, tn2, tn3:          float[:]           knot vectors for mapping
        pn:                     int[:]              spline degrees for mapping
        nbase_n:                int[:]              dimensions of univariate spline spaces for mapping 
        cx, cy, cz:             float[:, :, :]     control points of (f_1, f_2, f_3)

    Returns:
    --------
        value:  float
            point value df^(-1)_ij(eta1, eta2, eta3)
    """
    
    value = 0.

    df_11 = df(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_12 = df(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_13 = df(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    df_21 = df(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_22 = df(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_23 = df(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    df_31 = df(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_32 = df(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_33 = df(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    detdf = df_11*(df_22*df_33 - df_32*df_23) + df_21*(df_32*df_13 - df_12*df_33) + df_31*(df_12*df_23 - df_22*df_13)

    if   component == 11:
        value = (df_22*df_33 - df_32*df_23)/detdf
    elif component == 12:
        value = (df_32*df_13 - df_12*df_33)/detdf
    elif component == 13:
        value = (df_12*df_23 - df_22*df_13)/detdf
    elif component == 21:
        value = (df_23*df_31 - df_33*df_21)/detdf
    elif component == 22:
        value = (df_33*df_11 - df_13*df_31)/detdf
    elif component == 23:
        value = (df_13*df_21 - df_23*df_11)/detdf
    elif component == 31:
        value = (df_21*df_32 - df_31*df_22)/detdf
    elif component == 32:
        value = (df_31*df_12 - df_11*df_32)/detdf
    elif component == 33:
        value = (df_11*df_22 - df_21*df_12)/detdf
            
    return value


# =======================================================================
def g(eta1 : 'float', eta2 : 'float', eta3 : 'float', component : 'int', kind_map : 'int', params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> 'float':
    """Point-wise evaluation of ij-th component of metric tensor g_ij = sum_k (df^T)_ik (df)_kj (i,j,k=1,2,3). 
    
    Parameters:
    -----------
        eta1, eta2, eta3:       float              logical coordinates in [0, 1]
        component:              int                 index ij (11, 12, 13, 21, 22, 23, 31, 32, 33)
        kind_map:               int                 kind of mapping (see module docstring)
        params_map:             float[:]           parameters for the mapping
        tn1, tn2, tn3:          float[:]           knot vectors for mapping
        pn:                     int[:]              spline degrees for mapping
        nbase_n:                int[:]              dimensions of univariate spline spaces for mapping 
        cx, cy, cz:             float[:, :, :]     control points of (f_1, f_2, f_3)

    Returns:
    --------
        value:  float
            point value g_ij(eta1, eta2, eta3)
    """
    
    value = 0.

    if   component == 11:
        df_11 = df(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_21 = df(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_31 = df(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        value = df_11*df_11 + df_21*df_21 + df_31*df_31
        
    elif component == 22:                                              
        df_12 = df(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_22 = df(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_32 = df(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        value = df_12*df_12 + df_22*df_22 + df_32*df_32
                 
    elif component == 33:                                              
        df_13 = df(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_23 = df(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_33 = df(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        value = df_13*df_13 + df_23*df_23 + df_33*df_33
                 
    elif component == 12 or component == 21:
        df_11 = df(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_21 = df(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_31 = df(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_12 = df(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_22 = df(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_32 = df(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        value = df_11*df_12 + df_21*df_22 + df_31*df_32
                 
    elif component == 13 or component == 31:
        df_11 = df(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_21 = df(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_31 = df(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_13 = df(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_23 = df(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_33 = df(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        value = df_11*df_13 + df_21*df_23 + df_31*df_33
                 
    elif component == 23 or component == 32:  
        df_12 = df(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_22 = df(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_32 = df(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_13 = df(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_23 = df(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_33 = df(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        value = df_12*df_13 + df_22*df_23 + df_32*df_33
               
    return value


# =======================================================================
def g_inv(eta1 : 'float', eta2 : 'float', eta3 : 'float', component : 'int', kind_map : 'int', params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> 'float':
    """Point-wise evaluation of ij-th component of inverse metric tensor g^(-1)_ij = sum_k (df^-1)_ik (df^-T)_kj (i,j,k=1,2,3). 
    
    Parameters:
    -----------
        eta1, eta2, eta3:       float              logical coordinates in [0, 1]
        component:              int                 index ij (11, 12, 13, 21, 22, 23, 31, 32, 33)
        kind_map:               int                 kind of mapping (see module docstring)
        params_map:             float[:]           parameters for the mapping
        tn1, tn2, tn3:          float[:]           knot vectors for mapping
        pn:                     int[:]              spline degrees for mapping
        nbase_n:                int[:]              dimensions of univariate spline spaces for mapping 
        cx, cy, cz:             float[:, :, :]     control points of (f_1, f_2, f_3)

    Returns:
    --------
        value:  float
            point value g^(-1)_ij(eta1, eta2, eta3) 
    """
    
    value = 0.

    df_11 = df(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_12 = df(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_13 = df(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    df_21 = df(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_22 = df(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_23 = df(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    df_31 = df(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_32 = df(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_33 = df(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    detdf = df_11*(df_22*df_33 - df_32*df_23) + df_21*(df_32*df_13 - df_12*df_33) + df_31*(df_12*df_23 - df_22*df_13)
    
    df_inv_11 = (df_22*df_33 - df_32*df_23)/detdf
    df_inv_12 = (df_32*df_13 - df_12*df_33)/detdf
    df_inv_13 = (df_12*df_23 - df_22*df_13)/detdf
    
    df_inv_21 = (df_23*df_31 - df_33*df_21)/detdf
    df_inv_22 = (df_33*df_11 - df_13*df_31)/detdf
    df_inv_23 = (df_13*df_21 - df_23*df_11)/detdf
        
    df_inv_31 = (df_21*df_32 - df_31*df_22)/detdf
    df_inv_32 = (df_31*df_12 - df_11*df_32)/detdf
    df_inv_33 = (df_11*df_22 - df_21*df_12)/detdf
    
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


# ==========================================================================================
def mappings_all(eta1 : 'float', eta2 : 'float', eta3 : 'float', kind_fun : 'int', kind_map : 'int', params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> 'float':
    """Point-wise evaluation of
        - f      : mapping x_i = f_i(eta1, eta2, eta3)
        - df     : Jacobian matrix df_i/deta_j
        - det_df : Jacobian determinant det(df)
        - df_inv : inverse Jacobian matrix (df_i/deta_j)^(-1)
        - g      : metric tensor df^T * df 
        - g_inv  : inverse metric tensor df^(-1) * df^(-T)  .
    
    Parameters:
    -----------
        eta1, eta2, eta3:       float              logical coordinates in [0, 1]
        kind_fun:               int                 function to evaluate (see keys_map in 'domain_3d.py')
        kind_map:               int                 kind of mapping (see module docstring)
        params_map:             float[:]           parameters for the mapping
        tn1, tn2, tn3:          float[:]           knot vectors for mapping
        pn:                     int[:]              spline degrees for mapping
        nbase_n:                int[:]              dimensions of univariate spline spaces for mapping 
        cx, cy, cz:             float[:, :, :]     control points of (f_1, f_2, f_3)

    Returns:
    --------
        value:  float
            point value of mapping/metric coefficient at (eta1, eta2, eta3)
    """
    
    value = 0.
    
    # mapping f
    if   kind_fun == 1:
        value = f(eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 2:
        value = f(eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 3:
        value = f(eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    # Jacobian matrix df
    elif kind_fun == 11:
        value = df(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 12:
        value = df(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 13:
        value = df(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 14:
        value = df(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 15:
        value = df(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 16:
        value = df(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 17:
        value = df(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 18:
        value = df(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 19:
        value = df(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
    # Jacobian determinant det_df
    elif kind_fun == 4:
        value = det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
    # inverse Jacobian matrix df_inv
    elif kind_fun == 21:
        value = df_inv(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 22:
        value = df_inv(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 23:
        value = df_inv(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 24:
        value = df_inv(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 25:
        value = df_inv(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 26:
        value = df_inv(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 27:
        value = df_inv(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 28:
        value = df_inv(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 29:
        value = df_inv(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
    # metric tensor g
    elif kind_fun == 31:
        value = g(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 32:
        value = g(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 33:
        value = g(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 34:
        value = g(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 35:
        value = g(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 36:
        value = g(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 37:
        value = g(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)     
    elif kind_fun == 38:
        value = g(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 39:
        value = g(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    # metric tensor g_inv
    elif kind_fun == 41:
        value = g_inv(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 42:
        value = g_inv(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 43:
        value = g_inv(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)  
    elif kind_fun == 44:
        value = g_inv(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 45:
        value = g_inv(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 46:
        value = g_inv(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 47:
        value = g_inv(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 48:
        value = g_inv(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 49:
        value = g_inv(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    return value


# ==========================================================================================   
def kernel_evaluate(eta1 : 'float[:,:,:]', eta2 : 'float[:,:,:]', eta3 : 'float[:,:,:]', kind_fun : 'int', kind_map : 'int', params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', mat_f : 'float[:,:,:]'):
    """Matrix-wise evaluation of
        - f      : mapping x_i = f_i(eta1, eta2, eta3)
        - df     : Jacobian matrix df_i/deta_j
        - det_df : Jacobian determinant det(df)
        - df_inv : inverse Jacobian matrix (df_i/deta_j)^(-1)
        - g      : metric tensor df^T * df 
        - g_inv  : inverse metric tensor df^(-1) * df^(-T)  .

    Parameters
    ----------
        eta1, eta2, eta3:       float[:, :, :]     matrices of logical coordinates in [0, 1]
        kind_fun:               int                 function to evaluate (see keys_map in 'domain_3d.py')
        kind_map:               int                 kind of mapping (see module docstring)
        params_map:             float[:]           parameters for the mapping
        tn1, tn2, tn3:          float[:]           knot vectors for mapping
        pn:                     int[:]              spline degrees for mapping
        nbase_n:                int[:]              dimensions of univariate spline spaces for mapping 
        cx, cy, cz:             float[:, :, :]     control points of (f_1, f_2, f_3)

    Returns
    -------
        mat_f : np.array
            matrix-valued mapping/metric coefficient evaluated at (eta1, eta2, eta3)
    """

    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                mat_f[i1, i2, i3] = mappings_all(eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)


# ==========================================================================================     
def kernel_evaluate_sparse(eta1 : 'float[:,:,:]', eta2 : 'float[:,:,:]', eta3 : 'float[:,:,:]', kind_fun : 'int', kind_map : 'int', params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', mat_f : 'float[:,:,:]'):
    """Same as `kernel_evaluate`, but for sparse meshgrid.

    Returns
    -------
        mat_f:  ndarray
            matrix-valued mapping/metric coefficient evaluated at (eta1, eta2, eta3)
    """

    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                mat_f[i1, i2, i3] = mappings_all(eta1[i1, 0, 0], eta2[0, i2, 0], eta3[0, 0, i3], kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

                
# ==========================================================================================     
def kernel_evaluate_flat(eta1 : 'float[:]', eta2 : 'float[:]', eta3 : 'float[:]', kind_fun : 'int', kind_map : 'int', params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', mat_f : 'float[:]'):
    """Same as `kernel_evaluate`, but for flat evaluation.

    Returns
    -------
        mat_f:  np.array
            1d array [f(x1, y1, z1) f(x2, y2, z2) etc.]
    """

    for i in range(len(eta1)):
        mat_f[i] = mappings_all(eta1[i], eta2[i], eta3[i], kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
