# coding: utf-8


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
- kind_map = 11 : orthogonal,         params_map = [Lx, Ly, alpha, Lz].
- kind_map = 12 : colella,            params_map = [Lx, Ly, alpha, Lz].
- kind_map = 20 : hollow cylinder,    params_map = [a1, a2, R0].
- kind_map = 22 : hollow torus,       params_map = [a1, a2, R0].
- kind_map = 30 : shafranov shift,    params_map = [x0, y0, z0, rx, ry, Lz, delta].
- kind_map = 31 : shafranov sqrt,     params_map = [x0, y0, z0, rx, ry, Lz, delta].
- kind_map = 32 : shafranov D-shaped, params_map = [x0, y0, z0, R0, Lz, delta_x, delta_y, delta_gs, epsilon_gs, kappa_gs].
"""

from numpy import arcsin, arctan2, cos, empty, pi, shape, sin, sqrt

import struphy.pic.tests.test_pic_legacy_files.spline_evaluation_2d as eva_2d
import struphy.pic.tests.test_pic_legacy_files.spline_evaluation_3d as eva_3d


# =======================================================================
def f(
    eta1: "float",
    eta2: "float",
    eta3: "float",
    component: "int",
    kind_map: "int",
    params_map: "float[:]",
    tn1: "float[:]",
    tn2: "float[:]",
    tn3: "float[:]",
    pn: "int[:]",
    nbase_n: "int[:]",
    cx: "float[:,:,:]",
    cy: "float[:,:,:]",
    cz: "float[:,:,:]",
) -> "float":
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

    value = 0.0

    # =========== 3d spline ========================
    if kind_map == 0:
        if component == 1:
            value = eva_3d.evaluate_n_n_n(
                tn1,
                tn2,
                tn3,
                pn[0],
                pn[1],
                pn[2],
                nbase_n[0],
                nbase_n[1],
                nbase_n[2],
                cx,
                eta1,
                eta2,
                eta3,
            )

        elif component == 2:
            value = eva_3d.evaluate_n_n_n(
                tn1,
                tn2,
                tn3,
                pn[0],
                pn[1],
                pn[2],
                nbase_n[0],
                nbase_n[1],
                nbase_n[2],
                cy,
                eta1,
                eta2,
                eta3,
            )

        elif component == 3:
            value = eva_3d.evaluate_n_n_n(
                tn1,
                tn2,
                tn3,
                pn[0],
                pn[1],
                pn[2],
                nbase_n[0],
                nbase_n[1],
                nbase_n[2],
                cz,
                eta1,
                eta2,
                eta3,
            )

    # ==== 2d spline (straight in 3rd direction) ===
    elif kind_map == 1:
        Lz = params_map[0]

        if component == 1:
            value = eva_2d.evaluate_n_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx[:, :, 0], eta1, eta2)

            if eta1 == 0.0 and cx[0, 0, 0] == cx[0, 1, 0]:
                value = cx[0, 0, 0]

        elif component == 2:
            value = eva_2d.evaluate_n_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cy[:, :, 0], eta1, eta2)

            if eta1 == 0.0 and cy[0, 0, 0] == cy[0, 1, 0]:
                value = cy[0, 0, 0]

        elif component == 3:
            value = Lz * eta3

    # ==== 2d spline (curvature in 3rd direction) ===
    elif kind_map == 2:
        if component == 1:
            value = eva_2d.evaluate_n_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx[:, :, 0], eta1, eta2) * cos(
                2 * pi * eta3,
            )

            if eta1 == 0.0 and cx[0, 0, 0] == cx[0, 1, 0]:
                value = cx[0, 0, 0] * cos(2 * pi * eta3)

        elif component == 2:
            value = eva_2d.evaluate_n_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cy[:, :, 0], eta1, eta2)

            if eta1 == 0.0 and cy[0, 0, 0] == cy[0, 1, 0]:
                value = cy[0, 0, 0]

        elif component == 3:
            value = eva_2d.evaluate_n_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx[:, :, 0], eta1, eta2) * sin(
                2 * pi * eta3,
            )

            if eta1 == 0.0 and cx[0, 0, 0] == cx[0, 1, 0]:
                value = cx[0, 0, 0] * sin(2 * pi * eta3)

    # ============== cuboid =========================
    elif kind_map == 10:
        b1 = params_map[0]
        e1 = params_map[1]
        b2 = params_map[2]
        e2 = params_map[3]
        b3 = params_map[4]
        e3 = params_map[5]

        # value =  begin + (end - begin) * eta
        if component == 1:
            value = b1 + (e1 - b1) * eta1
        elif component == 2:
            value = b2 + (e2 - b2) * eta2
        elif component == 3:
            value = b3 + (e3 - b3) * eta3

    # ========= hollow cylinder =====================
    elif kind_map == 20:
        a1 = params_map[0]
        a2 = params_map[1]
        lz = params_map[2]

        da = a2 - a1

        if component == 1:
            value = (a1 + eta1 * da) * cos(2 * pi * eta2)
        elif component == 2:
            value = (a1 + eta1 * da) * sin(2 * pi * eta2)
        elif component == 3:
            value = lz * eta3

    # ============ colella ==========================
    elif kind_map == 12:
        Lx = params_map[0]
        Ly = params_map[1]
        alpha = params_map[2]
        Lz = params_map[3]

        if component == 1:
            value = Lx * (eta1 + alpha * sin(2 * pi * eta1) * sin(2 * pi * eta2))
        elif component == 2:
            value = Ly * (eta2 + alpha * sin(2 * pi * eta1) * sin(2 * pi * eta2))
        elif component == 3:
            value = Lz * eta3

    # =========== orthogonal ========================
    elif kind_map == 11:
        Lx = params_map[0]
        Ly = params_map[1]
        alpha = params_map[2]
        Lz = params_map[3]

        if component == 1:
            value = Lx * (eta1 + alpha * sin(2 * pi * eta1))
        elif component == 2:
            value = Ly * (eta2 + alpha * sin(2 * pi * eta2))
        elif component == 3:
            value = Lz * eta3

    # ========= hollow torus ========================
    elif kind_map == 22:
        a1 = params_map[0]
        a2 = params_map[1]
        r0 = params_map[2]

        da = a2 - a1

        if component == 1:
            value = ((a1 + eta1 * da) * cos(2 * pi * eta2) + r0) * cos(2 * pi * eta3)
        elif component == 2:
            value = (a1 + eta1 * da) * sin(2 * pi * eta2)
        elif component == 3:
            value = ((a1 + eta1 * da) * cos(2 * pi * eta2) + r0) * sin(2 * pi * eta3)

    # ========= shafranov shift =====================
    elif kind_map == 30:
        rx = params_map[0]
        ry = params_map[1]
        Lz = params_map[2]
        de = params_map[3]  # Domain: [0,0.1]

        if component == 1:
            value = (eta1 * rx) * cos(2 * pi * eta2) + (1 - eta1**2) * rx * de
        elif component == 2:
            value = (eta1 * ry) * sin(2 * pi * eta2)
        elif component == 3:
            value = eta3 * Lz

    # ========= shafranov sqrt =====================
    elif kind_map == 31:
        rx = params_map[0]
        ry = params_map[1]
        Lz = params_map[2]
        de = params_map[3]  # Domain: [0,0.1]

        if component == 1:
            value = (eta1 * rx) * cos(2 * pi * eta2) + (1 - sqrt(eta1)) * rx * de
        elif component == 2:
            value = (eta1 * ry) * sin(2 * pi * eta2)
        elif component == 3:
            value = eta3 * Lz

    # ========= shafranov D-shaped =====================
    elif kind_map == 32:
        r0 = params_map[0]
        Lz = params_map[1]
        dx = params_map[2]  # Grad-Shafranov shift along x-axis.
        dy = params_map[3]  # Grad-Shafranov shift along y-axis.
        dg = params_map[4]  # Delta = sin(alpha): Triangularity, shift of high point.
        eg = params_map[5]  # Epsilon: Inverse aspect ratio a/r0.
        kg = params_map[6]  # Kappa: Ellipticity (elongation).

        if component == 1:
            value = r0 * (
                1 + (1 - eta1**2) * dx + eg * eta1 * cos(2 * pi * eta2 + arcsin(dg) * eta1 * sin(2 * pi * eta2))
            )
        elif component == 2:
            value = r0 * ((1 - eta1**2) * dy + eg * kg * eta1 * sin(2 * pi * eta2))
        elif component == 3:
            value = eta3 * Lz

    return value


# =======================================================================
def df(
    eta1: "float",
    eta2: "float",
    eta3: "float",
    component: "int",
    kind_map: "int",
    params_map: "float[:]",
    tn1: "float[:]",
    tn2: "float[:]",
    tn3: "float[:]",
    pn: "int[:]",
    nbase_n: "int[:]",
    cx: "float[:,:,:]",
    cy: "float[:,:,:]",
    cz: "float[:,:,:]",
) -> "float":
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

    value = 0.0

    # =========== 3d spline ========================
    if kind_map == 0:
        if component == 11:
            value = eva_3d.evaluate_diffn_n_n(
                tn1,
                tn2,
                tn3,
                pn[0],
                pn[1],
                pn[2],
                nbase_n[0],
                nbase_n[1],
                nbase_n[2],
                cx,
                eta1,
                eta2,
                eta3,
            )
        elif component == 12:
            value = eva_3d.evaluate_n_diffn_n(
                tn1,
                tn2,
                tn3,
                pn[0],
                pn[1],
                pn[2],
                nbase_n[0],
                nbase_n[1],
                nbase_n[2],
                cx,
                eta1,
                eta2,
                eta3,
            )
        elif component == 13:
            value = eva_3d.evaluate_n_n_diffn(
                tn1,
                tn2,
                tn3,
                pn[0],
                pn[1],
                pn[2],
                nbase_n[0],
                nbase_n[1],
                nbase_n[2],
                cx,
                eta1,
                eta2,
                eta3,
            )
        elif component == 21:
            value = eva_3d.evaluate_diffn_n_n(
                tn1,
                tn2,
                tn3,
                pn[0],
                pn[1],
                pn[2],
                nbase_n[0],
                nbase_n[1],
                nbase_n[2],
                cy,
                eta1,
                eta2,
                eta3,
            )
        elif component == 22:
            value = eva_3d.evaluate_n_diffn_n(
                tn1,
                tn2,
                tn3,
                pn[0],
                pn[1],
                pn[2],
                nbase_n[0],
                nbase_n[1],
                nbase_n[2],
                cy,
                eta1,
                eta2,
                eta3,
            )
        elif component == 23:
            value = eva_3d.evaluate_n_n_diffn(
                tn1,
                tn2,
                tn3,
                pn[0],
                pn[1],
                pn[2],
                nbase_n[0],
                nbase_n[1],
                nbase_n[2],
                cy,
                eta1,
                eta2,
                eta3,
            )
        elif component == 31:
            value = eva_3d.evaluate_diffn_n_n(
                tn1,
                tn2,
                tn3,
                pn[0],
                pn[1],
                pn[2],
                nbase_n[0],
                nbase_n[1],
                nbase_n[2],
                cz,
                eta1,
                eta2,
                eta3,
            )
        elif component == 32:
            value = eva_3d.evaluate_n_diffn_n(
                tn1,
                tn2,
                tn3,
                pn[0],
                pn[1],
                pn[2],
                nbase_n[0],
                nbase_n[1],
                nbase_n[2],
                cz,
                eta1,
                eta2,
                eta3,
            )
        elif component == 33:
            value = eva_3d.evaluate_n_n_diffn(
                tn1,
                tn2,
                tn3,
                pn[0],
                pn[1],
                pn[2],
                nbase_n[0],
                nbase_n[1],
                nbase_n[2],
                cz,
                eta1,
                eta2,
                eta3,
            )

    # ==== 2d spline (straight in 3rd direction) ===
    elif kind_map == 1:
        Lz = 2 * pi * cx[0, 0, 0]

        if component == 11:
            value = eva_2d.evaluate_diffn_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx[:, :, 0], eta1, eta2)
        elif component == 12:
            value = eva_2d.evaluate_n_diffn(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx[:, :, 0], eta1, eta2)

            if eta1 == 0.0 and cx[0, 0, 0] == cx[0, 1, 0]:
                value = 0.0

        elif component == 13:
            value = 0.0
        elif component == 21:
            value = eva_2d.evaluate_diffn_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cy[:, :, 0], eta1, eta2)
        elif component == 22:
            value = eva_2d.evaluate_n_diffn(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cy[:, :, 0], eta1, eta2)

            if eta1 == 0.0 and cy[0, 0, 0] == cy[0, 1, 0]:
                value = 0.0

        elif component == 23:
            value = 0.0
        elif component == 31:
            value = 0.0
        elif component == 32:
            value = 0.0
        elif component == 33:
            value = Lz

    # ==== 2d spline (curvature in 3rd direction) ===
    elif kind_map == 2:
        if component == 11:
            value = eva_2d.evaluate_diffn_n(
                tn1,
                tn2,
                pn[0],
                pn[1],
                nbase_n[0],
                nbase_n[1],
                cx[:, :, 0],
                eta1,
                eta2,
            ) * cos(2 * pi * eta3)
        elif component == 12:
            value = eva_2d.evaluate_n_diffn(
                tn1,
                tn2,
                pn[0],
                pn[1],
                nbase_n[0],
                nbase_n[1],
                cx[:, :, 0],
                eta1,
                eta2,
            ) * cos(2 * pi * eta3)

            if eta1 == 0.0 and cx[0, 0, 0] == cx[0, 1, 0]:
                value = 0.0

        elif component == 13:
            value = (
                eva_2d.evaluate_n_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx[:, :, 0], eta1, eta2)
                * sin(2 * pi * eta3)
                * (-2 * pi)
            )
        elif component == 21:
            value = eva_2d.evaluate_diffn_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cy[:, :, 0], eta1, eta2)
        elif component == 22:
            value = eva_2d.evaluate_n_diffn(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cy[:, :, 0], eta1, eta2)

            if eta1 == 0.0 and cy[0, 0, 0] == cy[0, 1, 0]:
                value = 0.0

        elif component == 23:
            value = 0.0
        elif component == 31:
            value = eva_2d.evaluate_diffn_n(
                tn1,
                tn2,
                pn[0],
                pn[1],
                nbase_n[0],
                nbase_n[1],
                cx[:, :, 0],
                eta1,
                eta2,
            ) * sin(2 * pi * eta3)
        elif component == 32:
            value = eva_2d.evaluate_n_diffn(
                tn1,
                tn2,
                pn[0],
                pn[1],
                nbase_n[0],
                nbase_n[1],
                cx[:, :, 0],
                eta1,
                eta2,
            ) * sin(2 * pi * eta3)

            if eta1 == 0.0 and cx[0, 0, 0] == cx[0, 1, 0]:
                value = 0.0

        elif component == 33:
            value = (
                eva_2d.evaluate_n_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx[:, :, 0], eta1, eta2)
                * cos(2 * pi * eta3)
                * 2
                * pi
            )

    # ============== cuboid ===================
    elif kind_map == 10:
        b1 = params_map[0]
        e1 = params_map[1]
        b2 = params_map[2]
        e2 = params_map[3]
        b3 = params_map[4]
        e3 = params_map[5]

        if component == 11:
            value = e1 - b1
        elif component == 12:
            value = 0.0
        elif component == 13:
            value = 0.0
        elif component == 21:
            value = 0.0
        elif component == 22:
            value = e2 - b2
        elif component == 23:
            value = 0.0
        elif component == 31:
            value = 0.0
        elif component == 32:
            value = 0.0
        elif component == 33:
            value = e3 - b3

    # ======== hollow cylinder =================
    elif kind_map == 20:
        a1 = params_map[0]
        a2 = params_map[1]
        lz = params_map[2]

        da = a2 - a1

        if component == 11:
            value = da * cos(2 * pi * eta2)
        elif component == 12:
            value = -2 * pi * (a1 + eta1 * da) * sin(2 * pi * eta2)
        elif component == 13:
            value = 0.0
        elif component == 21:
            value = da * sin(2 * pi * eta2)
        elif component == 22:
            value = 2 * pi * (a1 + eta1 * da) * cos(2 * pi * eta2)
        elif component == 23:
            value = 0.0
        elif component == 31:
            value = 0.0
        elif component == 32:
            value = 0.0
        elif component == 33:
            value = lz

    # ============ colella =================
    elif kind_map == 12:
        Lx = params_map[0]
        Ly = params_map[1]
        alpha = params_map[2]
        Lz = params_map[3]

        if component == 11:
            value = Lx * (1 + alpha * cos(2 * pi * eta1) * sin(2 * pi * eta2) * 2 * pi)
        elif component == 12:
            value = Lx * alpha * sin(2 * pi * eta1) * cos(2 * pi * eta2) * 2 * pi
        elif component == 13:
            value = 0.0
        elif component == 21:
            value = Ly * alpha * cos(2 * pi * eta1) * sin(2 * pi * eta2) * 2 * pi
        elif component == 22:
            value = Ly * (1 + alpha * sin(2 * pi * eta1) * cos(2 * pi * eta2) * 2 * pi)
        elif component == 23:
            value = 0.0
        elif component == 31:
            value = 0.0
        elif component == 32:
            value = 0.0
        elif component == 33:
            value = Lz

    # =========== orthogonal ================
    elif kind_map == 11:
        Lx = params_map[0]
        Ly = params_map[1]
        alpha = params_map[2]
        Lz = params_map[3]

        if component == 11:
            value = Lx * (1 + alpha * cos(2 * pi * eta1) * 2 * pi)
        elif component == 12:
            value = 0.0
        elif component == 13:
            value = 0.0
        elif component == 21:
            value = 0.0
        elif component == 22:
            value = Ly * (1 + alpha * cos(2 * pi * eta2) * 2 * pi)
        elif component == 23:
            value = 0.0
        elif component == 31:
            value = 0.0
        elif component == 32:
            value = 0.0
        elif component == 33:
            value = Lz

    # ========= hollow torus ==================
    elif kind_map == 22:
        a1 = params_map[0]
        a2 = params_map[1]
        r0 = params_map[2]

        da = a2 - a1

        if component == 11:
            value = da * cos(2 * pi * eta2) * cos(2 * pi * eta3)
        elif component == 12:
            value = -2 * pi * (a1 + eta1 * da) * sin(2 * pi * eta2) * cos(2 * pi * eta3)
        elif component == 13:
            value = -2 * pi * ((a1 + eta1 * da) * cos(2 * pi * eta2) + r0) * sin(2 * pi * eta3)
        elif component == 21:
            value = da * sin(2 * pi * eta2)
        elif component == 22:
            value = (a1 + eta1 * da) * cos(2 * pi * eta2) * 2 * pi
        elif component == 23:
            value = 0.0
        elif component == 31:
            value = da * cos(2 * pi * eta2) * sin(2 * pi * eta3)
        elif component == 32:
            value = -2 * pi * (a1 + eta1 * da) * sin(2 * pi * eta2) * sin(2 * pi * eta3)
        elif component == 33:
            value = ((a1 + eta1 * da) * cos(2 * pi * eta2) + r0) * cos(2 * pi * eta3) * 2 * pi

    # ========= shafranov shift =====================
    elif kind_map == 30:
        rx = params_map[0]
        ry = params_map[1]
        Lz = params_map[2]
        de = params_map[3]  # Domain: [0,0.1]

        if component == 11:
            value = rx * cos(2 * pi * eta2) - 2 * eta1 * rx * de
        elif component == 12:
            value = -2 * pi * (eta1 * rx) * sin(2 * pi * eta2)
        elif component == 13:
            value = 0.0
        elif component == 21:
            value = ry * sin(2 * pi * eta2)
        elif component == 22:
            value = 2 * pi * (eta1 * ry) * cos(2 * pi * eta2)
        elif component == 23:
            value = 0.0
        elif component == 31:
            value = 0.0
        elif component == 32:
            value = 0.0
        elif component == 33:
            value = Lz

    # ========= shafranov sqrt =====================
    elif kind_map == 31:
        rx = params_map[0]
        ry = params_map[1]
        Lz = params_map[2]
        de = params_map[3]  # Domain: [0,0.1]

        if component == 11:
            value = rx * cos(2 * pi * eta2) - 0.5 / sqrt(eta1) * rx * de
        elif component == 12:
            value = -2 * pi * (eta1 * rx) * sin(2 * pi * eta2)
        elif component == 13:
            value = 0.0
        elif component == 21:
            value = ry * sin(2 * pi * eta2)
        elif component == 22:
            value = 2 * pi * (eta1 * ry) * cos(2 * pi * eta2)
        elif component == 23:
            value = 0.0
        elif component == 31:
            value = 0.0
        elif component == 32:
            value = 0.0
        elif component == 33:
            value = Lz

    # ========= shafranov D-shaped =====================
    elif kind_map == 32:
        r0 = params_map[0]
        Lz = params_map[1]
        dx = params_map[2]  # Grad-Shafranov shift along x-axis.
        dy = params_map[3]  # Grad-Shafranov shift along y-axis.
        dg = params_map[4]  # Delta = sin(alpha): Triangularity, shift of high point.
        eg = params_map[5]  # Epsilon: Inverse aspect ratio a/R0.
        kg = params_map[6]  # Kappa: Ellipticity (elongation).

        if component == 11:
            value = r0 * (
                -2 * dx * eta1
                - eg
                * eta1
                * sin(2 * pi * eta2)
                * arcsin(dg)
                * sin(eta1 * sin(2 * pi * eta2) * arcsin(dg) + 2 * pi * eta2)
                + eg * cos(eta1 * sin(2 * pi * eta2) * arcsin(dg) + 2 * pi * eta2)
            )
        elif component == 12:
            value = (
                -r0
                * eg
                * eta1
                * (2 * pi * eta1 * cos(2 * pi * eta2) * arcsin(dg) + 2 * pi)
                * sin(eta1 * sin(2 * pi * eta2) * arcsin(dg) + 2 * pi * eta2)
            )
        elif component == 13:
            value = 0.0
        elif component == 21:
            value = r0 * (-2 * dy * eta1 + eg * kg * sin(2 * pi * eta2))
        elif component == 22:
            value = 2 * pi * r0 * eg * eta1 * kg * cos(2 * pi * eta2)
        elif component == 23:
            value = 0.0
        elif component == 31:
            value = 0.0
        elif component == 32:
            value = 0.0
        elif component == 33:
            value = Lz

    return value
