import cunumpy as xp
import scipy.sparse as spa

import struphy.feec.basics.kernels_3d as ker
import struphy.feec.control_variates.kinetic_extended.fB_massless_kernels_control_variate as ker_cv
import struphy.feec.control_variates.kinetic_extended.fnB_massless_cv_kernel_2 as ker_cv2


def bv_right(
    p,
    indN,
    indD,
    Nel,
    G_inv_11,
    G_inv_12,
    G_inv_13,
    G_inv_22,
    G_inv_23,
    G_inv_33,
    DFI_11,
    DFI_12,
    DFI_13,
    DFI_21,
    DFI_22,
    DFI_23,
    DFI_31,
    DFI_32,
    DFI_33,
    df_det,
    Jeqx,
    Jeqy,
    Jeqz,
    temp_dft,
    generate_weight1,
    generate_weight2,
    generate_weight3,
    temp_twoform1,
    temp_twoform2,
    temp_twoform3,
    b1,
    b2,
    b3,
    uvalue,
    b1value,
    b2value,
    b3value,
    tensor_space_FEM,
):
    r"""
    Computes the matrix Q4_(ab,ij) = \int e^{-U} (DF^{-T}B)_{6-a-b} (Lambda^2_(a,i) x Lambda^2_(b,j))_{6-a-b} deta, no summatio over a,b!! a is not equal to b
    2-form space V2 = span(Lambda^2_(a,i)) with a in [1,2,3] and i in [1, N^2_a]
    Number of basis functions is N2 = N2_1 + N2_2 + N2_3

    ab is a 3x3 block matrix structure: skew-symmetric matrix, we compute elements 12, 31, 23
    M12 = Lambda^2_1 x Lambda^2_2 contracts with the 3rd component of DF^{-T}(B1, B2, B3)
    M31 = Lambda^2_3 x Lambda^2_1 contracts with the 2nd component of DF^{-T}(B1, B2, B3)
    M23 = Lambda^2_2 x Lambda^2_3 contracts with the 1st component of DF^{-T}(B1, B2, B3)

    Parameters:
            Mab_kernel: ndarray
            [b1, b2, b3]: coefficents of B
            u: coefficients of U
            tensor_space_FEM: STRUPHY 3D spline object
            mapping: STRUPHY mapping type



    Returns:
            M: Q4 block matrix, csc sparse
    """
    # ============= load information about B-splines =============
    d = [p[0] - 1, p[1] - 1, p[2] - 1]  # D splin degrees
    nq1 = tensor_space_FEM.n_quad[0]
    nq2 = tensor_space_FEM.n_quad[1]
    nq3 = tensor_space_FEM.n_quad[2]
    bn1 = tensor_space_FEM.basisN[0]
    bn2 = tensor_space_FEM.basisN[1]
    bn3 = tensor_space_FEM.basisN[2]
    bd1 = tensor_space_FEM.basisD[0]
    bd2 = tensor_space_FEM.basisD[1]
    bd3 = tensor_space_FEM.basisD[2]
    pts1 = tensor_space_FEM.pts[0]
    pts2 = tensor_space_FEM.pts[1]
    pts3 = tensor_space_FEM.pts[2]
    wts1 = tensor_space_FEM.wts[0]
    wts2 = tensor_space_FEM.wts[1]
    wts3 = tensor_space_FEM.wts[2]

    ker_cv.bvright1(
        G_inv_11,
        G_inv_12,
        G_inv_13,
        G_inv_22,
        G_inv_23,
        G_inv_33,
        indN[0],
        indN[1],
        indN[2],
        indD[0],
        indD[1],
        indD[2],
        Nel[0],
        Nel[1],
        Nel[2],
        nq1,
        nq2,
        nq3,
        p[0],
        p[1],
        p[2],
        d[0],
        d[1],
        d[2],
        b1value,
        b2value,
        b3value,
        b1,
        b2,
        b3,
        temp_dft,
        generate_weight1,
        generate_weight2,
        generate_weight3,
        bn1,
        bn2,
        bn3,
        bd1,
        bd2,
        bd3,
    )
    ker_cv.bvright2(
        DFI_11,
        DFI_12,
        DFI_13,
        DFI_21,
        DFI_22,
        DFI_23,
        DFI_31,
        DFI_32,
        DFI_33,
        df_det,
        Jeqx,
        Jeqy,
        Jeqz,
        Nel[0],
        Nel[1],
        Nel[2],
        nq1,
        nq2,
        nq3,
        p[0],
        p[1],
        p[2],
        d[0],
        d[1],
        d[2],
        b1value,
        b2value,
        b3value,
        uvalue,
        temp_dft,
        generate_weight1,
        generate_weight2,
        generate_weight3,
        pts1,
        pts2,
        pts3,
        wts1,
        wts2,
        wts3,
    )
    ker_cv.bvfinal(
        indN[0],
        indN[1],
        indN[2],
        indD[0],
        indD[1],
        indD[2],
        Nel[0],
        Nel[1],
        Nel[2],
        nq1,
        nq2,
        nq3,
        p[0],
        p[1],
        p[2],
        d[0],
        d[1],
        d[2],
        b1value,
        b2value,
        b3value,
        uvalue,
        bn1,
        bn2,
        bn3,
        bd1,
        bd2,
        bd3,
        temp_twoform1,
        temp_twoform2,
        temp_twoform3,
    )
    # ========================= C.T ===========================
    return tensor_space_FEM.C.T.dot(
        xp.concatenate((temp_twoform1.flatten(), temp_twoform2.flatten(), temp_twoform3.flatten()))
    )


def vv_right(
    stage_index,
    tol,
    Np_loc,
    LO_inv,
    domain,
    acc,
    NbaseN,
    NbaseD,
    temp_particle,
    p,
    Nel,
    tensor_space_FEM,
    b1,
    b2,
    b3,
    particles_loc,
):
    r"""
    Computes the matrix Q4_(ab,ij) = \int e^{-U} (DF^{-T}B)_{6-a-b} (Lambda^2_(a,i) x Lambda^2_(b,j))_{6-a-b} deta, no summatio over a,b!! a is not equal to b
    2-form space V2 = span(Lambda^2_(a,i)) with a in [1,2,3] and i in [1, N^2_a]
    Number of basis functions is N2 = N2_1 + N2_2 + N2_3

    ab is a 3x3 block matrix structure: skew-symmetric matrix, we compute elements 12, 31, 23
    M12 = Lambda^2_1 x Lambda^2_2 contracts with the 3rd component of DF^{-T}(B1, B2, B3)
    M31 = Lambda^2_3 x Lambda^2_1 contracts with the 2nd component of DF^{-T}(B1, B2, B3)
    M23 = Lambda^2_2 x Lambda^2_3 contracts with the 1st component of DF^{-T}(B1, B2, B3)

    Parameters:
            Mab_kernel: ndarray
            [b1, b2, b3]: coefficents of B
            u: coefficients of U
            tensor_space_FEM: STRUPHY 3D spline object
            mapping: STRUPHY mapping type



    Returns:
            M: Q4 block matrix, csc sparse
    """
    # =====we can just calculate 3 matrices=====
    # ============= load information about B-splines =============
    if stage_index == 1:
        ker_cv.vv(
            tol,
            acc.stage1_out_loc,
            temp_particle,
            b1,
            b2,
            b3,
            LO_inv,
            Np_loc,
            NbaseN,
            NbaseD,
            Nel,
            p,
            tensor_space_FEM.T[0],
            tensor_space_FEM.T[1],
            tensor_space_FEM.T[2],
            particles_loc,
            domain.kind_map,
            domain.params,
            domain.T[0],
            domain.T[1],
            domain.T[2],
            domain.p,
            domain.Nel,
            domain.NbaseN,
            domain.cx,
            domain.cy,
            domain.cz,
        )
    elif stage_index == 2:
        ker_cv.vv(
            tol,
            acc.stage2_out_loc,
            temp_particle,
            b1,
            b2,
            b3,
            LO_inv,
            Np_loc,
            NbaseN,
            NbaseD,
            Nel,
            p,
            tensor_space_FEM.T[0],
            tensor_space_FEM.T[1],
            tensor_space_FEM.T[2],
            particles_loc,
            domain.kind_map,
            domain.params,
            domain.T[0],
            domain.T[1],
            domain.T[2],
            domain.p,
            domain.Nel,
            domain.NbaseN,
            domain.cx,
            domain.cy,
            domain.cz,
        )
    elif stage_index == 3:
        ker_cv.vv(
            tol,
            acc.stage3_out_loc,
            temp_particle,
            b1,
            b2,
            b3,
            LO_inv,
            Np_loc,
            NbaseN,
            NbaseD,
            Nel,
            p,
            tensor_space_FEM.T[0],
            tensor_space_FEM.T[1],
            tensor_space_FEM.T[2],
            particles_loc,
            domain.kind_map,
            domain.params,
            domain.T[0],
            domain.T[1],
            domain.T[2],
            domain.p,
            domain.Nel,
            domain.NbaseN,
            domain.cx,
            domain.cy,
            domain.cz,
        )
    else:
        ker_cv.vv(
            tol,
            acc.stage4_out_loc,
            temp_particle,
            b1,
            b2,
            b3,
            LO_inv,
            Np_loc,
            NbaseN,
            NbaseD,
            Nel,
            p,
            tensor_space_FEM.T[0],
            tensor_space_FEM.T[1],
            tensor_space_FEM.T[2],
            particles_loc,
            domain.kind_map,
            domain.params,
            domain.T[0],
            domain.T[1],
            domain.T[2],
            domain.p,
            domain.Nel,
            domain.NbaseN,
            domain.cx,
            domain.cy,
            domain.cz,
        )


def quadrature_density(gather, domain):
    """
    Computes the matrix

    Parameters:
            Mab_kernel: ndarray
            [b1, b2, b3]: coefficents of B
            u: coefficients of U
            tensor_space_FEM: STRUPHY 3D spline object
            mapping: STRUPHY mapping type


    Returns:
            M: Q4 block matrix, csc sparse
    """
    # =====we can just calculate 3 matrices=====
    # ============= load information about B-splines =============

    ker_cv2.quadrature_density(
        gather.Nel,
        gather.pts[0],
        gather.pts[1],
        gather.pts[2],
        gather.n_quad,
        gather.gather_quadrature,
        domain.kind_map,
        domain.params,
        domain.T[0],
        domain.T[1],
        domain.T[2],
        domain.p,
        domain.NbaseN,
        domain.cx,
        domain.cy,
        domain.cz,
    )


def quadrature_grid(gather, domain):
    """
    Computes the matrix

    Parameters:
            Mab_kernel: ndarray
            [b1, b2, b3]: coefficents of B
            u: coefficients of U
            tensor_space_FEM: STRUPHY 3D spline object
            mapping: STRUPHY mapping type


    Returns:
            M: Q4 block matrix, csc sparse
    """
    # =====we can just calculate 3 matrices=====
    # ============= load information about B-splines =============

    ker_cv.grid_density(
        gather.Nel,
        gather.gather_grid,
        domain.kind_map,
        domain.params,
        domain.T[0],
        domain.T[1],
        domain.T[2],
        domain.p,
        domain.NbaseN,
        domain.cx,
        domain.cy,
        domain.cz,
    )
