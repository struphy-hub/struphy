from numpy import zeros, empty, sqrt

import struphy.kinetic_equil.analytical.background_sol as background_sol
import struphy.pic.mat_vec_filler as mvf
import struphy.geometry.mappings_3d_fast as mapping_fast
import struphy.linear_algebra.core as linalg


def _docstring():
    '''
    MODULE DOCSTRING for **struphy.pic.accumulators**.

    The module contains model-specific accumulation routines, to be defined by the user.

    Naming conventions:
        - use the model name, all lower-case letters (e.g. lin_vlasov_maxwell)
        - in case of multiple accumulations in one model, attach _1, _2, etc. 

    Arguments have to be passed in the following order (copy and paste from existing accum. function):

    - First, the marker info:
        - markers: 'float[:,:]',          # positions [0:3,], velocities [3:6,], and weights [6,] of the markers
        - np: 'int'                       # number of markers

    - then, the spline bases info:
        - p_n: 'int[:]',                  # spline degree in each direction
        - t1: 'float[:]',                 # knot vector in eta1
        - t2: 'float[:]',                 # knot vector in eta2
        - t3: 'float[:]',                 # knot vector in eta3

    - then, the mapping info:
        - kind_map: 'int',                # mapping identifier 
        - params_map: 'float[:]',         # mapping parameters
        - p_map: 'int[:]',                # spline degree
        - t1_map: 'float[:]',             # knot vector in eta1
        - t2_map: 'float[:]',             # knot vector in eta2
        - t3_map: 'float[:]',             # knot vector in eta3
        - cx: 'float[:,:,:]',             # control points for Fx
        - cy: 'float[:,:,:]',             # control points for Fy
        - cz: 'float[:,:,:]',             # control points for Fz

    - then, the mpi.comm info of the space:
        - starts1: 'int[:]'              # start indices in the three directions of current process of component 1
        - in case of vector-valued spaces:

            - starts2: 'int[:]'              
            - starts3: 'int[:]'              

        - ends1: 'int[:]'              # end indices in the three directions of current process of component 1
        - in case of vector-valued spaces:

            - ends2: 'int[:]'              
            - ends3: 'int[:]'              

        - pads1: 'int[:]'              # paddings in the three directions of current process of component 1
        - in case of vector-valued spaces:

            - pads2: 'int[:]'              
            - pads3: 'int[:]'              

    - then, the data objects (number depends on model, but at least one matrix has to be passed)
        - mat11: 'float[:,:,:,:,:,:]',    # _data attribute of StencilMatrix
        - optional:

            - mat12: 'float[:,:,:,:,:,:]',
            - mat13: 'float[:,:,:,:,:,:]',
            - mat21: 'float[:,:,:,:,:,:]',
            - mat22: 'float[:,:,:,:,:,:]',
            - mat23: 'float[:,:,:,:,:,:]',
            - mat31: 'float[:,:,:,:,:,:]',
            - mat32: 'float[:,:,:,:,:,:]',
            - mat33: 'float[:,:,:,:,:,:]',
            - vec1: 'float[:,:,:]',           # _data attribute of StencilVector
            - vec2: 'float[:,:,:]',
            - vec3: 'float[:,:,:]'

    - optional: additional parameters, for example
        - b2_1: 'float[:,:,:]',           # spline coefficients of b2_1
        - b2_2: 'float[:,:,:]',           # spline coefficients of b2_2
        - b2_3: 'float[:,:,:]'            # spline coefficients of b2_3
        - f0_params: 'float[:]',          # parameters of equilibrium background
    '''

    print('This is just the docstring function.')


def linear_vlasov_maxwell(markers: 'float[:,:]', np: 'int',
                          p_n: 'int[:]', t1: 'float[:]', t2: 'float[:]', t3: 'float[:]',
                          kind_map: 'int', params_map: 'float[:]',
                          p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                          cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                          starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]',
                          ends1: 'int[:]', ends2: 'int[:]', ends3: 'int[:]',
                          pads1: 'int[:]', pads2: 'int[:]', pads3: 'int[:]',
                          mat11: 'float[:,:,:,:,:,:]',
                          mat12: 'float[:,:,:,:,:,:]',
                          mat13: 'float[:,:,:,:,:,:]',
                          mat21: 'float[:,:,:,:,:,:]',
                          mat22: 'float[:,:,:,:,:,:]',
                          mat23: 'float[:,:,:,:,:,:]',
                          mat31: 'float[:,:,:,:,:,:]',
                          mat32: 'float[:,:,:,:,:,:]',
                          mat33: 'float[:,:,:,:,:,:]',
                          vec1: 'float[:,:,:]',
                          vec2: 'float[:,:,:]',
                          vec3: 'float[:,:,:]',
                          f0_params: 'float[:]'):
    """
    Accumulates into V1 with the filling functions

    .. math::

        A_p^{\mu, \\nu} &= f_0(\eta_p, v_p) * [ G^{-1}(\eta_p) * v_p ]_\mu * [ DF^{-1}(\eta_p) * v_p ]_\\nu    

        B_p^\mu &= \sqrt{f_0(\eta_p, v_p)} * w_p * [ G^{-1}(\eta_p) * v_p ]_\mu                  
    """

    # allocate for metric coeffs
    df = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)
    fx = empty(3, dtype=float)
    Gv = empty(3, dtype=float)
    Dv = empty(3, dtype=float)

    # $ omp parallel private (ip, eta1, eta2, eta3, df, fx, df_inv, g_inv, Gv, Dv, v, v1, v2, v3, weight, f0, filling_m11, filling_m12, filling_m13, filling_m21, filling_m22, filling_m23, filling_m31, filling_m32, filling_m33, filling_v1, filling_v2, filling_v3)
    # $ omp for reduction ( + :mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32, mat33, vec1, vec2, vec3)
    for ip in range(np):

        # position
        eta1 = markers[0, ip]
        eta2 = markers[1, ip]
        eta3 = markers[2, ip]

        # velocity
        v1 = markers[3, ip]
        v2 = markers[4, ip]
        v3 = markers[5, ip]

        weight = markers[6, ip]

        # background solution
        # TODO: possibility to choose from user-added background function (pyccelized)
        f0 = background_sol.maxwellian_point(
            eta1, eta2, eta3, v1, v2, v3, f0_params)

        # TODO: which fast mapping routine to use?
        mapping_fast.dl_all(kind_map, params_map, t1, t2, t3, p, cx,
                            cy, cz, indN1, indN2, indN3, eta1, eta2, eta3, df, fx, 0)

        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, df_inv)

        # evaluate inverse metric tensor
        mapping_fast.g_inv_all(df_inv, g_inv)

        Gv[0] = g_inv[0, 0] * v1 + g_inv[0, 1] * v2 + g_inv[0, 2] * v3
        Gv[1] = g_inv[1, 0] * v1 + g_inv[1, 1] * v2 + g_inv[1, 2] * v3
        Gv[2] = g_inv[2, 0] * v1 + g_inv[2, 1] * v2 + g_inv[2, 2] * v3

        Dv[0] = df_inv[0, 0] * v1 + df_inv[0, 1] * v2 + df_inv[0, 2] * v3
        Dv[1] = df_inv[1, 0] * v1 + df_inv[1, 1] * v2 + df_inv[1, 2] * v3
        Dv[2] = df_inv[2, 0] * v1 + df_inv[2, 1] * v2 + df_inv[2, 2] * v3

        # Compute filling for matrix: f_0(x,v) * ( G^{-1} v )_mu ( DL^{-1} v )_nu
        filling_m11 = f0 * Gv[0] * Dv[0]
        filling_m12 = f0 * Gv[0] * Dv[1]
        filling_m13 = f0 * Gv[0] * Dv[2]
        filling_m21 = f0 * Gv[1] * Dv[0]
        filling_m22 = f0 * Gv[1] * Dv[1]
        filling_m23 = f0 * Gv[1] * Dv[2]
        filling_m31 = f0 * Gv[2] * Dv[0]
        filling_m32 = f0 * Gv[2] * Dv[1]
        filling_m33 = f0 * Gv[2] * Dv[2]

        # Compute filling for vector: sqrt(f_0(x,v)) * w_p * ( G^{-1} v )
        filling_v1 = sqrt(f0) * weight * Gv[0]
        filling_v2 = sqrt(f0) * weight * Gv[1]
        filling_v3 = sqrt(f0) * weight * Gv[2]

        # fill matrix and vector
        # TODO: do we need the indN arrays?
        mvf.m_v_fill_b_v1_full(p, t1, t2, t3, indN1, indN2, indN3, eta1, eta2, eta3, mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32, mat33, filling_m11,
                               filling_m12, filling_m13, filling_m21, filling_m22, filling_m23, filling_m31, filling_m32, filling_m33, vec1, vec2, vec3, filling_v1, filling_v2, filling_v3)
    # $ omp end parallel


def cc_lin_mhd_6d_1(markers: 'float[:,:]', np: 'int',
                    p_n: 'int[:]', t1: 'float[:]', t2: 'float[:]', t3: 'float[:]',
                    kind_map: 'int', params_map: 'float[:]',
                    p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                    cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                    starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]',
                    ends1: 'int[:]', ends2: 'int[:]', ends3: 'int[:]',
                    pads1: 'int[:]', pads2: 'int[:]', pads3: 'int[:]',
                    mat12: 'float[:,:,:,:,:,:]',
                    mat13: 'float[:,:,:,:,:,:]',
                    mat23: 'float[:,:,:,:,:,:]',
                    vec1: 'float[:,:,:]',
                    vec2: 'float[:,:,:]',
                    vec3: 'float[:,:,:]',
                    b2_1: 'float[:,:,:]',
                    b2_2: 'float[:,:,:]',
                    b2_3: 'float[:,:,:]'):
    '''Accumulates into V1 with the filling functions

    .. math::

        A_p^{\mu, \\nu} = w_p * [ G^{-1}(\eta_p) * B2_{\\times}(\eta_p) * G^{-1}(\eta_p) ]_{\mu, \\nu}     

    where :math:`B2_{\\times} * a := B2 \\times a` for :math:`a \in \mathbb R^3`. 
    '''

    # magnetic field at particle position
    b = empty(3, dtype=float)
    b_prod = zeros((3, 3), dtype=float)

    # ================ for mapping evaluation ==================
    # spline degrees
    pf1 = p_map[0]
    pf2 = p_map[1]
    pf3 = p_map[2]

    # pf + 1 non-vanishing basis functions up tp degree pf
    b1f = empty((pf1 + 1, pf1 + 1), dtype=float)
    b2f = empty((pf2 + 1, pf2 + 1), dtype=float)
    b3f = empty((pf3 + 1, pf3 + 1), dtype=float)

    # left and right values for spline evaluation
    l1f = empty(pf1, dtype=float)
    l2f = empty(pf2, dtype=float)
    l3f = empty(pf3, dtype=float)

    r1f = empty(pf1, dtype=float)
    r2f = empty(pf2, dtype=float)
    r3f = empty(pf3, dtype=float)

    # scaling arrays for M-splines
    d1f = empty(pf1, dtype=float)
    d2f = empty(pf2, dtype=float)
    d3f = empty(pf3, dtype=float)

    # pf + 1 derivatives
    der1f = empty(pf1 + 1, dtype=float)
    der2f = empty(pf2 + 1, dtype=float)
    der3f = empty(pf3 + 1, dtype=float)

    # needed mapping quantities
    df = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    ginv = empty((3, 3), dtype=float)
    fx = empty(3, dtype=float)

    temp_mat1 = empty((3, 3), dtype=float)
    temp_mat2 = empty((3, 3), dtype=float)

    # $ omp parallel private(ip, eta1, eta2, eta3, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, det_df, dfinv, ginv, span1, span2, span3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, bn1, bn2, bn3, bd1, bd2, bd3, b, ie1, ie2, ie3, temp_mat1, temp_mat2, w_over_det2, temp12, temp13, temp23, il1, il2, il3, jl1, jl2, jl3, i1, i2, i3, bi1, bi2, bi3, bj1, bj2, bj3) firstprivate(b_prod)
    # $ omp for reduction ( + : mat12, mat13, mat23)
    for ip in range(np):

        # TODO: set marker to -1 when lost
        # only do something if particle is inside the logical domain (s < 1)
        # if markers[0, ip] > 1.0:
        #     continue

        eta1 = markers[0, ip]
        eta2 = markers[1, ip]
        eta3 = markers[2, ip]

        # ========= mapping evaluation =============
        span1f = int(eta1*nelf[0]) + pf1
        span2f = int(eta2*nelf[1]) + pf2
        span3f = int(eta3*nelf[2]) + pf3

        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f,
                            l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta1, eta2, eta3, df, fx, 0)

        # evaluate Jacobian determinant
        det_df = abs(linalg.det(df))

        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)

        # evaluate inverse metric tensor
        mapping_fast.g_inv_all(dfinv, ginv)
        # ==========================================

        # ========== field evaluation ==============
        span1 = int(eta1*nel[0]) + pn1
        span2 = int(eta2*nel[1]) + pn2
        span3 = int(eta3*nel[2]) + pn3

        bsp.basis_funs_all(t1, pn1, eta1, span1, l1, r1, b1, d1)
        bsp.basis_funs_all(t2, pn2, eta2, span2, l2, r2, b2, d2)
        bsp.basis_funs_all(t3, pn3, eta3, span3, l3, r3, b3, d3)

        # N-splines and D-splines at particle positions
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        bn3[:] = b3[pn3, :]

        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        bd3[:] = b3[pd3, :pn3] * d3[:]

        b[0] = eva3.evaluation_kernel_3d(
            pn1, pd2, pd3, bn1, bd2, bd3, span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], b2_1)
        b[1] = eva3.evaluation_kernel_3d(
            pd1, pn2, pd3, bd1, bn2, bd3, span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], b2_2)
        b[2] = eva3.evaluation_kernel_3d(
            pd1, pd2, pn3, bd1, bd2, bn3, span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], b2_3)

        b_prod[0, 1] = -b[2]
        b_prod[0, 2] = b[1]

        b_prod[1, 0] = b[2]
        b_prod[1, 2] = -b[0]

        b_prod[2, 0] = -b[1]
        b_prod[2, 1] = b[0]
        # ==========================================

        # ========= charge accumulation ============
        # element indices
        ie1 = span1 - pn1
        ie2 = span2 - pn2
        ie3 = span3 - pn3

        mvf.mat_fill_v1_asym(...)


def cc_lin_mhd_6d_2(markers: 'float[:,:]', np: 'int',
                    p_n: 'int[:]', t1: 'float[:]', t2: 'float[:]', t3: 'float[:]',
                    kind_map: 'int', params_map: 'float[:]',
                    p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                    cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                    starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]',
                    ends1: 'int[:]', ends2: 'int[:]', ends3: 'int[:]',
                    pads1: 'int[:]', pads2: 'int[:]', pads3: 'int[:]',
                    mat11: 'float[:,:,:,:,:,:]',
                    mat12: 'float[:,:,:,:,:,:]',
                    mat13: 'float[:,:,:,:,:,:]',
                    mat22: 'float[:,:,:,:,:,:]',
                    mat23: 'float[:,:,:,:,:,:]',
                    mat33: 'float[:,:,:,:,:,:]',
                    vec1: 'float[:,:,:]',
                    vec2: 'float[:,:,:]',
                    vec3: 'float[:,:,:]',
                    b2_1: 'float[:,:,:]',
                    b2_2: 'float[:,:,:]',
                    b2_3: 'float[:,:,:]'):
    '''Accumulates into V1 with the filling functions

    .. math::

        A_p^{\mu, \\nu} &= w_p * [ G^{-1}(\eta_p) * B2_{\\times}(\eta_p) * G^{-1}(\eta_p) * B2_{\\times}(\eta_p)^\\top * G^{-1}(\eta_p) ]_{\mu, \\nu}  

        B_p^\mu &= w_p * [ G^{-1}(\eta_p) * B2_{\\times}(\eta_p) * DF^{-1}(\eta_p) * v_p ]_\mu   

    where :math:`B2_{\\times} * a := B2 \\times a` for :math:`a \in \mathbb R^3`.                             
    '''

    # magnetic field at particle position
    b = empty(3, dtype=float)
    b_prod = zeros((3, 3), dtype=float)
    b_prod_t = zeros((3, 3), dtype=float)

    # ================ for mapping evaluation ==================
    # spline degrees
    pf1 = pf[0]
    pf2 = pf[1]
    pf3 = pf[2]

    # pf + 1 non-vanishing basis functions up tp degree pf
    b1f = empty((pf1 + 1, pf1 + 1), dtype=float)
    b2f = empty((pf2 + 1, pf2 + 1), dtype=float)
    b3f = empty((pf3 + 1, pf3 + 1), dtype=float)

    # left and right values for spline evaluation
    l1f = empty(pf1, dtype=float)
    l2f = empty(pf2, dtype=float)
    l3f = empty(pf3, dtype=float)

    r1f = empty(pf1, dtype=float)
    r2f = empty(pf2, dtype=float)
    r3f = empty(pf3, dtype=float)

    # scaling arrays for M-splines
    d1f = empty(pf1, dtype=float)
    d2f = empty(pf2, dtype=float)
    d3f = empty(pf3, dtype=float)

    # pf + 1 derivatives
    der1f = empty(pf1 + 1, dtype=float)
    der2f = empty(pf2 + 1, dtype=float)
    der3f = empty(pf3 + 1, dtype=float)

    # needed mapping quantities
    df = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    ginv = empty((3, 3), dtype=float)
    fx = empty(3, dtype=float)

    temp_mat1 = empty((3, 3), dtype=float)
    temp_mat2 = empty((3, 3), dtype=float)

    temp_mat_vec = empty((3, 3), dtype=float)

    temp_vec = empty(3, dtype=float)

    # particle velocity
    v = empty(3, dtype=float)

    # $ omp parallel private(ip, eta1, eta2, eta3, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, det_df, dfinv, ginv, span1, span2, span3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, bn1, bn2, bn3, bd1, bd2, bd3, b, b_prod_t, ie1, ie2, ie3, v, temp_mat_vec, temp_mat1, temp_mat2, temp_vec, w_over_det1, w_over_det2, temp11, temp12, temp13, temp22, temp23, temp33, temp1, temp2, temp3, il1, il2, il3, jl1, jl2, jl3, i1, i2, i3, bi1, bi2, bi3, bj1, bj2, bj3) firstprivate(b_prod)
    # $ omp for reduction ( + : mat11, mat12, mat13, mat22, mat23, mat33, vec1, vec2, vec3)
    for ip in range(np):

        # only do something if particle is inside the logical domain (s < 1)
        if markers[0, ip] > 1.0:
            continue

        eta1 = markers[0, ip]
        eta2 = markers[1, ip]
        eta3 = markers[2, ip]

        # ========= mapping evaluation =============
        span1f = int(eta1*nelf[0]) + pf1
        span2f = int(eta2*nelf[1]) + pf2
        span3f = int(eta3*nelf[2]) + pf3

        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f,
                            l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta1, eta2, eta3, df, fx, 0)

        # evaluate Jacobian determinant
        det_df = abs(linalg.det(df))

        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)

        # evaluate inverse metric tensor
        mapping_fast.g_inv_all(dfinv, ginv)
        # ==========================================

        # ========== field evaluation ==============
        span1 = int(eta1*nel[0]) + pn1
        span2 = int(eta2*nel[1]) + pn2
        span3 = int(eta3*nel[2]) + pn3

        bsp.basis_funs_all(t1, pn1, eta1, span1, l1, r1, b1, d1)
        bsp.basis_funs_all(t2, pn2, eta2, span2, l2, r2, b2, d2)
        bsp.basis_funs_all(t3, pn3, eta3, span3, l3, r3, b3, d3)

        # N-splines and D-splines at particle positions
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        bn3[:] = b3[pn3, :]

        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        bd3[:] = b3[pd3, :pn3] * d3[:]

        b[0] = eva3.evaluation_kernel_3d(
            pn1, pd2, pd3, bn1, bd2, bd3, span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], b2_1)
        b[1] = eva3.evaluation_kernel_3d(
            pd1, pn2, pd3, bd1, bn2, bd3, span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], b2_2)
        b[2] = eva3.evaluation_kernel_3d(
            pd1, pd2, pn3, bd1, bd2, bn3, span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], b2_3)

        b_prod[0, 1] = -b[2]
        b_prod[0, 2] = b[1]

        b_prod[1, 0] = b[2]
        b_prod[1, 2] = -b[0]

        b_prod[2, 0] = -b[1]
        b_prod[2, 1] = b[0]

        linalg.transpose(b_prod, b_prod_t)
        # ==========================================

        # ========= current accumulation ===========
        # element indices
        ie1 = span1 - pn1
        ie2 = span2 - pn2
        ie3 = span3 - pn3

        # particle velocity
        v[:] = markers[3:6, ip]

        mvf.m_v_fill_v1_symm(...)
