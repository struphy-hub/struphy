from numpy import zeros, empty, sqrt

from struphy.geometry import mappings_3d
from struphy.feec import bsplines_kernels as bsp
from struphy.kinetic_equil.analytical import background_sol

import struphy.pic.mat_vec_filler as mvf
import struphy.linear_algebra.core as linalg


def _docstring():
    '''
    MODULE DOCSTRING for **struphy.pic.accum_kernels**.

    The module contains model-specific accumulation routines (pyccelized), to be defined by the user.

    Naming conventions:
        - use the model name, all lower-case letters (e.g. lin_vlasov_maxwell)
        - in case of multiple accumulations in one model, attach _1, _2, etc. 

    Arguments have to be passed in the following order (copy and paste from existing accum. function):

    - First, the marker info:
        - markers: 'float[:,:]',          # positions [0:3,], velocities [3:6,], and weights [6,] of the markers
        - n_markers: 'int'                # number of markers

    - then, the Derham spline bases info:
        - pn: 'int[:]',                   # N-spline degree in each direction
        - tn1: 'float[:]',                # N-spline knot vector 
        - tn2: 'float[:]',
        - tn3: 'float[:]',                
        - ind_n1: int[:,:],               # Indices of non-vanishing N-splines in format (number of elements, pn + 1)       
        - ind_n2: int[:,:], 
        - ind_n3: int[:,:],
        - ind_d1: int[:,:],               # Indices of non-vanishing D-splines in format (number of elements, pn)
        - ind_d2: int[:,:],
        - ind_d3: int[:,:],

    - then, the mapping info:
        - kind_map: 'int',                # mapping identifier 
        - params_map: 'float[:]',         # mapping parameters
        - p_map: 'int[:]',                # spline degree
        - t1_map: 'float[:]',             # knot vector 
        - t2_map: 'float[:]',             
        - t3_map: 'float[:]', 
        - ind1_map: int[:,:],             # Indices of non-vanishing splines in format (number of mapping grid cells, p_map + 1)       
        - ind2_map: int[:,:], 
        - ind3_map: int[:,:],            
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


def linear_vlasov_maxwell(markers: 'float[:,:]', n_markers: 'int',
                          pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                          ind_n1: 'int[:,:]', ind_n2: 'int[:,:]', ind_n3: 'int[:,:]',
                          ind_d1: 'int[:,:]', ind_d2: 'int[:,:]', ind_d3: 'int[:,:]',
                          kind_map: 'int', params_map: 'float[:]',
                          p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                          ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
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

    # ================ for mapping evaluation ==================
    # spline degrees
    p1_map = p_map[0]
    p2_map = p_map[1]
    p3_map = p_map[2]

    # pf + 1 non-vanishing basis functions up tp degree pf
    b1_map = empty((p1_map + 1, p1_map + 1), dtype=float)
    b2_map = empty((p2_map + 1, p2_map + 1), dtype=float)
    b3_map = empty((p3_map + 1, p3_map + 1), dtype=float)

    # left and right values for spline evaluation
    l1_map = empty(p1_map, dtype=float)
    l2_map = empty(p2_map, dtype=float)
    l3_map = empty(p3_map, dtype=float)

    r1_map = empty(p1_map, dtype=float)
    r2_map = empty(p2_map, dtype=float)
    r3_map = empty(p3_map, dtype=float)

    # scaling arrays for M-splines
    d1_map = empty(p1_map, dtype=float)
    d2_map = empty(p2_map, dtype=float)
    d3_map = empty(p3_map, dtype=float)

    # pf + 1 derivatives
    der1_map = empty(p1_map + 1, dtype=float)
    der2_map = empty(p2_map + 1, dtype=float)
    der3_map = empty(p3_map + 1, dtype=float)

    # allocate for metric coeffs
    f = empty(3, dtype=float)
    df = empty((3, 3), dtype=float)
    df_t = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    g = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)
    g_inv_times_v = empty(3, dtype=float)
    df_inv_times_v = empty(3, dtype=float)

    # $ omp parallel private (ip, eta1, eta2, eta3, df, fx, df_inv, g_inv, Gv, Dv, v, v1, v2, v3, weight, f0, filling_m11, filling_m12, filling_m13, filling_m21, filling_m22, filling_m23, filling_m31, filling_m32, filling_m33, filling_v1, filling_v2, filling_v3)
    # $ omp for reduction ( + :mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32, mat33, vec1, vec2, vec3)
    for ip in range(n_markers):

        # marker data
        eta1 = markers[0, ip]
        eta2 = markers[1, ip]
        eta3 = markers[2, ip]
        v = markers[3:6, ip]
        weight = markers[6, ip]

        # background solution
        # TODO: possibility to choose from user-added background function (pyccelized)
        f0 = background_sol.maxwellian_point(
            eta1, eta2, eta3, v[0], v[1], v[2], f0_params)

        # mapping spans
        span1_map = bsp.find_span(t1_map, p1_map, eta1)
        span2_map = bsp.find_span(t2_map, p2_map, eta2)
        span3_map = bsp.find_span(t3_map, p3_map, eta3)

        # evaluate Jacobian, result in df
        mappings_3d.f_df_pic(kind_map, params_map,
                             t1_map, t2_map, t3_map, p_map,
                             span1_map, span2_map, span3_map,
                             ind1_map, ind2_map, ind3_map,
                             cx, cy, cz,
                             l1_map, l2_map, l3_map,
                             r1_map, r2_map, r3_map,
                             b1_map, b2_map, b3_map,
                             d1_map, d2_map, d3_map,
                             der1_map, der2_map, der3_map,
                             eta1, eta2, eta3,
                             f, df, 1)

        # evaluate inverse Jacobian matrix
        linalg.matrix_inv(df, df_inv)

        # evaluate inverse metric tensor, result in g_inv
        linalg.transpose(df, df_t)
        linalg.matrix_matrix(df, df_t, g)
        linalg.matrix_inv(g, g_inv)

        # filling functions
        linalg.matrix_vector(g_inv, v, g_inv_times_v)
        linalg.matrix_vector(df_inv, v, df_inv_times_v)

        filling_m11 = f0 * g_inv_times_v[0] * df_inv_times_v[0]
        filling_m12 = f0 * g_inv_times_v[0] * df_inv_times_v[1]
        filling_m13 = f0 * g_inv_times_v[0] * df_inv_times_v[2]
        filling_m21 = f0 * g_inv_times_v[1] * df_inv_times_v[0]
        filling_m22 = f0 * g_inv_times_v[1] * df_inv_times_v[1]
        filling_m23 = f0 * g_inv_times_v[1] * df_inv_times_v[2]
        filling_m31 = f0 * g_inv_times_v[2] * df_inv_times_v[0]
        filling_m32 = f0 * g_inv_times_v[2] * df_inv_times_v[1]
        filling_m33 = f0 * g_inv_times_v[2] * df_inv_times_v[2]

        filling_v1 = sqrt(f0) * weight * g_inv_times_v[0]
        filling_v2 = sqrt(f0) * weight * g_inv_times_v[1]
        filling_v3 = sqrt(f0) * weight * g_inv_times_v[2]

        # call the appropriate matvec filler
        mvf.m_v_fill_b_v1_full(pn, tn1, tn2, tn3,
                               start1, start2, start3,
                               pad1, pad2, pad3,
                               eta1, eta2, eta3,
                               mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32, mat33,
                               filling_m11, filling_m12, filling_m13,
                               filling_m21, filling_m22, filling_m23,
                               filling_m31, filling_m32, filling_m33,
                               vec1, vec2, vec3,
                               filling_v1, filling_v2, filling_v3)
    # $ omp end parallel


def cc_lin_mhd_6d_1(markers: 'float[:,:]', n_markers: 'int',
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
    p1_map = p_map[0]
    p2_map = p_map[1]
    p3_map = p_map[2]

    # pf + 1 non-vanishing basis functions up tp degree pf
    b1f = empty((p1_map + 1, p1_map + 1), dtype=float)
    b2f = empty((p2_map + 1, p2_map + 1), dtype=float)
    b3f = empty((p3_map + 1, p3_map + 1), dtype=float)

    # left and right values for spline evaluation
    l1f = empty(p1_map, dtype=float)
    l2f = empty(p2_map, dtype=float)
    l3f = empty(p3_map, dtype=float)

    r1f = empty(p1_map, dtype=float)
    r2f = empty(p2_map, dtype=float)
    r3f = empty(p3_map, dtype=float)

    # scaling arrays for M-splines
    d1f = empty(p1_map, dtype=float)
    d2f = empty(p2_map, dtype=float)
    d3f = empty(p3_map, dtype=float)

    # pf + 1 derivatives
    der1f = empty(p1_map + 1, dtype=float)
    der2f = empty(p2_map + 1, dtype=float)
    der3f = empty(p3_map + 1, dtype=float)

    # needed mapping quantities
    df = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    ginv = empty((3, 3), dtype=float)
    fx = empty(3, dtype=float)

    temp_mat1 = empty((3, 3), dtype=float)
    temp_mat2 = empty((3, 3), dtype=float)

    # $ omp parallel private(ip, eta1, eta2, eta3, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, det_df, dfinv, ginv, span1, span2, span3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, bn1, bn2, bn3, bd1, bd2, bd3, b, ie1, ie2, ie3, temp_mat1, temp_mat2, w_over_det2, temp12, temp13, temp23, il1, il2, il3, jl1, jl2, jl3, i1, i2, i3, bi1, bi2, bi3, bj1, bj2, bj3) firstprivate(b_prod)
    # $ omp for reduction ( + : mat12, mat13, mat23)
    for ip in range(n_markers):

        # TODO: set marker to -1 when lost
        # only do something if particle is inside the logical domain (s < 1)
        # if markers[0, ip] > 1.0:
        #     continue

        eta1 = markers[0, ip]
        eta2 = markers[1, ip]
        eta3 = markers[2, ip]

        # ========= mapping evaluation =============
        span1f = int(eta1*nelf[0]) + p1_map
        span2f = int(eta2*nelf[1]) + p2_map
        span3f = int(eta3*nelf[2]) + p3_map

        # evaluate Jacobian matrix
        mappings_3d.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f,
                           l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta1, eta2, eta3, df, fx, 0)

        # evaluate Jacobian determinant
        det_df = abs(linalg.det(df))

        # evaluate inverse Jacobian matrix
        mappings_3d.df_inv_all(df, dfinv)

        # evaluate inverse metric tensor
        mappings_3d.g_inv_all(dfinv, ginv)
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


def cc_lin_mhd_6d_2(markers: 'float[:,:]', n_markers: 'int',
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
    p1_map = pf[0]
    p2_map = pf[1]
    p3_map = pf[2]

    # pf + 1 non-vanishing basis functions up tp degree pf
    b1f = empty((p1_map + 1, p1_map + 1), dtype=float)
    b2f = empty((p2_map + 1, p2_map + 1), dtype=float)
    b3f = empty((p3_map + 1, p3_map + 1), dtype=float)

    # left and right values for spline evaluation
    l1f = empty(p1_map, dtype=float)
    l2f = empty(p2_map, dtype=float)
    l3f = empty(p3_map, dtype=float)

    r1f = empty(p1_map, dtype=float)
    r2f = empty(p2_map, dtype=float)
    r3f = empty(p3_map, dtype=float)

    # scaling arrays for M-splines
    d1f = empty(p1_map, dtype=float)
    d2f = empty(p2_map, dtype=float)
    d3f = empty(p3_map, dtype=float)

    # pf + 1 derivatives
    der1f = empty(p1_map + 1, dtype=float)
    der2f = empty(p2_map + 1, dtype=float)
    der3f = empty(p3_map + 1, dtype=float)

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
    for ip in range(n_markers):

        # only do something if particle is inside the logical domain (s < 1)
        if markers[0, ip] > 1.0:
            continue

        eta1 = markers[0, ip]
        eta2 = markers[1, ip]
        eta3 = markers[2, ip]

        # ========= mapping evaluation =============
        span1f = int(eta1*nelf[0]) + p1_map
        span2f = int(eta2*nelf[1]) + p2_map
        span3f = int(eta3*nelf[2]) + p3_map

        # evaluate Jacobian matrix
        mappings_3d.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f,
                           l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta1, eta2, eta3, df, fx, 0)

        # evaluate Jacobian determinant
        det_df = abs(linalg.det(df))

        # evaluate inverse Jacobian matrix
        mappings_3d.df_inv_all(df, dfinv)

        # evaluate inverse metric tensor
        mappings_3d.g_inv_all(dfinv, ginv)
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
