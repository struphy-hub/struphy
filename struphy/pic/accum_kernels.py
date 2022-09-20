from numpy import zeros, empty, sqrt

import struphy.geometry.map_eval as map_eval
import struphy.feec.bsplines_kernels as bsp
import struphy.feec.basics.spline_evaluation_3d as eval_3d
import struphy.kinetic_background.background_eval as background_eval

import struphy.pic.mat_vec_filler as mvf
import struphy.linear_algebra.core as linalg


def _docstring():
    '''
    MODULE DOCSTRING for :ref:`accumulators`.

    The module contains model-specific accumulation routines (pyccelized), to be defined by the user.

    Naming conventions:
        - use the model name, all lower-case letters (e.g. lin_vlasov_maxwell)
        - in case of multiple accumulations in one model, attach _1, _2, etc. 

    Arguments have to be passed in the following order (copy and paste from existing accum_kernels function):

    - First, the marker info:
        - markers: 'float[:,:]',          # positions [0:3,], velocities [3:6,], and weights [6,] of the markers
        - n_markers: 'int'                # number of markers

    - then, the Derham spline bases info:
        - pn: 'int[:]',                   # N-spline degree in each direction
        - tn1: 'float[:]',                # N-spline knot vector 
        - tn2: 'float[:]',
        - tn3: 'float[:]',    

    - then, the mpi.comm info of the accumulation space:
        - starts1: 'int[:]'              # start indices in the three directions of current process of component 1
        - in case of vector-valued spaces:

            - starts2: 'int[:]'              
            - starts3: 'int[:]'             

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
                          starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]',
                          kind_map: 'int', params_map: 'float[:]',
                          p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                          ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                          cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
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
                          f0_spec: 'int',  # model specific arguments
                          moms_spec: 'int[:]',  # model specific arguments
                          f0_params: 'float[:]'):  # model specific arguments
    r"""
    Accumulates into V1 with the filling functions

    .. math::

        A_p^{\mu, \nu} &= f_0(\eta_p, v_p) * [ G^{-1}(\eta_p) * v_p ]_\mu * [ DF^{-1}(\eta_p) * v_p ]_\nu    

        B_p^\mu &= \sqrt{f_0(\eta_p, v_p)} * w_p * [ G^{-1}(\eta_p) * v_p ]_\mu  

    Parameters
    ----------
        f0_spec : int
            Specifier for kinetic background, see :ref:`kinetic_backgrounds`  

        moms_spec : array[int]
            Specifier for the seven moments n0, u0x, u0y, u0z, vth0x, vth0y, vth0z (in this order).
            Is 0 for constant moment, for more see :meth:`struphy.kinetic_background.moments_kernels.moments`.

        f0_params : array[float]
            Parameters needed to specify the moments; the order is specified in :ref:`kinetic_moments` for the respective functions available.

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    # allocate for metric coeffs
    df = empty((3, 3), dtype=float)
    df_t = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    g = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # allocate for filling
    g_inv_times_v = empty(3, dtype=float)
    df_inv_times_v = empty(3, dtype=float)
    filling_m = empty((3, 3), dtype=float)
    filling_v = empty(3, dtype=float)

    #$ omp parallel private (ip, eta1, eta2, eta3, v, weight, f0, df, df_t, df_inv, g, g_inv, g_inv_times_v, df_inv_times_v, filling_m, filling_v)
    #$ omp for reduction ( + : mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32, mat33, vec1, vec2, vec3)
    for ip in range(n_markers):

        # marker data
        eta1 = markers[0, ip]
        eta2 = markers[1, ip]
        eta3 = markers[2, ip]
        v = markers[3:6, ip]
        weight = markers[6, ip]

        # evaluate background
        f0 = background_eval.f0(
            markers[:3, ip], v, f0_spec, moms_spec, f0_params)

        # evaluate Jacobian, result in df
        map_eval.df(eta1, eta2, eta3,
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # Avoid second computation of df, use linear_algebra.core routines to get g_inv:
        linalg.matrix_inv(df, df_inv)
        linalg.transpose(df, df_t)
        linalg.matrix_matrix(df, df_t, g)
        linalg.matrix_inv(g, g_inv)

        # filling functions
        linalg.matrix_vector(g_inv, v, g_inv_times_v)
        linalg.matrix_vector(df_inv, v, df_inv_times_v)

        linalg.outer(g_inv_times_v, df_inv_times_v, filling_m)
        filling_m[:] = f0 * filling_m
        filling_v[:] = sqrt(f0) * weight * g_inv_times_v

        # call the appropriate matvec filler
        mvf.m_v_fill_b_v1_full(pn, tn1, tn2, tn3,
                               starts1, starts2, starts3,
                               eta1, eta2, eta3,
                               mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32, mat33,
                               filling_m[0, 0], filling_m[0, 1], filling_m[0, 2],
                               filling_m[1, 0], filling_m[1, 1], filling_m[1, 2],
                               filling_m[2, 0], filling_m[2, 1], filling_m[2, 2],
                               vec1, vec2, vec3,
                               filling_v[0], filling_v[1], filling_v[2])
    #$ omp end parallel


def cc_lin_mhd_6d_1(markers: 'float[:,:]', n_markers: 'int',
                    pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                    kind_map: 'int', params_map: 'float[:]',
                    p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                    ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                    cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                    starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]',
                    mat12: 'float[:,:,:,:,:,:]',
                    mat13: 'float[:,:,:,:,:,:]',
                    mat23: 'float[:,:,:,:,:,:]',
                    b2_1: 'float[:,:,:]',  # model specific parameters
                    b2_2: 'float[:,:,:]',  # model specific parameters
                    b2_3: 'float[:,:,:]',  # model specific parameters
                    starts_21: 'int[:]',  # model specific parameters
                    starts_22: 'int[:]',  # model specific parameters
                    starts_23: 'int[:]'):  # model specific parameters
    r'''Accumulates into V1 with the filling functions

    .. math::

        A_p^{\mu, \nu} = w_p * [ G^{-1}(\eta_p) * B2_{\times}(\eta_p) * G^{-1}(\eta_p) ]_{\mu, \nu}     

    where :math:`B2_{\times} * a := B2 \times a` for :math:`a \in \mathbb R^3`. 

    Parameters
    ----------
        b2_1, b2_2, b2_3 : array[float]
            FE coefficients c_ijk of the magnetic field as a 2-form.  

        starts_21, starts_22, starts_23 : array[int]
            Start indices of 2-forms on current process.

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    '''

    # allocate for metric coeffs
    g_inv = empty((3, 3), dtype=float)

    # allocate for filling
    tmp1 = empty((3, 3), dtype=float)
    tmp2 = empty((3, 3), dtype=float)

    # allocate for magnetic field evaluation
    b = empty(3, dtype=float)
    b_prod = zeros((3, 3), dtype=float) 

    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    #$ omp parallel firstprivate(b_prod) private(ip, eta1, eta2, eta3, weight, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, b, g_inv, tmp1, tmp2, filling_m12, filling_m13, filling_m23) 
    #$ omp for reduction ( + : mat12, mat13, mat23)
    for ip in range(n_markers):

        # marker data
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        weight = markers[ip, 6]

        # b-field evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        b[0] = eval_3d.eval_spline_mpi_3d(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b2_1, starts_21, pn)
        b[1] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2_2, starts_22, pn)
        b[2] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b2_3, starts_23, pn)

        # operator bx() as matrix
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] = b[1]
        b_prod[1, 0] = b[2]
        b_prod[1, 2] = -b[0]
        b_prod[2, 0] = -b[1]
        b_prod[2, 1] = b[0]

        # evaluate inverse metric tensor, result in g_inv
        map_eval.g_inv(eta1, eta2, eta3,
                       kind_map, params_map,
                       t1_map, t2_map, t3_map, p_map,
                       ind1_map, ind2_map, ind3_map,
                       cx, cy, cz,
                       g_inv)

        # filling functions
        linalg.matrix_matrix(g_inv, b_prod, tmp1)
        linalg.matrix_matrix(tmp1, g_inv, tmp2)

        filling_m12 = - weight * tmp2[0, 1]
        filling_m13 = - weight * tmp2[0, 2]
        filling_m23 = - weight * tmp2[1, 2]

        # call the appropriate matvec filler
        mvf.mat_fill_v1_asym(pn, span1, span2, span3,
                             bn1, bn2, bn3,
                             bd1, bd2, bd3,
                             starts1, starts2, starts3,
                             mat12, mat13, mat23,
                             filling_m12, filling_m13, filling_m23)
    #$ omp end parallel


def cc_lin_mhd_6d_2(markers: 'float[:,:]', n_markers: 'int',
                    pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                    kind_map: 'int', params_map: 'float[:]',
                    p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                    ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                    cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                    starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]',
                    mat11: 'float[:,:,:,:,:,:]',
                    mat12: 'float[:,:,:,:,:,:]',
                    mat13: 'float[:,:,:,:,:,:]',
                    mat22: 'float[:,:,:,:,:,:]',
                    mat23: 'float[:,:,:,:,:,:]',
                    mat33: 'float[:,:,:,:,:,:]',
                    vec1: 'float[:,:,:]',
                    vec2: 'float[:,:,:]',
                    vec3: 'float[:,:,:]',
                    b2_1: 'float[:,:,:]',  # model specific parameters
                    b2_2: 'float[:,:,:]',  # model specific parameters
                    b2_3: 'float[:,:,:]',  # model specific parameters
                    starts_21: 'int[:]',  # model specific parameters
                    starts_22: 'int[:]',  # model specific parameters
                    starts_23: 'int[:]'):  # model specific parameters
    r'''Accumulates into V1 with the filling functions

    .. math::

        A_p^{\mu, \nu} &= w_p * [ G^{-1}(\eta_p) * B2_{\times}(\eta_p) * G^{-1}(\eta_p) * B2_{\times}(\eta_p)^\top * G^{-1}(\eta_p) ]_{\mu, \nu}

        B_p^\mu &= w_p * [ G^{-1}(\eta_p) * B2_{\times}(\eta_p) * DF^{-1}(\eta_p) * v_p ]_\mu

    where :math:`B2_{\times} * a := B2 \times a` for :math:`a \in \mathbb R^3`.

    Parameters
    ----------
        b2_1, b2_2, b2_3 : array[float]
            FE coefficients c_ijk of the magnetic field as a 2-form.  

        starts_21, starts_22, starts_23 : array[int]
            Start indices of 2-forms on current process.

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    '''

    # allocate for metric coeffs
    df = empty((3, 3), dtype=float)
    df_t = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    g = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # allocate for filling
    filling_m = empty((3, 3), dtype=float)
    filling_v = empty(3, dtype=float)

    tmp1 = empty((3, 3), dtype=float)
    tmp1_t = empty((3, 3), dtype=float)
    tmp2 = empty((3, 3), dtype=float)
    tmp3 = empty((3, 3), dtype=float)

    tmp_v = empty(3, dtype=float)
    df_inv_times_v = empty(3, dtype=float)

    # allocate for magnetic field evaluation
    b = empty(3, dtype=float)
    b_prod = zeros((3, 3), dtype=float)

    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    #$ omp parallel firstprivate(b_prod) private(ip, eta1, eta2, eta3, v, weight, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, b, df, df_inv, df_t, g, g_inv, tmp1, tmp1_t, tmp2, tmp3, tmp_v, df_inv_times_v, filling_m, filling_v) 
    #$ omp for reduction ( + : mat12, mat13, mat23)
    for ip in range(n_markers):

        # marker data
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        v = markers[ip, 3:6]
        weight = markers[ip, 6]

        # b-field evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        b[0] = eval_3d.eval_spline_mpi_3d(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b2_1, starts_21, pn)
        b[1] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2_2, starts_22, pn)
        b[2] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b2_3, starts_23, pn)

        # operator bx() as matrix
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] = b[1]
        b_prod[1, 0] = b[2]
        b_prod[1, 2] = -b[0]
        b_prod[2, 0] = -b[1]
        b_prod[2, 1] = b[0]

        # evaluate Jacobian, result in df
        map_eval.df(eta1, eta2, eta3,
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # Avoid second computation of df, use linear_algebra.core routines to get g_inv:
        linalg.matrix_inv(df, df_inv)
        linalg.transpose(df, df_t)
        linalg.matrix_matrix(df, df_t, g)
        linalg.matrix_inv(g, g_inv)

        # filling functions
        linalg.matrix_matrix(g_inv, b_prod, tmp1)
        linalg.transpose(tmp1, tmp1_t)
        linalg.matrix_matrix(tmp1, g_inv, tmp2)
        linalg.matrix_matrix(tmp2, tmp1_t, tmp3)

        linalg.matrix_vector(df_inv, v, df_inv_times_v)
        linalg.matrix_vector(tmp1, df_inv_times_v, tmp_v)

        filling_m[:] = weight * tmp3
        filling_v[:] = weight * tmp_v

        # call the appropriate matvec filler
        mvf.m_v_fill_v1_symm(pn, span1, span2, span3,
                             bn1, bn2, bn3,
                             bd1, bd2, bd3,
                             starts1, starts2, starts3,
                             mat11, mat12, mat13, 
                             mat22, mat23, 
                             mat33, 
                             filling_m[0, 0], filling_m[0, 1], filling_m[0, 2], 
                             filling_m[1, 1], filling_m[1, 2], 
                             filling_m[2, 2],
                             vec1, vec2, vec3,
                             filling_v[0], filling_v[1], filling_v[2])
    #$ omp end parallel


def pc_lin_mhd_6d(markers: 'float[:,:]', n_markers: 'int',
                  pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                  kind_map: 'int', params_map: 'float[:]',
                  p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                  ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                  cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                  starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]',
                  mat11_11: 'float[:,:,:,:,:,:]',
                  mat12_11: 'float[:,:,:,:,:,:]',
                  mat13_11: 'float[:,:,:,:,:,:]',
                  mat22_11: 'float[:,:,:,:,:,:]',
                  mat23_11: 'float[:,:,:,:,:,:]',
                  mat33_11: 'float[:,:,:,:,:,:]',
                  mat11_12: 'float[:,:,:,:,:,:]',
                  mat12_12: 'float[:,:,:,:,:,:]',
                  mat13_12: 'float[:,:,:,:,:,:]',
                  mat22_12: 'float[:,:,:,:,:,:]',
                  mat23_12: 'float[:,:,:,:,:,:]',
                  mat33_12: 'float[:,:,:,:,:,:]',
                  mat11_13: 'float[:,:,:,:,:,:]',
                  mat12_13: 'float[:,:,:,:,:,:]',
                  mat13_13: 'float[:,:,:,:,:,:]',
                  mat22_13: 'float[:,:,:,:,:,:]',
                  mat23_13: 'float[:,:,:,:,:,:]',
                  mat33_13: 'float[:,:,:,:,:,:]',
                  mat11_22: 'float[:,:,:,:,:,:]',
                  mat12_22: 'float[:,:,:,:,:,:]',
                  mat13_22: 'float[:,:,:,:,:,:]',
                  mat22_22: 'float[:,:,:,:,:,:]',
                  mat23_22: 'float[:,:,:,:,:,:]',
                  mat33_22: 'float[:,:,:,:,:,:]',
                  mat11_23: 'float[:,:,:,:,:,:]',
                  mat12_23: 'float[:,:,:,:,:,:]',
                  mat13_23: 'float[:,:,:,:,:,:]',
                  mat22_23: 'float[:,:,:,:,:,:]',
                  mat23_23: 'float[:,:,:,:,:,:]',
                  mat33_23: 'float[:,:,:,:,:,:]',
                  mat11_33: 'float[:,:,:,:,:,:]',
                  mat12_33: 'float[:,:,:,:,:,:]',
                  mat13_33: 'float[:,:,:,:,:,:]',
                  mat22_33: 'float[:,:,:,:,:,:]',
                  mat23_33: 'float[:,:,:,:,:,:]',
                  mat33_33: 'float[:,:,:,:,:,:]',
                  vec1_1: 'float[:,:,:]',
                  vec2_1: 'float[:,:,:]',
                  vec3_1: 'float[:,:,:]',
                  vec1_2: 'float[:,:,:]',
                  vec2_2: 'float[:,:,:]',
                  vec3_2: 'float[:,:,:]',
                  vec1_3: 'float[:,:,:]',
                  vec2_3: 'float[:,:,:]',
                  vec3_3: 'float[:,:,:]'):
    '''Accumulates into V1 with the filling functions

    .. math::

        V_{p,i} A_p^{\mu, \\nu} V_{p,j} &= w_p * [ DF^{-1}(\eta_p) DF^{-\\top}(\eta_p) ]_{\mu, \\nu} * V_{p,i} * V_{p,j} \,,
        
        V_{p,i} B_p^\mu &= w_p * [DF^{-1}(\eta_p) V_p]_\mu * V_{p,i} \,.

    Parameters
    ----------

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    '''

    # allocate for metric coeffs
    df = empty((3, 3), dtype=float)
    df_t = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)
    g = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # allocate for filling
    filling_m = empty((3, 3), dtype=float)
    filling_v = empty(3, dtype=float)

    tmp1 = empty((3, 3), dtype=float)
    tmp1_t = empty((3, 3), dtype=float)
    tmp2 = empty((3, 3), dtype=float)
    tmp3 = empty((3, 3), dtype=float)

    tmp_v = empty(3, dtype=float)
    df_inv_times_v = empty(3, dtype=float)

    # allocate for magnetic field evaluation
    b = empty(3, dtype=float)
    b_prod = zeros((3, 3), dtype=float)

    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    #$ omp parallel private(ip, eta1, eta2, eta3, v, weight, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, df, df_inv, df_t, df_inv_t, tmp1, tmp_v, filling_m, filling_v) 
    #$ omp for reduction ( + : mat11_11, mat11_12, mat11_13, mat11_22, mat11_23, mat11_33, mat12_11, mat12_12, mat12_13, mat12_22, mat12_23, mat12_33, mat13_11, mat13_12, mat13_13, mat13_22, mat13_23, mat13_33, mat22_11, mat22_12, mat22_13, mat22_22, mat22_23, mat22_33, mat23_11, mat23_12, mat23_13, mat23_22, mat23_23, mat23_33, mat33_11, mat33_12, mat33_13, mat33_22, mat33_23, mat33_33, vec1_1, vec1_2, vec1_3, vec2_1, vec2_2, vec2_3, vec3_1, vec3_2, vec3_3)
    for ip in range(n_markers):

        # marker data
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        v = markers[ip, 3:6]
        weight = markers[ip, 8]

        # b-field evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        # evaluate Jacobian, result in df
        map_eval.df(eta1, eta2, eta3,
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # Avoid second computation of df, use linear_algebra.core routines to get g_inv:
        linalg.matrix_inv(df, df_inv)
        linalg.transpose(df, df_t)
        linalg.transpose(df_inv, df_inv_t)

        # filling functions
        linalg.matrix_matrix(df_inv, df_inv_t, tmp1)
        linalg.matrix_vector(df_inv, v, tmp_v)

        filling_m[:] = weight * tmp1
        filling_v[:] = weight * tmp_v

        # call the appropriate matvec filler
        mvf.m_v_fill_v1_pressure(pn, span1, span2, span3,
                                 bn1, bn2, bn3,
                                 bd1, bd2, bd3,
                                 starts1, starts2, starts3,
                                 mat11_11, mat12_11, mat13_11, 
                                 mat22_11, mat23_11, 
                                 mat33_11,
                                 mat11_12, mat12_12, mat13_12, 
                                 mat22_12, mat23_12, 
                                 mat33_12,
                                 mat11_13, mat12_13, mat13_13, 
                                 mat22_13, mat23_13, 
                                 mat33_13,
                                 mat11_22, mat12_22, mat13_22, 
                                 mat22_22, mat23_22, 
                                 mat33_22,
                                 mat11_23, mat12_23, mat13_23, 
                                 mat22_23, mat23_23, 
                                 mat33_23,
                                 mat11_33, mat12_33, mat13_33, 
                                 mat22_33, mat23_33, 
                                 mat33_33, 
                                 filling_m[0, 0], filling_m[0, 1], filling_m[0, 2], 
                                 filling_m[1, 1], filling_m[1, 2], 
                                 filling_m[2, 2],
                                 vec1_1, vec2_1, vec3_1,
                                 vec1_2, vec2_2, vec3_2,
                                 vec1_3, vec2_3, vec3_3,
                                 filling_v[0], filling_v[1], filling_v[2],
                                 v[0], v[1], v[2])
    #$ omp end parallel