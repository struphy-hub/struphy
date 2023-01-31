from pyccel.decorators import stack_array

from numpy import zeros, empty, sqrt, shape, floor

import struphy.geometry.map_eval as map_eval
import struphy.b_splines.bsplines_kernels as bsp
import struphy.b_splines.bspline_evaluation_3d as eval_3d
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

    - then, the Derham spline bases info:
        - pn: 'int[:]',                   # N-spline degree in each direction
        - tn1: 'float[:]',                # N-spline knot vector 
        - tn2: 'float[:]',
        - tn3: 'float[:]',    

    - then, the mpi.comm info of all spaces:
        - starts0: 'int[:]'               # start indices of current process of elements in space V0
        - starts1: 'int[:,:]'             # start indices of current process of elements in space V1 in format (component, direction)
        - starts2: 'int[:,:]'             # start indices of current process of elements in space V2 in format (component, direction)
        - starts3: 'int[:]'               # start indices of current process of elements in space V3

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


@stack_array('cell_left', 'point_left', 'point_right', 'cell_number', 'temp1', 'temp4', 'compact', 'grids_shapex', 'grids_shapey', 'grids_shapez')
def hybrid_fA_density(markers: 'float[:,:]', n_markers_tot: 'int',
                          pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                          starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                          kind_map: 'int', params_map: 'float[:]',
                          p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                          ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                          cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                          mat: 'float[:,:,:,:,:,:]', Nel: 'int[:]', quad: 'int[:]', quad_pts_x: 'float[:]', quad_pts_y: 'float[:]', quad_pts_z: 'float[:]',
                          p_shape: 'int[:]', p_size: 'float[:]'):  # model specific argument
    r"""
    Accumulates the values of density at quadrature points with the filling functions

    .. math::
        n = \sum_p w_p S(x - x_p)

    Parameters
    ----------
        To do 
    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    # allocate
    cell_left    = empty(3, dtype=int)
    point_left   = zeros(3, dtype=float)
    point_right  = zeros(3, dtype=float)
    cell_number  = empty(3, dtype=int)

    temp1        = zeros(3, dtype=float)
    temp4        = zeros(3, dtype=float)

    compact      = zeros(3, dtype=float)
    compact[0]   = (p_shape[0]+1.0)*p_size[0]
    compact[1]   = (p_shape[1]+1.0)*p_size[1]
    compact[2]   = (p_shape[2]+1.0)*p_size[2]

    grids_shapex = zeros(p_shape[0] + 2, dtype=float)
    grids_shapey = zeros(p_shape[1] + 2, dtype=float)
    grids_shapez = zeros(p_shape[2] + 2, dtype=float)
    
    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private (cell_left, point_left, point_right, cell_number, temp1, temp4, compact, grids_shapex, grids_shapey, grids_shapez, n_markers, ip, eta1, eta2, eta3, weight, ie1, ie2, ie3, il1, il2, il3, jl1, jl2, jl3, i1, i2, i3, value_x, value_y, value_z, span1, span2, span3)
    #$ omp for reduction ( + : mat)
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        weight = markers[ip, 6]/(p_size[0]*p_size[1]*p_size[2])/n_markers_tot

        ie1 = int(eta1*Nel[0])
        ie2 = int(eta2*Nel[1])
        ie3 = int(eta3*Nel[2])

        #the points here are still not put in the periodic box [0, 1] x [0, 1] x [0, 1]
        point_left[0]  = eta1 - 0.5*compact[0]
        point_right[0] = eta1 + 0.5*compact[0]
        point_left[1]  = eta2 - 0.5*compact[1]
        point_right[1] = eta2 + 0.5*compact[1]
        point_left[2]  = eta3 - 0.5*compact[2]
        point_right[2] = eta3 + 0.5*compact[2]

        cell_left[0] = int(floor(point_left[0]*Nel[0]))
        cell_left[1] = int(floor(point_left[1]*Nel[1]))
        cell_left[2] = int(floor(point_left[2]*Nel[2]))

        cell_number[0] = int(floor(point_right[0]*Nel[0])) - cell_left[0] + 1
        cell_number[1] = int(floor(point_right[1]*Nel[1])) - cell_left[1] + 1
        cell_number[2] = int(floor(point_right[2]*Nel[2])) - cell_left[2] + 1

        for i in range(p_shape[0] + 1):
            grids_shapex[i] = point_left[0] + i * p_size[0]
        grids_shapex[p_shape[0] + 1] = point_right[0]

        for i in range(p_shape[1] + 1):
            grids_shapey[i] = point_left[1] + i * p_size[1]
        grids_shapey[p_shape[1] + 1] = point_right[1]

        for i in range(p_shape[2] + 1):
            grids_shapez[i] = point_left[2] + i * p_size[2]
        grids_shapez[p_shape[2] + 1] = point_right[2]

        span1 = int(eta1*Nel[0]) + pn[0]
        span2 = int(eta2*Nel[1]) + pn[1]
        span3 = int(eta3*Nel[2]) + pn[2]

        # =========== kernel part (periodic bundary case) ==========
        mvf.hybrid_density(Nel, pn, cell_left, cell_number, span1, span2, span3, starts0, ie1, ie2, ie3, temp1, temp4, quad, quad_pts_x, quad_pts_y, quad_pts_z, compact, eta1, eta2, eta3, mat, weight, p_shape, p_size, grids_shapex, grids_shapey, grids_shapez)
    #$ omp end parallel


@stack_array('df', 'df_t', 'df_inv', 'df_inv_times_v', 'filling_m', 'filling_v')
def hybrid_fA_Arelated(markers: 'float[:,:]', n_markers_tot: 'int',
                          pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                          starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                          kind_map: 'int', params_map: 'float[:]',
                          p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                          ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                          cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                          mat11: 'float[:,:,:,:,:,:]',
                          mat12: 'float[:,:,:,:,:,:]',
                          mat13: 'float[:,:,:,:,:,:]',
                          mat22: 'float[:,:,:,:,:,:]',
                          mat23: 'float[:,:,:,:,:,:]',
                          mat33: 'float[:,:,:,:,:,:]',
                          vec1: 'float[:,:,:]',
                          vec2: 'float[:,:,:]',
                          vec3: 'float[:,:,:]'):  # model specific argument
    r"""
    Accumulates into V1 with the filling functions

    .. math::

        A_p^{\mu, \nu} &= f_0(\eta_p, v_p) * [ DF^{-1}(\eta_p) * v_p ]_\mu * [ DF^{-1}(\eta_p) * v_p ]_\nu    

        B_p^\mu &= \sqrt{f_0(\eta_p, v_p)} * w_p * [ DF^{-1}(\eta_p) * v_p ]_\mu  

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
    df_inv = empty((3, 3), dtype=float)

    # allocate for filling
    df_inv_times_v = empty(3, dtype=float)
    filling_m = empty((3, 3), dtype=float)
    filling_v = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private (ip, eta1, eta2, eta3, v, df, df_inv, df_inv_times_v, weight, filling_m, filling_v)
    #$ omp for reduction ( + : mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32, mat33, vec1, vec2, vec3)
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # evaluate background
        v = markers[ip, 3:6]

        # evaluate Jacobian, result in df
        map_eval.df(eta1, eta2, eta3,
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # filling functions
        linalg.matrix_inv(df, df_inv)
        linalg.matrix_vector(df_inv, v, df_inv_times_v)

        weight = markers[ip, 6]

        # filling_m
        filling_m[0,0] = weight / n_markers_tot * (df_inv[0,0]*df_inv[0,0] + df_inv[0,1]*df_inv[0,1] + df_inv[0,2]*df_inv[0,2])
        filling_m[0,1] = weight / n_markers_tot * (df_inv[0,0]*df_inv[1,0] + df_inv[0,1]*df_inv[1,1] + df_inv[0,2]*df_inv[1,2])
        filling_m[0,2] = weight / n_markers_tot * (df_inv[0,0]*df_inv[2,0] + df_inv[0,1]*df_inv[2,1] + df_inv[0,2]*df_inv[2,2])

        filling_m[1,1] = weight / n_markers_tot * (df_inv[1,0]*df_inv[1,0] + df_inv[1,1]*df_inv[1,1] + df_inv[1,2]*df_inv[1,2])
        filling_m[1,2] = weight / n_markers_tot * (df_inv[1,0]*df_inv[2,0] + df_inv[1,1]*df_inv[2,1] + df_inv[1,2]*df_inv[2,2])

        filling_m[2,2] = weight / n_markers_tot * (df_inv[2,0]*df_inv[2,0] + df_inv[2,1]*df_inv[2,1] + df_inv[2,2]*df_inv[2,2])

        # filling_v
        filling_v[:] = weight / n_markers_tot * df_inv_times_v

        # call the appropriate matvec filler
        mvf.m_v_fill_b_v1_symm(pn, tn1, tn2, tn3, starts1,
                               eta1, eta2, eta3,
                               mat11, mat12, mat13, mat22, mat23, mat33,
                               filling_m[0, 0], filling_m[0, 1], filling_m[0, 2],
                               filling_m[1, 1], filling_m[1, 2], filling_m[2, 2],
                               vec1, vec2, vec3,
                               filling_v[0], filling_v[1], filling_v[2])

    #$ omp end parallel


@stack_array('df', 'df_t', 'df_inv', 'df_inv_times_v', 'filling_m', 'filling_v')
def linear_vlasov_maxwell(markers: 'float[:,:]', n_markers_tot: 'int',
                          pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                          starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                          kind_map: 'int', params_map: 'float[:]',
                          p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                          ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                          cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                          mat11: 'float[:,:,:,:,:,:]',
                          mat12: 'float[:,:,:,:,:,:]',
                          mat13: 'float[:,:,:,:,:,:]',
                          mat22: 'float[:,:,:,:,:,:]',
                          mat23: 'float[:,:,:,:,:,:]',
                          mat33: 'float[:,:,:,:,:,:]',
                          vec1: 'float[:,:,:]',
                          vec2: 'float[:,:,:]',
                          vec3: 'float[:,:,:]',
                          f0_spec: 'int',          # model specific argument
                          moms_spec: 'int[:]',     # model specific argument
                          f0_params: 'float[:]',   # model specific argument
                          alpha: 'float'):  # model specific argument
    r"""
    Accumulates into V1 with the filling functions

    .. math::

        A_p^{\mu, \nu} &= \frac{\alpha^2}{v_{\text{th}}^2} \frac{1}{N\, s_0} f_0(\mathbf{\eta}_p, \mathbf{v}_p)
            [ DF^{-1}(\mathbf{\eta}_p) v_p ]_\mu [ DF^{-1}(\mathbf{\eta}_p) \mathbf{v}_p ]_\nu \,,

        B_p^\mu &= \alpha^2 \sqrt{f_0(\mathbf{\eta}_p, \mathbf{v}_p)} w_p [ DF^{-1}(\mathbf{\eta}_p) \mathbf{v}_p ]_\mu \,.

    Parameters
    ----------
        f0_spec : int
            Specifier for kinetic background, see :ref:`kinetic_backgrounds`  

        moms_spec : array[int]
            Specifier for the seven moments n0, u0x, u0y, u0z, vth0x, vth0y, vth0z (in this order).
            Is 0 for constant moment, for more see :meth:`struphy.kinetic_background.moments_kernels.moments`.

        f0_params : array[float]
            Parameters needed to specify the moments; the order is specified in :ref:`kinetic_moments` for the respective functions available.

        alpha : float
            = Omega_c / Omega_p ; Parameter determining the coupling strength between particles and fields

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    # allocate for metric coeffs
    df = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)

    # allocate for filling
    df_inv_times_v = empty(3, dtype=float)
    filling_m = empty((3, 3), dtype=float)
    filling_v = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private (ip, eta1, eta2, eta3, f0, df, df_inv, df_inv_times_v, filling_m, filling_v)
    #$ omp for reduction ( + : mat11, mat12, mat13, mat22, mat23, mat33, vec1, vec2, vec3)
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        f0 = background_eval.f0(markers[ip, 0:3], markers[ip, 3:6],
                                f0_spec, moms_spec, f0_params)

        # evaluate Jacobian, result in df
        map_eval.df(eta1, eta2, eta3,
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # filling functions
        linalg.matrix_inv(df, df_inv)
        linalg.matrix_vector(df_inv, markers[ip, 3:6], df_inv_times_v)

        # filling_m = alpha^2 * f0 / (N * s_0 * v_th^2) * (DF^{-1} v_p)_mu * (DF^{-1} v_p)_nu
        linalg.outer(df_inv_times_v, df_inv_times_v, filling_m)
        filling_m[:, :] *= alpha**2 * f0 / (n_markers_tot * markers[ip, 7] * f0_params[4]**2)

        # filling_v = alpha^2 * w_p * sqrt{f_0} DL^{-1} v_p
        filling_v[:] = alpha**2 * sqrt(f0) * markers[ip, 6] * df_inv_times_v[:]

        # call the appropriate matvec filler
        mvf.m_v_fill_b_v1_symm(pn, tn1, tn2, tn3, starts1,
                               eta1, eta2, eta3,
                               mat11, mat12, mat13, mat22, mat23, mat33,
                               filling_m[0, 0], filling_m[0, 1], filling_m[0, 2],
                               filling_m[1, 1], filling_m[1, 2], filling_m[2, 2],
                               vec1, vec2, vec3,
                               filling_v[0], filling_v[1], filling_v[2])

    #$ omp end parallel


@stack_array('g_inv', 'tmp1', 'tmp2', 'b', 'b_prod', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def cc_lin_mhd_6d_1(markers: 'float[:,:]', n_markers_tot: 'int',
                    pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                    starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                    kind_map: 'int', params_map: 'float[:]',
                    p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                    ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                    cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                    mat12: 'float[:,:,:,:,:,:]',
                    mat13: 'float[:,:,:,:,:,:]',
                    mat23: 'float[:,:,:,:,:,:]',
                    b2_1: 'float[:,:,:]',   # model specific argument
                    b2_2: 'float[:,:,:]',   # model specific argument
                    b2_3: 'float[:,:,:]',   # model specific argument
                    basis_u : 'int', scale_mat : 'float'):  # model specific argument
    r"""Accumulates into V1 with the filling functions

    .. math::

        A_p^{\mu, \nu} = w_p * [ G^{-1}(\eta_p) * B2_{\times}(\eta_p) * G^{-1}(\eta_p) ]_{\mu, \nu}     

    where :math:`B2_{\times} * a := B2 \times a` for :math:`a \in \mathbb R^3`. 

    Parameters
    ----------
        b2_1, b2_2, b2_3 : array[float]
            FE coefficients c_ijk of the magnetic field as a 2-form.

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    # allocate for magnetic field evaluation
    b = empty(3, dtype=float)
    b_prod = zeros((3, 3), dtype=float) 

    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)
    
    # allocate for metric coefficients
    df       = empty((3, 3), dtype=float)
    df_inv   = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)
    g_inv    = empty((3, 3), dtype=float)

    # allocate some temporary buffers for filling
    tmp1 = empty((3, 3), dtype=float)
    tmp2 = empty((3, 3), dtype=float)

    # get local number of markers
    n_markers_loc = shape(markers)[0]
    
    #$ omp parallel firstprivate(b_prod) private(ip, eta1, eta2, eta3, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, b, df, det_df, weight, df_inv, df_inv_t, g_inv, tmp1, tmp2, filling_m12, filling_m13, filling_m23) 
    #$ omp for reduction ( + : mat12, mat13, mat23)
    for ip in range(n_markers_loc):
        
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # b-field evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        b[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b2_1, starts2[0])
        b[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2_2, starts2[1])
        b[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b2_3, starts2[2])

        # operator bx() as matrix
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] = +b[1]
        b_prod[1, 0] = +b[2]
        b_prod[1, 2] = -b[0]
        b_prod[2, 0] = -b[1]
        b_prod[2, 1] = +b[0]

        # evaluate Jacobian matrix and Jacobian determinant
        map_eval.df(eta1, eta2, eta3,
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)
        
        det_df = linalg.det(df)

        # marker weight
        weight = markers[ip, 6]
        
        if basis_u == 0:
        
            # filling functions
            filling_m12 = - weight * b_prod[0, 1] * scale_mat
            filling_m13 = - weight * b_prod[0, 2] * scale_mat
            filling_m23 = - weight * b_prod[1, 2] * scale_mat

            # call the appropriate matvec filler
            mvf.mat_fill_v0vec_asym(pn, span1, span2, span3,
                                 bn1, bn2, bn3,
                                 starts0,
                                 mat12, mat13, mat23,
                                 filling_m12, filling_m13, filling_m23)
        
        elif basis_u == 1:
        
            # filling functions
            linalg.matrix_inv_with_det(df, det_df, df_inv)
            linalg.transpose(df_inv, df_inv_t)
            linalg.matrix_matrix(df_inv, df_inv_t, g_inv)
            linalg.matrix_matrix(g_inv, b_prod, tmp1)
            linalg.matrix_matrix(tmp1, g_inv, tmp2)

            filling_m12 = - weight * tmp2[0, 1] * scale_mat
            filling_m13 = - weight * tmp2[0, 2] * scale_mat
            filling_m23 = - weight * tmp2[1, 2] * scale_mat

            # call the appropriate matvec filler
            mvf.mat_fill_v1_asym(pn, span1, span2, span3,
                                 bn1, bn2, bn3,
                                 bd1, bd2, bd3,
                                 starts1,
                                 mat12, mat13, mat23,
                                 filling_m12, filling_m13, filling_m23)
            
        elif basis_u == 2:
            
            # filling functions
            filling_m12 = - weight * b_prod[0, 1] * scale_mat / det_df**2
            filling_m13 = - weight * b_prod[0, 2] * scale_mat / det_df**2
            filling_m23 = - weight * b_prod[1, 2] * scale_mat / det_df**2

            # call the appropriate matvec filler
            mvf.mat_fill_v2_asym(pn, span1, span2, span3,
                                 bn1, bn2, bn3,
                                 bd1, bd2, bd3,
                                 starts1,
                                 mat12, mat13, mat23,
                                 filling_m12, filling_m13, filling_m23)
            
    #$ omp end parallel
    
    mat12 /= n_markers_tot
    mat13 /= n_markers_tot
    mat23 /= n_markers_tot


@stack_array('df', 'df_t', 'df_inv', 'g', 'g_inv', 'filling_m', 'filling_v', 'tmp1', 'tmp1_t', 'tmp2', 'tmp3', 'tmp_v', 'df_inv_times_v', 'b', 'b_prod', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def cc_lin_mhd_6d_2(markers: 'float[:,:]', n_markers_tot: 'int',
                    pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                    starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                    kind_map: 'int', params_map: 'float[:]',
                    p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                    ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                    cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                    mat11: 'float[:,:,:,:,:,:]',
                    mat12: 'float[:,:,:,:,:,:]',
                    mat13: 'float[:,:,:,:,:,:]',
                    mat22: 'float[:,:,:,:,:,:]',
                    mat23: 'float[:,:,:,:,:,:]',
                    mat33: 'float[:,:,:,:,:,:]',
                    vec1: 'float[:,:,:]',
                    vec2: 'float[:,:,:]',
                    vec3: 'float[:,:,:]',
                    b2_1: 'float[:,:,:]',   # model specific argument
                    b2_2: 'float[:,:,:]',   # model specific argument
                    b2_3: 'float[:,:,:]',   # model specific argument
                    basis_u : 'int', scale_mat : 'float', scale_vec : 'float'): # model specific argument
    r"""Accumulates into V1 with the filling functions

    .. math::

        A_p^{\mu, \nu} &= w_p * [ G^{-1}(\eta_p) * B2_{\times}(\eta_p) * G^{-1}(\eta_p) * B2_{\times}(\eta_p)^\top * G^{-1}(\eta_p) ]_{\mu, \nu}

        B_p^\mu &= w_p * [ G^{-1}(\eta_p) * B2_{\times}(\eta_p) * DF^{-1}(\eta_p) * v_p ]_\mu

    where :math:`B2_{\times} * a := B2 \times a` for :math:`a \in \mathbb R^3`.

    Parameters
    ----------
        b2_1, b2_2, b2_3 : array[float]
            FE coefficients c_ijk of the magnetic field as a 2-form.

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    # allocate for magnetic field evaluation
    b = empty(3, dtype=float)
    b_prod = zeros((3, 3), dtype=float)

    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)
    
    # allocate for metric coeffs
    df       = empty((3, 3), dtype=float)
    df_inv   = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)
    g_inv    = empty((3, 3), dtype=float)

    # allocate for filling
    filling_m = empty((3, 3), dtype=float)
    filling_v = empty(3, dtype=float)

    tmp1   = empty((3, 3), dtype=float)
    tmp2   = empty((3, 3), dtype=float)
    
    tmp_t  = empty((3, 3), dtype=float)
    tmp_m  = empty((3, 3), dtype=float)

    tmp_v = empty(3, dtype=float)

    # get number of markers
    n_markers_loc = shape(markers)[0]
    
    #$ omp parallel firstprivate(b_prod) private(ip, eta1, eta2, eta3, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, b, df, det_df, weight, v, df_inv, df_inv_t, g_inv, tmp1, tmp2, tmp_t, tmp_m, tmp_v, filling_m, filling_v) 
    #$ omp for reduction ( + : mat12, mat13, mat23)
    for ip in range(n_markers_loc):
        
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # b-field evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        b[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b2_1, starts2[0])
        b[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2_2, starts2[1])
        b[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b2_3, starts2[2])

        # operator bx() as matrix
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] = +b[1]
        b_prod[1, 0] = +b[2]
        b_prod[1, 2] = -b[0]
        b_prod[2, 0] = -b[1]
        b_prod[2, 1] = +b[0]

        # evaluate Jacobian, result in df
        map_eval.df(eta1, eta2, eta3,
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)
        
        det_df = linalg.det(df)
        
        # marker weight and velocity
        weight = markers[ip, 6]
        v = markers[ip, 3:6]
        
        if basis_u == 0:
            
            # needed metric coefficients
            linalg.matrix_inv_with_det(df, det_df, df_inv)
            linalg.transpose(df_inv, df_inv_t)
            linalg.matrix_matrix(df_inv, df_inv_t, g_inv)
            
            # filling functions tmp_m = tmp1 * tmp1^T and tmp_v = tmp1 * v, where tmp1 = B^x * DF^(-1)
            linalg.matrix_matrix(b_prod, df_inv, tmp1)
            
            linalg.transpose(tmp1, tmp_t)
            
            linalg.matrix_matrix(tmp1, tmp_t, tmp_m)
            linalg.matrix_vector(tmp1, v, tmp_v)

            filling_m[:, :] = weight * tmp_m * scale_mat
            filling_v[:] = weight * tmp_v * scale_vec

            # call the appropriate matvec filler
            mvf.m_v_fill_v0vec_symm(pn, span1, span2, span3,
                                    bn1, bn2, bn3,
                                    starts0,
                                    mat11, mat12, mat13, 
                                    mat22, mat23, 
                                    mat33, 
                                    filling_m[0, 0], filling_m[0, 1], filling_m[0, 2], 
                                    filling_m[1, 1], filling_m[1, 2], 
                                    filling_m[2, 2],
                                    vec1, vec2, vec3,
                                    filling_v[0], filling_v[1], filling_v[2])
        
        elif basis_u == 1:
            
            # needed metric coefficients
            linalg.matrix_inv_with_det(df, det_df, df_inv)
            linalg.transpose(df_inv, df_inv_t)
            linalg.matrix_matrix(df_inv, df_inv_t, g_inv)
            
            # filling functions tmp_m = tmp2 * tmp2^T and tmp_v = tmp2 * v, where tmp2 = G^(-1) * B^x * DF^(-1)
            linalg.matrix_matrix(g_inv, b_prod, tmp1)
            linalg.matrix_matrix(tmp1, df_inv, tmp2)
            
            linalg.transpose(tmp2, tmp_t)
            
            linalg.matrix_matrix(tmp2, tmp_t, tmp_m)
            linalg.matrix_vector(tmp2, v, tmp_v)

            filling_m[:, :] = weight * tmp_m * scale_mat
            filling_v[:] = weight * tmp_v * scale_vec

            # call the appropriate matvec filler
            mvf.m_v_fill_v1_symm(pn, span1, span2, span3,
                                 bn1, bn2, bn3,
                                 bd1, bd2, bd3,
                                 starts1,
                                 mat11, mat12, mat13, 
                                 mat22, mat23, 
                                 mat33, 
                                 filling_m[0, 0], filling_m[0, 1], filling_m[0, 2], 
                                 filling_m[1, 1], filling_m[1, 2], 
                                 filling_m[2, 2],
                                 vec1, vec2, vec3,
                                 filling_v[0], filling_v[1], filling_v[2])
            
        elif basis_u == 2:
            
            # needed metric coefficients
            linalg.matrix_inv_with_det(df, det_df, df_inv)
            linalg.transpose(df_inv, df_inv_t)
            linalg.matrix_matrix(df_inv, df_inv_t, g_inv)
            
            # filling functions tmp_m = tmp1 * tmp1^T and tmp_v = tmp1 * v, where tmp1 = B^x * DF^(-1) / det(DF)
            linalg.matrix_matrix(b_prod, df_inv, tmp1)
            
            linalg.transpose(tmp1, tmp_t)
            
            linalg.matrix_matrix(tmp1, tmp_t, tmp_m)
            linalg.matrix_vector(tmp1, v, tmp_v)

            filling_m[:, :] = weight * tmp_m * scale_mat / det_df**2
            filling_v[:] = weight * tmp_v * scale_vec / det_df

            # call the appropriate matvec filler
            mvf.m_v_fill_v2_symm(pn, span1, span2, span3,
                                 bn1, bn2, bn3,
                                 bd1, bd2, bd3,
                                 starts2,
                                 mat11, mat12, mat13, 
                                 mat22, mat23, 
                                 mat33, 
                                 filling_m[0, 0], filling_m[0, 1], filling_m[0, 2], 
                                 filling_m[1, 1], filling_m[1, 2], 
                                 filling_m[2, 2],
                                 vec1, vec2, vec3,
                                 filling_v[0], filling_v[1], filling_v[2])
            
    #$ omp end parallel
    
    mat11 /= n_markers_tot
    mat12 /= n_markers_tot
    mat13 /= n_markers_tot
    mat22 /= n_markers_tot
    mat23 /= n_markers_tot
    mat33 /= n_markers_tot
    
    vec1 /= n_markers_tot
    vec2 /= n_markers_tot
    vec3 /= n_markers_tot


@stack_array('df', 'df_t', 'df_inv', 'filling_m', 'filling_v', 'tmp1', 'tmp_v', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def pc_lin_mhd_6d_full(markers: 'float[:,:]', n_markers_tot: 'int',
                       pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                       starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                       kind_map: 'int', params_map: 'float[:]',
                       p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                       ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                       cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
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

    r"""Accumulates into V1 with the filling functions

    .. math::

        V_{p,i} A_p^{\mu, \nu} V_{p,j} &= w_p * [ DF^{-1}(\eta_p) DF^{-\top}(\eta_p) ]_{\mu, \nu} * V_{p,i} * V_{p,j}
        
        V_{p,i} B_p^\mu &= w_p * [DF^{-1}(\eta_p)]_\mu * V_{p,i}

    Parameters
    ----------

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    # allocate for metric coeffs
    df = empty((3, 3), dtype=float)
    df_t = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)

    # allocate for filling
    filling_m = empty((3, 3), dtype=float)
    filling_v = empty(3, dtype=float)

    tmp1 = empty((3, 3), dtype=float)

    tmp_v = empty(3, dtype=float)

    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]
    
    #$ omp parallel private(ip, eta1, eta2, eta3, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, df, df_inv, df_t, df_inv_t, v, tmp1, tmp_v, weight, filling_m, filling_v) 
    #$ omp for reduction ( + : mat11_11, mat11_12, mat11_13, mat11_22, mat11_23, mat11_33, mat12_11, mat12_12, mat12_13, mat12_22, mat12_23, mat12_33, mat13_11, mat13_12, mat13_13, mat13_22, mat13_23, mat13_33, mat22_11, mat22_12, mat22_13, mat22_22, mat22_23, mat22_33, mat23_11, mat23_12, mat23_13, mat23_22, mat23_23, mat23_33, mat33_11, mat33_12, mat33_13, mat33_22, mat33_23, mat33_33, vec1_1, vec1_2, vec1_3, vec2_1, vec2_2, vec2_3, vec3_1, vec3_2, vec3_3)
    for ip in range(n_markers):
        
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        
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
        v = markers[ip, 3:6]
        
        linalg.matrix_matrix(df_inv, df_inv_t, tmp1)
        linalg.matrix_vector(df_inv, v, tmp_v)

        weight = markers[ip, 8]
        
        filling_m[:] = weight * tmp1 / n_markers_tot
        filling_v[:] = weight * tmp_v / n_markers_tot

        # call the appropriate matvec filler
        mvf.m_v_fill_v1_pressure_full(pn, span1, span2, span3,
                                      bn1, bn2, bn3,
                                      bd1, bd2, bd3,
                                      starts1,
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


@stack_array('df', 'df_t', 'df_inv', 'filling_m', 'filling_v', 'tmp1', 'tmp_v', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def pc_lin_mhd_6d(markers: 'float[:,:]', n_markers_tot: 'int',
                  pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                  starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                  kind_map: 'int', params_map: 'float[:]',
                  p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                  ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                  cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
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
    r"""Accumulates into V1 with the filling functions

    .. math::

        {V_{p,i}}_\perp A_p^{\mu, \nu} {V_{p,j}}_\perp &= w_p * [ DF^{-1}(\eta_p) DF^{-\top}(\eta_p) ]_{\mu, \nu} * {V_{p,i}}_\perp * {V_{p,j}}_\perp
        
        {V_{p,i}}_\perp B_p^\mu &= w_p * [DF^{-1}(\eta_p)]_\mu * {V_{p,i}}_\perp

    Parameters
    ----------

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    # allocate for metric coeffs
    df = empty((3, 3), dtype=float)
    df_t = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)

    # allocate for filling
    filling_m = empty((3, 3), dtype=float)
    filling_v = empty(3, dtype=float)

    tmp1 = empty((3, 3), dtype=float)

    tmp_v = empty(3, dtype=float)

    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]
    
    #$ omp parallel private(ip, eta1, eta2, eta3, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, df, df_inv, df_t, df_inv_t, v, tmp1, tmp_v, weight, filling_m, filling_v) 
    #$ omp for reduction ( + : mat11_11, mat11_12, mat11_22, mat12_11, mat12_12, mat12_22, mat13_11, mat13_12, mat13_22, mat22_11, mat22_12, mat22_22, mat23_11, mat23_12, mat23_22, mat33_11, mat33_12, mat33_22, vec1_1, vec1_2, vec2_1, vec2_2, vec3_1, vec3_2)
    for ip in range(n_markers):
        
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        
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
        v = markers[ip, 3:6]
        
        linalg.matrix_matrix(df_inv, df_inv_t, tmp1)
        linalg.matrix_vector(df_inv, v, tmp_v)

        weight = markers[ip, 8]
        
        filling_m[:] = weight * tmp1 / n_markers_tot
        filling_v[:] = weight * tmp_v / n_markers_tot

        # call the appropriate matvec filler
        mvf.m_v_fill_v1_pressure(pn, span1, span2, span3,
                                 bn1, bn2, bn3,
                                 bd1, bd2, bd3,
                                 starts1,
                                 mat11_11, mat12_11, mat13_11, 
                                 mat22_11, mat23_11, 
                                 mat33_11,
                                 mat11_12, mat12_12, mat13_12, 
                                 mat22_12, mat23_12, 
                                 mat33_12,
                                 mat11_22, mat12_22, mat13_22, 
                                 mat22_22, mat23_22, 
                                 mat33_22,
                                 filling_m[0, 0], filling_m[0, 1], filling_m[0, 2], 
                                 filling_m[1, 1], filling_m[1, 2], 
                                 filling_m[2, 2],
                                 vec1_1, vec2_1, vec3_1,
                                 vec1_2, vec2_2, vec3_2,
                                 filling_v[0], filling_v[1], filling_v[2],
                                 v[0], v[1])
    #$ omp end parallel

@stack_array('df', 'df_t', 'df_inv', 'g_inv', 'filling_m', 'filling_v', 'tmp1',  'tmp_t', 'tmp_m', 'tmp_v', 'b', 'b_prod', 'b_star', 'norm_b1', 'curl_norm_b', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def cc_lin_mhd_5d_J1(markers: 'float[:,:]', n_markers_tot: 'int',
                     pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                     starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                     kind_map: 'int', params_map: 'float[:]',
                     p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                     ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                     cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                     mat11: 'float[:,:,:,:,:,:]',
                     mat12: 'float[:,:,:,:,:,:]',
                     mat13: 'float[:,:,:,:,:,:]',
                     mat22: 'float[:,:,:,:,:,:]',
                     mat23: 'float[:,:,:,:,:,:]',
                     mat33: 'float[:,:,:,:,:,:]',
                     vec1: 'float[:,:,:]',
                     vec2: 'float[:,:,:]',
                     vec3: 'float[:,:,:]',
                     b1: 'float[:,:,:]',          # model specific argument
                     b2: 'float[:,:,:]',          # model specific argument 
                     b3: 'float[:,:,:]',          # model specific argument 
                     norm_b11: 'float[:,:,:]',       # model specific argument    
                     norm_b12: 'float[:,:,:]',       # model specific argument
                     norm_b13: 'float[:,:,:]',       # model specific argument
                     curl_norm_b1: 'float[:,:,:]',  # model specific argument
                     curl_norm_b2: 'float[:,:,:]',  # model specific argument
                     curl_norm_b3: 'float[:,:,:]',  # model specific argument
                     epsilon: float,    # model specific argument
                     basis_u : 'int', scale_mat : 'float', scale_vec : 'float'): # model specific argument

    r"""Accumulates into V1 with the filling functions

    .. math::

        A_p^{\mu, \nu} &= w_p * [G^{-1}(\eta_p) * B2_{\times}(\eta_p) * B2_{\times}(\eta_p)^\top * G^{-1}(\eta_p) * v^2_{\parallel,p} * \left( 1/B^*_\parallel \right)^2 * |1/\sqrt{g} \hat \nabla \times \hat b^1_0|_p^2]_{\mu, \nu}

        B_p^\mu &= w_p *[ G^{-1}(\eta_p) * B2_{\times}(\eta_p)* v^2_{\parallel,p} * \left( 1/B^*_\parallel \right)]_\mu

    where :math:`B2_{\times} * a := B2 \times a` for :math:`a \in \mathbb R^3`.

    Parameters
    ----------
        b1, b2, b3 : array[float]
            FE coefficients c_ijk of the magnetic field as a 2-form.

        norm_b11, norm_b12, norm_b13 : array[float]
            FE coefficients c_ijk of the normalized magnetic field as a 1-form.

        curl_norm_b1, curl_norm_b2, curl_norm_b3 : array[float]
            FE coefficients c_ijk of the curl of normalized magnetic field as a 2-form.

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    # allocate for magnetic field evaluation
    b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    b_prod = zeros((3, 3), dtype=float)
    norm_b1 = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)

    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)
    
    # allocate for metric coeffs
    df       = empty((3, 3), dtype=float)
    df_inv   = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)
    g_inv    = empty((3, 3), dtype=float)

    # allocate for filling
    filling_m = empty((3, 3), dtype=float)
    filling_v = empty(3, dtype=float)

    tmp1   = empty((3, 3), dtype=float)
    
    tmp_t  = empty((3, 3), dtype=float)
    tmp_m  = empty((3, 3), dtype=float)

    tmp_v = empty(3, dtype=float)

    # get number of markers
    n_markers_loc = shape(markers)[0]
    
    #$ omp parallel firstprivate(b_prod) private(ip, eta1, eta2, eta3, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, b, b_Star, norm_b1, curl_norm_b, df, det_df, weight, v, df_inv, df_inv_t, g_inv, tmp1, tmp2, tmp_t, tmp_m, tmp_v, filling_m, filling_v) 
    #$ omp for reduction ( + : mat11, mat12, mat13, mat22, mat23, mat33)
    for ip in range(n_markers_loc):
        
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # marker weight and velocity
        weight = markers[ip, 6]
        v = markers[ip, 3]

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
        
        det_df = linalg.det(df)

        # b; 2form
        b[0] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts2[0])
        b[1] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts2[1])
        b[2] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts2[2])

        # norm_b1; 1form
        norm_b1[0] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts1[0])
        norm_b1[1] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts1[1])
        norm_b1[2] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts1[2])

        # curl_norm_b; 2form
        curl_norm_b[0] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts2[0])
        curl_norm_b[1] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts2[1])
        curl_norm_b[2] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts2[2])

        # b_star; 2form in H1vec
        b_star[:] = (b + curl_norm_b*v*epsilon)/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg.scalar_dot(norm_b1, b_star)

        # calculate square of curl_norm_b
        curl_norm_b_square = linalg.scalar_dot(curl_norm_b, curl_norm_b)

        # operator bx() as matrix
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] = +b[1]
        b_prod[1, 0] = +b[2]
        b_prod[1, 2] = -b[0]
        b_prod[2, 0] = -b[1]
        b_prod[2, 1] = +b[0]

        if basis_u == 0:

            linalg.transpose(b_prod, tmp_t)
            
            linalg.matrix_matrix(b_prod, tmp_t, tmp_m)
            linalg.matrix_vector(b_prod, curl_norm_b, tmp_v)

            filling_m[:, :] = weight * tmp_m * curl_norm_b_square * v**2 / abs_b_star_para**2 / det_df**2 * scale_mat
            filling_v[:] =    weight * tmp_v                      * v**2 / abs_b_star_para    / det_df    * scale_vec

            # call the appropriate matvec filler
            mvf.m_v_fill_v0vec_symm(pn, span1, span2, span3,
                                    bn1, bn2, bn3,
                                    starts0,
                                    mat11, mat12, mat13, 
                                    mat22, mat23, 
                                    mat33, 
                                    filling_m[0, 0], filling_m[0, 1], filling_m[0, 2], 
                                    filling_m[1, 1], filling_m[1, 2], 
                                    filling_m[2, 2],
                                    vec1, vec2, vec3,
                                    filling_v[0], filling_v[1], filling_v[2])
        
        elif basis_u == 1:
            
            # needed metric coefficients
            linalg.matrix_inv_with_det(df, det_df, df_inv)
            linalg.transpose(df_inv, df_inv_t)
            linalg.matrix_matrix(df_inv, df_inv_t, g_inv)
            
            linalg.matrix_matrix(g_inv, b_prod, tmp1)
            
            linalg.transpose(tmp1, tmp_t)
            
            linalg.matrix_matrix(tmp1, tmp_t, tmp_m)
            linalg.matrix_vector(tmp1, curl_norm_b, tmp_v)

            filling_m[:, :] = weight * tmp_m * curl_norm_b_square * v**2 / abs_b_star_para**2 / det_df**2 * scale_mat
            filling_v[:] =    weight * tmp_v                      * v**2 / abs_b_star_para    / det_df    * scale_vec

            # call the appropriate matvec filler
            mvf.m_v_fill_v1_symm(pn, span1, span2, span3,
                                 bn1, bn2, bn3,
                                 bd1, bd2, bd3,
                                 starts1,
                                 mat11, mat12, mat13, 
                                 mat22, mat23, 
                                 mat33, 
                                 filling_m[0, 0], filling_m[0, 1], filling_m[0, 2], 
                                 filling_m[1, 1], filling_m[1, 2], 
                                 filling_m[2, 2],
                                 vec1, vec2, vec3,
                                 filling_v[0], filling_v[1], filling_v[2])
            
        elif basis_u == 2:
            
            linalg.transpose(b_prod, tmp_t)
            
            linalg.matrix_matrix(b_prod, tmp_t, tmp_m)
            linalg.matrix_vector(b_prod, curl_norm_b, tmp_v)

            filling_m[:, :] = weight * tmp_m * curl_norm_b_square * v**2 / abs_b_star_para**2 / det_df**4 * scale_mat
            filling_v[:] =    weight * tmp_v                      * v**2 / abs_b_star_para    / det_df**2 * scale_vec

            # call the appropriate matvec filler
            mvf.m_v_fill_v2_symm(pn, span1, span2, span3,
                                 bn1, bn2, bn3,
                                 bd1, bd2, bd3,
                                 starts2,
                                 mat11, mat12, mat13, 
                                 mat22, mat23, 
                                 mat33, 
                                 filling_m[0, 0], filling_m[0, 1], filling_m[0, 2], 
                                 filling_m[1, 1], filling_m[1, 2], 
                                 filling_m[2, 2],
                                 vec1, vec2, vec3,
                                 filling_v[0], filling_v[1], filling_v[2])
            
    #$ omp end parallel
    
    mat11 /= n_markers_tot
    mat12 /= n_markers_tot
    mat13 /= n_markers_tot
    mat22 /= n_markers_tot
    mat23 /= n_markers_tot
    mat33 /= n_markers_tot
    
    vec1 /= n_markers_tot
    vec2 /= n_markers_tot
    vec3 /= n_markers_tot

@stack_array('df', 'df_t', 'df_inv', 'g_inv', 'filling_m', 'filling_v', 'tmp1', 'tmp2', 'tmp_t', 'tmp_m', 'tmp_v', 'b', 'b_prod', 'norm_b2_prod', 'b_star', 'curl_norm_b', 'norm_b1', 'norm_b2', 'grad_PB', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def cc_lin_mhd_5d_J2(markers: 'float[:,:]', n_markers_tot: 'int',
                    pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                    starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                    kind_map: 'int', params_map: 'float[:]',
                    p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                    ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                    cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                    mat11: 'float[:,:,:,:,:,:]',
                    mat12: 'float[:,:,:,:,:,:]',
                    mat13: 'float[:,:,:,:,:,:]',
                    mat22: 'float[:,:,:,:,:,:]',
                    mat23: 'float[:,:,:,:,:,:]',
                    mat33: 'float[:,:,:,:,:,:]',
                    vec1: 'float[:,:,:]',
                    vec2: 'float[:,:,:]',
                    vec3: 'float[:,:,:]',
                    b1: 'float[:,:,:]',   # model specific argument
                    b2: 'float[:,:,:]',   # model specific argument
                    b3: 'float[:,:,:]',   # model specific argument
                    norm_b11: 'float[:,:,:]',       # model specific argument    
                    norm_b12: 'float[:,:,:]',       # model specific argument
                    norm_b13: 'float[:,:,:]',       # model specific argument
                    norm_b21: 'float[:,:,:]',       # model specific argument    
                    norm_b22: 'float[:,:,:]',       # model specific argument
                    norm_b23: 'float[:,:,:]',       # model specific argument
                    curl_norm_b1: 'float[:,:,:]',  # model specific argument
                    curl_norm_b2: 'float[:,:,:]',  # model specific argument
                    curl_norm_b3: 'float[:,:,:]',  # model specific argument
                    grad_PB1: 'float[:,:,:]',  # model specific argument
                    grad_PB2: 'float[:,:,:]',  # model specific argument
                    grad_PB3: 'float[:,:,:]',  # model specific argument
                    epsilon: float,    # model specific argument
                    basis_u : 'int', scale_mat : 'float', scale_vec : 'float'): # model specific argument
    
    r"""Accumulates into V1 with the filling functions

    .. math::

        A_p^{\mu, \nu} &= w_p * [G^{-1}(\eta_p) * B2_{\times}(\eta_p) * B2_{\times}(\eta_p)^\top * G^{-1}(\eta_p) * v^2_{\parallel,p} * \left( 1/B^*_\parallel \right)^2 * |1/\sqrt{g} \hat \nabla \times \hat b^1_0|_p^2]_{\mu, \nu}

        B_p^\mu &= w_p *[ G^{-1}(\eta_p) * B2_{\times}(\eta_p)* v^2_{\parallel,p} * \left( 1/B^*_\parallel \right)]_\mu

    where :math:`B2_{\times} * a := B2 \times a` for :math:`a \in \mathbb R^3`.

    Parameters
    ----------
        b1, b2, b3 : array[float]
            FE coefficients c_ijk of the magnetic field as a 2-form.

        norm_b11, norm_b12, norm_b13 : array[float]
            FE coefficients c_ijk of the normalized magnetic field as a 1-form.

        norm_b21, norm_b22, norm_b23 : array[float]
            FE coefficients c_ijk of the normalized magnetic field as a 2-form.

        curl_norm_b1, curl_norm_b2, curl_norm_b3 : array[float]
            FE coefficients c_ijk of the curl of normalized magnetic field as a 2-form.

        grad_PB1, grad_PB2, grad_PB3 : array[float]
            FE coefficients c_ijk of gradient of parallel magnetic field as a 1-form.

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    # allocate for magnetic field evaluation
    b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    b_prod = zeros((3, 3), dtype=float)
    norm_b2_prod = zeros((3, 3), dtype=float)
    curl_norm_b = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    norm_b2 = empty(3, dtype=float)
    grad_PB = empty(3, dtype=float)

    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)
    
    # allocate for metric coeffs
    df       = empty((3, 3), dtype=float)
    df_inv   = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)
    g_inv    = empty((3, 3), dtype=float)

    # allocate for filling
    filling_m = empty((3, 3), dtype=float)
    filling_v = empty(3, dtype=float)

    tmp1   = empty((3, 3), dtype=float)
    tmp2   = empty((3, 3), dtype=float)
    tmp_t  = empty((3, 3), dtype=float)
    tmp_m  = empty((3, 3), dtype=float)

    tmp_v = empty(3, dtype=float)

    # get number of markers
    n_markers_loc = shape(markers)[0]
    
    #$ omp parallel firstprivate(b_prod) private(ip, eta1, eta2, eta3, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, b, b_star, b_prob, norm_b2_prod, curl_norm_b, norm_b1, norm_b2, grad_PB, df, det_df, weight, v, df_inv, df_inv_t, g_inv, tmp1,tmp2, tmp_t, tmp_m, tmp_v, filling_m, filling_v) 
    #$ omp for reduction ( + : mat11, mat12, mat13, mat22, mat23, mat33)
    for ip in range(n_markers_loc):
        
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # marker weight and velocity
        weight = markers[ip, 6]
        v = markers[ip, 3]
        mu = markers[ip, 4]

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
        
        det_df = linalg.det(df)

        # needed metric coefficients
        linalg.matrix_inv_with_det(df, det_df, df_inv)
        linalg.transpose(df_inv, df_inv_t)
        linalg.matrix_matrix(df_inv, df_inv_t, g_inv)

        # b; 2form
        b[0] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts2[0])
        b[1] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts2[1])
        b[2] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts2[2])

        # norm_b1; 1form
        norm_b1[0] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts1[0])
        norm_b1[1] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts1[1])
        norm_b1[2] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts1[2])

        # norm_b2; 2form
        norm_b2[0] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, norm_b21, starts2[0])
        norm_b2[1] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, norm_b22, starts2[1])
        norm_b2[2] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, norm_b23, starts2[2])

        # curl_norm_b; 2form
        curl_norm_b[0] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts2[0])
        curl_norm_b[1] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts2[1])
        curl_norm_b[2] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts2[2])

        # grad_PB; 1form
        grad_PB[0] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_PB1, starts1[0])
        grad_PB[1] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_PB2, starts1[1])
        grad_PB[2] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_PB3, starts1[2])

        # b_star; 2form transformed into H1vec
        b_star[:] = (b + curl_norm_b*v*epsilon)/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg.scalar_dot(norm_b1, b_star)

        # operator bx() as matrix
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] = +b[1]
        b_prod[1, 0] = +b[2]
        b_prod[1, 2] = -b[0]
        b_prod[2, 0] = -b[1]
        b_prod[2, 1] = +b[0]

        norm_b2_prod[0, 1] = -norm_b2[2]
        norm_b2_prod[0, 2] = +norm_b2[1]
        norm_b2_prod[1, 0] = +norm_b2[2]
        norm_b2_prod[1, 2] = -norm_b2[0]
        norm_b2_prod[2, 0] = -norm_b2[1]
        norm_b2_prod[2, 1] = +norm_b2[0]

        if basis_u == 0:
            
            linalg.matrix_matrix(b_prod, g_inv, tmp1)
            linalg.matrix_matrix(tmp1, norm_b2_prod, tmp2)
            linalg.matrix_matrix(tmp2, g_inv, tmp1)

            linalg.transpose(tmp1, tmp_t)
            
            linalg.matrix_matrix(tmp1, tmp_t, tmp_m)
            linalg.matrix_vector(tmp1, grad_PB, tmp_v)

            filling_m[:, :] = weight * tmp_m * mu / abs_b_star_para**2 * scale_mat
            filling_v[:] =    weight * tmp_v * mu / abs_b_star_para    * scale_vec

            # call the appropriate matvec filler
            mvf.m_v_fill_v0vec_symm(pn, span1, span2, span3,
                                    bn1, bn2, bn3,
                                    starts0,
                                    mat11, mat12, mat13, 
                                    mat22, mat23, 
                                    mat33, 
                                    filling_m[0, 0], filling_m[0, 1], filling_m[0, 2], 
                                    filling_m[1, 1], filling_m[1, 2], 
                                    filling_m[2, 2],
                                    vec1, vec2, vec3,
                                    filling_v[0], filling_v[1], filling_v[2])
        
        elif basis_u == 1:
            
            linalg.matrix_matrix(g_inv, b_prod, tmp1)
            linalg.matrix_matrix(tmp1, g_inv, tmp2)
            linalg.matrix_matrix(tmp2, norm_b2_prod, tmp1)
            linalg.matrix_matrix(tmp1, g_inv, tmp2)

            linalg.transpose(tmp2, tmp_t)
            
            linalg.matrix_matrix(tmp2, tmp_t, tmp_m)
            linalg.matrix_vector(tmp2, grad_PB, tmp_v)

            filling_m[:, :] = weight * tmp_m * mu / abs_b_star_para**2 * scale_mat
            filling_v[:] =    weight * tmp_v * mu / abs_b_star_para    * scale_vec

            # call the appropriate matvec filler
            mvf.m_v_fill_v1_symm(pn, span1, span2, span3,
                                 bn1, bn2, bn3,
                                 bd1, bd2, bd3,
                                 starts1,
                                 mat11, mat12, mat13, 
                                 mat22, mat23, 
                                 mat33, 
                                 filling_m[0, 0], filling_m[0, 1], filling_m[0, 2], 
                                 filling_m[1, 1], filling_m[1, 2], 
                                 filling_m[2, 2],
                                 vec1, vec2, vec3,
                                 filling_v[0], filling_v[1], filling_v[2])
            
        elif basis_u == 2:
            
            linalg.matrix_matrix(b_prod, g_inv, tmp1)
            linalg.matrix_matrix(tmp1, norm_b2_prod, tmp2)
            linalg.matrix_matrix(tmp2, g_inv, tmp1)

            linalg.transpose(tmp1, tmp_t)
            
            linalg.matrix_matrix(tmp1, tmp_t, tmp_m)
            linalg.matrix_vector(tmp1, grad_PB, tmp_v)

            filling_m[:, :] = weight * tmp_m * mu / abs_b_star_para**2 / det_df**2 * scale_mat
            filling_v[:] =    weight * tmp_v * mu / abs_b_star_para    / det_df    * scale_vec

            # call the appropriate matvec filler
            mvf.m_v_fill_v2_symm(pn, span1, span2, span3,
                                 bn1, bn2, bn3,
                                 bd1, bd2, bd3,
                                 starts2,
                                 mat11, mat12, mat13, 
                                 mat22, mat23, 
                                 mat33, 
                                 filling_m[0, 0], filling_m[0, 1], filling_m[0, 2], 
                                 filling_m[1, 1], filling_m[1, 2], 
                                 filling_m[2, 2],
                                 vec1, vec2, vec3,
                                 filling_v[0], filling_v[1], filling_v[2])
            
    #$ omp end parallel
    
    mat11 /= n_markers_tot
    mat12 /= n_markers_tot
    mat13 /= n_markers_tot
    mat22 /= n_markers_tot
    mat23 /= n_markers_tot
    mat33 /= n_markers_tot
    
    vec1 /= n_markers_tot
    vec2 /= n_markers_tot
    vec3 /= n_markers_tot

