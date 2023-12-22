'Accumulation kernels for full-orbit (6D) particles.'


from pyccel.decorators import stack_array

from numpy import zeros, empty, sqrt, shape, floor, log

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.bsplines.bsplines_kernels as bsplines_kernels
import struphy.bsplines.evaluation_kernels_3d as evaluation_kernels_3d
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.accumulation.particle_to_mat_kernels as particle_to_mat_kernels
import struphy.pic.accumulation.filler_kernels as filler_kernels


def a_documentation():
    r'''
    Explainer for arguments of accumulation kernels.

    Function naming conventions:

    * use the model name, all lower-case letters (e.g. ``lin_vlasov_maxwell``)
    * in case of multiple accumulations in one model, attach ``_1``, ``_2`` or the species name.

    These kernels are passed to :class:`struphy.pic.accumulation.particles_to_grid.Accumulator` and called via::

        Accumulator.accumulate()

    The arguments passed to each kernel have a pre-defined order, defined in :class:`struphy.pic.accumulation.particles_to_grid.Accumulator`.
    This order is as follows (you can copy and paste from existing accum_kernels functions):

    1. Marker info:
        * ``markers: 'float[:,:]'``          # local marker array
        * ``n_markers_tot: 'int'``           # total number of markers :math:`N` (all processes)

    2. Derham spline bases info:
        * ``pn: 'int[:]'``                   # N-spline degree in each direction
        * ``tn1: 'float[:]'``                # N-spline knot vector 
        * ``tn2: 'float[:]'``
        * ``tn3: 'float[:]'``    

    3. mpi.comm start indices of FE coeffs on current process:
        - ``starts: 'int[:]'``               # start indices of current process 

    4. Mapping info:
        - ``kind_map: 'int'``                # mapping identifier 
        - ``params_map: 'float[:]'``         # mapping parameters
        - ``p_map: 'int[:]'``                # spline degree
        - ``t1_map: 'float[:]'``             # knot vector 
        - ``t2_map: 'float[:]'``             
        - ``t3_map: 'float[:]'`` 
        - ``ind1_map: int[:,:]``             # Indices of non-vanishing splines in format (number of mapping grid cells, p_map + 1)       
        - ``ind2_map: int[:,:]`` 
        - ``ind3_map: int[:,:]``            
        - ``cx: 'float[:,:,:]'``             # control points for Fx
        - ``cy: 'float[:,:,:]'``             # control points for Fy
        - ``cz: 'float[:,:,:]'``             # control points for Fz                         

    5. Data objects to accumulate into (number depends on model, but at least one matrix or vector has to be passed)
        - mat11: ``'float[:,:,:,:,:,:]'``    # _data attribute of StencilMatrix
        - optional:

            - ``mat12: 'float[:,:,:,:,:,:]'``
            - ``mat13: 'float[:,:,:,:,:,:]'``
            - ``mat21: 'float[:,:,:,:,:,:]'``
            - ``mat22: 'float[:,:,:,:,:,:]'``
            - ``mat23: 'float[:,:,:,:,:,:]'``
            - ``mat31: 'float[:,:,:,:,:,:]'``
            - ``mat32: 'float[:,:,:,:,:,:]'``
            - ``mat33: 'float[:,:,:,:,:,:]'``
            - ``vec1: 'float[:,:,:]'``           # _data attribute of StencilVector
            - ``vec2: 'float[:,:,:]'``
            - ``vec3: 'float[:,:,:]'``

    6. Optional: additional parameters, for example
        - ``b2_1: 'float[:,:,:]'``           # spline coefficients of b2_1
        - ``b2_2: 'float[:,:,:]'``           # spline coefficients of b2_2
        - ``b2_3: 'float[:,:,:]'``            # spline coefficients of b2_3
        - ``f0_params: 'float[:]'``          # parameters of equilibrium background
    '''

    print('This is just the docstring function.')


@stack_array('bn1', 'bn2', 'bn3')
def poisson(markers: 'float[:,:]', n_markers_tot: 'int',
            pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
            starts: 'int[:]',
            kind_map: 'int', params_map: 'float[:]',
            p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
            ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
            cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
            vec: 'float[:,:,:]',
            alpha: 'float',  # model specific argument
            epsilon: 'float'):  # model specific argument
    r"""
    Kernel for :class:`struphy.pic.accumulation.particles_to_grid.AccumulatorVector` with the filling 

    .. math::

        B_p^\mu = \frac{\alpha^2}{\epsilon} w_p \,.

    Parameters
    ----------
    alpha : float
        Omega_c / Omega_p.

    epsilon : float
        omega / Omega_c.

    Note
    ----
    The above parameter list contains only the model specific input arguments (`*args_add`).
    """

    # non-vanishing B-splines at particle position
    bn1 = empty(int(pn[0]) + 1, dtype=float)
    bn2 = empty(int(pn[1]) + 1, dtype=float)
    bn3 = empty(int(pn[2]) + 1, dtype=float)

    #$ omp parallel private (ip, eta1, eta2, eta3, f0, filling)
    #$ omp for reduction ( + :vec)
    for ip in range(shape(markers)[0]):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # filling = alpha^2 / epsilon * w_p
        filling = alpha**2 / epsilon * markers[ip, 6] / n_markers_tot

        # spans (i.e. index for non-vanishing B-spline basis functions)
        span1 = bsplines_kernels.find_span(tn1, int(pn[0]), eta1)
        span2 = bsplines_kernels.find_span(tn2, int(pn[1]), eta2)
        span3 = bsplines_kernels.find_span(tn3, int(pn[2]), eta3)

        # compute bn, bd, i.e. values for non-vanishing B-/splines at position eta
        bsplines_kernels.b_splines_slim(tn1, int(pn[0]), eta1, span1, bn1)
        bsplines_kernels.b_splines_slim(tn2, int(pn[1]), eta2, span2, bn2)
        bsplines_kernels.b_splines_slim(tn3, int(pn[2]), eta3, span3, bn3)

        # call the appropriate matvec filler
        filler_kernels.fill_vec(int(pn[0]), int(pn[1]), int(pn[2]), bn1, bn2, bn3, span1, span2, span3,
                                starts, vec, filling)

    #$ omp end parallel


@stack_array('cell_left', 'point_left', 'point_right', 'cell_number', 'temp1', 'temp4', 'compact', 'grids_shapex', 'grids_shapey', 'grids_shapez')
def hybrid_fA_density(markers: 'float[:,:]', n_markers_tot: 'int',
                      pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                      starts: 'int[:]',
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
    cell_left = empty(3, dtype=int)
    point_left = zeros(3, dtype=float)
    point_right = zeros(3, dtype=float)
    cell_number = empty(3, dtype=int)

    temp1 = zeros(3, dtype=float)
    temp4 = zeros(3, dtype=float)

    compact = zeros(3, dtype=float)
    compact[0] = (p_shape[0]+1.0)*p_size[0]
    compact[1] = (p_shape[1]+1.0)*p_size[1]
    compact[2] = (p_shape[2]+1.0)*p_size[2]

    grids_shapex = zeros(p_shape[0] + 2, dtype=float)
    grids_shapey = zeros(p_shape[1] + 2, dtype=float)
    grids_shapez = zeros(p_shape[2] + 2, dtype=float)

    dfm = zeros((3, 3), dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private (dfm, det_df, cell_left, point_left, point_right, cell_number, temp1, temp4, compact, grids_shapex, grids_shapey, grids_shapez, n_markers, ip, eta1, eta2, eta3, weight, ie1, ie2, ie3, il1, il2, il3, jl1, jl2, jl3, i1, i2, i3, value_x, value_y, value_z, span1, span2, span3)
    #$ omp for reduction ( + : mat)
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta1, eta2, eta3,
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        weight = markers[ip, 6] / \
            (p_size[0]*p_size[1]*p_size[2])/n_markers_tot/det_df

        ie1 = int(eta1*Nel[0])
        ie2 = int(eta2*Nel[1])
        ie3 = int(eta3*Nel[2])

        # the points here are still not put in the periodic box [0, 1] x [0, 1] x [0, 1]
        point_left[0] = eta1 - 0.5*compact[0]
        point_right[0] = eta1 + 0.5*compact[0]
        point_left[1] = eta2 - 0.5*compact[1]
        point_right[1] = eta2 + 0.5*compact[1]
        point_left[2] = eta3 - 0.5*compact[2]
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

        span1 = int(eta1*Nel[0]) + int(pn[0])
        span2 = int(eta2*Nel[1]) + int(pn[1])
        span3 = int(eta3*Nel[2]) + int(pn[2])

        # =========== kernel part (periodic bundary case) ==========
        particle_to_mat_kernels.hybrid_density(Nel, pn, cell_left, cell_number, span1, span2, span3, starts, ie1, ie2, ie3, temp1, temp4, quad, quad_pts_x,
                                               quad_pts_y, quad_pts_z, compact, eta1, eta2, eta3, mat, weight, p_shape, p_size, grids_shapex, grids_shapey, grids_shapez)
    #$ omp end parallel


@stack_array('dfm', 'df_t', 'df_inv', 'df_inv_times_v', 'filling_m', 'filling_v', 'v')
def hybrid_fA_Arelated(markers: 'float[:,:]', n_markers_tot: 'int',
                       pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                       starts: 'int[:]',
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

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    # allocate for metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)

    # allocate for filling
    df_inv_times_v = empty(3, dtype=float)
    filling_m = empty((3, 3), dtype=float)
    filling_v = empty(3, dtype=float)
    v = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private (ip, eta1, eta2, eta3, v, dfm, df_inv, df_inv_times_v, weight, filling_m, filling_v)
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
        v[:] = markers[ip, 3:6]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta1, eta2, eta3,
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # filling functions
        linalg_kernels.matrix_inv(dfm, df_inv)
        linalg_kernels.matrix_vector(df_inv, v, df_inv_times_v)

        weight = markers[ip, 6]

        # filling_m
        filling_m[0, 0] = weight / n_markers_tot * \
            (df_inv[0, 0]*df_inv[0, 0] + df_inv[0, 1]
             * df_inv[0, 1] + df_inv[0, 2]*df_inv[0, 2])
        filling_m[0, 1] = weight / n_markers_tot * \
            (df_inv[0, 0]*df_inv[1, 0] + df_inv[0, 1]
             * df_inv[1, 1] + df_inv[0, 2]*df_inv[1, 2])
        filling_m[0, 2] = weight / n_markers_tot * \
            (df_inv[0, 0]*df_inv[2, 0] + df_inv[0, 1]
             * df_inv[2, 1] + df_inv[0, 2]*df_inv[2, 2])

        filling_m[1, 1] = weight / n_markers_tot * \
            (df_inv[1, 0]*df_inv[1, 0] + df_inv[1, 1]
             * df_inv[1, 1] + df_inv[1, 2]*df_inv[1, 2])
        filling_m[1, 2] = weight / n_markers_tot * \
            (df_inv[1, 0]*df_inv[2, 0] + df_inv[1, 1]
             * df_inv[2, 1] + df_inv[1, 2]*df_inv[2, 2])

        filling_m[2, 2] = weight / n_markers_tot * \
            (df_inv[2, 0]*df_inv[2, 0] + df_inv[2, 1]
             * df_inv[2, 1] + df_inv[2, 2]*df_inv[2, 2])

        # filling_v
        filling_v[:] = weight / n_markers_tot * df_inv_times_v

        # call the appropriate matvec filler
        particle_to_mat_kernels.m_v_fill_b_v1_symm(pn, tn1, tn2, tn3, starts,
                                                   eta1, eta2, eta3,
                                                   mat11, mat12, mat13, mat22, mat23, mat33,
                                                   filling_m[0, 0], filling_m[0,
                                                                              1], filling_m[0, 2],
                                                   filling_m[1, 1], filling_m[1,
                                                                              2], filling_m[2, 2],
                                                   vec1, vec2, vec3,
                                                   filling_v[0], filling_v[1], filling_v[2])

    #$ omp end parallel


@stack_array('bn1', 'bn2', 'bn3')
def linear_vlasov_maxwell_poisson(markers: 'float[:,:]', n_markers_tot: 'int',
                                  pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                                  starts: 'int[:]',
                                  kind_map: 'int', params_map: 'float[:]',
                                  p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                                  ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                                  cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                                  vec: 'float[:,:,:]',
                                  # model specific argument
                                  f0_values: 'float[:]',
                                  # model specific argument
                                  f0_params: 'float[:]',
                                  alpha: 'float',  # model specific argument
                                  kappa: 'float'):  # model specific argument
    r"""
    Accumulates the charge density in V0 

    .. math::

        \rho_p^\mu = \alpha^2 \sqrt{f_0(\mathbf{\eta}_p, \mathbf{v}_p)} w_p [ DF^{-1}(\mathbf{\eta}_p) \mathbf{v}_p ]_\mu \,.

    Parameters
    ----------
        f0_values ; array[float]
            Value of f0 for each particle.

        f0_params : array[float]
            Parameters needed to specify the moments; the order is specified in :ref:`kinetic_moments` for the respective functions available.

        alpha : float
            = Omega_c / Omega_p ; Parameter determining the coupling strength between particles and fields

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    # get number of markers
    n_markers = shape(markers)[0]

    # non-vanishing B-splines at particle position
    bn1 = empty(int(pn[0]) + 1, dtype=float)
    bn2 = empty(int(pn[1]) + 1, dtype=float)
    bn3 = empty(int(pn[2]) + 1, dtype=float)

    #$ omp parallel private (ip, eta1, eta2, eta3, f0, filling)
    #$ omp for reduction ( + :vec)
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        f0 = f0_values[ip]

        # filling = alpha^2 * kappa * w_p * sqrt{f_0} / N
        filling = alpha**2 * kappa * markers[ip, 6] * sqrt(f0) / n_markers_tot

        # spans (i.e. index for non-vanishing B-spline basis functions)
        span1 = bsplines_kernels.find_span(tn1, int(pn[0]), eta1)
        span2 = bsplines_kernels.find_span(tn2, int(pn[1]), eta2)
        span3 = bsplines_kernels.find_span(tn3, int(pn[2]), eta3)

        # compute bn, bd, i.e. values for non-vanishing B-/splines at position eta
        bsplines_kernels.b_splines_slim(tn1, int(pn[0]), eta1, span1, bn1)
        bsplines_kernels.b_splines_slim(tn2, int(pn[1]), eta2, span2, bn2)
        bsplines_kernels.b_splines_slim(tn3, int(pn[2]), eta3, span3, bn3)

        # call the appropriate matvec filler
        filler_kernels.fill_vec(int(pn[0]), int(pn[1]), int(pn[2]), bn1, bn2, bn3, span1, span2, span3,
                                starts, vec, filling)

    #$ omp end parallel


@stack_array('dfm', 'df_inv', 'v', 'df_inv_times_v', 'filling_m', 'filling_v')
def linear_vlasov_maxwell(markers: 'float[:,:]', n_markers_tot: 'int',
                          pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                          starts: 'int[:]',
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
                          f0_values: 'float[:]',  # model specific argument
                          vth: 'float',  # model specific argument
                          alpha: 'float',  # model specific argument
                          kappa: 'float'):  # model specific argument
    r"""
    Accumulates into V1 with the filling functions

    .. math::

        A_p^{\mu, \nu} &= \frac{\alpha^2 \kappa^2}{v_{\text{th}}^2} \frac{1}{N\, s_0} f_0(\mathbf{\eta}_p, \mathbf{v}_p)
            [ DF^{-1}(\mathbf{\eta}_p) v_p ]_\mu [ DF^{-1}(\mathbf{\eta}_p) \mathbf{v}_p ]_\nu \,,

        B_p^\mu &= \alpha^2 \kappa \sqrt{f_0(\mathbf{\eta}_p, \mathbf{v}_p)} w_p [ DF^{-1}(\mathbf{\eta}_p) \mathbf{v}_p ]_\mu \,.

    Parameters
    ----------
        f0_values ; array[float]
            Value of f0 for each particle.

        f0_params : array[float]
            Parameters needed to specify the moments; the order is specified in :ref:`kinetic_moments` for the respective functions available.

        alpha : float
            = Omega_p / Omega_c ; Parameter determining the coupling strength between particles and fields

        kappa : float
            = 2 * pi * Omega_c / omega ; Parameter determining the coupling strength between particles and fields

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    # allocate for metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)

    # allocate for filling
    v = empty(3, dtype=float)
    df_inv_v = empty(3, dtype=float)
    filling_m = empty((3, 3), dtype=float)
    filling_v = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private (ip, eta1, eta2, eta3, f0, dfm, df_inv, v, df_inv_times_v, filling_m, filling_v)
    #$ omp for reduction ( + : mat11, mat12, mat13, mat22, mat23, mat33, vec1, vec2, vec3)
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # get velocity
        v[0] = markers[ip, 3]
        v[1] = markers[ip, 4]
        v[2] = markers[ip, 5]

        f0 = f0_values[ip]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta1, eta2, eta3,
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # invert Jacobian matrix
        linalg_kernels.matrix_inv(dfm, df_inv)

        # compute DF^{-1} v
        linalg_kernels.matrix_vector(df_inv, v, df_inv_v)

        # filling_m = alpha^2 * kappa^2 * f0 / (N * s_0 * v_th^2) * (DF^{-1} v_p)_mu * (DF^{-1} v_p)_nu
        linalg_kernels.outer(df_inv_v, df_inv_v, filling_m)
        filling_m[:, :] *= alpha**2 * kappa**2 * f0 / \
            (vth**2 * n_markers_tot * markers[ip, 7])

        # filling_v = alpha^2 * kappa / N * w_p * sqrt{f_0} DL^{-1} * v_p
        filling_v[:] = alpha**2 * kappa * \
            sqrt(f0) * markers[ip, 6] * df_inv_v / n_markers_tot

        # call the appropriate matvec filler
        particle_to_mat_kernels.m_v_fill_b_v1_symm(pn,
                                                   tn1, tn2, tn3,
                                                   starts,
                                                   eta1, eta2, eta3,
                                                   mat11, mat12, mat13, mat22, mat23, mat33,
                                                   filling_m[0, 0], filling_m[0,
                                                                              1], filling_m[0, 2],
                                                   filling_m[1, 1], filling_m[1,
                                                                              2], filling_m[2, 2],
                                                   vec1, vec2, vec3,
                                                   filling_v[0], filling_v[1], filling_v[2])

    #$ omp end parallel


@stack_array('bn1', 'bn2', 'bn3')
def vlasov_maxwell_poisson(markers: 'float[:,:]', n_markers_tot: 'int',
                           pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                           starts: 'int[:]',
                           kind_map: 'int', params_map: 'float[:]',
                           p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                           ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                           cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                           vec: 'float[:,:,:]'):
    r"""
    Accumulates the charge density in V0 

    .. math::

        \rho_p^\mu = w_p \,.

    Parameters
    ----------

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    # non-vanishing B-splines at particle position
    bn1 = empty(int(pn[0]) + 1, dtype=float)
    bn2 = empty(int(pn[1]) + 1, dtype=float)
    bn3 = empty(int(pn[2]) + 1, dtype=float)

    #$ omp parallel private (ip, eta1, eta2, eta3, f0, filling)
    #$ omp for reduction ( + :vec)
    for ip in range(shape(markers)[0]):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # filling = w_p
        filling = markers[ip, 6] / n_markers_tot

        # spans (i.e. index for non-vanishing B-spline basis functions)
        span1 = bsplines_kernels.find_span(tn1, int(pn[0]), eta1)
        span2 = bsplines_kernels.find_span(tn2, int(pn[1]), eta2)
        span3 = bsplines_kernels.find_span(tn3, int(pn[2]), eta3)

        # compute bn, bd, i.e. values for non-vanishing B-/splines at position eta
        bsplines_kernels.b_splines_slim(tn1, int(pn[0]), eta1, span1, bn1)
        bsplines_kernels.b_splines_slim(tn2, int(pn[1]), eta2, span2, bn2)
        bsplines_kernels.b_splines_slim(tn3, int(pn[2]), eta3, span3, bn3)

        # call the appropriate matvec filler
        filler_kernels.fill_vec(int(pn[0]), int(pn[1]), int(pn[2]), bn1, bn2, bn3, span1, span2, span3,
                                starts, vec, filling)

    #$ omp end parallel


@stack_array('dfm', 'df_inv', 'df_inv_t', 'g_inv', 'v', 'df_inv_times_v', 'filling_m', 'filling_v')
def vlasov_maxwell(markers: 'float[:,:]', n_markers_tot: 'int',
                   pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                   starts: 'int[:]',
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
                   vec3: 'float[:,:,:]'):
    r"""
    Accumulates into V1 with the filling functions

    .. math::

        A_p^{\mu, \nu} &= w_p [ DF^{-1}(\eta_p) DF^{-\top}(\eta_p) ]_{\mu, \nu} \,,

        B_p^\mu &= w_p [ DF^{-1}(\mathbf{\eta}_p) \mathbf{v}_p ]_\mu \,.

    Parameters
    ----------

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    # allocate for metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # allocate for filling
    v = empty(3, dtype=float)
    df_inv_times_v = empty(3, dtype=float)
    filling_m = empty((3, 3), dtype=float)
    filling_v = empty(3, dtype=float)

    #$ omp parallel private (ip, eta1, eta2, eta3, f0, dfm, df_inv, v, df_inv_times_v, filling_m, filling_v)
    #$ omp for reduction ( + : mat11, mat12, mat13, mat22, mat23, mat33, vec1, vec2, vec3)
    for ip in range(shape(markers)[0]):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta1, eta2, eta3,
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # compute shifted and stretched velocity
        v[0] = markers[ip, 3]
        v[1] = markers[ip, 4]
        v[2] = markers[ip, 5]

        # filling functions
        linalg_kernels.matrix_inv(dfm, df_inv)
        linalg_kernels.transpose(df_inv, df_inv_t)
        linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)
        linalg_kernels.matrix_vector(df_inv, v, df_inv_times_v)

        # filling_m = w_p * DF^{-1} * DF^{-T}
        filling_m[:, :] = markers[ip, 6] * g_inv / n_markers_tot

        # filling_v = w_p * DF^{-1} * \V
        filling_v[:] = markers[ip, 6] * df_inv_times_v / n_markers_tot

        # call the appropriate matvec filler
        particle_to_mat_kernels.m_v_fill_b_v1_symm(pn,
                                                   tn1, tn2, tn3,
                                                   starts,
                                                   eta1, eta2, eta3,
                                                   mat11, mat12, mat13, mat22, mat23, mat33,
                                                   filling_m[0, 0], filling_m[0,
                                                                              1], filling_m[0, 2],
                                                   filling_m[1, 1], filling_m[1,
                                                                              2], filling_m[2, 2],
                                                   vec1, vec2, vec3,
                                                   filling_v[0], filling_v[1], filling_v[2])

    #$ omp end parallel


def delta_f_vlasov_maxwell_poisson(markers: 'float[:,:]', n_markers_tot: 'int',
                                   pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                                   starts: 'int[:]',
                                   kind_map: 'int', params_map: 'float[:]',
                                   p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                                   ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                                   cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                                   vec: 'float[:,:,:]',
                                   # model specific argument
                                   f0_values: 'float[:]',
                                   # model specific argument
                                   f0_params: 'float[:]',
                                   alpha: 'float',  # model specific argument
                                   kappa: 'float'):  # model specific argument
    r"""
    Accumulates the charge density in V0 

    .. math::

        \rho_p^\mu = \alpha^2 \sqrt{f_0(\mathbf{\eta}_p, \mathbf{v}_p)} w_p [ DF^{-1}(\mathbf{\eta}_p) \mathbf{v}_p ]_\mu \,.

    Parameters
    ----------
    f0_values ; array[float]
        Value of f0 for each particle.

    f0_params : array[float]
        Parameters needed to specify the moments; the order is specified in :ref:`kinetic_moments` for the respective functions available.

    alpha : float
        = Omega_c / Omega_p ; Parameter determining the coupling strength between particles and fields

    Note
    ----
    The above parameter list contains only the model specific input arguments.
    """

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private (ip, eta1, eta2, eta3, f0, filling)
    #$ omp for reduction ( + :vec)
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        f0 = f0_values[ip]

        # filling = alpha^2 * kappa * (1 / (N * s_0) * (f_0 / log(f_0) - f_0) - w_p / log(f_0))
        filling = alpha**2 * kappa * ((f0 / log(f0) - f0) / (n_markers_tot * markers[ip, 7]) - markers[ip, 6] / log(
            f0)) * f0_params[4]**2 * f0_params[5]**2 * f0_params[6]**2

        # call the appropriate matvec filler
        particle_to_mat_kernels.scalar_fill_b_v0(pn, tn1, tn2, tn3,
                                                 starts, eta1, eta2, eta3,
                                                 vec, filling)

    #$ omp end parallel


@stack_array('dfm', 'df_inv', 'v', 'df_inv_times_v', 'filling_v')
def delta_f_vlasov_maxwell(markers: 'float[:,:]', n_markers_tot: 'int',
                           pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                           starts: 'int[:]',
                           kind_map: 'int', params_map: 'float[:]',
                           p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                           ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                           cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                           vec1: 'float[:,:,:]',
                           vec2: 'float[:,:,:]',
                           vec3: 'float[:,:,:]',
                           f0_values: 'float[:]',  # model specific argument
                           alpha: 'float',  # model specific argument
                           kappa: 'float',
                           substep: 'int'):  # model specific argument
    r"""
    Accumulates vector into V1 with the filling functions

    .. math::

        B_p^\mu &= \frac{\alpha^2 \kappa}{N \, s_0} \left( \frac{f_0}{\ln(f_0)} - f_0 \right) [ DF^{-1}(\mathbf{\eta}_p) \mathbf{v}_p ]_\mu \,.

    Parameters
    ----------
    f0_values ; array[float]
        Value of f0 for each particle.

    f0_params : array[float]
        Parameters needed to specify the moments; the order is specified in :ref:`kinetic_moments` for the respective functions available.

    alpha : float
        = Omega_c / Omega_p ; Parameter determining the coupling strength between particles and fields

    Note
    ----
    The above parameter list contains only the model specific input arguments.
    """

    # allocate for metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)

    # allocate for filling
    v = empty(3, dtype=float)
    df_inv_times_v = empty(3, dtype=float)
    filling_v = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private (ip, eta1, eta2, eta3, f0, dfm, df_inv, v, df_inv_times_v, filling_v)
    #$ omp for reduction ( + : mat11, mat12, mat13, mat22, mat23, mat33, vec1, vec2, vec3)
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        f0 = f0_values[ip]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta1, eta2, eta3,
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # compute shifted and stretched velocity
        v[0] = markers[ip, 3]
        v[1] = markers[ip, 4]
        v[2] = markers[ip, 5]

        # filling functions
        linalg_kernels.matrix_inv(dfm, df_inv)
        linalg_kernels.matrix_vector(df_inv, v, filling_v)

        if substep == 0:
            # filling_v = alpha^2 / (N * s_0) * (f_0 / ln(f_0) - f_0) * DL^{-1} * v_p
            filling_v[:] *= alpha**2 * kappa / (n_markers_tot * markers[ip, 7]) * \
                (f0 / log(f0) - f0)
        elif substep == 1:
            # filling_v = alpha^2 * kappa * w_p / (N * ln(f_0)) * DL^{-1} * v_p
            filling_v[:] *= alpha**2 * kappa * \
                markers[ip, 6] / (n_markers_tot * log(f0))

        # call the appropriate matvec filler
        particle_to_mat_kernels.vec_fill_b_v1(pn, tn1, tn2, tn3, starts,
                                              eta1, eta2, eta3,
                                              vec1, vec2, vec3,
                                              filling_v[0], filling_v[1], filling_v[2])

    #$ omp end parallel


@stack_array('dfm', 'df_inv', 'v', 'df_inv_times_v', 'filling_v', 'filling_m')
def delta_f_vlasov_maxwell_scn(markers: 'float[:,:]', n_markers_tot: 'int',
                               pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                               starts: 'int[:]',
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
                               # model specific argument
                               f0_values: 'float[:]',
                               vth: 'float',  # model specific argument
                               alpha: 'float',  # model specific argument
                               kappa: 'float'):  # model specific argument
    r"""
    Accumulates vector into V1 with the filling functions

    .. math::

        B_p^\mu &= \frac{\alpha^2 \kappa}{N \, s_0} \left( \frac{f_0}{\ln(f_0)} - f_0 \right) [ DF^{-1}(\mathbf{\eta}_p) \mathbf{v}_p ]_\mu \,.

    Parameters
    ----------
    f0_values ; array[float]
        Value of f0 for each particle.

    f0_params : array[float]
        Parameters needed to specify the moments; the order is specified in :ref:`kinetic_moments` for the respective functions available.

    alpha : float
        = Omega_c / Omega_p ; Parameter determining the coupling strength between particles and fields

    Note
    ----
    The above parameter list contains only the model specific input arguments.
    """

    # allocate for metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)

    # allocate for filling
    v = empty(3, dtype=float)
    df_inv_times_v = empty(3, dtype=float)
    filling_m = empty((3, 3), dtype=float)
    filling_v = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private (ip, eta1, eta2, eta3, f0, dfm, df_inv, v, df_inv_times_v, filling_v)
    #$ omp for reduction ( + : mat11, mat12, mat13, mat22, mat23, mat33, vec1, vec2, vec3)
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        f0 = f0_values[ip]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta1, eta2, eta3,
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # compute shifted and stretched velocity
        v[0] = markers[ip, 3]
        v[1] = markers[ip, 4]
        v[2] = markers[ip, 5]

        # filling functions
        linalg_kernels.matrix_inv(dfm, df_inv)
        linalg_kernels.matrix_vector(df_inv, v, df_inv_times_v)
        linalg_kernels.outer(df_inv_times_v, df_inv_times_v, filling_m)

        # filling_m = alpha^2 * kappa^2 * w_p / (N * vth^2 * log^2(f_0))
        filling_m[:, :] *= alpha**2 * kappa**2 * markers[ip, 6] / \
            (n_markers_tot * vth**2 * log(f0)**2)

        # filling_v = alpha^2 * kappa * w_p / (N * ln(f_0)) * DL^{-1} * v_p
        filling_v[:] = alpha**2 * kappa * markers[ip, 6] * df_inv_times_v[:] / \
            (n_markers_tot * log(f0))

        # call the appropriate matvec filler
        particle_to_mat_kernels.m_v_fill_b_v1_symm(pn, tn1, tn2, tn3, starts,
                                                   eta1, eta2, eta3,
                                                   mat11, mat12, mat13,
                                                   mat22, mat23, mat33,
                                                   filling_m[0, 0],
                                                   filling_m[0, 1],
                                                   filling_m[0, 2],
                                                   filling_m[1, 1],
                                                   filling_m[1, 2],
                                                   filling_m[2, 2],
                                                   vec1, vec2, vec3,
                                                   filling_v[0], filling_v[1], filling_v[2])

    #$ omp end parallel


@stack_array('b', 'b_prod', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'dfm', 'df_inv', 'df_inv_t' 'g_inv', 'tmp1', 'tmp2')
def cc_lin_mhd_6d_1(markers: 'float[:,:]', n_markers_tot: 'int',
                    pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                    starts: 'int[:]',
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
                    basis_u: 'int', scale_mat: 'float'):  # model specific argument
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

    bn1 = empty(int(pn[0]) + 1, dtype=float)
    bn2 = empty(int(pn[1]) + 1, dtype=float)
    bn3 = empty(int(pn[2]) + 1, dtype=float)

    bd1 = empty(int(pn[0]), dtype=float)
    bd2 = empty(int(pn[1]), dtype=float)
    bd3 = empty(int(pn[2]), dtype=float)

    # allocate for metric coefficients
    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # allocate some temporary buffers for filling
    tmp1 = empty((3, 3), dtype=float)
    tmp2 = empty((3, 3), dtype=float)

    # get local number of markers
    n_markers_loc = shape(markers)[0]

    #$ omp parallel firstprivate(b_prod) private(ip, eta1, eta2, eta3, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, b, dfm, det_df, weight, df_inv, df_inv_t, g_inv, tmp1, tmp2, filling_m12, filling_m13, filling_m23)
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
        span1 = bsplines_kernels.find_span(tn1, int(pn[0]), eta1)
        span2 = bsplines_kernels.find_span(tn2, int(pn[1]), eta2)
        span3 = bsplines_kernels.find_span(tn3, int(pn[2]), eta3)

        bsplines_kernels.b_d_splines_slim(
            tn1, int(pn[0]), eta1, span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(
            tn2, int(pn[1]), eta2, span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(
            tn3, int(pn[2]), eta3, span3, bn3, bd3)

        b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]), int(pn[1]) - 1, int(pn[2]) - 1, bn1, bd2, bd3, span1, span2, span3, b2_1, starts)
        b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]) - 1, int(pn[1]), int(pn[2]) - 1, bd1, bn2, bd3, span1, span2, span3, b2_2, starts)
        b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]) - 1, int(pn[1]) - 1, int(pn[2]), bd1, bd2, bn3, span1, span2, span3, b2_3, starts)

        # operator bx() as matrix
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] = +b[1]
        b_prod[1, 0] = +b[2]
        b_prod[1, 2] = -b[0]
        b_prod[2, 0] = -b[1]
        b_prod[2, 1] = +b[0]

        # evaluate Jacobian matrix and Jacobian determinant
        evaluation_kernels.df(eta1, eta2, eta3,
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        det_df = linalg_kernels.det(dfm)

        # marker weight
        weight = markers[ip, 6]

        if basis_u == 0:

            # filling functions
            filling_m12 = - weight * b_prod[0, 1] * scale_mat
            filling_m13 = - weight * b_prod[0, 2] * scale_mat
            filling_m23 = - weight * b_prod[1, 2] * scale_mat

            # call the appropriate matvec filler
            particle_to_mat_kernels.mat_fill_v0vec_asym(pn, span1, span2, span3,
                                                        bn1, bn2, bn3,
                                                        starts,
                                                        mat12, mat13, mat23,
                                                        filling_m12, filling_m13, filling_m23)

        elif basis_u == 1:

            # filling functions
            linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
            linalg_kernels.transpose(df_inv, df_inv_t)
            linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)
            linalg_kernels.matrix_matrix(g_inv, b_prod, tmp1)
            linalg_kernels.matrix_matrix(tmp1, g_inv, tmp2)

            filling_m12 = - weight * tmp2[0, 1] * scale_mat
            filling_m13 = - weight * tmp2[0, 2] * scale_mat
            filling_m23 = - weight * tmp2[1, 2] * scale_mat

            # call the appropriate matvec filler
            particle_to_mat_kernels.mat_fill_v1_asym(pn, span1, span2, span3,
                                                     bn1, bn2, bn3,
                                                     bd1, bd2, bd3,
                                                     starts,
                                                     mat12, mat13, mat23,
                                                     filling_m12, filling_m13, filling_m23)

        elif basis_u == 2:

            # filling functions
            filling_m12 = - weight * b_prod[0, 1] * scale_mat / det_df**2
            filling_m13 = - weight * b_prod[0, 2] * scale_mat / det_df**2
            filling_m23 = - weight * b_prod[1, 2] * scale_mat / det_df**2

            # call the appropriate matvec filler
            particle_to_mat_kernels.mat_fill_v2_asym(pn, span1, span2, span3,
                                                     bn1, bn2, bn3,
                                                     bd1, bd2, bd3,
                                                     starts,
                                                     mat12, mat13, mat23,
                                                     filling_m12, filling_m13, filling_m23)

    #$ omp end parallel

    mat12 /= n_markers_tot
    mat13 /= n_markers_tot
    mat23 /= n_markers_tot


@stack_array('b', 'b_prod', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'dfm', 'df_inv', 'df_inv_t', 'g_inv', 'filling_m', 'filling_v', 'tmp1', 'tmp2', 'tmp_t', 'tmp_v', 'tmp_m', 'v')
def cc_lin_mhd_6d_2(markers: 'float[:,:]', n_markers_tot: 'int',
                    pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                    starts: 'int[:]',
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
                    basis_u: 'int', scale_mat: 'float', scale_vec: 'float'):  # model specific argument
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

    bn1 = empty(int(pn[0]) + 1, dtype=float)
    bn2 = empty(int(pn[1]) + 1, dtype=float)
    bn3 = empty(int(pn[2]) + 1, dtype=float)

    bd1 = empty(int(pn[0]), dtype=float)
    bd2 = empty(int(pn[1]), dtype=float)
    bd3 = empty(int(pn[2]), dtype=float)

    # allocate for metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # allocate for filling
    filling_m = empty((3, 3), dtype=float)
    filling_v = empty(3, dtype=float)
    v = empty(3, dtype=float)

    tmp1 = empty((3, 3), dtype=float)
    tmp2 = empty((3, 3), dtype=float)

    tmp_t = empty((3, 3), dtype=float)
    tmp_m = empty((3, 3), dtype=float)

    tmp_v = empty(3, dtype=float)

    # get number of markers
    n_markers_loc = shape(markers)[0]

    #$ omp parallel firstprivate(b_prod) private(ip, eta1, eta2, eta3, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, b, dfm, det_df, weight, v, df_inv, df_inv_t, g_inv, tmp1, tmp2, tmp_t, tmp_m, tmp_v, filling_m, filling_v)
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
        span1 = bsplines_kernels.find_span(tn1, int(pn[0]), eta1)
        span2 = bsplines_kernels.find_span(tn2, int(pn[1]), eta2)
        span3 = bsplines_kernels.find_span(tn3, int(pn[2]), eta3)

        bsplines_kernels.b_d_splines_slim(
            tn1, int(pn[0]), eta1, span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(
            tn2, int(pn[1]), eta2, span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(
            tn3, int(pn[2]), eta3, span3, bn3, bd3)

        b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]), int(pn[1]) - 1, int(pn[2]) - 1, bn1, bd2, bd3, span1, span2, span3, b2_1, starts)
        b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]) - 1, int(pn[1]), int(pn[2]) - 1, bd1, bn2, bd3, span1, span2, span3, b2_2, starts)
        b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]) - 1, int(pn[1]) - 1, int(pn[2]), bd1, bd2, bn3, span1, span2, span3, b2_3, starts)

        # operator bx() as matrix
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] = +b[1]
        b_prod[1, 0] = +b[2]
        b_prod[1, 2] = -b[0]
        b_prod[2, 0] = -b[1]
        b_prod[2, 1] = +b[0]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta1, eta2, eta3,
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        det_df = linalg_kernels.det(dfm)

        # marker weight and velocity
        weight = markers[ip, 6]
        v[:] = markers[ip, 3:6]

        if basis_u == 0:

            # needed metric coefficients
            linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
            linalg_kernels.transpose(df_inv, df_inv_t)
            linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)

            # filling functions tmp_m = tmp1 * tmp1^T and tmp_v = tmp1 * v, where tmp1 = B^x * DF^(-1)
            linalg_kernels.matrix_matrix(b_prod, df_inv, tmp1)

            linalg_kernels.transpose(tmp1, tmp_t)

            linalg_kernels.matrix_matrix(tmp1, tmp_t, tmp_m)
            linalg_kernels.matrix_vector(tmp1, v, tmp_v)

            filling_m[:, :] = weight * tmp_m * scale_mat
            filling_v[:] = weight * tmp_v * scale_vec

            # call the appropriate matvec filler
            particle_to_mat_kernels.m_v_fill_v0vec_symm(pn, span1, span2, span3,
                                                        bn1, bn2, bn3,
                                                        starts,
                                                        mat11, mat12, mat13,
                                                        mat22, mat23,
                                                        mat33,
                                                        filling_m[0, 0], filling_m[0,
                                                                                   1], filling_m[0, 2],
                                                        filling_m[1,
                                                                  1], filling_m[1, 2],
                                                        filling_m[2, 2],
                                                        vec1, vec2, vec3,
                                                        filling_v[0], filling_v[1], filling_v[2])

        elif basis_u == 1:

            # needed metric coefficients
            linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
            linalg_kernels.transpose(df_inv, df_inv_t)
            linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)

            # filling functions tmp_m = tmp2 * tmp2^T and tmp_v = tmp2 * v, where tmp2 = G^(-1) * B^x * DF^(-1)
            linalg_kernels.matrix_matrix(g_inv, b_prod, tmp1)
            linalg_kernels.matrix_matrix(tmp1, df_inv, tmp2)

            linalg_kernels.transpose(tmp2, tmp_t)

            linalg_kernels.matrix_matrix(tmp2, tmp_t, tmp_m)
            linalg_kernels.matrix_vector(tmp2, v, tmp_v)

            filling_m[:, :] = weight * tmp_m * scale_mat
            filling_v[:] = weight * tmp_v * scale_vec

            # call the appropriate matvec filler
            particle_to_mat_kernels.m_v_fill_v1_symm(pn, span1, span2, span3,
                                                     bn1, bn2, bn3,
                                                     bd1, bd2, bd3,
                                                     starts,
                                                     mat11, mat12, mat13,
                                                     mat22, mat23,
                                                     mat33,
                                                     filling_m[0, 0], filling_m[0,
                                                                                1], filling_m[0, 2],
                                                     filling_m[1,
                                                               1], filling_m[1, 2],
                                                     filling_m[2, 2],
                                                     vec1, vec2, vec3,
                                                     filling_v[0], filling_v[1], filling_v[2])

        elif basis_u == 2:

            # needed metric coefficients
            linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
            linalg_kernels.transpose(df_inv, df_inv_t)
            linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)

            # filling functions tmp_m = tmp1 * tmp1^T and tmp_v = tmp1 * v, where tmp1 = B^x * DF^(-1) / det(DF)
            linalg_kernels.matrix_matrix(b_prod, df_inv, tmp1)

            linalg_kernels.transpose(tmp1, tmp_t)

            linalg_kernels.matrix_matrix(tmp1, tmp_t, tmp_m)
            linalg_kernels.matrix_vector(tmp1, v, tmp_v)

            filling_m[:, :] = weight * tmp_m * scale_mat / det_df**2
            filling_v[:] = weight * tmp_v * scale_vec / det_df

            # call the appropriate matvec filler
            particle_to_mat_kernels.m_v_fill_v2_symm(pn, span1, span2, span3,
                                                     bn1, bn2, bn3,
                                                     bd1, bd2, bd3,
                                                     starts,
                                                     mat11, mat12, mat13,
                                                     mat22, mat23,
                                                     mat33,
                                                     filling_m[0, 0], filling_m[0,
                                                                                1], filling_m[0, 2],
                                                     filling_m[1,
                                                               1], filling_m[1, 2],
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


@stack_array('dfm', 'df_t', 'df_inv', 'df_inv_t', 'filling_m', 'filling_v', 'tmp1', 'v', 'tmp_v', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def pc_lin_mhd_6d_full(markers: 'float[:,:]', n_markers_tot: 'int',
                       pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                       starts: 'int[:]',
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
                       vec3_3: 'float[:,:,:]',
                       scale_mat: 'float', scale_vec: 'float'):
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
    dfm = empty((3, 3), dtype=float)
    df_t = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)

    # allocate for filling
    filling_m = empty((3, 3), dtype=float)
    filling_v = empty(3, dtype=float)

    tmp1 = empty((3, 3), dtype=float)

    v = empty(3, dtype=float)
    tmp_v = empty(3, dtype=float)

    bn1 = empty(int(pn[0]) + 1, dtype=float)
    bn2 = empty(int(pn[1]) + 1, dtype=float)
    bn3 = empty(int(pn[2]) + 1, dtype=float)

    bd1 = empty(int(pn[0]), dtype=float)
    bd2 = empty(int(pn[1]), dtype=float)
    bd3 = empty(int(pn[2]), dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # b-field evaluation
        span1 = bsplines_kernels.find_span(tn1, int(pn[0]), eta1)
        span2 = bsplines_kernels.find_span(tn2, int(pn[1]), eta2)
        span3 = bsplines_kernels.find_span(tn3, int(pn[2]), eta3)

        bsplines_kernels.b_d_splines_slim(
            tn1, int(pn[0]), eta1, span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(
            tn2, int(pn[1]), eta2, span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(
            tn3, int(pn[2]), eta3, span3, bn3, bd3)

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta1, eta2, eta3,
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # Avoid second computation of dfm, use linear_algebra.linalg_kernels routines to get g_inv:
        linalg_kernels.matrix_inv(dfm, df_inv)
        linalg_kernels.transpose(dfm, df_t)
        linalg_kernels.transpose(df_inv, df_inv_t)

        # filling functions
        v[:] = markers[ip, 3:6]

        linalg_kernels.matrix_matrix(df_inv, df_inv_t, tmp1)
        linalg_kernels.matrix_vector(df_inv, v, tmp_v)

        weight = markers[ip, 8]

        filling_m[:, :] = weight * tmp1 / n_markers_tot * scale_mat
        filling_v[:] = weight * tmp_v / n_markers_tot * scale_vec

        # call the appropriate matvec filler
        particle_to_mat_kernels.m_v_fill_v1_pressure_full(pn, span1, span2, span3,
                                                          bn1, bn2, bn3,
                                                          bd1, bd2, bd3,
                                                          starts,
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
                                                          filling_m[0, 0], filling_m[0,
                                                                                     1], filling_m[0, 2],
                                                          filling_m[1,
                                                                    1], filling_m[1, 2],
                                                          filling_m[2, 2],
                                                          vec1_1, vec2_1, vec3_1,
                                                          vec1_2, vec2_2, vec3_2,
                                                          vec1_3, vec2_3, vec3_3,
                                                          filling_v[0], filling_v[1], filling_v[2],
                                                          v[0], v[1], v[2])


@stack_array('dfm', 'df_inv_t', 'df_inv', 'filling_m', 'filling_v', 'tmp1', 'v', 'tmp_v', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def pc_lin_mhd_6d(markers: 'float[:,:]', n_markers_tot: 'int',
                  pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                  starts: 'int[:]',
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
                  vec3_3: 'float[:,:,:]',
                  scale_mat: 'float', scale_vec: 'float'):
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
    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)

    # allocate for filling
    filling_m = empty((3, 3), dtype=float)
    filling_v = empty(3, dtype=float)

    tmp1 = empty((3, 3), dtype=float)

    v = empty(3, dtype=float)
    tmp_v = empty(3, dtype=float)

    bn1 = empty(int(pn[0]) + 1, dtype=float)
    bn2 = empty(int(pn[1]) + 1, dtype=float)
    bn3 = empty(int(pn[2]) + 1, dtype=float)

    bd1 = empty(int(pn[0]), dtype=float)
    bd2 = empty(int(pn[1]), dtype=float)
    bd3 = empty(int(pn[2]), dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # marker weight and velocity
        weight = markers[ip, 6]
        v[:] = markers[ip, 3:6]

        # evaluation
        span1 = bsplines_kernels.find_span(tn1, int(pn[0]), eta1)
        span2 = bsplines_kernels.find_span(tn2, int(pn[1]), eta2)
        span3 = bsplines_kernels.find_span(tn3, int(pn[2]), eta3)

        bsplines_kernels.b_d_splines_slim(
            tn1, int(pn[0]), eta1, span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(
            tn2, int(pn[1]), eta2, span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(
            tn3, int(pn[2]), eta3, span3, bn3, bd3)

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta1, eta2, eta3,
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        det_df = linalg_kernels.det(dfm)

        # Avoid second computation of dfm, use linear_algebra.linalg_kernels routines to get g_inv:
        linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
        linalg_kernels.transpose(df_inv, df_inv_t)

        linalg_kernels.matrix_matrix(df_inv, df_inv_t, tmp1)
        linalg_kernels.matrix_vector(df_inv, v, tmp_v)

        filling_m[:, :] = weight * tmp1 * scale_mat
        filling_v[:] = weight * tmp_v * scale_vec

        # call the appropriate matvec filler
        particle_to_mat_kernels.m_v_fill_v1_pressure(pn, span1, span2, span3,
                                                     bn1, bn2, bn3,
                                                     bd1, bd2, bd3,
                                                     starts,
                                                     mat11_11, mat12_11, mat13_11,
                                                     mat22_11, mat23_11,
                                                     mat33_11,
                                                     mat11_12, mat12_12, mat13_12,
                                                     mat22_12, mat23_12,
                                                     mat33_12,
                                                     mat11_22, mat12_22, mat13_22,
                                                     mat22_22, mat23_22,
                                                     mat33_22,
                                                     filling_m[0, 0], filling_m[0,
                                                                                1], filling_m[0, 2],
                                                     filling_m[1,
                                                               1], filling_m[1, 2],
                                                     filling_m[2, 2],
                                                     vec1_1, vec2_1, vec3_1,
                                                     vec1_2, vec2_2, vec3_2,
                                                     filling_v[0], filling_v[1], filling_v[2],
                                                     v[0], v[1])

    mat11_11 /= n_markers_tot
    mat12_11 /= n_markers_tot
    mat13_11 /= n_markers_tot
    mat22_11 /= n_markers_tot
    mat23_11 /= n_markers_tot
    mat33_11 /= n_markers_tot
    mat11_12 /= n_markers_tot
    mat12_12 /= n_markers_tot
    mat13_12 /= n_markers_tot
    mat22_12 /= n_markers_tot
    mat23_12 /= n_markers_tot
    mat33_12 /= n_markers_tot
    mat11_22 /= n_markers_tot
    mat12_22 /= n_markers_tot
    mat13_22 /= n_markers_tot
    mat22_22 /= n_markers_tot
    mat23_22 /= n_markers_tot
    mat33_22 /= n_markers_tot

    vec1_1 /= n_markers_tot
    vec2_1 /= n_markers_tot
    vec3_1 /= n_markers_tot
    vec1_2 /= n_markers_tot
    vec2_2 /= n_markers_tot
    vec3_2 /= n_markers_tot
