'Accumulation kernels for gyro-center (5D) particles.'


from pyccel.decorators import stack_array

from numpy import zeros, empty, shape

import struphy.geometry.evaluation_kernels as evaluation_kernels 
import struphy.bsplines.bsplines_kernels as bsplines_kernels 
import struphy.bsplines.evaluation_kernels_3d as evaluation_kernels_3d 
import struphy.linear_algebra.linalg_kernels as linalg_kernels 
import struphy.pic.accumulation.particle_to_mat_kernels as particle_to_mat_kernels 


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

    5. Data objects to accumulate into (number depends on model, but at least one matrix has to be passed)
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


@stack_array('dfm', 'df_inv', 'df_inv_t', 'g_inv', 'tmp1', 'tmp2', 'b', 'b_prod', 'bstar', 'norm_b1', 'curl_norm_b','bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def cc_lin_mhd_5d_D(markers: 'float[:,:]', n_markers_tot: 'int',
                    pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                    starts: 'int[:]',
                    kind_map: 'int', params_map: 'float[:]',
                    p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                    ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                    cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                    mat12: 'float[:,:,:,:,:,:]',
                    mat13: 'float[:,:,:,:,:,:]',
                    mat23: 'float[:,:,:,:,:,:]',
                    epsilon: float,           # model specific argument
                    PB0: 'float[:,:,:]',    # model specific argument
                    b2_1: 'float[:,:,:]',   # model specific argument
                    b2_2: 'float[:,:,:]',   # model specific argument
                    b2_3: 'float[:,:,:]',   # model specific argument
                    norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',               # model specific argument
                    curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',   # model specific argument
                    basis_u: 'int', scale_mat: 'float'):  # model specific argument
    r"""Accumulation kernel for the propagator `CurrentCoupling5DDensity, <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.propagators.propagators_fields.CurrentCoupling5DDensity>`_ .

    Accumulates math:`\alpha` -form matrix with the filling functions

    .. math::

        A_p^{\mu, \nu} = w_p * (1-\frac{B_\parallel}{B^*_\parallel}) * [ G^{-1}(\eta_p) * B2_{\times}(\eta_p) * G^{-1}(\eta_p) ]_{\mu, \nu}     

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

    # containers for fields
    curl_norm_b = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    b_star = empty(3, dtype=float)

    # allocate some temporary buffers for filling
    tmp1 = empty((3, 3), dtype=float)
    tmp2 = empty((3, 3), dtype=float)

    # get local number of markers
    n_markers_loc = shape(markers)[0]

    #$ omp parallel firstprivate(b_prod) private(ip, eta1, eta2, eta3, v, weight, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, b, b_para, curl_norm_b, b_star, norm_b1, b_star_para, density_const, dfm, df_inv, df_inv_t, g_inv, det_df, tmp1, tmp2, filling_m12, filling_m13, filling_m23)
    #$ omp for reduction ( + : mat12, mat13, mat23)
    for ip in range(n_markers_loc):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        v = markers[ip, 3]

        # b-field evaluation
        span1 = bsplines_kernels.find_span(tn1, int(pn[0]), eta1)
        span2 = bsplines_kernels.find_span(tn2, int(pn[1]), eta2)
        span3 = bsplines_kernels.find_span(tn3, int(pn[2]), eta3)

        bsplines_kernels.b_d_splines_slim(tn1, int(pn[0]), eta1, span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, int(pn[1]), eta2, span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, int(pn[2]), eta3, span3, bn3, bd3)

        b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]), int(pn[1]) - 1, int(pn[2]) - 1, bn1, bd2, bd3, span1, span2, span3, b2_1, starts)
        b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]) - 1, int(pn[1]), int(pn[2]) - 1, bd1, bn2, bd3, span1, span2, span3, b2_2, starts)
        b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]) - 1, int(pn[1]) - 1, int(pn[2]), bd1, bd2, bn3, span1, span2, span3, b2_3, starts)

        # abs_b; 0form
        b_para = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, PB0, starts)

        # norm_b1; 1form
        norm_b1[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts)
        norm_b1[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts)
        norm_b1[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts)

        # curl_norm_b; 2form
        curl_norm_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts)
        curl_norm_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts)
        curl_norm_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts)

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

        # calculate Bstar and transform to H1vec
        b_star[:] = b + epsilon*v*curl_norm_b
        b_star /= det_df

        # calculate abs_b_star_para
        b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # calculate scaling constant
        density_const = 1 - b_para/b_star_para

        # marker weight
        weight = markers[ip, 5]

        if basis_u == 0:

            # filling functions
            filling_m12 = - weight * density_const * b_prod[0, 1] * scale_mat
            filling_m13 = - weight * density_const * b_prod[0, 2] * scale_mat
            filling_m23 = - weight * density_const * b_prod[1, 2] * scale_mat

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

            filling_m12 = - weight * density_const * tmp2[0, 1] * scale_mat
            filling_m13 = - weight * density_const * tmp2[0, 2] * scale_mat
            filling_m23 = - weight * density_const * tmp2[1, 2] * scale_mat

            # call the appropriate matvec filler
            particle_to_mat_kernels.mat_fill_v1_asym(pn, span1, span2, span3,
                                 bn1, bn2, bn3,
                                 bd1, bd2, bd3,
                                 starts,
                                 mat12, mat13, mat23,
                                 filling_m12, filling_m13, filling_m23)

        elif basis_u == 2:

            # filling functions
            filling_m12 = - weight * density_const * b_prod[0, 1] * scale_mat / det_df**2
            filling_m13 = - weight * density_const * b_prod[0, 2] * scale_mat / det_df**2
            filling_m23 = - weight * density_const * b_prod[1, 2] * scale_mat / det_df**2

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


@stack_array('dfm', 'df_inv_t', 'df_inv', 'g_inv', 'filling_m', 'filling_v', 'tmp', 'tmp1', 'tmp2', 'tmp_m', 'tmp_v', 'b', 'b_prod', 'b_prod_neg' 'b_star', 'norm_b1', 'curl_norm_b', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def cc_lin_mhd_5d_J1(markers: 'float[:,:]', n_markers_tot: 'int',
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
                     epsilon: float,                  # model specific argument
                     b1: 'float[:,:,:]',            # model specific argument
                     b2: 'float[:,:,:]',            # model specific argument
                     b3: 'float[:,:,:]',            # model specific argument
                     norm_b11: 'float[:,:,:]',      # model specific argument
                     norm_b12: 'float[:,:,:]',      # model specific argument
                     norm_b13: 'float[:,:,:]',      # model specific argument
                     curl_norm_b1: 'float[:,:,:]',  # model specific argument
                     curl_norm_b2: 'float[:,:,:]',  # model specific argument
                     curl_norm_b3: 'float[:,:,:]',  # model specific argument
                     basis_u: 'int',               # model specific argument
                     scale_mat: 'float',           # model specific argument
                     scale_vec: 'float'):          # model specific argument
    r"""Accumulation kernel for the propagator `CurrentCoupling5DCurlb, <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.propagators.propagators_coupling.CurrentCoupling5DCurlb>`_ .

    Accumulates :math:`\alpha` -form matrix and vector with the filling functions

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
    b_prod_neg = zeros((3, 3), dtype=float)
    norm_b1 = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)

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

    tmp = empty((3, 3), dtype=float)
    tmp1 = empty((3, 3), dtype=float)
    tmp2 = empty((3, 3), dtype=float)
    tmp_m = empty((3, 3), dtype=float)

    tmp_v = empty(3, dtype=float)

    # get number of markers
    n_markers_loc = shape(markers)[0]

    #$ omp parallel firstprivate(b_prod) private(ip, eta1, eta2, eta3, v, weight, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, b, b_star, b_prod_neg, norm_b1, curl_norm_b, abs_b_star_para, dfm, df_inv, df_inv_t, g_inv, det_df, tmp, tmp1, tmp2, tmp_m, tmp_v, filling_m, filling_v)
    #$ omp for reduction ( + : mat11, mat12, mat13, mat22, mat23, mat33, vec1, vec2, vec3)
    for ip in range(n_markers_loc):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # marker weight and velocity
        weight = markers[ip, 5]
        v = markers[ip, 3]

        # b-field evaluation
        span1 = bsplines_kernels.find_span(tn1, int(pn[0]), eta1)
        span2 = bsplines_kernels.find_span(tn2, int(pn[1]), eta2)
        span3 = bsplines_kernels.find_span(tn3, int(pn[2]), eta3)

        bsplines_kernels.b_d_splines_slim(tn1, int(pn[0]), eta1, span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, int(pn[1]), eta2, span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, int(pn[2]), eta3, span3, bn3, bd3)

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta1, eta2, eta3,
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        det_df = linalg_kernels.det(dfm)

        # b; 2form
        b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]), int(pn[1]) - 1, int(pn[2]) - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts)
        b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]) - 1, int(pn[1]), int(pn[2]) - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts)
        b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]) - 1, int(pn[1]) - 1, int(pn[2]), bd1, bd2, bn3, span1, span2, span3, b3, starts)

        # norm_b1; 1form
        norm_b1[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]) - 1, int(pn[1]), int(pn[2]), bd1, bn2, bn3, span1, span2, span3, norm_b11, starts)
        norm_b1[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]), int(pn[1]) - 1, int(pn[2]), bn1, bd2, bn3, span1, span2, span3, norm_b12, starts)
        norm_b1[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]), int(pn[1]), int(pn[2]) - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts)

        # curl_norm_b; 2form
        curl_norm_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]), int(pn[1]) - 1, int(pn[2]) - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts)
        curl_norm_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]) - 1, int(pn[1]), int(pn[2]) - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts)
        curl_norm_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]) - 1, int(pn[1]) - 1, int(pn[2]), bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts)

        # b_star; 2form in H1vec
        b_star[:] = (b + curl_norm_b*v*epsilon)/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # calculate tensor product of two curl_norm_b
        linalg_kernels.outer(curl_norm_b, curl_norm_b, tmp)

        # operator bx() as matrix
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] = +b[1]
        b_prod[1, 0] = +b[2]
        b_prod[1, 2] = -b[0]
        b_prod[2, 0] = -b[1]
        b_prod[2, 1] = +b[0]

        b_prod_neg[:] = -1.0*b_prod

        if basis_u == 0:

            linalg_kernels.matrix_matrix(b_prod, tmp, tmp1)
            linalg_kernels.matrix_matrix(tmp1, b_prod_neg, tmp_m)
            linalg_kernels.matrix_vector(b_prod, curl_norm_b, tmp_v)

            filling_m[:, :] = weight * tmp_m * v**2 / \
                abs_b_star_para**2 / det_df**2 * scale_mat
            filling_v[:] = weight * tmp_v * v**2 / \
                abs_b_star_para / det_df * scale_vec

            # call the appropriate matvec filler
            particle_to_mat_kernels.m_v_fill_v0vec_symm(pn, span1, span2, span3,
                                    bn1, bn2, bn3,
                                    starts,
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
            linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
            linalg_kernels.transpose(df_inv, df_inv_t)
            linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)
            linalg_kernels.matrix_matrix(g_inv, b_prod, tmp1)
            linalg_kernels.matrix_vector(tmp1, curl_norm_b, tmp_v)

            linalg_kernels.matrix_matrix(tmp1, tmp, tmp2)
            linalg_kernels.matrix_matrix(tmp2, b_prod_neg, tmp1)
            linalg_kernels.matrix_matrix(tmp1, g_inv, tmp_m)

            filling_m[:, :] = weight * tmp_m * v**2 / \
                abs_b_star_para**2 / det_df**2 * scale_mat
            filling_v[:] = weight * tmp_v * v**2 / \
                abs_b_star_para / det_df * scale_vec

            # call the appropriate matvec filler
            particle_to_mat_kernels.m_v_fill_v1_symm(pn, span1, span2, span3,
                                 bn1, bn2, bn3,
                                 bd1, bd2, bd3,
                                 starts,
                                 mat11, mat12, mat13,
                                 mat22, mat23,
                                 mat33,
                                 filling_m[0, 0], filling_m[0, 1], filling_m[0, 2],
                                 filling_m[1, 1], filling_m[1, 2],
                                 filling_m[2, 2],
                                 vec1, vec2, vec3,
                                 filling_v[0], filling_v[1], filling_v[2])

        elif basis_u == 2:

            linalg_kernels.matrix_matrix(b_prod, tmp, tmp1)
            linalg_kernels.matrix_matrix(tmp1, b_prod_neg, tmp_m)
            linalg_kernels.matrix_vector(b_prod, curl_norm_b, tmp_v)

            filling_m[:, :] = weight * tmp_m * v**2 / \
                abs_b_star_para**2 / det_df**4 * scale_mat
            filling_v[:] = weight * tmp_v * v**2 / \
                abs_b_star_para / det_df**2 * scale_vec

            # call the appropriate matvec filler
            particle_to_mat_kernels.m_v_fill_v2_symm(pn, span1, span2, span3,
                                 bn1, bn2, bn3,
                                 bd1, bd2, bd3,
                                 starts,
                                 mat11, mat12, mat13,
                                 mat22, mat23,
                                 mat33,
                                 filling_m[0, 0], filling_m[0, 1], filling_m[0, 2],
                                 filling_m[1, 1], filling_m[1, 2],
                                 filling_m[2, 2],
                                 vec1, vec2, vec3,
                                 filling_v[0], filling_v[1], filling_v[2])

    mat11 /= n_markers_tot
    mat12 /= n_markers_tot
    mat13 /= n_markers_tot
    mat22 /= n_markers_tot
    mat23 /= n_markers_tot
    mat33 /= n_markers_tot

    vec1 /= n_markers_tot
    vec2 /= n_markers_tot
    vec3 /= n_markers_tot

    #$ omp end parallel


@stack_array('bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def cc_lin_mhd_5d_mu(markers: 'float[:,:]', n_markers_tot: 'int',
                     pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                     starts: 'int[:]',
                     kind_map: 'int', params_map: 'float[:]',
                     p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                     ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                     cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                     mat: 'float[:,:,:,:,:,:]',
                     vec: 'float[:,:,:]',
                     coupling_const: 'float'):
    r"""Accumulation kernel for the propagator `ShearAlfvénCurrentCoupling5D, <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.propagators.propagators_fields.ShearAlfvénCurrentCoupling5D>`_ .

    Accumulates 0-form scalar with the filling functions:

    .. math::

        C_p = \omega_p * \mu_p \,.

    """
    bn1 = empty(int(pn[0]) + 1, dtype=float)
    bn2 = empty(int(pn[1]) + 1, dtype=float)
    bn3 = empty(int(pn[2]) + 1, dtype=float)

    bd1 = empty(int(pn[0]), dtype=float)
    bd2 = empty(int(pn[1]), dtype=float)
    bd3 = empty(int(pn[2]), dtype=float)

    # get number of markers
    n_markers_loc = shape(markers)[0]

    #$ omp parallel private(ip, eta1, eta2, eta3, mu, weight, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, filling)
    #$ omp for reduction ( + : vec)

    for ip in range(n_markers_loc):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # marker weight and velocity
        weight = markers[ip, 5]
        mu = markers[ip, 4]

        # b-field evaluation
        span1 = bsplines_kernels.find_span(tn1, int(pn[0]), eta1)
        span2 = bsplines_kernels.find_span(tn2, int(pn[1]), eta2)
        span3 = bsplines_kernels.find_span(tn3, int(pn[2]), eta3)

        bsplines_kernels.b_d_splines_slim(tn1, int(pn[0]), eta1, span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, int(pn[1]), eta2, span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, int(pn[2]), eta3, span3, bn3, bd3)

        filling = weight * mu * coupling_const

        # call the appropriate matvec filler
        particle_to_mat_kernels.scalar_fill_v0(pn, span1, span2, span3, bn1,
                           bn2, bn3, starts, vec, filling)

    vec /= n_markers_tot

    #$ omp end parallel


@stack_array('dfm', 'df_inv', 'df_inv_t', 'g_inv', 'filling_v', 'tmp_v1', 'tmp_v2', 'b', 'curl_norm_b', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def cc_lin_mhd_5d_curlMxB(markers: 'float[:,:]', n_markers_tot: 'int',
                          pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                          starts: 'int[:]',
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
                          b1: 'float[:,:,:]',           # model specific argument
                          b2: 'float[:,:,:]',           # model specific argument
                          b3: 'float[:,:,:]',           # model specific argument
                          curl_norm_b1: 'float[:,:,:]', # model specific argument
                          curl_norm_b2: 'float[:,:,:]', # model specific argument
                          curl_norm_b3: 'float[:,:,:]', # model specific argument
                          basis_u: 'int', scale_vec: 'float'):  # model specific argument
    r"""Accumulation kernel for the propagator `MagnetosonicCurrentCoupling5D, <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.propagators.propagators_fields.MagnetosonicCurrentCoupling5D>`_ .

    Accumulates :math:`\alpha` -form vector with the filling functions:

    .. math::

        B^\mu_p = \omega_p * \mu_p * [(\hat \nabla \times \hat{\mathbf b}^1_0)(\boldsymbol \eta_k) \times \hat{\mathbf B}^2 (\boldsymbol \eta_k)]_\mu \,.

    Parameters
    ----------
        b2_1, b2_2, b2_3 : array[float]
            FE coefficients c_ijk of the magnetic field as a 2-form.

        curl_norm_b1, curl_norm_b2, curl_norm_b3 : array[float]
            FE coefficients c_ijk of the curl of the unit magnetic field as a 1-form.

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    # allocate for magnetic field evaluation
    b = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)

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
    filling_v = empty(3, dtype=float)

    tmp_v1 = empty(3, dtype=float)
    tmp_v2 = empty(3, dtype=float)

    # get number of markers
    n_markers_loc = shape(markers)[0]

    #$ omp parallel private(ip, eta1, eta2, eta3, mu, weight, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, b, curl_norm_b, dfm, df_inv, df_inv_t, g_inv, det_df, tmp_v1, tmp_v2, filling_v)
    #$ omp for reduction ( + : vec1, vec2, vec3)
    for ip in range(n_markers_loc):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # marker weight and velocity
        weight = markers[ip, 5]
        mu = markers[ip, 4]

        # b-field evaluation
        span1 = bsplines_kernels.find_span(tn1, int(pn[0]), eta1)
        span2 = bsplines_kernels.find_span(tn2, int(pn[1]), eta2)
        span3 = bsplines_kernels.find_span(tn3, int(pn[2]), eta3)

        bsplines_kernels.b_d_splines_slim(tn1, int(pn[0]), eta1, span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, int(pn[1]), eta2, span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, int(pn[2]), eta3, span3, bn3, bd3)

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta1, eta2, eta3,
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        det_df = linalg_kernels.det(dfm)

        # needed metric coefficients
        linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
        linalg_kernels.transpose(df_inv, df_inv_t)
        linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)

        # b; 2form
        b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]), int(pn[1]) - 1, int(pn[2]) - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts)
        b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]) - 1, int(pn[1]), int(pn[2]) - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts)
        b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]) - 1, int(pn[1]) - 1, int(pn[2]), bd1, bd2, bn3, span1, span2, span3, b3, starts)

        # curl_norm_b; 2form
        curl_norm_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]), int(pn[1]) - 1, int(pn[2]) - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts)
        curl_norm_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]) - 1, int(pn[1]), int(pn[2]) - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts)
        curl_norm_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]) - 1, int(pn[1]) - 1, int(pn[2]), bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts)

        linalg_kernels.cross(curl_norm_b, b, tmp_v1)
        tmp_v1 /= det_df

        if basis_u == 0:

            filling_v[:] = weight * mu * tmp_v1 * scale_vec

            particle_to_mat_kernels.vec_fill_v0(pn, span1, span2, span3,
                            bn1, bn2, bn3,
                            starts,
                            vec1, vec2, vec3,
                            filling_v[0], filling_v[1], filling_v[2])

        elif basis_u == 1:

            linalg_kernels.matrix_vector(g_inv, tmp_v1, tmp_v2)

            filling_v[:] = weight * mu * tmp_v2 * scale_vec

            particle_to_mat_kernels.vec_fill_v1(pn, span1, span2, span3,
                            bn1, bn2, bn3, bd1, bd2, bd3,
                            starts,
                            vec1, vec2, vec3,
                            filling_v[0], filling_v[1], filling_v[2])

        elif basis_u == 2:

            filling_v[:] = weight * mu * tmp_v1 / det_df * scale_vec

            particle_to_mat_kernels.vec_fill_v2(pn, span1, span2, span3,
                            bn1, bn2, bn3, bd1, bd2, bd3,
                            starts,
                            vec1, vec2, vec3,
                            filling_v[0], filling_v[1], filling_v[2])

    vec1 /= n_markers_tot
    vec2 /= n_markers_tot
    vec3 /= n_markers_tot

    #$ omp end parallel


@stack_array('dfm', 'df_inv_t', 'df_inv', 'g_inv', 'filling_m', 'filling_v', 'tmp1', 'tmp2', 'tmp_t', 'tmp_m', 'tmp_v', 'b', 'b_prod', 'norm_b2_prod', 'b_star', 'curl_norm_b', 'norm_b1', 'norm_b2', 'grad_PB', 'grad_PB_mat', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def cc_lin_mhd_5d_J2(markers: 'float[:,:]', n_markers_tot: 'int',
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
                     epsilon: float,    # model specific argument
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
                     basis_u: 'int', scale_mat: 'float', scale_vec: 'float'):  # model specific argument
    r"""Accumulation kernel for the propagator `CurrentCoupling5DGradBxB, <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.propagators.propagators_coupling.CurrentCoupling5DGradBxB>`_ .

    Accumulates math:`\alpha` -form matrix and vector with the filling functions

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
    grad_PB_mat = zeros((3, 3), dtype=float)

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

    tmp1 = empty((3, 3), dtype=float)
    tmp2 = empty((3, 3), dtype=float)
    tmp_t = empty((3, 3), dtype=float)
    tmp_m = empty((3, 3), dtype=float)

    tmp_v = empty(3, dtype=float)

    # get number of markers
    n_markers_loc = shape(markers)[0]

    #$ omp parallel firstprivate(b_prod) private(ip, eta1, eta2, eta3, v, mu, weight, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, b, b_star, norm_b1, norm_b2, norm_b2_prod, curl_norm_b, grad_PB, grad_PB_mat, abs_b_star_para, dfm, df_inv, df_inv_t, g_inv, det_df, tmp_t, tmp1, tmp2, tmp_m, tmp_v, filling_m, filling_v)
    #$ omp for reduction ( + : mat11, mat12, mat13, mat22, mat23, mat33, vec1, vec2, vec3)
    for ip in range(n_markers_loc):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # marker weight and velocity
        weight = markers[ip, 5]
        v = markers[ip, 3]
        mu = markers[ip, 4]

        # b-field evaluation
        span1 = bsplines_kernels.find_span(tn1, int(pn[0]), eta1)
        span2 = bsplines_kernels.find_span(tn2, int(pn[1]), eta2)
        span3 = bsplines_kernels.find_span(tn3, int(pn[2]), eta3)

        bsplines_kernels.b_d_splines_slim(tn1, int(pn[0]), eta1, span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, int(pn[1]), eta2, span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, int(pn[2]), eta3, span3, bn3, bd3)

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta1, eta2, eta3,
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        det_df = linalg_kernels.det(dfm)

        # needed metric coefficients
        linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
        linalg_kernels.transpose(df_inv, df_inv_t)
        linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)

        # b; 2form
        b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]), int(pn[1]) - 1, int(pn[2]) - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts)
        b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]) - 1, int(pn[1]), int(pn[2]) - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts)
        b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]) - 1, int(pn[1]) - 1, int(pn[2]), bd1, bd2, bn3, span1, span2, span3, b3, starts)

        # norm_b1; 1form
        norm_b1[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]) - 1, int(pn[1]), int(pn[2]), bd1, bn2, bn3, span1, span2, span3, norm_b11, starts)
        norm_b1[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]), int(pn[1]) - 1, int(pn[2]), bn1, bd2, bn3, span1, span2, span3, norm_b12, starts)
        norm_b1[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]), int(pn[1]), int(pn[2]) - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts)

        # norm_b2; 2form
        norm_b2[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]), int(pn[1]) - 1, int(pn[2]) - 1, bn1, bd2, bd3, span1, span2, span3, norm_b21, starts)
        norm_b2[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]) - 1, int(pn[1]), int(pn[2]) - 1, bd1, bn2, bd3, span1, span2, span3, norm_b22, starts)
        norm_b2[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]) - 1, int(pn[1]) - 1, int(pn[2]), bd1, bd2, bn3, span1, span2, span3, norm_b23, starts)

        # curl_norm_b; 2form
        curl_norm_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]), int(pn[1]) - 1, int(pn[2]) - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts)
        curl_norm_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]) - 1, int(pn[1]), int(pn[2]) - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts)
        curl_norm_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]) - 1, int(pn[1]) - 1, int(pn[2]), bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts)

        # grad_PB; 1form
        grad_PB[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]) - 1, int(pn[1]), int(pn[2]), bd1, bn2, bn3, span1, span2, span3, grad_PB1, starts)
        grad_PB[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]), int(pn[1]) - 1, int(pn[2]), bn1, bd2, bn3, span1, span2, span3, grad_PB2, starts)
        grad_PB[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            int(pn[0]), int(pn[1]), int(pn[2]) - 1, bn1, bn2, bd3, span1, span2, span3, grad_PB3, starts)

        grad_PB_mat[0, 0] = grad_PB[0]
        grad_PB_mat[1, 1] = grad_PB[1]
        grad_PB_mat[2, 2] = grad_PB[2]

        # b_star; 2form transformed into H1vec
        b_star[:] = (b + curl_norm_b*v*epsilon)/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

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

            linalg_kernels.matrix_matrix(b_prod, g_inv, tmp1)
            linalg_kernels.matrix_matrix(tmp1, norm_b2_prod, tmp2)
            linalg_kernels.matrix_matrix(tmp2, g_inv, tmp1)

            linalg_kernels.transpose(tmp1, tmp_t)

            linalg_kernels.matrix_vector(tmp1, grad_PB, tmp_v)

            linalg_kernels.matrix_matrix(tmp1, grad_PB_mat, tmp2)
            linalg_kernels.matrix_matrix(tmp2, tmp_t, tmp_m)

            filling_m[:, :] = weight * tmp_m * \
                mu / abs_b_star_para**2 * scale_mat
            filling_v[:] = weight * tmp_v * mu / abs_b_star_para * scale_vec

            # call the appropriate matvec filler
            particle_to_mat_kernels.m_v_fill_v0vec_symm(pn, span1, span2, span3,
                                    bn1, bn2, bn3,
                                    starts,
                                    mat11, mat12, mat13,
                                    mat22, mat23,
                                    mat33,
                                    filling_m[0, 0], filling_m[0, 1], filling_m[0, 2],
                                    filling_m[1, 1], filling_m[1, 2],
                                    filling_m[2, 2],
                                    vec1, vec2, vec3,
                                    filling_v[0], filling_v[1], filling_v[2])

        elif basis_u == 1:

            linalg_kernels.matrix_matrix(g_inv, b_prod, tmp1)
            linalg_kernels.matrix_matrix(tmp1, g_inv, tmp2)
            linalg_kernels.matrix_matrix(tmp2, norm_b2_prod, tmp1)
            linalg_kernels.matrix_matrix(tmp1, g_inv, tmp2)

            linalg_kernels.transpose(tmp2, tmp_t)

            linalg_kernels.matrix_vector(tmp2, grad_PB, tmp_v)

            linalg_kernels.matrix_matrix(tmp2, grad_PB_mat, tmp1)
            linalg_kernels.matrix_matrix(tmp1, tmp_t, tmp_m)

            filling_m[:, :] = weight * tmp_m * \
                mu / abs_b_star_para**2 * scale_mat
            filling_v[:] = weight * tmp_v * mu / abs_b_star_para * scale_vec

            # call the appropriate matvec filler
            particle_to_mat_kernels.m_v_fill_v1_symm(pn, span1, span2, span3,
                                 bn1, bn2, bn3,
                                 bd1, bd2, bd3,
                                 starts,
                                 mat11, mat12, mat13,
                                 mat22, mat23,
                                 mat33,
                                 filling_m[0, 0], filling_m[0, 1], filling_m[0, 2],
                                 filling_m[1, 1], filling_m[1, 2],
                                 filling_m[2, 2],
                                 vec1, vec2, vec3,
                                 filling_v[0], filling_v[1], filling_v[2])

        elif basis_u == 2:

            linalg_kernels.matrix_matrix(b_prod, g_inv, tmp1)
            linalg_kernels.matrix_matrix(tmp1, norm_b2_prod, tmp2)
            linalg_kernels.matrix_matrix(tmp2, g_inv, tmp1)

            linalg_kernels.transpose(tmp1, tmp_t)

            linalg_kernels.matrix_vector(tmp1, grad_PB, tmp_v)

            linalg_kernels.matrix_matrix(tmp1, grad_PB_mat, tmp2)
            linalg_kernels.matrix_matrix(tmp2, tmp_t, tmp_m)

            filling_m[:, :] = weight * tmp_m * mu / \
                abs_b_star_para**2 / det_df**2 * scale_mat
            filling_v[:] = weight * tmp_v * mu / \
                abs_b_star_para / det_df * scale_vec

            # call the appropriate matvec filler
            particle_to_mat_kernels.m_v_fill_v2_symm(pn, span1, span2, span3,
                                 bn1, bn2, bn3,
                                 bd1, bd2, bd3,
                                 starts,
                                 mat11, mat12, mat13,
                                 mat22, mat23,
                                 mat33,
                                 filling_m[0, 0], filling_m[0, 1], filling_m[0, 2],
                                 filling_m[1, 1], filling_m[1, 2],
                                 filling_m[2, 2],
                                 vec1, vec2, vec3,
                                 filling_v[0], filling_v[1], filling_v[2])

    mat11 /= n_markers_tot
    mat12 /= n_markers_tot
    mat13 /= n_markers_tot
    mat22 /= n_markers_tot
    mat23 /= n_markers_tot
    mat33 /= n_markers_tot

    vec1 /= n_markers_tot
    vec2 /= n_markers_tot
    vec3 /= n_markers_tot

    #$ omp end parallel


# def cc_lin_mhd_5d_J2_dg(markers: 'float[:,:]', n_markers_tot: 'int',
#                         pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
#                         starts: 'int[:]',
#                         kind_map: 'int', params_map: 'float[:]',
#                         p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
#                         ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
#                         cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
#                         mat11: 'float[:,:,:,:,:,:]',
#                         mat12: 'float[:,:,:,:,:,:]',
#                         mat13: 'float[:,:,:,:,:,:]',
#                         mat22: 'float[:,:,:,:,:,:]',
#                         mat23: 'float[:,:,:,:,:,:]',
#                         mat33: 'float[:,:,:,:,:,:]',
#                         vec1: 'float[:,:,:]',
#                         vec2: 'float[:,:,:]',
#                         vec3: 'float[:,:,:]',
#                         epsilon: float,               # model specific argument
#                         b1: 'float[:,:,:]',           # model specific argument
#                         b2: 'float[:,:,:]',           # model specific argument
#                         b3: 'float[:,:,:]',           # model specific argument
#                         norm_b11: 'float[:,:,:]',     # model specific argument
#                         norm_b12: 'float[:,:,:]',     # model specific argument
#                         norm_b13: 'float[:,:,:]',     # model specific argument
#                         norm_b21: 'float[:,:,:]',     # model specific argument
#                         norm_b22: 'float[:,:,:]',     # model specific argument
#                         norm_b23: 'float[:,:,:]',     # model specific argument
#                         # model specific argument
#                         curl_norm_b1: 'float[:,:,:]',
#                         # model specific argument
#                         curl_norm_b2: 'float[:,:,:]',
#                         # model specific argument
#                         curl_norm_b3: 'float[:,:,:]',
#                         grad_PB1: 'float[:,:,:]',     # model specific argument
#                         grad_PB2: 'float[:,:,:]',     # model specific argument
#                         grad_PB3: 'float[:,:,:]',     # model specific argument
#                         gradI_const: 'float',         # model specific argument
#                         basis_u: 'int', scale_vec: 'float'):  # model specific argument
#     r"""Draft
#     """
#     # allocate for particle position
#     e_diff = empty(3, dtype=float)

#     # allocate for magnetic field evaluation
#     b = empty(3, dtype=float)
#     b_star = empty(3, dtype=float)
#     b_prod = zeros((3, 3), dtype=float)
#     norm_b2_prod = zeros((3, 3), dtype=float)
#     curl_norm_b = empty(3, dtype=float)
#     norm_b1 = empty(3, dtype=float)
#     norm_b2 = empty(3, dtype=float)
#     grad_PB = empty(3, dtype=float)

#     bn1 = empty(int(pn[0]) + 1, dtype=float)
#     bn2 = empty(int(pn[1]) + 1, dtype=float)
#     bn3 = empty(int(pn[2]) + 1, dtype=float)

#     bd1 = empty(int(pn[0]), dtype=float)
#     bd2 = empty(int(pn[1]), dtype=float)
#     bd3 = empty(int(pn[2]), dtype=float)

#     # allocate for metric coeffs
#     dfm = empty((3, 3), dtype=float)
#     df_inv = empty((3, 3), dtype=float)
#     df_inv_t = empty((3, 3), dtype=float)
#     g_inv = empty((3, 3), dtype=float)

#     # allocate for filling
#     filling_v = empty(3, dtype=float)

#     tmp1 = empty((3, 3), dtype=float)
#     tmp2 = empty((3, 3), dtype=float)
#     tmp_v1 = empty(3, dtype=float)
#     tmp_v2 = empty(3, dtype=float)

#     # get number of markers
#     n_markers_loc = shape(markers)[0]

#     for ip in range(n_markers_loc):

#         # only do something if particle is a "true" particle (i.e. not a hole)
#         if markers[ip, 0] == -1.:
#             continue

#         # marker positions
#         eta1 = markers[ip, 0]
#         eta2 = markers[ip, 1]
#         eta3 = markers[ip, 2]

#         # marker weight and velocity
#         weight = markers[ip, 5]
#         v = markers[ip, 3]
#         mu = markers[ip, 4]

#         # b-field evaluation
#         span1 = bsplines_kernels.find_span(tn1, int(pn[0]), eta1)
#         span2 = bsplines_kernels.find_span(tn2, int(pn[1]), eta2)
#         span3 = bsplines_kernels.find_span(tn3, int(pn[2]), eta3)

#         bsplines_kernels.b_d_splines_slim(tn1, int(pn[0]), eta1, span1, bn1, bd1)
#         bsplines_kernels.b_d_splines_slim(tn2, int(pn[1]), eta2, span2, bn2, bd2)
#         bsplines_kernels.b_d_splines_slim(tn3, int(pn[2]), eta3, span3, bn3, bd3)

#         # evaluate Jacobian, result in dfm
#         map_eval.df(eta1, eta2, eta3,
#                     kind_map, params_map,
#                     t1_map, t2_map, t3_map, p_map,
#                     ind1_map, ind2_map, ind3_map,
#                     cx, cy, cz,
#                     dfm)

#         det_df = linalg_kernels.det(dfm)

#         # needed metric coefficients
#         linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
#         linalg_kernels.transpose(df_inv, df_inv_t)
#         linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)

#         # b; 2form
#         b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]), int(pn[1]) - 1, int(pn[2]) - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts)
#         b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]) - 1, int(pn[1]), int(pn[2]) - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts)
#         b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]) - 1, int(pn[1]) - 1, int(pn[2]), bd1, bd2, bn3, span1, span2, span3, b3, starts)

#         # norm_b1; 1form
#         norm_b1[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]) - 1, int(pn[1]), int(pn[2]), bd1, bn2, bn3, span1, span2, span3, norm_b11, starts)
#         norm_b1[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]), int(pn[1]) - 1, int(pn[2]), bn1, bd2, bn3, span1, span2, span3, norm_b12, starts)
#         norm_b1[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]), int(pn[1]), int(pn[2]) - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts)

#         # norm_b2; 2form
#         norm_b2[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]), int(pn[1]) - 1, int(pn[2]) - 1, bn1, bd2, bd3, span1, span2, span3, norm_b21, starts)
#         norm_b2[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]) - 1, int(pn[1]), int(pn[2]) - 1, bd1, bn2, bd3, span1, span2, span3, norm_b22, starts)
#         norm_b2[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]) - 1, int(pn[1]) - 1, int(pn[2]), bd1, bd2, bn3, span1, span2, span3, norm_b23, starts)

#         # curl_norm_b; 2form
#         curl_norm_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]), int(pn[1]) - 1, int(pn[2]) - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts)
#         curl_norm_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]) - 1, int(pn[1]), int(pn[2]) - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts)
#         curl_norm_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]) - 1, int(pn[1]) - 1, int(pn[2]), bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts)

#         # grad_PB; 1form
#         grad_PB[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]) - 1, int(pn[1]), int(pn[2]), bd1, bn2, bn3, span1, span2, span3, grad_PB1, starts)
#         grad_PB[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]), int(pn[1]) - 1, int(pn[2]), bn1, bd2, bn3, span1, span2, span3, grad_PB2, starts)
#         grad_PB[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]), int(pn[1]), int(pn[2]) - 1, bn1, bn2, bd3, span1, span2, span3, grad_PB3, starts)

#         # b_star; 2form transformed into H1vec
#         b_star[:] = (b + curl_norm_b*v*epsilon)/det_df

#         # calculate abs_b_star_para
#         abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

#         # operator bx() as matrix
#         b_prod[0, 1] = -b[2]
#         b_prod[0, 2] = +b[1]
#         b_prod[1, 0] = +b[2]
#         b_prod[1, 2] = -b[0]
#         b_prod[2, 0] = -b[1]
#         b_prod[2, 1] = +b[0]

#         norm_b2_prod[0, 1] = -norm_b2[2]
#         norm_b2_prod[0, 2] = +norm_b2[1]
#         norm_b2_prod[1, 0] = +norm_b2[2]
#         norm_b2_prod[1, 2] = -norm_b2[0]
#         norm_b2_prod[2, 0] = -norm_b2[1]
#         norm_b2_prod[2, 1] = +norm_b2[0]

#         if basis_u == 0:

#             linalg_kernels.matrix_matrix(b_prod, g_inv, tmp1)
#             linalg_kernels.matrix_matrix(tmp1, norm_b2_prod, tmp2)
#             linalg_kernels.matrix_matrix(tmp2, g_inv, tmp1)
#             linalg_kernels.matrix_vector(tmp1, grad_PB, tmp_v1)
#             linalg_kernels.matrix_vector(tmp1, markers[ip, 15:18], tmp_v2)

#             filling_v[:] = (weight*tmp_v1*mu + tmp_v2 *
#                             gradI_const)/abs_b_star_para * scale_vec

#             # call the appropriate matvec filler
#             mvf.vec_fill_v0(pn, span1, span2, span3,
#                             bn1, bn2, bn3,
#                             starts,
#                             vec1, vec2, vec3,
#                             filling_v[0], filling_v[1], filling_v[2])

#         elif basis_u == 1:

#             linalg_kernels.matrix_matrix(g_inv, b_prod, tmp1)
#             linalg_kernels.matrix_matrix(tmp1, g_inv, tmp2)
#             linalg_kernels.matrix_matrix(tmp2, norm_b2_prod, tmp1)
#             linalg_kernels.matrix_matrix(tmp1, g_inv, tmp2)
#             linalg_kernels.matrix_vector(tmp2, grad_PB, tmp_v1)
#             linalg_kernels.matrix_vector(tmp2, markers[ip, 15:18], tmp_v2)

#             filling_v[:] = (weight*tmp_v1*mu + tmp_v2 *
#                             gradI_const)/abs_b_star_para * scale_vec

#             # call the appropriate matvec filler
#             mvf.vec_fill_v1(pn, span1, span2, span3,
#                             bn1, bn2, bn3,
#                             bd1, bd2, bd3,
#                             starts,
#                             vec1, vec2, vec3,
#                             filling_v[0], filling_v[1], filling_v[2])

#         elif basis_u == 2:

#             linalg_kernels.matrix_matrix(b_prod, g_inv, tmp1)
#             linalg_kernels.matrix_matrix(tmp1, norm_b2_prod, tmp2)
#             linalg_kernels.matrix_matrix(tmp2, g_inv, tmp1)
#             linalg_kernels.matrix_vector(tmp1, grad_PB, tmp_v1)
#             linalg_kernels.matrix_vector(tmp1, markers[ip, 15:18], tmp_v2)

#             filling_v[:] = (weight*tmp_v1*mu + tmp_v2*gradI_const) / \
#                 abs_b_star_para/det_df * scale_vec

#             # call the appropriate matvec filler
#             mvf.vec_fill_v2(pn, span1, span2, span3,
#                             bn1, bn2, bn3,
#                             bd1, bd2, bd3,
#                             starts,
#                             vec1, vec2, vec3,
#                             filling_v[0], filling_v[1], filling_v[2])

#     vec1 /= n_markers_tot
#     vec2 /= n_markers_tot
#     vec3 /= n_markers_tot


# def cc_lin_mhd_5d_J2_dg_prepare(markers: 'float[:,:]', n_markers_tot: 'int',
#                                 pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
#                                 starts: 'int[:]',
#                                 kind_map: 'int', params_map: 'float[:]',
#                                 p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
#                                 ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
#                                 cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
#                                 mat11: 'float[:,:,:,:,:,:]',
#                                 mat12: 'float[:,:,:,:,:,:]',
#                                 mat13: 'float[:,:,:,:,:,:]',
#                                 mat22: 'float[:,:,:,:,:,:]',
#                                 mat23: 'float[:,:,:,:,:,:]',
#                                 mat33: 'float[:,:,:,:,:,:]',
#                                 vec1: 'float[:,:,:]',
#                                 vec2: 'float[:,:,:]',
#                                 vec3: 'float[:,:,:]',
#                                 epsilon: float,    # model specific argument
#                                 b1: 'float[:,:,:]',   # model specific argument
#                                 b2: 'float[:,:,:]',   # model specific argument
#                                 b3: 'float[:,:,:]',   # model specific argument
#                                 # model specific argument
#                                 norm_b11: 'float[:,:,:]',
#                                 # model specific argument
#                                 norm_b12: 'float[:,:,:]',
#                                 # model specific argument
#                                 norm_b13: 'float[:,:,:]',
#                                 # model specific argument
#                                 norm_b21: 'float[:,:,:]',
#                                 # model specific argument
#                                 norm_b22: 'float[:,:,:]',
#                                 # model specific argument
#                                 norm_b23: 'float[:,:,:]',
#                                 # model specific argument
#                                 curl_norm_b1: 'float[:,:,:]',
#                                 # model specific argument
#                                 curl_norm_b2: 'float[:,:,:]',
#                                 # model specific argument
#                                 curl_norm_b3: 'float[:,:,:]',
#                                 # model specific argument
#                                 grad_PB1: 'float[:,:,:]',
#                                 # model specific argument
#                                 grad_PB2: 'float[:,:,:]',
#                                 # model specific argument
#                                 grad_PB3: 'float[:,:,:]',
#                                 basis_u: 'int', scale_vec: 'float'):  # model specific argument
#     r"""Draft
#     """

#     # allocate for magnetic field evaluation
#     b = empty(3, dtype=float)
#     b_star = empty(3, dtype=float)
#     b_prod = zeros((3, 3), dtype=float)
#     norm_b2_prod = zeros((3, 3), dtype=float)
#     curl_norm_b = empty(3, dtype=float)
#     norm_b1 = empty(3, dtype=float)
#     norm_b2 = empty(3, dtype=float)
#     grad_PB = empty(3, dtype=float)

#     bn1 = empty(int(pn[0]) + 1, dtype=float)
#     bn2 = empty(int(pn[1]) + 1, dtype=float)
#     bn3 = empty(int(pn[2]) + 1, dtype=float)

#     bd1 = empty(int(pn[0]), dtype=float)
#     bd2 = empty(int(pn[1]), dtype=float)
#     bd3 = empty(int(pn[2]), dtype=float)

#     # allocate for metric coeffs
#     dfm = empty((3, 3), dtype=float)
#     df_inv = empty((3, 3), dtype=float)
#     df_inv_t = empty((3, 3), dtype=float)
#     g_inv = empty((3, 3), dtype=float)

#     # allocate for filling
#     filling_v = empty(3, dtype=float)

#     tmp1 = empty((3, 3), dtype=float)
#     tmp2 = empty((3, 3), dtype=float)
#     tmp_v = empty(3, dtype=float)

#     # get number of markers
#     n_markers_loc = shape(markers)[0]

#     for ip in range(n_markers_loc):

#         # only do something if particle is a "true" particle (i.e. not a hole)
#         if markers[ip, 0] == -1.:
#             continue

#         # marker positions
#         eta1 = markers[ip, 0]
#         eta2 = markers[ip, 1]
#         eta3 = markers[ip, 2]

#         # marker weight and velocity
#         weight = markers[ip, 5]
#         v = markers[ip, 3]
#         mu = markers[ip, 4]

#         # b-field evaluation
#         span1 = bsplines_kernels.find_span(tn1, int(pn[0]), eta1)
#         span2 = bsplines_kernels.find_span(tn2, int(pn[1]), eta2)
#         span3 = bsplines_kernels.find_span(tn3, int(pn[2]), eta3)

#         bsplines_kernels.b_d_splines_slim(tn1, int(pn[0]), eta1, span1, bn1, bd1)
#         bsplines_kernels.b_d_splines_slim(tn2, int(pn[1]), eta2, span2, bn2, bd2)
#         bsplines_kernels.b_d_splines_slim(tn3, int(pn[2]), eta3, span3, bn3, bd3)

#         # evaluate Jacobian, result in dfm
#         map_eval.df(eta1, eta2, eta3,
#                     kind_map, params_map,
#                     t1_map, t2_map, t3_map, p_map,
#                     ind1_map, ind2_map, ind3_map,
#                     cx, cy, cz,
#                     dfm)

#         det_df = linalg_kernels.det(dfm)

#         # needed metric coefficients
#         linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
#         linalg_kernels.transpose(df_inv, df_inv_t)
#         linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)

#         # b; 2form
#         b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]), int(pn[1]) - 1, int(pn[2]) - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts)
#         b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]) - 1, int(pn[1]), int(pn[2]) - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts)
#         b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]) - 1, int(pn[1]) - 1, int(pn[2]), bd1, bd2, bn3, span1, span2, span3, b3, starts)

#         # norm_b1; 1form
#         norm_b1[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]) - 1, int(pn[1]), int(pn[2]), bd1, bn2, bn3, span1, span2, span3, norm_b11, starts)
#         norm_b1[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]), int(pn[1]) - 1, int(pn[2]), bn1, bd2, bn3, span1, span2, span3, norm_b12, starts)
#         norm_b1[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]), int(pn[1]), int(pn[2]) - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts)

#         # norm_b2; 2form
#         norm_b2[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]), int(pn[1]) - 1, int(pn[2]) - 1, bn1, bd2, bd3, span1, span2, span3, norm_b21, starts)
#         norm_b2[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]) - 1, int(pn[1]), int(pn[2]) - 1, bd1, bn2, bd3, span1, span2, span3, norm_b22, starts)
#         norm_b2[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]) - 1, int(pn[1]) - 1, int(pn[2]), bd1, bd2, bn3, span1, span2, span3, norm_b23, starts)

#         # curl_norm_b; 2form
#         curl_norm_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]), int(pn[1]) - 1, int(pn[2]) - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts)
#         curl_norm_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]) - 1, int(pn[1]), int(pn[2]) - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts)
#         curl_norm_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]) - 1, int(pn[1]) - 1, int(pn[2]), bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts)

#         # grad_PB; 1form
#         grad_PB[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]) - 1, int(pn[1]), int(pn[2]), bd1, bn2, bn3, span1, span2, span3, grad_PB1, starts)
#         grad_PB[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]), int(pn[1]) - 1, int(pn[2]), bn1, bd2, bn3, span1, span2, span3, grad_PB2, starts)
#         grad_PB[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]), int(pn[1]), int(pn[2]) - 1, bn1, bn2, bd3, span1, span2, span3, grad_PB3, starts)

#         # b_star; 2form transformed into H1vec
#         b_star[:] = (b + curl_norm_b*v*epsilon)/det_df

#         # calculate abs_b_star_para
#         abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

#         # operator bx() as matrix
#         b_prod[0, 1] = -b[2]
#         b_prod[0, 2] = +b[1]
#         b_prod[1, 0] = +b[2]
#         b_prod[1, 2] = -b[0]
#         b_prod[2, 0] = -b[1]
#         b_prod[2, 1] = +b[0]

#         norm_b2_prod[0, 1] = -norm_b2[2]
#         norm_b2_prod[0, 2] = +norm_b2[1]
#         norm_b2_prod[1, 0] = +norm_b2[2]
#         norm_b2_prod[1, 2] = -norm_b2[0]
#         norm_b2_prod[2, 0] = -norm_b2[1]
#         norm_b2_prod[2, 1] = +norm_b2[0]

#         if basis_u == 0:

#             linalg_kernels.matrix_matrix(b_prod, g_inv, tmp1)
#             linalg_kernels.matrix_matrix(tmp1, norm_b2_prod, tmp2)
#             linalg_kernels.matrix_matrix(tmp2, g_inv, tmp1)
#             linalg_kernels.matrix_vector(tmp1, grad_PB, tmp_v)

#             filling_v[:] = weight * tmp_v * mu / abs_b_star_para * scale_vec

#             # call the appropriate matvec filler
#             mvf.vec_fill_v0(pn, span1, span2, span3,
#                             bn1, bn2, bn3,
#                             starts,
#                             vec1, vec2, vec3,
#                             filling_v[0], filling_v[1], filling_v[2])

#         elif basis_u == 1:

#             linalg_kernels.matrix_matrix(g_inv, b_prod, tmp1)
#             linalg_kernels.matrix_matrix(tmp1, g_inv, tmp2)
#             linalg_kernels.matrix_matrix(tmp2, norm_b2_prod, tmp1)
#             linalg_kernels.matrix_matrix(tmp1, g_inv, tmp2)
#             linalg_kernels.matrix_vector(tmp2, grad_PB, tmp_v)

#             filling_v[:] = weight * tmp_v * mu / abs_b_star_para * scale_vec

#             # call the appropriate matvec filler
#             mvf.vec_fill_v1(pn, span1, span2, span3,
#                             bn1, bn2, bn3,
#                             bd1, bd2, bd3,
#                             starts,
#                             vec1, vec2, vec3,
#                             filling_v[0], filling_v[1], filling_v[2])

#         elif basis_u == 2:

#             linalg_kernels.matrix_matrix(b_prod, g_inv, tmp1)
#             linalg_kernels.matrix_matrix(tmp1, norm_b2_prod, tmp2)
#             linalg_kernels.matrix_matrix(tmp2, g_inv, tmp1)
#             linalg_kernels.matrix_vector(tmp1, grad_PB, tmp_v)

#             filling_v[:] = weight * tmp_v * mu / \
#                 abs_b_star_para / det_df * scale_vec

#             # call the appropriate matvec filler
#             mvf.vec_fill_v2(pn, span1, span2, span3,
#                             bn1, bn2, bn3, bd1, bd2, bd3,
#                             starts,
#                             vec1, vec2, vec3,
#                             filling_v[0], filling_v[1], filling_v[2])

#     vec1 /= n_markers_tot
#     vec2 /= n_markers_tot
#     vec3 /= n_markers_tot


# def cc_lin_mhd_5d_J2_dg_faster(markers: 'float[:,:]', n_markers_tot: 'int',
#                                pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
#                                starts: 'int[:]',
#                                kind_map: 'int', params_map: 'float[:]',
#                                p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
#                                ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
#                                cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
#                                mat11: 'float[:,:,:,:,:,:]',
#                                mat12: 'float[:,:,:,:,:,:]',
#                                mat13: 'float[:,:,:,:,:,:]',
#                                mat22: 'float[:,:,:,:,:,:]',
#                                mat23: 'float[:,:,:,:,:,:]',
#                                mat33: 'float[:,:,:,:,:,:]',
#                                vec1: 'float[:,:,:]',
#                                vec2: 'float[:,:,:]',
#                                vec3: 'float[:,:,:]',
#                                epsilon: float,    # model specific argument
#                                b1: 'float[:,:,:]',  # model specific argument
#                                b2: 'float[:,:,:]',  # model specific argument
#                                b3: 'float[:,:,:]',  # model specific argument
#                                # model specific argument
#                                norm_b11: 'float[:,:,:]',
#                                # model specific argument
#                                norm_b12: 'float[:,:,:]',
#                                # model specific argument
#                                norm_b13: 'float[:,:,:]',
#                                # model specific argument
#                                norm_b21: 'float[:,:,:]',
#                                # model specific argument
#                                norm_b22: 'float[:,:,:]',
#                                # model specific argument
#                                norm_b23: 'float[:,:,:]',
#                                # model specific argument
#                                curl_norm_b1: 'float[:,:,:]',
#                                # model specific argument
#                                curl_norm_b2: 'float[:,:,:]',
#                                # model specific argument
#                                curl_norm_b3: 'float[:,:,:]',
#                                # model specific argument
#                                grad_PB1: 'float[:,:,:]',
#                                # model specific argument
#                                grad_PB2: 'float[:,:,:]',
#                                # model specific argument
#                                grad_PB3: 'float[:,:,:]',
#                                gradI_const: 'float',         # model specific argument
#                                basis_u: 'int', scale_vec: 'float'):  # model specific argument
#     r"""Draft
#     """
#     # allocate for magnetic field evaluation
#     grad_PB = empty(3, dtype=float)

#     bn1 = empty(int(pn[0]) + 1, dtype=float)
#     bn2 = empty(int(pn[1]) + 1, dtype=float)
#     bn3 = empty(int(pn[2]) + 1, dtype=float)

#     bd1 = empty(int(pn[0]), dtype=float)
#     bd2 = empty(int(pn[1]), dtype=float)
#     bd3 = empty(int(pn[2]), dtype=float)

#     # allocate for filling
#     filling_v = empty(3, dtype=float)

#     tmp = empty((3, 3), dtype=float)
#     tmp_v1 = empty(3, dtype=float)
#     tmp_v2 = empty(3, dtype=float)

#     # get number of markers
#     n_markers_loc = shape(markers)[0]

#     for ip in range(n_markers_loc):

#         # only do something if particle is a "true" particle (i.e. not a hole)
#         if markers[ip, 0] == -1.:
#             continue

#         # marker positions
#         eta1 = markers[ip, 0]  # mid
#         eta2 = markers[ip, 1]  # mid
#         eta3 = markers[ip, 2]  # mid

#         # marker weight and velocity
#         weight = markers[ip, 5]
#         mu = markers[ip, 4]

#         # b-field evaluation
#         span1 = bsplines_kernels.find_span(tn1, int(pn[0]), eta1)
#         span2 = bsplines_kernels.find_span(tn2, int(pn[1]), eta2)
#         span3 = bsplines_kernels.find_span(tn3, int(pn[2]), eta3)

#         bsplines_kernels.b_d_splines_slim(tn1, int(pn[0]), eta1, span1, bn1, bd1)
#         bsplines_kernels.b_d_splines_slim(tn2, int(pn[1]), eta2, span2, bn2, bd2)
#         bsplines_kernels.b_d_splines_slim(tn3, int(pn[2]), eta3, span3, bn3, bd3)

#         # grad_PB; 1form
#         grad_PB[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]) - 1, int(pn[1]), int(pn[2]), bd1, bn2, bn3, span1, span2, span3, grad_PB1, starts)
#         grad_PB[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]), int(pn[1]) - 1, int(pn[2]), bn1, bd2, bn3, span1, span2, span3, grad_PB2, starts)
#         grad_PB[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]), int(pn[1]), int(pn[2]) - 1, bn1, bn2, bd3, span1, span2, span3, grad_PB3, starts)

#         tmp[:, :] = ((markers[ip, 18], markers[ip, 19], markers[ip, 20]),
#                      (markers[ip, 19], markers[ip, 21], markers[ip, 22]),
#                      (markers[ip, 20], markers[ip, 22], markers[ip, 23]))

#         if basis_u == 0:

#             linalg_kernels.matrix_vector(tmp, grad_PB, tmp_v1)
#             linalg_kernels.matrix_vector(tmp, markers[ip, 15:18], tmp_v2)

#             filling_v[:] = (tmp_v1 * mu * scale_vec +
#                             tmp_v2*gradI_const) * weight

#             # call the appropriate matvec filler
#             mvf.vec_fill_v0(pn, span1, span2, span3,
#                             bn1, bn2, bn3,
#                             starts,
#                             vec1, vec2, vec3,
#                             filling_v[0], filling_v[1], filling_v[2])

#         elif basis_u == 1:

#             linalg_kernels.matrix_vector(tmp, grad_PB, tmp_v1)
#             linalg_kernels.matrix_vector(tmp, markers[ip, 15:18], tmp_v2)

#             filling_v[:] = (tmp_v1 * mu * scale_vec +
#                             tmp_v2*gradI_const) * weight

#             # call the appropriate matvec filler
#             mvf.vec_fill_v1(pn, span1, span2, span3,
#                             bn1, bn2, bn3,
#                             bd1, bd2, bd3,
#                             starts,
#                             vec1, vec2, vec3,
#                             filling_v[0], filling_v[1], filling_v[2])

#         elif basis_u == 2:

#             linalg_kernels.matrix_vector(tmp, grad_PB, tmp_v1)
#             linalg_kernels.matrix_vector(tmp, markers[ip, 15:18], tmp_v2)

#             filling_v[:] = (tmp_v1 * mu * scale_vec +
#                             tmp_v2*gradI_const) * weight

#             # call the appropriate matvec filler
#             mvf.vec_fill_v2(pn, span1, span2, span3,
#                             bn1, bn2, bn3,
#                             bd1, bd2, bd3,
#                             starts,
#                             vec1, vec2, vec3,
#                             filling_v[0], filling_v[1], filling_v[2])

#     vec1 /= n_markers_tot
#     vec2 /= n_markers_tot
#     vec3 /= n_markers_tot


# def cc_lin_mhd_5d_J2_dg_prepare_faster(markers: 'float[:,:]', n_markers_tot: 'int',
#                                        pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
#                                        starts: 'int[:]',
#                                        kind_map: 'int', params_map: 'float[:]',
#                                        p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
#                                        ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
#                                        cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
#                                        mat11: 'float[:,:,:,:,:,:]',
#                                        mat12: 'float[:,:,:,:,:,:]',
#                                        mat13: 'float[:,:,:,:,:,:]',
#                                        mat22: 'float[:,:,:,:,:,:]',
#                                        mat23: 'float[:,:,:,:,:,:]',
#                                        mat33: 'float[:,:,:,:,:,:]',
#                                        vec1: 'float[:,:,:]',
#                                        vec2: 'float[:,:,:]',
#                                        vec3: 'float[:,:,:]',
#                                        epsilon: float,    # model specific argument
#                                        # model specific argument
#                                        b1: 'float[:,:,:]',
#                                        # model specific argument
#                                        b2: 'float[:,:,:]',
#                                        # model specific argument
#                                        b3: 'float[:,:,:]',
#                                        # model specific argument
#                                        norm_b11: 'float[:,:,:]',
#                                        # model specific argument
#                                        norm_b12: 'float[:,:,:]',
#                                        # model specific argument
#                                        norm_b13: 'float[:,:,:]',
#                                        # model specific argument
#                                        norm_b21: 'float[:,:,:]',
#                                        # model specific argument
#                                        norm_b22: 'float[:,:,:]',
#                                        # model specific argument
#                                        norm_b23: 'float[:,:,:]',
#                                        # model specific argument
#                                        curl_norm_b1: 'float[:,:,:]',
#                                        # model specific argument
#                                        curl_norm_b2: 'float[:,:,:]',
#                                        # model specific argument
#                                        curl_norm_b3: 'float[:,:,:]',
#                                        # model specific argument
#                                        grad_PB1: 'float[:,:,:]',
#                                        # model specific argument
#                                        grad_PB2: 'float[:,:,:]',
#                                        # model specific argument
#                                        grad_PB3: 'float[:,:,:]',
#                                        basis_u: 'int', scale_vec: 'float'):  # model specific argument
#     r"""Draft
#     """

#     # allocate for magnetic field evaluation
#     b = empty(3, dtype=float)
#     b_star = empty(3, dtype=float)
#     b_prod = zeros((3, 3), dtype=float)
#     norm_b2_prod = zeros((3, 3), dtype=float)
#     curl_norm_b = empty(3, dtype=float)
#     norm_b1 = empty(3, dtype=float)
#     norm_b2 = empty(3, dtype=float)
#     grad_PB = empty(3, dtype=float)

#     bn1 = empty(int(pn[0]) + 1, dtype=float)
#     bn2 = empty(int(pn[1]) + 1, dtype=float)
#     bn3 = empty(int(pn[2]) + 1, dtype=float)

#     bd1 = empty(int(pn[0]), dtype=float)
#     bd2 = empty(int(pn[1]), dtype=float)
#     bd3 = empty(int(pn[2]), dtype=float)

#     # allocate for metric coeffs
#     dfm = empty((3, 3), dtype=float)
#     df_inv = empty((3, 3), dtype=float)
#     df_inv_t = empty((3, 3), dtype=float)
#     g_inv = empty((3, 3), dtype=float)

#     # allocate for filling
#     filling_v = empty(3, dtype=float)

#     tmp1 = empty((3, 3), dtype=float)
#     tmp2 = empty((3, 3), dtype=float)
#     tmp_v = empty(3, dtype=float)

#     # get number of markers
#     n_markers_loc = shape(markers)[0]

#     for ip in range(n_markers_loc):

#         # only do something if particle is a "true" particle (i.e. not a hole)
#         if markers[ip, 0] == -1.:
#             continue

#         # marker positions
#         eta1 = markers[ip, 0]
#         eta2 = markers[ip, 1]
#         eta3 = markers[ip, 2]

#         # marker weight and velocity
#         weight = markers[ip, 5]
#         v = markers[ip, 3]
#         mu = markers[ip, 4]

#         # b-field evaluation
#         span1 = bsplines_kernels.find_span(tn1, int(pn[0]), eta1)
#         span2 = bsplines_kernels.find_span(tn2, int(pn[1]), eta2)
#         span3 = bsplines_kernels.find_span(tn3, int(pn[2]), eta3)

#         bsplines_kernels.b_d_splines_slim(tn1, int(pn[0]), eta1, span1, bn1, bd1)
#         bsplines_kernels.b_d_splines_slim(tn2, int(pn[1]), eta2, span2, bn2, bd2)
#         bsplines_kernels.b_d_splines_slim(tn3, int(pn[2]), eta3, span3, bn3, bd3)

#         # evaluate Jacobian, result in dfm
#         map_eval.df(eta1, eta2, eta3,
#                     kind_map, params_map,
#                     t1_map, t2_map, t3_map, p_map,
#                     ind1_map, ind2_map, ind3_map,
#                     cx, cy, cz,
#                     dfm)

#         det_df = linalg_kernels.det(dfm)

#         # needed metric coefficients
#         linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
#         linalg_kernels.transpose(df_inv, df_inv_t)
#         linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)

#         # b; 2form
#         b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]), int(pn[1]) - 1, int(pn[2]) - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts)
#         b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]) - 1, int(pn[1]), int(pn[2]) - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts)
#         b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]) - 1, int(pn[1]) - 1, int(pn[2]), bd1, bd2, bn3, span1, span2, span3, b3, starts)

#         # norm_b1; 1form
#         norm_b1[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]) - 1, int(pn[1]), int(pn[2]), bd1, bn2, bn3, span1, span2, span3, norm_b11, starts)
#         norm_b1[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]), int(pn[1]) - 1, int(pn[2]), bn1, bd2, bn3, span1, span2, span3, norm_b12, starts)
#         norm_b1[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]), int(pn[1]), int(pn[2]) - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts)

#         # norm_b2; 2form
#         norm_b2[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]), int(pn[1]) - 1, int(pn[2]) - 1, bn1, bd2, bd3, span1, span2, span3, norm_b21, starts)
#         norm_b2[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]) - 1, int(pn[1]), int(pn[2]) - 1, bd1, bn2, bd3, span1, span2, span3, norm_b22, starts)
#         norm_b2[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]) - 1, int(pn[1]) - 1, int(pn[2]), bd1, bd2, bn3, span1, span2, span3, norm_b23, starts)

#         # curl_norm_b; 2form
#         curl_norm_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]), int(pn[1]) - 1, int(pn[2]) - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts)
#         curl_norm_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]) - 1, int(pn[1]), int(pn[2]) - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts)
#         curl_norm_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]) - 1, int(pn[1]) - 1, int(pn[2]), bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts)

#         # grad_PB; 1form
#         grad_PB[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]) - 1, int(pn[1]), int(pn[2]), bd1, bn2, bn3, span1, span2, span3, grad_PB1, starts)
#         grad_PB[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]), int(pn[1]) - 1, int(pn[2]), bn1, bd2, bn3, span1, span2, span3, grad_PB2, starts)
#         grad_PB[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             int(pn[0]), int(pn[1]), int(pn[2]) - 1, bn1, bn2, bd3, span1, span2, span3, grad_PB3, starts)

#         # b_star; 2form transformed into H1vec
#         b_star[:] = (b + curl_norm_b*v*epsilon)/det_df

#         # calculate abs_b_star_para
#         abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

#         # operator bx() as matrix
#         b_prod[0, 1] = -b[2]
#         b_prod[0, 2] = +b[1]
#         b_prod[1, 0] = +b[2]
#         b_prod[1, 2] = -b[0]
#         b_prod[2, 0] = -b[1]
#         b_prod[2, 1] = +b[0]

#         norm_b2_prod[0, 1] = -norm_b2[2]
#         norm_b2_prod[0, 2] = +norm_b2[1]
#         norm_b2_prod[1, 0] = +norm_b2[2]
#         norm_b2_prod[1, 2] = -norm_b2[0]
#         norm_b2_prod[2, 0] = -norm_b2[1]
#         norm_b2_prod[2, 1] = +norm_b2[0]

#         if basis_u == 0:

#             linalg_kernels.matrix_matrix(b_prod, g_inv, tmp1)
#             linalg_kernels.matrix_matrix(tmp1, norm_b2_prod, tmp2)
#             linalg_kernels.matrix_matrix(tmp2, g_inv, tmp1)
#             linalg_kernels.matrix_vector(tmp1, grad_PB, tmp_v)

#             # saving S(H_n)
#             markers[ip, 18:21] = tmp1[0, :]/abs_b_star_para
#             markers[ip, 21:23] = tmp1[1, 1:3]/abs_b_star_para
#             markers[ip, 23] = tmp1[2, 2]/abs_b_star_para

#             filling_v[:] = weight * tmp_v * mu / abs_b_star_para * scale_vec

#             # call the appropriate matvec filler
#             mvf.vec_fill_v0(pn, span1, span2, span3,
#                             bn1, bn2, bn3,
#                             starts,
#                             vec1, vec2, vec3,
#                             filling_v[0], filling_v[1], filling_v[2])

#         elif basis_u == 1:

#             linalg_kernels.matrix_matrix(g_inv, b_prod, tmp1)
#             linalg_kernels.matrix_matrix(tmp1, g_inv, tmp2)
#             linalg_kernels.matrix_matrix(tmp2, norm_b2_prod, tmp1)
#             linalg_kernels.matrix_matrix(tmp1, g_inv, tmp2)
#             linalg_kernels.matrix_vector(tmp2, grad_PB, tmp_v)

#             # saving S(H_n)
#             markers[ip, 18:21] = tmp2[0, :]/abs_b_star_para
#             markers[ip, 21:23] = tmp2[1, 1:3]/abs_b_star_para
#             markers[ip, 23] = tmp2[2, 2]/abs_b_star_para

#             filling_v[:] = weight * tmp_v * mu / abs_b_star_para * scale_vec

#             # call the appropriate matvec filler
#             mvf.vec_fill_v1(pn, span1, span2, span3,
#                             bn1, bn2, bn3,
#                             bd1, bd2, bd3,
#                             starts,
#                             vec1, vec2, vec3,
#                             filling_v[0], filling_v[1], filling_v[2])

#         elif basis_u == 2:

#             linalg_kernels.matrix_matrix(b_prod, g_inv, tmp1)
#             linalg_kernels.matrix_matrix(tmp1, norm_b2_prod, tmp2)
#             linalg_kernels.matrix_matrix(tmp2, g_inv, tmp1)
#             linalg_kernels.matrix_vector(tmp1, grad_PB, tmp_v)

#             # saving S(H_n)
#             markers[ip, 18:21] = tmp1[0, :]/det_df/abs_b_star_para
#             markers[ip, 21:23] = tmp1[1, 1:3]/det_df/abs_b_star_para
#             markers[ip, 23] = tmp1[2, 2]/det_df/abs_b_star_para

#             filling_v[:] = weight * tmp_v * mu / \
#                 abs_b_star_para / det_df * scale_vec

#             # call the appropriate matvec filler
#             mvf.vec_fill_v2(pn, span1, span2, span3,
#                             bn1, bn2, bn3,
#                             bd1, bd2, bd3,
#                             starts,
#                             vec1, vec2, vec3,
#                             filling_v[0], filling_v[1], filling_v[2])

#     vec1 /= n_markers_tot
#     vec2 /= n_markers_tot
#     vec3 /= n_markers_tot
