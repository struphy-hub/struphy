from pyccel.decorators import stack_array

import struphy.b_splines.bsplines_kernels as bsp
from struphy.b_splines.bspline_evaluation_3d import eval_spline_mpi_kernel
import struphy.linear_algebra.core as linalg
import struphy.geometry.map_eval as map_eval

from numpy import empty, shape


@stack_array('bn1', 'bn2', 'bn3')
def eval_0_form_at_particles(markers: 'float[:,:]',
                             pn: 'int[:]',
                             tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                             starts: 'int[:]',
                             coeffs: 'float[:,:,:]') -> 'float':
    """
    Evaluate a 0 form at all the particle positions and sum up the result.

    Parameters
    ----------
        markers : array[float]
            .markers attribute of a struphy.pic.particles.Particles object

        pn : array[int]
            spline degrees

        tn1, tn2, tn3 : array[float]
            knot vectors

        starts : array[int]
            starts of the stencil objects

        coeffs : array[float]
            3d array of FE coeffs of the 0-form.
    """

    res = 0.0

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private(ip, eta1, eta2, eta3, span1, span2, span3, bn1, bn2, bn3)
    #$ omp for reduction( + : res)
    for ip in range(n_markers):

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.b_splines_slim(tn1, pn[0], eta1, span1, bn1)
        bsp.b_splines_slim(tn2, pn[1], eta2, span2, bn2)
        bsp.b_splines_slim(tn3, pn[2], eta3, span3, bn3)

        # sum up result
        res = res + eval_spline_mpi_kernel(pn[0], pn[1], pn[2],
                                       bn1, bn2, bn3, span1, span2, span3, coeffs, starts)

    #$ omp end parallel

    return res


@stack_array('bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def eval_1_form_at_particles(markers: 'float[:,:]',
                             pn: 'int[:]',
                             tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                             starts: 'int[:,:]',
                             coeffs1: 'float[:,:,:]', coeffs2: 'float[:,:,:]', coeffs3: 'float[:,:,:]',
                             res: 'float[:]'):
    """
    Evaluate a 1 form at all the particle positions and sum up the result.

    Parameters
    ----------
        markers : array[float]
            .markers attribute of a struphy.pic.particles.Particles object

        pn : array[int]
            spline degrees

        tn1, tn2, tn3 : array[float]
            knot vectors

        starts : array[int]
            starts of the stencil objects

        coeffs1, coeffs2, coeffs3 : array[float]
            3d array of FE coeffs of the 1-form.
    """

    res[:] = 0.0

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private(ip, eta1, eta2, eta3, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3)
    #$ omp for reduction( + : res)
    for ip in range(n_markers):

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        # sum up result
        res[0] = res[0] + eval_spline_mpi_kernel(pn[0], pn[1], pn[2],
                                             bd1, bn2, bn3, span1, span2, span3, coeffs1, starts[0])
        res[1] = res[1] + eval_spline_mpi_kernel(pn[0], pn[1], pn[2],
                                             bn1, bd2, bn3, span1, span2, span3, coeffs2, starts[1])
        res[2] = res[2] + eval_spline_mpi_kernel(pn[0], pn[1], pn[2],
                                             bn1, bn2, bd3, span1, span2, span3, coeffs3, starts[2])

    #$ omp end parallel


@stack_array('bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def eval_2_form_at_particles(markers: 'float[:,:]',
                             pn: 'int[:]',
                             tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                             starts: 'int[:,:]',
                             coeffs1: 'float[:,:,:]', coeffs2: 'float[:,:,:]', coeffs3: 'float[:,:,:]',
                             res: 'float[:]'):
    """
    Evaluate a 2 form at all the particle positions and sum up the result.

    Parameters
    ----------
        markers : array[float]
            .markers attribute of a struphy.pic.particles.Particles object

        pn : array[int]
            spline degrees

        tn1, tn2, tn3 : array[float]
            knot vectors

        starts : array[int]
            starts of the stencil objects

        coeffs1, coeffs2, coeffs3 : array[float]
            3d array of FE coeffs of the 2-form.
    """

    res[:] = 0.0

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private(ip, eta1, eta2, eta3, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3)
    #$ omp for
    for ip in range(n_markers):

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        # sum up result
        res[0] = res[0] + eval_spline_mpi_kernel(pn[0], pn[1], pn[2],
                                             bn1, bd2, bd3, span1, span2, span3, coeffs1, starts[0])
        res[1] = res[1] + eval_spline_mpi_kernel(pn[0], pn[1], pn[2],
                                             bd1, bn2, bd3, span1, span2, span3, coeffs2, starts[1])
        res[2] = res[2] + eval_spline_mpi_kernel(pn[0], pn[1], pn[2],
                                             bd1, bd2, bn3, span1, span2, span3, coeffs3, starts[2])

    #$ omp end parallel


@stack_array('bd1', 'bd2', 'bd3')
def eval_3_form_at_particles(markers: 'float[:,:]',
                             pn: 'int[:]',
                             tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                             starts: 'int[:]',
                             coeffs: 'float[:,:,:]') -> 'float':
    """
    Evaluate a 3 form at all the particle positions and sum up the result.

    Parameters
    ----------
        markers : array[float]
            .markers attribute of a struphy.pic.particles.Particles object

        pn : array[int]
            spline degrees

        tn1, tn2, tn3 : array[float]
            knot vectors

        starts : array[int]
            starts of the stencil objects

        coeffs : array[float]
            3d array of FE coeffs of the 3-form.
    """

    res = 0.0

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private(ip, eta1, eta2, eta3, span1, span2, span3, bd1, bd2, bd3)
    #$ omp for
    for ip in range(n_markers):

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.d_splines_slim(tn1, pn[0], eta1, span1, bd1)
        bsp.d_splines_slim(tn2, pn[1], eta2, span2, bd2)
        bsp.d_splines_slim(tn3, pn[2], eta3, span3, bd3)

        # sum up result
        res = res + eval_spline_mpi_kernel(pn[0], pn[1], pn[2],
                                       bd1, bd2, bd3, span1, span2, span3, coeffs, starts)

    #$ omp end parallel

    return res


@stack_array('bn1', 'bn2', 'bn3')
def eval_H1vec_at_particles(markers: 'float[:,:]',
                            pn: 'int[:]',
                            tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                            starts: 'int[:]',
                            coeffs1: 'float[:,:,:]', coeffs2: 'float[:,:,:]', coeffs3: 'float[:,:,:]',
                            res: 'float[:]'):
    """
    Evaluate a vector of 0-forms at all the particle positions and sum up the result.

    Parameters
    ----------
        markers : array[float]
            .markers attribute of a struphy.pic.particles.Particles object

        pn : array[int]
            spline degrees

        tn1, tn2, tn3 : array[float]
            knot vectors

        starts : array[int]
            starts of the stencil objects

        coeffs1, coeffs2, coeffs3 : array[float]
            3d array of FE coeffs of the H1vec-form.
    """

    res[:] = 0.0

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private(ip, eta1, eta2, eta3, span1, span2, span3, bn1, bn2, bn3)
    #$ omp for
    for ip in range(n_markers):

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.b_splines_slim(tn1, pn[0], eta1, span1, bn1)
        bsp.b_splines_slim(tn2, pn[1], eta2, span2, bn2)
        bsp.b_splines_slim(tn3, pn[2], eta3, span3, bn3)

        # sum up result
        res[0] = res[0] + eval_spline_mpi_kernel(pn[0], pn[1], pn[2],
                                             bn1, bn2, bn3, span1, span2, span3, coeffs1, starts)
        res[1] = res[1] + eval_spline_mpi_kernel(pn[0], pn[1], pn[2],
                                             bn1, bn2, bn3, span1, span2, span3, coeffs2, starts)
        res[2] = res[2] + eval_spline_mpi_kernel(pn[0], pn[1], pn[2],
                                             bn1, bn2, bn3, span1, span2, span3, coeffs3, starts)

    #$ omp end parallel


@stack_array('bn1', 'bn2', 'bn3')
def eval_magnetic_moments(markers: 'float[:,:]',
                          pn: 'int[:]',
                          tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                          starts0: 'int[:]',
                          b0: 'float[:,:,:]'):
    """
    Evaluate magnetic moments of each particles

    Parameters
    ----------
        markers : array[float]
            .markers attribute of a struphy.pic.particles.Particles object

        pn : array[int]
            spline degrees

        tn1, tn2, tn3 : array[float]
            knot vectors

        starts0 : array[int]
            starts of the stencil objects (0-form)

        b0 : array[float]
            3d array of FE coeffs of the absolute value of static magnetic field (0-form).
    """
    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        vperp_square = markers[ip,4]**2 + markers[ip,5]**2

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.b_splines_slim(tn1, pn[0], eta1, span1, bn1)
        bsp.b_splines_slim(tn2, pn[1], eta2, span2, bn2)
        bsp.b_splines_slim(tn3, pn[2], eta3, span3, bn3)

        b = eval_spline_mpi_kernel(pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, b0, starts0)

        markers[ip,4] = 1/2 * vperp_square / b


@stack_array('bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e', 'e_mid')
def push_gc1_discrete_gradients_stage_initial_guess(markers: 'float[:,:]', dt: float,
                                                    domain_array: 'float[:]',
                                                    pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                                                    starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                                                    kind_map: int, params_map: 'float[:]',
                                                    p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                                                    ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                                                    cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                                                    epsilon: float, 
                                                    abs_b: 'float[:,:,:]',
                                                    b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
                                                    norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
                                                    norm_b21: 'float[:,:,:]', norm_b22: 'float[:,:,:]', norm_b23: 'float[:,:,:]',
                                                    curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
                                                    grad_abs_b1: 'float[:,:,:]', grad_abs_b2: 'float[:,:,:]', grad_abs_b3: 'float[:,:,:]'):
    r"""
    Evaluate initial guess of the iterative solver

    ..math::

        \mathbf{H}^0_{n+1} = \mathbf{H}_n + h S_1(\mathbf{H}_n) \nabla I_1(\mathbf{H}_n)

    initial guess -> particles.marakers[:, 9:12]
    """
    # allocate metric coeffs
    df = empty((3, 3), dtype=float)
    df_t = empty((3, 3), dtype=float)
    g = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers
    S = empty((3, 3), dtype=float)
    temp = empty(3, dtype=float)
    grad_abs_b = empty(3, dtype=float)
    bcross = empty((3, 3), dtype=float)
    temp1 = empty((3, 3), dtype=float)
    norm_b2 = empty(3, dtype=float)
    temp2 = empty((3, 3), dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)

    # marker position e
    e = empty(3, dtype=float)
    e_mid = empty(3, dtype=float)
    
    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):
        
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        v = markers[ip,3]
        mu = markers[ip, 4]

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], e[0])
        span2 = bsp.find_span(tn2, pn[1], e[1])
        span3 = bsp.find_span(tn3, pn[2], e[2])

        bsp.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # eval all the needed field
        # grad_abs_b; 1form
        grad_abs_b[0] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts1[0])
        grad_abs_b[1] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts1[1])
        grad_abs_b[2] = eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts1[2])

        # assemble S
        S[:,:] = ( ( markers[ip, 13],  markers[ip, 14], markers[ip, 15]), 
                   (-markers[ip, 14],  markers[ip, 16], markers[ip, 17]), 
                   (-markers[ip, 15], -markers[ip, 17], markers[ip, 18]) )        

        # calculate S1 * grad I1
        linalg.matrix_vector(S, mu*grad_abs_b, temp)

        # save at the markers
        markers[ip, 0:3] = markers[ip, 0:3] + dt*temp[:]

        e_mid[:] = (markers[ip, 0:3] + markers[ip, 9:12])/2

        # check whether the mid position is already outside of the proc domain
        if (e_mid[0] < domain_array[0]) or (e_mid[0] > domain_array[1]):
            markers[ip,20] = -1.
            continue
        if (e_mid[1] < domain_array[3]) or (e_mid[1] > domain_array[4]):
            markers[ip,20] = -1.
            continue
        if (e_mid[2] < domain_array[6]) or (e_mid[2] > domain_array[7]):
            markers[ip,20] = -1.
            continue

        # evaluate Jacobian, result in df
        map_eval.df(e_mid[0], e_mid[1], e_mid[2],
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # evaluate inverse of G
        linalg.transpose(df, df_t)
        linalg.matrix_matrix(df_t, df, g)
        linalg.matrix_inv(g, g_inv)

        # metric coeffs
        det_df = linalg.det(df)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], e_mid[0])
        span2 = bsp.find_span(tn2, pn[1], e_mid[1])
        span3 = bsp.find_span(tn3, pn[2], e_mid[2])

        bsp.b_d_splines_slim(tn1, pn[0], e_mid[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], e_mid[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], e_mid[2], span3, bn3, bd3)

        # eval all the needed field
        # grad_abs_b; 1form
        grad_abs_b[0] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts1[0])
        grad_abs_b[1] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts1[1])
        grad_abs_b[2] = eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts1[2])

        # norm_b1; 1form
        norm_b1[0] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts1[0])
        norm_b1[1] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts1[1])
        norm_b1[2] = eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts1[2])

        # norm_b2; 2form
        norm_b2[0] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, norm_b21, starts2[0])
        norm_b2[1] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, norm_b22, starts2[1])
        norm_b2[2] = eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, norm_b23, starts2[2])

        # b_star; 2form
        b_star[0] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1 + epsilon*v*curl_norm_b1, starts2[0])
        b_star[1] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2 + epsilon*v*curl_norm_b2, starts2[1])
        b_star[2] = eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3 + epsilon*v*curl_norm_b3, starts2[2])

        # transform to H1vec
        b_star[:] = b_star/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg.scalar_dot(norm_b1, b_star)

        # assemble b cross (.) as 3x3 matrix
        bcross[:,:] = ( (     0.     ,  -norm_b2[2],   norm_b2[1]), 
                        (  norm_b2[2],      0.     ,  -norm_b2[0]), 
                        ( -norm_b2[1],   norm_b2[0],      0.     ) )

        # calculate G-1 b cross G-1 
        linalg.matrix_matrix(bcross, g_inv, temp1)
        linalg.matrix_matrix(g_inv, temp1, temp2)

        # calculate S
        S[:,:] = (epsilon*temp2)/abs_b_star_para

        # save at the markers
        markers[ip, 13:16] = S[0,:]
        markers[ip, 16:18] = S[1,1:3]
        markers[ip, 18]    = S[2,2]
        markers[ip, 20:23] = mu*grad_abs_b[:]



@stack_array('df', 'dfinv', 'g', 'g_inv', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e')
def push_gc1_discrete_gradients_stage_eval_S_and_I(markers: 'float[:,:]',
                                                   domain_array: 'float[:]',
                                                   pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                                                   starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                                                   kind_map: int, params_map: 'float[:]',
                                                   p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                                                   ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                                                   cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                                                   epsilon: float, 
                                                   abs_b: 'float[:,:,:]',
                                                   b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
                                                   norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
                                                   norm_b21: 'float[:,:,:]', norm_b22: 'float[:,:,:]', norm_b23: 'float[:,:,:]',
                                                   curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
                                                   grad_abs_b1: 'float[:,:,:]', grad_abs_b2: 'float[:,:,:]', grad_abs_b3: 'float[:,:,:]'):
    r"""
    Evaluate initial skew-symmetric :math:`S(\mathbf{H}_n)` matrix and :math:`I(\mathbf{H}_n)`in the following ODE system

    ..math::

        \dot \mathbf{H} = S_1(\mathbf{H},v_\parallel) \nabla I_1(\mathbf{H}) \,.

    where

    ..maht::

        \tilde S_1(\mathbf{H}_n, \mathbf{H}_{n+1}) \approx S_1(\mathbf{H}) = \frac{1}{\hat B^*_\parallel (\mathbf{H}, v_{\parallel,n})} \left[ \epsilon G^{-1} (\mathbf{H}) \mathbb{b}_{0,\otimes} G^{-1}(\mathbf{H})\right] \in \mathbb{R}^{3 \times 3} \,.
        \\
        I_1(\mathbf{H}_n) = \mu |\hat B^0_0 (\mathbf{H}_n)|

    | S1  S2  S3|
    |-S2  S4  S5|
    |-S3 -S5  S6|

    |S1 ~ S6| -> particles.marakers[:, 13:19]
    """

    # allocate metric coeffs
    df = empty((3, 3), dtype=float)
    df_t = empty((3, 3), dtype=float)
    g = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers 
    S = empty((3, 3), dtype=float)
    bcross = empty((3, 3), dtype=float)
    temp1 = empty((3, 3), dtype=float)
    norm_b2 = empty(3, dtype=float)
    temp2 = empty((3, 3), dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)

    # marker position e
    e = empty(3, dtype=float)
    
    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):
        
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        v = markers[ip, 3]
        mu = markers[ip, 4]

        # evaluate Jacobian, result in df
        map_eval.df(e[0], e[1], e[2],
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # evaluate inverse of G
        linalg.transpose(df, df_t)
        linalg.matrix_matrix(df_t, df, g)
        linalg.matrix_inv(g, g_inv)

        # metric coeffs
        det_df = linalg.det(df)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], e[0])
        span2 = bsp.find_span(tn2, pn[1], e[1])
        span3 = bsp.find_span(tn3, pn[2], e[2])

        bsp.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # eval all the needed field
        # abs_b; 0form
        abs_b0 = eval_spline_mpi_kernel(pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, abs_b, starts0)

        # norm_b1; 1form
        norm_b1[0] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts1[0])
        norm_b1[1] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts1[1])
        norm_b1[2] = eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts1[2])

        # norm_b2; 2form
        norm_b2[0] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, norm_b21, starts2[0])
        norm_b2[1] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, norm_b22, starts2[1])
        norm_b2[2] = eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, norm_b23, starts2[2])

        # b_star; 2form
        b_star[0] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1 + epsilon*v*curl_norm_b1, starts2[0])
        b_star[1] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2 + epsilon*v*curl_norm_b2, starts2[1])
        b_star[2] = eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3 + epsilon*v*curl_norm_b3, starts2[2])

        # transform to H1vec
        b_star[:] = b_star/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg.scalar_dot(norm_b1, b_star)

        # assemble b cross (.) as 3x3 matrix
        bcross[:,:] = ( (     0.     ,  -norm_b2[2],   norm_b2[1]), 
                        (  norm_b2[2],      0.     ,  -norm_b2[0]), 
                        ( -norm_b2[1],   norm_b2[0],      0.     ) )

        # calculate G-1 b cross G-1 
        linalg.matrix_matrix(bcross, g_inv, temp1)
        linalg.matrix_matrix(g_inv, temp1, temp2)

        # calculate S
        S[:,:] = (epsilon*temp2)/abs_b_star_para

        # save at the markers
        markers[ip, 13:16] = S[0,:]
        markers[ip, 16:18] = S[1,1:3]
        markers[ip, 18]    = S[2,2]
        markers[ip, 19]    = abs_b0*mu


@stack_array('bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e', 'e_mid')
def push_gc2_discrete_gradients_stage_initial_guess(markers: 'float[:,:]', dt: float,
                                                    domain_array: 'float[:]',
                                                    pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                                                    starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                                                    kind_map: int, params_map: 'float[:]',
                                                    p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                                                    ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                                                    cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                                                    epsilon: float, 
                                                    abs_b: 'float[:,:,:]',
                                                    b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
                                                    norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
                                                    norm_b21: 'float[:,:,:]', norm_b22: 'float[:,:,:]', norm_b23: 'float[:,:,:]',
                                                    curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
                                                    grad_abs_b1: 'float[:,:,:]', grad_abs_b2: 'float[:,:,:]', grad_abs_b3: 'float[:,:,:]'):
    r"""
    Evaluate initial guess of the iterative solver

    ..math::

        \mathbf{z}^0_{n+1} = \mathbf{z}_n + h S_2(\mathbf{H}_n) \nabla I_2(\mathbf{H}_n)

    initial guess -> particles.marakers[:, 9:12]
    """
    # allocate metric coeffs
    df = empty((3, 3), dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers
    grad_abs_b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)

    # marker position e
    e = empty(3, dtype=float)
    e_mid = empty(3, dtype=float)
    
    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):
        
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        v = markers[ip,3]
        mu = markers[ip, 4]

        # evaluate Jacobian, result in df
        map_eval.df(e[0], e[1], e[2],
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # metric coeffs
        det_df = linalg.det(df)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], e[0])
        span2 = bsp.find_span(tn2, pn[1], e[1])
        span3 = bsp.find_span(tn3, pn[2], e[2])

        bsp.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # eval all the needed field
        # grad_abs_b; 1form
        grad_abs_b[0] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts1[0])
        grad_abs_b[1] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts1[1])
        grad_abs_b[2] = eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts1[2])

        # norm_b1; 1form
        norm_b1[0] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts1[0])
        norm_b1[1] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts1[1])
        norm_b1[2] = eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts1[2])

        # b_star; 2form
        b_star[0] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1 + epsilon*v*curl_norm_b1, starts2[0])
        b_star[1] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2 + epsilon*v*curl_norm_b2, starts2[1])
        b_star[2] = eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3 + epsilon*v*curl_norm_b3, starts2[2])

        # transform to H1vec
        b_star[:] = b_star/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg.scalar_dot(norm_b1, b_star)

        # calculate b_star . grad_abs_b
        b_star_dot_grad_abs_b = linalg.scalar_dot(b_star, mu*grad_abs_b)

        # save at the markers
        markers[ip, 0:3] = markers[ip, 9:12] + dt*b_star[:]/abs_b_star_para*v
        markers[ip, 3] = markers[ip, 12] - dt*b_star_dot_grad_abs_b/abs_b_star_para

        e_mid[:] = (markers[ip, 0:3] + markers[ip, 9:12])/2.
        v_mid = (markers[ip, 3] + markers[ip, 12])/2.

        # check whether the mid position is already outside of the proc domain
        if (e_mid[0] < domain_array[0]) or (e_mid[0] > domain_array[1]):
            markers[ip,20] = -1.
            continue
        if (e_mid[1] < domain_array[3]) or (e_mid[1] > domain_array[4]):
            markers[ip,20] = -1.
            continue
        if (e_mid[2] < domain_array[6]) or (e_mid[2] > domain_array[7]):
            markers[ip,20] = -1.
            continue

        # evaluate Jacobian, result in df
        map_eval.df(e_mid[0], e_mid[1], e_mid[2],
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # metric coeffs
        det_df = linalg.det(df)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], e_mid[0])
        span2 = bsp.find_span(tn2, pn[1], e_mid[1])
        span3 = bsp.find_span(tn3, pn[2], e_mid[2])

        bsp.b_d_splines_slim(tn1, pn[0], e_mid[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], e_mid[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], e_mid[2], span3, bn3, bd3)

        # eval all the needed field
        # grad_abs_b; 1form
        grad_abs_b[0] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts1[0])
        grad_abs_b[1] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts1[1])
        grad_abs_b[2] = eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts1[2])

        # norm_b1; 1form
        norm_b1[0] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts1[0])
        norm_b1[1] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts1[1])
        norm_b1[2] = eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts1[2])

        # b_star; 2form
        b_star[0] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1 + epsilon*v_mid*curl_norm_b1, starts2[0])
        b_star[1] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2 + epsilon*v_mid*curl_norm_b2, starts2[1])
        b_star[2] = eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3 + epsilon*v_mid*curl_norm_b3, starts2[2])

        # transform to H1vec
        b_star[:] = b_star/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg.scalar_dot(norm_b1, b_star)

        # save at the markers
        markers[ip, 13:16] = b_star[:]/abs_b_star_para
        markers[ip, 20:23] = mu*grad_abs_b[:]


@stack_array('df', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e')
def push_gc2_discrete_gradients_stage_eval_S_and_I(markers: 'float[:,:]',
                                                   domain_array: 'float[:]',
                                                   pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                                                   starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                                                   kind_map: int, params_map: 'float[:]',
                                                   p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                                                   ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                                                   cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                                                   epsilon: float, 
                                                   abs_b: 'float[:,:,:]',
                                                   b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
                                                   norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
                                                   norm_b21: 'float[:,:,:]', norm_b22: 'float[:,:,:]', norm_b23: 'float[:,:,:]',
                                                   curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
                                                   grad_abs_b1: 'float[:,:,:]', grad_abs_b2: 'float[:,:,:]', grad_abs_b3: 'float[:,:,:]'):
    r"""TODO

    """

    # allocate metric coeffs
    df = empty((3, 3), dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers 
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)

    # marker position e
    e = empty(3, dtype=float)
    
    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):
        
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        v = markers[ip, 3]
        mu = markers[ip, 4]

        # evaluate Jacobian, result in df
        map_eval.df(e[0], e[1], e[2],
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # metric coeffs
        det_df = linalg.det(df)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], e[0])
        span2 = bsp.find_span(tn2, pn[1], e[1])
        span3 = bsp.find_span(tn3, pn[2], e[2])

        bsp.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # eval all the needed field
        # abs_b; 0form
        abs_b0 = eval_spline_mpi_kernel(pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, abs_b, starts0)

        # norm_b1; 1form
        norm_b1[0] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts1[0])
        norm_b1[1] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts1[1])
        norm_b1[2] = eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts1[2])

        # b_star; 2form
        b_star[0] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1 + epsilon*v*curl_norm_b1, starts2[0])
        b_star[1] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2 + epsilon*v*curl_norm_b2, starts2[1])
        b_star[2] = eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3 + epsilon*v*curl_norm_b3, starts2[2])

        # transform to H1vec
        b_star[:] = b_star/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg.scalar_dot(norm_b1, b_star)

        # save at the markers
        markers[ip, 13:16] = b_star[:]/abs_b_star_para
        markers[ip, 19] = mu*abs_b0