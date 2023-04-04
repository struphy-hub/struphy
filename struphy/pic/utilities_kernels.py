from pyccel.decorators import stack_array

import struphy.b_splines.bsplines_kernels as bsp
from struphy.b_splines.bspline_evaluation_3d import eval_spline_mpi_kernel
import struphy.linear_algebra.core as linalg
import struphy.geometry.map_eval as map_eval

from numpy import empty, shape, zeros, sqrt, log


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


@stack_array('bn1', 'bn2', 'bn3', 'b_cart', 'norm_b_cart', 'v', 'temp', 'v_perp')
def eval_magnetic_moment(markers: 'float[:,:]',
                         pn: 'int[:]',
                         tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                         starts0: 'int[:]',
                         b_cart_1: 'float[:,:,:]',        
                         b_cart_2: 'float[:,:,:]',      
                         b_cart_3: 'float[:,:,:]'):
    """
    Evaluate parallel velocity and magnetic moment of each particles and asign it into markers[ip,3] and markers[ip,4] respectively.

    Parameters
    ----------
        markers : array[float]
            .markers attribute of a struphy.pic.particles.Particles object

        epsilon : array[float]
            omega_th/omega_c = k*rho = rhostar

        pn : array[int]
            spline degrees

        tn1, tn2, tn3 : array[float]
            knot vectors

        starts0 : array[int]
            starts of the stencil objects (0-form)

        unit_b_cart_x : array[float]
            3d array of FE coeffs of the x component of unit cartesian equilibrium magnetic field 
    """
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    b_cart = empty(3, dtype=float)
    norm_b_cart = empty(3, dtype=float)
    v = empty(3, dtype=float)
    temp = empty(3, dtype=float)
    v_perp = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        v[:] = markers[ip, 3:6]

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.b_splines_slim(tn1, pn[0], eta1, span1, bn1)
        bsp.b_splines_slim(tn2, pn[1], eta2, span2, bn2)
        bsp.b_splines_slim(tn3, pn[2], eta3, span3, bn3)

        # b_cart
        b_cart[0] = eval_spline_mpi_kernel(pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, b_cart_1, starts0)
        b_cart[1] = eval_spline_mpi_kernel(pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, b_cart_2, starts0)
        b_cart[2] = eval_spline_mpi_kernel(pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, b_cart_3, starts0)

        # calculate absB
        absB = sqrt(b_cart[0]**2 + b_cart[1]**2 + b_cart[2]**2)

        if absB != 0.:
            norm_b_cart[:] = b_cart/absB
        else:
            norm_b_cart[:] = b_cart

        # calculate parallel velocity
        v_parallel = linalg.scalar_dot(norm_b_cart, v)

        # extract perpendicular velocity
        linalg.cross(v, norm_b_cart, temp)
        linalg.cross(norm_b_cart, temp, v_perp)

        v_perp_square = (v_perp[0]**2 + v_perp[1]**2 +v_perp[2]**2)

        # parallel velocity
        markers[ip,3] = v_parallel
        # magnetic moment
        markers[ip,4] = 1/2 * v_perp_square / absB
        # empty leftovers
        markers[ip,5] = 0.

@stack_array('bn1', 'bn2', 'bn3', 'b_cart', 'norm_b_cart', 'v', 'temp', 'v_perp')
def transform_6D_to_5D(markers: 'float[:,:]', epsilon: 'float',
                         pn: 'int[:]',
                         tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                         starts0: 'int[:]',
                         b_cart_1: 'float[:,:,:]',        
                         b_cart_2: 'float[:,:,:]',      
                         b_cart_3: 'float[:,:,:]'):
    """
    Evaluate parallel velocity and magnetic moment of each particles and asign it into markers[ip,3] and markers[ip,4] respectively.

    Parameters
    ----------
        markers : array[float]
            .markers attribute of a struphy.pic.particles.Particles object

        epsilon : array[float]
            omega_th/omega_c = k*rho = rhostar

        pn : array[int]
            spline degrees

        tn1, tn2, tn3 : array[float]
            knot vectors

        starts0 : array[int]
            starts of the stencil objects (0-form)

        unit_b_cart_x : array[float]
            3d array of FE coeffs of the x component of unit cartesian equilibrium magnetic field 
    """
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    b_cart = empty(3, dtype=float)
    norm_b_cart = empty(3, dtype=float)
    v = empty(3, dtype=float)
    temp = empty(3, dtype=float)
    v_perp = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        v[:] = markers[ip, 3:6]

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.b_splines_slim(tn1, pn[0], eta1, span1, bn1)
        bsp.b_splines_slim(tn2, pn[1], eta2, span2, bn2)
        bsp.b_splines_slim(tn3, pn[2], eta3, span3, bn3)

        # b_cart
        b_cart[0] = eval_spline_mpi_kernel(pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, b_cart_1, starts0)
        b_cart[1] = eval_spline_mpi_kernel(pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, b_cart_2, starts0)
        b_cart[2] = eval_spline_mpi_kernel(pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, b_cart_3, starts0)

        # calculate absB
        absB = sqrt(b_cart[0]**2 + b_cart[1]**2 + b_cart[2]**2)

        if absB != 0.:
            norm_b_cart[:] = b_cart/absB
        else:
            norm_b_cart[:] = b_cart

        # calculate parallel velocity
        v_parallel = linalg.scalar_dot(norm_b_cart, v)

        # extract perpendicular velocity
        linalg.cross(v, norm_b_cart, temp)
        linalg.cross(norm_b_cart, temp, v_perp)

        # applying epsilon
        v_parallel /= epsilon
        v_perp /= epsilon

        v_perp_square = (v_perp[0]**2 + v_perp[1]**2 +v_perp[2]**2)

        # parallel velocity
        markers[ip,3] = v_parallel
        # magnetic moment
        markers[ip,4] = 1/2 * v_perp_square / absB
        # empty leftovers
        markers[ip,5] = 0.

@stack_array('bn1', 'bn2', 'bn3')
def eval_magnetic_energy(markers: 'float[:,:]',
                         pn: 'int[:]',
                         tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                         starts0: 'int[:]',
                         b0: 'float[:,:,:]'):
    """
    Evaluate magnetic field energy of each particles

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

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.b_splines_slim(tn1, pn[0], eta1, span1, bn1)
        bsp.b_splines_slim(tn2, pn[1], eta2, span2, bn2)
        bsp.b_splines_slim(tn3, pn[2], eta3, span3, bn3)

        b = eval_spline_mpi_kernel(pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, b0, starts0)

        markers[ip, 5] = b*markers[ip, 4]


@stack_array('df', 'dfinv', 'g', 'g_inv', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e')
def push_gc1_discrete_gradients_prepare(markers: 'float[:,:]', dt: float,
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
    b = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)
    S = empty((3, 3), dtype=float)
    temp = empty(3, dtype=float)
    bcross = empty((3, 3), dtype=float)
    temp1 = empty((3, 3), dtype=float)
    norm_b2 = empty(3, dtype=float)
    temp2 = empty((3, 3), dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    grad_abs_b = empty(3, dtype=float)

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

        # b; 2form
        b[0] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts2[0])
        b[1] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts2[1])
        b[2] = eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts2[2])

        # curl_norm_b; 2form
        curl_norm_b[0] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts2[0])
        curl_norm_b[1] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts2[1])
        curl_norm_b[2] = eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts2[2])

        # grad_abs_b; 1form
        grad_abs_b[0] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts1[0])
        grad_abs_b[1] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts1[1])
        grad_abs_b[2] = eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts1[2])

        # transform to H1vec
        b_star[:] = b + epsilon*v*curl_norm_b
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

        # calculate S1 * grad I1
        linalg.matrix_vector(S, grad_abs_b, temp)

        # save at the markers
        markers[ip, 0:3] = markers[ip, 0:3] + dt*temp[:]*mu

        markers[ip, 20:24] = markers[ip, 0:4]
        markers[ip, 0:4] = (markers[ip, 0:4] + markers[ip, 9:13])/2.

@stack_array('df', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e')
def push_gc2_discrete_gradients_prepare(markers: 'float[:,:]', dt: float,
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
    b = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)
    grad_abs_b = empty(3, dtype=float)
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

        # b; 2form
        b[0] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts2[0])
        b[1] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts2[1])
        b[2] = eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts2[2])

        # curl_norm_b; 2form
        curl_norm_b[0] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts2[0])
        curl_norm_b[1] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts2[1])
        curl_norm_b[2] = eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts2[2])

        # grad_abs_b; 1form
        grad_abs_b[0] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts1[0])
        grad_abs_b[1] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts1[1])
        grad_abs_b[2] = eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts1[2])

        # transform to H1vec
        b_star[:] = b + epsilon*v*curl_norm_b
        b_star[:] = b_star/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg.scalar_dot(norm_b1, b_star)

        # save at the markers
        markers[ip, 13:16] = b_star[:]/abs_b_star_para
        markers[ip, 19] = mu*abs_b0

        # calculate b_star . grad_abs_b
        b_star_dot_grad_abs_b = linalg.scalar_dot(b_star, grad_abs_b)*mu

        # save at the markers
        markers[ip, 0:3] = markers[ip, 9:12] + dt*b_star[:]/abs_b_star_para*v
        markers[ip, 3] = markers[ip, 12] - dt*b_star_dot_grad_abs_b/abs_b_star_para

        markers[ip, 20:24] = markers[ip, 0:4]
        markers[ip, 0:4] = (markers[ip, 0:4] + markers[ip, 9:13])/2.

@stack_array('df', 'dfinv', 'g', 'g_inv', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e')
def push_gc1_discrete_gradients_faster_prepare(markers: 'float[:,:]', dt: float,
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
    b = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)
    S = empty((3, 3), dtype=float)
    temp = empty(3, dtype=float)
    bcross = empty((3, 3), dtype=float)
    temp1 = empty((3, 3), dtype=float)
    norm_b2 = empty(3, dtype=float)
    temp2 = empty((3, 3), dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    grad_abs_b = empty(3, dtype=float)

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

        # b; 2form
        b[0] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts2[0])
        b[1] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts2[1])
        b[2] = eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts2[2])

        # curl_norm_b; 2form
        curl_norm_b[0] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts2[0])
        curl_norm_b[1] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts2[1])
        curl_norm_b[2] = eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts2[2])

        # grad_abs_b; 1form
        grad_abs_b[0] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts1[0])
        grad_abs_b[1] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts1[1])
        grad_abs_b[2] = eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts1[2])

        # transform to H1vec
        b_star[:] = b + epsilon*v*curl_norm_b
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

        # calculate S1 * grad I1
        linalg.matrix_vector(S, grad_abs_b, temp)

        # save at the markers
        markers[ip, 0:3] = markers[ip, 0:3] + dt*temp[:]*mu

        markers[ip, 20:23] = markers[ip, 0:3]
        markers[ip, 0:3] = (markers[ip, 0:3] + markers[ip, 9:12])/2.

@stack_array('df', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e')
def push_gc2_discrete_gradients_faster_prepare(markers: 'float[:,:]', dt: float,
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
    b = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)
    grad_abs_b = empty(3, dtype=float)
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

        # b; 2form
        b[0] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts2[0])
        b[1] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts2[1])
        b[2] = eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts2[2])

        # curl_norm_b; 2form
        curl_norm_b[0] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts2[0])
        curl_norm_b[1] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts2[1])
        curl_norm_b[2] = eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts2[2])

        # grad_abs_b; 1form
        grad_abs_b[0] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts1[0])
        grad_abs_b[1] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts1[1])
        grad_abs_b[2] = eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts1[2])

        # transform to H1vec
        b_star[:] = b + epsilon*v*curl_norm_b
        b_star[:] = b_star/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg.scalar_dot(norm_b1, b_star)

        # save at the markers
        markers[ip, 13:16] = b_star[:]/abs_b_star_para
        markers[ip, 19] = mu*abs_b0

        # calculate b_star . grad_abs_b
        b_star_dot_grad_abs_b = linalg.scalar_dot(b_star, grad_abs_b)*mu

        # save at the markers
        markers[ip, 0:3] = markers[ip, 9:12] + dt*b_star[:]/abs_b_star_para*v
        markers[ip, 3] = markers[ip, 12] - dt*b_star_dot_grad_abs_b/abs_b_star_para

        markers[ip, 20:24] = markers[ip, 0:4]
        markers[ip, 0:4] = (markers[ip, 0:4] + markers[ip, 9:13])/2.

@stack_array('df', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e_mid')
def push_gc1_discrete_gradients_faster_eval_gradI(markers: 'float[:,:]', dt: float,
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
    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers 
    grad_abs_b = empty(3, dtype=float)

    # marker position e
    e_mid = empty(3, dtype=float)
    
    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):
        
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 23] == -1.:
            continue

        e_mid[:] = markers[ip, 0:3]
        markers[ip, 0:3] = markers[ip, 20:23]
        mu = markers[ip, 4]

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

        markers[ip, 20:23] = mu*grad_abs_b[:]

@stack_array('df', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e_mid')
def push_gc2_discrete_gradients_faster_eval_gradI(markers: 'float[:,:]', dt: float,
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
    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers 
    grad_abs_b = empty(3, dtype=float)

    # marker position e
    e_mid = empty(3, dtype=float)
    
    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):
        
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 23] == -1.:
            continue

        e_mid[:] = markers[ip, 0:3]
        markers[ip, 0:3] = markers[ip, 20:23]
        mu = markers[ip, 4]

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

        markers[ip, 20:23] = mu*grad_abs_b[:]

@stack_array('df', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e_mid')
def push_gc1_discrete_gradients_eval_gradI(markers: 'float[:,:]', dt: float,
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
    temp2 = empty((3, 3), dtype=float)
    grad_abs_b = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    norm_b2 = empty(3, dtype=float)
    b = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)

    # marker position e
    e_mid = empty(3, dtype=float)
    
    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):
        
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 23] == -1.:
            continue

        e_mid[:] = markers[ip, 0:3]
        v_mid = markers[ip, 3]
        markers[ip, 0:4] = markers[ip, 20:24]
        mu = markers[ip, 4]

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
        # norm_b1; 1form
        norm_b1[0] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts1[0])
        norm_b1[1] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts1[1])
        norm_b1[2] = eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts1[2])

        # norm_b2; 2form
        norm_b2[0] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, norm_b21, starts2[0])
        norm_b2[1] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, norm_b22, starts2[1])
        norm_b2[2] = eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, norm_b23, starts2[2])

        # b; 2form
        b[0] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts2[0])
        b[1] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts2[1])
        b[2] = eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts2[2])

        # curl_norm_b; 2form
        curl_norm_b[0] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts2[0])
        curl_norm_b[1] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts2[1])
        curl_norm_b[2] = eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts2[2])

        # grad_abs_b; 1form
        grad_abs_b[0] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts1[0])
        grad_abs_b[1] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts1[1])
        grad_abs_b[2] = eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts1[2])

        # transform to H1vec
        b_star[:] = b + epsilon*v_mid*curl_norm_b
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

@stack_array('df', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e_mid')
def push_gc2_discrete_gradients_eval_gradI(markers: 'float[:,:]', dt: float,
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
    grad_abs_b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    b = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)

    # marker position e
    e_mid = empty(3, dtype=float)
    
    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):
        
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 23] == -1.:
            continue

        e_mid[:] = markers[ip, 0:3]
        v_mid = markers[ip, 3]
        markers[ip, 0:4] = markers[ip, 20:24]
        mu = markers[ip, 4]

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
        # norm_b1; 1form
        norm_b1[0] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts1[0])
        norm_b1[1] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts1[1])
        norm_b1[2] = eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts1[2])

        # b; 2form
        b[0] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts2[0])
        b[1] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts2[1])
        b[2] = eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts2[2])

        # curl_norm_b; 2form
        curl_norm_b[0] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts2[0])
        curl_norm_b[1] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts2[1])
        curl_norm_b[2] = eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts2[2])

        # grad_abs_b; 1form
        grad_abs_b[0] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts1[0])
        grad_abs_b[1] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts1[1])
        grad_abs_b[2] = eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts1[2])

        # transform to H1vec
        b_star[:] = b + epsilon*v_mid*curl_norm_b
        b_star[:] = b_star/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg.scalar_dot(norm_b1, b_star)

        # save at the markers
        markers[ip, 13:16] = b_star[:]/abs_b_star_para
        markers[ip, 20:23] = mu*grad_abs_b[:]


@stack_array('grad_PB', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def accum_gradI_const(markers: 'float[:,:]', n_markers_tot: 'int',
                      pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                      starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                      grad_PB1: 'float[:,:,:]', grad_PB2: 'float[:,:,:]', grad_PB3: 'float[:,:,:]',
                      scale: 'float'):
    
    r"""TODO
    """
    # allocate for magnetic field evaluation
    grad_PB = empty(3, dtype=float)

    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # allocate for filling
    res = zeros(1, dtype=float)

    # get number of markers
    n_markers_loc = shape(markers)[0]
    
    for ip in range(n_markers_loc):
        
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker positions
        eta1 = markers[ip, 0] # mid
        eta2 = markers[ip, 1] # mid
        eta3 = markers[ip, 2] # mid

        # marker weight and velocity
        weight = markers[ip, 6]
        mu = markers[ip, 4]

        # b-field evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        # grad_PB; 1form
        grad_PB[0] = eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_PB1, starts1[0])
        grad_PB[1] = eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_PB2, starts1[1])
        grad_PB[2] = eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_PB3, starts1[2])

        res += linalg.scalar_dot(markers[ip, 15:18] , grad_PB) * weight * mu * scale
        
    return res/n_markers_tot

@stack_array('e', 'e_diff')
def check_eta_diff(markers: 'float[:,:]'):
    r'''TODO
    '''
    # marker position e
    e = empty(3, dtype=float)
    e_diff = empty(3, dtype=float)

    # get number of markers
    n_markers_loc = shape(markers)[0]

    for ip in range(n_markers_loc):
        
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        e_diff[:] = e[:] - markers[ip, 9:12]

        for axis in range(3):
            if e_diff[axis] > 0.5:
                e_diff[axis] -= 1.
            elif e_diff[axis] < -0.5:
                e_diff[axis] += 1.

        markers[ip,15:18] = e_diff[:]

@stack_array('e', 'e_diff', 'e_mid')
def check_eta_mid(markers: 'float[:,:]'):
    r'''TODO
    '''
    # marker position e
    e = empty(3, dtype=float)
    e_diff = empty(3, dtype=float)
    e_mid = empty(3, dtype=float)

    # get number of markers
    n_markers_loc = shape(markers)[0]

    for ip in range(n_markers_loc):
        
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        e_diff[:] = e[:] - markers[ip, 9:12]
        e_mid[:] = (e[:] + markers[ip, 9:12])/2.

        for axis in range(3):
            if e_diff[axis] > 0.5:
                e_mid[axis] += 0.5
            elif e_diff[axis] < -0.5:
                e_mid[axis] += 0.5

        markers[ip,12:15] = e_mid[:]




@stack_array('df', 'dfinv', 'dfinv_t', 'e', 'v', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'a_form', 'dfta_form')
def canonical_kinetic_particles(res: 'float[:]', markers: 'float[:,:]',
                      pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                      starts1: 'int[:,:]',
                      kind_map: int, params_map: 'float[:]',
                      p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                      ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                      cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                      a1_1: 'float[:,:,:]', a1_2: 'float[:,:,:]', a1_3: 'float[:,:,:]'):
    
    r'''
    Calculate kinetic energy of each particle and sum up the result.

    Parameters
    ----------
    	res : array[float]
    		array to store the sum of kinetic energy of particles

        markers : array[float]
            markers attribute of a struphy.pic.particles.Particles object

        pn : array[int]
            spline degrees

        tn1, tn2, tn3 : array[float]
            knot vectors

        starts1 : array[int]
            starts of the stencil objects

        kind_map ->  cz:
            domain information

        a1_1, a1_2, a1_3 : array[float]
        	coefficients of one form (vector potential)

    .. math:: 
    	\begin{align*}
			\frac{1}{2} \sum_p w_p |{\mathbf p} -  \hat{\mathbf A}^1({\boldsymbol \eta}_p)|^2.
        \end{align*}
    '''

    res[:] = 0.0 
    # allocate metric coeffs
    df = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)

    # allocate for field evaluations (1-form components)
    a_form = empty(3, dtype=float)
    dfta_form =  empty(3, dtype=float)
    # particle position and velocity
    e = empty(3, dtype=float)
    v = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private (ip, e, v, w, df, dfinv, dfinv_t, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, a_form, dfta_form)
    #$ omp for reduction( + : res)
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        v[:] = markers[ip, 3:6]
        w    = markers[ip,   6]
        # evaluate Jacobian, result in df
        map_eval.df(e[0], e[1], e[2],
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        linalg.matrix_inv(df, dfinv)
        linalg.transpose(dfinv, dfinv_t)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], e[0])
        span2 = bsp.find_span(tn2, pn[1], e[1])
        span3 = bsp.find_span(tn3, pn[2], e[2])

        bsp.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # magnetic field: 2-form components
        a_form[0] = eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, a1_1, starts1[0])
        a_form[1] = eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, a1_2, starts1[1])
        a_form[2] = eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, a1_3, starts1[2])

        dfta_form[0] = dfinv_t[0,0] * a_form[0] + dfinv_t[0,1] * a_form[1] + dfinv_t[0,2] * a_form[2]
        dfta_form[1] = dfinv_t[1,0] * a_form[0] + dfinv_t[1,1] * a_form[1] + dfinv_t[1,2] * a_form[2]
        dfta_form[2] = dfinv_t[2,0] * a_form[0] + dfinv_t[2,1] * a_form[1] + dfinv_t[2,2] * a_form[2]

        res[0] += 0.5 * w * ( (v[0] - dfta_form[0]) ** 2.0 + (v[1] - dfta_form[1]) ** 2.0 + (v[2] - dfta_form[2]) ** 2.0 )

    #$ omp end parallel





@stack_array('det_df', 'df')
def thermal_energy(res: 'float[:]', density: 'float[:,:,:,:,:,:]', 
                  pads1 : int, pads2 : int, pads3 : int,
                  nel1 : 'int', nel2 : 'int', nel3 : 'int', 
                  nq1 : int, nq2 : int, nq3 : int, 
                  w1 : 'float[:,:]', w2 : 'float[:,:]', w3 : 'float[:,:]', 
                  pts1 : 'float[:,:]', pts2 : 'float[:,:]', pts3 : 'float[:,:]', 
                  kind_map: int, params_map: 'float[:]',
                  p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                  ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                  cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]'):
    
    r'''
    Calculate thermal energy of electron.

    Parameters
    ----------
        res : array[float]
            array to store the thermal energy of electrons

        density : array[float]
            array to store values of density at quadrature points in each cell

        pads1 - pads3 : int
            size of ghost region in each direction

        nel1 - nel3 : array[int]
            number of cells in each direction

        nq1 - nq3 : array[int]
            number of quadrature points in each direction of each cell

        w1 - w3: array[float]
            quadrature weights in each cell 

        pts1 - pts3: array[float]
            quadrature points in each cell 

        starts1 : array[int]
            starts of the stencil objects

        kind_map ->  cz:
            domain information

    .. math:: 
        \begin{align*}
            \int \hat{n}^0 \ln \hat{n}^0 \sqrt{g} \mathrm{d}{\boldsymbol \eta}.
        \end{align*}
    '''

    res[:] = 0.0

    # allocate metric coeffs
    df = empty((3, 3), dtype=float)

    #$ omp parallel private (iel1, iel2, iel3, q1, q2, q3, eta1, eta2, eta3, wvol, vv, df, det_df)
    #$ omp for reduction( + : res)

    for iel1 in range(nel1):
        for iel2 in range(nel2):
            for iel3 in range(nel3):

                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):

                            eta1 = pts1[iel1, q1]
                            eta2 = pts2[iel2, q2]
                            eta3 = pts3[iel3, q3]

                            wvol = w1[iel1, q1] * w2[iel2, q2] * w3[iel3, q3]

                            vv   = density[pads1 + iel1, pads2 + iel2, pads3 + iel3, q1, q2, q3]
                                
                            if abs(vv) < 0.00001:
                                vv = 1.0 

                            # evaluate Jacobian, result in df
                            map_eval.df(eta1, eta2, eta3, kind_map, params_map, t1_map, t2_map, t3_map, p_map, ind1_map, ind2_map, ind3_map, cx, cy, cz, df)

                            det_df = linalg.det(df)

                            res[0] += vv * det_df * log(vv) * wvol

    #$ omp end parallel
