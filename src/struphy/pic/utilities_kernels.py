from pyccel.decorators import stack_array

import struphy.bsplines.bsplines_kernels as bsplines_kernels
import struphy.bsplines.evaluation_kernels_3d as evaluation_kernels_3d
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.geometry.evaluation_kernels as evaluation_kernels

from numpy import empty, shape, zeros, sqrt, log, abs


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
        span1 = bsplines_kernels.find_span(tn1, pn[0], eta1)
        span2 = bsplines_kernels.find_span(tn2, pn[1], eta2)
        span3 = bsplines_kernels.find_span(tn3, pn[2], eta3)

        bsplines_kernels.b_splines_slim(tn1, pn[0], eta1, span1, bn1)
        bsplines_kernels.b_splines_slim(tn2, pn[1], eta2, span2, bn2)
        bsplines_kernels.b_splines_slim(tn3, pn[2], eta3, span3, bn3)

        # sum up result
        res = res + evaluation_kernels_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2],
                                                                 bn1, bn2, bn3, span1, span2, span3, coeffs, starts)

    #$ omp end parallel

    return res


@stack_array('bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def eval_1_form_at_particles(markers: 'float[:,:]',
                             pn: 'int[:]',
                             tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                             starts: 'int[:]',
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

    #$ omp parallel private(ip, eta1, eta2, eta3, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3) shared(res)
    #$ omp for reduction (+:res)
    for ip in range(n_markers):

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], eta1)
        span2 = bsplines_kernels.find_span(tn2, pn[1], eta2)
        span3 = bsplines_kernels.find_span(tn3, pn[2], eta3)

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        # sum up result
        res[0] = res[0] + evaluation_kernels_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2],
                                                                       bd1, bn2, bn3, span1, span2, span3, coeffs1, starts)
        res[1] = res[1] + evaluation_kernels_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2],
                                                                       bn1, bd2, bn3, span1, span2, span3, coeffs2, starts)
        res[2] = res[2] + evaluation_kernels_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2],
                                                                       bn1, bn2, bd3, span1, span2, span3, coeffs3, starts)

    #$ omp end parallel


@stack_array('bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def eval_2_form_at_particles(markers: 'float[:,:]',
                             pn: 'int[:]',
                             tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                             starts: 'int[:]',
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
        span1 = bsplines_kernels.find_span(tn1, pn[0], eta1)
        span2 = bsplines_kernels.find_span(tn2, pn[1], eta2)
        span3 = bsplines_kernels.find_span(tn3, pn[2], eta3)

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        # sum up result
        res[0] = res[0] + evaluation_kernels_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2],
                                                                       bn1, bd2, bd3, span1, span2, span3, coeffs1, starts)
        res[1] = res[1] + evaluation_kernels_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2],
                                                                       bd1, bn2, bd3, span1, span2, span3, coeffs2, starts)
        res[2] = res[2] + evaluation_kernels_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2],
                                                                       bd1, bd2, bn3, span1, span2, span3, coeffs3, starts)

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
        span1 = bsplines_kernels.find_span(tn1, pn[0], eta1)
        span2 = bsplines_kernels.find_span(tn2, pn[1], eta2)
        span3 = bsplines_kernels.find_span(tn3, pn[2], eta3)

        bsplines_kernels.d_splines_slim(tn1, pn[0], eta1, span1, bd1)
        bsplines_kernels.d_splines_slim(tn2, pn[1], eta2, span2, bd2)
        bsplines_kernels.d_splines_slim(tn3, pn[2], eta3, span3, bd3)

        # sum up result
        res = res + evaluation_kernels_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2],
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
        span1 = bsplines_kernels.find_span(tn1, pn[0], eta1)
        span2 = bsplines_kernels.find_span(tn2, pn[1], eta2)
        span3 = bsplines_kernels.find_span(tn3, pn[2], eta3)

        bsplines_kernels.b_splines_slim(tn1, pn[0], eta1, span1, bn1)
        bsplines_kernels.b_splines_slim(tn2, pn[1], eta2, span2, bn2)
        bsplines_kernels.b_splines_slim(tn3, pn[2], eta3, span3, bn3)

        # sum up result
        res[0] = res[0] + evaluation_kernels_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2],
                                                                       bn1, bn2, bn3, span1, span2, span3, coeffs1, starts)
        res[1] = res[1] + evaluation_kernels_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2],
                                                                       bn1, bn2, bn3, span1, span2, span3, coeffs2, starts)
        res[2] = res[2] + evaluation_kernels_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2],
                                                                       bn1, bn2, bn3, span1, span2, span3, coeffs3, starts)

    #$ omp end parallel


@stack_array('bn1', 'bn2', 'bn3', 'b_cart', 'norm_b_cart', 'v', 'temp', 'v_perp')
def eval_magnetic_moment_6d(markers: 'float[:,:]',
                            pn: 'int[:]',
                            tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                            starts: 'int[:]',
                            b_cart_1: 'float[:,:,:]',
                            b_cart_2: 'float[:,:,:]',
                            b_cart_3: 'float[:,:,:]'):
    """
    Evaluate parallel velocity and magnetic moment of each particles and asign it into markers[ip,3] and markers[ip,4] respectively.

    Parameters
    ----------
        markers : array[float]
            .markers attribute of a struphy.pic.particles.Particles object

        kappa : array[float]
            omega_c/omega_unit

        pn : array[int]
            spline degrees

        tn1, tn2, tn3 : array[float]
            knot vectors

        starts : array[int]
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
        span1 = bsplines_kernels.find_span(tn1, pn[0], eta1)
        span2 = bsplines_kernels.find_span(tn2, pn[1], eta2)
        span3 = bsplines_kernels.find_span(tn3, pn[2], eta3)

        bsplines_kernels.b_splines_slim(tn1, pn[0], eta1, span1, bn1)
        bsplines_kernels.b_splines_slim(tn2, pn[1], eta2, span2, bn2)
        bsplines_kernels.b_splines_slim(tn3, pn[2], eta3, span3, bn3)

        # b_cart
        b_cart[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, b_cart_1, starts)
        b_cart[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, b_cart_2, starts)
        b_cart[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, b_cart_3, starts)

        # calculate absB
        absB = sqrt(b_cart[0]**2 + b_cart[1]**2 + b_cart[2]**2)

        if absB != 0.:
            norm_b_cart[:] = b_cart/absB
        else:
            norm_b_cart[:] = b_cart

        # calculate parallel velocity
        v_parallel = linalg_kernels.scalar_dot(norm_b_cart, v)

        # extract perpendicular velocity
        linalg_kernels.cross(v, norm_b_cart, temp)
        linalg_kernels.cross(norm_b_cart, temp, v_perp)

        v_perp_square = (v_perp[0]**2 + v_perp[1]**2 + v_perp[2]**2)

        # parallel velocity
        markers[ip, 3] = v_parallel
        # magnetic moment
        markers[ip, 4] = 1/2 * v_perp_square / absB
        # empty leftovers
        markers[ip, 5] = 0.


@stack_array('bn1', 'bn2', 'bn3')
def eval_magnetic_moment_5d(markers: 'float[:,:]',
                            pn: 'int[:]',
                            tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                            starts: 'int[:]',
                            absB: 'float[:,:,:]'):
    """
    Evaluate parallel velocity and magnetic moment of each particles and asign it into markers[ip,3] and markers[ip,4] respectively.

    Parameters
    ----------
        markers : array[float]
            .markers attribute of a struphy.pic.particles.Particles object

        kappa : array[float]
            omega_c/omega_unit

        pn : array[int]
            spline degrees

        tn1, tn2, tn3 : array[float]
            knot vectors

        starts : array[int]
            starts of the stencil objects (0-form)

        unit_b_cart_x : array[float]
            3d array of FE coeffs of the x component of unit cartesian equilibrium magnetic field 
    """
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

        v_perp = markers[ip, 4]

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], eta1)
        span2 = bsplines_kernels.find_span(tn2, pn[1], eta2)
        span3 = bsplines_kernels.find_span(tn3, pn[2], eta3)

        bsplines_kernels.b_splines_slim(tn1, pn[0], eta1, span1, bn1)
        bsplines_kernels.b_splines_slim(tn2, pn[1], eta2, span2, bn2)
        bsplines_kernels.b_splines_slim(tn3, pn[2], eta3, span3, bn3)

        B0 = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, absB, starts)

        # magnetic moment
        markers[ip, 4] = 1/2 * v_perp**2 / abs(B0)


@stack_array('bn1', 'bn2', 'bn3')
def eval_magnetic_energy(markers: 'float[:,:]',
                         pn: 'int[:]',
                         tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                         starts: 'int[:]',
                         PB: 'float[:,:,:]'):
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

        starts : array[int]
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
        span1 = bsplines_kernels.find_span(tn1, pn[0], eta1)
        span2 = bsplines_kernels.find_span(tn2, pn[1], eta2)
        span3 = bsplines_kernels.find_span(tn3, pn[2], eta3)

        bsplines_kernels.b_splines_slim(tn1, pn[0], eta1, span1, bn1)
        bsplines_kernels.b_splines_slim(tn2, pn[1], eta2, span2, bn2)
        bsplines_kernels.b_splines_slim(tn3, pn[2], eta3, span3, bn3)

        B0 = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, PB, starts)

        # if B0 < 0:
        #     print('minus', B0)

        markers[ip, 8] = B0*markers[ip, 4]


@stack_array('grad_PB', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'tmp')
def accum_gradI_const(markers: 'float[:,:]', n_markers_tot: 'int',
                      pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                      starts: 'int[:]',
                      grad_PB1: 'float[:,:,:]', grad_PB2: 'float[:,:,:]', grad_PB3: 'float[:,:,:]',
                      scale: 'float'):
    r"""TODO
    """
    # allocate for magnetic field evaluation
    grad_PB = empty(3, dtype=float)
    tmp = empty(3, dtype=float)

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
        eta1 = markers[ip, 0]  # mid
        eta2 = markers[ip, 1]  # mid
        eta3 = markers[ip, 2]  # mid

        # marker weight and velocity
        weight = markers[ip, 5]
        mu = markers[ip, 4]

        # b-field evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], eta1)
        span2 = bsplines_kernels.find_span(tn2, pn[1], eta2)
        span3 = bsplines_kernels.find_span(tn3, pn[2], eta3)

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        # grad_PB; 1form
        grad_PB[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_PB1, starts)
        grad_PB[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_PB2, starts)
        grad_PB[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_PB3, starts)

        tmp[:] = markers[ip, 15:18]
        res += linalg_kernels.scalar_dot(tmp, grad_PB) * weight * mu * scale

    return res/n_markers_tot


@stack_array('bn1', 'bn2', 'bn3')
def accum_en_fB(markers: 'float[:,:]', n_markers_tot: 'int',
                pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                starts: 'int[:]',
                PB: 'float[:,:,:]'):
    r"""TODO
    """
    # allocate for magnetic field evaluation
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # allocate for filling
    res = zeros(1, dtype=float)

    # get number of markers
    n_markers_loc = shape(markers)[0]

    for ip in range(n_markers_loc):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # marker weight and velocity
        mu = markers[ip, 4]
        weight = markers[ip, 5]

        # b-field evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], eta1)
        span2 = bsplines_kernels.find_span(tn2, pn[1], eta2)
        span3 = bsplines_kernels.find_span(tn3, pn[2], eta3)

        bsplines_kernels.b_splines_slim(tn1, pn[0], eta1, span1, bn1)
        bsplines_kernels.b_splines_slim(tn2, pn[1], eta2, span2, bn2)
        bsplines_kernels.b_splines_slim(tn3, pn[2], eta3, span3, bn3)

        B0 = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, PB, starts)

        res += abs(B0)*mu*weight

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

        markers[ip, 15:18] = e_diff[:]


@stack_array('e', 'e_diff')
def check_eta_diff2(markers: 'float[:,:]'):
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
        e_diff[:] = e[:] - markers[ip, 12:15]

        for axis in range(3):
            if e_diff[axis] > 0.5:
                e_diff[axis] -= 1.
            elif e_diff[axis] < -0.5:
                e_diff[axis] += 1.

        markers[ip, 15:18] = e_diff[:]


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
        markers[ip, 12:15] = e[:]

        e_diff[:] = e[:] - markers[ip, 9:12]
        e_mid[:] = (e[:] + markers[ip, 9:12])/2.

        for axis in range(3):
            if e_diff[axis] > 0.5:
                e_mid[axis] += 0.5
            elif e_diff[axis] < -0.5:
                e_mid[axis] += 0.5

        markers[ip, 0:3] = e_mid[:]


@stack_array('dfm', 'dfinv', 'dfinv_t', 'e', 'v', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'a_form', 'dfta_form')
def canonical_kinetic_particles(res: 'float[:]', markers: 'float[:,:]',
                                pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                                starts: 'int[:]',
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
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)

    # allocate for field evaluations (1-form components)
    a_form = empty(3, dtype=float)
    dfta_form = empty(3, dtype=float)
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

    #$ omp parallel private (ip, e, v, w, dfm, dfinv, dfinv_t, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, a_form, dfta_form)
    #$ omp for reduction( + : res)
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        v[:] = markers[ip, 3:6]
        w = markers[ip,   6]
        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(e[0], e[1], e[2],
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        linalg_kernels.matrix_inv(dfm, dfinv)
        linalg_kernels.transpose(dfinv, dfinv_t)

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], e[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], e[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], e[2])

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # magnetic field: 2-form components
        a_form[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, a1_1, starts)
        a_form[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, a1_2, starts)
        a_form[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, a1_3, starts)

        dfta_form[0] = dfinv_t[0, 0] * a_form[0] + \
            dfinv_t[0, 1] * a_form[1] + dfinv_t[0, 2] * a_form[2]
        dfta_form[1] = dfinv_t[1, 0] * a_form[0] + \
            dfinv_t[1, 1] * a_form[1] + dfinv_t[1, 2] * a_form[2]
        dfta_form[2] = dfinv_t[2, 0] * a_form[0] + \
            dfinv_t[2, 1] * a_form[1] + dfinv_t[2, 2] * a_form[2]

        res[0] += 0.5 * w * ((v[0] - dfta_form[0]) ** 2.0 +
                             (v[1] - dfta_form[1]) ** 2.0 + (v[2] - dfta_form[2]) ** 2.0)

    #$ omp end parallel


@stack_array('det_df', 'dfm')
def thermal_energy(res: 'float[:]', density: 'float[:,:,:,:,:,:]',
                   pads1: int, pads2: int, pads3: int,
                   nel1: 'int', nel2: 'int', nel3: 'int',
                   nq1: int, nq2: int, nq3: int,
                   w1: 'float[:,:]', w2: 'float[:,:]', w3: 'float[:,:]',
                   pts1: 'float[:,:]', pts2: 'float[:,:]', pts3: 'float[:,:]',
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
    dfm = empty((3, 3), dtype=float)

    #$ omp parallel private (iel1, iel2, iel3, q1, q2, q3, eta1, eta2, eta3, wvol, vv, dfm, det_df)
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

                            vv = density[pads1 + iel1, pads2 +
                                         iel2, pads3 + iel3, q1, q2, q3]

                            if abs(vv) < 0.00001:
                                vv = 1.0

                            # evaluate Jacobian, result in dfm
                            evaluation_kernels.df(eta1, eta2, eta3, kind_map, params_map, t1_map, t2_map,
                                                  t3_map, p_map, ind1_map, ind2_map, ind3_map, cx, cy, cz, dfm)

                            det_df = linalg_kernels.det(dfm)

                            res[0] += vv * det_df * log(vv) * wvol

    #$ omp end parallel
