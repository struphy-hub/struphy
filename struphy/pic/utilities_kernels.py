from pyccel.decorators import stack_array

import struphy.feec.bsplines_kernels as bsp
from struphy.feec.basics.spline_evaluation_3d import eval_spline_mpi_kernel

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
