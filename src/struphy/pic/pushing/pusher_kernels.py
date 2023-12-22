'Pusher kernels for full orbit (6D) particles.'


from pyccel.decorators import stack_array

import struphy.linear_algebra.linalg_kernels as linalg_kernels 
import struphy.geometry.evaluation_kernels as evaluation_kernels 
import struphy.bsplines.bsplines_kernels as bsplines_kernels 
import struphy.bsplines.evaluation_kernels_3d as evaluation_kernels_3d 
import struphy.pic.pushing.pusher_utilities_kernels as pusher_utilities_kernels 

from numpy import zeros, empty, shape, sqrt, cos, sin, floor, log


def a_documentation():
    r'''
    Explainer for arguments of pusher kernels.

    Function naming conventions:

    * starts with ``push_``
    * add a short description of the pusher, e.g. ``push_bxu_H1vec``.

    These kernels are passed to :class:`struphy.pic.pushing.pusher.Pusher` and called via::

        Pusher()

    The arguments passed to each kernel have a pre-defined order, defined in :class:`struphy.pic.pushing.pusher.Pusher`.
    This order is as follows (you can copy and paste from existing pusher_kernels functions):

    1. Marker info:
        * ``markers: 'float[:,:]'``          # local marker array

    2. Step info:
        * ``dt: 'float'``                    # time step
        * ``stage: 'int'``                   # current stage of the pusher (e.g. 0,1,2,3 for RK4)

    3. Derham spline bases info:
        * ``pn: 'int[:]'``                   # N-spline degree in each direction
        * ``tn1: 'float[:]'``                # N-spline knot vector 
        * ``tn2: 'float[:]'``
        * ``tn3: 'float[:]'``    

    4. mpi.comm start indices of FE coeffs on current process:
        - ``starts: 'int[:]'``               # start indices of current process 

    5. Mapping info:
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

    6. Optional: additional parameters, for example
        - ``b2_1: 'float[:,:,:]'``           # spline coefficients of b2_1
        - ``b2_2: 'float[:,:,:]'``           # spline coefficients of b2_2
        - ``b2_3: 'float[:,:,:]'``            # spline coefficients of b2_3
        - ``f0_params: 'float[:]'``          # parameters of equilibrium background
    '''

    print('This is just the docstring function.')


@stack_array('dfm', 'dfinv', 'dfinvt', 'e_form', 'e_cart', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def push_v_with_efield(markers: 'float[:,:]', dt: float, stage: int,
                       pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', 
                       starts: 'int[:]',
                       kind_map: int, params_map: 'float[:]',
                       p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                       ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                       cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                       e1_1: 'float[:,:,:]', e1_2: 'float[:,:,:]', e1_3: 'float[:,:,:]',
                       kappa: 'float'):
    r'''Updates

    .. math::

        \frac{\mathbf v^{n+1}_p - \mathbf v^n_p}{\Delta t} = \kappa DF^{-\top} \hat{\mathbf E}^1(\eta^n_p)

    for each marker :math:`p` in markers array, where :math:`\hat{\mathbf E}^1 in H(\textnormal{curl})`.

    Parameters
    ----------
        e1_1, e1_2, e1_3: array[float]
            3d array of FE coeffs of E-field as 1-form.
    '''

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinvt = empty((3, 3), dtype=float)

    # allocate for field evaluations (1-form and Cartesian components)
    e_form = empty(3, dtype=float)
    e_cart = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private(ip, eta1, eta2, eta3, dfm, dfinv, dfinvt, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, e_form, e_cart)
    #$ omp for
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

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
        linalg_kernels.matrix_inv_with_det(dfm, det_df, dfinv)
        linalg_kernels.transpose(dfinv, dfinvt)

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], eta1)
        span2 = bsplines_kernels.find_span(tn2, pn[1], eta2)
        span3 = bsplines_kernels.find_span(tn3, pn[2], eta3)

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        # electric field: 1-form components
        e_form[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, e1_1, starts)
        e_form[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, e1_2, starts)
        e_form[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, e1_3, starts)

        # electric field: Cartesian components
        linalg_kernels.matrix_vector(dfinvt, e_form, e_cart)

        # update velocities
        markers[ip, 3:6] += dt * kappa * e_cart

    #$ omp end parallel


@stack_array('dfm', 'b_form', 'b_cart', 'b_norm', 'e', 'v', 'vperp', 'vxb_norm', 'b_normxvperp', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def push_vxb_analytic(markers: 'float[:,:]', dt: float, stage: int,
                      pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                      starts: 'int[:]',
                      kind_map: int, params_map: 'float[:]',
                      p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                      ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                      cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                      b2_1: 'float[:,:,:]', b2_2: 'float[:,:,:]', b2_3: 'float[:,:,:]'):
    r'''Solves exactly the rotation

    .. math::

        \frac{\textnormal d \mathbf v_p(t)}{\textnormal d t} =  \mathbf v_p(t) \times \frac{DF\, \hat{\mathbf B}^2}{\sqrt g}

    for each marker :math:`p` in markers array, with fixed rotation vector.

    Parameters
    ----------
        b2_1, b2_2, b2_3: array[float]
            3d array of FE coeffs of B-field as 2-form.
    '''

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # allocate for field evaluations (2-form components, Cartesian components and normalized Cartesian components)
    b_form = empty(3, dtype=float)
    b_cart = empty(3, dtype=float)
    b_norm = empty(3, dtype=float)

    # particle position and velocity
    e = empty(3, dtype=float)
    v = empty(3, dtype=float)

    # perpendicular velocity, v x b_norm and b_norm x vperp
    vperp = empty(3, dtype=float)
    vxb_norm = empty(3, dtype=float)
    b_normxvperp = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private (ip, e, v, dfm, det_df, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, b_form, b_cart, b_abs, b_norm, vpar, vxb_norm, vperp, b_normxvperp)
    #$ omp for
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        v[:] = markers[ip, 3:6]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(e[0], e[1], e[2],
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], e[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], e[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], e[2])

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], e[0], int(span1), bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], e[1], int(span2), bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], e[2], int(span3), bn3, bd3)

        # magnetic field: 2-form components
        b_form[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b2_1, starts)
        b_form[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2_2, starts)
        b_form[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b2_3, starts)

        # magnetic field: Cartesian components
        linalg_kernels.matrix_vector(dfm, b_form, b_cart)
        b_cart[:] = b_cart/det_df

        # magnetic field: magnitude
        b_abs = sqrt(b_cart[0]**2 + b_cart[1]**2 + b_cart[2]**2)

        # only push vxb if magnetic field is non-zero
        if b_abs != 0.:
            # normalized magnetic field direction
            b_norm[:] = b_cart/b_abs

            # parallel velocity v.b_norm
            vpar = linalg_kernels.scalar_dot(v, b_norm)

            # first component of perpendicular velocity
            linalg_kernels.cross(v, b_norm, vxb_norm)
            linalg_kernels.cross(b_norm, vxb_norm, vperp)

            # second component of perpendicular velocity
            linalg_kernels.cross(b_norm, vperp, b_normxvperp)

            # analytic rotation
            markers[ip, 3:6] = vpar*b_norm + \
                cos(b_abs*dt)*vperp - sin(b_abs*dt)*b_normxvperp

    #$ omp end parallel


@stack_array('dfm', 'b_form', 'b_cart', 'b_prod', 'e', 'v', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'identity', 'rhs', 'lhs', 'lhs_inv', 'vec', 'res')
def push_vxb_implicit(markers: 'float[:,:]', dt: float, stage: int,
                      pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                      starts: 'int[:]',
                      kind_map: int, params_map: 'float[:]',
                      p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                      ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                      cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                      b2_1: 'float[:,:,:]', b2_2: 'float[:,:,:]', b2_3: 'float[:,:,:]'):
    r'''Solves the rotation

    .. math::

        \frac{\textnormal d \mathbf v_p(t)}{\textnormal d t} =  \mathbf v_p(t) \times \frac{DF\, \hat{\mathbf B}^2}{\sqrt g}

    with the Crank-Nicolson method for each marker :math:`p` in markers array, with fixed rotation vector.

    Parameters
    ----------
        b2_1, b2_2, b2_3: array[float]
            3d array of FE coeffs of B-field as 2-form.
    '''

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # allocate for field evaluations (2-form components, Cartesian components and rotation matrix such that vxB = B_prod.v)
    b_form = empty(3, dtype=float)
    b_cart = empty(3, dtype=float)
    b_prod = zeros((3, 3), dtype=float)

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

    # identity matrix
    identity = zeros((3, 3), dtype=float)

    identity[0, 0] = 1.
    identity[1, 1] = 1.
    identity[2, 2] = 1.

    # right-hand side and left-hand side of Crank-Nicolson scheme
    rhs = empty((3, 3), dtype=float)
    lhs = empty((3, 3), dtype=float)

    lhs_inv = empty((3, 3), dtype=float)

    vec = empty(3, dtype=float)
    res = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel firstprivate(b_prod) private (ip, e, v, dfm, det_df, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, b_form, b_cart, rhs, lhs, lhs_inv, vec, res)
    #$ omp for
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        v[:] = markers[ip, 3:6]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(e[0], e[1], e[2],
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], e[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], e[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], e[2])

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # magnetic field: 2-form components
        b_form[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b2_1, starts)
        b_form[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2_2, starts)
        b_form[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b2_3, starts)

        # magnetic field: Cartesian components
        linalg_kernels.matrix_vector(dfm, b_form, b_cart)
        b_cart[:] = b_cart/det_df

        # magnetic field: rotation matrix
        b_prod[0, 1] = b_cart[2]
        b_prod[0, 2] = -b_cart[1]

        b_prod[1, 0] = -b_cart[2]
        b_prod[1, 2] = b_cart[0]

        b_prod[2, 0] = b_cart[1]
        b_prod[2, 1] = -b_cart[0]

        # solve 3x3 system
        rhs[:, :] = identity + dt/2*b_prod
        lhs[:, :] = identity - dt/2*b_prod

        linalg_kernels.matrix_inv(lhs, lhs_inv)

        linalg_kernels.matrix_vector(rhs, v, vec)
        linalg_kernels.matrix_vector(lhs_inv, vec, res)

        markers[ip, 3:6] = res

    #$ omp end parallel


@stack_array('dfm', 'dfinv', 'dfinv_t', 'rot_temp', 'b_form', 'b_cart', 'b_norm', 'e', 'v', 'vperp', 'vxb_norm', 'b_normxvperp', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def push_pxb_analytic(markers: 'float[:,:]', dt: float, stage: int,
                      pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                      starts: 'int[:]',
                      kind_map: int, params_map: 'float[:]',
                      p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                      ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                      cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                      b2_1: 'float[:,:,:]', b2_2: 'float[:,:,:]', b2_3: 'float[:,:,:]',
                      a1_1: 'float[:,:,:]', a1_2: 'float[:,:,:]', a1_3: 'float[:,:,:]'):
    r'''Solves exactly the rotation

    .. math::

        \frac{\textnormal d \mathbf v_p(t)}{\textnormal d t} =  \mathbf v_p(t) \times \frac{DF\, \hat{\mathbf B}^2}{\sqrt g}

    for each marker :math:`p` in markers array, with fixed rotation vector.

    Parameters
    ----------
        b2_1, b2_2, b2_3: array[float]
            3d array of FE coeffs of B-field as 2-form.
    '''

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)

    rot_temp = empty(3, dtype=float)

    # allocate for field evaluations (2-form components, Cartesian components and normalized Cartesian components)
    b_form = empty(3, dtype=float)
    b_cart = empty(3, dtype=float)
    b_norm = empty(3, dtype=float)

    a_form = empty(3, dtype=float)

    # particle position and velocity
    e = empty(3, dtype=float)
    v = empty(3, dtype=float)

    # perpendicular velocity, v x b_norm and b_norm x vperp
    vperp = empty(3, dtype=float)
    vxb_norm = empty(3, dtype=float)
    b_normxvperp = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private (ip, e, v, dfm, dfinv, dfinv_t, det_df, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, b_form, a_form, b_cart, b_abs, b_norm, vpar, vxb_norm, vperp, b_normxvperp)
    #$ omp for
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        v[:] = markers[ip, 3:6]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(e[0], e[1], e[2],
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        linalg_kernels.matrix_inv(dfm, dfinv)
        linalg_kernels.transpose(dfinv, dfinv_t)
        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], e[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], e[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], e[2])

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # magnetic field: 2-form components
        b_form[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b2_1, starts)
        b_form[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2_2, starts)
        b_form[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b2_3, starts)

        # magnetic field: 2-form components
        a_form[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, a1_1, starts)
        a_form[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, a1_2, starts)
        a_form[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, a1_3, starts)

        rot_temp[0] = dfinv_t[0, 0] * a_form[0] + \
            dfinv_t[0, 1] * a_form[1] + dfinv_t[0, 2] * a_form[2]
        rot_temp[1] = dfinv_t[1, 0] * a_form[0] + \
            dfinv_t[1, 1] * a_form[1] + dfinv_t[1, 2] * a_form[2]
        rot_temp[2] = dfinv_t[2, 0] * a_form[0] + \
            dfinv_t[2, 1] * a_form[1] + dfinv_t[2, 2] * a_form[2]

        v[0] = v[0] - rot_temp[0]
        v[1] = v[1] - rot_temp[1]
        v[2] = v[2] - rot_temp[2]

        # magnetic field: Cartesian components
        linalg_kernels.matrix_vector(dfm, b_form, b_cart)
        b_cart[:] = b_cart/det_df

        # normalized magnetic field direction
        b_abs = sqrt(b_cart[0]**2 + b_cart[1]**2 + b_cart[2]**2)

        if b_abs != 0.:
            b_norm[:] = b_cart/b_abs
        else:
            b_norm[:] = b_cart

        # parallel velocity v.b_norm
        vpar = linalg_kernels.scalar_dot(v, b_norm)

        # first component of perpendicular velocity
        linalg_kernels.cross(v, b_norm, vxb_norm)
        linalg_kernels.cross(b_norm, vxb_norm, vperp)

        # second component of perpendicular velocity
        linalg_kernels.cross(b_norm, vperp, b_normxvperp)

        # analytic rotation
        markers[ip, 3:6] = vpar*b_norm + \
            cos(b_abs*dt)*vperp - sin(b_abs*dt)*b_normxvperp + rot_temp

    #$ omp end parallel


@stack_array('dfm', 'dfinv', 'dfinv_t', 'eta1', 'eta2', 'eta3')
def push_hybrid_xp_lnn(markers: 'float[:,:]', dt: float, stage: int,
                       pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                       starts: 'int[:]',
                       kind_map: int, params_map: 'float[:]',
                       p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                       ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                       cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                       p_shape: 'int[:]', p_size: 'float[:]', Nel: 'int[:]',
                       pts1: 'float[:]', pts2: 'float[:]', pts3: 'float[:]',
                       wts1: 'float[:]', wts2: 'float[:]', wts3: 'float[:]',
                       weight: 'float[:,:,:,:,:,:]', thermal: 'float', n_quad: 'int[:]'):
    r'''Solves exactly the rotation

    .. math::

        \frac{\textnormal d \mathbf v_p(t)}{\textnormal d t} =  \mathbf v_p(t) \times \frac{DF\, \hat{\mathbf B}^2}{\sqrt g}

    for each marker :math:`p` in markers array, with fixed rotation vector.
    '''

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)

    compact = zeros(3, dtype=float)
    compact[0] = (p_shape[0]+1.0)*p_size[0]
    compact[1] = (p_shape[1]+1.0)*p_size[1]
    compact[2] = (p_shape[2]+1.0)*p_size[2]

    cell_left = empty(3, dtype=int)
    point_left = zeros(3, dtype=float)
    point_right = zeros(3, dtype=float)
    cell_number = empty(3, dtype=int)

    grids_shapex = zeros(p_shape[0] + 2, dtype=float)
    grids_shapey = zeros(p_shape[1] + 2, dtype=float)
    grids_shapez = zeros(p_shape[2] + 2, dtype=float)

    temp1 = empty(3, dtype=float)
    temp4 = empty(3, dtype=float)
    temp6 = empty(3, dtype=float)
    temp8 = empty(3, dtype=float)
    ww = empty(1, dtype=float)

    value = empty(3, dtype=float)
    valuexyz = empty(3, dtype=float)
    dvaluexyz = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private (ip, eta1, eta2, eta3, dfm, dfinv, dfinv_t, det_df, point_left, point_right, cell_left, cell_number, i, grids_shapex, grids_shapey, grids_shapez, x_ii, y_ii, z_ii, il1, il2, il3, q1, q2, q3, temp1, temp4, temp6, valuexyz, dvaluexyz, temp8, ww)
    #$ omp for
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

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

        linalg_kernels.matrix_inv(dfm, dfinv)
        linalg_kernels.transpose(dfinv, dfinv_t)
        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        point_left[0] = eta1 - 0.5*compact[0]
        point_right[0] = eta1 + 0.5*compact[0]
        point_left[1] = eta2 - 0.5*compact[1]
        point_right[1] = eta2 + 0.5*compact[1]
        point_left[2] = eta3 - 0.5*compact[2]
        point_right[2] = eta3 + 0.5*compact[2]

        cell_left[0] = int(floor(point_left[0]*Nel[0]))
        cell_left[1] = int(floor(point_left[1]*Nel[1]))
        cell_left[2] = int(floor(point_left[2]*Nel[2]))

        cell_number[0] = int(floor(point_right[0]*Nel[0])) - cell_left[0] + 1.0
        cell_number[1] = int(floor(point_right[1]*Nel[1])) - cell_left[1] + 1.0
        cell_number[2] = int(floor(point_right[2]*Nel[2])) - cell_left[2] + 1.0

        for i in range(p_shape[0] + 1):
            grids_shapex[i] = point_left[0] + i * p_size[0]
        grids_shapex[p_shape[0] + 1] = point_right[0]

        for i in range(p_shape[1] + 1):
            grids_shapey[i] = point_left[1] + i * p_size[1]
        grids_shapey[p_shape[1] + 1] = point_right[1]

        for i in range(p_shape[2] + 1):
            grids_shapez[i] = point_left[2] + i * p_size[2]
        grids_shapez[p_shape[2] + 1] = point_right[2]

        # if periodic
        x_ii = pn[0] + cell_left[0] - starts[0]
        y_ii = pn[1] + cell_left[1] - starts[1]
        z_ii = pn[2] + cell_left[2] - starts[2]

        # ======================================
        for il1 in range(cell_number[0]):
            for il2 in range(cell_number[1]):
                for il3 in range(cell_number[2]):
                    for q1 in range(n_quad[0]):
                        for q2 in range(n_quad[1]):
                            for q3 in range(n_quad[2]):
                                # quadrature points in the cell x direction
                                temp1[0] = (cell_left[0] + il1) / \
                                    Nel[0] + pts1[q1]
                                # if > 0, result is 0
                                temp4[0] = abs(temp1[0] - eta1) - compact[0]/2

                                temp1[1] = (cell_left[1] + il2) / \
                                    Nel[1] + pts2[q2]
                                # if > 0, result is 0
                                temp4[1] = abs(temp1[1] - eta2) - compact[1]/2

                                temp1[2] = (cell_left[2] + il3) / \
                                    Nel[2] + pts3[q3]
                                # if > 0, result is 0
                                temp4[2] = abs(temp1[2] - eta3) - compact[2]/2

                                if temp4[0] < 0 and temp4[1] < 0 and temp4[2] < 0:

                                    valuexyz[0] = bsplines_kernels.convolution(
                                        p_shape[0], grids_shapex, temp1[0])
                                    dvaluexyz[0] = bsplines_kernels.convolution_der(
                                        p_shape[0], grids_shapex, temp1[0])

                                    valuexyz[1] = bsplines_kernels.piecewise(
                                        p_shape[1], p_size[1], temp1[1] - eta2)
                                    dvaluexyz[1] = bsplines_kernels.piecewise(
                                        p_shape[2], p_size[2], temp1[2] - eta3)

                                    valuexyz[2] = bsplines_kernels.piecewise_der(
                                        p_shape[1], p_size[1], temp1[1] - eta2)
                                    dvaluexyz[2] = bsplines_kernels.piecewise_der(
                                        p_shape[2], p_size[2], temp1[2] - eta3)

                                    temp8[0] = dvaluexyz[0] * \
                                        valuexyz[1] * valuexyz[2]
                                    temp8[1] = valuexyz[0] * \
                                        dvaluexyz[1] * valuexyz[2]
                                    temp8[2] = valuexyz[0] * \
                                        valuexyz[1] * dvaluexyz[2]

                                    ww[0] = weight[x_ii + il1, y_ii + il2, z_ii + il3,
                                                   q1, q2, q3] * wts1[q1] * wts2[q2] * wts3[q3]

                                    temp6[0] = dfinv_t[0, 0]*temp8[0] + \
                                        dfinv_t[0, 1]*temp8[1] + \
                                        dfinv_t[0, 2]*temp8[2]
                                    temp6[1] = dfinv_t[1, 0]*temp8[0] + \
                                        dfinv_t[1, 1]*temp8[1] + \
                                        dfinv_t[1, 2]*temp8[2]
                                    temp6[2] = dfinv_t[2, 0]*temp8[0] + \
                                        dfinv_t[2, 1]*temp8[1] + \
                                        dfinv_t[2, 2]*temp8[2]
                                    # check weight_123 index
                                    markers[ip, 3] += dt * \
                                        ww[0] * thermal * temp6[0]
                                    markers[ip, 4] += dt * \
                                        ww[0] * thermal * temp6[1]
                                    markers[ip, 5] += dt * \
                                        ww[0] * thermal * temp6[2]

    #$ omp end parallel


@stack_array('dfm', 'dfinv', 'dfinv_t', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def push_hybrid_xp_ap(markers: 'float[:,:]', dt: float, stage: int,
                      pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                      starts: 'int[:]',
                      kind_map: int, params_map: 'float[:]',
                      p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                      ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                      cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                      a1_1: 'float[:,:,:]', a1_2: 'float[:,:,:]', a1_3: 'float[:,:,:]'):
    r'''Solves exactly the rotation

    .. math::

        \frac{\textnormal d \mathbf v_p(t)}{\textnormal d t} =  \mathbf v_p(t) \times \frac{DF\, \hat{\mathbf B}^2}{\sqrt g}

    for each marker :math:`p` in markers array, with fixed rotation vector.
    '''

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)

    a_form = empty(3, dtype=float)
    a_xx = empty((3, 3), dtype=float)
    a_xxtrans = empty((3, 3), dtype=float)

    matrixp = empty((3, 3), dtype=float)
    matrixpp = empty((3, 3), dtype=float)
    matrixppp = empty((3, 3), dtype=float)

    lhs = empty((3, 3), dtype=float)
    lhsinv = empty((3, 3), dtype=float)
    rhs = empty(3, dtype=float)

    # particle position and velocity
    e = empty(3, dtype=float)
    v = empty(3, dtype=float)

    # p + 1 non-vanishing basis functions up tp degree p
    b1 = zeros((pn[0] + 1, pn[0] + 1), dtype=float)
    b2 = zeros((pn[1] + 1, pn[1] + 1), dtype=float)
    b3 = zeros((pn[2] + 1, pn[2] + 1), dtype=float)

    l1 = zeros(pn[0], dtype=float)
    l2 = zeros(pn[1], dtype=float)
    l3 = zeros(pn[2], dtype=float)

    r1 = zeros(pn[0], dtype=float)
    r2 = zeros(pn[1], dtype=float)
    r3 = zeros(pn[2], dtype=float)

    # scaling arrays for M-splines
    d1 = zeros(pn[0], dtype=float)
    d2 = zeros(pn[1], dtype=float)
    d3 = zeros(pn[2], dtype=float)

    # non-vanishing N-splines
    bn1 = zeros(pn[0] + 1, dtype=float)
    bn2 = zeros(pn[1] + 1, dtype=float)
    bn3 = zeros(pn[2] + 1, dtype=float)

    # non-vanishing D-splines
    bd1 = zeros(pn[0], dtype=float)
    bd2 = zeros(pn[1], dtype=float)
    bd3 = zeros(pn[2], dtype=float)

    pd1 = pn[0] - 1
    pd2 = pn[1] - 1
    pd3 = pn[2] - 1

    bdd1 = zeros(pd1, dtype=float)
    bdd2 = zeros(pd2, dtype=float)
    bdd3 = zeros(pd3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private (ip, e, v, dfm, dfinv, dfinv_t, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, bdd1, bdd2, bdd3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, a_form, a_xx, a_xxtrans, matrixp, matrixpp, matrixppp, lhs, rhs, lhsinv)
    #$ omp for
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        v[:] = markers[ip, 3:6]

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

        bsplines_kernels.basis_funs_all(tn1, pn[0], e[0], span1, l1, r1, b1, d1)
        bsplines_kernels.basis_funs_all(tn2, pn[1], e[1], span2, l2, r2, b2, d2)
        bsplines_kernels.basis_funs_all(tn3, pn[2], e[2], span3, l3, r3, b3, d3)

        # N-splines and D-splines
        bn1[:] = b1[pn[0], :]
        bn2[:] = b2[pn[1], :]
        bn3[:] = b3[pn[2], :]

        for il1 in range(pn[0]):
            bd1[il1] = b1[pd1, il1] * d1[il1]

        for il2 in range(pn[1]):
            bd2[il2] = b2[pd2, il2] * d2[il2]

        for il3 in range(pn[2]):
            bd3[il3] = b3[pd3, il3] * d3[il3]

        for il1 in range(pd1):
            bdd1[il1] = b1[pd1 - 1, il1] * d1[il1] * d1[il1]

        for il2 in range(pd2):
            bdd2[il2] = b2[pd2 - 1, il2] * d2[il2] * d2[il2]

        for il3 in range(pd3):
            bdd3[il3] = b3[pd3 - 1, il3] * d3[il3] * d3[il3]

        # vector potential: 1-form components
        a_form[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, a1_1, starts)
        a_form[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, a1_2, starts)
        a_form[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, a1_3, starts)

        a_xx[0, 0] = evaluation_kernels_3d.eval_spline_derivative_mpi_kernel(
            pn[0] - 2, pn[1], pn[2], bdd1, bn2, bn3, span1, span2, span3, a1_1, starts, int(1))
        a_xx[0, 1] = evaluation_kernels_3d.eval_spline_derivative_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, a1_1, starts, int(2))
        a_xx[0, 2] = evaluation_kernels_3d.eval_spline_derivative_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, a1_1, starts, int(3))

        a_xx[1, 0] = evaluation_kernels_3d.eval_spline_derivative_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, a1_2, starts, int(1))
        a_xx[1, 1] = evaluation_kernels_3d.eval_spline_derivative_mpi_kernel(
            pn[0], pn[1] - 2, pn[2], bn1, bdd2, bn3, span1, span2, span3, a1_2, starts, int(2))
        a_xx[1, 2] = evaluation_kernels_3d.eval_spline_derivative_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, a1_2, starts, int(3))

        a_xx[2, 0] = evaluation_kernels_3d.eval_spline_derivative_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, a1_3, starts, int(1))
        a_xx[2, 1] = evaluation_kernels_3d.eval_spline_derivative_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, a1_3, starts, int(2))
        a_xx[2, 2] = evaluation_kernels_3d.eval_spline_derivative_mpi_kernel(
            pn[0], pn[1], pn[2] - 2, bn1, bn2, bdd3, span1, span2, span3, a1_3, starts, int(3))

        linalg_kernels.transpose(a_xx, a_xxtrans)
        linalg_kernels.matrix_matrix(a_xxtrans, dfinv, matrixp)
        linalg_kernels.matrix_matrix(dfinv_t, matrixp, matrixpp)  # left matrix
        linalg_kernels.matrix_matrix(matrixpp, dfinv_t, matrixppp)  # right matrix

        lhs[0, 0] = 1.0 - dt*matrixpp[0, 0]
        lhs[0, 1] = - dt*matrixpp[0, 1]
        lhs[0, 2] = - dt*matrixpp[0, 2]

        lhs[1, 0] = - dt*matrixpp[0, 0]
        lhs[1, 1] = 1.0 - dt*matrixpp[1, 1]
        lhs[1, 2] = - dt*matrixpp[1, 2]

        lhs[2, 0] = - dt*matrixpp[2, 0]
        lhs[2, 1] = - dt*matrixpp[2, 1]
        lhs[2, 2] = 1.0 - dt*matrixpp[2, 2]

        linalg_kernels.matrix_vector(matrixppp, a_form, rhs)
        rhs[0] = v[0] - dt*rhs[0]
        rhs[1] = v[1] - dt*rhs[1]
        rhs[2] = v[2] - dt*rhs[2]

        linalg_kernels.matrix_inv(lhs, lhsinv)
        # update velocity
        markers[ip, 3] = lhsinv[0, 0]*rhs[0] + \
            lhsinv[0, 1]*rhs[1] + lhsinv[0, 2]*rhs[2]
        markers[ip, 4] = lhsinv[1, 0]*rhs[0] + \
            lhsinv[1, 1]*rhs[1] + lhsinv[1, 2]*rhs[2]
        markers[ip, 5] = lhsinv[2, 0]*rhs[0] + \
            lhsinv[2, 1]*rhs[1] + lhsinv[2, 2]*rhs[2]

        # update position
        linalg_kernels.matrix_vector(dfinv_t, a_form, rhs)
        rhs[0] = markers[ip, 3] - rhs[0]
        rhs[1] = markers[ip, 4] - rhs[1]
        rhs[2] = markers[ip, 5] - rhs[2]

        markers[ip, 0] = e[0] + dt * \
            (dfinv[0, 0]*rhs[0] + dfinv[0, 1]*rhs[1] + dfinv[0, 2]*rhs[2])
        markers[ip, 1] = e[1] + dt * \
            (dfinv[1, 0]*rhs[0] + dfinv[1, 1]*rhs[1] + dfinv[1, 2]*rhs[2])
        markers[ip, 2] = e[2] + dt * \
            (dfinv[2, 0]*rhs[0] + dfinv[2, 1]*rhs[1] + dfinv[2, 2]*rhs[2])

    #$ omp end parallel


@stack_array('dfm', 'b_form', 'u_form', 'b_cart', 'u_cart', 'e_cart', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def push_bxu_Hdiv(markers: 'float[:,:]', dt: float, stage: int,
                  pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                  starts: 'int[:]',
                  kind_map: int, params_map: 'float[:]',
                  p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                  ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                  cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                  b2_1: 'float[:,:,:]', b2_2: 'float[:,:,:]', b2_3: 'float[:,:,:]',
                  u2_1: 'float[:,:,:]', u2_2: 'float[:,:,:]', u2_3: 'float[:,:,:]'):
    r'''Updates

    .. math::

        \frac{\mathbf v^{n+1}_p - \mathbf v^n_p}{\Delta t} = DF^{-\top} \left(  \hat{\mathbf B}^2 \times \frac{\hat{\mathbf U}^2}{\sqrt g}  \right)^n_p

    for each marker :math:`p` in markers array, where :math:`\hat{\mathbf U}^2 \in H(\textnormal{div})`.

    Parameters
    ----------
        b2_1, b2_2, b2_3: array[float]
            3d array of FE coeffs of B-field as 2-form.

        u2_1, u2_2, u2_3: array[float]
            3d array of FE coeffs of U-field as 2-form.
    '''

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # allocate for field evaluations (2-form and Cartesian components)
    b_form = empty(3, dtype=float)
    u_form = empty(3, dtype=float)

    b_cart = empty(3, dtype=float)
    u_cart = empty(3, dtype=float)

    e_cart = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private(ip, eta1, eta2, eta3, dfm, det_df, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, b_form, b_cart, u_form, u_cart, e_cart)
    #$ omp for
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker data
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

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], eta1)
        span2 = bsplines_kernels.find_span(tn2, pn[1], eta2)
        span3 = bsplines_kernels.find_span(tn3, pn[2], eta3)

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        # magnetic field: 2-form components
        b_form[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b2_1, starts)
        b_form[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2_2, starts)
        b_form[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b2_3, starts)

        # magnetic field: Cartesian components
        linalg_kernels.matrix_vector(dfm, b_form, b_cart)
        b_cart[:] = b_cart/det_df

        # velocity field: 2-form components
        u_form[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, u2_1, starts)
        u_form[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, u2_2, starts)
        u_form[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, u2_3, starts)

        linalg_kernels.matrix_vector(dfm, u_form, u_cart)
        u_cart[:] = u_cart/det_df

        # electric field E = B x U
        linalg_kernels.cross(b_cart, u_cart, e_cart)

        # update velocities
        markers[ip, 3:6] += dt*e_cart

    #$ omp end parallel


@stack_array('dfm', 'dfinv', 'dfinv_t', 'b_form', 'u_form', 'b_cart', 'u_cart', 'e_cart', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def push_bxu_Hcurl(markers: 'float[:,:]', dt: float, stage: int,
                   pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                   starts: 'int[:]',
                   kind_map: int, params_map: 'float[:]',
                   p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                   ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                   cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                   b2_1: 'float[:,:,:]', b2_2: 'float[:,:,:]', b2_3: 'float[:,:,:]',
                   u1_1: 'float[:,:,:]', u1_2: 'float[:,:,:]', u1_3: 'float[:,:,:]'):
    r'''Updates

    .. math::

        \frac{\mathbf v^{n+1}_p - \mathbf v^n_p}{\Delta t} = DF^{-\top} \left(  \hat{\mathbf B}^2 \times G^{-1}\hat{\mathbf U}^1  \right)^n_p

    for each marker :math:`p` in markers array, where :math:`\hat{\mathbf U}^1 \in H(\textnormal{curl})`.

    Parameters
    ----------
        b2_1, b2_2, b2_3: array[float]
            3d array of FE coeffs of B-field as 2-form.

        u1_1, u1_2, u1_3: array[float]
            3d array of FE coeffs of U-field as 1-form.
    '''

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)

    # allocate for field evaluations (2-form and Cartesian components)
    b_form = empty(3, dtype=float)
    u_form = empty(3, dtype=float)

    b_cart = empty(3, dtype=float)
    u_cart = empty(3, dtype=float)

    e_cart = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private(ip, eta1, eta2, eta3, dfm, det_df, dfinv, dfinv_t, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, b_form, b_cart, u_form, u_cart, e_cart)
    #$ omp for
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker data
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
        linalg_kernels.matrix_inv_with_det(dfm, det_df, dfinv)
        linalg_kernels.transpose(dfinv, dfinv_t)

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], eta1)
        span2 = bsplines_kernels.find_span(tn2, pn[1], eta2)
        span3 = bsplines_kernels.find_span(tn3, pn[2], eta3)

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        # magnetic field: 2-form components
        b_form[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b2_1, starts)
        b_form[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2_2, starts)
        b_form[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b2_3, starts)

        # magnetic field: Cartesian components
        linalg_kernels.matrix_vector(dfm, b_form, b_cart)
        b_cart[:] = b_cart/det_df

        # velocity field: 1-form components
        u_form[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, u1_1, starts)
        u_form[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, u1_2, starts)
        u_form[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, u1_3, starts)

        # velocity field: Cartesian components
        linalg_kernels.matrix_vector(dfinv_t, u_form, u_cart)

        # electric field E = B x U
        linalg_kernels.cross(b_cart, u_cart, e_cart)

        # update velocities
        markers[ip, 3:6] += dt*e_cart

    #$ omp end parallel


@stack_array('dfm', 'b_form', 'u_form', 'b_cart', 'u_cart', 'e_cart', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def push_bxu_H1vec(markers: 'float[:,:]', dt: float, stage: int,
                   pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                   starts: 'int[:]',
                   kind_map: int, params_map: 'float[:]',
                   p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                   ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                   cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                   b2_1: 'float[:,:,:]', b2_2: 'float[:,:,:]', b2_3: 'float[:,:,:]',
                   uv_1: 'float[:,:,:]', uv_2: 'float[:,:,:]', uv_3: 'float[:,:,:]'):
    r'''Updates

    .. math::

        \frac{\mathbf v^{n+1}_p - \mathbf v^n_p}{\Delta t} = DF^{-\top} \left(  \hat{\mathbf B}^2 \times \hat{\mathbf U}  \right)^n_p

    for each marker :math:`p` in markers array, where :math:`\hat{\mathbf U}` is a vector-field (dual to 1-form) in :math:`(H^1)^3`.

    Parameters
    ----------
        b2_1, b2_2, b2_3: array[float]
            3d array of FE coeffs of B-field as 2-form.

        uv_1, uv_2, uv_3: array[float]
            3d array of FE coeffs of U-field as vector field in (H^1)^3.
    '''

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # allocate for field evaluations (2-form and Cartesian components)
    b_form = empty(3, dtype=float)
    u_form = empty(3, dtype=float)

    b_cart = empty(3, dtype=float)
    u_cart = empty(3, dtype=float)

    e_cart = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private(ip, eta1, eta2, eta3, dfm, det_df, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, b_form, b_cart, u_form, u_cart, e_cart)
    #$ omp for
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker data
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

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], eta1)
        span2 = bsplines_kernels.find_span(tn2, pn[1], eta2)
        span3 = bsplines_kernels.find_span(tn3, pn[2], eta3)

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        # magnetic field: 2-form components
        b_form[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b2_1, starts)
        b_form[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2_2, starts)
        b_form[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b2_3, starts)

        # magnetic field: Cartesian components
        linalg_kernels.matrix_vector(dfm, b_form, b_cart)
        b_cart[:] = b_cart/det_df

        # velocity field: vector field components
        u_form[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, uv_1, starts)
        u_form[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, uv_2, starts)
        u_form[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, uv_3, starts)

        # velocity field: Cartesian components
        linalg_kernels.matrix_vector(dfm, u_form, u_cart)

        # electric field E = B x U
        linalg_kernels.cross(b_cart, u_cart, e_cart)

        # update velocities
        markers[ip, 3:6] += dt*e_cart

    #$ omp end parallel


@stack_array('dfm', 'dfinv', 'dfinv_t', 'b_form', 'u_form', 'b_diff', 'b_cart', 'u_cart', 'b_grad', 'e_cart', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'der1', 'der2', 'der3')
def push_bxu_Hdiv_pauli(markers: 'float[:,:]', dt: float, stage: int,
                        pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                        starts: 'int[:]',
                        kind_map: int, params_map: 'float[:]',
                        p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                        ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                        cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                        b2_1: 'float[:,:,:]', b2_2: 'float[:,:,:]', b2_3: 'float[:,:,:]',
                        u2_1: 'float[:,:,:]', u2_2: 'float[:,:,:]', u2_3: 'float[:,:,:]',
                        b0: 'float[:,:,:]',
                        mu: 'float[:]'):
    r'''Updates

    .. math::

        \frac{\mathbf v^{n+1}_p - \mathbf v^n_p}{\Delta t} = DF^{-\top} \left(  \hat{\mathbf B}^2 \times \frac{\hat{\mathbf U}^2}{\sqrt g} - \mu\,\nabla \hat{|\mathbf B|}^0  \right)^n_p

    for each marker :math:`p` in markers array, where :math:`\hat{\mathbf U}^2 \in H(\textnormal{div})` and :math:`\hat{|\mathbf B|}^0 \in H^1`.

    Parameters
    ----------
        b2_1, b2_2, b2_3: array[float]
            3d array of FE coeffs of B-field as 2-form.

        u2_1, u2_2, u2_3: array[float]
            3d array of FE coeffs of U-field as 2-form.

        b0 : array[float]
            3d array of FE coeffs of abs(B) as 0-form.

        mu : array[float]
            1d array of size n_markers holding particle magnetic moments.
    '''

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)

    # allocate for field evaluations (2-form and Cartesian components)
    b_form = empty(3, dtype=float)
    u_form = empty(3, dtype=float)
    b_diff = empty(3, dtype=float)

    b_cart = empty(3, dtype=float)
    u_cart = empty(3, dtype=float)
    b_grad = empty(3, dtype=float)

    e_cart = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    der1 = empty(pn[0] + 1, dtype=float)
    der2 = empty(pn[1] + 1, dtype=float)
    der3 = empty(pn[2] + 1, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private(ip, eta1, eta2, eta3, dfm, det_df, dfinv, dfinv_t, span1, span2, span3, bn1, bn2, bn3, der1, der2, der3, bd1, bd2, bd3, b_form, b_cart, b_diff, b_grad, u_form, u_cart, e_cart)
    #$ omp for
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker data
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
        linalg_kernels.matrix_inv_with_det(dfm, det_df, dfinv)
        linalg_kernels.transpose(dfinv, dfinv_t)

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], eta1)
        span2 = bsplines_kernels.find_span(tn2, pn[1], eta2)
        span3 = bsplines_kernels.find_span(tn3, pn[2], eta3)

        bsplines_kernels.b_der_splines_slim(tn1, pn[0], eta1, span1, bn1, der1)
        bsplines_kernels.b_der_splines_slim(tn2, pn[1], eta2, span2, bn2, der2)
        bsplines_kernels.b_der_splines_slim(tn3, pn[2], eta3, span3, bn3, der3)

        bsplines_kernels.d_splines_slim(tn1, pn[0], eta1, span1, bd1)
        bsplines_kernels.d_splines_slim(tn2, pn[1], eta2, span2, bd2)
        bsplines_kernels.d_splines_slim(tn3, pn[2], eta3, span3, bd3)

        # magnetic field: 2-form components
        b_form[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b2_1, starts)
        b_form[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2_2, starts)
        b_form[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b2_3, starts)

        # magnetic field: Cartesian components
        linalg_kernels.matrix_vector(dfm, b_form, b_cart)
        b_cart[:] = b_cart/det_df

        # magnetic field: evaluation of gradient (1-form)
        b_diff[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], der1, bn2, bn3, span1, span2, span3, b0, starts)
        b_diff[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, der2, bn3, span1, span2, span3, b0, starts)
        b_diff[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, der3, span1, span2, span3, b0, starts)

        # magnetic field: evaluation of gradient (Cartesian components)
        linalg_kernels.matrix_vector(dfinv_t, b_diff, b_grad)

        # velocity field: 2-form components
        u_form[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, u2_1, starts)
        u_form[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, u2_2, starts)
        u_form[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, u2_3, starts)

        linalg_kernels.matrix_vector(dfm, u_form, u_cart)
        u_cart[:] = u_cart/det_df

        # electric field E = B x U
        linalg_kernels.cross(b_cart, u_cart, e_cart)

        # additional artificial electric field of Pauli markers
        e_cart[:] = e_cart - mu[ip]*b_grad

        # update velocities
        markers[ip, 3:6] += dt*e_cart

    #$ omp end parallel


def push_pc_GXu_full(markers: 'float[:,:]', dt: float, stage: int,
                     pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                     starts: 'int[:]',
                     kind_map: int, params_map: 'float[:]',
                     p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                     ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                     cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                     GXu_11: 'float[:,:,:]', GXu_12: 'float[:,:,:]', GXu_13: 'float[:,:,:]',
                     GXu_21: 'float[:,:,:]', GXu_22: 'float[:,:,:]', GXu_23: 'float[:,:,:]',
                     GXu_31: 'float[:,:,:]', GXu_32: 'float[:,:,:]', GXu_33: 'float[:,:,:]'):
    r'''Updates

    .. math::

        \frac{\mathbf v^{n+1}_p - \mathbf v^n_p}{\Delta t} = - DF^{-\top} \left(  \boldsymbol \Lambda^1 \mathbb G \mathcal X(\mathbf u, \mathbf v)  \right)^n_p

    for each marker :math:`p` in markers array, where :math:`\mathbf u` 
    are the coefficients of the mhd velocity field (either 1-form or 2-form) and :math:`\mathcal X`
    is either the MHD operator :meth:`struphy.feec.basis_projection_ops.MHDOperators.assemble_X1` (if u is 1-form)
    or :meth:`struphy.feec.basis_projection_ops.MHDOperators.assemble_X2` (if u is 2-form).

    Parameters
    ----------
        grad_Xu_ij: array[float]
            3d array of FE coeffs of :math:`\nabla_j(\mathcal X \cdot \mathbf u)_i`. i,j=1,2,3.
    '''

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)

    # allocate for field evaluations
    e = empty(3, dtype=float)
    e_cart = empty(3, dtype=float)
    GXu = empty((3, 3), dtype=float)

    # particle velocity
    v = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    n_markers = shape(markers)[0]

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        v[:] = markers[ip, 3:6]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta1, eta2, eta3,
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # metric coeffs
        linalg_kernels.matrix_inv(dfm, dfinv)
        linalg_kernels.transpose(dfinv, dfinv_t)

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], eta1)
        span2 = bsplines_kernels.find_span(tn2, pn[1], eta2)
        span3 = bsplines_kernels.find_span(tn3, pn[2], eta3)

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        # Evaluate grad(X(u, v)) at the particle positions
        GXu[0, 0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, GXu_11, starts)
        GXu[1, 0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, GXu_21, starts)
        GXu[2, 0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, GXu_31, starts)
        GXu[0, 1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, GXu_12, starts)
        GXu[1, 1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, GXu_22, starts)
        GXu[2, 1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, GXu_32, starts)
        GXu[0, 2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, GXu_13, starts)
        GXu[1, 2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, GXu_23, starts)
        GXu[2, 2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, GXu_33, starts)

        e[0] = GXu[0, 0] * v[0] + GXu[1, 0] * v[1] + GXu[2, 0] * v[2]
        e[1] = GXu[0, 1] * v[0] + GXu[1, 1] * v[1] + GXu[2, 1] * v[2]
        e[2] = GXu[0, 2] * v[0] + GXu[1, 2] * v[1] + GXu[2, 2] * v[2]

        linalg_kernels.matrix_vector(dfinv_t, e, e_cart)

        # update velocities
        markers[ip, 3:6] -= dt*e_cart/2.


def push_pc_GXu(markers: 'float[:,:]', dt: float, stage: int,
                pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                starts: 'int[:]',
                kind_map: int, params_map: 'float[:]',
                p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                GXu_11: 'float[:,:,:]', GXu_12: 'float[:,:,:]', GXu_13: 'float[:,:,:]',
                GXu_21: 'float[:,:,:]', GXu_22: 'float[:,:,:]', GXu_23: 'float[:,:,:]',
                GXu_31: 'float[:,:,:]', GXu_32: 'float[:,:,:]', GXu_33: 'float[:,:,:]'):
    r'''Updates

    .. math::

        \frac{\mathbf v^{n+1}_p - \mathbf v^n_p}{\Delta t} = - DF^{-\top} \left(  \boldsymbol \Lambda^1 \mathbb G \mathcal X(\mathbf u, \mathbf v)  \right)^n_p

    for each marker :math:`p` in markers array, where :math:`\mathbf u`
    are the coefficients of the mhd velocity field (either 1-form or 2-form) and :math:`\mathcal X`
    is either the MHD operator :meth:`struphy.feec.basis_projection_ops.MHDOperators.assemble_X1` (if u is 1-form)
    or :meth:`struphy.feec.basis_projection_ops.MHDOperators.assemble_X2` (if u is 2-form).

    Parameters
    ----------
        grad_Xu_ij: array[float]
            3d array of FE coeffs of :math:`\nabla_j(\mathcal X \cdot \mathbf u)_i`. i,j=1,2,3.
    '''

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)

    # allocate for field evaluations
    e = empty(3, dtype=float)
    e_cart = empty(3, dtype=float)
    GXu = empty((3, 3), dtype=float)
    GXu_t = empty((3, 3), dtype=float)

    # particle velocity
    v = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    n_markers = shape(markers)[0]

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        v[:] = markers[ip, 3:6]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta1, eta2, eta3,
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # metric coeffs
        linalg_kernels.matrix_inv(dfm, dfinv)
        linalg_kernels.transpose(dfinv, dfinv_t)

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], eta1)
        span2 = bsplines_kernels.find_span(tn2, pn[1], eta2)
        span3 = bsplines_kernels.find_span(tn3, pn[2], eta3)

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        # Evaluate grad(X(u, v)) at the particle positions
        GXu[0, 0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, GXu_11, starts)
        GXu[1, 0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, GXu_21, starts)
        GXu[0, 1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, GXu_12, starts)
        GXu[1, 1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, GXu_22, starts)
        GXu[0, 2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, GXu_13, starts)
        GXu[1, 2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, GXu_23, starts)

        e[0] = GXu[0, 0] * v[0] + GXu[1, 0] * v[1]
        e[1] = GXu[0, 1] * v[0] + GXu[1, 1] * v[1]
        e[2] = GXu[0, 2] * v[0] + GXu[1, 2] * v[1]

        linalg_kernels.matrix_vector(dfinv_t, e, e_cart)

        # update velocities
        markers[ip, 3:6] -= dt*e_cart/2.


@stack_array('dfm', 'dfinv', 'e', 'v', 'k')
def push_eta_stage(markers: 'float[:,:]', dt: float, stage: int,
                   pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                   starts: 'int[:]',
                   kind_map: int, params_map: 'float[:]',
                   p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                   ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                   cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                   a: 'float[:]', b: 'float[:]', c: 'float[:]'):
    r'''Single stage of a s-stage Runge-Kutta solve of

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant.
    '''

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)

    # marker position e and velocity v
    e = empty(3, dtype=float)
    v = empty(3, dtype=float)

    # intermediate k-vector
    k = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    # get number of stages
    n_stages = shape(b)[0]

    if stage == n_stages - 1:
        last = 1.
    else:
        last = 0.

    #$ omp parallel private(ip, e, v, dfm, dfinv, k)
    #$ omp for
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        v[:] = markers[ip, 3:6]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(e[0], e[1], e[2],
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # evaluate inverse Jacobian matrix
        linalg_kernels.matrix_inv(dfm, dfinv)

        # pull-back of velocity
        linalg_kernels.matrix_vector(dfinv, v, k)

        # accumulation for last stage
        markers[ip, 12:15] += dt*b[stage]*k

        # update positions for intermediate stages or last stage
        markers[ip, 0:3] = markers[ip, 9:12] + \
            dt*a[stage]*k + last*markers[ip, 12:15]

    #$ omp end parallel


@stack_array('dfm', 'dfinv', 'dfinv_t', 'ginv', 'eta', 'v', 'u', 'k', 'k_v', 'k_u', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def push_pc_eta_rk4_Hcurl_full(markers: 'float[:,:]', dt: float, stage: int,
                               pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                               starts: 'int[:]',
                               kind_map: int, params_map: 'float[:]',
                               p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                               ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                               cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                               u_1: 'float[:,:,:]', u_2: 'float[:,:,:]', u_3: 'float[:,:,:]'):
    r'''Fourth order Runge-Kutta solve of

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v + \textnormal{vec}( \hat{\mathbf U}^{1(2)})

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant and

    .. math::

        \textnormal{vec}( \hat{\mathbf U}^{1}) = G^{-1}\hat{\mathbf U}^{1}\,,\qquad \textnormal{vec}( \hat{\mathbf U}^{2}) = \frac{\hat{\mathbf U}^{2}}{\sqrt g}\,.

    Parameters
    ----------
        u_1, u_2, u_3: array[float]
            3d array of FE coeffs of U-field, either as 1-form or as 2-form.

        u_basis : int
            U is 1-form (u_basis=1) or a 2-form (u_basis=2).
    '''

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)
    ginv = empty((3, 3), dtype=float)

    # marker position and velocity
    eta = empty(3, dtype=float)
    v = empty(3, dtype=float)

    # U-fiels
    u = empty(3, dtype=float)

    # intermediate stages in RK4
    k = empty(3, dtype=float)
    k_v = empty(3, dtype=float)
    k_u = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    # assign factor of k for each stage
    if stage == 0 or stage == 3:
        nk = 1.
    else:
        nk = 2.

    # which stage
    if stage == 3:
        last = 1.
        cont = 0.
    elif stage == 2:
        last = 0.
        cont = 2.
    else:
        last = 0.
        cont = 1.

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta[:] = markers[ip, 0:3]
        v[:] = markers[ip, 3:6]

        # ----------------- stage n in Runge-Kutta method -------------------
        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta[0], eta[1], eta[2],
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # metric coeffs
        linalg_kernels.matrix_inv(dfm, dfinv)
        linalg_kernels.transpose(dfinv, dfinv_t)
        linalg_kernels.matrix_matrix(dfinv, dfinv_t, ginv)

        # pull-back of velocity
        linalg_kernels.matrix_vector(dfinv, v, k_v)

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], eta[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], eta[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], eta[2])

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

        # U-field
        u[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, u_1, starts)
        u[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, u_2, starts)
        u[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, u_3, starts)

        # transform to vector field
        linalg_kernels.matrix_vector(ginv, u, k_u)

        # sum contribs
        k[:] = k_v + k_u

        # accum k
        markers[ip, 12:15] += k*nk/6.

        # update markers for the next stage
        markers[ip, 0:3] = (markers[ip, 9:12] + dt*k/2 *
                            cont + dt*markers[ip, 12:15] * last)


@stack_array('dfm', 'dfinv', 'dfinv_t', 'ginv', 'eta', 'v', 'u', 'k', 'k_v', 'k_u', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def push_pc_eta_rk4_Hdiv_full(markers: 'float[:,:]', dt: float, stage: int,
                              pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                              starts: 'int[:]',
                              kind_map: int, params_map: 'float[:]',
                              p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                              ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                              cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                              u_1: 'float[:,:,:]', u_2: 'float[:,:,:]', u_3: 'float[:,:,:]'):
    r'''Fourth order Runge-Kutta solve of

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v + \textnormal{vec}( \hat{\mathbf U}^{1(2)})

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant and

    .. math::

        \textnormal{vec}( \hat{\mathbf U}^{1}) = G^{-1}\hat{\mathbf U}^{1}\,,\qquad \textnormal{vec}( \hat{\mathbf U}^{2}) = \frac{\hat{\mathbf U}^{2}}{\sqrt g}\,.

    Parameters
    ----------
        u_1, u_2, u_3: array[float]
            3d array of FE coeffs of U-field, either as 1-form or as 2-form.

        u_basis : int
            U is 1-form (u_basis=1) or a 2-form (u_basis=2).
    '''

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)
    ginv = empty((3, 3), dtype=float)

    # marker position and velocity
    eta = empty(3, dtype=float)
    v = empty(3, dtype=float)

    # U-fiels
    u = empty(3, dtype=float)

    # intermediate stages in RK4
    k = empty(3, dtype=float)
    k_v = empty(3, dtype=float)
    k_u = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    # assign factor of k for each stage
    if stage == 0 or stage == 3:
        nk = 1.
    else:
        nk = 2.

    # is it the last stage?
    if stage == 3:
        last = 1.
        cont = 0.
    else:
        last = 0.
        cont = 1.

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta[:] = markers[ip, 0:3]
        v[:] = markers[ip, 3:6]

        # ----------------- stage n in Runge-Kutta method -------------------
        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta[0], eta[1], eta[2],
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # metric coeffs
        det_df = linalg_kernels.det(dfm)
        linalg_kernels.matrix_inv(dfm, dfinv)
        linalg_kernels.transpose(dfinv, dfinv_t)
        linalg_kernels.matrix_matrix(dfinv, dfinv_t, ginv)

        # pull-back of velocity
        linalg_kernels.matrix_vector(dfinv, v, k_v)

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], eta[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], eta[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], eta[2])

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

        # U-field
        u[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, u_1, starts)
        u[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, u_2, starts)
        u[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, u_3, starts)

        # transform to vector field
        k_u[:] = u/det_df

        # sum contribs
        k[:] = k_v + k_u

        # accum k
        markers[ip, 12:15] += k*nk/6.

        # update markers for the next stage
        markers[ip, 0:3] = (markers[ip, 9:12] + dt*k/2 *
                            cont + dt*markers[ip, 12:15] * last)


@stack_array('dfm', 'dfinv', 'dfinv_t', 'ginv', 'eta', 'v', 'u', 'k', 'k_v', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def push_pc_eta_rk4_H1vec_full(markers: 'float[:,:]', dt: float, stage: int,
                               pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                               starts: 'int[:]',
                               kind_map: int, params_map: 'float[:]',
                               p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                               ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                               cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                               u_1: 'float[:,:,:]', u_2: 'float[:,:,:]', u_3: 'float[:,:,:]'):
    r'''Fourth order Runge-Kutta solve of

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v + \textnormal{vec}( \hat{\mathbf U}^{1(2)})

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant and

    .. math::

        \textnormal{vec}( \hat{\mathbf U}^{1}) = G^{-1}\hat{\mathbf U}^{1}\,,\qquad \textnormal{vec}( \hat{\mathbf U}^{2}) = \frac{\hat{\mathbf U}^{2}}{\sqrt g}\,.

    Parameters
    ----------
        u_1, u_2, u_3: array[float]
            3d array of FE coeffs of U-field, either as 1-form or as 2-form.

        u_basis : int
            U is 1-form (u_basis=1) or a 2-form (u_basis=2).
    '''

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)
    ginv = empty((3, 3), dtype=float)

    # marker position and velocity
    eta = empty(3, dtype=float)
    v = empty(3, dtype=float)

    # U-fiels
    u = empty(3, dtype=float)

    # intermediate stages in RK4
    k = empty(3, dtype=float)
    k_v = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    # assign factor of k for each stage
    if stage == 0 or stage == 3:
        nk = 1.
    else:
        nk = 2.

    # which stage
    if stage == 3:
        last = 1.
        cont = 0.
    elif stage == 2:
        last = 0.
        cont = 2.
    else:
        last = 0.
        cont = 1.

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta[:] = markers[ip, 0:3]
        v[:] = markers[ip, 3:6]

        # ----------------- stage n in Runge-Kutta method -------------------
        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta[0], eta[1], eta[2],
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # metric coeffs
        linalg_kernels.matrix_inv(dfm, dfinv)
        linalg_kernels.transpose(dfinv, dfinv_t)
        linalg_kernels.matrix_matrix(dfinv, dfinv_t, ginv)

        # pull-back of velocity
        linalg_kernels.matrix_vector(dfinv, v, k_v)

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], eta[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], eta[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], eta[2])

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

        # U-field
        u[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u_1, starts)
        u[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u_2, starts)
        u[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u_3, starts)

        # sum contribs
        k[:] = k_v + u

        # accum k
        markers[ip, 12:15] += k*nk/6.

        # update markers for the next stage
        markers[ip, 0:3] = (markers[ip, 9:12] + dt*k/2 *
                            cont + dt*markers[ip, 12:15] * last)


@stack_array('dfm', 'dfinv', 'dfinv_t', 'ginv', 'eta', 'v', 'u', 'k', 'k_v', 'k_u', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def push_pc_eta_rk4_Hcurl(markers: 'float[:,:]', dt: float, stage: int,
                          pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                          starts: 'int[:]',
                          kind_map: int, params_map: 'float[:]',
                          p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                          ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                          cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                          u_1: 'float[:,:,:]', u_2: 'float[:,:,:]', u_3: 'float[:,:,:]'):
    r'''Fourth order Runge-Kutta solve of

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v + \textnormal{vec}( \hat{\mathbf U}^{1(2)})

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant and

    .. math::

        \textnormal{vec}( \hat{\mathbf U}^{1}) = G^{-1}\hat{\mathbf U}^{1}\,,\qquad \textnormal{vec}( \hat{\mathbf U}^{2}) = \frac{\hat{\mathbf U}^{2}}{\sqrt g}\,.

    Parameters
    ----------
        u_1, u_2, u_3: array[float]
            3d array of FE coeffs of U-field, either as 1-form or as 2-form.

        u_basis : int
            U is 1-form (u_basis=1) or a 2-form (u_basis=2).
    '''

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)
    ginv = empty((3, 3), dtype=float)

    # marker position and velocity
    eta = empty(3, dtype=float)
    v = empty(3, dtype=float)

    # U-fiels
    u = empty(3, dtype=float)

    # intermediate stages in RK4
    k = empty(3, dtype=float)
    k_v = empty(3, dtype=float)
    k_u = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    # assign factor of k for each stage
    if stage == 0 or stage == 3:
        nk = 1.
    else:
        nk = 2.

    # which stage
    if stage == 3:
        last = 1.
        cont = 0.
    elif stage == 2:
        last = 0.
        cont = 2.
    else:
        last = 0.
        cont = 1.

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta[:] = markers[ip, 0:3]
        v[:] = markers[ip, 3:6]

        # ----------------- stage n in Runge-Kutta method -------------------
        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta[0], eta[1], eta[2],
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # metric coeffs
        linalg_kernels.matrix_inv(dfm, dfinv)
        linalg_kernels.transpose(dfinv, dfinv_t)
        linalg_kernels.matrix_matrix(dfinv, dfinv_t, ginv)

        # pull-back of velocity
        linalg_kernels.matrix_vector(dfinv, v, k_v)

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], eta[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], eta[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], eta[2])

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

        # U-field
        u[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, u_1, starts)
        u[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, u_2, starts)
        u[2] = 0.

        # transform to vector field
        linalg_kernels.matrix_vector(ginv, u, k_u)

        # sum contribs
        k[:] = k_v + k_u

        # accum k
        markers[ip, 12:15] += k*nk/6.

        # update markers for the next stage
        markers[ip, 0:3] = (markers[ip, 9:12] + dt*k/2 *
                            cont + dt*markers[ip, 12:15] * last)


@stack_array('dfm', 'dfinv', 'dfinv_t', 'ginv', 'eta', 'v', 'u', 'k', 'k_v', 'k_u', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def push_pc_eta_rk4_Hdiv(markers: 'float[:,:]', dt: float, stage: int,
                         pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                         starts: 'int[:]',
                         kind_map: int, params_map: 'float[:]',
                         p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                         ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                         cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                         u_1: 'float[:,:,:]', u_2: 'float[:,:,:]', u_3: 'float[:,:,:]'):
    r'''Fourth order Runge-Kutta solve of

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v + \textnormal{vec}( \hat{\mathbf U}^{1(2)})

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant and

    .. math::

        \textnormal{vec}( \hat{\mathbf U}^{1}) = G^{-1}\hat{\mathbf U}^{1}\,,\qquad \textnormal{vec}( \hat{\mathbf U}^{2}) = \frac{\hat{\mathbf U}^{2}}{\sqrt g}\,.

    Parameters
    ----------
        u_1, u_2, u_3: array[float]
            3d array of FE coeffs of U-field, either as 1-form or as 2-form.

        u_basis : int
            U is 1-form (u_basis=1) or a 2-form (u_basis=2).
    '''

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)
    ginv = empty((3, 3), dtype=float)

    # marker position and velocity
    eta = empty(3, dtype=float)
    v = empty(3, dtype=float)

    # U-fiels
    u = empty(3, dtype=float)

    # intermediate stages in RK4
    k = empty(3, dtype=float)
    k_v = empty(3, dtype=float)
    k_u = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    # assign factor of k for each stage
    if stage == 0 or stage == 3:
        nk = 1.
    else:
        nk = 2.

    # is it the last stage?
    if stage == 3:
        last = 1.
        cont = 0.
    else:
        last = 0.
        cont = 1.

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta[:] = markers[ip, 0:3]
        v[:] = markers[ip, 3:6]

        # ----------------- stage n in Runge-Kutta method -------------------
        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta[0], eta[1], eta[2],
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # metric coeffs
        det_df = linalg_kernels.det(dfm)
        linalg_kernels.matrix_inv(dfm, dfinv)
        linalg_kernels.transpose(dfinv, dfinv_t)
        linalg_kernels.matrix_matrix(dfinv, dfinv_t, ginv)

        # pull-back of velocity
        linalg_kernels.matrix_vector(dfinv, v, k_v)

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], eta[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], eta[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], eta[2])

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

        # U-field
        u[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, u_1, starts)
        u[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, u_2, starts)
        u[2] = 0.

        # transform to vector field
        k_u[:] = u/det_df

        # sum contribs
        k[:] = k_v + k_u

        # accum k
        markers[ip, 12:15] += k*nk/6.

        # update markers for the next stage
        markers[ip, 0:3] = (markers[ip, 9:12] + dt*k/2 *
                            cont + dt*markers[ip, 12:15] * last)


@stack_array('dfm', 'dfinv', 'dfinv_t', 'ginv', 'eta', 'v', 'u', 'k', 'k_v', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def push_pc_eta_rk4_H1vec(markers: 'float[:,:]', dt: float, stage: int,
                          pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                          starts: 'int[:]',
                          kind_map: int, params_map: 'float[:]',
                          p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                          ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                          cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                          u_1: 'float[:,:,:]', u_2: 'float[:,:,:]', u_3: 'float[:,:,:]'):
    r'''Fourth order Runge-Kutta solve of

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v + \textnormal{vec}( \hat{\mathbf U}^{1(2)})

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant and

    .. math::

        \textnormal{vec}( \hat{\mathbf U}^{1}) = G^{-1}\hat{\mathbf U}^{1}\,,\qquad \textnormal{vec}( \hat{\mathbf U}^{2}) = \frac{\hat{\mathbf U}^{2}}{\sqrt g}\,.

    Parameters
    ----------
        u_1, u_2, u_3: array[float]
            3d array of FE coeffs of U-field, either as 1-form or as 2-form.

        u_basis : int
            U is 1-form (u_basis=1) or a 2-form (u_basis=2).
    '''

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)
    ginv = empty((3, 3), dtype=float)

    # marker position and velocity
    eta = empty(3, dtype=float)
    v = empty(3, dtype=float)

    # U-fiels
    u = empty(3, dtype=float)

    # intermediate stages in RK4
    k = empty(3, dtype=float)
    k_v = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    # assign factor of k for each stage
    if stage == 0 or stage == 3:
        nk = 1.
    else:
        nk = 2.

    # which stage
    if stage == 3:
        last = 1.
        cont = 0.
    elif stage == 2:
        last = 0.
        cont = 2.
    else:
        last = 0.
        cont = 1.

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta[:] = markers[ip, 0:3]
        v[:] = markers[ip, 3:6]

        # ----------------- stage n in Runge-Kutta method -------------------
        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta[0], eta[1], eta[2],
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # metric coeffs
        linalg_kernels.matrix_inv(dfm, dfinv)
        linalg_kernels.transpose(dfinv, dfinv_t)
        linalg_kernels.matrix_matrix(dfinv, dfinv_t, ginv)

        # pull-back of velocity
        linalg_kernels.matrix_vector(dfinv, v, k_v)

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], eta[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], eta[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], eta[2])

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

        # U-field
        u[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u_1, starts)
        u[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u_2, starts)
        u[2] = 0.

        # sum contribs
        k[:] = k_v + u

        # accum k
        markers[ip, 12:15] += k*nk/6.

        # update markers for the next stage
        markers[ip, 0:3] = (markers[ip, 9:12] + dt*k/2 *
                            cont + dt*markers[ip, 12:15] * last)


@stack_array('bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'dfm', 'df_inv', 'v', 'df_inv_v')
def push_weights_with_efield_lin_vm(markers: 'float[:,:]', dt: float, stage: int,
                                    pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                                    starts: 'int[:]',
                                    kind_map: int, params_map: 'float[:]',
                                    p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                                    ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                                    cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                                    e1_1: 'float[:,:,:]', e1_2: 'float[:,:,:]', e1_3: 'float[:,:,:]',
                                    f0_values: 'float[:]', f0_params: 'float[:]',
                                    n_markers_tot: 'int', kappa: 'float'):
    r'''
    updates the single weights in the e_W substep of the Vlasov Maxwell system with delta-f;
    c.f. struphy.propagators.propagators.StepEfieldWeights

    Parameters :
    ------------
        e1_1, e1_2, e1_3: array[float]
            3d array of FE coeffs of E-field as 1-form.

        f0_values ; array[float]
            Value of f0 for each particle.

        f0_params : array[float]
            Parameters needed to specify the moments, in ascending order.

        n_markers_tot : int
            total number of particles

        kappa : float
            = 2 * pi * Omega_c / omega ; Parameter determining the coupling strength between particles and fields
    '''

    # total number of basis functions : B-splines (pn) and D-splines (pn-1)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # non-vanishing N-splines at particle position
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    v = empty(3, dtype=float)
    df_inv_v = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private (ip, eta1, eta2, eta3, dfm, df_inv, v, df_inv_v, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, f0, e_vec_1, e_vec_2, e_vec_3, update)
    #$ omp for
    for ip in range(n_markers):
        if markers[ip, 0] == -1:
            continue

        # position
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # get velocity
        v[0] = markers[ip, 3] / f0_params[4]**2
        v[1] = markers[ip, 4] / f0_params[5]**2
        v[2] = markers[ip, 5] / f0_params[6]**2

        # spans (i.e. index for non-vanishing basis functions)
        span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
        span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
        span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

        # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
        bsplines_kernels.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

        f0 = f0_values[ip]

        # Compute Jacobian matrix
        evaluation_kernels.df(eta1, eta2, eta3,
                              kind_map, params_map,
                              t1_map, t2_map, t3_map,
                              p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # invert Jacobian matrix
        linalg_kernels.matrix_inv(dfm, df_inv)

        # compute DF^{-1} v
        linalg_kernels.matrix_vector(df_inv, v, df_inv_v)

        # E-field (1-form)
        e_vec_1 = evaluation_kernels_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3,
                                                 span1, span2, span3, e1_1, starts)
        e_vec_2 = evaluation_kernels_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3,
                                                 span1, span2, span3, e1_2, starts)
        e_vec_3 = evaluation_kernels_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3,
                                                 span1, span2, span3, e1_3, starts)

        # w_{n+1} = w_n + dt * kappa / (2 * s_0) * sqrt(f_0) * ( DF^{-1} \V_th * v_p ) \cdot ( e_{n+1} + e_n )
        update = (df_inv_v[0] * e_vec_1 + df_inv_v[1] * e_vec_2 + df_inv_v[2] * e_vec_3) * \
            sqrt(f0) * dt * kappa / (2 * markers[ip, 7])
        markers[ip, 6] += update

    #$ omp end parallel


@stack_array('bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'dfm', 'df_inv', 'v', 'df_inv_v')
def push_weights_with_efield_delta_f_vm(markers: 'float[:,:]', dt: float, stage: int,
                                        pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                                        starts: 'int[:]',
                                        kind_map: int, params_map: 'float[:]',
                                        p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                                        ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                                        cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                                        e1_1: 'float[:,:,:]', e1_2: 'float[:,:,:]', e1_3: 'float[:,:,:]',
                                        f0_values: 'float[:]', vth: 'float',
                                        kappa: 'float', substep: 'int'):
    r"""
    updates the single weights in one of the e_W substep of the Vlasov Maxwell system;
    c.f. struphy.propagators.propagators.StepEfieldWeights

    Parameters :
    ------------
    e1_1, e1_2, e1_3: array[float]
        3d array of FE coeffs of E-field as 1-form.

    f0_values ; array[float]
        Value of f0 for each particle.

    f0_params : array[float]
        Parameters needed to specify the moments, in ascending order.

    n_markers_tot : int
        total number of particles

    kappa : float
        = 2 * pi * Omega_c / omega ; Parameter determining the coupling strength between particles and fields

    substep : int
        0 for explicit substep, 1 for symplectic substep
    """

    # total number of basis functions : B-splines (pn) and D-splines (pn-1)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # non-vanishing N-splines at particle position
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    v = empty(3, dtype=float)
    df_inv_v = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private (ip, eta1, eta2, eta3, dfm, df_inv, v, df_inv_v, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, f0, e_vec_1, e_vec_2, e_vec_3, update)
    #$ omp for
    for ip in range(n_markers):
        if markers[ip, 0] == -1:
            continue

        # position
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # spans (i.e. index for non-vanishing basis functions)
        span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
        span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
        span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

        # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
        bsplines_kernels.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

        f0 = f0_values[ip]

        # Compute Jacobian matrix
        evaluation_kernels.df(eta1, eta2, eta3,
                              kind_map, params_map,
                              t1_map, t2_map, t3_map,
                              p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # compute shifted and stretched velocity
        v[0] = markers[ip, 3]
        v[1] = markers[ip, 4]
        v[2] = markers[ip, 5]

        # invert Jacobian matrix
        linalg_kernels.matrix_inv(dfm, df_inv)
        linalg_kernels.matrix_vector(df_inv, v, df_inv_v)

        # E-field (1-form)
        e_vec_1 = evaluation_kernels_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3,
                                                 span1, span2, span3, e1_1, starts)
        e_vec_2 = evaluation_kernels_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3,
                                                 span1, span2, span3, e1_2, starts)
        e_vec_3 = evaluation_kernels_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3,
                                                 span1, span2, span3, e1_3, starts)

        update = kappa * (df_inv_v[0] * e_vec_1 +
                          df_inv_v[1] * e_vec_2 + df_inv_v[2] * e_vec_3)
        if substep == 0:
            # w_p += dt * kappa / s_0 * (DL^{-1} v_p) * e_vec
            # with e_vec = e(0) - dt / 2 * M_1^{-1} accum_vec
            update *= dt * (f0 / log(f0) - f0) / markers[ip, 7]
        elif substep == 1:
            # w_p -= dt * kappa * w_p / (vth^2 * ln(f_0)) * (DL^{-1} v_p) * e_vec
            # with e_vec = (e^{n+1} + e^n) / 2
            update *= (-1) * dt * markers[ip, 6] / (vth**2 * log(f0))

        markers[ip, 6] += update

    #$ omp end parallel


@stack_array('particle', 'dfm', 'df_inv', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'taus')
def push_x_v_static_efield(markers: 'float[:,:]', dt: float, stage: int,
                           pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                           starts: 'int[:]',
                           kind_map: int, params_map: 'float[:]',
                           p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                           ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                           cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                           loc1: 'float[:]', loc2: 'float[:]', loc3: 'float[:]',
                           weight1: 'float[:]', weight2: 'float[:]', weight3: 'float[:]',
                           e1_1: 'float[:,:,:]', e1_2: 'float[:,:,:]', e1_3: 'float[:,:,:]',
                           kappa: 'float',
                           eps: 'float[:]', maxiter: int):
    r"""
    particle pusher for ODE

    .. math::
        \frac{\text{d} \mathbf{\eta}}{\text{d} t} & = DL^{-1} \mathbf{v} \,

        \frac{\text{d} \mathbf{v}}{\text{d} t} & = \kappa \, DL^{-T} \mathbf{E}_0(\mathbf{\eta})

    Parameters 
    ----------
    loc1, loc2, loc3 : array
        contain the positions of the Legendre-Gauss quadrature points of necessary order to integrate basis splines exactly in each direction

    weight1, weight2, weight3 : array
        contain the values of the weights for the Legendre-Gauss quadrature in each direction

    e1_1, e1_2, e1_3: array[float]
        3d array of FE coeffs of the background E-field as 1-form.

    kappa : float
        = 2 * pi * Omega_c / omega ; Parameter determining the coupling strength between particles and fields

    eps: array
        determines the accuracy for the position (0th element) and velocity (1st element) with which the implicit scheme is executed

    maxiter : integer
        sets the maximum number of iterations for the iterative scheme
    """

    particle = zeros(9, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    # non-vanishing B-splines at particle position
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)

    # number of quadrature points in direction 1
    n_quad1 = int(floor((pn[0] - 1) * pn[1] * pn[2] / 2 + 1))
    # number of quadrature points in direction 2
    n_quad2 = int(floor(pn[0] * (pn[1] - 1) * pn[2] / 2 + 1))
    # number of quadrature points in direction 3
    n_quad3 = int(floor(pn[0] * pn[1] * (pn[2] - 1) / 2 + 1))

    # Create array for storing the tau values
    taus = empty(20, dtype=float)

    #$ omp parallel private(ip, run, bn1, bn2, bn3, bd1, bd2, bd3, dfm, df_inv, taus, temp, k, particle, dt2)
    #$ omp for
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        particle[:] = markers[ip, :]

        run = 1
        k = 0

        while run != 0:
            k += 1
            if k == 5:
                print(
                    'Splitting the time steps into 4 has not been enough, aborting the iteration.')
                print()
                break

            run = 0

            dt2 = dt/k

            for _ in range(k):
                temp = pusher_utilities_kernels.aux_fun_x_v_stat_e(particle,
                                          pn, tn1, tn2, tn3,
                                          starts, starts, starts,
                                          kind_map, params_map,
                                          p_map, t1_map, t2_map, t3_map,
                                          ind1_map, ind2_map, ind3_map,
                                          cx, cy, cz,
                                          n_quad1, n_quad2, n_quad3,
                                          dfm, df_inv,
                                          bn1, bn2, bn3, bd1, bd2, bd3,
                                          taus,
                                          dt2,
                                          loc1, loc2, loc3, weight1, weight2, weight3,
                                          e1_1, e1_2, e1_3,
                                          kappa,
                                          eps, maxiter)
                run = run + temp

        # write the results in the particles array
        markers[ip, :] = particle[:]

    #$ omp end parallel
