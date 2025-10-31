"Pusher kernels for full orbit (6D) particles."

from numpy import cos, empty, floor, log, shape, sin, sqrt, zeros
from pyccel.decorators import stack_array

import struphy.bsplines.bsplines_kernels as bsplines_kernels
import struphy.bsplines.evaluation_kernels_3d as evaluation_kernels_3d
import struphy.geometry.evaluation_kernels as evaluation_kernels

# do not remove; needed to identify dependencies
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.pushing.pusher_utilities_kernels as pusher_utilities_kernels
import struphy.pic.sph_eval_kernels as sph_eval_kernels
from struphy.bsplines.evaluation_kernels_3d import (
    eval_0form_spline_mpi,
    eval_1form_spline_mpi,
    eval_2form_spline_mpi,
    eval_3form_spline_mpi,
    eval_vectorfield_spline_mpi,
    get_spans,
)
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("dfm", "dfinv", "dfinvt", "e_form", "e_cart")
def push_v_with_efield(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    e1_1: "float[:,:,:]",
    e1_2: "float[:,:,:]",
    e1_3: "float[:,:,:]",
    const: "float",
):
    r"""Updates particle velocities as

    .. math::

        \frac{\mathbf v^{n+1} - \mathbf v^n}{\Delta t} = c \, \bar{DF}^{-\top}  (\mathbb L^1)^\top \mathbf e

    where :math:`\mathbf e \in \mathbb R^{N_1}` are given FE coefficients of the 1-form spline field
    and :math:`c \in \mathbb R` is some constant.

    Parameters
    ----------
        e1_1, e1_2, e1_3 : ndarray[float]
            3d array of FE coeffs of E-field as 1-form.

        const : float
            A constant (usuallly related to the charge-to-mass ratio).
    """

    # allocate metric coeffs
    dfm = zeros((3, 3), dtype=float)
    dfinv = zeros((3, 3), dtype=float)
    dfinvt = zeros((3, 3), dtype=float)

    # allocate for field evaluations (1-form and Cartesian components)
    e_form = zeros(3, dtype=float)
    e_cart = zeros(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers

    # -- removed omp: #$ omp parallel private(ip, eta1, eta2, eta3, dfm, dfinv, dfinvt, span1, span2, span3, e_form, e_cart)
    # -- removed omp: #$ omp for
    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta1,
            eta2,
            eta3,
            args_domain,
            dfm,
        )

        # metric coeffs
        linalg_kernels.matrix_inv(dfm, dfinv)
        linalg_kernels.transpose(dfinv, dfinvt)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # electric field: 1-form components
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            e1_1,
            e1_2,
            e1_3,
            e_form,
        )

        # electric field: Cartesian components
        linalg_kernels.matrix_vector(dfinvt, e_form, e_cart)

        # update velocities
        markers[ip, 3:6] += dt * const * e_cart

    # -- removed omp: #$ omp end parallel


@stack_array("dfm", "b_form", "b_cart", "b_norm", "v", "vperp", "vxb_norm", "b_normxvperp")
def push_vxb_analytic(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    b2_1: "float[:,:,:]",
    b2_2: "float[:,:,:]",
    b2_3: "float[:,:,:]",
):
    r"""Solves exactly the rotation

    .. math::

        \frac{\textnormal d \mathbf v_p(t)}{\textnormal d t} =  \mathbf v_p(t) \times \frac{DF\, \hat{\mathbf B}^2}{\sqrt g}

    for each marker :math:`p` in markers array, with fixed rotation vector.

    Parameters
    ----------
        b2_1, b2_2, b2_3: array[float]
            3d array of FE coeffs of B-field as 2-form.
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # allocate for field evaluations (2-form components, Cartesian components and normalized Cartesian components)
    b_form = empty(3, dtype=float)
    b_cart = empty(3, dtype=float)
    b_norm = empty(3, dtype=float)

    # particle velocity
    v = empty(3, dtype=float)

    # perpendicular velocity, v x b_norm and b_norm x vperp
    vperp = empty(3, dtype=float)
    vxb_norm = empty(3, dtype=float)
    b_normxvperp = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    first_init_idx = args_markers.first_init_idx

    # -- removed omp: #$ omp parallel private (ip, e1, e2, e3, v, dfm, det_df, span1, span2, span3, b_form, b_cart, b_abs, b_norm, vpar, vxb_norm, vperp, b_normxvperp)
    # -- removed omp: #$ omp for
    for ip in range(n_markers):
        # check if marker is a hole
        if markers[ip, first_init_idx] == -1.0 or markers[ip, -1] == -2.0:
            continue

        e1 = markers[ip, 0]
        e2 = markers[ip, 1]
        e3 = markers[ip, 2]
        v[:] = markers[ip, 3:6]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            e1,
            e2,
            e3,
            args_domain,
            dfm,
        )

        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1, span2, span3 = get_spans(e1, e2, e3, args_derham)

        # magnetic field 2-form
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            b2_1,
            b2_2,
            b2_3,
            b_form,
        )

        # magnetic field: Cartesian components
        linalg_kernels.matrix_vector(dfm, b_form, b_cart)
        b_cart[:] = b_cart / det_df

        # magnetic field: magnitude
        b_abs = sqrt(b_cart[0] ** 2 + b_cart[1] ** 2 + b_cart[2] ** 2)

        # only push vxb if magnetic field is non-zero
        if b_abs != 0.0:
            # normalized magnetic field direction
            b_norm[:] = b_cart / b_abs

            # parallel velocity v.b_norm
            vpar = linalg_kernels.scalar_dot(v, b_norm)

            # first component of perpendicular velocity
            linalg_kernels.cross(v, b_norm, vxb_norm)
            linalg_kernels.cross(b_norm, vxb_norm, vperp)

            # second component of perpendicular velocity
            linalg_kernels.cross(b_norm, vperp, b_normxvperp)

            # analytic rotation
            markers[ip, 3:6] = vpar * b_norm + cos(b_abs * dt) * vperp - sin(b_abs * dt) * b_normxvperp

    # -- removed omp: #$ omp end parallel


@stack_array("dfm", "b_form", "b_cart", "b_prod", "v", "identity", "rhs", "lhs", "lhs_inv", "vec", "res")
def push_vxb_implicit(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    b2_1: "float[:,:,:]",
    b2_2: "float[:,:,:]",
    b2_3: "float[:,:,:]",
):
    r"""Solves the rotation

    .. math::

        \frac{\textnormal d \mathbf v_p(t)}{\textnormal d t} =  \mathbf v_p(t) \times \frac{DF\, \hat{\mathbf B}^2}{\sqrt g}

    with the Crank-Nicolson method for each marker :math:`p` in markers array, with fixed rotation vector.

    Parameters
    ----------
        b2_1, b2_2, b2_3: array[float]
            3d array of FE coeffs of B-field as 2-form.
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # allocate for field evaluations (2-form components, Cartesian components and rotation matrix such that vxB = B_prod.v)
    b_form = empty(3, dtype=float)
    b_cart = empty(3, dtype=float)
    b_prod = zeros((3, 3), dtype=float)

    # particle position and velocity
    v = empty(3, dtype=float)

    # identity matrix
    identity = zeros((3, 3), dtype=float)

    identity[0, 0] = 1.0
    identity[1, 1] = 1.0
    identity[2, 2] = 1.0

    # right-hand side and left-hand side of Crank-Nicolson scheme
    rhs = empty((3, 3), dtype=float)
    lhs = empty((3, 3), dtype=float)

    lhs_inv = empty((3, 3), dtype=float)

    vec = empty(3, dtype=float)
    res = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    first_init_idx = args_markers.first_init_idx

    # -- removed omp: #$ omp parallel firstprivate(b_prod) private (ip, v, dfm, det_df, span1, span2, span3, b_form, b_cart, rhs, lhs, lhs_inv, vec, res)
    # -- removed omp: #$ omp for
    for ip in range(n_markers):
        # check if marker is a hole
        if markers[ip, first_init_idx] == -1.0:
            continue

        e1 = markers[ip, 0]
        e2 = markers[ip, 1]
        e3 = markers[ip, 2]
        v[:] = markers[ip, 3:6]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            e1,
            e2,
            e3,
            args_domain,
            dfm,
        )

        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1, span2, span3 = get_spans(e1, e2, e3, args_derham)

        # magnetic field 2-form
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            b2_1,
            b2_2,
            b2_3,
            b_form,
        )

        # magnetic field: Cartesian components
        linalg_kernels.matrix_vector(dfm, b_form, b_cart)
        b_cart[:] = b_cart / det_df

        # magnetic field: rotation matrix
        b_prod[0, 1] = b_cart[2]
        b_prod[0, 2] = -b_cart[1]

        b_prod[1, 0] = -b_cart[2]
        b_prod[1, 2] = b_cart[0]

        b_prod[2, 0] = b_cart[1]
        b_prod[2, 1] = -b_cart[0]

        # solve 3x3 system
        rhs[:, :] = identity + dt / 2 * b_prod
        lhs[:, :] = identity - dt / 2 * b_prod

        linalg_kernels.matrix_inv(lhs, lhs_inv)

        linalg_kernels.matrix_vector(rhs, v, vec)
        linalg_kernels.matrix_vector(lhs_inv, vec, res)

        markers[ip, 3:6] = res

    # -- removed omp: #$ omp end parallel


@stack_array(
    "dfm",
    "dfinv",
    "dfinv_t",
    "rot_temp",
    "b_form",
    "b_cart",
    "b_norm",
    "v",
    "vperp",
    "vxb_norm",
    "b_normxvperp",
    "bn1",
    "bn2",
    "bn3",
    "bd1",
    "bd2",
    "bd3",
)
def push_pxb_analytic(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    b2_1: "float[:,:,:]",
    b2_2: "float[:,:,:]",
    b2_3: "float[:,:,:]",
    a1_1: "float[:,:,:]",
    a1_2: "float[:,:,:]",
    a1_3: "float[:,:,:]",
):
    r"""Solves exactly the rotation

    .. math::

        \frac{\textnormal d \mathbf v_p(t)}{\textnormal d t} =  \mathbf v_p(t) \times \frac{DF\, \hat{\mathbf B}^2}{\sqrt g}

    for each marker :math:`p` in markers array, with fixed rotation vector.

    Parameters
    ----------
        b2_1, b2_2, b2_3: array[float]
            3d array of FE coeffs of B-field as 2-form.
    """

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

    # particle velocity
    v = empty(3, dtype=float)

    # perpendicular velocity, v x b_norm and b_norm x vperp
    vperp = empty(3, dtype=float)
    vxb_norm = empty(3, dtype=float)
    b_normxvperp = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers

    # -- removed omp: #$ omp parallel private (ip, v, dfm, dfinv, dfinv_t, det_df, span1, span2, span3, b_form, a_form, b_cart, b_abs, b_norm, vpar, vxb_norm, vperp, b_normxvperp)
    # -- removed omp: #$ omp for
    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        e1 = markers[ip, 0]
        e2 = markers[ip, 1]
        e3 = markers[ip, 2]
        v[:] = markers[ip, 3:6]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            e1,
            e2,
            e3,
            args_domain,
            dfm,
        )

        linalg_kernels.matrix_inv(dfm, dfinv)
        linalg_kernels.transpose(dfinv, dfinv_t)
        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1, span2, span3 = get_spans(e1, e2, e3, args_derham)

        # magnetic field: 2-form components
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            b2_1,
            b2_2,
            b2_3,
            b_form,
        )

        # vector potential: 1-form components
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            a1_1,
            a1_2,
            a1_3,
            a_form,
        )

        rot_temp[0] = dfinv_t[0, 0] * a_form[0] + dfinv_t[0, 1] * a_form[1] + dfinv_t[0, 2] * a_form[2]
        rot_temp[1] = dfinv_t[1, 0] * a_form[0] + dfinv_t[1, 1] * a_form[1] + dfinv_t[1, 2] * a_form[2]
        rot_temp[2] = dfinv_t[2, 0] * a_form[0] + dfinv_t[2, 1] * a_form[1] + dfinv_t[2, 2] * a_form[2]

        v[0] = v[0] - rot_temp[0]
        v[1] = v[1] - rot_temp[1]
        v[2] = v[2] - rot_temp[2]

        # magnetic field: Cartesian components
        linalg_kernels.matrix_vector(dfm, b_form, b_cart)
        b_cart[:] = b_cart / det_df

        # normalized magnetic field direction
        b_abs = sqrt(b_cart[0] ** 2 + b_cart[1] ** 2 + b_cart[2] ** 2)

        if b_abs != 0.0:
            b_norm[:] = b_cart / b_abs
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
        markers[ip, 3:6] = vpar * b_norm + cos(b_abs * dt) * vperp - sin(b_abs * dt) * b_normxvperp + rot_temp

    # -- removed omp: #$ omp end parallel


@stack_array("dfm", "dfinv", "dfinv_t")
def push_hybrid_xp_lnn(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    p_shape: "int[:]",
    p_size: "float[:]",
    Nel: "int[:]",
    pts1: "float[:]",
    pts2: "float[:]",
    pts3: "float[:]",
    wts1: "float[:]",
    wts2: "float[:]",
    wts3: "float[:]",
    weight: "float[:,:,:,:,:,:]",
    thermal: "float",
    n_quad: "int[:]",
):
    r"""Solves exactly the rotation

    .. math::

        \frac{\textnormal d \mathbf v_p(t)}{\textnormal d t} =  \mathbf v_p(t) \times \frac{DF\, \hat{\mathbf B}^2}{\sqrt g}

    for each marker :math:`p` in markers array, with fixed rotation vector.
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)

    compact = zeros(3, dtype=float)
    compact[0] = (p_shape[0] + 1.0) * p_size[0]
    compact[1] = (p_shape[1] + 1.0) * p_size[1]
    compact[2] = (p_shape[2] + 1.0) * p_size[2]

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

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers

    # -- removed omp: #$ omp parallel private (ip, eta1, eta2, eta3, dfm, dfinv, dfinv_t, det_df, point_left, point_right, cell_left, cell_number, i, grids_shapex, grids_shapey, grids_shapez, x_ii, y_ii, z_ii, il1, il2, il3, q1, q2, q3, temp1, temp4, temp6, valuexyz, dvaluexyz, temp8, ww)
    # -- removed omp: #$ omp for
    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta1,
            eta2,
            eta3,
            args_domain,
            dfm,
        )

        linalg_kernels.matrix_inv(dfm, dfinv)
        linalg_kernels.transpose(dfinv, dfinv_t)
        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        point_left[0] = eta1 - 0.5 * compact[0]
        point_right[0] = eta1 + 0.5 * compact[0]
        point_left[1] = eta2 - 0.5 * compact[1]
        point_right[1] = eta2 + 0.5 * compact[1]
        point_left[2] = eta3 - 0.5 * compact[2]
        point_right[2] = eta3 + 0.5 * compact[2]

        cell_left[0] = int(floor(point_left[0] * Nel[0]))
        cell_left[1] = int(floor(point_left[1] * Nel[1]))
        cell_left[2] = int(floor(point_left[2] * Nel[2]))

        cell_number[0] = int(floor(point_right[0] * Nel[0])) - cell_left[0] + 1.0
        cell_number[1] = int(floor(point_right[1] * Nel[1])) - cell_left[1] + 1.0
        cell_number[2] = int(floor(point_right[2] * Nel[2])) - cell_left[2] + 1.0

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
        x_ii = args_derham.pn[0] + cell_left[0] - args_derham.starts[0]
        y_ii = args_derham.pn[1] + cell_left[1] - args_derham.starts[1]
        z_ii = args_derham.pn[2] + cell_left[2] - args_derham.starts[2]

        # ======================================
        for il1 in range(cell_number[0]):
            for il2 in range(cell_number[1]):
                for il3 in range(cell_number[2]):
                    for q1 in range(n_quad[0]):
                        for q2 in range(n_quad[1]):
                            for q3 in range(n_quad[2]):
                                # quadrature points in the cell x direction
                                temp1[0] = (cell_left[0] + il1) / Nel[0] + pts1[q1]
                                # if > 0, result is 0
                                temp4[0] = abs(temp1[0] - eta1) - compact[0] / 2

                                temp1[1] = (cell_left[1] + il2) / Nel[1] + pts2[q2]
                                # if > 0, result is 0
                                temp4[1] = abs(temp1[1] - eta2) - compact[1] / 2

                                temp1[2] = (cell_left[2] + il3) / Nel[2] + pts3[q3]
                                # if > 0, result is 0
                                temp4[2] = abs(temp1[2] - eta3) - compact[2] / 2

                                if temp4[0] < 0 and temp4[1] < 0 and temp4[2] < 0:
                                    valuexyz[0] = bsplines_kernels.convolution(
                                        p_shape[0],
                                        grids_shapex,
                                        temp1[0],
                                    )
                                    dvaluexyz[0] = bsplines_kernels.convolution_der(
                                        p_shape[0],
                                        grids_shapex,
                                        temp1[0],
                                    )

                                    valuexyz[1] = bsplines_kernels.piecewise(
                                        p_shape[1],
                                        p_size[1],
                                        temp1[1] - eta2,
                                    )
                                    dvaluexyz[1] = bsplines_kernels.piecewise(
                                        p_shape[2],
                                        p_size[2],
                                        temp1[2] - eta3,
                                    )

                                    valuexyz[2] = bsplines_kernels.piecewise_der(
                                        p_shape[1],
                                        p_size[1],
                                        temp1[1] - eta2,
                                    )
                                    dvaluexyz[2] = bsplines_kernels.piecewise_der(
                                        p_shape[2],
                                        p_size[2],
                                        temp1[2] - eta3,
                                    )

                                    temp8[0] = dvaluexyz[0] * valuexyz[1] * valuexyz[2]
                                    temp8[1] = valuexyz[0] * dvaluexyz[1] * valuexyz[2]
                                    temp8[2] = valuexyz[0] * valuexyz[1] * dvaluexyz[2]

                                    ww[0] = (
                                        weight[
                                            x_ii + il1,
                                            y_ii + il2,
                                            z_ii + il3,
                                            q1,
                                            q2,
                                            q3,
                                        ]
                                        * wts1[q1]
                                        * wts2[q2]
                                        * wts3[q3]
                                    )

                                    temp6[0] = (
                                        dfinv_t[0, 0] * temp8[0] + dfinv_t[0, 1] * temp8[1] + dfinv_t[0, 2] * temp8[2]
                                    )
                                    temp6[1] = (
                                        dfinv_t[1, 0] * temp8[0] + dfinv_t[1, 1] * temp8[1] + dfinv_t[1, 2] * temp8[2]
                                    )
                                    temp6[2] = (
                                        dfinv_t[2, 0] * temp8[0] + dfinv_t[2, 1] * temp8[1] + dfinv_t[2, 2] * temp8[2]
                                    )
                                    # check weight_123 index
                                    markers[ip, 3] += dt * ww[0] * thermal * temp6[0]
                                    markers[ip, 4] += dt * ww[0] * thermal * temp6[1]
                                    markers[ip, 5] += dt * ww[0] * thermal * temp6[2]

    # -- removed omp: #$ omp end parallel


@stack_array("dfm", "dfinv", "dfinv_t", "b1", "b2", "b3", "d1", "d2", "d3")
def push_hybrid_xp_ap(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    pn1: int,
    pn2: int,
    pn3: int,
    a1_1: "float[:,:,:]",
    a1_2: "float[:,:,:]",
    a1_3: "float[:,:,:]",
):
    r"""Solves exactly the rotation

    .. math::

        \frac{\textnormal d \mathbf v_p(t)}{\textnormal d t} =  \mathbf v_p(t) \times \frac{DF\, \hat{\mathbf B}^2}{\sqrt g}

    for each marker :math:`p` in markers array, with fixed rotation vector.
    """

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

    # TODO: use newer spline evlaution with kernels
    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    l1 = zeros(pn1, dtype=float)
    l2 = zeros(pn2, dtype=float)
    l3 = zeros(pn3, dtype=float)

    r1 = zeros(pn1, dtype=float)
    r2 = zeros(pn2, dtype=float)
    r3 = zeros(pn3, dtype=float)

    b1 = zeros((pn1 + 1, pn1 + 1), dtype=float)
    b2 = zeros((pn2 + 1, pn2 + 1), dtype=float)
    b3 = zeros((pn3 + 1, pn3 + 1), dtype=float)

    d1 = zeros(pn1, dtype=float)
    d2 = zeros(pn2, dtype=float)
    d3 = zeros(pn3, dtype=float)

    bdd1 = zeros(pd1, dtype=float)
    bdd2 = zeros(pd2, dtype=float)
    bdd3 = zeros(pd3, dtype=float)

    # particle position and velocity
    v = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers

    # -- removed omp: #$ omp parallel private (ip, v, dfm, dfinv, dfinv_t, span1, span2, span3, bdd1, bdd2, bdd3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, a_form, a_xx, a_xxtrans, matrixp, matrixpp, matrixppp, lhs, rhs, lhsinv)
    # -- removed omp: #$ omp for
    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        e1 = markers[ip, 0]
        e2 = markers[ip, 1]
        e3 = markers[ip, 2]
        v[:] = markers[ip, 3:6]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            e1,
            e2,
            e3,
            args_domain,
            dfm,
        )

        linalg_kernels.matrix_inv(dfm, dfinv)
        linalg_kernels.transpose(dfinv, dfinv_t)

        for il1 in range(pd1):
            bdd1[il1] = b1[pd1 - 1, il1] * d1[il1] * d1[il1]

        for il2 in range(pd2):
            bdd2[il2] = b2[pd2 - 1, il2] * d2[il2] * d2[il2]

        for il3 in range(pd3):
            bdd3[il3] = b3[pd3 - 1, il3] * d3[il3] * d3[il3]

        # spline evaluation
        span1, span2, span3 = get_spans(e1, e2, e3, args_derham)

        bsplines_kernels.basis_funs_all(
            args_derham.tn1,
            pn1,
            e1,
            span1,
            l1,
            r1,
            b1,
            d1,
        )
        bsplines_kernels.basis_funs_all(
            args_derham.tn2,
            pn2,
            e2,
            span2,
            l2,
            r2,
            b2,
            d2,
        )
        bsplines_kernels.basis_funs_all(
            args_derham.tn3,
            pn3,
            e3,
            span3,
            l3,
            r3,
            b3,
            d3,
        )

        # vector potential: 1-form components
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            a1_1,
            a1_2,
            a1_3,
            a_form,
        )

        a_xx[0, 0] = evaluation_kernels_3d.eval_spline_derivative_mpi_kernel(
            args_derham.pn[0] - 2,
            args_derham.pn[1],
            args_derham.pn[2],
            bdd1,
            args_derham.bn2,
            args_derham.bn3,
            span1,
            span2,
            span3,
            a1_1,
            args_derham.starts,
            int(
                1,
            ),
        )
        a_xx[0, 1] = evaluation_kernels_3d.eval_spline_derivative_mpi_kernel(
            args_derham.pn[0] - 1,
            args_derham.pn[1] - 1,
            args_derham.pn[2],
            args_derham.bd1,
            args_derham.bd2,
            args_derham.bn3,
            span1,
            span2,
            span3,
            a1_1,
            args_derham.starts,
            int(
                2,
            ),
        )
        a_xx[0, 2] = evaluation_kernels_3d.eval_spline_derivative_mpi_kernel(
            args_derham.pn[0] - 1,
            args_derham.pn[1],
            args_derham.pn[2] - 1,
            args_derham.bd1,
            args_derham.bn2,
            args_derham.bd3,
            span1,
            span2,
            span3,
            a1_1,
            args_derham.starts,
            int(3),
        )

        a_xx[1, 0] = evaluation_kernels_3d.eval_spline_derivative_mpi_kernel(
            args_derham.pn[0] - 1,
            args_derham.pn[1] - 1,
            args_derham.pn[2],
            args_derham.bd1,
            args_derham.bd2,
            args_derham.bn3,
            span1,
            span2,
            span3,
            a1_2,
            args_derham.starts,
            int(
                1,
            ),
        )
        a_xx[1, 1] = evaluation_kernels_3d.eval_spline_derivative_mpi_kernel(
            args_derham.pn[0],
            args_derham.pn[1] - 2,
            args_derham.pn[2],
            args_derham.bn1,
            bdd2,
            args_derham.bn3,
            span1,
            span2,
            span3,
            a1_2,
            args_derham.starts,
            int(
                2,
            ),
        )
        a_xx[1, 2] = evaluation_kernels_3d.eval_spline_derivative_mpi_kernel(
            args_derham.pn[0],
            args_derham.pn[1] - 1,
            args_derham.pn[2] - 1,
            args_derham.bn1,
            args_derham.bd2,
            args_derham.bd3,
            span1,
            span2,
            span3,
            a1_2,
            args_derham.starts,
            int(3),
        )

        a_xx[2, 0] = evaluation_kernels_3d.eval_spline_derivative_mpi_kernel(
            args_derham.pn[0] - 1,
            args_derham.pn[1],
            args_derham.pn[2] - 1,
            args_derham.bd1,
            args_derham.bn2,
            args_derham.bd3,
            span1,
            span2,
            span3,
            a1_3,
            args_derham.starts,
            int(1),
        )
        a_xx[2, 1] = evaluation_kernels_3d.eval_spline_derivative_mpi_kernel(
            args_derham.pn[0],
            args_derham.pn[1] - 1,
            args_derham.pn[2] - 1,
            args_derham.bn1,
            args_derham.bd2,
            args_derham.bd3,
            span1,
            span2,
            span3,
            a1_3,
            args_derham.starts,
            int(2),
        )
        a_xx[2, 2] = evaluation_kernels_3d.eval_spline_derivative_mpi_kernel(
            args_derham.pn[0],
            args_derham.pn[1],
            args_derham.pn[2] - 2,
            args_derham.bn1,
            args_derham.bn2,
            bdd3,
            span1,
            span2,
            span3,
            a1_3,
            args_derham.starts,
            int(3),
        )

        linalg_kernels.transpose(a_xx, a_xxtrans)
        linalg_kernels.matrix_matrix(a_xxtrans, dfinv, matrixp)
        linalg_kernels.matrix_matrix(dfinv_t, matrixp, matrixpp)  # left matrix
        linalg_kernels.matrix_matrix(
            matrixpp,
            dfinv_t,
            matrixppp,
        )  # right matrix

        lhs[0, 0] = 1.0 - dt * matrixpp[0, 0]
        lhs[0, 1] = -dt * matrixpp[0, 1]
        lhs[0, 2] = -dt * matrixpp[0, 2]

        lhs[1, 0] = -dt * matrixpp[0, 0]
        lhs[1, 1] = 1.0 - dt * matrixpp[1, 1]
        lhs[1, 2] = -dt * matrixpp[1, 2]

        lhs[2, 0] = -dt * matrixpp[2, 0]
        lhs[2, 1] = -dt * matrixpp[2, 1]
        lhs[2, 2] = 1.0 - dt * matrixpp[2, 2]

        linalg_kernels.matrix_vector(matrixppp, a_form, rhs)
        rhs[0] = v[0] - dt * rhs[0]
        rhs[1] = v[1] - dt * rhs[1]
        rhs[2] = v[2] - dt * rhs[2]

        linalg_kernels.matrix_inv(lhs, lhsinv)
        # update velocity
        markers[ip, 3] = lhsinv[0, 0] * rhs[0] + lhsinv[0, 1] * rhs[1] + lhsinv[0, 2] * rhs[2]
        markers[ip, 4] = lhsinv[1, 0] * rhs[0] + lhsinv[1, 1] * rhs[1] + lhsinv[1, 2] * rhs[2]
        markers[ip, 5] = lhsinv[2, 0] * rhs[0] + lhsinv[2, 1] * rhs[1] + lhsinv[2, 2] * rhs[2]

        # update position
        linalg_kernels.matrix_vector(dfinv_t, a_form, rhs)
        rhs[0] = markers[ip, 3] - rhs[0]
        rhs[1] = markers[ip, 4] - rhs[1]
        rhs[2] = markers[ip, 5] - rhs[2]

        markers[ip, 0] = e1 + dt * (dfinv[0, 0] * rhs[0] + dfinv[0, 1] * rhs[1] + dfinv[0, 2] * rhs[2])
        markers[ip, 1] = e2 + dt * (dfinv[1, 0] * rhs[0] + dfinv[1, 1] * rhs[1] + dfinv[1, 2] * rhs[2])
        markers[ip, 2] = e3 + dt * (dfinv[2, 0] * rhs[0] + dfinv[2, 1] * rhs[1] + dfinv[2, 2] * rhs[2])

    # -- removed omp: #$ omp end parallel


@stack_array("dfm", "b_form", "u_form", "b_cart", "u_cart", "e_cart")
def push_bxu_Hdiv(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    b2_1: "float[:,:,:]",
    b2_2: "float[:,:,:]",
    b2_3: "float[:,:,:]",
    u2_1: "float[:,:,:]",
    u2_2: "float[:,:,:]",
    u2_3: "float[:,:,:]",
    boundary_cut: "float",
):
    r"""Updates

    .. math::

        \frac{\mathbf v^{n+1}_p - \mathbf v^n_p}{\Delta t} = DF^{-\top} \left(  \hat{\mathbf B}^2 \times \frac{\hat{\mathbf U}^2}{\sqrt g}  \right)^n_p

    for each marker :math:`p` in markers array, where :math:`\hat{\mathbf U}^2 \in H(\textnormal{div})`.

    Parameters
    ----------
        b2_1, b2_2, b2_3: array[float]
            3d array of FE coeffs of B-field as 2-form.

        u2_1, u2_2, u2_3: array[float]
            3d array of FE coeffs of U-field as 2-form.
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # allocate for field evaluations (2-form and Cartesian components)
    b_form = empty(3, dtype=float)
    u_form = empty(3, dtype=float)

    b_cart = empty(3, dtype=float)
    u_cart = empty(3, dtype=float)

    e_cart = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers

    # -- removed omp: #$ omp parallel private(ip, eta1, eta2, eta3, dfm, det_df, span1, span2, span3, b_form, b_cart, u_form, u_cart, e_cart)
    # -- removed omp: #$ omp for
    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        # boundary cut
        if markers[ip, 0] < boundary_cut or markers[ip, 0] > 1.0 - boundary_cut:
            continue

        # marker data
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta1,
            eta2,
            eta3,
            args_domain,
            dfm,
        )

        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # magnetic field: 2-form components
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            b2_1,
            b2_2,
            b2_3,
            b_form,
        )

        # magnetic field: Cartesian components
        linalg_kernels.matrix_vector(dfm, b_form, b_cart)
        b_cart[:] = b_cart / det_df

        # velocity field: 2-form components
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            u2_1,
            u2_2,
            u2_3,
            u_form,
        )

        linalg_kernels.matrix_vector(dfm, u_form, u_cart)
        u_cart[:] = u_cart / det_df

        # electric field E = B x U
        linalg_kernels.cross(b_cart, u_cart, e_cart)

        # update velocities
        markers[ip, 3:6] += dt * e_cart

    # -- removed omp: #$ omp end parallel


@stack_array("dfm", "dfinv", "dfinv_t", "b_form", "u_form", "b_cart", "u_cart", "e_cart")
def push_bxu_Hcurl(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    b2_1: "float[:,:,:]",
    b2_2: "float[:,:,:]",
    b2_3: "float[:,:,:]",
    u1_1: "float[:,:,:]",
    u1_2: "float[:,:,:]",
    u1_3: "float[:,:,:]",
    boundary_cut: "float",
):
    r"""Updates

    .. math::

        \frac{\mathbf v^{n+1}_p - \mathbf v^n_p}{\Delta t} = DF^{-\top} \left(  \hat{\mathbf B}^2 \times G^{-1}\hat{\mathbf U}^1  \right)^n_p

    for each marker :math:`p` in markers array, where :math:`\hat{\mathbf U}^1 \in H(\textnormal{curl})`.

    Parameters
    ----------
        b2_1, b2_2, b2_3: array[float]
            3d array of FE coeffs of B-field as 2-form.

        u1_1, u1_2, u1_3: array[float]
            3d array of FE coeffs of U-field as 1-form.
    """

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

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers

    # -- removed omp: #$ omp parallel private(ip, eta1, eta2, eta3, dfm, det_df, dfinv, dfinv_t, span1, span2, span3, b_form, b_cart, u_form, u_cart, e_cart)
    # -- removed omp: #$ omp for
    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        # boundary cut
        if markers[ip, 0] < boundary_cut or markers[ip, 0] > 1.0 - boundary_cut:
            continue

        # marker data
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta1,
            eta2,
            eta3,
            args_domain,
            dfm,
        )

        # metric coeffs
        det_df = linalg_kernels.det(dfm)
        linalg_kernels.matrix_inv_with_det(dfm, det_df, dfinv)
        linalg_kernels.transpose(dfinv, dfinv_t)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # magnetic field: 2-form components
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            b2_1,
            b2_2,
            b2_3,
            b_form,
        )

        # magnetic field: Cartesian components
        linalg_kernels.matrix_vector(dfm, b_form, b_cart)
        b_cart[:] = b_cart / det_df

        # velocity field: 1-form components
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            u1_1,
            u1_2,
            u1_3,
            u_form,
        )

        # velocity field: Cartesian components
        linalg_kernels.matrix_vector(dfinv_t, u_form, u_cart)

        # electric field E = B x U
        linalg_kernels.cross(b_cart, u_cart, e_cart)

        # update velocities
        markers[ip, 3:6] += dt * e_cart

    # -- removed omp: #$ omp end parallel


@stack_array("dfm", "b_form", "u_form", "b_cart", "u_cart", "e_cart")
def push_bxu_H1vec(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    b2_1: "float[:,:,:]",
    b2_2: "float[:,:,:]",
    b2_3: "float[:,:,:]",
    uv_1: "float[:,:,:]",
    uv_2: "float[:,:,:]",
    uv_3: "float[:,:,:]",
    boundary_cut: "float",
):
    r"""Updates

    .. math::

        \frac{\mathbf v^{n+1}_p - \mathbf v^n_p}{\Delta t} = DF^{-\top} \left(  \hat{\mathbf B}^2 \times \hat{\mathbf U}  \right)^n_p

    for each marker :math:`p` in markers array, where :math:`\hat{\mathbf U}` is a vector-field (dual to 1-form) in :math:`(H^1)^3`.

    Parameters
    ----------
        b2_1, b2_2, b2_3: array[float]
            3d array of FE coeffs of B-field as 2-form.

        uv_1, uv_2, uv_3: array[float]
            3d array of FE coeffs of U-field as vector field in (H^1)^3.
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # allocate for field evaluations (2-form and Cartesian components)
    b_form = empty(3, dtype=float)
    u_form = empty(3, dtype=float)

    b_cart = empty(3, dtype=float)
    u_cart = empty(3, dtype=float)

    e_cart = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers

    # -- removed omp: #$ omp parallel private(ip, eta1, eta2, eta3, dfm, det_df, span1, span2, span3, b_form, b_cart, u_form, u_cart, e_cart)
    # -- removed omp: #$ omp for
    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        # boundary cut
        if markers[ip, 0] < boundary_cut or markers[ip, 0] > 1.0 - boundary_cut:
            continue

        # marker data
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta1,
            eta2,
            eta3,
            args_domain,
            dfm,
        )

        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # magnetic field: 2-form components
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            b2_1,
            b2_2,
            b2_3,
            b_form,
        )

        # magnetic field: Cartesian components
        linalg_kernels.matrix_vector(dfm, b_form, b_cart)
        b_cart[:] = b_cart / det_df

        # velocity field: vector field components
        eval_vectorfield_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            uv_1,
            uv_2,
            uv_3,
            u_form,
        )

        # velocity field: Cartesian components
        linalg_kernels.matrix_vector(dfm, u_form, u_cart)

        # electric field E = B x U
        linalg_kernels.cross(b_cart, u_cart, e_cart)

        # update velocities
        markers[ip, 3:6] += dt * e_cart

    # -- removed omp: #$ omp end parallel


@stack_array(
    "dfm",
    "dfinv",
    "dfinv_t",
    "b_form",
    "u_form",
    "b_diff",
    "b_cart",
    "u_cart",
    "b_grad",
    "e_cart",
    "der1",
    "der2",
    "der3",
)
def push_bxu_Hdiv_pauli(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    pn1: int,
    pn2: int,
    pn3: int,
    b2_1: "float[:,:,:]",
    b2_2: "float[:,:,:]",
    b2_3: "float[:,:,:]",
    u2_1: "float[:,:,:]",
    u2_2: "float[:,:,:]",
    u2_3: "float[:,:,:]",
    b0: "float[:,:,:]",
    mu: "float[:]",
):
    r"""Updates

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
    """

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

    # allocate spline derivatives
    der1 = empty(pn1 + 1, dtype=float)
    der2 = empty(pn2 + 1, dtype=float)
    der3 = empty(pn3 + 1, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers

    # -- removed omp: #$ omp parallel private(ip, eta1, eta2, eta3, dfm, det_df, dfinv, dfinv_t, span1, span2, span3, der1, der2, der3, b_form, b_cart, b_diff, b_grad, u_form, u_cart, e_cart)
    # -- removed omp: #$ omp for
    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        # marker data
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta1,
            eta2,
            eta3,
            args_domain,
            dfm,
        )

        # metric coeffs
        det_df = linalg_kernels.det(dfm)
        linalg_kernels.matrix_inv_with_det(dfm, det_df, dfinv)
        linalg_kernels.transpose(dfinv, dfinv_t)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        bsplines_kernels.b_der_splines_slim(args_derham.tn1, args_derham.pn[0], eta1, span1, args_derham.bn1, der1)
        bsplines_kernels.b_der_splines_slim(args_derham.tn2, args_derham.pn[1], eta2, span2, args_derham.bn2, der2)
        bsplines_kernels.b_der_splines_slim(args_derham.tn3, args_derham.pn[2], eta3, span3, args_derham.bn3, der3)

        # magnetic field: 2-form components
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            b2_1,
            b2_2,
            b2_3,
            b_form,
        )

        # magnetic field: Cartesian components
        linalg_kernels.matrix_vector(dfm, b_form, b_cart)
        b_cart[:] = b_cart / det_df

        # magnetic field: evaluation of gradient (vector field)
        b_diff[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            args_derham.pn[0],
            args_derham.pn[1],
            args_derham.pn[2],
            der1,
            args_derham.bn2,
            args_derham.bn3,
            span1,
            span2,
            span3,
            b0,
            args_derham.starts,
        )
        b_diff[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            args_derham.pn[0],
            args_derham.pn[1],
            args_derham.pn[2],
            args_derham.bn1,
            der2,
            args_derham.bn3,
            span1,
            span2,
            span3,
            b0,
            args_derham.starts,
        )
        b_diff[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            args_derham.pn[0],
            args_derham.pn[1],
            args_derham.pn[2],
            args_derham.bn1,
            args_derham.bn2,
            der3,
            span1,
            span2,
            span3,
            b0,
            args_derham.starts,
        )

        # magnetic field: evaluation of gradient (Cartesian components)
        linalg_kernels.matrix_vector(dfinv_t, b_diff, b_grad)

        # velocity field: 2-form components
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            u2_1,
            u2_2,
            u2_3,
            u_form,
        )

        linalg_kernels.matrix_vector(dfm, u_form, u_cart)
        u_cart[:] = u_cart / det_df

        # electric field E = B x U
        linalg_kernels.cross(b_cart, u_cart, e_cart)

        # additional artificial electric field of Pauli markers
        e_cart[:] = e_cart - mu[ip] * b_grad

        # update velocities
        markers[ip, 3:6] += dt * e_cart

    # -- removed omp: #$ omp end parallel


def push_pc_GXu_full(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    GXu_11: "float[:,:,:]",
    GXu_12: "float[:,:,:]",
    GXu_13: "float[:,:,:]",
    GXu_21: "float[:,:,:]",
    GXu_22: "float[:,:,:]",
    GXu_23: "float[:,:,:]",
    GXu_31: "float[:,:,:]",
    GXu_32: "float[:,:,:]",
    GXu_33: "float[:,:,:]",
    boundary_cut: "float",
):
    r"""Updates

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
    """

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

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        # boundary cut
        if markers[ip, 0] < boundary_cut or markers[ip, 0] > 1.0 - boundary_cut:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        v[:] = markers[ip, 3:6]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta1,
            eta2,
            eta3,
            args_domain,
            dfm,
        )

        # metric coeffs
        linalg_kernels.matrix_inv(dfm, dfinv)
        linalg_kernels.transpose(dfinv, dfinv_t)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # Evaluate grad(X(u, v)) at the particle positions
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            GXu_11,
            GXu_12,
            GXu_13,
            GXu[0, :],
        )

        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            GXu_21,
            GXu_22,
            GXu_23,
            GXu[1, :],
        )

        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            GXu_31,
            GXu_32,
            GXu_33,
            GXu[2, :],
        )

        e[0] = GXu[0, 0] * v[0] + GXu[1, 0] * v[1] + GXu[2, 0] * v[2]
        e[1] = GXu[0, 1] * v[0] + GXu[1, 1] * v[1] + GXu[2, 1] * v[2]
        e[2] = GXu[0, 2] * v[0] + GXu[1, 2] * v[1] + GXu[2, 2] * v[2]

        linalg_kernels.matrix_vector(dfinv_t, e, e_cart)

        # update velocities
        markers[ip, 3:6] -= dt * e_cart / 2.0


def push_pc_GXu(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    GXu_11: "float[:,:,:]",
    GXu_12: "float[:,:,:]",
    GXu_13: "float[:,:,:]",
    GXu_21: "float[:,:,:]",
    GXu_22: "float[:,:,:]",
    GXu_23: "float[:,:,:]",
    GXu_31: "float[:,:,:]",
    GXu_32: "float[:,:,:]",
    GXu_33: "float[:,:,:]",
    boundary_cut: "float",
):
    r"""Updates

    .. math::

        \frac{\mathbf v^{n+1}_p - \mathbf v^n_p}{\Delta t} = - DF^{-\top} \left(  \boldsymbol \Lambda^1 \mathbb G \mathcal X(\mathbf u, \mathbf v)  \right)^n_p

    for each marker :math:`p` in markers array, where :math:`\mathbf u`
    are the coefficients of the mhd velocity field (either 1-form or 2-form) and :math:`\mathcal X`
    is either the MHD operator :meth:`struphy.feec.basis_projection_ops.MHDOperators.assemble_X1` (if u is 1-form)
    or :meth:`struphy.feec.basis_projection_ops.MHDOperators.assemble_X2` (if u is 2-form).

    Parameters
    ----------
    grad_Xu_ij : array[float]
        3d array of FE coeffs of :math:`\nabla_j(\mathcal X \cdot \mathbf u)_i`. i,j=1,2,3.
    """

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

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        # boundary cut
        if markers[ip, 0] < boundary_cut or markers[ip, 0] > 1.0 - boundary_cut:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        v[:] = markers[ip, 3:6]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta1,
            eta2,
            eta3,
            args_domain,
            dfm,
        )

        # metric coeffs
        linalg_kernels.matrix_inv(dfm, dfinv)
        linalg_kernels.transpose(dfinv, dfinv_t)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # Evaluate grad(X(u, v)) at the particle positions
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            GXu_11,
            GXu_12,
            GXu_13,
            GXu[0, :],
        )

        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            GXu_21,
            GXu_22,
            GXu_23,
            GXu[1, :],
        )

        e[0] = GXu[0, 0] * v[0] + GXu[1, 0] * v[1]
        e[1] = GXu[0, 1] * v[0] + GXu[1, 1] * v[1]
        e[2] = GXu[0, 2] * v[0] + GXu[1, 2] * v[1]

        linalg_kernels.matrix_vector(dfinv_t, e, e_cart)

        # update velocities
        markers[ip, 3:6] -= dt * e_cart / 2.0


@stack_array("dfm", "dfinv", "v", "k")
def push_eta_stage(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    a: "float[:]",
    b: "float[:]",
    c: "float[:]",
):
    r"""Single stage of a s-stage Runge-Kutta solve of

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant.
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)

    # marker position e and velocity v
    v = empty(3, dtype=float)

    # intermediate k-vector
    k = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    first_init_idx = args_markers.first_init_idx
    first_free_idx = args_markers.first_free_idx

    # get number of stages
    n_stages = shape(b)[0]

    if stage == n_stages - 1:
        last = 1.0
    else:
        last = 0.0

    # -- removed omp: #$ omp parallel private(ip, v, dfm, dfinv, k)
    # -- removed omp: #$ omp for
    for ip in range(n_markers):
        # check if marker is a hole or a boundary particle
        if markers[ip, first_init_idx] == -1.0 or markers[ip, -1] == -2.0:
            continue

        e1 = markers[ip, 0]
        e2 = markers[ip, 1]
        e3 = markers[ip, 2]
        v[:] = markers[ip, 3:6]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            e1,
            e2,
            e3,
            args_domain,
            dfm,
        )

        # evaluate inverse Jacobian matrix
        linalg_kernels.matrix_inv(dfm, dfinv)

        # pull-back of velocity
        linalg_kernels.matrix_vector(dfinv, v, k)

        # accumulation for last stage
        markers[ip, first_free_idx : first_free_idx + 3] += dt * b[stage] * k

        # update positions for intermediate stages or last stage
        markers[ip, 0:3] = (
            markers[ip, first_init_idx : first_init_idx + 3]
            + dt * a[stage] * k
            + last * markers[ip, first_free_idx : first_free_idx + 3]
        )

    # -- removed omp: #$ omp end parallel


@stack_array("dfm", "dfinv", "dfinv_t", "ginv", "v", "u", "k", "k_v", "k_u")
def push_pc_eta_rk4_Hcurl_full(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    u_1: "float[:,:,:]",
    u_2: "float[:,:,:]",
    u_3: "float[:,:,:]",
):
    r"""Fourth order Runge-Kutta solve of

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
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)
    ginv = empty((3, 3), dtype=float)

    # marker velocity
    v = empty(3, dtype=float)

    # U-fiels
    u = empty(3, dtype=float)

    # intermediate stages in RK4
    k = empty(3, dtype=float)
    k_v = empty(3, dtype=float)
    k_u = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    first_init_idx = args_markers.first_init_idx
    first_free_idx = args_markers.first_free_idx

    # assign factor of k for each stage
    if stage == 0 or stage == 3:
        nk = 1.0
    else:
        nk = 2.0

    # which stage
    if stage == 3:
        last = 1.0
        cont = 0.0
    elif stage == 2:
        last = 0.0
        cont = 2.0
    else:
        last = 0.0
        cont = 1.0

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        e1 = markers[ip, 0]
        e2 = markers[ip, 1]
        e3 = markers[ip, 2]
        v[:] = markers[ip, 3:6]

        # ----------------- stage n in Runge-Kutta method -------------------
        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            e1,
            e2,
            e3,
            args_domain,
            dfm,
        )

        # metric coeffs
        linalg_kernels.matrix_inv(dfm, dfinv)
        linalg_kernels.transpose(dfinv, dfinv_t)
        linalg_kernels.matrix_matrix(dfinv, dfinv_t, ginv)

        # pull-back of velocity
        linalg_kernels.matrix_vector(dfinv, v, k_v)

        # spline evaluation
        span1, span2, span3 = get_spans(e1, e2, e3, args_derham)

        # U-field
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            u_1,
            u_2,
            u_3,
            u,
        )

        # transform to vector field
        linalg_kernels.matrix_vector(ginv, u, k_u)

        # sum contribs
        k[:] = k_v + k_u

        # accum k
        markers[ip, first_free_idx : first_free_idx + 3] += k * nk / 6.0

        # update markers for the next stage
        markers[ip, 0:3] = (
            markers[ip, first_init_idx : first_init_idx + 3]
            + dt * k / 2 * cont
            + dt * markers[ip, first_free_idx : first_free_idx + 3] * last
        )


@stack_array("dfm", "dfinv", "dfinv_t", "ginv", "v", "u", "k", "k_v", "k_u")
def push_pc_eta_rk4_Hdiv_full(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    u_1: "float[:,:,:]",
    u_2: "float[:,:,:]",
    u_3: "float[:,:,:]",
):
    r"""Fourth order Runge-Kutta solve of

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
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)
    ginv = empty((3, 3), dtype=float)

    # marker velocity
    v = empty(3, dtype=float)

    # U-fiels
    u = empty(3, dtype=float)

    # intermediate stages in RK4
    k = empty(3, dtype=float)
    k_v = empty(3, dtype=float)
    k_u = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    first_init_idx = args_markers.first_init_idx
    first_free_idx = args_markers.first_free_idx

    # assign factor of k for each stage
    if stage == 0 or stage == 3:
        nk = 1.0
    else:
        nk = 2.0

    # is it the last stage?
    if stage == 3:
        last = 1.0
        cont = 0.0
    else:
        last = 0.0
        cont = 1.0

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        e1 = markers[ip, 0]
        e2 = markers[ip, 1]
        e3 = markers[ip, 2]
        v[:] = markers[ip, 3:6]

        # ----------------- stage n in Runge-Kutta method -------------------
        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            e1,
            e2,
            e3,
            args_domain,
            dfm,
        )

        # metric coeffs
        det_df = linalg_kernels.det(dfm)
        linalg_kernels.matrix_inv(dfm, dfinv)
        linalg_kernels.transpose(dfinv, dfinv_t)
        linalg_kernels.matrix_matrix(dfinv, dfinv_t, ginv)

        # pull-back of velocity
        linalg_kernels.matrix_vector(dfinv, v, k_v)

        # spline evaluation
        span1, span2, span3 = get_spans(e1, e2, e3, args_derham)

        # U-field
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            u_1,
            u_2,
            u_3,
            u,
        )

        # transform to vector field
        k_u[:] = u / det_df

        # sum contribs
        k[:] = k_v + k_u

        # accum k
        markers[ip, first_free_idx : first_free_idx + 3] += k * nk / 6.0

        # update markers for the next stage
        markers[ip, 0:3] = (
            markers[ip, first_init_idx : first_init_idx + 3]
            + dt * k / 2 * cont
            + dt * markers[ip, first_free_idx : first_free_idx + 3] * last
        )


@stack_array("dfm", "dfinv", "dfinv_t", "ginv", "v", "u", "k", "k_v")
def push_pc_eta_rk4_H1vec_full(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    u_1: "float[:,:,:]",
    u_2: "float[:,:,:]",
    u_3: "float[:,:,:]",
):
    r"""Fourth order Runge-Kutta solve of

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v + \textnormal{vec}( \hat{\mathbf U}^{1(2)})

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant and

    .. math::

        \textnormal{vec}( \hat{\mathbf U}^{1}) = G^{-1}\hat{\mathbf U}^{1}\,,\qquad \textnormal{vec}( \hat{\mathbf U}^{2}) = \frac{\hat{\mathbf U}^{2}}{\sqrt g}\,.

    Parameters
    ----------
    u_1, u_2, u_3 : array[float]
        3d array of FE coeffs of U-field, either as 1-form or as 2-form.

    u_basis : int
        U is 1-form (u_basis=1) or a 2-form (u_basis=2).
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)
    ginv = empty((3, 3), dtype=float)

    # marker and velocity
    v = empty(3, dtype=float)

    # U-fiels
    u = empty(3, dtype=float)

    # intermediate stages in RK4
    k = empty(3, dtype=float)
    k_v = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    first_init_idx = args_markers.first_init_idx
    first_free_idx = args_markers.first_free_idx

    # assign factor of k for each stage
    if stage == 0 or stage == 3:
        nk = 1.0
    else:
        nk = 2.0

    # which stage
    if stage == 3:
        last = 1.0
        cont = 0.0
    elif stage == 2:
        last = 0.0
        cont = 2.0
    else:
        last = 0.0
        cont = 1.0

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        e1 = markers[ip, 0]
        e2 = markers[ip, 1]
        e3 = markers[ip, 2]
        v[:] = markers[ip, 3:6]

        # ----------------- stage n in Runge-Kutta method -------------------
        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            e1,
            e2,
            e3,
            args_domain,
            dfm,
        )

        # metric coeffs
        linalg_kernels.matrix_inv(dfm, dfinv)
        linalg_kernels.transpose(dfinv, dfinv_t)
        linalg_kernels.matrix_matrix(dfinv, dfinv_t, ginv)

        # pull-back of velocity
        linalg_kernels.matrix_vector(dfinv, v, k_v)

        # spline evaluation
        span1, span2, span3 = get_spans(e1, e2, e3, args_derham)

        # U-field
        eval_vectorfield_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            u_1,
            u_2,
            u_3,
            u,
        )

        # sum contribs
        k[:] = k_v + u

        # accum k
        markers[ip, first_free_idx : first_free_idx + 3] += k * nk / 6.0

        # update markers for the next stage
        markers[ip, 0:3] = (
            markers[ip, first_init_idx : first_init_idx + 3]
            + dt * k / 2 * cont
            + dt * markers[ip, first_free_idx : first_free_idx + 3] * last
        )


@stack_array("dfm", "dfinv", "dfinv_t", "ginv", "v", "u", "k", "k_v", "k_u")
def push_pc_eta_rk4_Hcurl(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    u_1: "float[:,:,:]",
    u_2: "float[:,:,:]",
    u_3: "float[:,:,:]",
):
    r"""Fourth order Runge-Kutta solve of

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v + \textnormal{vec}( \hat{\mathbf U}^{1(2)})

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant and

    .. math::

        \textnormal{vec}( \hat{\mathbf U}^{1}) = G^{-1}\hat{\mathbf U}^{1}\,,\qquad \textnormal{vec}( \hat{\mathbf U}^{2}) = \frac{\hat{\mathbf U}^{2}}{\sqrt g}\,.

    Parameters
    ----------
    u_1, u_2, u_3 : array[float]
        3d array of FE coeffs of U-field, either as 1-form or as 2-form.

    u_basis : int
        U is 1-form (u_basis=1) or a 2-form (u_basis=2).
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)
    ginv = empty((3, 3), dtype=float)

    # marker velocity
    v = empty(3, dtype=float)

    # U-fiels
    u = empty(3, dtype=float)

    # intermediate stages in RK4
    k = empty(3, dtype=float)
    k_v = empty(3, dtype=float)
    k_u = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    first_init_idx = args_markers.first_init_idx
    first_free_idx = args_markers.first_free_idx

    # assign factor of k for each stage
    if stage == 0 or stage == 3:
        nk = 1.0
    else:
        nk = 2.0

    # which stage
    if stage == 3:
        last = 1.0
        cont = 0.0
    elif stage == 2:
        last = 0.0
        cont = 2.0
    else:
        last = 0.0
        cont = 1.0

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        e1 = markers[ip, 0]
        e2 = markers[ip, 1]
        e3 = markers[ip, 2]
        v[:] = markers[ip, 3:6]

        # ----------------- stage n in Runge-Kutta method -------------------
        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            e1,
            e2,
            e3,
            args_domain,
            dfm,
        )

        # metric coeffs
        linalg_kernels.matrix_inv(dfm, dfinv)
        linalg_kernels.transpose(dfinv, dfinv_t)
        linalg_kernels.matrix_matrix(dfinv, dfinv_t, ginv)

        # pull-back of velocity
        linalg_kernels.matrix_vector(dfinv, v, k_v)

        # spline evaluation
        span1, span2, span3 = get_spans(e1, e2, e3, args_derham)

        # U-field
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            u_1,
            u_2,
            u_3,
            u,
        )
        u[2] = 0.0

        # transform to vector field
        linalg_kernels.matrix_vector(ginv, u, k_u)

        # sum contribs
        k[:] = k_v + k_u

        # accum k
        markers[ip, first_free_idx : first_free_idx + 3] += k * nk / 6.0

        # update markers for the next stage
        markers[ip, 0:3] = (
            markers[ip, first_init_idx : first_init_idx + 3]
            + dt * k / 2 * cont
            + dt * markers[ip, first_free_idx : first_free_idx + 3] * last
        )


@stack_array("dfm", "dfinv", "dfinv_t", "ginv", "v", "u", "k", "k_v", "k_u")
def push_pc_eta_rk4_Hdiv(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    u_1: "float[:,:,:]",
    u_2: "float[:,:,:]",
    u_3: "float[:,:,:]",
):
    r"""Fourth order Runge-Kutta solve of

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v + \textnormal{vec}( \hat{\mathbf U}^{1(2)})

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant and

    .. math::

        \textnormal{vec}( \hat{\mathbf U}^{1}) = G^{-1}\hat{\mathbf U}^{1}\,,\qquad \textnormal{vec}( \hat{\mathbf U}^{2}) = \frac{\hat{\mathbf U}^{2}}{\sqrt g}\,.

    Parameters
    ----------
    u_1, u_2, u_3 : array[float]
        3d array of FE coeffs of U-field, either as 1-form or as 2-form.

    u_basis : int
        U is 1-form (u_basis=1) or a 2-form (u_basis=2).
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)
    ginv = empty((3, 3), dtype=float)

    # marker velocity
    v = empty(3, dtype=float)

    # U-fiels
    u = empty(3, dtype=float)

    # intermediate stages in RK4
    k = empty(3, dtype=float)
    k_v = empty(3, dtype=float)
    k_u = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    first_init_idx = args_markers.first_init_idx
    first_free_idx = args_markers.first_free_idx

    # assign factor of k for each stage
    if stage == 0 or stage == 3:
        nk = 1.0
    else:
        nk = 2.0

    # is it the last stage?
    if stage == 3:
        last = 1.0
        cont = 0.0
    else:
        last = 0.0
        cont = 1.0

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        e1 = markers[ip, 0]
        e2 = markers[ip, 1]
        e3 = markers[ip, 2]
        v[:] = markers[ip, 3:6]

        # ----------------- stage n in Runge-Kutta method -------------------
        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            e1,
            e2,
            e3,
            args_domain,
            dfm,
        )

        # metric coeffs
        det_df = linalg_kernels.det(dfm)
        linalg_kernels.matrix_inv(dfm, dfinv)
        linalg_kernels.transpose(dfinv, dfinv_t)
        linalg_kernels.matrix_matrix(dfinv, dfinv_t, ginv)

        # pull-back of velocity
        linalg_kernels.matrix_vector(dfinv, v, k_v)

        # spline evaluation
        span1, span2, span3 = get_spans(e1, e2, e3, args_derham)

        # U-field
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            u_1,
            u_2,
            u_3,
            u,
        )
        u[2] = 0.0

        # transform to vector field
        k_u[:] = u / det_df

        # sum contribs
        k[:] = k_v + k_u

        # accum k
        markers[ip, first_free_idx : first_free_idx + 3] += k * nk / 6.0

        # update markers for the next stage
        markers[ip, 0:3] = (
            markers[ip, first_init_idx : first_init_idx + 3]
            + dt * k / 2 * cont
            + dt * markers[ip, first_free_idx : first_free_idx + 3] * last
        )


@stack_array("dfm", "dfinv", "dfinv_t", "ginv", "v", "u", "k", "k_v")
def push_pc_eta_rk4_H1vec(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    u_1: "float[:,:,:]",
    u_2: "float[:,:,:]",
    u_3: "float[:,:,:]",
):
    r"""Fourth order Runge-Kutta solve of

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v + \textnormal{vec}( \hat{\mathbf U}^{1(2)})

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant and

    .. math::

        \textnormal{vec}( \hat{\mathbf U}^{1}) = G^{-1}\hat{\mathbf U}^{1}\,,\qquad \textnormal{vec}( \hat{\mathbf U}^{2}) = \frac{\hat{\mathbf U}^{2}}{\sqrt g}\,.

    Parameters
    ----------
    u_1, u_2, u_3 : array[float]
        3d array of FE coeffs of U-field, either as 1-form or as 2-form.

    u_basis : int
        U is 1-form (u_basis=1) or a 2-form (u_basis=2).
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)
    ginv = empty((3, 3), dtype=float)

    # marker velocity
    v = empty(3, dtype=float)

    # U-fiels
    u = empty(3, dtype=float)

    # intermediate stages in RK4
    k = empty(3, dtype=float)
    k_v = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    first_init_idx = args_markers.first_init_idx
    first_free_idx = args_markers.first_free_idx

    # assign factor of k for each stage
    if stage == 0 or stage == 3:
        nk = 1.0
    else:
        nk = 2.0

    # which stage
    if stage == 3:
        last = 1.0
        cont = 0.0
    elif stage == 2:
        last = 0.0
        cont = 2.0
    else:
        last = 0.0
        cont = 1.0

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        e1 = markers[ip, 0]
        e2 = markers[ip, 1]
        e3 = markers[ip, 2]
        v[:] = markers[ip, 3:6]

        # ----------------- stage n in Runge-Kutta method -------------------
        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            e1,
            e2,
            e3,
            args_domain,
            dfm,
        )

        # metric coeffs
        linalg_kernels.matrix_inv(dfm, dfinv)
        linalg_kernels.transpose(dfinv, dfinv_t)
        linalg_kernels.matrix_matrix(dfinv, dfinv_t, ginv)

        # pull-back of velocity
        linalg_kernels.matrix_vector(dfinv, v, k_v)

        # spline evaluation
        span1, span2, span3 = get_spans(e1, e2, e3, args_derham)

        # U-field
        eval_vectorfield_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            u_1,
            u_2,
            u_3,
            u,
        )
        u[2] = 0.0

        # sum contribs
        k[:] = k_v + u

        # accum k
        markers[ip, first_free_idx : first_free_idx + 3] += k * nk / 6.0

        # update markers for the next stage
        markers[ip, 0:3] = (
            markers[ip, first_init_idx : first_init_idx + 3]
            + dt * k / 2 * cont
            + dt * markers[ip, first_free_idx : first_free_idx + 3] * last
        )


@stack_array("dfm", "df_inv", "v", "df_inv_v", "e_vec")
def push_weights_with_efield_lin_va(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    e1_1: "float[:,:,:]",
    e1_2: "float[:,:,:]",
    e1_3: "float[:,:,:]",
    f0_values: "float[:]",
    kappa: "float",
    vth: "float",
):
    r"""
    updates the single weights in the e_W substep of the linear Vlasov Ampère system with delta-f;
    c.f. :class:`~struphy.propagators.propagators_coupling.EfieldWeights`.

    Parameters
    ----------
    e1_1, e1_2, e1_3 : array[float]
        3d array of FE coeffs of E-field as 1-form.

    f0_values : array[float]
        Value of f0 for each particle.

    kappa : float
        = 2 * pi * Omega_c / omega ; Parameter determining the coupling strength between particles and fields
    """

    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    v = empty(3, dtype=float)
    df_inv_v = empty(3, dtype=float)

    e_vec = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    valid_mks = args_markers.valid_mks

    # -- removed omp: #$ omp parallel private (ip, eta1, eta2, eta3, dfm, df_inv, v, df_inv_v, span1, span2, span3, e_vec, update)
    # -- removed omp: #$ omp for
    for ip in range(n_markers):
        if markers[ip, 0] == -1.0 or markers[ip, -1] == -2.0:
            continue

        # position
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # get velocity
        v[0] = markers[ip, 3]
        v[1] = markers[ip, 4]
        v[2] = markers[ip, 5]

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # Compute Jacobian matrix
        evaluation_kernels.df(
            eta1,
            eta2,
            eta3,
            args_domain,
            dfm,
        )

        # invert Jacobian matrix
        linalg_kernels.matrix_inv(dfm, df_inv)

        # compute DF^{-1} v
        linalg_kernels.matrix_vector(df_inv, v, df_inv_v)

        # E-field (1-form)
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            e1_1,
            e1_2,
            e1_3,
            e_vec,
        )

        # w_{n+1} = w_n + dt / (2 * s_0) * sqrt(f_0) * ( DF^{-1} \V_th * v_p ) \cdot ( e_{n+1} + e_n )
        update = (
            (df_inv_v[0] * e_vec[0] + df_inv_v[1] * e_vec[1] + df_inv_v[2] * e_vec[2])
            * f0_values[ip]
            * kappa
            * dt
            / (2 * markers[ip, 7] * vth**2)
        )
        markers[ip, 6] += update

    # -- removed omp: #$ omp end parallel


@stack_array("ginv", "k", "tmp", "pi_du_value")
def push_deterministic_diffusion_stage(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    pi_u: "float[:,:,:]",
    pi_grad_u1: "float[:,:,:]",
    pi_grad_u2: "float[:,:,:]",
    pi_grad_u3: "float[:,:,:]",
    diffusion_coeff: float,
    a: "float[:]",
    b: "float[:]",
    c: "float[:]",
):
    r"""Single stage of a s-stage Runge-Kutta solve of

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = - D \,G^{-1}(\boldsymbol \eta_p(t)) \frac{\nabla \hat u^0}{\hat u^0}(\boldsymbol \eta_p(t))

    for each marker :math:`p` in markers array, where :math:`\frac{\nabla \hat u^0}{\hat u^0}` is constant in time. :math:`D>0` is a positive, constant diffusion coefficient.
    """

    # allocate arrays
    tmp1 = zeros((3, 3), dtype=float)
    tmp2 = zeros((3, 3), dtype=float)
    tmp3 = zeros((3, 3), dtype=float)
    ginv = zeros((3, 3), dtype=float)

    # intermediate k-vector
    k = empty(3, dtype=float)
    tmp = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    first_init_idx = args_markers.first_init_idx
    first_free_idx = args_markers.first_free_idx

    # get number of stages
    n_stages = shape(b)[0]

    if stage == n_stages - 1:
        last = 1.0
    else:
        last = 0.0

    pi_du_value = empty(3, dtype=float)

    # -- removed omp: #$ omp parallel private(ip, span1, span2, span3, pi_u_value, pi_du_value, k, tmp, ginv)
    # -- removed omp: #$ omp for
    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        e1 = markers[ip, 0]
        e2 = markers[ip, 1]
        e3 = markers[ip, 2]

        # spline evaluation
        span1, span2, span3 = get_spans(e1, e2, e3, args_derham)

        # density function: 0-form components
        pi_u_value = eval_0form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            pi_u,
        )

        # gradient of the density function: 1-form components
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            pi_grad_u1,
            pi_grad_u2,
            pi_grad_u3,
            pi_du_value,
        )

        # evaluate Metric tensor, result in gm
        evaluation_kernels.g_inv(
            e1,
            e2,
            e3,
            args_domain,
            tmp1,
            tmp2,
            tmp3,
            False,
            ginv,
        )

        # updating k
        tmp = -diffusion_coeff * pi_du_value / pi_u_value
        linalg_kernels.matrix_vector(ginv, tmp, k)

        # accumulation for last stage
        markers[ip, first_free_idx : first_free_idx + 3] += dt * b[stage] * k

        # update positions for intermediate stages or last stage
        markers[ip, 0:3] = (
            markers[ip, first_init_idx : first_init_idx + 3]
            + dt * a[stage] * k
            + last * markers[ip, first_free_idx : first_free_idx + 3]
        )

    # -- removed omp: #$ omp end parallel


def push_random_diffusion_stage(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    noise: "float[:,:]",
    diffusion_coeff: float,
    a: "float[:]",
    b: "float[:]",
    c: "float[:]",
):
    r"""Single stage of a s-stage Runge-Kutta solve of

    .. math::

        {\textnormal d \boldsymbol \eta_p(t)} = \sqrt{2 \, D}\, \textnormal d \boldsymbol B_t\,,

    for each marker :math:`p` in markers array, where :math:`\textnormal d \boldsymbol B_t` is a Brownian Motion and $D$ is a positive diffusion coefficient.
    """

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers

    # get number of stages. +
    # TODO: Multistage to be added later.
    n_stages = shape(b)[0]

    if stage == n_stages - 1:
        last = 1.0
    else:
        last = 0.0

    # -- removed omp: #$ omp parallel private(ip)
    # -- removed omp: #$ omp for
    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        markers[ip, 0:3] += sqrt(2 * dt * diffusion_coeff) * noise[ip, :]

    # -- removed omp: #$ omp end parallel


@stack_array("grad_u", "grad_u_cart", "tmp1", "dfinv", "dfinvT")
def push_v_sph_pressure(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    boxes: "int[:,:]",
    neighbours: "int[:, :]",
    holes: "bool[:]",
    periodic1: "bool",
    periodic2: "bool",
    periodic3: "bool",
    kernel_type: "int",
    h1: "float",
    h2: "float",
    h3: "float",
    gravity: "float[:]",
):
    r"""Updates particle velocities as

    .. math::

        \frac{\mathbf v^{n+1} - \mathbf v^n}{\Delta t} = \kappa_p \sum_{q} w_p\,w_q \left( \frac{1}{\rho^{N,h}(\boldsymbol \eta_p)} + \frac{1}{\rho^{N,h}(\boldsymbol \eta_q)} \right) G^{-1}\nabla W_h(\boldsymbol \eta_p - \boldsymbol \eta_q) \,,

    where :math:`G^{-1}` denotes the inverse metric tensor, and with the smoothed density

    .. math::

        \rho^{N,h}(\boldsymbol \eta_p) = \frac 1N \sum_q w_q \, W_h(\boldsymbol \eta_p - \boldsymbol \eta_q)\,,

    where :math:`W_h(\boldsymbol \eta)` is a smoothing kernel from :mod:`~struphy.pic.sph_smoothing_kernels`.

    Parameters
    ----------
    boxes : 2d array
        Box array of the sorting boxes structure.

    neighbours : 2d array
        Array containing the 27 neighbouring boxes of each box.

    holes : bool
        1D array of length markers.shape[0]. True if markers[i] is a hole.

    periodic1, periodic2, periodic3 : bool
        True if periodic in that dimension.

    kernel_type : int
        Number of the smoothing kernel.

    h1, h2, h3 : float
        Kernel width in respective dimension.

    gravity: np.ndarray
        Constant gravitational force as 3-vector.
    """
    # allocate arrays
    grad_u = zeros(3, dtype=float)
    grad_u_cart = zeros(3, dtype=float)
    tmp1 = zeros((3, 3), dtype=float)
    dfinv = zeros((3, 3), dtype=float)
    dfinvT = zeros((3, 3), dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    Np = args_markers.Np
    weight_idx = args_markers.weight_idx
    first_free_idx = args_markers.first_free_idx
    valid_mks = args_markers.valid_mks
    n_cols = shape(markers)[1]

    # -- removed omp: #$ omp parallel private(ip, eta1, eta2, eta3, dfinv)
    # -- removed omp: #$ omp for
    for ip in range(n_markers):
        if not valid_mks[ip]:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        kappa = 1.0  # markers[ip, first_diagnostics_idx]
        n_at_eta = markers[ip, first_free_idx]
        loc_box = int(markers[ip, n_cols - 2])

        # first component
        grad_u[0] = sph_eval_kernels.boxed_based_kernel(
            args_markers,
            eta1,
            eta2,
            eta3,
            loc_box,
            boxes,
            neighbours,
            holes,
            periodic1,
            periodic2,
            periodic3,
            weight_idx,
            kernel_type + 1,
            h1,
            h2,
            h3,
        )
        grad_u[0] *= kappa / n_at_eta

        sum2 = sph_eval_kernels.boxed_based_kernel(
            args_markers,
            eta1,
            eta2,
            eta3,
            loc_box,
            boxes,
            neighbours,
            holes,
            periodic1,
            periodic2,
            periodic3,
            first_free_idx + 1,
            kernel_type + 1,
            h1,
            h2,
            h3,
        )
        sum2 *= kappa
        grad_u[0] += sum2

        if kernel_type >= 340:
            # second component
            grad_u[1] = sph_eval_kernels.boxed_based_kernel(
                args_markers,
                eta1,
                eta2,
                eta3,
                loc_box,
                boxes,
                neighbours,
                holes,
                periodic1,
                periodic2,
                periodic3,
                weight_idx,
                kernel_type + 2,
                h1,
                h2,
                h3,
            )
            grad_u[1] *= kappa / n_at_eta

            sum4 = sph_eval_kernels.boxed_based_kernel(
                args_markers,
                eta1,
                eta2,
                eta3,
                loc_box,
                boxes,
                neighbours,
                holes,
                periodic1,
                periodic2,
                periodic3,
                first_free_idx + 1,
                kernel_type + 2,
                h1,
                h2,
                h3,
            )
            sum4 *= kappa
            grad_u[1] += sum4

        if kernel_type >= 670:
            # third component
            grad_u[2] = sph_eval_kernels.boxed_based_kernel(
                args_markers,
                eta1,
                eta2,
                eta3,
                loc_box,
                boxes,
                neighbours,
                holes,
                periodic1,
                periodic2,
                periodic3,
                weight_idx,
                kernel_type + 3,
                h1,
                h2,
                h3,
            )
            grad_u[2] *= kappa / n_at_eta

            sum6 = sph_eval_kernels.boxed_based_kernel(
                args_markers,
                eta1,
                eta2,
                eta3,
                loc_box,
                boxes,
                neighbours,
                holes,
                periodic1,
                periodic2,
                periodic3,
                first_free_idx + 1,
                kernel_type + 3,
                h1,
                h2,
                h3,
            )
            sum6 *= kappa
            grad_u[2] += sum6

        # push to Cartesian coordinates
        evaluation_kernels.df_inv(
            eta1,
            eta2,
            eta3,
            args_domain,
            tmp1,
            False,
            dfinv,
        )
        linalg_kernels.transpose(dfinv, dfinvT)
        linalg_kernels.matrix_vector(dfinvT, grad_u, grad_u_cart)

        # update velocities
        markers[ip, 3:6] -= dt * (grad_u_cart - gravity)

    # -- removed omp: #$ omp end parallel


@stack_array("grad_u", "grad_u_cart", "tmp1", "dfinv", "dfinvT")
def push_v_sph_pressure_ideal_gas(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    boxes: "int[:,:]",
    neighbours: "int[:, :]",
    holes: "bool[:]",
    periodic1: "bool",
    periodic2: "bool",
    periodic3: "bool",
    kernel_type: "int",
    h1: "float",
    h2: "float",
    h3: "float",
    gravity: "float[:]",
):
    r"""Updates particle velocities as

    .. math::

        \frac{\mathbf v^{n+1} - \mathbf v^n}{\Delta t} = \kappa_p \sum_{q} w_p\,w_q \left( \frac{1}{\rho^{N,h}(\boldsymbol \eta_p)} + \frac{1}{\rho^{N,h}(\boldsymbol \eta_q)} \right) G^{-1}\nabla W_h(\boldsymbol \eta_p - \boldsymbol \eta_q) \,,

    where :math:`G^{-1}` denotes the inverse metric tensor, and with the smoothed density

    .. math::

        \rho^{N,h}(\boldsymbol \eta_p) = \frac 1N \sum_q w_q \, W_h(\boldsymbol \eta_p - \boldsymbol \eta_q)\,,

    where :math:`W_h(\boldsymbol \eta)` is a smoothing kernel from :mod:`~struphy.pic.sph_smoothing_kernels`.

    Parameters
    ----------
    boxes : 2d array
        Box array of the sorting boxes structure.

    neighbours : 2d array
        Array containing the 27 neighbouring boxes of each box.

    holes : bool
        1D array of length markers.shape[0]. True if markers[i] is a hole.

    periodic1, periodic2, periodic3 : bool
        True if periodic in that dimension.

    kernel_type : int
        Number of the smoothing kernel.

    h1, h2, h3 : float
        Kernel width in respective dimension.

    gravity: np.ndarray
        Constant gravitational force as 3-vector.
    """
    # allocate arrays
    grad_u = zeros(3, dtype=float)
    grad_u_cart = zeros(3, dtype=float)
    tmp1 = zeros((3, 3), dtype=float)
    dfinv = zeros((3, 3), dtype=float)
    dfinvT = zeros((3, 3), dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    Np = args_markers.Np
    weight_idx = args_markers.weight_idx
    first_free_idx = args_markers.first_free_idx
    valid_mks = args_markers.valid_mks
    n_cols = shape(markers)[1]

    gamma = 5 / 3
    kappa = 1 / (gamma - 1)

    # -- removed omp: #$ omp parallel private(ip, eta1, eta2, eta3, dfinv)
    # -- removed omp: #$ omp for
    for ip in range(n_markers):
        if not valid_mks[ip]:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        n_at_eta = markers[ip, first_free_idx]
        loc_box = int(markers[ip, n_cols - 2])

        # first component
        grad_u[0] = sph_eval_kernels.boxed_based_kernel(
            args_markers,
            eta1,
            eta2,
            eta3,
            loc_box,
            boxes,
            neighbours,
            holes,
            periodic1,
            periodic2,
            periodic3,
            weight_idx,
            kernel_type + 1,
            h1,
            h2,
            h3,
        )
        grad_u[0] *= kappa * n_at_eta ** (gamma - 2)

        sum2 = sph_eval_kernels.boxed_based_kernel(
            args_markers,
            eta1,
            eta2,
            eta3,
            loc_box,
            boxes,
            neighbours,
            holes,
            periodic1,
            periodic2,
            periodic3,
            first_free_idx + 2,
            kernel_type + 1,
            h1,
            h2,
            h3,
        )
        sum2 *= kappa
        grad_u[0] += sum2

        if kernel_type >= 340:
            # second component
            grad_u[1] = sph_eval_kernels.boxed_based_kernel(
                args_markers,
                eta1,
                eta2,
                eta3,
                loc_box,
                boxes,
                neighbours,
                holes,
                periodic1,
                periodic2,
                periodic3,
                weight_idx,
                kernel_type + 2,
                h1,
                h2,
                h3,
            )
            grad_u[1] *= kappa * (n_at_eta) ** (gamma - 2)

            sum4 = sph_eval_kernels.boxed_based_kernel(
                args_markers,
                eta1,
                eta2,
                eta3,
                loc_box,
                boxes,
                neighbours,
                holes,
                periodic1,
                periodic2,
                periodic3,
                first_free_idx + 2,
                kernel_type + 2,
                h1,
                h2,
                h3,
            )
            sum4 *= kappa
            grad_u[1] += sum4

        if kernel_type >= 670:
            # third component
            grad_u[2] = sph_eval_kernels.boxed_based_kernel(
                args_markers,
                eta1,
                eta2,
                eta3,
                loc_box,
                boxes,
                neighbours,
                holes,
                periodic1,
                periodic2,
                periodic3,
                weight_idx,
                kernel_type + 3,
                h1,
                h2,
                h3,
            )
            grad_u[2] *= kappa * (n_at_eta) ** (gamma - 2)

            sum6 = sph_eval_kernels.boxed_based_kernel(
                args_markers,
                eta1,
                eta2,
                eta3,
                loc_box,
                boxes,
                neighbours,
                holes,
                periodic1,
                periodic2,
                periodic3,
                first_free_idx + 2,
                kernel_type + 3,
                h1,
                h2,
                h3,
            )
            sum6 *= kappa
            grad_u[2] += sum6

        # push to Cartesian coordinates
        evaluation_kernels.df_inv(
            eta1,
            eta2,
            eta3,
            args_domain,
            tmp1,
            False,
            dfinv,
        )
        linalg_kernels.transpose(dfinv, dfinvT)
        linalg_kernels.matrix_vector(dfinvT, grad_u, grad_u_cart)

        # update velocities
        markers[ip, 3:6] -= dt * (grad_u_cart - gravity)

    # -- removed omp: #$ omp end parallel


@stack_array("grad_u", "grad_u_cart", "tmp1", "dfinv", "dfinvT")
def push_v_viscosity(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    boxes: "int[:,:]",
    neighbours: "int[:, :]",
    holes: "bool[:]",
    periodic1: "bool",
    periodic2: "bool",
    periodic3: "bool",
    kernel_type: "int",
    h1: "float",
    h2: "float",
    h3: "float",
):
    r"""Updates particle velocities as

    .. math::

        \frac{\mathbf v^{n+1} - \mathbf v^n}{\Delta t} = \kappa_p \sum_{q} w_p\,w_q \left( \frac{1}{\rho^{N,h}(\boldsymbol \eta_p)} + \frac{1}{\rho^{N,h}(\boldsymbol \eta_q)} \right) G^{-1}\nabla W_h(\boldsymbol \eta_p - \boldsymbol \eta_q) \,,

    where :math:`G^{-1}` denotes the inverse metric tensor, and with the smoothed density

    .. math::

        \rho^{N,h}(\boldsymbol \eta_p) = \frac 1N \sum_q w_q \, W_h(\boldsymbol \eta_p - \boldsymbol \eta_q)\,,

    where :math:`W_h(\boldsymbol \eta)` is a smoothing kernel from :mod:`~struphy.pic.sph_smoothing_kernels`.

    Parameters
    ----------
    boxes : 2d array
        Box array of the sorting boxes structure.

    neighbours : 2d array
        Array containing the 27 neighbouring boxes of each box.

    holes : bool
        1D array of length markers.shape[0]. True if markers[i] is a hole.

    periodic1, periodic2, periodic3 : bool
        True if periodic in that dimension.

    kernel_type : int
        Number of the smoothing kernel.

    h1, h2, h3 : float
        Kernel width in respective dimension.

    gravity: np.ndarray
        Constant gravitational force as 3-vector.
    """
    # allocate arrays
    grad_u = zeros(3, dtype=float)
    grad_u_cart = zeros(3, dtype=float)
    tmp1 = zeros((3, 3), dtype=float)
    dfinv = zeros((3, 3), dtype=float)
    dfinvT = zeros((3, 3), dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    Np = args_markers.Np
    weight_idx = args_markers.weight_idx
    first_free_idx = args_markers.first_free_idx
    valid_mks = args_markers.valid_mks
    n_cols = shape(markers)[1]
    f_visc = zeros(3, dtype=float)
    f_visc_cart = zeros(3, dtype=float)

    # -- removed omp: #$ omp parallel private(ip, eta1, eta2, eta3, dfinv)
    # -- removed omp: #$ omp for
    for ip in range(n_markers):
        if not valid_mks[ip]:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        kappa = 1.0  # markers[ip, first_diagnostics_idx]
        # n_at_eta = markers[ip, first_free_idx]
        loc_box = int(markers[ip, n_cols - 2])

        for j in range(3):  # row of viscosity tensor
            for k in range(3):  # column = derivative direction
                coeff_idx = first_free_idx + 3 * j + k + 15

                # if k == 0:
                #     deriv_type = kernel_type + 1
                #     use_component = True
                # elif k == 1 and kernel_type >= 340:
                #     deriv_type = kernel_type + 2
                #     use_component = True
                # elif k == 2 and kernel_type >= 670:
                #     deriv_type = kernel_type + 3
                #     use_component = True
                # else:
                #     use_component = False

                # if use_component:
                f_visc[j] += sph_eval_kernels.boxed_based_kernel(
                    args_markers,
                    eta1,
                    eta2,
                    eta3,
                    loc_box,
                    boxes,
                    neighbours,
                    holes,
                    periodic1,
                    periodic2,
                    periodic3,
                    coeff_idx,
                    kernel_type + 1 + k,
                    h1,
                    h2,
                    h3,
                )

        # push to Cartesian coordinates
        evaluation_kernels.df_inv(
            eta1,
            eta2,
            eta3,
            args_domain,
            tmp1,
            False,
            dfinv,
        )
        linalg_kernels.transpose(dfinv, dfinvT)
        linalg_kernels.matrix_vector(dfinvT, f_visc, f_visc_cart)

        # update velocities
        markers[ip, 3:6] -= dt * (f_visc_cart)

    # -- removed omp: #$ omp end parallel
