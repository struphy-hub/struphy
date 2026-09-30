"Pusher kernels for full orbit (6D) particles."

from numpy import cos, empty, shape, sin, sqrt, zeros
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
    valid_mks = args_markers.valid_mks

    # fmt: off
    #$ omp parallel private(ip, eta1, eta2, eta3, dfm, dfinv, dfinvt, span1, span2, span3, e_form, e_cart)
    #$ omp for
    # fmt: on
    for ip in range(n_markers):
        # only do something if particle is valid (i.e. not a hole or ghost)
        if not valid_mks[ip]:
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

    # fmt: off
    #$ omp end parallel
    # fmt: on


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

    # fmt: off
    #$ omp parallel private (ip, e1, e2, e3, v, dfm, det_df, span1, span2, span3, b_form, b_cart, b_abs, b_norm, vpar, vxb_norm, vperp, b_normxvperp)
    #$ omp for
    # fmt: on
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

    # fmt: off
    #$ omp end parallel
    # fmt: on


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

    # fmt: off
    #$ omp parallel firstprivate(b_prod) private (ip, e1, e2, e3, v, dfm, det_df, span1, span2, span3, b_form, b_cart, rhs, lhs, lhs_inv, vec, res)
    #$ omp for
    # fmt: on
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

    # fmt: off
    #$ omp end parallel
    # fmt: on


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

    # fmt: off
    #$ omp parallel private (ip, e1, e2, e3, v, dfm, dfinv, dfinv_t, det_df, span1, span2, span3, b_form, a_form, b_cart, b_abs, b_norm, vpar, vxb_norm, vperp, b_normxvperp, rot_temp)
    #$ omp for
    # fmt: on
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

    # fmt: off
    #$ omp end parallel
    # fmt: on


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

    # fmt: off
    #$ omp parallel private(ip, eta1, eta2, eta3, dfm, det_df, span1, span2, span3, b_form, b_cart, u_form, u_cart, e_cart)
    #$ omp for
    # fmt: on
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

    # fmt: off
    #$ omp end parallel
    # fmt: on


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

    # fmt: off
    #$ omp parallel private(ip, eta1, eta2, eta3, dfm, det_df, dfinv, dfinv_t, span1, span2, span3, b_form, b_cart, u_form, u_cart, e_cart)
    #$ omp for
    # fmt: on
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

    # fmt: off
    #$ omp end parallel
    # fmt: on


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

    # fmt: off
    #$ omp parallel private(ip, eta1, eta2, eta3, dfm, det_df, span1, span2, span3, b_form, b_cart, u_form, u_cart, e_cart)
    #$ omp for
    # fmt: on
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

    # fmt: off
    #$ omp end parallel
    # fmt: on


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

    # fmt: off
    #$ omp parallel private(ip, eta1, eta2, eta3, dfm, det_df, dfinv, dfinv_t, span1, span2, span3, der1, der2, der3, b_form, b_cart, b_diff, b_grad, u_form, u_cart, e_cart)
    #$ omp for
    # fmt: on
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

    # fmt: off
    #$ omp end parallel
    # fmt: on


@stack_array(
    "dfm",
    "dfinv",
    "dfinv_t",
    "e",
    "e_cart",
    "GXu",
    "v",
)
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


@stack_array(
    "dfm",
    "dfinv",
    "dfinv_t",
    "e",
    "e_cart",
    "GXu",
    "v",
)
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

    # particle velocity
    v = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
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

    # fmt: off
    #$ omp parallel private(ip, e1, e2, e3, v, dfm, dfinv, k)
    #$ omp for
    # fmt: on
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

        # apply kinetic boundary conditions
        pusher_utilities_kernels.apply_kinetic_bc_marker(ip, args_markers, args_domain, False)

    # fmt: off
    #$ omp end parallel
    # fmt: on


@stack_array("dfm", "dfinv", "dfinv_t", "ginv", "v", "u", "k", "k_v", "k_u")
def push_pc_eta_stage_Hcurl(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    u_1: "float[:,:,:]",
    u_2: "float[:,:,:]",
    u_3: "float[:,:,:]",
    use_perp_model: "bool",
    a: "float[:]",
    b: "float[:]",
    c: "float[:]",
):
    r"""Fourth order Runge-Kutta solve of

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v + \textnormal{vec}( \hat{\mathbf U}^{1(2)})

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant and

    .. math::

        \textnormal{vec}( \hat{\mathbf U}^{1}) = G^{-1}\hat{\mathbf U}^{1}\,,\qquad \textnormal{vec}( \hat{\mathbf U}^{2}) = \frac{\hat{\mathbf U}^{2}}{\sqrt g}\,.
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

    # get number of stages
    n_stages = shape(b)[0]

    if stage == n_stages - 1:
        last = 1.0
    else:
        last = 0.0

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

        if use_perp_model:
            u[2] = 0.0

        # transform to vector field
        linalg_kernels.matrix_vector(ginv, u, k_u)

        # sum contribs
        k[:] = k_v + k_u

        # accum k
        markers[ip, first_free_idx : first_free_idx + 3] += dt * b[stage] * k

        # update markers for the next stage
        markers[ip, 0:3] = (
            markers[ip, first_init_idx : first_init_idx + 3]
            + dt * k * a[stage]
            + last * markers[ip, first_free_idx : first_free_idx + 3]
        )

        # apply kinetic boundary conditions
        pusher_utilities_kernels.apply_kinetic_bc_marker(ip, args_markers, args_domain, False)


@stack_array("dfm", "dfinv", "dfinv_t", "ginv", "v", "u", "k", "k_v", "k_u")
def push_pc_eta_stage_Hdiv(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    u_1: "float[:,:,:]",
    u_2: "float[:,:,:]",
    u_3: "float[:,:,:]",
    use_perp_model: "bool",
    a: "float[:]",
    b: "float[:]",
    c: "float[:]",
):
    r"""Fourth order Runge-Kutta solve of

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v + \textnormal{vec}( \hat{\mathbf U}^{1(2)})

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant and

    .. math::

        \textnormal{vec}( \hat{\mathbf U}^{1}) = G^{-1}\hat{\mathbf U}^{1}\,,\qquad \textnormal{vec}( \hat{\mathbf U}^{2}) = \frac{\hat{\mathbf U}^{2}}{\sqrt g}\,.
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

    # get number of stages
    n_stages = shape(b)[0]

    if stage == n_stages - 1:
        last = 1.0
    else:
        last = 0.0

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

        if use_perp_model:
            u[2] = 0.0

        # transform to vector field
        k_u[:] = u / det_df

        # sum contribs
        k[:] = k_v + k_u

        # accum k
        markers[ip, first_free_idx : first_free_idx + 3] += dt * b[stage] * k

        # update markers for the next stage
        markers[ip, 0:3] = (
            markers[ip, first_init_idx : first_init_idx + 3]
            + dt * k * a[stage]
            + last * markers[ip, first_free_idx : first_free_idx + 3]
        )

        # apply kinetic boundary conditions
        pusher_utilities_kernels.apply_kinetic_bc_marker(ip, args_markers, args_domain, False)


@stack_array("dfm", "dfinv", "dfinv_t", "ginv", "v", "u", "k", "k_v")
def push_pc_eta_stage_H1vec(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    u_1: "float[:,:,:]",
    u_2: "float[:,:,:]",
    u_3: "float[:,:,:]",
    use_perp_model: "bool",
    a: "float[:]",
    b: "float[:]",
    c: "float[:]",
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

    # get number of stages
    n_stages = shape(b)[0]

    if stage == n_stages - 1:
        last = 1.0
    else:
        last = 0.0

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

        if use_perp_model:
            u[2] = 0.0

        # sum contribs
        k[:] = k_v + u

        # accum k
        markers[ip, first_free_idx : first_free_idx + 3] += dt * b[stage] * k

        # update markers for the next stage
        markers[ip, 0:3] = (
            markers[ip, first_init_idx : first_init_idx + 3]
            + dt * k * a[stage]
            + last * markers[ip, first_free_idx : first_free_idx + 3]
        )

        # apply kinetic boundary conditions
        pusher_utilities_kernels.apply_kinetic_bc_marker(ip, args_markers, args_domain, False)


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

    # fmt: off
    #$ omp parallel private (ip, eta1, eta2, eta3, dfm, df_inv, v, df_inv_v, span1, span2, span3, e_vec, update)
    #$ omp for
    # fmt: on
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

    # fmt: off
    #$ omp end parallel
    # fmt: on


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

    # fmt: off
    #$ omp parallel private(ip, e1, e2, e3, span1, span2, span3, pi_u_value, pi_du_value, k, tmp, tmp1, tmp2, tmp3, ginv)
    #$ omp for
    # fmt: on
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

        # apply kinetic boundary conditions
        pusher_utilities_kernels.apply_kinetic_bc_marker(ip, args_markers, args_domain, False)

    # fmt: off
    #$ omp end parallel
    # fmt: on


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

    The right-hand side is the same Wiener increment
    :math:`\sqrt{2 D \Delta t}\, \boldsymbol \xi_p` in every stage, hence stage ``i``
    adds the fraction ``b[i]`` of it. Since :math:`\sum_i b_i = 1`, one full step adds the
    increment exactly once for any explicit tableau (Euler-Maruyama).
    """

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers

    # fraction of the Wiener increment added in this stage
    scale = b[stage] * sqrt(2 * dt * diffusion_coeff)

    # fmt: off
    #$ omp parallel private(ip)
    #$ omp for
    # fmt: on
    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        markers[ip, 0:3] += scale * noise[ip, :]

        # apply kinetic boundary conditions
        pusher_utilities_kernels.apply_kinetic_bc_marker(ip, args_markers, args_domain, False)

    # fmt: off
    #$ omp end parallel
    # fmt: on
