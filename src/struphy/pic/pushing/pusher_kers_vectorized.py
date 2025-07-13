"Pusher kernels for full orbit (6D) particles and AMReX data structures."

from numpy import array, cos, empty, matmul, newaxis, shape, sin, sqrt, transpose
from numpy.linalg import det, inv

from struphy.bsplines.evaluation_kernels_3d import (
    get_spans_and_eval_1form_spline_vectorized,
    get_spans_and_eval_2form_spline_vectorized,
)
from struphy.geometry.base import Domain
from struphy.linear_algebra.linalg_kernels import cross_vectorized, cross_vectorized_flat, scalar_dot_vectorized_flat
from struphy.pic.base import Particles
from struphy.pic.pushing.pusher_args_kernels import DerhamArguments


def push_vxb_analytic(
    dt: float,
    stage: int,
    particles: "Particles",
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

    for pti in particles.markers.iterator(particles.markers, 0):
        markers_array = particles.get_amrex_markers_array(pti.soa())
        e1 = markers_array["x"]
        e2 = markers_array["y"]
        e3 = markers_array["z"]
        v1 = markers_array["v1"]
        v2 = markers_array["v2"]
        v3 = markers_array["v3"]

        n_p = len(e1)
        b_form = empty((n_p, 3, 1), dtype=float)
        b_abs = empty(n_p, dtype=float)
        b_norm = empty((n_p, 3), dtype=float)

        # evaluate Jacobian
        jacobian = particles.domain.jacobian(e1, e2, e3, change_out_order=True, flat_eval=True)  # Npx3x3

        # metric coeffs
        det_df = det(jacobian)

        # spline evaluation and magnetic field 2-form
        get_spans_and_eval_2form_spline_vectorized(
            e1,
            e2,
            e3,
            args_derham,
            b2_1,
            b2_2,
            b2_3,
            b_form,
        )

        # magnetic field: Cartesian components
        b_cart = matmul(jacobian, b_form)  # Npx3x1
        b_cart = b_cart / det_df[:, newaxis, newaxis]

        # magnetic field: magnitude
        b_abs[:] = sqrt(b_cart[:, 0, 0] ** 2 + b_cart[:, 1, 0] ** 2 + b_cart[:, 2, 0] ** 2)

        # only push vxb if magnetic field is non-zero
        non_zero_idx = b_abs != 0

        # normalized magnetic field direction
        b_norm[non_zero_idx, :] = b_cart[non_zero_idx, :, 0] / b_abs[non_zero_idx, newaxis]

        # parallel velocity v.b_norm
        vpar = scalar_dot_vectorized_flat(v1, v2, v3, b_norm)

        # first component of perpendicular velocity
        vxb_norm = cross_vectorized_flat(v1, v2, v3, b_norm)
        vperp = cross_vectorized(b_norm, vxb_norm)

        # second component of perpendicular velocity
        b_normxvperp = cross_vectorized(b_norm, vperp)

        # analytic rotation
        temp = (
            vpar[:, newaxis] * b_norm + cos(b_abs * dt)[:, newaxis] * vperp - sin(b_abs * dt)[:, newaxis] * b_normxvperp
        )
        v1[:] = temp[:, 0]
        v2[:] = temp[:, 1]
        v3[:] = temp[:, 2]


def push_vxb_implicit(
    dt: float,
    stage: int,
    particles: "Particles",
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

    for pti in particles.markers.iterator(particles.markers, 0):
        markers_array = particles.get_amrex_markers_array(pti.soa())
        e1 = markers_array["x"]
        e2 = markers_array["y"]
        e3 = markers_array["z"]
        v1 = markers_array["v1"]
        v2 = markers_array["v2"]
        v3 = markers_array["v3"]

        n_p = len(e1)
        # allocate for field evaluations (2-form components, Cartesian components and rotation matrix such that vxB = B_prod.v)
        b_form = empty((n_p, 3, 1), dtype=float)
        b_prod = empty((n_p, 3, 3), dtype=float)

        # identity matrix
        identity = empty((n_p, 3, 3), dtype=float)

        identity[:, 0, 0] = 1.0
        identity[:, 1, 1] = 1.0
        identity[:, 2, 2] = 1.0

        # right-hand side and left-hand side of Crank-Nicolson scheme
        rhs = empty((n_p, 3, 3), dtype=float)
        lhs = empty((n_p, 3, 3), dtype=float)

        lhs_inv = empty((n_p, 3, 3), dtype=float)

        vec = empty((n_p, 3), dtype=float)
        res = empty((n_p, 3), dtype=float)

        # evaluate Jacobian
        jacobian = particles.domain.jacobian(e1, e2, e3, change_out_order=True, flat_eval=True)  # Npx3x3

        # metric coeffs
        det_df = det(jacobian)

        # spline evaluation and magnetic field 2-form
        get_spans_and_eval_2form_spline_vectorized(
            e1,
            e2,
            e3,
            args_derham,
            b2_1,
            b2_2,
            b2_3,
            b_form,
        )

        # magnetic field: Cartesian components
        b_cart = matmul(jacobian, b_form)  # Npx3x1
        b_cart = b_cart / det_df[:, newaxis, newaxis]

        # magnetic field: rotation matrix
        b_prod[:, 0, 1] = b_cart[:, 2, 0]
        b_prod[:, 0, 2] = -b_cart[:, 1, 0]

        b_prod[:, 1, 0] = -b_cart[:, 2, 0]
        b_prod[:, 1, 2] = b_cart[:, 0, 0]

        b_prod[:, 2, 0] = b_cart[:, 1, 0]
        b_prod[:, 2, 1] = -b_cart[:, 0, 0]

        # solve 3x3 system
        rhs[:, :, :] = identity + dt / 2 * b_prod
        lhs[:, :, :] = identity - dt / 2 * b_prod

        lhs_inv = inv(lhs)

        v = array([v1, v2, v3]).T
        v = v[..., newaxis]  # Npx3x1

        vec = matmul(rhs, v)
        res = matmul(lhs_inv, vec)

        v1[:] = res[:, 0, 0]
        v2[:] = res[:, 1, 0]
        v3[:] = res[:, 2, 0]


def push_v_with_efield(
    dt: float,
    stage: int,
    particles: "Particles",
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
            A constant (usually related to the charge-to-mass ratio).
    """

    for pti in particles.markers.iterator(particles.markers, 0):
        markers_array = particles.get_amrex_markers_array(pti.soa())
        e1 = markers_array["x"]
        e2 = markers_array["y"]
        e3 = markers_array["z"]

        n_p = len(e1)
        e_form = empty((n_p, 3, 1), dtype=float)

        # evaluate Jacobian
        jacobian = particles.domain.jacobian(e1, e2, e3, change_out_order=True, flat_eval=True)  # Npx3x3

        # metric coeffs
        jacobian_inverse = inv(jacobian)
        jacobian_inverse_T = transpose(jacobian_inverse, [0, 2, 1])

        # spline evaluation and electric field: 1-form components
        get_spans_and_eval_1form_spline_vectorized(
            e1,
            e2,
            e3,
            args_derham,
            e1_1,
            e1_2,
            e1_3,
            e_form,
        )

        # If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly. Squeeze to take away the unnecessary 1 dim
        e_cart = matmul(jacobian_inverse_T, e_form)

        # update velocities
        temp = dt * const * e_cart
        markers_array["v1"][:] = markers_array["v1"][:] + temp[:, 0, 0]
        markers_array["v2"][:] = markers_array["v2"][:] + temp[:, 1, 0]
        markers_array["v3"][:] = markers_array["v3"][:] + temp[:, 2, 0]


def push_eta_stage(
    dt: float,
    stage: int,
    particles: "Particles",
    a: "float[:]",
    b: "float[:]",
    c: "float[:]",
):
    r"""Single stage of a s-stage Runge-Kutta solve of

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant.
    """

    # get number of stages
    n_stages = shape(b)[0]

    if stage == n_stages - 1:
        last = 1.0
    else:
        last = 0.0

    for pti in particles.markers.iterator(particles.markers, 0):
        markers_array = particles.get_amrex_markers_array(pti.soa())

        e1 = markers_array["x"]
        e2 = markers_array["y"]
        e3 = markers_array["z"]
        v1 = markers_array["v1"]
        v2 = markers_array["v2"]
        v3 = markers_array["v3"]

        # evaluate inverse Jacobian matrices for each point
        jacobian_inv = particles.domain.jacobian_inv(e1, e2, e3, change_out_order=True, flat_eval=True)  # Npx3x3

        # pull-back of velocity
        v = array([v1, v2, v3]).T
        v = v[..., newaxis]  # Npx3x1
        # If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly. Squeeze to take away the unnecessary 1 dim
        k = matmul(jacobian_inv, v)
        # accumulation for last stage
        temp = dt * b[stage] * k
        markers_array["real_comp0"][:] = markers_array["real_comp0"][:] + temp[:, 0, 0]
        markers_array["real_comp1"][:] = markers_array["real_comp1"][:] + temp[:, 1, 0]
        markers_array["real_comp2"][:] = markers_array["real_comp2"][:] + temp[:, 2, 0]

        # update positions for intermediate stages or last stage
        temp = dt * a[stage] * k
        markers_array["x"][:] = markers_array["init_x"][:] + temp[:, 0, 0] + last * markers_array["real_comp0"][:]
        markers_array["y"][:] = markers_array["init_y"][:] + temp[:, 1, 0] + last * markers_array["real_comp1"][:]
        markers_array["z"][:] = markers_array["init_z"][:] + temp[:, 2, 0] + last * markers_array["real_comp2"][:]


def amrex_reflect(markers_array: "dict", outside_inds: "int[:]", axis: "int", domain: "Domain"):
    r"""
    Reflect the particles which are pushed outside of the logical cube.

    .. math::

        \hat{v} = DF^{-1} v \,, \\
        \hat{v}_\text{reflected}[\text{axis}] = -1 * \hat{v} \,, \\
        v_\text{reflected} = DF \hat{v}_\text{reflected} \,.

    Parameters
    ----------
        markers : array[float]
            Local markers array

        outside_inds : array[int]
            inds indicate the particles which are pushed outside of the local cube

        axis : int
            0, 1 or 2
    """

    e1 = markers_array["x"][outside_inds]
    e2 = markers_array["y"][outside_inds]
    e3 = markers_array["z"][outside_inds]
    v1 = markers_array["v1"][outside_inds]
    v2 = markers_array["v2"][outside_inds]
    v3 = markers_array["v3"][outside_inds]

    # evaluate inverse Jacobian matrices for each point
    jacobian = domain.jacobian(e1, e2, e3, change_out_order=True, flat_eval=True)  # Npx3x3
    jacobian_inv = domain.jacobian_inv(e1, e2, e3, change_out_order=True, flat_eval=True)  # Npx3x3

    # pull-back of velocity
    v = array([v1, v2, v3]).T
    v = v[..., newaxis]  # Npx3x1
    # If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly. Squeeze to take away the unnecessary 1 dim
    v_logical = matmul(jacobian_inv, v)

    # reverse the velocity
    v_logical[:, axis] *= -1

    # push forward of the velocity
    v = matmul(jacobian, v_logical)

    # update the particle velocities
    markers_array["v1"][outside_inds] = v[:, 0, 0]
    markers_array["v2"][outside_inds] = v[:, 1, 0]
    markers_array["v3"][outside_inds] = v[:, 2, 0]
