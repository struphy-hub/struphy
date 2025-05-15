"Pusher kernels for full orbit (6D) particles and AMReX data structures."

from numpy import array, matmul, newaxis, shape, transpose, zeros
from numpy.linalg import inv

from struphy.bsplines.evaluation_kernels_3d import eval_1form_spline_vectorized, get_spans_vectorized
from struphy.geometry.base import Domain


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
            A constant (usuallly related to the charge-to-mass ratio).
    """

    span1 = zeros(particles.Np, dtype=int)
    span2 = zeros(particles.Np, dtype=int)
    span3 = zeros(particles.Np, dtype=int)
    e_form = zeros((particles.Np, 3, 1), dtype=float)

    for pti in particles.markers.iterator(particles.markers, 0):
        markers_array = pti.soa().to_numpy()[0]
        e1 = markers_array["x"]
        e2 = markers_array["y"]
        e3 = markers_array["z"]

        # evaluate Jacobian
        jacobian = particles.domain.jacobian(e1, e2, e3, change_out_order=True, flat_eval=True)  # Npx3x3

        # metric coeffs
        jacobian_inverse = inv(jacobian)
        jacobian_inverse_T = transpose(jacobian_inverse, [0, 2, 1])

        # spline evaluation
        get_spans_vectorized(e1, e2, e3, args_derham, span1, span2, span3)

        # electric field: 1-form components
        eval_1form_spline_vectorized(
            span1,
            span2,
            span3,
            args_derham,
            e1_1,
            e1_2,
            e1_3,
            e_form,
        )

        # If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly. Squeeze to take away the unnecessary 1 dim
        e_cart = matmul(jacobian_inverse_T, e_form)  # TODO (Mati) maybe better to write our own piccelized

        # update velocities
        temp = dt * const * e_cart
        markers_array["v1"][:] = markers_array["v1"][:] + temp[:, 0].squeeze()
        markers_array["v2"][:] = markers_array["v2"][:] + temp[:, 1].squeeze()
        markers_array["v3"][:] = markers_array["v3"][:] + temp[:, 2].squeeze()


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

    # TODO (Mati) preallocate outside of the time loop and pass to the kernel, create slices (particles.velocity_buffer?)
    # attach to the propagator?

    for pti in particles.markers.iterator(particles.markers, 0):
        markers_array = pti.soa().to_numpy()[0]
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
        k = matmul(jacobian_inv, v)  # TODO (Mati) maybe better to write our own piccelized
        # accumulation for last stage
        temp = dt * b[stage] * k
        markers_array["real_comp0"][:] = markers_array["real_comp0"][:] + temp[:, 0].squeeze()
        markers_array["real_comp1"][:] = markers_array["real_comp1"][:] + temp[:, 1].squeeze()
        markers_array["real_comp2"][:] = markers_array["real_comp2"][:] + temp[:, 2].squeeze()

        # update positions for intermediate stages or last stage
        temp = dt * a[stage] * k
        markers_array["x"][:] = (
            markers_array["init_x"][:] + temp[:, 0].squeeze() + last * markers_array["real_comp0"][:]
        )
        markers_array["y"][:] = (
            markers_array["init_y"][:] + temp[:, 1].squeeze() + last * markers_array["real_comp1"][:]
        )
        markers_array["z"][:] = (
            markers_array["init_z"][:] + temp[:, 2].squeeze() + last * markers_array["real_comp2"][:]
        )


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
    markers_array["v1"][outside_inds] = v[:, 0].squeeze()
    markers_array["v2"][outside_inds] = v[:, 1].squeeze()
    markers_array["v3"][outside_inds] = v[:, 2].squeeze()
