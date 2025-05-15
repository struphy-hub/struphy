"Pusher kernels for full orbit (6D) particles and AMReX data structures."

from numpy import array, matmul, newaxis, shape, stack

from struphy.geometry.base import Domain


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

    #$ omp parallel private(ip, eta1, eta2, eta3, dfm, dfinv, dfinvt, span1, span2, span3, e_form, e_cart)
    #$ omp for
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

    #$ omp end parallel


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
        markers_array = pti.soa().to_numpy()[0]
        e1 = markers_array["x"][:]
        e2 = markers_array["y"][:]
        e3 = markers_array["z"][:]
        v1 = markers_array["v1"][:]
        v2 = markers_array["v2"][:]
        v3 = markers_array["v3"][:]

        # evaluate inverse Jacobian matrices for each point
        jacobian_inv = particles.domain.jacobian_inv(e1, e2, e3, change_out_order=True, flat_eval=True)  # Npx3x3

        # pull-back of velocity
        v = array([v1, v2, v3]).T
        v = v[..., newaxis]  # Npx3x1
        # If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly. Squeeze to take away the unnecessary 1 dim
        k = matmul(jacobian_inv, v)
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
