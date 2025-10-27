"Initialization routines (initial guess, evaluations) for 5D gyro-center pusher kernels."

from numpy import abs, empty, log, mod, shape, size, sqrt, zeros
from pyccel.decorators import stack_array

import struphy.bsplines.bsplines_kernels as bsplines_kernels
import struphy.bsplines.evaluation_kernels_3d as evaluation_kernels_3d
import struphy.geometry.evaluation_kernels as evaluation_kernels

# do not remove; needed to identify dependencies
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels
import struphy.linear_algebra.linalg_kernels as linalg_kernels
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


@stack_array("eta_k", "eta_n", "eta")
def driftkinetic_hamiltonian(
    alpha: "float[:]",
    column_nr: int,
    comps: "int[:]",
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    epsilon: float,
    B_dot_b_coeffs: "float[:,:, :]",
    phi_coeffs: "float[:,:, :]",
    evaluate_e_field: bool,
):
    r"""Evaluate the Hamiltonian

    .. math::

        H(\mathbf Z_p) = H(\boldsymbol \eta_p, v_{\parallel,p}) = \varepsilon \frac{v_{\parallel,p}^2}{2}
        + \varepsilon\mu |\hat \mathbf B| (\boldsymbol \eta_p) + \hat \phi(\boldsymbol \eta_p)\,,

    where the evaluation point is the weighted average
    :math:`Z_{p,i} = \alpha_i Z_{p,i}^{n+1,k} + (1 - \alpha_i) Z_{p,i}^n`,
    for :math:`i=1,2,3,4`. Markers must be sorted according to the evaluation point
    :math:`\boldsymbol \eta_p` beforehand.

    The result is saved at ``column_nr`` in markers array for each particle.
    """

    # allocate stack arrays
    eta_k = empty(3, dtype=float)
    eta_n = empty(3, dtype=float)
    eta = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    mu_idx = args_markers.mu_idx
    first_init_idx = args_markers.first_init_idx
    first_shift_idx = args_markers.first_shift_idx

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        eta_k[:] = markers[ip, 0:3] + markers[ip, first_shift_idx : first_shift_idx + 3]
        eta_n[:] = markers[ip, first_init_idx : first_init_idx + 3]

        eta[:] = alpha[:3] * eta_k + (1.0 - alpha[:3]) * eta_n
        eta[:] = mod(eta, 1.0)

        v_k = markers[ip, 3]
        v_n = markers[ip, first_init_idx + 3]
        v = alpha[3] * v_k + (1.0 - alpha[3]) * v_n

        mu = markers[ip, mu_idx]

        # spline evaluation
        span1, span2, span3 = get_spans(eta[0], eta[1], eta[2], args_derham)

        if evaluate_e_field:
            phi = eval_0form_spline_mpi(
                span1,
                span2,
                span3,
                args_derham,
                phi_coeffs,
            )
        else:
            phi = 0.0

        B_dot_b = eval_0form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            B_dot_b_coeffs,
        )

        # save
        markers[ip, column_nr] = epsilon * v**2 / 2.0 + epsilon * mu * B_dot_b + phi


@stack_array("eta_k", "eta_n", "eta", "grad_H", "e_field")
def grad_driftkinetic_hamiltonian(
    alpha: "float[:]",
    column_nr: int,
    comps: "int[:]",
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    epsilon: float,
    grad_b_full_1: "float[:,:,:]",
    grad_b_full_2: "float[:,:,:]",
    grad_b_full_3: "float[:,:,:]",
    e_field_1: "float[:,:,:]",
    e_field_2: "float[:,:,:]",
    e_field_3: "float[:,:,:]",
    evaluate_e_field: bool,
):
    r"""Evaluate the :math:`\boldsymbol \eta`-gradient of the Hamiltonian

    .. math::

        H(\mathbf Z_p) = H(\boldsymbol \eta_p, v_{\parallel,p}) = \varepsilon \frac{v_{\parallel,p}^2}{2}
        + \varepsilon \mu |\hat \mathbf B| (\boldsymbol \eta_p) + \hat \phi(\boldsymbol \eta_p)\,,

    that is

    .. math::

        \hat \nabla H(\mathbf Z_p) = \varepsilon \mu \hat \nabla |\hat \mathbf B| (\boldsymbol \eta_p)
        + \hat \nabla \hat \phi(\boldsymbol \eta_p)\,,

    where the evaluation point is the weighted average
    :math:`Z_{p,i} = \alpha_i Z_{p,i}^{n+1,k} + (1 - \alpha_i) Z_{p,i}^n`,
    for :math:`i=1,2,3,4`. Markers must be sorted according to the evaluation point
    :math:`\boldsymbol \eta_p` beforehand.

    The components specified in ``comps`` are save at ``column_nr:column_nr + len(comps)``
    in markers array for each particle.
    """

    # allocate stack arrays
    eta_k = empty(3, dtype=float)
    eta_n = empty(3, dtype=float)
    eta = empty(3, dtype=float)
    grad_H = empty(3, dtype=float)
    e_field = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    mu_idx = args_markers.mu_idx
    first_init_idx = args_markers.first_init_idx
    first_shift_idx = args_markers.first_shift_idx

    # for saving
    n_comps = size(comps)

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        eta_k[:] = markers[ip, 0:3] + markers[ip, first_shift_idx : first_shift_idx + 3]
        eta_n[:] = markers[ip, first_init_idx : first_init_idx + 3]

        eta[:] = alpha[:3] * eta_k + (1.0 - alpha[:3]) * eta_n
        eta[:] = mod(eta, 1.0)

        mu = markers[ip, mu_idx]

        # spline evaluation
        span1, span2, span3 = get_spans(eta[0], eta[1], eta[2], args_derham)

        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            grad_b_full_1,
            grad_b_full_2,
            grad_b_full_3,
            grad_H,
        )

        grad_H *= epsilon * mu

        if evaluate_e_field:
            eval_1form_spline_mpi(
                span1,
                span2,
                span3,
                args_derham,
                e_field_1,
                e_field_2,
                e_field_3,
                e_field,
            )

            e_field *= -1.0
            grad_H += e_field

        # save
        for j in range(n_comps):
            markers[ip, column_nr + j] = grad_H[comps[j]]


@stack_array("eta_k", "eta_n", "eta", "dfm")
def bstar_parallel_3form(
    alpha: "float[:]",
    column_nr: int,
    comps: "int[:]",
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    epsilon: float,
    B_dot_b_coeffs: "float[:,:,:]",
    curl_unit_b_dot_b0: "float[:,:,:]",
):
    r"""Evaluate

    .. math::

        \hat B^{*3}_\parallel(\mathbf Z_p) = \sqrt g \left(\hat B + \varepsilon v_{\parallel,p} \widehat{\left[(\nabla \times \mathbf b_0) \cdot \mathbf b_0\right]}(\boldsymbol \eta_p) \right)\,,

    where the evaluation point is the weighted average
    :math:`Z_{p,i} = \alpha_i Z_{p,i}^{n+1,k} + (1 - \alpha_i) Z_{p,i}^n`,
    for :math:`i=1,2,3,4`. Markers must be sorted according to the evaluation point
    :math:`\boldsymbol \eta_p` beforehand.

    The result is saved at ``column_nr``  in markers array for each particle.
    """

    # allocate stack arrays
    eta_k = empty(3, dtype=float)
    eta_n = empty(3, dtype=float)
    eta = empty(3, dtype=float)
    dfm = empty((3, 3), dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    mu_idx = args_markers.mu_idx
    first_init_idx = args_markers.first_init_idx
    first_shift_idx = args_markers.first_shift_idx

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        eta_k[:] = markers[ip, 0:3] + markers[ip, first_shift_idx : first_shift_idx + 3]
        eta_n[:] = markers[ip, first_init_idx : first_init_idx + 3]

        eta[:] = alpha[:3] * eta_k + (1.0 - alpha[:3]) * eta_n
        eta[:] = mod(eta, 1.0)

        v_k = markers[ip, 3]
        v_n = markers[ip, first_init_idx + 3]
        v = alpha[3] * v_k + (1.0 - alpha[3]) * v_n

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta[0],
            eta[1],
            eta[2],
            args_domain,
            dfm,
        )

        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1, span2, span3 = get_spans(eta[0], eta[1], eta[2], args_derham)

        # compute B*_parallel
        B_dot_b = eval_0form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            B_dot_b_coeffs,
        )

        b_star_parallel = eval_0form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            curl_unit_b_dot_b0,
        )

        b_star_parallel *= epsilon * v
        b_star_parallel += B_dot_b
        b_star_parallel *= det_df

        markers[ip, column_nr] = b_star_parallel


@stack_array("eta_k", "eta_n", "eta", "b2", "b_star")
def bstar_2form(
    alpha: "float[:]",
    column_nr: int,
    comps: "int[:]",
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    epsilon: float,
    b2_1: "float[:,:,:]",
    b2_2: "float[:,:,:]",
    b2_3: "float[:,:,:]",
    curl_unit_b2_1: "float[:,:,:]",
    curl_unit_b2_2: "float[:,:,:]",
    curl_unit_b2_3: "float[:,:,:]",
):
    r"""Evaluate

    .. math::

        \hat{\mathbf B}^{*2}(\mathbf Z_p) = \hat{\mathbf B}^2(\boldsymbol \eta_p)
        + \varepsilon v_{\parallel,p} \hat \nabla \times \hat{\mathbf b}^1_0 (\boldsymbol \eta_p)\,,

    where the evaluation point is the weighted average
    :math:`Z_{p,i} = \alpha_i Z_{p,i}^{n+1,k} + (1 - \alpha_i) Z_{p,i}^n`,
    for :math:`i=1,2,3,4`. Markers must be sorted according to the evaluation point
    :math:`\boldsymbol \eta_p` beforehand.

    The components specified in ``comps`` are save at ``column_nr:column_nr + len(comps)``
    in markers array for each particle.
    """

    # allocate stack arrays
    eta_k = empty(3, dtype=float)
    eta_n = empty(3, dtype=float)
    eta = empty(3, dtype=float)
    b2 = empty(3, dtype=float)
    b_star = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    mu_idx = args_markers.mu_idx
    first_init_idx = args_markers.first_init_idx
    first_shift_idx = args_markers.first_shift_idx

    # for saving
    n_comps = size(comps)

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        eta_k[:] = markers[ip, 0:3] + markers[ip, first_shift_idx : first_shift_idx + 3]
        eta_n[:] = markers[ip, first_init_idx : first_init_idx + 3]

        eta[:] = alpha[:3] * eta_k + (1.0 - alpha[:3]) * eta_n
        eta[:] = mod(eta, 1.0)

        v_k = markers[ip, 3]
        v_n = markers[ip, first_init_idx + 3]
        v = alpha[3] * v_k + (1.0 - alpha[3]) * v_n

        # spline evaluation
        span1, span2, span3 = get_spans(eta[0], eta[1], eta[2], args_derham)

        # compute B*
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            b2_1,
            b2_2,
            b2_3,
            b2,
        )

        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            curl_unit_b2_1,
            curl_unit_b2_2,
            curl_unit_b2_3,
            b_star,
        )

        b_star *= epsilon * v
        b_star += b2

        # save
        for j in range(n_comps):
            markers[ip, column_nr + j] = b_star[comps[j]]


@stack_array("eta_k", "eta_n", "eta", "unit_b1")
def unit_b_1form(
    alpha: "float[:]",
    column_nr: int,
    comps: "int[:]",
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    unit_b1_1: "float[:,:,:]",
    unit_b1_2: "float[:,:,:]",
    unit_b1_3: "float[:,:,:]",
):
    r"""Evaluate :math:`\hat{\mathbf b}^1_0(\boldsymbol \eta_p)`,
    where the evaluation point is the weighted average
    :math:`\eta_{p,i} = \alpha_i \eta_{p,i}^{n+1,k} + (1 - \alpha_i) \eta_{p,i}^n`,
    for :math:`i=1,2,3`. Markers must be sorted according to the evaluation point
    :math:`\boldsymbol \eta_p` beforehand.

    The components specified in ``comps`` are save at ``column_nr:column_nr + len(comps)``
    in markers array for each particle.
    """

    # allocate stack arrays
    eta_k = empty(3, dtype=float)
    eta_n = empty(3, dtype=float)
    eta = empty(3, dtype=float)
    unit_b1 = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    mu_idx = args_markers.mu_idx
    first_init_idx = args_markers.first_init_idx
    first_shift_idx = args_markers.first_shift_idx

    # for saving
    n_comps = size(comps)

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        eta_k[:] = markers[ip, 0:3] + markers[ip, first_shift_idx : first_shift_idx + 3]
        eta_n[:] = markers[ip, first_init_idx : first_init_idx + 3]

        eta[:] = alpha[:3] * eta_k + (1.0 - alpha[:3]) * eta_n
        eta[:] = mod(eta, 1.0)

        # spline evaluation
        span1, span2, span3 = get_spans(eta[0], eta[1], eta[2], args_derham)

        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            unit_b1_1,
            unit_b1_2,
            unit_b1_3,
            unit_b1,
        )

        # save
        for j in range(n_comps):
            markers[ip, column_nr + j] = unit_b1[comps[j]]


@stack_array("eta_k", "eta_n", "eta", "grad_H", "e_field")
def sph_pressure_coeffs(
    alpha: "float[:]",
    column_nr: int,
    comps: "int[:]",
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    boxes: "int[:, :]",
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
    r"""Evaluate the :math:`\boldsymbol \eta`-gradient of the Hamiltonian

    .. math::

        H(\mathbf Z_p) = H(\boldsymbol \eta_p, v_{\parallel,p}) = \varepsilon \frac{v_{\parallel,p}^2}{2}
        + \varepsilon \mu |\hat \mathbf B| (\boldsymbol \eta_p) + \hat \phi(\boldsymbol \eta_p)\,,

    that is

    .. math::

        \hat \nabla H(\mathbf Z_p) = \varepsilon \mu \hat \nabla |\hat \mathbf B| (\boldsymbol \eta_p)
        + \hat \nabla \hat \phi(\boldsymbol \eta_p)\,,

    where the evaluation point is the weighted average
    :math:`Z_{p,i} = \alpha_i Z_{p,i}^{n+1,k} + (1 - \alpha_i) Z_{p,i}^n`,
    for :math:`i=1,2,3,4`. Markers must be sorted according to the evaluation point
    :math:`\boldsymbol \eta_p` beforehand.

    The components specified in ``comps`` are save at ``column_nr:column_nr + len(comps)``
    in markers array for each particle.
    """

    gamma = 5 / 3

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    n_cols = shape(markers)[1]
    Np = args_markers.Np
    weight_idx = args_markers.weight_idx
    valid_mks = args_markers.valid_mks

    for ip in range(n_markers):
        # only do something if particle is a "true" particle
        if not valid_mks[ip]:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        loc_box = int(markers[ip, n_cols - 2])
        n_at_eta = sph_eval_kernels.boxed_based_kernel(
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
            kernel_type,
            h1,
            h2,
            h3,
        )
        weight = markers[ip, weight_idx]
        # save
        markers[ip, column_nr] = n_at_eta
        markers[ip, column_nr + 1] = weight / n_at_eta
        markers[ip, column_nr + 2] = weight * n_at_eta ** (gamma - 2)


@stack_array("eta_k", "eta_n", "eta", "grad_H", "e_field")
def sph_isotherm_kappa(
    alpha: "float[:]",
    column_nr: int,
    comps: "int[:]",
    args_markers: "MarkerArguments",
):
    r"""Evaluate the :math:`\boldsymbol \eta`-gradient of the Hamiltonian

    .. math::

        H(\mathbf Z_p) = H(\boldsymbol \eta_p, v_{\parallel,p}) = \varepsilon \frac{v_{\parallel,p}^2}{2}
        + \varepsilon \mu |\hat \mathbf B| (\boldsymbol \eta_p) + \hat \phi(\boldsymbol \eta_p)\,,

    that is

    .. math::

        \hat \nabla H(\mathbf Z_p) = \varepsilon \mu \hat \nabla |\hat \mathbf B| (\boldsymbol \eta_p)
        + \hat \nabla \hat \phi(\boldsymbol \eta_p)\,,

    where the evaluation point is the weighted average
    :math:`Z_{p,i} = \alpha_i Z_{p,i}^{n+1,k} + (1 - \alpha_i) Z_{p,i}^n`,
    for :math:`i=1,2,3,4`. Markers must be sorted according to the evaluation point
    :math:`\boldsymbol \eta_p` beforehand.

    The components specified in ``comps`` are save at ``column_nr:column_nr + len(comps)``
    in markers array for each particle.
    """

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    first_diagnostic_idx = args_markers.first_diagnostics_idx

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        markers[ip, first_diagnostic_idx] = 1.0


@stack_array("eta_k", "eta_n", "eta", "grad_H", "e_field")
def sph_mean_velocity_coeffs(
    alpha: "float[:]",
    column_nr: int,
    comps: "int[:]",
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    boxes: "int[:, :]",
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
    r"""Evaluate the :math:`\boldsymbol \eta`-gradient of the Hamiltonian

    .. math::

        H(\mathbf Z_p) = H(\boldsymbol \eta_p, v_{\parallel,p}) = \varepsilon \frac{v_{\parallel,p}^2}{2}
        + \varepsilon \mu |\hat \mathbf B| (\boldsymbol \eta_p) + \hat \phi(\boldsymbol \eta_p)\,,

    that is

    .. math::

        \hat \nabla H(\mathbf Z_p) = \varepsilon \mu \hat \nabla |\hat \mathbf B| (\boldsymbol \eta_p)
        + \hat \nabla \hat \phi(\boldsymbol \eta_p)\,,

    where the evaluation point is the weighted average
    :math:`Z_{p,i} = \alpha_i Z_{p,i}^{n+1,k} + (1 - \alpha_i) Z_{p,i}^n`,
    for :math:`i=1,2,3,4`. Markers must be sorted according to the evaluation point
    :math:`\boldsymbol \eta_p` beforehand.

    The components specified in ``comps`` are save at ``column_nr:column_nr + len(comps)``
    in markers array for each particle.
    """

    gamma = 5 / 3

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    n_cols = shape(markers)[1]
    Np = args_markers.Np
    vdim = args_markers.vdim
    weight_idx = args_markers.weight_idx
    valid_mks = args_markers.valid_mks

    for ip in range(n_markers):
        # only do something if particle is a "true" particle
        if not valid_mks[ip]:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        loc_box = int(markers[ip, n_cols - 2])
        n_at_eta = sph_eval_kernels.boxed_based_kernel(
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
            kernel_type,
            h1,
            h2,
            h3,
        )
        weight = markers[ip, weight_idx]
        velocities = markers[ip, 3:6]
        # save
        markers[ip, column_nr] = weight / n_at_eta * velocities[0]
        markers[ip, column_nr + 1] = weight / n_at_eta * velocities[1]
        markers[ip, column_nr + 2] = weight / n_at_eta * velocities[2]
        
        # print(f"{ip = }, {weight = }, {n_at_eta = }, {velocities[0] = }")


@stack_array("eta_k", "eta_n", "eta", "grad_H", "e_field")
def sph_mean_velocity(
    alpha: "float[:]",
    column_nr: int,
    comps: "int[:]",
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    boxes: "int[:, :]",
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
    r"""Evaluate the :math:`\boldsymbol \eta`-gradient of the Hamiltonian

    .. math::

        H(\mathbf Z_p) = H(\boldsymbol \eta_p, v_{\parallel,p}) = \varepsilon \frac{v_{\parallel,p}^2}{2}
        + \varepsilon \mu |\hat \mathbf B| (\boldsymbol \eta_p) + \hat \phi(\boldsymbol \eta_p)\,,

    that is

    .. math::

        \hat \nabla H(\mathbf Z_p) = \varepsilon \mu \hat \nabla |\hat \mathbf B| (\boldsymbol \eta_p)
        + \hat \nabla \hat \phi(\boldsymbol \eta_p)\,,

    where the evaluation point is the weighted average
    :math:`Z_{p,i} = \alpha_i Z_{p,i}^{n+1,k} + (1 - \alpha_i) Z_{p,i}^n`,
    for :math:`i=1,2,3,4`. Markers must be sorted according to the evaluation point
    :math:`\boldsymbol \eta_p` beforehand.

    The components specified in ``comps`` are save at ``column_nr:column_nr + len(comps)``
    in markers array for each particle.
    """

    gamma = 5 / 3

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    n_cols = shape(markers)[1]
    Np = args_markers.Np
    vdim = args_markers.vdim
    weight_idx = args_markers.weight_idx
    first_free_idx = args_markers.first_free_idx
    valid_mks = args_markers.valid_mks

    for ip in range(n_markers):
        # only do something if particle is a "true" particle
        if not valid_mks[ip]:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        loc_box = int(markers[ip, n_cols - 2])
        v1_at_eta = sph_eval_kernels.boxed_based_kernel(
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
            first_free_idx,
            kernel_type,
            h1,
            h2,
            h3,
        )

        v2_at_eta = sph_eval_kernels.boxed_based_kernel(
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
            kernel_type,
            h1,
            h2,
            h3,
        )

        v3_at_eta = sph_eval_kernels.boxed_based_kernel(
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
            kernel_type,
            h1,
            h2,
            h3,
        )
        # save
        markers[ip, column_nr] = v1_at_eta
        markers[ip, column_nr + 1] = v2_at_eta
        markers[ip, column_nr + 2] = v3_at_eta


@stack_array("eta_k", "eta_n", "eta", "grad_H", "e_field")
def sph_grad_mean_velocity(
    alpha: "float[:]",
    column_nr: int,
    comps: "int[:]",
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    boxes: "int[:, :]",
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
    r"""Evaluate the :math:`\boldsymbol \eta`-gradient of the Hamiltonian

    .. math::

        H(\mathbf Z_p) = H(\boldsymbol \eta_p, v_{\parallel,p}) = \varepsilon \frac{v_{\parallel,p}^2}{2}
        + \varepsilon \mu |\hat \mathbf B| (\boldsymbol \eta_p) + \hat \phi(\boldsymbol \eta_p)\,,

    that is

    .. math::

        \hat \nabla H(\mathbf Z_p) = \varepsilon \mu \hat \nabla |\hat \mathbf B| (\boldsymbol \eta_p)
        + \hat \nabla \hat \phi(\boldsymbol \eta_p)\,,

    where the evaluation point is the weighted average
    :math:`Z_{p,i} = \alpha_i Z_{p,i}^{n+1,k} + (1 - \alpha_i) Z_{p,i}^n`,
    for :math:`i=1,2,3,4`. Markers must be sorted according to the evaluation point
    :math:`\boldsymbol \eta_p` beforehand.

    The components specified in ``comps`` are save at ``column_nr:column_nr + len(comps)``
    in markers array for each particle.
    """

    gamma = 5 / 3

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    n_cols = shape(markers)[1]
    Np = args_markers.Np
    vdim = args_markers.vdim
    weight_idx = args_markers.weight_idx
    first_free_idx = args_markers.first_free_idx
    valid_mks = args_markers.valid_mks

    grad_v_at_eta = zeros((3, 3), dtype=float)
    for ip in range(n_markers):
        # only do something if particle is a "true" particle
        if not valid_mks[ip]:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        loc_box = int(markers[ip, n_cols - 2])
        for j in range(3):
            for k in range(3):
                grad_v_at_eta[j, k] = sph_eval_kernels.boxed_based_kernel(
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
                    first_free_idx + j,
                    kernel_type + 1 + k,
                    h1,
                    h2,
                    h3,
                )

                # save
                markers[ip, column_nr + 3 * j + k] = grad_v_at_eta[j, k]


@stack_array("eta_k", "eta_n", "eta", "grad_H", "e_field")
def sph_viscosity_tensor(
    alpha: "float[:]",
    column_nr: int,
    comps: "int[:]",
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    boxes: "int[:, :]",
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
    r"""Evaluate the :math:`\boldsymbol \eta`-gradient of the Hamiltonian

    .. math::

        H(\mathbf Z_p) = H(\boldsymbol \eta_p, v_{\parallel,p}) = \varepsilon \frac{v_{\parallel,p}^2}{2}
        + \varepsilon \mu |\hat \mathbf B| (\boldsymbol \eta_p) + \hat \phi(\boldsymbol \eta_p)\,,

    that is

    .. math::

        \hat \nabla H(\mathbf Z_p) = \varepsilon \mu \hat \nabla |\hat \mathbf B| (\boldsymbol \eta_p)
        + \hat \nabla \hat \phi(\boldsymbol \eta_p)\,,

    where the evaluation point is the weighted average
    :math:`Z_{p,i} = \alpha_i Z_{p,i}^{n+1,k} + (1 - \alpha_i) Z_{p,i}^n`,
    for :math:`i=1,2,3,4`. Markers must be sorted according to the evaluation point
    :math:`\boldsymbol \eta_p` beforehand.

    The components specified in ``comps`` are save at ``column_nr:column_nr + len(comps)``
    in markers array for each particle.
    """

    gamma = 5 / 3

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    n_cols = shape(markers)[1]
    Np = args_markers.Np
    vdim = args_markers.vdim
    weight_idx = args_markers.weight_idx
    first_free_idx = args_markers.first_free_idx
    valid_mks = args_markers.valid_mks

    grad_v_at_eta = zeros((3, 3), dtype=float)
    d_dev = zeros((3, 3), dtype=float)
    for ip in range(n_markers):
        # only do something if particle is a "true" particle
        if not valid_mks[ip]:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        loc_box = int(markers[ip, n_cols - 2])
        n_at_eta = sph_eval_kernels.boxed_based_kernel(
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
            kernel_type,
            h1,
            h2,
            h3,
        )
        weight = markers[ip, weight_idx]
        for j in range(3):
            for k in range(3):
                grad_v_at_eta[j, k] = sph_eval_kernels.boxed_based_kernel(
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
                    first_free_idx + j,
                    kernel_type + 1 + k,
                    h1,
                    h2,
                    h3,
                )

        mu = 0.007
        d = 0.5 * (grad_v_at_eta + grad_v_at_eta.T)
        trace_d = d[0, 0] + d[1, 1] + d[2, 2]
        d_dev[0, 0] = d[0, 0] - (trace_d / 3.0)
        d_dev[1, 1] = d[1, 1] - (trace_d / 3.0)
        d_dev[2, 2] = d[2, 2] - (trace_d / 3.0)
        d_dev *= 2 * mu * weight / n_at_eta
        for j in range(3):
            for k in range(3):
                markers[ip, column_nr + 3 * j + k] = d_dev[j, k]
