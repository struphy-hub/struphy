"Initialization routine (initial guess, evaluations) for 5D gyro-center pusher kernels."

from numpy import empty, mod
from pyccel.decorators import stack_array

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
from struphy.bsplines.evaluation_kernels_3d import eval_0form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("eta_k", "eta_n", "eta")
def driftkinetic_hamiltonian(
    alpha: "float[:]",
    output_indices: "int[:]",
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

    The result is saved at ``output_indices[0]`` in the marker array; -1 skips the output.
    """

    # allocate stack arrays
    eta_k = empty(3, dtype=float)
    eta_n = empty(3, dtype=float)
    eta = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    mu_idx = args_markers.mu_idx
    first_pusher_idx = args_markers.first_pusher_idx
    first_shift_idx = args_markers.first_shift_idx

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        eta_k[:] = markers[ip, 0:3] + markers[ip, first_shift_idx : first_shift_idx + 3]
        eta_n[:] = markers[ip, first_pusher_idx : first_pusher_idx + 3]

        eta[:] = alpha[:3] * eta_k + (1.0 - alpha[:3]) * eta_n
        eta[:] = mod(eta, 1.0)

        v_k = markers[ip, 3]
        v_n = markers[ip, first_pusher_idx + 3]
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
        if output_indices[0] >= 0:
            markers[ip, output_indices[0]] = epsilon * v**2 / 2.0 + epsilon * mu * B_dot_b + phi
