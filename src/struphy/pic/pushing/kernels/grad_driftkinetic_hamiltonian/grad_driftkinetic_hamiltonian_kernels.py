"Initialization routine (initial guess, evaluations) for 5D gyro-center pusher kernels."

from numpy import empty, mod
from pyccel.decorators import stack_array

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
from struphy.bsplines.evaluation_kernels_3d import eval_1form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("eta_k", "eta_n", "eta", "grad_H", "e_field")
def grad_driftkinetic_hamiltonian(
    alpha: "float[:]",
    output_indices: "int[:]",
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

    Component j is saved at marker column ``output_indices[j]``. Supply three indices;
    An index of -1 skips that component for every particle.
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
        for j in range(3):
            if output_indices[j] >= 0:
                markers[ip, output_indices[j]] = grad_H[j]
