"Pusher kernel for gyro-center (5D) particles."

from numpy import empty
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
from struphy.bsplines.evaluation_kernels_3d import (
    eval_0form_spline_mpi,
    eval_1form_spline_mpi,
    eval_2form_spline_mpi,
    get_spans,
)
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("dfm", "b2", "b_star", "e_form")
def push_vpar_with_efield(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    epsilon: float,
    e1_1: "float[:,:,:]",
    e1_2: "float[:,:,:]",
    e1_3: "float[:,:,:]",
    b2_1: "float[:,:,:]",
    b2_2: "float[:,:,:]",
    b2_3: "float[:,:,:]",
    curl_unit_b2_1: "float[:,:,:]",
    curl_unit_b2_2: "float[:,:,:]",
    curl_unit_b2_3: "float[:,:,:]",
    B_dot_b_coeffs: "float[:,:,:]",
    curl_unit_b_dot_b0: "float[:,:,:]",
    const_factor: float,
):
    r"""Updates the parallel velocities of guiding centers at fixed positions as

    .. math::

        \frac{v_{\parallel,p}^{n+1} - v_{\parallel,p}^n}{\Delta t}
        = c\, \frac{\hat{\mathbf B}^{*2}}{\hat B^{*3}_\parallel} \cdot \hat{\mathbf E}^1 (\boldsymbol \eta_p)\,,

    where :math:`\hat{\mathbf E}^1` is the 1-form spline field of the given FE coefficients, :math:`c`
    (``const_factor``) a constant, and :math:`\hat{\mathbf B}^{*2}`, :math:`\hat B^{*3}_\parallel` are evaluated with
    :math:`v_{\parallel,p}^n` as in
    :func:`~struphy.pic.accumulation.kernels.dk_maxwell.dk_maxwell_kernels.dk_maxwell`.
    """

    dfm = empty((3, 3), dtype=float)
    b2 = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    e_form = empty(3, dtype=float)

    markers = args_markers.markers
    n_markers = args_markers.n_markers
    valid_mks = args_markers.valid_mks

    for ip in range(n_markers):
        # only do something if particle is valid (i.e. not a hole or ghost)
        if not valid_mks[ip]:
            continue

        eta1 = markers[0, ip]
        eta2 = markers[1, ip]
        eta3 = markers[2, ip]
        v = markers[3, ip]

        evaluation_kernels.df(eta1, eta2, eta3, args_domain, dfm)
        det_df = linalg_kernels.det(dfm)

        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # B* (2-form) and B*_parallel (3-form)
        eval_2form_spline_mpi(span1, span2, span3, args_derham, b2_1, b2_2, b2_3, b2)
        eval_2form_spline_mpi(span1, span2, span3, args_derham, curl_unit_b2_1, curl_unit_b2_2, curl_unit_b2_3, b_star)
        b_star *= epsilon * v
        b_star += b2

        b_star_parallel = eval_0form_spline_mpi(span1, span2, span3, args_derham, curl_unit_b_dot_b0)
        b_star_parallel *= epsilon * v
        b_star_parallel += eval_0form_spline_mpi(span1, span2, span3, args_derham, B_dot_b_coeffs)
        b_star_parallel *= det_df

        # electric field (1-form)
        eval_1form_spline_mpi(span1, span2, span3, args_derham, e1_1, e1_2, e1_3, e_form)

        markers[3, ip] += dt * const_factor * linalg_kernels.scalar_dot(b_star, e_form) / b_star_parallel
