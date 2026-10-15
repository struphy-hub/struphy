"Pusher kernel for gyro-center (5D) dynamics."

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
def gk_push_vpar_efield(
    dt: float,
    stage: int,
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
    B_dot_b_coeffs: "float[:,:,:]",
    curl_unit_b_dot_b0: "float[:,:,:]",
    e1_1: "float[:,:,:]",
    e1_2: "float[:,:,:]",
    e1_3: "float[:,:,:]",
):
    r"""Updates the parallel velocity of each marker :math:`p` as

    .. math::

        \frac{v_{\parallel,p}^{n+1} - v_{\parallel,p}^n}{\Delta t} = \frac{1}{\varepsilon}\, \mathbf c_p \cdot \hat{\mathbf e}^1(\boldsymbol \eta_p)\,,
        \qquad \mathbf c_p = \frac{\hat{\mathbf B}^{*2}}{\sqrt g\, \hat B^*_\parallel}(\boldsymbol \eta_p)\,,

    where :math:`\hat{\mathbf e}^1` are given FE coefficients of the electric field (1-form) and
    :math:`\mathbf c_p` is evaluated with the parallel velocity :math:`v_{\parallel,p}^n` before the update,
    exactly as in the accumulation kernel
    :func:`~struphy.pic.accumulation.kernels.gk_epar_coupling.gk_epar_coupling_kernels.gk_epar_coupling`.

    Parameters
    ----------
        epsilon : float
            Normalized cyclotron period of the species.

        b2_1, b2_2, b2_3 : array[float]
            FE coefficients c_ijk of the full magnetic field (equilibrium + perturbation) as a 2-form.

        curl_unit_b2_1, curl_unit_b2_2, curl_unit_b2_3 : array[float]
            FE coefficients c_ijk of the curl of the equilibrium unit magnetic field as a 2-form.

        B_dot_b_coeffs : array[float]
            FE coefficients c_ijk of the parallel magnetic field (equilibrium + perturbation) as a 0-form.

        curl_unit_b_dot_b0 : array[float]
            FE coefficients c_ijk of (curl b_0) . b_0 as a 0-form.

        e1_1, e1_2, e1_3 : array[float]
            FE coefficients c_ijk of the electric field as a 1-form.
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # containers for fields
    b2 = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    e_form = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    valid_mks = args_markers.valid_mks

    # fmt: off
    #$ omp parallel private(b2, b_star, b_star_para, B_dot_b, det_df, dfm, e_form, eta1, eta2, eta3, ip, span1, span2, span3, thread_derham, v)
    # fmt: on
    # Each thread owns the mutable spline-evaluation buffers.
    thread_derham = DerhamArguments(
        args_derham.pn, args_derham.tn1, args_derham.tn2, args_derham.tn3, args_derham.starts
    )
    # fmt: off
    #$ omp for
    # fmt: on
    for ip in range(n_markers):
        # only do something if particle is valid (i.e. not a hole or ghost)
        if not valid_mks[ip]:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        v = markers[ip, 3]

        # evaluate Jacobian determinant
        evaluation_kernels.df(eta1, eta2, eta3, args_domain, dfm)
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, thread_derham)

        # B*; 2-form
        eval_2form_spline_mpi(span1, span2, span3, thread_derham, b2_1, b2_2, b2_3, b2)
        eval_2form_spline_mpi(
            span1, span2, span3, thread_derham, curl_unit_b2_1, curl_unit_b2_2, curl_unit_b2_3, b_star
        )
        b_star *= epsilon * v
        b_star += b2

        # B*_parallel (times det DF)
        B_dot_b = eval_0form_spline_mpi(span1, span2, span3, thread_derham, B_dot_b_coeffs)
        b_star_para = eval_0form_spline_mpi(span1, span2, span3, thread_derham, curl_unit_b_dot_b0)
        b_star_para *= epsilon * v
        b_star_para += B_dot_b
        b_star_para *= det_df

        # electric field; 1-form
        eval_1form_spline_mpi(span1, span2, span3, thread_derham, e1_1, e1_2, e1_3, e_form)

        # update parallel velocity
        markers[ip, 3] += dt / epsilon * linalg_kernels.scalar_dot(b_star, e_form) / b_star_para

    # fmt: off
    #$ omp end parallel
    # fmt: on
