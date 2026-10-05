"Pusher kernel for full orbit (6D) particles."

from numpy import empty
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
from struphy.bsplines.evaluation_kernels_3d import eval_1form_spline_mpi, eval_2form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


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
