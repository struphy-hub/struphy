"Pusher kernel for full orbit (6D) particles."

from numpy import empty
from pyccel.decorators import stack_array

import struphy.bsplines.bsplines_kernels as bsplines_kernels
import struphy.bsplines.evaluation_kernels_3d as evaluation_kernels_3d
import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
from struphy.bsplines.evaluation_kernels_3d import eval_2form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


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
