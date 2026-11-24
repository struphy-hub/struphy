"Pusher kernel for full orbit (6D) particles, magnetic field as a 1-form."

from numpy import empty
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
from struphy.bsplines.evaluation_kernels_3d import eval_1form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("dfm", "b_form", "u_form", "e_form", "e_cart")
def push_bxu_Hcurl_1form(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    b1_1: "float[:,:,:]",
    b1_2: "float[:,:,:]",
    b1_3: "float[:,:,:]",
    u1_1: "float[:,:,:]",
    u1_2: "float[:,:,:]",
    u1_3: "float[:,:,:]",
    boundary_cut: "float",
):
    r"""Updates

    .. math::

        \frac{\mathbf v^{n+1}_p - \mathbf v^n_p}{\Delta t} = \left( DF^{-\top} \hat{\mathbf B}^1 \times DF^{-\top} \hat{\mathbf U}^1 \right)^n_p
        = \left( \frac{DF}{\sqrt g} \left( \hat{\mathbf B}^1 \times \hat{\mathbf U}^1 \right) \right)^n_p

    for each marker :math:`p` in markers array, where :math:`\hat{\mathbf B}^1, \hat{\mathbf U}^1 \in H(\textnormal{curl})`
    (using :math:`M \mathbf a \times M \mathbf c = \det(M) M^{-\top} (\mathbf a \times \mathbf c)`). The 1-form variant
    of :func:`~struphy.pic.pushing.kernels.push_bxu_Hcurl.push_bxu_Hcurl_kernels.push_bxu_Hcurl`.

    Parameters
    ----------
        b1_1, b1_2, b1_3: array[float]
            3d array of FE coeffs of B-field as 1-form.

        u1_1, u1_2, u1_3: array[float]
            3d array of FE coeffs of U-field as 1-form.
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # allocate for field evaluations (1-form components)
    b_form = empty(3, dtype=float)
    u_form = empty(3, dtype=float)

    # b1 x u1 and Cartesian components of the electric field
    e_form = empty(3, dtype=float)
    e_cart = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers

    # fmt: off
    #$ omp parallel private(b_form, det_df, dfm, e_cart, e_form, eta1, eta2, eta3, ip, span1, span2, span3, thread_derham, u_form)
    # fmt: on
    # Each thread owns the mutable spline-evaluation buffers.
    thread_derham = DerhamArguments(
        args_derham.pn, args_derham.tn1, args_derham.tn2, args_derham.tn3, args_derham.starts
    )
    # fmt: off
    #$ omp for
    # fmt: on
    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[0, ip] == -1.0:
            continue

        # boundary cut
        if markers[0, ip] < boundary_cut or markers[0, ip] > 1.0 - boundary_cut:
            continue

        # marker data
        eta1 = markers[0, ip]
        eta2 = markers[1, ip]
        eta3 = markers[2, ip]

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

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, thread_derham)

        # magnetic field: 1-form components
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            thread_derham,
            b1_1,
            b1_2,
            b1_3,
            b_form,
        )

        # velocity field: 1-form components
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            thread_derham,
            u1_1,
            u1_2,
            u1_3,
            u_form,
        )

        # electric field E = B x U = DF (b1 x u1) / det_df
        linalg_kernels.cross(b_form, u_form, e_form)
        linalg_kernels.matrix_vector(dfm, e_form, e_cart)

        # update velocities
        markers[3:6, ip] += dt * e_cart / det_df

    # fmt: off
    #$ omp end parallel
    # fmt: on
