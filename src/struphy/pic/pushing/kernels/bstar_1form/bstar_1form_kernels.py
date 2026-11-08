"Initialization routine (initial guess, evaluations) for 5D gyro-center pusher kernels, magnetic field as a 1-form."

from numpy import empty, mod
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
from struphy.bsplines.evaluation_kernels_3d import eval_1form_spline_mpi, eval_2form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("eta_k", "eta_n", "eta", "dfm", "dfinv", "dfinvt", "b1", "tmp", "bv", "b_star")
def bstar_1form(
    alpha: "float[:]",
    output_indices: "int[:]",
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    epsilon: float,
    b1_1: "float[:,:,:]",
    b1_2: "float[:,:,:]",
    b1_3: "float[:,:,:]",
    curl_unit_b2_1: "float[:,:,:]",
    curl_unit_b2_2: "float[:,:,:]",
    curl_unit_b2_3: "float[:,:,:]",
):
    r"""Evaluate, for the magnetic field given as a 1-form :math:`\hat{\mathbf B}^1`,

    .. math::

        \hat{\mathbf B}^{*2}(\mathbf Z_p) = \sqrt g \, G^{-1} \hat{\mathbf B}^1(\boldsymbol \eta_p)
        + \varepsilon v_{\parallel,p} \hat \nabla \times \hat{\mathbf b}^1_0 (\boldsymbol \eta_p)\,,

    i.e. the 2-form components of :math:`\mathbf B^*` read by
    :func:`~struphy.pic.pushing.kernels.push_gc_Bstar_discrete_gradient_1st_order.push_gc_Bstar_discrete_gradient_1st_order_kernels.push_gc_Bstar_discrete_gradient_1st_order`
    and its Newton variant, which divide them by
    :math:`\hat B^{*3}_\parallel = \sqrt g \, \hat B^*_\parallel` (from ``bstar_parallel_3form``).
    The 1-form is converted pointwise to the vector field :math:`\hat{\mathbf B}^\textnormal{v} = G^{-1} \hat{\mathbf B}^1`
    (never to FE coefficients of a 2-form); the factor :math:`\sqrt g` is needed because the equilibrium term
    :math:`\hat \nabla \times \hat{\mathbf b}^1_0` is a 2-form and the pushers divide by the 3-form
    :math:`\hat B^{*3}_\parallel`.

    The evaluation point is the weighted average
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

    # metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinvt = empty((3, 3), dtype=float)

    # magnetic field: 1-form components, intermediate DF^{-T} b1, vector-field components G^{-1} b1
    b1 = empty(3, dtype=float)
    tmp = empty(3, dtype=float)
    bv = empty(3, dtype=float)
    b_star = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    first_pusher_idx = args_markers.first_pusher_idx
    first_shift_idx = args_markers.first_shift_idx

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[0, ip] == -1.0:
            continue

        eta_k[:] = markers[0:3, ip] + markers[first_shift_idx : first_shift_idx + 3, ip]
        eta_n[:] = markers[first_pusher_idx : first_pusher_idx + 3, ip]

        eta[:] = alpha[:3] * eta_k + (1.0 - alpha[:3]) * eta_n
        eta[:] = mod(eta, 1.0)

        v_k = markers[3, ip]
        v_n = markers[first_pusher_idx + 3, ip]
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
        linalg_kernels.matrix_inv_with_det(dfm, det_df, dfinv)
        linalg_kernels.transpose(dfinv, dfinvt)

        # spline evaluation
        span1, span2, span3 = get_spans(eta[0], eta[1], eta[2], args_derham)

        # magnetic field: 1-form components
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            b1_1,
            b1_2,
            b1_3,
            b1,
        )

        # vector-field components G^{-1} b1 = DF^{-1} DF^{-T} b1
        linalg_kernels.matrix_vector(dfinvt, b1, tmp)
        linalg_kernels.matrix_vector(dfinv, tmp, bv)

        # compute B* (2-form components)
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
        b_star += det_df * bv

        # save
        for j in range(3):
            if output_indices[j] >= 0:
                markers[output_indices[j], ip] = b_star[j]
