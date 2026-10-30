"Accumulation kernel for gyro-center (5D) particles, matrix-free variant of dk_maxwell."

from numpy import empty
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.accumulation.particle_to_mat_kernels as particle_to_mat_kernels
from struphy.bsplines.evaluation_kernels_3d import (
    eval_0form_spline_mpi,
    eval_1form_spline_mpi,
    eval_2form_spline_mpi,
    get_spans,
)
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("dfm", "b2", "b_star", "unit_b1", "a")
def dk_maxwell_mf(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    vec1: "float[:,:,:]",
    vec2: "float[:,:,:]",
    vec3: "float[:,:,:]",
    epsilon: float,
    b2_1: "float[:,:,:]",
    b2_2: "float[:,:,:]",
    b2_3: "float[:,:,:]",
    curl_unit_b2_1: "float[:,:,:]",
    curl_unit_b2_2: "float[:,:,:]",
    curl_unit_b2_3: "float[:,:,:]",
    unit_b1_1: "float[:,:,:]",
    unit_b1_2: "float[:,:,:]",
    unit_b1_3: "float[:,:,:]",
    B_dot_b_coeffs: "float[:,:,:]",
    curl_unit_b_dot_b0: "float[:,:,:]",
    column: "int",
):
    r"""Accumulation kernel for the propagator
    :class:`~struphy.propagators.drift_kinetic_maxwell_coupling.DriftKineticMaxwellCoupling` (matrix-free).
    Accumulates the vector :math:`B_p^\mu = w_p v_{\parallel,p} a_{p,\mu}` into :math:`V^1_h` and stores, per marker,
    the data of the two particle matrices of
    :func:`~struphy.pic.accumulation.kernels.dk_maxwell.dk_maxwell_kernels.dk_maxwell`
    (see :class:`~struphy.pic.accumulation.particles_to_grid.MarkerOperator`):

    - the rank-one filling :math:`w_p\, \mathbf a_p \mathbf a_p^\top` in the columns ``column`` to ``column + 3``
      (:math:`\mathbf a_p` and :math:`w_p`), applied by
      :func:`~struphy.pic.accumulation.kernels.marker_operator_rank1_dot.marker_operator_rank1_dot_kernels.marker_operator_rank1_dot`;
    - the antisymmetric filling :math:`w_p S_p`, :math:`S_p \mathbf x = \mathbf x \times \hat{\mathbf b}^1_0 /
      \hat B^{*3}_\parallel`, entries 12, 13, 23 in the columns ``column + 4`` to ``column + 6``, applied by
      :func:`~struphy.pic.accumulation.kernels.marker_operator_asym_dot.marker_operator_asym_dot_kernels.marker_operator_asym_dot`.
    """

    markers = args_markers.markers

    dfm = empty((3, 3), dtype=float)
    b2 = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    unit_b1 = empty(3, dtype=float)
    a = empty(3, dtype=float)

    for ip in range(args_markers.n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[0, ip] == -1.0:
            continue

        eta1 = markers[0, ip]
        eta2 = markers[1, ip]
        eta3 = markers[2, ip]
        v = markers[3, ip]
        weight = markers[5, ip]

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

        a[:] = b_star / b_star_parallel

        # unit vector b0 (1-form)
        eval_1form_spline_mpi(span1, span2, span3, args_derham, unit_b1_1, unit_b1_2, unit_b1_3, unit_b1)

        s = weight / b_star_parallel

        markers[column, ip] = a[0]
        markers[column + 1, ip] = a[1]
        markers[column + 2, ip] = a[2]
        markers[column + 3, ip] = weight
        markers[column + 4, ip] = s * unit_b1[2]
        markers[column + 5, ip] = -s * unit_b1[1]
        markers[column + 6, ip] = s * unit_b1[0]

        particle_to_mat_kernels.vec_fill_v1(
            args_derham,
            span1,
            span2,
            span3,
            vec1,
            vec2,
            vec3,
            weight * v * a[0],
            weight * v * a[1],
            weight * v * a[2],
        )
