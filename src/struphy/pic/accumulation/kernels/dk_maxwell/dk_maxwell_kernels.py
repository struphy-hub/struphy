"Accumulation kernel for gyro-center (5D) particles."

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


@stack_array("dfm", "b2", "b_star", "unit_b1", "a", "filling_m")
def dk_maxwell(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    mat11: "float[:,:,:,:,:,:]",
    mat12: "float[:,:,:,:,:,:]",
    mat13: "float[:,:,:,:,:,:]",
    mat21: "float[:,:,:,:,:,:]",
    mat22: "float[:,:,:,:,:,:]",
    mat23: "float[:,:,:,:,:,:]",
    mat31: "float[:,:,:,:,:,:]",
    mat32: "float[:,:,:,:,:,:]",
    mat33: "float[:,:,:,:,:,:]",
    vec1: "float[:,:,:]",
    vec2: "float[:,:,:]",
    vec3: "float[:,:,:]",
    epsilon: float,
    lam: float,
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
):
    r"""Accumulation kernel for the propagator
    :class:`~struphy.propagators.drift_kinetic_maxwell_coupling.DriftKineticMaxwellCoupling`. Accumulates into
    :math:`V^1_h` with the (non-symmetric) filling functions

    .. math::

        A_p^{\mu\nu} = w_p \left( a_{p,\mu} a_{p,\nu} + \lambda\, S_{p,\mu\nu} \right)\,,
        \qquad
        B_p^\mu = w_p\, v_{\parallel,p}\, a_{p,\mu}\,,
        \qquad
        \mathbf a_p = \frac{\hat{\mathbf B}^{*2}}{\hat B^{*3}_\parallel}(\boldsymbol \eta_p)\,,
        \qquad
        S_p \mathbf x = \frac{\mathbf x \times \hat{\mathbf b}^1_0}{\hat B^{*3}_\parallel}(\boldsymbol \eta_p)\,,

    with :math:`\hat{\mathbf B}^{*2} = \hat{\mathbf B}^2 + \varepsilon v_\parallel \hat \nabla \times
    \hat{\mathbf b}^1_0` and :math:`\hat B^{*3}_\parallel = \sqrt g\, (\hat B_\parallel + \varepsilon v_\parallel
    \widehat{(\nabla \times \mathbf b_0) \cdot \mathbf b_0})`, as in
    :class:`~struphy.propagators.push_guiding_center_parallel.PushGuidingCenterParallel` and
    :class:`~struphy.propagators.push_guiding_center_bx_estar.PushGuidingCenterBxEstar`. The rank-one part is the
    parallel response, the antisymmetric part :math:`S_p` the :math:`E \times B` current, scaled by ``lam``
    (:math:`\lambda`).
    """

    markers = args_markers.markers

    dfm = empty((3, 3), dtype=float)
    b2 = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    unit_b1 = empty(3, dtype=float)
    a = empty(3, dtype=float)
    filling_m = empty((3, 3), dtype=float)

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

        # filling: w (a a^T + lam S), with S x = x cross b0 / B*_parallel
        for i in range(3):
            for j in range(3):
                filling_m[i, j] = weight * a[i] * a[j]

        s = weight * lam / b_star_parallel
        filling_m[0, 1] += s * unit_b1[2]
        filling_m[0, 2] -= s * unit_b1[1]
        filling_m[1, 0] -= s * unit_b1[2]
        filling_m[1, 2] += s * unit_b1[0]
        filling_m[2, 0] += s * unit_b1[1]
        filling_m[2, 1] -= s * unit_b1[0]

        particle_to_mat_kernels.m_v_fill_b_v1_full(
            args_derham,
            eta1,
            eta2,
            eta3,
            mat11,
            mat12,
            mat13,
            mat21,
            mat22,
            mat23,
            mat31,
            mat32,
            mat33,
            filling_m[0, 0],
            filling_m[0, 1],
            filling_m[0, 2],
            filling_m[1, 0],
            filling_m[1, 1],
            filling_m[1, 2],
            filling_m[2, 0],
            filling_m[2, 1],
            filling_m[2, 2],
            vec1,
            vec2,
            vec3,
            weight * v * a[0],
            weight * v * a[1],
            weight * v * a[2],
        )
