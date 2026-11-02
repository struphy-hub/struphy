"Matrix-free application of the pressure-coupling particle matrix of full-orbit (6D) particles (MarkerOperator)."

from numpy import empty
from pyccel.decorators import stack_array

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.pic.accumulation.particle_to_mat_kernels as particle_to_mat_kernels
from struphy.bsplines.evaluation_kernels_3d import (
    eval_1form_spline_mpi,
    get_spans,
)
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("v", "s_h", "s_v")
def marker_operator_pressure_dot(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    vec1_1: "float[:,:,:]",
    vec2_1: "float[:,:,:]",
    vec3_1: "float[:,:,:]",
    vec1_2: "float[:,:,:]",
    vec2_2: "float[:,:,:]",
    vec3_2: "float[:,:,:]",
    vec1_3: "float[:,:,:]",
    vec2_3: "float[:,:,:]",
    vec3_3: "float[:,:,:]",
    s1_1: "float[:,:,:]",
    s2_1: "float[:,:,:]",
    s3_1: "float[:,:,:]",
    s1_2: "float[:,:,:]",
    s2_2: "float[:,:,:]",
    s3_2: "float[:,:,:]",
    s1_3: "float[:,:,:]",
    s2_3: "float[:,:,:]",
    s3_3: "float[:,:,:]",
    n_v: "int",
    column: "int",
):
    r"""Applies the 3x3 block particle matrix of the pressure coupling
    (:class:`~struphy.pic.accumulation.particles_to_grid.MarkerOperator` with ``n_blocks=3``),

    .. math::

        A_{ab} = \sum_p v_{p,a}\, v_{p,b}\, \vec \Lambda^1(\boldsymbol \eta_p)\, F_p\, \vec \Lambda^1(\boldsymbol \eta_p)^\top\,,
        \qquad a, b = 1, 2, 3\,,

    with the symmetric 3x3 filling :math:`F_p` (entries 11, 12, 13, 22, 23, 33 in the marker columns ``column`` to
    ``column + 5``, stored by
    :func:`~struphy.pic.accumulation.kernels.pc_lin_mhd_6d_mf.pc_lin_mhd_6d_mf_kernels.pc_lin_mhd_6d_mf`), to the
    1-form coefficients :math:`s = (s_1, s_2, s_3)` (component arrays ``s1_b, s2_b, s3_b`` of :math:`s_b`), with the
    filling of the block :math:`a`

    .. math::

        B_{p,a}^\mu = v_{p,a} \Big(F_p \sum_b v_{p,b}\, S_{h,b}(\boldsymbol \eta_p)\Big)_\mu\,,

    where :math:`S_{h,b}` is the 1-form spline function of :math:`s_b`. Only the first ``n_v`` velocity components
    are used (``n_v = 2``: :math:`\mathbf v_\perp = (v_1, v_2, 0)`, ``n_v = 3``: :math:`\mathbf v`).
    """

    markers = args_markers.markers

    v = empty(3, dtype=float)
    s_h = empty(3, dtype=float)
    s_v = empty(3, dtype=float)

    n_markers_loc = args_markers.n_markers

    for ip in range(n_markers_loc):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[0, ip] == -1.0:
            continue

        span1, span2, span3 = get_spans(markers[0, ip], markers[1, ip], markers[2, ip], args_derham)

        v[:] = markers[3:6, ip]
        if n_v == 2:
            v[2] = 0.0

        # s_v = sum_b v_b S_{h,b}
        eval_1form_spline_mpi(span1, span2, span3, args_derham, s1_1, s2_1, s3_1, s_h)
        s_v[:] = v[0] * s_h
        eval_1form_spline_mpi(span1, span2, span3, args_derham, s1_2, s2_2, s3_2, s_h)
        s_v[:] += v[1] * s_h
        if n_v == 3:
            eval_1form_spline_mpi(span1, span2, span3, args_derham, s1_3, s2_3, s3_3, s_h)
            s_v[:] += v[2] * s_h

        f11 = markers[column, ip]
        f12 = markers[column + 1, ip]
        f13 = markers[column + 2, ip]
        f22 = markers[column + 3, ip]
        f23 = markers[column + 4, ip]
        f33 = markers[column + 5, ip]

        y1 = f11 * s_v[0] + f12 * s_v[1] + f13 * s_v[2]
        y2 = f12 * s_v[0] + f22 * s_v[1] + f23 * s_v[2]
        y3 = f13 * s_v[0] + f23 * s_v[1] + f33 * s_v[2]

        particle_to_mat_kernels.vec_fill_v1(
            args_derham, span1, span2, span3, vec1_1, vec2_1, vec3_1, v[0] * y1, v[0] * y2, v[0] * y3
        )
        particle_to_mat_kernels.vec_fill_v1(
            args_derham, span1, span2, span3, vec1_2, vec2_2, vec3_2, v[1] * y1, v[1] * y2, v[1] * y3
        )
        if n_v == 3:
            particle_to_mat_kernels.vec_fill_v1(
                args_derham, span1, span2, span3, vec1_3, vec2_3, vec3_3, v[2] * y1, v[2] * y2, v[2] * y3
            )
