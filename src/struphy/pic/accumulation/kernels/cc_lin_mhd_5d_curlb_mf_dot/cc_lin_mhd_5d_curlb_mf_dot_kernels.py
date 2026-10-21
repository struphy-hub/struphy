"Matrix-free application of the accumulated matrix of cc_lin_mhd_5d_curlb_mf (gyro-center (5D) particles)."

from numpy import empty
from pyccel.decorators import stack_array

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.pic.accumulation.particle_to_mat_kernels as particle_to_mat_kernels
from struphy.bsplines.evaluation_kernels_3d import (
    eval_2form_spline_mpi,
    eval_vectorfield_spline_mpi,
    get_spans,
)
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("s_h")
def cc_lin_mhd_5d_curlb_mf_dot(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    vec1: "float[:,:,:]",
    vec2: "float[:,:,:]",
    vec3: "float[:,:,:]",
    s1: "float[:,:,:]",
    s2: "float[:,:,:]",
    s3: "float[:,:,:]",
    basis_u: "int",
    column: "int",
):
    r"""Applies the particle matrix with the rank-one filling :math:`F_p = c_p\, \mathbf e_p \mathbf e_p^\top`
    (:class:`~struphy.pic.accumulation.particles_to_grid.MarkerOperator`) stored by
    :func:`~struphy.pic.accumulation.kernels.cc_lin_mhd_5d_curlb_mf.cc_lin_mhd_5d_curlb_mf_kernels.cc_lin_mhd_5d_curlb_mf`
    (:math:`\mathbf e_p` and :math:`c_p` in the marker columns ``column`` to ``column + 3``) to the FE coefficients
    :math:`s` (component arrays ``s1, s2, s3``), with the filling

    .. math::

        B_p^\mu = \big(F_p\, S_h(\boldsymbol \eta_p)\big)_\mu
        = c_p \, \big(\mathbf e_p \cdot S_h(\boldsymbol \eta_p)\big)\, e_{p,\mu}\,,

    where :math:`S_h` is the spline function of :math:`s` (2-form for ``basis_u = 2``, vector field for
    ``basis_u = 0``).
    """

    markers = args_markers.markers

    s_h = empty(3, dtype=float)

    n_markers_loc = args_markers.n_markers

    for ip in range(n_markers_loc):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[0, ip] == -1.0:
            continue

        span1, span2, span3 = get_spans(markers[0, ip], markers[1, ip], markers[2, ip], args_derham)

        e1 = markers[column, ip]
        e2 = markers[column + 1, ip]
        e3 = markers[column + 2, ip]

        if basis_u == 0:
            eval_vectorfield_spline_mpi(span1, span2, span3, args_derham, s1, s2, s3, s_h)
        elif basis_u == 2:
            eval_2form_spline_mpi(span1, span2, span3, args_derham, s1, s2, s3, s_h)

        fill = markers[column + 3, ip] * (e1 * s_h[0] + e2 * s_h[1] + e3 * s_h[2])

        if basis_u == 0:
            particle_to_mat_kernels.vec_fill_v0vec(
                args_derham, span1, span2, span3, vec1, vec2, vec3, fill * e1, fill * e2, fill * e3
            )
        elif basis_u == 2:
            particle_to_mat_kernels.vec_fill_v2(
                args_derham, span1, span2, span3, vec1, vec2, vec3, fill * e1, fill * e2, fill * e3
            )
