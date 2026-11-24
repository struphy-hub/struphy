"Matrix-free application of a particle matrix with symmetric fillings (MarkerOperator)."

from numpy import empty
from pyccel.decorators import stack_array

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.pic.accumulation.particle_to_mat_kernels as particle_to_mat_kernels
from struphy.bsplines.evaluation_kernels_3d import (
    eval_1form_spline_mpi,
    eval_2form_spline_mpi,
    eval_vectorfield_spline_mpi,
    get_spans,
)
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("s_h")
def marker_operator_symm_dot(
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
    r"""Applies the particle matrix (:class:`~struphy.pic.accumulation.particles_to_grid.MarkerOperator`) with the
    symmetric 3x3 filling :math:`F_p`, with the entries 11, 12, 13, 22, 23, 33 stored in the marker columns
    ``column`` to ``column + 5`` (e.g. by
    :func:`~struphy.pic.accumulation.kernels.vlasov_maxwell_mf.vlasov_maxwell_mf_kernels.vlasov_maxwell_mf`),
    to the FE coefficients :math:`s` (component arrays ``s1, s2, s3``), with the filling

    .. math::

        B_p^\mu = \big(F_p\, S_h(\boldsymbol \eta_p)\big)_\mu\,,

    where :math:`S_h` is the spline function of :math:`s` (vector field, 1-form or 2-form for ``basis_u = 0, 1, 2``).
    """

    markers = args_markers.markers

    s_h = empty(3, dtype=float)

    n_markers_loc = args_markers.n_markers

    for ip in range(n_markers_loc):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[0, ip] == -1.0:
            continue

        span1, span2, span3 = get_spans(markers[0, ip], markers[1, ip], markers[2, ip], args_derham)

        f11 = markers[column, ip]
        f12 = markers[column + 1, ip]
        f13 = markers[column + 2, ip]
        f22 = markers[column + 3, ip]
        f23 = markers[column + 4, ip]
        f33 = markers[column + 5, ip]

        if basis_u == 0:
            eval_vectorfield_spline_mpi(span1, span2, span3, args_derham, s1, s2, s3, s_h)
        elif basis_u == 1:
            eval_1form_spline_mpi(span1, span2, span3, args_derham, s1, s2, s3, s_h)
        elif basis_u == 2:
            eval_2form_spline_mpi(span1, span2, span3, args_derham, s1, s2, s3, s_h)

        y1 = f11 * s_h[0] + f12 * s_h[1] + f13 * s_h[2]
        y2 = f12 * s_h[0] + f22 * s_h[1] + f23 * s_h[2]
        y3 = f13 * s_h[0] + f23 * s_h[1] + f33 * s_h[2]

        if basis_u == 0:
            particle_to_mat_kernels.vec_fill_v0vec(args_derham, span1, span2, span3, vec1, vec2, vec3, y1, y2, y3)
        elif basis_u == 1:
            particle_to_mat_kernels.vec_fill_v1(args_derham, span1, span2, span3, vec1, vec2, vec3, y1, y2, y3)
        elif basis_u == 2:
            particle_to_mat_kernels.vec_fill_v2(args_derham, span1, span2, span3, vec1, vec2, vec3, y1, y2, y3)
