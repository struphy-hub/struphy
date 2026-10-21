"Matrix-free application of the density matrix of cc_lin_mhd_5d_D_mf (gyro-center (5D) particles)."

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


@stack_array("x")
def cc_lin_mhd_5d_D_mf_dot(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    vec1: "float[:,:,:]",
    vec2: "float[:,:,:]",
    vec3: "float[:,:,:]",
    x1: "float[:,:,:]",
    x2: "float[:,:,:]",
    x3: "float[:,:,:]",
    basis_u: "int",
    column: "int",
):
    r"""Applies the matrix :math:`A = \sum_p \vec \Lambda_p F_p \vec \Lambda_p^\top` with the antisymmetric
    :math:`F_p` stored by
    :func:`~struphy.pic.accumulation.kernels.cc_lin_mhd_5d_D_mf.cc_lin_mhd_5d_D_mf_kernels.cc_lin_mhd_5d_D_mf`
    (entries 12, 13, 23 in the marker columns ``column`` to ``column + 2``) to the FE coefficients ``x``:

    .. math::

        B_p^\mu = \left(F_p\, \mathbf x_h(\boldsymbol \eta_p)\right)_\mu\,,

    with the spline :math:`\mathbf x_h` of the coefficients ``x`` (vector field, 1-form or 2-form for
    ``basis_u = 0, 1, 2``).
    """

    markers = args_markers.markers

    x = empty(3, dtype=float)

    n_markers_loc = args_markers.n_markers

    for ip in range(n_markers_loc):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[0, ip] == -1.0:
            continue

        span1, span2, span3 = get_spans(markers[0, ip], markers[1, ip], markers[2, ip], args_derham)

        f12 = markers[column, ip]
        f13 = markers[column + 1, ip]
        f23 = markers[column + 2, ip]

        if basis_u == 0:
            eval_vectorfield_spline_mpi(span1, span2, span3, args_derham, x1, x2, x3, x)
        elif basis_u == 1:
            eval_1form_spline_mpi(span1, span2, span3, args_derham, x1, x2, x3, x)
        elif basis_u == 2:
            eval_2form_spline_mpi(span1, span2, span3, args_derham, x1, x2, x3, x)

        # F x with F = [[0, f12, f13], [-f12, 0, f23], [-f13, -f23, 0]]
        y1 = f12 * x[1] + f13 * x[2]
        y2 = -f12 * x[0] + f23 * x[2]
        y3 = -f13 * x[0] - f23 * x[1]

        if basis_u == 0:
            particle_to_mat_kernels.vec_fill_v0vec(args_derham, span1, span2, span3, vec1, vec2, vec3, y1, y2, y3)
        elif basis_u == 1:
            particle_to_mat_kernels.vec_fill_v1(args_derham, span1, span2, span3, vec1, vec2, vec3, y1, y2, y3)
        elif basis_u == 2:
            particle_to_mat_kernels.vec_fill_v2(args_derham, span1, span2, span3, vec1, vec2, vec3, y1, y2, y3)
