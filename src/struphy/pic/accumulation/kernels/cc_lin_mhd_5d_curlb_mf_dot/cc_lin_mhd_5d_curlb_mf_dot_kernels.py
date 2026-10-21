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


@stack_array("x")
def cc_lin_mhd_5d_curlb_mf_dot(
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
    r"""Applies the matrix :math:`A = \sum_p s_p (\vec \Lambda_p \cdot \mathbf e_p) (\vec \Lambda_p \cdot \mathbf e_p)^\top`
    stored by
    :func:`~struphy.pic.accumulation.kernels.cc_lin_mhd_5d_curlb_mf.cc_lin_mhd_5d_curlb_mf_kernels.cc_lin_mhd_5d_curlb_mf`
    (:math:`\mathbf e_p` and :math:`s_p` in the marker columns ``column`` to ``column + 3``) to the FE coefficients
    ``x``:

    .. math::

        B_p^\mu = s_p \, \left(\mathbf e_p \cdot \mathbf x_h(\boldsymbol \eta_p)\right) e_{p,\mu}\,,

    with the spline :math:`\mathbf x_h` of the coefficients ``x`` (2-form for ``basis_u = 2``, vector field for
    ``basis_u = 0``).
    """

    markers = args_markers.markers

    x = empty(3, dtype=float)

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
            eval_vectorfield_spline_mpi(span1, span2, span3, args_derham, x1, x2, x3, x)
        elif basis_u == 2:
            eval_2form_spline_mpi(span1, span2, span3, args_derham, x1, x2, x3, x)

        fill = markers[column + 3, ip] * (e1 * x[0] + e2 * x[1] + e3 * x[2])

        if basis_u == 0:
            particle_to_mat_kernels.vec_fill_v0vec(
                args_derham, span1, span2, span3, vec1, vec2, vec3, fill * e1, fill * e2, fill * e3
            )
        elif basis_u == 2:
            particle_to_mat_kernels.vec_fill_v2(
                args_derham, span1, span2, span3, vec1, vec2, vec3, fill * e1, fill * e2, fill * e3
            )
