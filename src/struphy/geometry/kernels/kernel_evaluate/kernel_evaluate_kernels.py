"""Evaluation of metric coefficients on a given 3d grid of evaluation points."""

from numpy import shape, zeros
from pyccel.decorators import stack_array

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
from struphy.geometry.evaluation_kernels import select_metric_coeff
from struphy.kernel_arguments.pusher_args_kernels import DomainArguments


@stack_array("tmp0", "tmp1", "tmp2", "tmp3", "out")
def kernel_evaluate(
    eta1: "float[:,:,:]",
    eta2: "float[:,:,:]",
    eta3: "float[:,:,:]",
    kind_coeff: int,
    args: "DomainArguments",
    mat_f: "float[:,:,:,:,:]",
    is_sparse_meshgrid: bool,
    avoid_round_off: bool,
):
    """
    Evaluation of metric coefficients on a given 3d grid of evaluation points.

    Parameters
    ----------
    is_sparse_meshgrid : bool
        Whether the 3d evaluation points were obtained from a sparse meshgrid.
    """
    tmp0 = zeros(3, dtype=float)
    tmp1 = zeros((3, 3), dtype=float)
    tmp2 = zeros((3, 3), dtype=float)
    tmp3 = zeros((3, 3), dtype=float)
    out = zeros((3, 3), dtype=float)

    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]

    if is_sparse_meshgrid:
        sparse_factor = 0
    else:
        sparse_factor = 1

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                e1 = eta1[i1, i2 * sparse_factor, i3 * sparse_factor]
                e2 = eta2[i1 * sparse_factor, i2, i3 * sparse_factor]
                e3 = eta3[i1 * sparse_factor, i2 * sparse_factor, i3]

                out[:] = mat_f[i1, i2, i3, :, :]

                select_metric_coeff(
                    e1,
                    e2,
                    e3,
                    kind_coeff,
                    args,
                    tmp0,
                    tmp1,
                    tmp2,
                    tmp3,
                    avoid_round_off,
                    out,
                )

                mat_f[i1, i2, i3, :, :] = out
