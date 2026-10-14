"""Evaluation of metric coefficients for given markers."""

from numpy import shape, zeros
from pyccel.decorators import stack_array

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
from struphy.geometry.evaluation_kernels import select_metric_coeff
from struphy.kernel_arguments.pusher_args_kernels import DomainArguments


@stack_array("tmp0", "tmp1", "tmp2", "tmp3", "out")
def kernel_evaluate_pic(
    markers: "float[:,:]",
    kind_coeff: int,
    args: "DomainArguments",
    mat_f: "float[:,:,:]",
    remove_outside: bool,
    avoid_round_off: bool,
) -> int:
    """
    Evaluation of metric coefficients for given markers.

    Parameters
    ----------
    remove_outside : bool
        Whether to remove values that originate from markers outside of [0, 1]^d.

    Returns
    -------
    counter : int
        How many markers have been treated (not been skipped).
    """
    tmp0 = zeros(3, dtype=float)
    tmp1 = zeros((3, 3), dtype=float)
    tmp2 = zeros((3, 3), dtype=float)
    tmp3 = zeros((3, 3), dtype=float)
    out = zeros((3, 3), dtype=float)

    np = shape(markers)[0]
    counter = 0

    for i in range(np):
        e1 = markers[i, 0]
        e2 = markers[i, 1]
        e3 = markers[i, 2]

        if e1 < 0.0 or e1 > 1.0 or e2 < 0.0 or e2 > 1.0 or e3 < 0.0 or e3 > 1.0:
            if remove_outside:
                continue
            else:
                if kind_coeff >= 0:
                    mat_f[counter, :, :] = -1.0
                else:
                    mat_f[counter, 0, 0] = e1
                    mat_f[counter, 1, 0] = e2
                    mat_f[counter, 2, 0] = e3
                counter += 1
        else:
            out[:] = mat_f[counter, :, :]

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

            mat_f[counter, :, :] = out

            counter += 1

    return counter
