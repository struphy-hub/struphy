"""Naive SPH evaluation on a flat array of points, see :func:`~struphy.pic.sph_eval_kernels.naive_physical`."""

from numpy import zeros
from pyccel.decorators import stack_array

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.pic.sph_eval_kernels as sph_eval_kernels
from struphy.kernel_arguments.pusher_args_kernels import MarkerArguments


@stack_array("res")
def naive_evaluation_flat(
    args_markers: "MarkerArguments",
    eta1: "float[:]",
    eta2: "float[:]",
    eta3: "float[:]",
    x1: "float[:]",
    x2: "float[:]",
    x3: "float[:]",
    holes: "bool[:]",
    periodic1: "bool",
    periodic2: "bool",
    periodic3: "bool",
    period: "float[:,:]",
    proj: "float[:,:]",
    x_idx: "int",
    index: "int",
    kernel: "int",
    dim: "int",
    h: "float",
    scale: "float",
    component: "int",
    out: "float[:]",
):
    r"""Naive SPH evaluation (sum over all markers, ghost particles skipped) on a flat array of points, see
    :func:`~struphy.pic.sph_eval_kernels.naive_physical`. The cost is :math:`O(N)` per point; for testing.

    Parameters
    ----------
    args_markers : MarkerArguments
        Marker arguments.

    eta1, eta2, eta3 : float[:]
        Logical evaluation points.

    x1, x2, x3 : float[:]
        Physical evaluation points, :math:`F(\boldsymbol\eta)`.

    holes : bool[:]
        ``True`` for holes.

    periodic1, periodic2, periodic3 : bool
        Periodicity of the logical directions.

    period, proj : float[:,:]
        Period vectors and projection onto the resolved directions (3 x 3).

    x_idx, index : int
        Marker columns of the physical position and of the coefficients.

    kernel, dim : int
        Kernel id and number of resolved directions.

    h, scale : float
        Kernel width (physical) and kernel factor :math:`1/A_\perp`.

    component : int
        0 for the value, 1, 2, 3 for the components of the physical gradient.

    out : float[:]
        Output, same length as ``eta1``.
    """
    res = zeros(4, dtype=float)

    n_eval = len(eta1)
    for i in range(n_eval):
        sph_eval_kernels.naive_physical(
            args_markers,
            x1[i],
            x2[i],
            x3[i],
            eta1[i],
            eta2[i],
            eta3[i],
            holes,
            periodic1,
            periodic2,
            periodic3,
            period,
            proj,
            x_idx,
            index,
            kernel,
            dim,
            h,
            scale,
            res,
        )
        out[i] = res[component]
