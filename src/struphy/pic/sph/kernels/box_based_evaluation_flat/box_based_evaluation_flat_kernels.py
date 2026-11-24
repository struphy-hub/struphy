"""Box-based SPH evaluation on a flat array of points, see :func:`~struphy.pic.sph_eval_kernels.box_based_physical`."""

from numpy import zeros
from pyccel.decorators import stack_array

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.pic.sorting_kernels as sorting_kernels
from struphy.kernel_arguments.pusher_args_kernels import MarkerArguments
from struphy.pic.sph_eval_kernels import box_based_physical


@stack_array("res")
def box_based_evaluation_flat(
    args_markers: "MarkerArguments",
    eta1: "float[:]",
    eta2: "float[:]",
    eta3: "float[:]",
    x1: "float[:]",
    x2: "float[:]",
    x3: "float[:]",
    n1: "int",
    n2: "int",
    n3: "int",
    domain_array: "float[:]",
    boxes: "int[:,:]",
    neighbours: "int[:,:]",
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
    r"""Box-based SPH evaluation on a flat array of points, see
    :func:`~struphy.pic.sph_eval_kernels.box_based_physical`.

    Parameters
    ----------
    args_markers : MarkerArguments
        Marker arguments.

    eta1, eta2, eta3 : float[:]
        Logical evaluation points.

    x1, x2, x3 : float[:]
        Physical evaluation points, :math:`F(\boldsymbol\eta)`.

    n1, n2, n3 : int
        Number of sorting boxes per dimension on this process.

    domain_array : float[:]
        Local MPI sub-domain, used by :func:`~struphy.pic.sorting_kernels.find_box`.

    boxes, neighbours : int[:,:]
        Sorting-box arrays.

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
        Output, same length as ``eta1``; points outside the local domain are left at zero.
    """
    res = zeros(4, dtype=float)

    n_eval = len(eta1)
    out[:] = 0.0
    for i in range(n_eval):
        loc_box = sorting_kernels.find_box(eta1[i], eta2[i], eta3[i], n1, n2, n3, domain_array)
        if loc_box == -1:
            continue
        box_based_physical(
            args_markers,
            x1[i],
            x2[i],
            x3[i],
            eta1[i],
            eta2[i],
            eta3[i],
            loc_box,
            boxes,
            neighbours,
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
