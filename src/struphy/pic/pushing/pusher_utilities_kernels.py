from numpy import empty, floor, sqrt, zeros
from pyccel.decorators import pure, stack_array

import struphy.bsplines.bsplines_kernels as bsplines_kernels
import struphy.geometry.evaluation_kernels as evaluation_kernels

# do not remove; needed to identify dependencies
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels
import struphy.linear_algebra.linalg_kernels as linalg_kernels
from struphy.bsplines.evaluation_kernels_3d import get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments


@stack_array("dfm", "dfinv", "eta", "v", "v_logical")
def reflect(
    markers: "float[:,:]",
    args_domain: "DomainArguments",
    outside_inds: "int[:]",
    axis: "int",
):
    r"""
    Reflect the particles which are pushed outside of the logical cube.

    .. math::

        \hat{v} = DF^{-1} v \,, \\
        \hat{v}_\text{reflected}[\text{axis}] = -1 * \hat{v} \,, \\
        v_\text{reflected} = DF \hat{v}_\text{reflected} \,.

    Parameters
    ----------
        markers : array[float]
            Local markers array

        args_domain : DomainArguments
            kind_map, params_map, ..., cx, cy, cz

        outside_inds : array[int]
            inds indicate the particles which are pushed outside of the local cube

        axis : int
            0, 1 or 2
    """

    # allocate metric coeffs
    dfm = zeros((3, 3), dtype=float)
    dfinv = zeros((3, 3), dtype=float)

    # marker position and velocity
    eta = empty(3, dtype=float)
    v = empty(3, dtype=float)
    v_logical = empty(3, dtype=float)

    for ip in outside_inds:
        eta[:] = markers[ip, 0:3]
        v[:] = markers[ip, 3:6]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta[0],
            eta[1],
            eta[2],
            args_domain,
            dfm,
        )

        linalg_kernels.matrix_inv(dfm, dfinv)

        # pull back of the velocity
        linalg_kernels.matrix_vector(dfinv, v, v_logical)

        # reverse the velocity
        v_logical[axis] *= -1

        # push forwward of the velocity
        linalg_kernels.matrix_vector(dfm, v_logical, v)

        # update the particle velocities
        markers[ip, 3:6] = v[:]
