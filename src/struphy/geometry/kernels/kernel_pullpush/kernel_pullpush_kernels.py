"""Pull-backs, pushforwards and transformations on a given 3d grid of evaluation points."""

from numpy import shape, zeros
from pyccel.decorators import stack_array

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
from struphy.geometry.transform_kernels import pull, push, tran
from struphy.kernel_arguments.pusher_args_kernels import DomainArguments


@stack_array("tmp1", "tmp2")
def kernel_pullpush(
    a: "float[:,:,:,:]",
    eta1: "float[:,:,:]",
    eta2: "float[:,:,:]",
    eta3: "float[:,:,:]",
    kind_transform: int,
    kind_fun: int,
    args_domain: "DomainArguments",
    is_sparse_meshgrid: bool,
    out: "float[:,:,:,:]",
):
    """
    Pull-backs, pushforwards and transformations on a given 3d grid of evaluation points.

    Parameters
    ----------
    a : float[:,:,:,:]
        3d values of scalar function a[0, i, j, k] or 3d values of components of vector valued function a[:, i, j, k].

    eta1, eta2, eta3 : float[:,:,:]
        3d evaluation point sets.

    kind_transform : int
        Which general transformation to be performed (pull, push or tran).

    kind_fun : int
        Which detailed transformation to be performed.

    args_domain : DomainArguments
        Domain info.

    is_sparse_meshgrid : bool
        Whether the evaluation points were obtained from a sparse meshgrid.

    out : float[:,:,:,:]
        Output values.
    """

    tmp1 = zeros(shape(a)[-1], dtype=float)
    tmp2 = zeros(shape(out)[-1], dtype=float)
    # tmp1 = zeros(3, dtype=float)
    # tmp2 = zeros(3, dtype=float)

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

                tmp1[:] = a[i1, i2, i3, :]
                tmp2[:] = out[i1, i2, i3, :]

                if kind_transform == 0:
                    pull(tmp1, e1, e2, e3, kind_fun, args_domain, tmp2)
                elif kind_transform == 1:
                    push(tmp1, e1, e2, e3, kind_fun, args_domain, tmp2)
                else:
                    tran(tmp1, e1, e2, e3, kind_fun, args_domain, tmp2)

                out[i1, i2, i3, :] = tmp2
