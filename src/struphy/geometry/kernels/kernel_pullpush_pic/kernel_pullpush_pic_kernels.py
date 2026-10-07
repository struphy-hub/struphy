"""Pull-backs, pushforwards and transformations for given markers."""

from numpy import shape, zeros
from pyccel.decorators import stack_array

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
from struphy.geometry.transform_kernels import pull, push, tran
from struphy.kernel_arguments.pusher_args_kernels import DomainArguments


@stack_array("tmp1", "tmp2")
def kernel_pullpush_pic(
    a: "float[:,:]",
    markers: "float[:,:]",
    kind_transform: int,
    kind_fun: int,
    args_domain: "DomainArguments",
    out: "float[:,:]",
    remove_outside: bool,
) -> int:
    """
    Pull-backs, pushforwards and transformations for given markers.

    Parameters
    ----------
    a : float[:,:]
        Values of scalar function a[0, ip] or values of components of a vector valued function (a[0, ip], a[1, ip], a[2, ip]).

    markers : float[:,:]
        Evaluation points in marker format (eta1 = markers[:, 0], eta2 = markers[:, 1], eta3 = markers[:, 2]).

    kind_transform : int
        Which general transformation to be performed (pull, push or tran).

    kind_fun : int
        Which detailed transformation to be performed.

    args_domain : DomainArguments
        Domain info.

    out : float[:,:]
        Output values.

    remove_outside : bool
        Whether to remove values that originate from markers outside of [0, 1]^d.
    """

    tmp1 = zeros(shape(a)[1], dtype=float)
    tmp2 = zeros(shape(out)[1], dtype=float)
    # tmp1 = zeros((3,), dtype=float)
    # tmp2 = zeros((3,), dtype=float)

    np = shape(markers)[0]

    # check if a has holes or not
    if shape(a)[0] == np:
        a_has_holes = True
    else:
        a_has_holes = False

    counter_a = 0
    counter_o = 0

    for i in range(np):
        e1 = markers[i, 0]
        e2 = markers[i, 1]
        e3 = markers[i, 2]

        # treatment of a hole
        if e1 < 0.0 or e1 > 1.0 or e2 < 0.0 or e2 > 1.0 or e3 < 0.0 or e3 > 1.0:
            # skip value in a
            if a_has_holes:
                counter_a += 1

            if remove_outside:
                continue
            else:
                out[counter_o, :] = -1.0
                counter_o += 1

        # treatment of "true" marker
        else:
            tmp1[:] = a[counter_a, :]
            tmp2[:] = out[counter_o, :]

            if kind_transform == 0:
                pull(tmp1, e1, e2, e3, kind_fun, args_domain, tmp2)
            elif kind_transform == 1:
                push(tmp1, e1, e2, e3, kind_fun, args_domain, tmp2)
            else:
                tran(tmp1, e1, e2, e3, kind_fun, args_domain, tmp2)

            out[counter_o, :] = tmp2

            counter_a += 1
            counter_o += 1

    return counter_o
