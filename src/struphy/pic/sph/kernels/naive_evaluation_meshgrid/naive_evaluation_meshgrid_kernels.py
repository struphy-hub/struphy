"""Naive SPH evaluation on a 3-D meshgrid of points, see :func:`~struphy.pic.sph_eval_kernels.naive_evaluation_kernel`."""

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
from struphy.kernel_arguments.pusher_args_kernels import MarkerArguments
from struphy.pic.sph_eval_kernels import naive_evaluation_kernel


def naive_evaluation_meshgrid(
    args_markers: "MarkerArguments",
    eta1: "float[:,:,:]",
    eta2: "float[:,:,:]",
    eta3: "float[:,:,:]",
    holes: "bool[:]",
    periodic1: "bool",
    periodic2: "bool",
    periodic3: "bool",
    index: "int",
    kernel_type: "int",
    h1: "float",
    h2: "float",
    h3: "float",
    out: "float[:,:,:]",
):
    r"""Naive SPH evaluation on a 3-D meshgrid of points, see :func:`~struphy.pic.sph_eval_kernels.naive_evaluation_kernel`.

    Parameters
    ----------
    args_markers : MarkerArguments
        Container holding the markers array and the total number of particles ``Np``.

    eta1, eta2, eta3 : float[:,:,:]
        Evaluation points in logical space on a 3-D meshgrid.

    holes : bool[:]
        1D array of length ``markers.shape[0]``.  ``True`` if particle ``i`` is a hole (inactive).

    periodic1, periodic2, periodic3 : bool
        ``True`` if the domain is periodic in that dimension.

    index : int
        Column index in the markers array of the coefficient :math:`\beta_k` multiplying the kernel.

    kernel_type : int
        Integer identifier of the smoothing kernel.  See :ref:`smoothing_kernels`.

    h1, h2, h3 : float
        Kernel width in the respective dimension.

    out : float[:,:,:]
        Output array of the same shape as ``eta1``.  Modified in place.
    """

    markers = args_markers.markers
    Np = args_markers.Np

    n_eval_1 = eta1.shape[0]
    n_eval_2 = eta1.shape[1]
    n_eval_3 = eta1.shape[2]
    out[:] = 0.0
    for i in range(n_eval_1):
        for j in range(n_eval_2):
            for k in range(n_eval_3):
                e1 = eta1[i, j, k]
                e2 = eta2[i, j, k]
                e3 = eta3[i, j, k]
                out[i, j, k] = naive_evaluation_kernel(
                    args_markers,
                    e1,
                    e2,
                    e3,
                    holes,
                    periodic1,
                    periodic2,
                    periodic3,
                    index,
                    kernel_type,
                    h1,
                    h2,
                    h3,
                )
