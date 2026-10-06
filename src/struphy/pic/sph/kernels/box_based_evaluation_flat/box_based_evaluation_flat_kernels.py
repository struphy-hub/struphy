"""Box-based SPH evaluation on a flat array of points, see :func:`~struphy.pic.sph_eval_kernels.box_based_kernel`."""

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.pic.sorting_kernels as sorting_kernels
from struphy.kernel_arguments.pusher_args_kernels import MarkerArguments
from struphy.pic.sph_eval_kernels import box_based_kernel


def box_based_evaluation_flat(
    args_markers: "MarkerArguments",
    eta1: "float[:]",
    eta2: "float[:]",
    eta3: "float[:]",
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
    index: "int",
    kernel_type: "int",
    h1: "float",
    h2: "float",
    h3: "float",
    out: "float[:]",
):
    r"""Box-based SPH evaluation on a flat array of points, see :func:`~struphy.pic.sph_eval_kernels.box_based_kernel`.

    Parameters
    ----------
    args_markers : MarkerArguments
        Container holding the markers array and the total number of particles ``Np``.

    eta1, eta2, eta3 : float[:]
        Evaluation points in logical space.  The :math:`i`-th point is
        ``(eta1[i], eta2[i], eta3[i])``.

    n1, n2, n3 : int
        Number of sorting boxes in each dimension.

    domain_array : float[:]
        Flat description of the local MPI sub-domain, used by
        :func:`~struphy.pic.sorting_kernels.find_box` to locate boxes.

    boxes : int[:,:]
        Box array of the sorting-box structure (particles sorted into boxes).

    neighbours : int[:,:]
        ``neighbours[b, :]`` lists the 27 box indices neighbouring box ``b``.

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

    out : float[:]
        Output array of the same length as ``eta1``.  Modified in place.
        Points outside the local domain are left at zero.
    """

    markers = args_markers.markers
    Np = args_markers.Np

    n_eval = len(eta1)
    out[:] = 0.0
    for i in range(n_eval):
        e1 = eta1[i]
        e2 = eta2[i]
        e3 = eta3[i]
        loc_box = sorting_kernels.find_box(
            e1,
            e2,
            e3,
            n1,
            n2,
            n3,
            domain_array,
        )
        if loc_box == -1:
            continue
        else:
            out[i] = box_based_kernel(
                args_markers,
                e1,
                e2,
                e3,
                loc_box,
                boxes,
                neighbours,
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
