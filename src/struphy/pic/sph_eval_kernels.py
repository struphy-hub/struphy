# do not remove; needed to identify dependencies
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # noqa: PLR0402
import struphy.pic.sorting_kernels as sorting_kernels  # noqa: PLR0402
import struphy.pic.sph_smoothing_kernels as sph_smoothing_kernels  # noqa: PLR0402
from struphy.kernel_arguments.pusher_args_kernels import MarkerArguments


def distance(x: "float", y: "float", periodic: "bool") -> float:
    r"""Return the signed one-dimensional distance ``x - y``, adjusted for periodicity on ``[0, 1]``.

    Parameters
    ----------
    x, y : float
        Two coordinates on the domain.

    periodic : bool
        If ``True``, the result is folded into :math:`(-\tfrac{1}{2}, \tfrac{1}{2}]` so that
        the shortest path across the periodic boundary is returned.

    Returns
    -------
    float
        Signed distance ``x - y``, adjusted for periodicity.
    """
    d = x - y
    if periodic:
        if d > 0.5:
            while d > 0.5:
                d -= 1.0
        elif d < -0.5:
            while d < -0.5:
                d += 1.0
    return d


########################
# single-point kernels #
########################
def naive_evaluation_kernel(
    args_markers: "MarkerArguments",
    eta1: "float",
    eta2: "float",
    eta3: "float",
    holes: "bool[:]",
    periodic1: "bool",
    periodic2: "bool",
    periodic3: "bool",
    index: "int",
    kernel_type: "int",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    r"""Perform a single-point SPH evaluation of a function :math:`\rho: [0, 1]^3 \to \mathbb R` in the following sense:

    .. math::

        \rho(\boldsymbol \eta_i) = \sum_{j=0}^{N-1} \rho_j\, W_h(\boldsymbol \eta_i - \boldsymbol \eta_j)\,.

    The coefficients :math:`\rho_j` must be available in the marker array, stored at some index ``self.markers[j, index]``.
    In case that `derivative=k` where `k` is not zero, the `k`-th component of the gradient of :math:`\rho` is computed:

    .. math::

        \textrm{derivative}=k:\qquad [\nabla \rho(\boldsymbol \eta_i)]_k = \sum_{j=0}^{N-1} \rho_j \frac{\partial W_h}{\partial \eta_k}(\boldsymbol \eta_i - \boldsymbol \eta_j)\,.

    The possible choices for :math:`W_h` are listed in :ref:`smoothing_kernels`
    and in :meth:`~struphy.pic.base.Particles.ker_dct`.

    ATTENTION: The sum is done over all particles in the markers array (ignoring holes), no neighbour search is performed.
    Hence, the cost of this evaluation is :math:`\mathcal{O}(N)` in the number of particles, and it should only be used for testing and verification purposes.

    Parameters
    ----------
    args_markers : MarkerArguments
        Container holding the markers array and the total number of particles ``Np``.

    eta1, eta2, eta3 : float
        Evaluation point in logical space.

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

    Returns
    -------
    float
        SPH estimate of :math:`b` at the evaluation point.
    """

    markers = args_markers.markers
    Np = args_markers.Np

    n_particles = len(markers)
    out = 0.0
    for p in range(n_particles):
        if not holes[p]:
            r1 = distance(eta1, markers[p, 0], periodic1)
            r2 = distance(eta2, markers[p, 1], periodic2)
            r3 = distance(eta3, markers[p, 2], periodic3)
            out += markers[p, index] * sph_smoothing_kernels.smoothing_kernel(kernel_type, r1, r2, r3, h1, h2, h3)
    return out / Np


def box_based_kernel(
    args_markers: "MarkerArguments",
    eta1: "float",
    eta2: "float",
    eta3: "float",
    loc_box: "int",
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
) -> float:
    r"""Perform a single-point SPH evaluation of a function :math:`\rho: [0, 1]^3 \to \mathbb R` in the following sense:

    .. math::

        \rho(\boldsymbol \eta_i) = \sum_{j=0}^{N-1} \rho_j\, W_h(\boldsymbol \eta_i - \boldsymbol \eta_j)\,.

    The coefficients :math:`\rho_j` must be available in the marker array, stored at some index ``self.markers[j, index]``.
    In case that `derivative=k` where `k` is not zero, the `k`-th component of the gradient of :math:`\rho` is computed:

    .. math::

        \textrm{derivative}=k:\qquad [\nabla \rho(\boldsymbol \eta_i)]_k = \sum_{j=0}^{N-1} \rho_j \frac{\partial W_h}{\partial \eta_k}(\boldsymbol \eta_i - \boldsymbol \eta_j)\,.

    The possible choices for :math:`W_h` are listed in :ref:`smoothing_kernels`
    and in :meth:`~struphy.pic.base.Particles.ker_dct`.

    The sum is restricted to the 27 neighbouring boxes of the box containing
    :math:`\boldsymbol\eta_i`, making the cost :math:`\mathcal{O}(1)` in the number
    of particles when the kernel support is proportional to the box size.

    Parameters
    ----------
    eta1, eta2, eta3 : float
        Evaluation point in logical space.

    loc_box : int
        Box of the evaluation point.

    boxes : 2d array
        Box array of the sorting boxes structure.

    neighbours : 2d array
        Array containing the 27 neighbouring boxes of each box.

    markers : array[float]
        Markers array.

    Np : int
        Total number of particles.

    holes : bool
        1D array of length markers.shape[0]. True if markers[i] is a hole.

    periodic1, periodic2, periodic3 : bool
        True if periodic in that dimension.

    index : int
        Column index in markers array where the value multiplying the kernel in the evaluation is stored.

    kernel_type : int
        Number of the smoothing kernel.

    h1, h2, h3 : float
        Kernel width in respective dimension.
    """

    markers = args_markers.markers
    Np = args_markers.Np

    out = 0.0
    for neigh in range(27):
        box_to_search = neighbours[loc_box, neigh]
        c = 0
        # loop over all particles in a box
        while boxes[box_to_search, c] != -1:
            p = boxes[box_to_search, c]
            c += 1
            if not holes[p]:
                r1 = distance(eta1, markers[p, 0], periodic1)
                r2 = distance(eta2, markers[p, 1], periodic2)
                r3 = distance(eta3, markers[p, 2], periodic3)
                out += markers[p, index] * sph_smoothing_kernels.smoothing_kernel(kernel_type, r1, r2, r3, h1, h2, h3)
    return out


####################
# naive evaluation #
####################
def naive_evaluation_flat(
    args_markers: "MarkerArguments",
    eta1: "float[:]",
    eta2: "float[:]",
    eta3: "float[:]",
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
    r"""Naive SPH evaluation on a flat array of points, see :func:`~struphy.pic.sph_eval_kernels.naive_evaluation_kernel`.

    Parameters
    ----------
    args_markers : MarkerArguments
        Container holding the markers array and the total number of particles ``Np``.

    eta1, eta2, eta3 : float[:]
        Evaluation points in logical space.  The :math:`i`-th point is
        ``(eta1[i], eta2[i], eta3[i])``.

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
        Output array of the same length as ``eta1``.  Modified in place and also returned.
    """

    markers = args_markers.markers
    Np = args_markers.Np

    n_eval = len(eta1)
    out[:] = 0.0
    for i in range(n_eval):
        e1 = eta1[i]
        e2 = eta2[i]
        e3 = eta3[i]
        out[i] = naive_evaluation_kernel(
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
    return out


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


########################
# box-based evaluation #
########################
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


def box_based_evaluation_meshgrid(
    args_markers: "MarkerArguments",
    eta1: "float[:,:,:]",
    eta2: "float[:,:,:]",
    eta3: "float[:,:,:]",
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
    out: "float[:,:,:]",
):
    r"""Box-based SPH evaluation on a 3-D meshgrid of points, see :func:`~struphy.pic.sph_eval_kernels.box_based_kernel`.

    Parameters
    ----------
    args_markers : MarkerArguments
        Container holding the markers array and the total number of particles ``Np``.

    eta1, eta2, eta3 : float[:,:,:]
        Evaluation points in logical space on a 3-D meshgrid.

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

    out : float[:,:,:]
        Output array of the same shape as ``eta1``.  Modified in place.
        Points outside the local domain are left at zero.
    """

    markers = args_markers.markers
    Np = args_markers.Np

    n_eval_1 = eta1.shape[0]
    n_eval_2 = eta1.shape[1]
    n_eval_3 = eta1.shape[2]
    out[:] = 0.0
    for i in range(n_eval_1):
        e1 = eta1[i, 0, 0]

        if e1 < domain_array[0] or e1 >= domain_array[1] and e1 != 1.0:
            continue

        for j in range(n_eval_2):
            e2 = eta2[0, j, 0]

            if e2 < domain_array[3] or e2 >= domain_array[4] and e2 != 1.0:
                continue

            for k in range(n_eval_3):
                e3 = eta3[0, 0, k]

                if e3 < domain_array[6] or e3 >= domain_array[7] and e3 != 1.0:
                    continue

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
                    out[i, j, k] = box_based_kernel(
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
