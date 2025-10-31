import struphy.pic.sorting_kernels as sorting_kernels
import struphy.pic.sph_smoothing_kernels as sph_smoothing_kernels
from struphy.kernel_arguments.pusher_args_kernels import MarkerArguments


def distance(x: "float", y: "float", periodic: "bool") -> float:
    """Return the one dimensional distance of x and y taking in account the periodicity on [0,1]."""
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
    """Naive single-point sph evaluation.
    The sum is done over all particles in markers array.

    Parameters
    ----------
    eta1, eta2, eta3 : float
        Evaluation point in logical space.

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

    kernel_type : str
        Name of the smoothing kernel.

    h1, h2, h3 : float
        Kernel width in respective dimension.
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


def boxed_based_kernel(
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
    """Box-based single-point sph evaluation.
    The sum is done over the particles that are in the 26 + 1 neighboring boxes
    of the ``loc_box`` the evaluation point is in.

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
    return out / Np


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
    """Naive flat sph evaluation.
    The sum is done over all particles in markers array.

    Parameters
    ----------
    eta1, eta2, eta3 : array[float]
        Evaluation points in logical space for flat evaluation at (eta1[i], eta2[i], eta3[i]).

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

    out : array[float]
        Output array of same size as eta1, eta2, eta3.
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
    """Naive meshgrid sph evaluation.
    The sum is done over all particles in markers array.

    Parameters
    ----------
    eta1, eta2, eta3 : array[float]
        Evaluation points in logical space for meshgrid evaluation at (eta1[i,j,k], eta2[i,j,k], eta3[i,j,k]).

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

    out : array[float]
        Output array of same size as eta1, eta2, eta3.
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
                    args_markers, e1, e2, e3, holes, periodic1, periodic2, periodic3, index, kernel_type, h1, h2, h3
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
    """Box-based flat sph evaluation.
    The sum is done over the particles that are in the 26 + 1 neighboring boxes
    of the ``loc_box`` the evaluation point is in.

    Parameters
    ----------
    eta1, eta2, eta3 : array[float]
        Evaluation points in logical space for flat evaluation at (eta1[i], eta2[i], eta3[i]).

    n1, n2, n3 : int
        Number of boxes in each dimension.

    domain_array : array
        Information of the domain on the current mpi process.

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

    out : array[float]
        Output array of same size as eta1, eta2, eta3.
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
            out[i] = boxed_based_kernel(
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
    """Box-based meshgrid sph evaluation.
    The sum is done over the particles that are in the 26 + 1 neighboring boxes
    of the ``loc_box`` the evaluation point is in.

    Parameters
    ----------
    eta1, eta2, eta3 : array[float]
        Evaluation points in logical space for meshgrid evaluation at (eta1[i,j,k], eta2[i,j,k], eta3[i,j,k]).

    n1, n2, n3 : int
        Number of boxes in each dimension.

    domain_array : array
        Information of the domain on the current mpi process.

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

    out : array[float]
        Output array of same size as eta1, eta2, eta3.
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
                    out[i, j, k] = boxed_based_kernel(
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
