from numpy import sqrt

import struphy.pic.sorting_kernels as sorting_kernels


def smoothing_kernel(r: "float", h: "float"):
    """Evaluate the smoothing kernel S(r,h) = C(h)F(r/h)
    With F(x) = 1-x if x<1, 0 else
    and C(h)=3/(pi*h^3) is a normalization coefficient so the the kernel has unit integral."""
    if r / h > 1.0:
        return 0.0
    else:
        return (1.0 - r / h) / (1.0471975512 * h**3)  # normalization


def periodic_distance(x: "float", y: "float"):
    """Return the one dimensional distance of x and y taking in account the periodicity on [0,1]."""
    d = x - y
    if d > 0.5:
        while d > 0.5:
            d -= 1.0
    elif d < -0.5:
        while d < -0.5:
            d += 1.0
    return d


def naive_evaluation(
    eta1: "float[:]",
    eta2: "float[:]",
    eta3: "float[:]",
    markers: "float[:,:]",
    holes: "bool[:]",
    index: "int",
    h: "float",
    out: "float[:]",
):
    """Naive evaluation of a function defined by its values at the particles.
    This is done in an efficient way, looping over the particles.
    Entries have to be given as 3 1d array (representing the coordinate in each direction)."""
    n_eval = len(eta1)
    n_particles = len(markers)
    out[:] = 0.0
    for i in range(n_eval):
        for p in range(n_particles):
            if not holes[p]:
                r = sqrt(
                    periodic_distance(eta1[i], markers[p, 0]) ** 2
                    + periodic_distance(eta2[i], markers[p, 1]) ** 2
                    + periodic_distance(eta3[i], markers[p, 2]) ** 2
                )
                out[i] += markers[p, index] * smoothing_kernel(r, h)


def naive_evaluation_3d(
    eta1: "float[:,:,:]",
    eta2: "float[:,:,:]",
    eta3: "float[:,:,:]",
    markers: "float[:,:]",
    holes: "bool[:]",
    index: "int",
    h: "float",
    out: "float[:,:,:]",
):
    """Naive evaluation of a function defined by its values at the particles.
    This is done in an efficient way, looping over the particles.
    Entries have to be given as 3 3d array (meshgrid format)."""
    n_eval_1 = eta1.shape[0]
    n_eval_2 = eta1.shape[1]
    n_eval_3 = eta1.shape[2]
    n_particles = len(markers)
    out[:] = 0.0
    for i in range(n_eval_1):
        for j in range(n_eval_2):
            for k in range(n_eval_3):
                for p in range(n_particles):
                    if not holes[p]:
                        r = sqrt(
                            periodic_distance(eta1[i, j, k], markers[p, 0]) ** 2
                            + periodic_distance(eta2[i, j, k], markers[p, 1]) ** 2
                            + periodic_distance(eta3[i, j, k], markers[p, 2]) ** 2
                        )
                        out[i, j, k] += markers[p, index] * smoothing_kernel(r, h)


def box_based_evaluation(
    eta1: "float[:]",
    eta2: "float[:]",
    eta3: "float[:]",
    markers: "float[:,:]",
    nx: "int",
    ny: "int",
    nz: "int",
    boxes: "int[:,:]",
    neighbours: "int[:,:]",
    domain_array: "float[:]",
    holes: "bool[:]",
    index: "int",
    h: "float",
    out: "float[:]",
):
    """Optimized evaluation of a function defined by its values at the particles.
    This is done only evaluating the particles in the neighbouring cells of the evaluation point.
    This assumes that the smoothing radius h is smaller then the size of the boxes.
    Entries have to be given as 3 1d array (representing the coordinate in each direction).

    Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        markers : 2d array
            Marker array of the particles.

        nx, ny, nz : int
            Number of boxes in each direction.

        boxes : 2d array
            Box array of the sorting boxes structure.

        neighbours : 2d array
            Array containing the 27 neighbouring boxes of each box.

        domain_array : array
            Contain the information of the domain on the currnt processor.

        holes : array of bool
            Contain the information of the holes on the current processor.

        index : int
            At which index is the value we are computing stored in the particle marker array.

        h : float
            Radius of the smoothing kernel to use.

        out : array
            Array to fill with the result.
        """
    n_eval = len(eta1)
    out[:] = 0.0
    for i in range(n_eval):
        loc_box = sorting_kernels.find_box(eta1[i], eta2[i], eta3[i], nx, ny, nz, domain_array)
        if loc_box == -1:
            continue
        else:
            for neigh in range(27):
                box_to_search = neighbours[loc_box, neigh]
                c = 0
                while boxes[box_to_search, c] != -1:
                    p = boxes[box_to_search, c]
                    c += 1
                    if not holes[p]:
                        r = sqrt(
                            periodic_distance(eta1[i], markers[p, 0]) ** 2
                            + periodic_distance(eta2[i], markers[p, 1]) ** 2
                            + periodic_distance(eta3[i], markers[p, 2]) ** 2
                        )
                        out[i] += markers[p, index] * smoothing_kernel(r, h)

    

def box_based_evaluation_3d(
    eta1: "float[:,:,:]",
    eta2: "float[:,:,:]",
    eta3: "float[:,:,:]",
    markers: "float[:,:]",
    nx: "int",
    ny: "int",
    nz: "int",
    boxes: "int[:,:]",
    neighbours: "int[:,:]",
    domain_array: "float[:]",
    holes: "bool[:]",
    index: "int",
    h: "float",
    out: "float[:,:,:]",
):
    """Optimized evaluation of a function defined by its values at the particles.
    This is done only evaluating the particles in the neighbouring cells of the evaluation point.
    This assumes that the smoothing radius h is smaller then the size of the boxes.
    Entries have to be given as 3 3d array (meshgrid format).

    Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        markers : 2d array
            Marker array of the particles.

        nx, ny, nz : int
            Number of boxes in each direction.

        boxes : 2d array
            Box array of the sorting boxes structure.

        neighbours : 2d array
            Array containing the 27 neighbouring boxes of each box.

        domain_array : array
            Contain the information of the domain on the currnt processor.

        holes : array of bool
            Contain the information of the holes on the current processor.

        index : int
            At which index is the value we are computing stored in the particle marker array.

        h : float
            Radius of the smoothing kernel to use.

        out : array
            Array to fill with the result.
        """
    n_eval_1 = eta1.shape[0]
    n_eval_2 = eta1.shape[1]
    n_eval_3 = eta1.shape[2]
    out[:] = 0.0
    for i in range(n_eval_1):
        for j in range(n_eval_2):
            for k in range(n_eval_3):
                loc_box = sorting_kernels.find_box(
                    eta1[i, j, k], eta2[i, j, k], eta3[i, j, k], nx, ny, nz, domain_array
                )
                if loc_box == -1:
                    continue
                else:
                    for neigh in range(27):
                        box_to_search = neighbours[loc_box, neigh]
                        c = 0
                        while boxes[box_to_search, c] != -1:
                            p = boxes[box_to_search, c]
                            c += 1
                            if not holes[p]:
                                r = sqrt(
                                    periodic_distance(eta1[i, j, k], markers[p, 0]) ** 2
                                    + periodic_distance(eta2[i, j, k], markers[p, 1]) ** 2
                                    + periodic_distance(eta3[i, j, k], markers[p, 2]) ** 2
                                )
                                out[i, j, k] += markers[p, index] * smoothing_kernel(r, h)
