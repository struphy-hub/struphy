r"""SPH evaluation in physical space.

The smoothing kernel :math:`W_h` is radial and normalized in physical space (:mod:`~struphy.pic.sph_radial_kernels`)
and pulled back to the logical unit cube as a 0-form,

.. math::

    \hat W_h(\boldsymbol\eta, \boldsymbol\eta') = W_h\big(|F(\boldsymbol\eta) - F(\boldsymbol\eta')|\big)\,,
    \qquad
    \hat\rho_h(\boldsymbol\eta) = \sum_j c_j\, \hat W_h(\boldsymbol\eta, \boldsymbol\eta_j)\,.

The physical marker positions :math:`\mathbf x_j = F(\boldsymbol\eta_j)` are stored in the marker array (columns
``x_idx, x_idx + 1, x_idx + 2``, filled by :func:`physical_positions`). The physical distance vector between an
evaluation point and marker :math:`j` is

.. math::

    \mathbf r = P_\textnormal{res} \Big( \mathbf x - \mathbf x_j - \sum_{i\,\textnormal{periodic}} k_i\, \mathbf P_i \Big)\,,
    \qquad k_i = \textnormal{round}(\eta_i - \eta_{j,i})\,,

with the period vectors :math:`\mathbf P_i = F(\boldsymbol\eta + \mathbf e_i) - F(\boldsymbol\eta)` of the periodic
directions (constant, checked at setup) and the projection :math:`P_\textnormal{res}` onto the directions resolved
by the kernel (the identity for 3D kernels).
"""

from numpy import shape, sqrt, zeros
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.pic.sph_radial_kernels as sph_radial_kernels
from struphy.kernel_arguments.pusher_args_kernels import DomainArguments, MarkerArguments


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


def fold(eta: float) -> float:
    """Map a logical coordinate of a mirrored ghost particle back into the unit interval.

    Parameters
    ----------
    eta : float
        Logical coordinate, in (-1, 2).

    Returns
    -------
    float
        ``-eta`` if ``eta < 0``, ``2 - eta`` if ``eta > 1``, else ``eta``.
    """
    if eta < 0.0:
        return -eta
    elif eta > 1.0:
        return 2.0 - eta
    return eta


def clamp(eta: float) -> float:
    """Clamp a logical coordinate to the unit interval.

    Parameters
    ----------
    eta : float
        Logical coordinate.

    Returns
    -------
    float
        ``eta`` restricted to [0, 1].
    """
    if eta < 0.0:
        return 0.0
    elif eta > 1.0:
        return 1.0
    return eta


@stack_array("eta", "eta_f", "eta_w", "x_f", "dfm")
def physical_positions(
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    holes: "bool[:]",
    x_idx: "int",
):
    r"""Store the physical positions :math:`\mathbf x_p = F(\boldsymbol\eta_p)` of all markers (including ghost
    particles) in ``markers[x_idx:x_idx + 3, :]``.

    Mirrored ghost particles of wall boundary conditions sit outside the unit cube (at :math:`-\eta` or
    :math:`2 - \eta`). Their physical position is the first-order extrapolation from the foot point
    :math:`\boldsymbol\eta_w` on the wall (the clamped position),

    .. math::

        \mathbf x_g = F(\boldsymbol\eta_f) + DF(\boldsymbol\eta_w)\, (\boldsymbol\eta_g - \boldsymbol\eta_f)\,,

    where :math:`\boldsymbol\eta_f` is the folded position (the marker that was mirrored). This is the exact mirror
    image for planar walls (e.g. :class:`~struphy.geometry.domains.Cuboid`).

    Parameters
    ----------
    args_markers : MarkerArguments
        Marker arguments.

    args_domain : DomainArguments
        Arguments of the mapping.

    holes : bool[:]
        ``True`` for holes in the marker array.

    x_idx : int
        First of the three marker columns for the physical position.
    """
    markers = args_markers.markers
    n_cols_markers = shape(markers)[1]

    eta = zeros(3, dtype=float)
    eta_f = zeros(3, dtype=float)
    eta_w = zeros(3, dtype=float)
    x_f = zeros(3, dtype=float)
    dfm = zeros((3, 3), dtype=float)

    for p in range(n_cols_markers):
        if holes[p]:
            continue
        mirrored = False
        for i in range(3):
            eta[i] = markers[i, p]
            eta_f[i] = fold(eta[i])
            eta_w[i] = clamp(eta[i])
            if eta[i] < 0.0 or eta[i] > 1.0:
                mirrored = True

        evaluation_kernels.f(eta_f[0], eta_f[1], eta_f[2], args_domain, x_f)
        if mirrored:
            evaluation_kernels.df(eta_w[0], eta_w[1], eta_w[2], args_domain, dfm)
            for i in range(3):
                markers[x_idx + i, p] = x_f[i]
                for k in range(3):
                    markers[x_idx + i, p] += dfm[i, k] * (eta[k] - eta_f[k])
        else:
            for i in range(3):
                markers[x_idx + i, p] = x_f[i]


@stack_array("dx")
def distance_vector(
    markers: "float[:,:]",
    p: "int",
    x1: "float",
    x2: "float",
    x3: "float",
    eta1: "float",
    eta2: "float",
    eta3: "float",
    periodic1: "bool",
    periodic2: "bool",
    periodic3: "bool",
    period: "float[:,:]",
    proj: "float[:,:]",
    x_idx: "int",
    r_out: "float[:]",
) -> float:
    r"""Physical distance vector :math:`\mathbf r` from marker ``p`` to the point :math:`(\mathbf x, \boldsymbol\eta)`,
    with periodic images and projected onto the resolved directions; returns :math:`|\mathbf r|`.

    Parameters
    ----------
    markers : array[float]
        Marker array (physical positions at ``x_idx``).

    p : int
        Marker index.

    x1, x2, x3 : float
        Physical position of the evaluation point.

    eta1, eta2, eta3 : float
        Logical position of the evaluation point (to count periodic images).

    periodic1, periodic2, periodic3 : bool
        Periodicity of the logical directions.

    period : array[float]
        Period vectors, ``period[i, :]`` for logical direction ``i`` (zero if not periodic).

    proj : array[float]
        Projection onto the directions resolved by the kernel (3 x 3).

    x_idx : int
        First marker column of the physical position.

    r_out : array[float]
        Output: the distance vector (length 3).

    Returns
    -------
    float
        :math:`|\mathbf r|`.
    """
    dx = zeros(3, dtype=float)
    dx[0] = x1 - markers[x_idx, p]
    dx[1] = x2 - markers[x_idx + 1, p]
    dx[2] = x3 - markers[x_idx + 2, p]

    if periodic1:
        k = int(round(eta1 - markers[0, p]))
        for c in range(3):
            dx[c] -= k * period[0, c]
    if periodic2:
        k = int(round(eta2 - markers[1, p]))
        for c in range(3):
            dx[c] -= k * period[1, c]
    if periodic3:
        k = int(round(eta3 - markers[2, p]))
        for c in range(3):
            dx[c] -= k * period[2, c]

    r2 = 0.0
    for c in range(3):
        r_out[c] = proj[c, 0] * dx[0] + proj[c, 1] * dx[1] + proj[c, 2] * dx[2]
        r2 += r_out[c] * r_out[c]
    return sqrt(r2)


@stack_array("r_vec")
def naive_physical(
    args_markers: "MarkerArguments",
    x1: "float",
    x2: "float",
    x3: "float",
    eta1: "float",
    eta2: "float",
    eta3: "float",
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
    out: "float[:]",
):
    r"""SPH sum over all markers (no neighbour search, ghost particles skipped), for testing:
    ``out[0]`` :math:`= \sum_j c_j W_h(|\mathbf r_j|)`, ``out[1:4]`` :math:`= \sum_j c_j \nabla W_h(\mathbf r_j)`,
    with :math:`c_j` = ``markers[index, j]``. The cost is :math:`O(N)` per point.

    Parameters
    ----------
    args_markers : MarkerArguments
        Marker arguments.

    x1, x2, x3 : float
        Physical evaluation point.

    eta1, eta2, eta3 : float
        Logical evaluation point.

    holes : bool[:]
        ``True`` for holes.

    periodic1, periodic2, periodic3 : bool
        Periodicity of the logical directions.

    period : array[float]
        Period vectors (3 x 3).

    proj : array[float]
        Projection onto the resolved directions (3 x 3).

    x_idx : int
        First marker column of the physical position.

    index : int
        Marker column of the coefficients :math:`c_j`.

    kernel : int
        Kernel id, see :mod:`~struphy.pic.sph_radial_kernels`.

    dim : int
        Number of resolved directions.

    h : float
        Kernel width in physical units.

    scale : float
        Factor multiplying the kernel: :math:`1/A_\perp` with the physical measure :math:`A_\perp` of the
        directions not resolved by the kernel (1 for 3D kernels).

    out : array[float]
        Output (length 4): value and gradient.
    """
    markers = args_markers.markers
    n_cols_markers = shape(markers)[1]
    r_vec = zeros(3, dtype=float)

    for c in range(4):
        out[c] = 0.0
    for p in range(n_cols_markers):
        if holes[p] or markers[-1, p] == -2.0:
            continue
        r = distance_vector(
            markers, p, x1, x2, x3, eta1, eta2, eta3, periodic1, periodic2, periodic3, period, proj, x_idx, r_vec
        )
        if r >= h:
            continue
        coeff = markers[index, p] * scale
        out[0] += coeff * sph_radial_kernels.kernel_value(kernel, dim, r, h)
        if r > 0.0:
            dw = coeff * sph_radial_kernels.kernel_dr(kernel, dim, r, h) / r
            for c in range(3):
                out[1 + c] += dw * r_vec[c]


@stack_array("r_vec")
def box_based_physical(
    args_markers: "MarkerArguments",
    x1: "float",
    x2: "float",
    x3: "float",
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
    period: "float[:,:]",
    proj: "float[:,:]",
    x_idx: "int",
    index: "int",
    kernel: "int",
    dim: "int",
    h: "float",
    scale: "float",
    out: "float[:]",
):
    r"""SPH sum over the markers in the 27 neighbour boxes of ``loc_box`` (including ghost particles):
    ``out[0]`` :math:`= \sum_j c_j W_h(|\mathbf r_j|)` and ``out[1:4]`` :math:`= \sum_j c_j \nabla W_h(\mathbf r_j)`
    (the physical gradient with respect to the evaluation point), with :math:`c_j` = ``markers[index, j]``.
    The kernel width :math:`h` is chosen at setup such that the support lies in the 27 boxes.

    Parameters
    ----------
    args_markers : MarkerArguments
        Marker arguments.

    x1, x2, x3 : float
        Physical evaluation point.

    eta1, eta2, eta3 : float
        Logical evaluation point.

    loc_box : int
        Box of the evaluation point.

    boxes : array[int]
        Box array of the sorting boxes.

    neighbours : array[int]
        The 27 neighbour boxes of each box.

    holes : bool[:]
        ``True`` for holes.

    periodic1, periodic2, periodic3 : bool
        Periodicity of the logical directions.

    period : array[float]
        Period vectors (3 x 3).

    proj : array[float]
        Projection onto the resolved directions (3 x 3).

    x_idx : int
        First marker column of the physical position.

    index : int
        Marker column of the coefficients :math:`c_j`.

    kernel : int
        Kernel id, see :mod:`~struphy.pic.sph_radial_kernels`.

    dim : int
        Number of resolved directions.

    h : float
        Kernel width in physical units.

    scale : float
        Factor multiplying the kernel: :math:`1/A_\perp` with the physical measure :math:`A_\perp` of the
        directions not resolved by the kernel (1 for 3D kernels).

    out : array[float]
        Output (length 4): value and gradient.
    """
    markers = args_markers.markers
    r_vec = zeros(3, dtype=float)

    for c in range(4):
        out[c] = 0.0
    for neigh in range(27):
        box_to_search = neighbours[loc_box, neigh]
        n = 0
        while boxes[box_to_search, n] != -1:
            p = boxes[box_to_search, n]
            n += 1
            if holes[p]:
                continue
            r = distance_vector(
                markers, p, x1, x2, x3, eta1, eta2, eta3, periodic1, periodic2, periodic3, period, proj, x_idx, r_vec
            )
            if r >= h:
                continue
            coeff = markers[index, p] * scale
            out[0] += coeff * sph_radial_kernels.kernel_value(kernel, dim, r, h)
            if r > 0.0:
                dw = coeff * sph_radial_kernels.kernel_dr(kernel, dim, r, h) / r
                for c in range(3):
                    out[1 + c] += dw * r_vec[c]


@stack_array("r_vec")
def box_based_columns(
    args_markers: "MarkerArguments",
    x1: "float",
    x2: "float",
    x3: "float",
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
    period: "float[:,:]",
    proj: "float[:,:]",
    x_idx: "int",
    first_idx: "int",
    n_idx: "int",
    kernel: "int",
    dim: "int",
    h: "float",
    scale: "float",
    values: "float[:]",
    grads: "float[:,:]",
):
    r"""As :func:`box_based_physical`, for the ``n_idx`` consecutive coefficient columns
    ``first_idx, ..., first_idx + n_idx - 1`` in one sweep over the neighbour boxes:
    ``values[a]`` :math:`= \sum_j c_{a,j} W_h(|\mathbf r_j|)` and ``grads[a, :]``
    :math:`= \sum_j c_{a,j} \nabla W_h(\mathbf r_j)` with :math:`c_{a,j}` = ``markers[first_idx + a, j]``.

    Parameters
    ----------
    args_markers : MarkerArguments
        Marker arguments.

    x1, x2, x3 : float
        Physical evaluation point.

    eta1, eta2, eta3 : float
        Logical evaluation point.

    loc_box : int
        Box of the evaluation point.

    boxes, neighbours : array[int]
        Sorting-box arrays.

    holes : bool[:]
        ``True`` for holes.

    periodic1, periodic2, periodic3 : bool
        Periodicity of the logical directions.

    period, proj : array[float]
        Period vectors and projection onto the resolved directions (3 x 3).

    x_idx : int
        First marker column of the physical position.

    first_idx, n_idx : int
        First coefficient column and number of columns.

    kernel, dim : int
        Kernel id and number of resolved directions.

    h, scale : float
        Kernel width (physical) and kernel factor :math:`1/A_\perp`.

    values : array[float]
        Output, length at least ``n_idx``.

    grads : array[float]
        Output, shape at least ``(n_idx, 3)``.
    """
    markers = args_markers.markers
    r_vec = zeros(3, dtype=float)

    for a in range(n_idx):
        values[a] = 0.0
        for c in range(3):
            grads[a, c] = 0.0
    for neigh in range(27):
        box_to_search = neighbours[loc_box, neigh]
        n = 0
        while boxes[box_to_search, n] != -1:
            p = boxes[box_to_search, n]
            n += 1
            if holes[p]:
                continue
            r = distance_vector(
                markers, p, x1, x2, x3, eta1, eta2, eta3, periodic1, periodic2, periodic3, period, proj, x_idx, r_vec
            )
            if r >= h:
                continue
            w = scale * sph_radial_kernels.kernel_value(kernel, dim, r, h)
            dw = 0.0
            if r > 0.0:
                dw = scale * sph_radial_kernels.kernel_dr(kernel, dim, r, h) / r
            for a in range(n_idx):
                coeff = markers[first_idx + a, p]
                values[a] += coeff * w
                for c in range(3):
                    grads[a, c] += coeff * dw * r_vec[c]
