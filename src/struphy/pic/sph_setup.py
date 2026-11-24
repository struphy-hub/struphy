r"""Setup of the physical SPH smoothing kernel of a :class:`~struphy.pic.particles.ParticlesSPH` species.

The kernel :math:`W_h` is radial and normalized in physical space (:mod:`~struphy.pic.sph_radial_kernels`). The
neighbour search uses the sorting boxes of the logical unit cube: the sum for a marker runs over the 27 boxes around
it. The kernel width :math:`h` is therefore limited by

.. math::

    h_\max = \min_{i\ \textnormal{resolved}}\ \frac{1}{n_i\, \max_{\boldsymbol\eta} \big\| P_\textnormal{res}
    (DF^{-1})_{i,:}(\boldsymbol\eta)^\top \big\|}\,,

since a physical displacement :math:`\Delta\mathbf x` changes :math:`\eta_i` by
:math:`(DF^{-1}\Delta\mathbf x)_i \leq \|(DF^{-1})_{i,:}\|\, |\Delta\mathbf x|`, which must not exceed the box
width :math:`1/n_i`. The maximum is taken over a sampling grid finer than the boxes.

Directions with a single sorting box are not resolved by the kernel (1D and 2D kernels). The mapping must then be
affine and constant in these directions, orthogonal to the resolved ones; distances are projected onto the resolved
directions (:math:`P_\textnormal{res}`) and the kernel is divided by the physical measure :math:`A_\perp` of the
unresolved directions.
"""

from dataclasses import dataclass

import numpy as np

from struphy.geometry.base import Domain

KERNEL_IDS = {"gaussian": 1, "wendland_c2": 2, "cubic_spline": 3}
"""Kernel ids of :mod:`~struphy.pic.sph_radial_kernels`."""

SAFETY = 0.99
"""Safety factor on :math:`h_\\max` for the sampling of :math:`DF^{-1}`."""


@dataclass
class SPHKernelSetup:
    """Kernel data shared by all SPH evaluations of a species (computed by :func:`setup_sph_kernel`).

    Attributes
    ----------
    kernel : str
        Kernel name.

    kernel_id : int
        Kernel id for the Pyccel kernels.

    dim : int
        Number of resolved directions.

    resolved : tuple[bool]
        Resolved logical directions (more than one sorting box).

    h : float
        Kernel width (support radius) in physical units.

    h_max : float
        Largest width with the support inside the 27 neighbour boxes.

    h_min : float
        Lower bound of the width (``min_width_spacings`` times the mean marker spacing).

    spacing : float
        Mean marker spacing :math:`(V/N)^{1/d}`.

    periodic : tuple[bool]
        Periodic logical directions (``bc_sph``).

    period : array
        Period vectors (3 x 3), row ``i`` for logical direction ``i`` (zero if not periodic or not resolved).

    proj : array
        Projection onto the resolved directions (3 x 3).

    scale : float
        Kernel factor :math:`1/A_\\perp` (1 for 3D kernels).

    x_idx : int
        First of the three marker columns holding the physical positions.
    """

    kernel: str
    kernel_id: int
    dim: int
    resolved: tuple
    h: float
    h_max: float
    h_min: float
    spacing: float
    periodic: tuple
    period: np.ndarray
    proj: np.ndarray
    scale: float
    x_idx: int


def _grid(boxes_per_dim, resolved, max_points=200_000):
    """Sampling points (3, M) of the unit cube, finer than the sorting boxes in the resolved directions."""
    d = max(1, sum(resolved))
    cap = max(8, int(max_points ** (1.0 / d)))
    axes = []
    for n, res in zip(boxes_per_dim, resolved):
        # at least two points per box, up to four if the total number of points allows
        m = max(2 * n + 1, min(max(4 * n, 9), cap)) if res else 5
        axes.append(np.linspace(0.0, 1.0, m))
    e1, e2, e3 = np.meshgrid(*axes, indexing="ij")
    return np.vstack([e1.ravel(), e2.ravel(), e3.ravel()])


def _jacobian(domain: Domain, etas):
    """DF at sampling points, shape (M, 3, 3)."""
    jac = np.asarray(domain.jacobian(etas, remove_outside=False))
    return np.moveaxis(jac.reshape(3, 3, -1), -1, 0)


def setup_sph_kernel(
    domain: Domain,
    boxes_per_dim,
    bc_sph,
    n_markers: int,
    kernel: str,
    kernel_width: float,
    min_width_spacings: float,
    x_idx: int,
) -> SPHKernelSetup:
    """Compute the kernel data of an SPH species and check that the setup is consistent.

    Parameters
    ----------
    domain : Domain
        The mapping :math:`F`.

    boxes_per_dim : tuple[int]
        Number of sorting boxes per logical direction (global).

    bc_sph : tuple[str]
        Boundary conditions of the SPH sums per logical direction.

    n_markers : int
        Total number of markers :math:`N`.

    kernel : str
        Kernel name, see :data:`KERNEL_IDS`.

    kernel_width : float | None
        Kernel width set by the user (physical units); None for :math:`h_\\max`.

    min_width_spacings : float
        Lower bound of the width in units of the mean marker spacing.

    x_idx : int
        First of the three marker columns holding the physical positions.

    Returns
    -------
    SPHKernelSetup
        The kernel data.
    """
    boxes_per_dim = tuple(int(n) for n in boxes_per_dim)
    resolved = tuple(n > 1 for n in boxes_per_dim)
    dim = sum(resolved)
    if dim == 0:
        raise ValueError(
            "SPH needs more than one sorting box in at least one direction (SortingParameters.boxes_per_dim)."
        )
    for i in range(3):
        if bc_sph[i] == "periodic" and boxes_per_dim[i] == 2:
            # both neighbour boxes would hold the same markers (counted twice)
            raise ValueError(
                f"SPH with periodic direction {i} needs 1 or at least 3 sorting boxes in this direction, "
                f"got boxes_per_dim = {boxes_per_dim}."
            )

    etas = _grid(boxes_per_dim, resolved)
    jac = _jacobian(domain, etas)  # (M, 3, 3), jac[m, :, k] = dF/deta_k
    length = np.max(np.abs(jac)) + 1e-300

    # --- unresolved directions: constant, orthogonal columns; projection and transverse measure ---
    unres = [k for k in range(3) if not resolved[k]]
    res = [k for k in range(3) if resolved[k]]
    if unres:
        cols = jac[:, :, unres]  # (M, 3, n_unres)
        if np.max(np.abs(cols - cols[0])) > 1e-8 * length:
            raise ValueError(
                f"SPH with {dim}D kernels (sorting boxes {boxes_per_dim}) needs a mapping that is affine in the "
                f"directions {unres} without sorting boxes; DF varies along them."
            )
        dots = np.einsum("mik,mil->mkl", jac[:, :, res], cols)
        if np.max(np.abs(dots)) > 1e-8 * length**2:
            raise ValueError(
                f"SPH with {dim}D kernels needs the unresolved directions {unres} to be orthogonal to the "
                f"resolved ones {res} in physical space."
            )
        c = cols[0]  # (3, n_unres)
        q, _ = np.linalg.qr(c)
        proj = np.eye(3) - q @ q.T
        area_perp = float(np.sqrt(np.linalg.det(c.T @ c)))
    else:
        proj = np.eye(3)
        area_perp = 1.0

    # --- period vectors of the periodic, resolved directions ---
    periodic = tuple(bc == "periodic" for bc in bc_sph)
    period = np.zeros((3, 3))
    for i in range(3):
        if not (periodic[i] and resolved[i]):
            continue
        lo = etas.copy()
        hi = etas.copy()
        lo[i] = 0.0
        hi[i] = 1.0
        diff = np.asarray(domain(hi, remove_outside=False)).reshape(3, -1) - np.asarray(
            domain(lo, remove_outside=False)
        ).reshape(3, -1)
        if np.max(np.abs(diff - diff[:, :1])) > 1e-8 * length:
            raise ValueError(
                f"SPH with periodic direction {i} needs a constant period vector F(eta + e_{i}) - F(eta) "
                "(e.g. a translation or the identity for an angle); the mapping is not periodic in this way."
            )
        period[i] = diff[:, 0]

    # --- largest width with the support in the 27 neighbour boxes ---
    # singular points (e.g. the axis of a polar mapping) have an unbounded DF^{-1}: infinite row norms
    det = np.linalg.det(jac)
    singular = np.abs(det) <= 1e-12 * length**3
    jac_reg = np.where(singular[:, None, None], np.eye(3), jac)
    jac_inv = np.linalg.inv(jac_reg)  # (M, 3, 3), rows = grad eta_i
    h_max = np.inf
    limit = None
    for i in res:
        rows = jac_inv[:, i, :] @ proj  # gradient of eta_i projected on the resolved directions
        norms = np.linalg.norm(rows, axis=1)
        norms = np.where(singular | ~np.isfinite(norms), np.inf, norms)
        worst = int(np.argmax(norms))
        h_i = 1.0 / (boxes_per_dim[i] * norms[worst])
        if h_i < h_max:
            h_max = h_i
            limit = (i, etas[:, worst])
    h_max *= SAFETY

    # --- lower bound from the mean marker spacing ---
    det = np.abs(det)
    volume = float(np.mean(det[np.isfinite(det)]))  # volume of the unit cube under F (sampling average)
    spacing = (volume / area_perp / n_markers) ** (1.0 / dim)
    h_min = min_width_spacings * spacing

    hint = (
        f"The largest kernel width with the neighbour search in 27 sorting boxes is h_max = {h_max:.4g}, limited by "
        f"logical direction {limit[0]} (eta = {np.round(limit[1], 4)}); the lower bound is "
        f"h_min = {min_width_spacings} x mean marker spacing {spacing:.4g} = {h_min:.4g}. "
        f"Use fewer sorting boxes in direction {limit[0]} (boxes_per_dim = {boxes_per_dim}), more markers, a smaller "
        "SPHParameters.min_width_spacings or a less distorted mapping. For polar mappings, the kernel cannot reach "
        "the axis: use an annulus (inner radius a1 > 0) such that the angular box width at a1 is at least h_min."
    )
    if not h_max >= h_min:
        raise ValueError("SPH kernel width too small. " + hint)

    if kernel_width is None:
        h = h_max
    else:
        if kernel_width > h_max:
            raise ValueError(f"SPHParameters.kernel_width = {kernel_width} is larger than h_max. " + hint)
        if kernel_width < h_min:
            raise ValueError(f"SPHParameters.kernel_width = {kernel_width} is smaller than h_min. " + hint)
        h = float(kernel_width)

    return SPHKernelSetup(
        kernel=kernel,
        kernel_id=KERNEL_IDS[kernel],
        dim=dim,
        resolved=resolved,
        h=float(h),
        h_max=float(h_max),
        h_min=float(h_min),
        spacing=float(spacing),
        periodic=periodic,
        period=period,
        proj=proj,
        scale=1.0 / area_perp,
        x_idx=int(x_idx),
    )
