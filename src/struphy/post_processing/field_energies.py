"""Energy-type integrals of post-processed FEEC fields, computed by quadrature on the evaluation grid.

These reproduce the scalars saved during a simulation (e.g. ``en_U``, ``en_B`` of LinearMHD, which are
``1/2 u^T M u`` with a weighted mass matrix ``M``) from the point data in ``fields_data/``. Their main
purpose is to compute the same quantities for data that was never simulated, such as the time-FFT
filtered fields (``sim.spline_values.<species>.<var>.data.filtered()``).

The integrals use the trapezoidal rule on ``grids_log``, so their accuracy is set by the evaluation grid
(increase ``celldivide`` in ``sim.pproc`` to refine it).
"""

import logging

import cunumpy as xp

from struphy.geometry.base import Domain

logger = logging.getLogger("struphy")


def integrate_log(integrand: xp.ndarray, grids_log: list):
    """Trapezoidal rule of ``integrand`` (shape ``(n1, n2, n3)``) over the logical unit cube."""
    out = xp.trapezoid(integrand, grids_log[2], axis=2)
    out = xp.trapezoid(out, grids_log[1], axis=1)
    return xp.trapezoid(out, grids_log[0], axis=0)


def _clean_weight(weight: xp.ndarray, name: str):
    """Set non-finite entries of ``weight`` to zero (e.g. 1/p0 where p0 = 0 on the boundary)."""
    bad = ~xp.isfinite(weight)
    if xp.any(bad):
        logger.warning(f"{name}: {xp.count_nonzero(bad)} grid points with non-finite weight are left out.")
        weight = xp.where(bad, 0.0, weight)
    return weight


def form_energy(
    val_dict: dict,
    form: int,
    domain: Domain,
    grids_log: list,
    weight: xp.ndarray = None,
    normalization: float = 1.0,
):
    r"""Time series of the quadratic energy :math:`\alpha\,\frac12 \int w\, \omega^\top A\, \omega \,\mathrm d\boldsymbol\eta`.

    :math:`A` is the metric factor of the standard mass matrix of the given p-form,

    ======  =========================
    form    :math:`A`
    ======  =========================
    0       :math:`\sqrt g`
    1       :math:`G^{-1} \sqrt g`
    2       :math:`G / \sqrt g`
    3       :math:`1 / \sqrt g`
    ======  =========================

    e.g. LinearMHD's ``en_U`` is ``form_energy(u, 2, domain, grids_log, weight=n0)`` (mass matrix ``M2n``).

    Parameters
    ----------
    val_dict : dict
        ``{t: [comp_0, ...]}`` logical components of the p-form on ``grids_log``
        (e.g. ``sim.spline_values.mhd.velocity_log.data`` or its ``.filtered()``).
    form : int
        0, 1, 2 or 3.
    domain : Domain
        Mapping of the simulation (``sim.domain``).
    grids_log : list
        1D logical evaluation grids (``sim.grids_log``).
    weight : xp.ndarray, optional
        Additional weight :math:`w` on the grid, shape ``(n1, n2, n3)``. Non-finite entries are left out.
    normalization : float
        Prefactor :math:`\alpha`.

    Returns
    -------
    xp.ndarray
        Energy at each time of ``val_dict`` (in its key order).
    """
    assert form in (0, 1, 2, 3), f"{form = } must be 0, 1, 2 or 3."
    sqrt_g = domain.jacobian_det(*grids_log)

    w = xp.ones_like(sqrt_g) if weight is None else _clean_weight(xp.asarray(weight, dtype=float), "form_energy")
    if form == 0:
        A = w * sqrt_g
    elif form == 3:
        A = w / sqrt_g
    elif form == 2:
        A = domain.metric(*grids_log) * (w / sqrt_g)
    else:
        A = domain.metric_inv(*grids_log) * (w * sqrt_g)

    energies = []
    for comps in val_dict.values():
        if form in (0, 3):
            integrand = A * comps[0] ** 2
        else:
            integrand = sum(A[i, j] * comps[i] * comps[j] for i in range(3) for j in range(3))
        energies += [normalization * 0.5 * integrate_log(integrand, grids_log)]
    return xp.array(energies)


def volume_integral(val_dict: dict, grids_log: list, normalization: float = 1.0):
    r"""Time series of :math:`\alpha \int \omega^3 \,\mathrm d\boldsymbol\eta` of a 3-form (e.g. LinearMHD's ``en_p``)."""
    return xp.array([normalization * integrate_log(comps[0], grids_log) for comps in val_dict.values()])


def linear_mhd_energies(sim, filtered: bool = False, pad_bins: int = None, gamma: float = 5 / 3):
    """The energy scalars of :class:`~struphy.models.LinearMHD` computed from post-processed fields.

    Same definitions as the scalars saved during the run (``en_U``, ``en_B``, ``en_thermal``, ``en_p``,
    ``en_tot = en_U + en_B + en_thermal``), but computed from the point data, so that they can also be
    evaluated for the time-FFT filtered fields.

    Parameters
    ----------
    sim : Simulation
        Simulation after ``sim.load_plotting_data()``.
    filtered : bool
        If True, use the time-FFT filtered fields (``.filtered(pad_bins)``).
    pad_bins : int, optional
        Passed to ``.filtered()``; None uses the value from ``sim.pproc``.
    gamma : float
        Adiabatic index (5/3 in LinearMHD).

    Returns
    -------
    dict
        ``t`` and the energies as arrays; energies of fields that were not saved are missing,
        and ``en_tot`` is then the sum of the available quadratic ones.
    """
    domain, grids_log = sim.domain, sim.grids_log
    mhd, em = sim.spline_values.mhd, sim.spline_values.em_fields

    def get(holder, name):
        if not hasattr(holder, name):
            return None
        data = getattr(holder, name).data
        return data.filtered(pad_bins=pad_bins) if filtered else data

    out = {}
    u, b, p = get(mhd, "velocity_log"), get(em, "b_field_log"), get(mhd, "pressure_log")
    ref = next(d for d in (u, b, p) if d is not None)
    out["t"] = xp.array(list(ref.keys()))

    if u is not None:
        out["en_U"] = form_energy(u, 2, domain, grids_log, weight=sim.equil.n0(*grids_log))
    if b is not None:
        out["en_B"] = form_energy(b, 2, domain, grids_log)
    if p is not None:
        # p0 can vanish on the boundary (it is sampled there, unlike in the simulation's quadrature);
        # those points are left out
        p0 = sim.equil.p0(*grids_log)
        with xp.errstate(divide="ignore"):
            inv_p0 = xp.where(p0 > 1e-12 * xp.max(p0), 1.0 / p0, xp.nan)
        out["en_thermal"] = form_energy(p, 3, domain, grids_log, weight=inv_p0, normalization=1.0 / gamma)
        out["en_p"] = volume_integral(p, grids_log, normalization=1.0 / (gamma - 1))

    out["en_tot"] = sum(out[k] for k in ("en_U", "en_B", "en_thermal") if k in out)
    return out
