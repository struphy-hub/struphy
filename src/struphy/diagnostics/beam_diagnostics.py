"""Beam diagnostics for ion-optics runs, evaluated from saved marker histories."""

import numpy as np


def plane_crossings(states, plane, axis=0):
    """Interpolate each marker's state where it first crosses a plane.

    Parameters
    ----------
    states : ndarray
        Marker histories of shape ``(n_times, n_markers, n_components)``. Entries
        of a marker that has left the domain must be NaN.

    plane : float
        Position of the plane along ``axis``.

    axis : int
        Component of ``states`` holding the coordinate normal to the plane.

    Returns
    -------
    ndarray
        Shape ``(n_markers, n_components)``: the state linearly interpolated
        to the first crossing, or NaN for markers that never cross.
    """
    states = np.asarray(states, dtype=float)
    offset = states[:, :, axis] - plane
    crossing = np.full(states.shape[1:], np.nan)
    for j in range(states.shape[1]):
        before, after = offset[:-1, j], offset[1:, j]
        steps = np.nonzero((before <= 0.0) & (after > 0.0) | (before >= 0.0) & (after < 0.0))[0]
        if len(steps):
            k = steps[0]
            fraction = before[k] / (before[k] - after[k])
            crossing[j] = states[k, j] + fraction * (states[k + 1, j] - states[k, j])
    return crossing


def rms_moments(position, angle, weights=None):
    """RMS beam size, divergence and emittance in one transverse plane.

    Parameters
    ----------
    position, angle : ndarray
        Transverse position ``y`` and angle ``y' = v_y / v_x`` of the markers.
        NaN entries (lost markers) are ignored.

    weights : ndarray, optional
        Marker weights; uniform if omitted.

    Returns
    -------
    dict
        ``size`` (rms y), ``divergence`` (rms y'), ``emittance``
        (``sqrt(<y²><y'²> - <yy'>²)``, centred moments) and ``count``.
    """
    position, angle = np.asarray(position, dtype=float), np.asarray(angle, dtype=float)
    weights = np.ones_like(position) if weights is None else np.asarray(weights, dtype=float)
    valid = np.isfinite(position) & np.isfinite(angle)
    if not np.any(valid):
        return {"size": np.nan, "divergence": np.nan, "emittance": np.nan, "count": 0}
    y, yp, w = position[valid], angle[valid], weights[valid]
    y = y - np.average(y, weights=w)
    yp = yp - np.average(yp, weights=w)
    yy, ypyp, yyp = (np.average(q, weights=w) for q in (y * y, yp * yp, y * yp))
    return {
        "size": np.sqrt(yy),
        "divergence": np.sqrt(ypyp),
        "emittance": np.sqrt(max(yy * ypyp - yyp**2, 0.0)),
        "count": int(valid.sum()),
    }


def emittance_converged(history, n=5, tol=1e-3):
    """Stopping rule of the IBSimu Vlasov–Poisson iteration (Kalvas 2013, §5.8.2).

    Converged when the mean of the last ``n`` changes of the (exit) emittance,
    relative to the latest value, is below ``tol`` for two consecutive rounds.

    Parameters
    ----------
    history : sequence of float
        Emittance after each iteration round (or time window).

    Returns
    -------
    bool
    """
    eps = np.asarray(history, dtype=float)
    if len(eps) < n + 2:
        return False

    def mean_change(end):
        window = eps[end - n - 1 : end]
        return abs(np.mean(np.diff(window))) / abs(window[-1])

    return mean_change(len(eps)) < tol and mean_change(len(eps) - 1) < tol


def windowed_outlet_emittance(records, window):
    """RMS emittance of outlet records in consecutive time windows.

    Parameters
    ----------
    records : ndarray
        Ledger records ``(x, y, z, vx, vy, vz, weight, time)``; the transverse plane is (y, vy/vx).

    window : float
        Window length in time.

    Returns
    -------
    tuple[ndarray, ndarray]
        Window end times and the weighted rms emittance in each window (NaN if empty).
    """
    records = np.asarray(records, dtype=float)
    if len(records) == 0:
        return np.empty(0), np.empty(0)
    edges = np.arange(0.0, records[:, 7].max() + window, window)
    ends, values = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        rows = records[(records[:, 7] > low) & (records[:, 7] <= high)]
        ends.append(high)
        values.append(
            rms_moments(rows[:, 1], rows[:, 4] / rows[:, 3], rows[:, 6])["emittance"] if len(rows) > 2 else np.nan
        )
    return np.array(ends), np.array(values)


def steady_state_start(records, window, n=5, tol=0.05):
    """Start time of the steady state of a time-dependent run, from its outlet records.

    Applies :func:`emittance_converged` to the outlet emittance in consecutive time
    windows (:func:`windowed_outlet_emittance`) and returns the end of the first
    window at which the rule is satisfied, or ``None``. Per-window emittances carry
    marker noise, so ``tol`` must lie above their relative scatter.
    """
    ends, emittance = windowed_outlet_emittance(records, window)
    for k in range(len(emittance)):
        if np.isfinite(emittance[k]) and emittance_converged(
            emittance[: k + 1][np.isfinite(emittance[: k + 1])], n, tol
        ):
            return float(ends[k])
    return None
