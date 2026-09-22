"""Continuous beam sources and current accounting for ion-optics PIC runs.

Charge bookkeeping uses the marker weight as the marker charge in normalized
units: a source emitting ``rate`` markers per unit time with current ``current``
gives each marker the weight ``current / rate``.
"""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import numpy as np

from struphy.pic.base import Particles


class BeamSource(metaclass=ABCMeta):
    """Emits markers at a constant rate.

    Parameters
    ----------
    rate : float
        Number of markers emitted per unit (normalized) time.

    current : float
        Emitted charge per unit time in normalized units; sets the marker weight
        ``current / rate``. Irrelevant for zero-current runs except for accounting.

    seed : int
        Seed of the source's random number generator.
    """

    def __init__(self, rate: float, current: float = 1.0, seed: int = 1234):
        if rate <= 0.0:
            raise ValueError("The emission rate must be positive.")
        self.rate = rate
        self.current = current
        self.rng = np.random.default_rng(seed)

    @property
    def weight(self) -> float:
        """Charge carried by one marker."""
        return self.current / self.rate

    @abstractmethod
    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Return logical positions ``(n, 3)`` on the emitting surface and physical velocities ``(n, 3)``."""


class PlaneSource(BeamSource):
    """Markers emitted from a logical plane ``eta[axis] = eta_plane``.

    Transverse logical coordinates are uniform in ``eta_ranges`` (one interval for
    each of the other two axes, in increasing axis order). Velocities are the
    physical ``velocity`` plus independent Gaussian spreads ``velocity_spread``.

    Parameters
    ----------
    axis : int
        Logical axis normal to the emitting plane.

    eta_plane : float
        Logical coordinate of the plane.

    eta_ranges : tuple[tuple[float, float], tuple[float, float]]
        Logical intervals of the two transverse coordinates.

    velocity : tuple[float, float, float]
        Mean physical (Cartesian) velocity.

    velocity_spread : tuple[float, float, float]
        Standard deviation of each physical velocity component.
    """

    def __init__(
        self,
        rate: float,
        current: float = 1.0,
        axis: int = 0,
        eta_plane: float = 0.0,
        eta_ranges: tuple = ((0.0, 1.0), (0.0, 1.0)),
        velocity: tuple = (1.0, 0.0, 0.0),
        velocity_spread: tuple = (0.0, 0.0, 0.0),
        seed: int = 1234,
    ):
        super().__init__(rate=rate, current=current, seed=seed)
        if axis not in (0, 1, 2) or not 0.0 <= eta_plane <= 1.0:
            raise ValueError("axis must be 0, 1 or 2 and eta_plane must lie in [0, 1].")
        self.axis = axis
        self.eta_plane = eta_plane
        self.eta_ranges = tuple(tuple(r) for r in eta_ranges)
        self.velocity = np.asarray(velocity, dtype=float)
        self.velocity_spread = np.asarray(velocity_spread, dtype=float)

    def sample(self, n):
        eta = np.empty((n, 3))
        eta[:, self.axis] = self.eta_plane
        for (low, high), other in zip(self.eta_ranges, [a for a in range(3) if a != self.axis]):
            eta[:, other] = self.rng.uniform(low, high, n)
        v = self.velocity + self.velocity_spread * self.rng.standard_normal((n, 3))
        return eta, v


@dataclass(frozen=True)
class LossTag:
    """Names a part of the domain boundary for loss accounting.

    A removed marker matches if it left through logical ``axis`` on ``side``
    (0 = eta = 0, 1 = eta = 1, None = either) and, if ``interval`` is given, its
    physical ``coordinate`` at the loss point lies in ``interval``.
    """

    name: str
    axis: int
    side: int | None = None
    coordinate: int = 0
    interval: tuple[float, float] | None = None


class CurrentLedger:
    """Cumulative injected, lost (per tag) and live charge of one particle species.

    Parameters
    ----------
    tags : tuple[LossTag]
        Boundary parts, tested in order; unmatched losses are booked as ``"other"``.

    keep_records : tuple[str]
        Tag names whose removed markers are kept in :attr:`records` as rows
        ``(x, y, z, vx, vy, vz, weight, time)`` (physical loss point; ``time`` of the
        booking update), e.g. to get the phase space of the beam leaving through an outlet.
    """

    def __init__(self, tags: tuple = (), keep_records: tuple = ()):
        self.tags = tuple(tags)
        names = [tag.name for tag in self.tags]
        if len(set(names)) != len(names) or "other" in names:
            raise ValueError("Loss tag names must be unique and must not be 'other'.")
        self.names = (*names, "other")
        self.injected_charge = 0.0
        self.injected_markers = 0
        self.lost_charge = dict.fromkeys(self.names, 0.0)
        self.lost_markers = dict.fromkeys(self.names, 0)
        if not set(keep_records) <= set(self.names):
            raise ValueError(f"keep_records must be among the tag names {self.names}.")
        self._kept = {name: [] for name in keep_records}

    @property
    def records(self) -> dict[str, np.ndarray]:
        """Kept loss records per tag, see ``keep_records``."""
        return {name: np.concatenate(rows) if rows else np.empty((0, 8)) for name, rows in self._kept.items()}

    def book_injection(self, n_markers: int, charge: float):
        self.injected_markers += int(n_markers)
        self.injected_charge += float(charge)

    def update(self, particles: Particles, domain, time: float = 0.0):
        """Classify and book all markers removed since the last update (at ``time``)."""
        self.book(particles.pop_lost_markers(), particles.lost_index, domain, time)

    def book(self, records, index, domain, time: float = 0.0):
        """Classify and book removal records (rows of ``Particles.lost_markers``, columns ``index``)."""
        if len(records) == 0:
            return
        eta = np.clip(records[:, index["pos"]], 0.0, 1.0)
        x = np.asarray(domain(eta, change_out_order=True, remove_outside=False)).reshape(len(records), 3)
        axis = records[:, index["axis"]].astype(int)
        side = records[:, index["side"]].astype(int)
        weights = records[:, index["weights"]]
        state = np.column_stack([x, records[:, index["vel"]], weights, np.full(len(records), time)])
        unmatched = np.ones(len(records), dtype=bool)
        for tag in self.tags:
            match = unmatched & (axis == tag.axis)
            if tag.side is not None:
                match &= side == tag.side
            if tag.interval is not None:
                low, high = tag.interval
                match &= (x[:, tag.coordinate] >= low) & (x[:, tag.coordinate] <= high)
            self._book(tag.name, match, state)
            unmatched &= ~match
        self._book("other", unmatched, state)

    def _book(self, name, match, state):
        self.lost_markers[name] += int(np.count_nonzero(match))
        self.lost_charge[name] += float(np.sum(state[match, 6]))
        if name in self._kept and np.any(match):
            self._kept[name].append(state[match])
