"Only particle variables are updated."

import logging
from dataclasses import dataclass

import numpy as np
from feectools.ddm.mpi import mpi as MPI

from struphy.io.options import OptionsBase
from struphy.models.variables import PICVariable
from struphy.pic.ion_beams import BeamSource, CurrentLedger
from struphy.propagators.base import Propagator

logger = logging.getLogger("struphy")


class InjectMarkers(Propagator):
    r"""Continuous marker injection from a :class:`~struphy.pic.ion_beams.BeamSource`.

    A call with time step :math:`\Delta t` emits :math:`\lfloor r\,\Delta t + c \rfloor`
    markers (emission rate :math:`r`; the fractional remainder :math:`c` is carried over),
    each with weight ``source.current / source.rate``. Each marker gets a random emission
    time :math:`\tau \in [0, \Delta t)` and starts at

    .. math::

        \boldsymbol\eta = \boldsymbol\eta_\mathrm{s} + DF^{-1}(\boldsymbol\eta_\mathrm{s})\,\mathbf v\,\tau\,,

    so that a steady beam is not bunched at the time-step frequency. Markers are
    written into holes of the marker array; its size is set by ``bufsize`` in
    ``set_markers``. The markers loaded at start are removed unless ``keep_initial``.
    """

    class Variables:
        def __init__(self):
            self._var: PICVariable = None

        @property
        def var(self) -> PICVariable:
            return self._var

        @var.setter
        def var(self, new):
            assert isinstance(new, PICVariable)
            assert new.space in ("Particles6D", "DeltaFParticles6D")
            self._var = new

    def __init__(self, source: BeamSource, ledger: CurrentLedger = None, keep_initial: bool = False):
        assert isinstance(source, BeamSource)
        self.variables = self.Variables()
        self.source = source
        self.ledger = CurrentLedger() if ledger is None else ledger
        self.keep_initial = keep_initial

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`InjectMarkers` (none yet)."""

        def __post_init__(self):
            pass

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        self._options = new

    def allocate(self):
        if MPI.COMM_WORLD.Get_size() != 1:
            raise NotImplementedError("InjectMarkers currently supports serial runs only.")
        particles = self.variables.var.particles
        if not self.keep_initial:
            particles.markers[particles.valid_mks, :-1] = -1.0
            particles.update_holes()
        ids = particles.markers[:, -1]
        self._next_id = int(max(np.max(ids), particles.Np - 1)) + 1
        self._carry = 0.0

    def __call__(self, dt):
        expected = self.source.rate * dt + self._carry
        n = int(np.floor(expected))
        self._carry = expected - n
        if n == 0:
            return

        particles = self.variables.var.particles
        holes = np.nonzero(particles.holes)[0]
        if len(holes) < n:
            raise RuntimeError(
                f"No room for {n} injected markers ({len(holes)} holes); increase bufsize in set_markers."
            )
        rows = holes[:n]

        eta, v = self.source.sample(n)
        tau = self.source.rng.uniform(0.0, dt, n)
        dfinv = np.asarray(self.domain.jacobian_inv(eta, change_out_order=True, remove_outside=False)).reshape(n, 3, 3)
        eta = eta + np.einsum("nij,nj->ni", dfinv, v) * tau[:, None]

        index = particles.index
        weight = self.source.weight
        particles.markers[rows, :] = 0.0
        particles.markers[rows, index["pos"]] = eta
        particles.markers[rows, index["vel"]] = v
        particles.markers[rows, index["weights"]] = weight
        particles.markers[rows, index["s0"]] = 1.0
        particles.markers[rows, index["w0"]] = weight
        particles.markers[rows, -1] = self._next_id + np.arange(n)
        self._next_id += n
        particles.update_holes()
        self.ledger.book_injection(n, n * weight)
