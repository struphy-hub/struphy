"Only particle variables are updated."

import logging
from dataclasses import dataclass

from line_profiler import profile

from struphy.io.options import OptionsBase
from struphy.models.variables import PICVariable, SPHVariable
from struphy.ode.utils import ButcherTableau
from struphy.pic.pushing import pusher_kernels
from struphy.pic.pushing.pusher import Pusher
from struphy.propagators.base import Propagator
from cunumpy import PyccelKernel

logger = logging.getLogger("struphy")


class PushEta(Propagator):
    r"""For each marker :math:`p`, solves

    .. math::

        \frac{\textnormal d \mathbf{x}_p(t)}{\textnormal d t} = \mathbf{v}_p\,,

    in logical space given by :math:`\mathbf{x} = F(\boldsymbol{\eta})`:

    .. math::

        \frac{\textnormal d \boldsymbol{\eta}_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol{\eta}_p(t)) \,\mathbf{v}_p\,,

    for constant :math:`\mathbf{v}_p`. Available algorithms:

    - Explicit RK from :class:`~struphy.ode.utils.ButcherTableau`
    """

    class Variables:
        """Container for variables advanced by :class:`PushEta`.

        Attributes
        ----------
        var : PICVariable or SPHVariable
            Particle variable whose marker positions are advanced.
        """

        def __init__(self):
            self._var: PICVariable | SPHVariable = None

        @property
        def var(self) -> PICVariable | SPHVariable:
            return self._var

        @var.setter
        def var(self, new):
            assert isinstance(new, PICVariable | SPHVariable)
            self._var = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`PushEta`.

        Parameters
        ----------
        butcher : ButcherTableau, default=None
            Butcher tableau used for explicit Runge-Kutta marker pushing.
            If ``None``, defaults to ``ButcherTableau()``.
        """

        butcher: ButcherTableau = None

        def __post_init__(self):
            # defaults
            if self.butcher is None:
                self.butcher = ButcherTableau()

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        self._options = new
        logger.info(f"\nNew options for propagator '{self.__class__.__name__}':\n{self._options}")

    @profile
    def allocate(self):
        # get kernel
        kernel = PyccelKernel(pusher_kernels.push_eta_stage)

        # define algorithm
        butcher = self.options.butcher
        # temp fix due to refactoring of ButcherTableau:

        args_kernel = (
            butcher.a_stage,
            butcher.b,
            butcher.c,
        )

        self._pusher = Pusher(
            self.variables.var.particles,
            kernel,
            args_kernel,
            self.domain.args_domain,
            alpha_in_kernel=1.0,
            n_stages=butcher.n_stages,
            mpi_sort="each",
        )

    @profile
    def __call__(self, dt):
        self._pusher(dt)

        # update_weights
        if self.variables.var.particles.control_variate:
            self.variables.var.particles.update_weights()
