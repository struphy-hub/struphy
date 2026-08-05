"Only particle variables are updated."

import logging
from dataclasses import dataclass

from line_profiler import profile
from numpy import array, random

from struphy.io.options import OptionsBase
from struphy.models.variables import PICVariable
from struphy.ode.utils import ButcherTableau
from struphy.pic.pushing import pusher_kernels
from struphy.pic.pushing.pusher import Pusher
from struphy.propagators.base import Propagator
from cunumpy import Pyccelkernel

logger = logging.getLogger("struphy")


class PushRandomDiffusion(Propagator):
    r"""For each marker :math:`p`, solves

    .. math::

        \textnormal d \mathbf x_p(t) = \sqrt{2 D} \, \textnormal d \mathbf B_{t}\,,

    where :math:`D>0` is a positive diffusion coefficient and :math:`\textnormal d \mathbf B_{t}` is a Wiener process,

    .. math::

        \mathbf B_{t + \Delta t} - \mathbf B_{t} = \sqrt{\Delta t} \,\mathcal N(0;1)\,,

    with :math:`\mathcal N(0;1)` denoting the standard normal distribution with mean zero and variance one.

    Available algorithms:

    * ``forward_euler`` (1st order)
    """

    class Variables:
        """Container for variables advanced by :class:`PushRandomDiffusion`.

        Attributes
        ----------
        var : PICVariable
            Particle variable in ``"Particles3D"`` space whose marker
            positions are advanced.
        """

        def __init__(self):
            self._var: PICVariable = None

        @property
        def var(self) -> PICVariable:
            return self._var

        @var.setter
        def var(self, new):
            assert isinstance(new, PICVariable)
            assert new.space == "Particles3D"
            self._var = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`PushRandomDiffusion`.

        Parameters
        ----------
        butcher : ButcherTableau, default=None
            Butcher tableau used for explicit integration.
            If ``None``, defaults to ``ButcherTableau()``.

        bc_type : tuple, default=("periodic", "periodic", "periodic")
            Boundary-condition types per logical coordinate.

        diff_coeff : float, default=1.0
            Positive diffusion coefficient :math:`D` used in the stochastic
            increment.
        """

        butcher: ButcherTableau = None
        bc_type: tuple = ("periodic", "periodic", "periodic")
        diff_coeff: float = 1.0

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
        self._bc_type = self.options.bc_type
        self._diffusion = self.options.diff_coeff

        particles = self.variables.var.particles

        self._noise = array(particles.markers[:, :3])

        self._butcher = self.options.butcher
        # temp fix due to refactoring of ButcherTableau:

        # instantiate Pusher
        args_kernel = (
            self._noise,
            self._diffusion,
            self._butcher.a_stage,
            self._butcher.b,
            self._butcher.c,
        )

        self._pusher = Pusher(
            particles,
            Pyccelkernel(pusher_kernels.push_random_diffusion_stage),
            args_kernel,
            self.domain.args_domain,
            alpha_in_kernel=1.0,
            n_stages=self._butcher.n_stages,
            mpi_sort="each",
        )

        # self._tmp = self.derham.V1.zeros()
        self._mean = [0, 0, 0]
        self._cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def __call__(self, dt):
        """
        TODO
        """

        particles = self.variables.var.particles

        self._noise[:] = random.multivariate_normal(
            self._mean,
            self._cov,
            len(particles.markers),
        )

        # push markers
        self._pusher(dt)

        # update_weights
        if particles.control_variate:
            particles.update_weights()
