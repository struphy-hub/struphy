"Only particle variables are updated."

import logging
from dataclasses import dataclass

from line_profiler import profile

from struphy.io.options import LiteralOptions, OptionsBase
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable
from struphy.ode.utils import ButcherTableau
from struphy.pic.pushing import pusher_kernels
from struphy.pic.pushing.pusher import Pusher
from struphy.propagators.base import Propagator
from struphy.utils.pyccel import Pyccelkernel
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class PushEtaPC(Propagator):
    r"""For each marker :math:`p`, solves

    .. math::

        \frac{\textnormal d \mathbf x_p(t)}{\textnormal d t} = \mathbf v_p + \mathbf U (\mathbf x_p(t))\,,

    for constant :math:`\mathbf v_p` and :math:`\mathbf U` in logical space given by :math:`\mathbf x = F(\boldsymbol \eta)`:

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \,\mathbf v_p + \textnormal{vec}(\hat{\mathbf U}) \,,

    where

    .. math::

        \textnormal{vec}( \hat{\mathbf U}^{1}) = G^{-1}\hat{\mathbf U}^{1}\,,\qquad \textnormal{vec}( \hat{\mathbf U}^{2}) = \frac{\hat{\mathbf U}^{2}}{\sqrt g}\,, \qquad \textnormal{vec}( \hat{\mathbf U}) = \hat{\mathbf U}\,.

    Available algorithms:

    * ``rk4`` (4th order, default)
    * ``forward_euler`` (1st order)
    * ``heun2`` (2nd order)
    * ``rk2`` (2nd order)
    * ``heun3`` (3rd order)
    """

    class Variables:
        """Container for variables advanced by :class:`PushEtaPC`.

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

    def __init__(self, u_tilde: FEECVariable = None):
        self.variables = self.Variables()
        self.u_tilde = u_tilde

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`PushEtaPC`.

        Parameters
        ----------
        butcher : ButcherTableau, default=None
            Butcher tableau used for explicit Runge-Kutta marker pushing.
            If ``None``, defaults to ``ButcherTableau()``.

        use_perp_model : bool, default=True
            Flag forwarded to the particle kernel to select the perpendicular
            model formulation.

        u_space : LiteralOptions.OptsVecSpace, default="Hdiv"
            FEEC space used to interpret ``u_tilde`` in the pusher kernel.
        """

        butcher: ButcherTableau = None
        use_perp_model: bool = True
        u_space: LiteralOptions.OptsVecSpace = "Hdiv"

        def __post_init__(self):
            # checks
            check_option(self.u_space, LiteralOptions.OptsVecSpace)

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
        self._u_tilde = self.u_tilde.spline.vector

        # get kernell:
        if self.options.u_space == "Hcurl":
            kernel = Pyccelkernel(pusher_kernels.push_pc_eta_stage_Hcurl)
        elif self.options.u_space == "Hdiv":
            kernel = Pyccelkernel(pusher_kernels.push_pc_eta_stage_Hdiv)
        elif self.options.u_space == "H1vec":
            kernel = Pyccelkernel(pusher_kernels.push_pc_eta_stage_H1vec)
        else:
            raise ValueError(
                f'{self.options.u_space =} not valid, choose from "Hcurl", "Hdiv" or "H1vec.',
            )

        # define algorithm
        butcher = self.options.butcher
        # temp fix due to refactoring of ButcherTableau:

        args_kernel = (
            self.derham.args_derham,
            self._u_tilde[0]._data,
            self._u_tilde[1]._data,
            self._u_tilde[2]._data,
            self.options.use_perp_model,
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

    def __call__(self, dt):
        self._u_tilde.update_ghost_regions()

        self._pusher(dt)

        # update_weights
        if self.variables.var.particles.control_variate:
            self.variables.var.particles.update_weights()
