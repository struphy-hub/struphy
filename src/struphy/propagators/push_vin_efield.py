"Only particle variables are updated."

import logging
from collections.abc import Callable
from dataclasses import dataclass

from line_profiler import profile

from struphy.io.options import OptionsBase
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable
from struphy.pic.pushing import pusher_kernels
from struphy.pic.pushing.pusher import Pusher
from struphy.propagators.base import Propagator
from struphy.utils.pyccel import Pyccelkernel

logger = logging.getLogger("struphy")


class PushVinEfield(Propagator):
    r"""Push the velocities according to

    .. math::

        \frac{\text{d} \mathbf{v}_p}{\text{d} t} = \frac{1}{\varepsilon}\mathbf{E}(\mathbf{x}_p) \,,

    where :math:`\varepsilon \in \mathbb R` is a constant. In logical coordinates, given by :math:`\mathbf x = F(\boldsymbol \eta)`:

    .. math::

        \frac{\text{d} \mathbf{v}_p}{\text{d} t} = \frac{1}{\varepsilon}DF^{-\top}\hat{\mathbf E}^1(\boldsymbol \eta_p)  \,,

    which is solved analytically. :math:`\mathbf E` can optionally be defined
    through a potential, :math:`\mathbf E = - \nabla \phi`.
    """

    class Variables:
        """Container for variables advanced by :class:`PushVinEfield`.

        Attributes
        ----------
        var : PICVariable or SPHVariable
            Particle variable whose velocities are advanced.
        """

        def __init__(self):
            self._var: PICVariable | SPHVariable = None

        @property
        def var(self) -> PICVariable | SPHVariable:
            return self._var

        @var.setter
        def var(self, new):
            assert isinstance(new, PICVariable | SPHVariable)
            assert new.space in ("Particles6D", "DeltaFParticles6D", "ParticlesSPH")
            self._var = new

    def __init__(
        self,
        phi: FEECVariable | Callable = None,
        e_field: FEECVariable | tuple[Callable] = None,
    ):
        """
        Parameters
        ----------
        phi : FEECVariable or Callable, default=None
            Electrostatic potential from which the electric field is built as
            ``-grad(phi)``. If provided, it overrides ``e_field``.
            Accepted forms are an ``H1`` FEEC variable or a callable projected
            via ``L2Projector``.
        e_field : FEECVariable or tuple of Callables, default=None
            Electric field used directly in velocity pushing.
            Accepted forms are an ``Hcurl`` FEEC variable or a tuple of
            callables to be projected. Ignored when ``phi`` is set.
        """
        self.variables = self.Variables()

        if isinstance(phi, FEECVariable):
            assert phi.space == "H1"
        if isinstance(e_field, FEECVariable):
            assert e_field.space == "Hcurl"

        self.phi = phi
        self.e_field = e_field

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`PushVinEfield`."""

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
        logger.info(f"\nNew options for propagator '{self.__class__.__name__}':\n{self._options}")

    @profile
    def allocate(self):
        # scaling factor
        self._epsilon = self.variables.var.species.equation_params.epsilon

        self._e_field = None

        if self.e_field is not None:
            if isinstance(self.e_field, tuple[Callable]):
                self._e_field = self.derham.P1(self.e_field)
            else:
                self._e_field = self.e_field.spline.vector

        if self.phi is not None:
            if isinstance(self.phi, Callable):
                _phi = self.derham.P0(self.phi)
            else:
                _phi = self.phi.spline.vector
            self._e_field = self.derham.grad.dot(_phi)
            self._e_field.update_ghost_regions()  # very important, we will move it inside grad
            self._e_field *= -1.0

        if self._e_field is not None:
            # instantiate Pusher
            args_kernel = (
                self.derham.args_derham,
                self._e_field[0]._data,
                self._e_field[1]._data,
                self._e_field[2]._data,
                1.0 / self._epsilon,
            )

            self._pusher = Pusher(
                self.variables.var.particles,
                Pyccelkernel(pusher_kernels.push_v_with_efield),
                args_kernel,
                self.domain.args_domain,
                alpha_in_kernel=1.0,
            )

    def __call__(self, dt):
        if self._e_field is not None:
            self._pusher(dt)
