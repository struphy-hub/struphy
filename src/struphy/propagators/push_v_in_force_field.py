"Only particle variables are updated."

import logging
from dataclasses import dataclass
from typing import Callable

from cunumpy import PyccelKernel
from line_profiler import profile

from struphy.io.options import OptionsBase
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable
from struphy.pic.pushing import pusher_kernels
from struphy.pic.pushing.pusher import Pusher
from struphy.propagators.base import Propagator

logger = logging.getLogger("struphy")


class PushVinForceField(Propagator):
    r"""Push the velocities according to

    .. math::

        \frac{\text{d} \mathbf{v}_p}{\text{d} t} = \frac{1}{\varepsilon}\mathbf{F}(\mathbf{x}_p) \,,

    where :math:`\varepsilon \in \mathbb R` is a constant species parameter. In logical coordinates, given by :math:`\mathbf x = F(\boldsymbol \eta)`:

    .. math::

        \frac{\text{d} \mathbf{v}_p}{\text{d} t} = \frac{1}{\varepsilon}DF^{-\top}\hat{\mathbf F}^1(\boldsymbol \eta_p)  \,,

    which is solved analytically. :math:`\mathbf F` can optionally be defined
    through a potential, :math:`\mathbf F = - \nabla \phi`.
    """

    class Variables:
        """Container for variables advanced by :class:`PushVinForceField`.

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
        force_field: FEECVariable | tuple[Callable] = None,
        potential: FEECVariable | Callable = None,
    ):
        """
        Parameters
        ----------
        force_field : FEECVariable or tuple of Callables, default=None
            Force field used directly in velocity pushing.
            Accepted forms are an ``Hcurl`` FEEC variable or a tuple of
            callables to be projected. If provided, ``potential`` is ignored.
        potential : FEECVariable or Callable, default=None
            Scalar potential from which the force field is built as
            ``-grad(potential)``. Accepted forms are an ``H1`` FEEC variable or a callable projected
            via ``L2Projector``.
        """
        self.variables = self.Variables()

        if force_field is not None:
            if isinstance(force_field, FEECVariable):
                assert force_field.space == "Hcurl"
            else:
                assert isinstance(force_field, tuple) and all(callable(x) for x in force_field)
            potential = None
        elif potential is not None:
            if isinstance(potential, FEECVariable):
                assert potential.space == "H1"
            else:
                assert callable(potential)

        self.force_field = force_field
        self.potential = potential

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`PushVinForceField`."""

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
        # scaling factor, retrieved from variable's species
        self.epsilon = self.variables.var.species.equation_params.epsilon

        if self.force_field is not None:
            self.potential_vector = None
            if isinstance(self.force_field, FEECVariable):
                self.force_vector = self.force_field.spline.vector
            else:
                self.force_vector = self.derham.P1(self.force_field)
        elif self.potential is not None:
            if isinstance(self.potential, FEECVariable):
                self.potential_vector = self.potential.spline.vector
            else:
                self.potential_vector = self.derham.P0(self.potential)
            self.force_vector = self.derham.grad.dot(self.potential_vector)
            self.force_vector *= -1.0
            self.force_vector.update_ghost_regions()
        else:
            self.force_vector = self.derham.V1.zeros()

        # instantiate Pusher
        args_kernel = (
            self.derham.args_derham,
            self.force_vector[0]._data,
            self.force_vector[1]._data,
            self.force_vector[2]._data,
            1.0 / self.epsilon,
        )

        self._pusher = Pusher(
            self.variables.var.particles,
            PyccelKernel(pusher_kernels.push_v_with_efield),
            args_kernel,
            self.domain.args_domain,
            alpha_in_kernel=1.0,
        )

    def __call__(self, dt):
        if self.force_field is not None:
            self._pusher(dt)
        elif self.potential is not None:
            self.derham.grad.dot(self.potential_vector, out=self.force_vector)
            self.force_vector *= -1.0
            self.force_vector.update_ghost_regions()
            self._pusher(dt)
        else:
            pass
