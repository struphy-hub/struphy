"Only particle variables are updated."

import logging
from dataclasses import dataclass
from typing import Callable

from line_profiler import profile

from struphy.io.options import OptionsBase
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable
from struphy.pic.pushing.kernels.push_v_with_efield import push_v_with_efield
from struphy.pic.pushing.kernels.push_v_with_efield_mu import push_v_with_efield_mu
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

    With ``magnetic_moment=True``, each marker carries a magnetic moment :math:`\mu_p` (classical Pauli particle,
    see :class:`~struphy.pic.particles.PauliParticles6D`) and the force is augmented by the magnetic-moment force
    of the equilibrium field,

    .. math::

        \frac{\text{d} \mathbf{v}_p}{\text{d} t} = \frac{1}{\varepsilon}\mathbf{F}(\mathbf{x}_p)
        - \mu_p \nabla |\mathbf B_0|(\mathbf{x}_p) \,,

    i.e. :math:`\mathbf F = -\nabla(\phi + \varepsilon \mu_p |\mathbf B_0|)`, where :math:`\nabla |\mathbf B_0|`
    is the projected 1-form ``gradB1`` of the equilibrium (the same as in the guiding-center pushers).

    Without force field, potential and magnetic moment the propagator does nothing.
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
            assert new.space in ("Particles6D", "DeltaFParticles6D", "PauliParticles6D", "ParticlesSPH")
            self._var = new

    def __init__(
        self,
        force_field: FEECVariable | tuple[Callable] = None,
        potential: FEECVariable | Callable = None,
        magnetic_moment: bool = False,
    ):
        r"""
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
        magnetic_moment : bool, default=False
            Whether to add the magnetic-moment force :math:`-\mu_p \nabla |\mathbf B_0|` of each marker
            (needs :class:`~struphy.pic.particles.PauliParticles6D` and an equilibrium with magnetic field).
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
        self.magnetic_moment = magnetic_moment

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
        elif self.magnetic_moment:
            self.force_vector = self.derham.V1.zeros()
        else:
            # no force: the propagator does nothing
            return

        # instantiate Pusher
        args_kernel = (
            self.derham.args_derham,
            self.force_vector[0]._data,
            self.force_vector[1]._data,
            self.force_vector[2]._data,
        )

        if self.magnetic_moment:
            assert self.variables.var.space == "PauliParticles6D", (
                f"magnetic_moment=True needs markers with a magnetic moment (PauliParticles6D), not {self.variables.var.space}."
            )
            assert self.projected_equil is not None, "magnetic_moment=True needs an equilibrium with magnetic field."
            self._gradB1 = self.projected_equil.gradB1
            kernel = push_v_with_efield_mu
            args_kernel += (
                self._gradB1[0]._data,
                self._gradB1[1]._data,
                self._gradB1[2]._data,
            )
        else:
            kernel = push_v_with_efield

        args_kernel += (1.0 / self.epsilon,)

        self._pusher = Pusher(
            self.variables.var.particles,
            kernel,
            args_kernel,
            self.domain.args_domain,
            alpha_in_kernel=1.0,
            pushes_eta=False,
        )

    def __call__(self, dt):
        if self.force_field is not None:
            self._pusher(dt)
        elif self.potential is not None:
            self.derham.grad.dot(self.potential_vector, out=self.force_vector)
            self.force_vector *= -1.0
            self.force_vector.update_ghost_regions()
            self._pusher(dt)
        elif self.magnetic_moment:
            # zero force field, only the magnetic-moment force
            self._pusher(dt)
