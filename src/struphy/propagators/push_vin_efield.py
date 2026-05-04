"Only particle variables are updated."

from dataclasses import dataclass
from typing import Callable

from line_profiler import profile

from struphy.models.variables import FEECVariable, PICVariable, SPHVariable
from struphy.pic.pushing import pusher_kernels
from struphy.pic.pushing.pusher import Pusher
from struphy.propagators.base import Propagator
from struphy.utils.pyccel import Pyccelkernel
class PushVinEfield(Propagator):
    r"""Push the velocities according to

    .. math::

        \frac{\text{d} \mathbf{v}_p}{\text{d} t} = \frac{1}{\varepsilon} \, \mathbf{E}(\mathbf{x}_p) \,,

    where :math:`\varepsilon \in \mathbb R` is a constant. In logical coordinates, given by :math:`\mathbf x = F(\boldsymbol \eta)`:

    .. math::

        \frac{\text{d} \mathbf{v}_p}{\text{d} t} = \frac{1}{\varepsilon} \, DF^{-\top} \hat{\mathbf E}^1(\boldsymbol \eta_p)  \,,

    which is solved analytically. :math:`\mathbf E` can optionally be defined
    through a potential, :math:`\mathbf E = - \nabla \phi`.
    """

    class Variables:
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

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
        # propagator options
        e_field: FEECVariable | tuple[Callable] = None
        phi: FEECVariable | Callable = None

        def __post_init__(self):
            # checks
            if self.e_field is not None:
                assert isinstance(self.e_field, tuple[Callable]) or self.e_field.space == "Hcurl"
            else:
                if self.phi is not None:
                    assert isinstance(self.phi, Callable) or self.phi.space == "H1"

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        self._options = new

    @profile
    def allocate(self, verbose: bool = False):
        # scaling factor
        self._epsilon = self.variables.var.species.equation_params.epsilon

        self._e_field = None

        if self.options.e_field is not None:
            if isinstance(self.options.e_field, tuple[Callable]):
                self._e_field = self.derham.P1(self.options.e_field)
            else:
                self._e_field = self.options.e_field.spline.vector

        if self.options.phi is not None:
            if isinstance(self.options.phi, Callable):
                _phi = self.derham.P0(self.options.phi)
            else:
                _phi = self.options.phi.spline.vector
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


