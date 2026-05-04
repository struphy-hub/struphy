"Only particle variables are updated."

from dataclasses import dataclass

from line_profiler import profile

from struphy.models.variables import PICVariable
from struphy.ode.utils import ButcherTableau
from struphy.pic.accumulation import accum_kernels
from struphy.pic.accumulation.particles_to_grid import AccumulatorVector
from struphy.pic.pushing import pusher_kernels
from struphy.pic.pushing.pusher import Pusher
from struphy.propagators.base import Propagator
from struphy.utils.pyccel import Pyccelkernel
class PushDeterministicDiffusion(Propagator):
    r"""For each marker :math:`p`, solves

    .. math::

        \frac{\textnormal d \mathbf x_p(t)}{\textnormal d t} = - D \, \frac{\nabla u}{ u}\mathbf (\mathbf x_p(t))\,,

    in logical space given by :math:`\mathbf x = F(\boldsymbol \eta)`:

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = - G\, D \, \frac{\nabla \Pi^0_{L^2}u_h}{\Pi^0_{L^2} u_h}\mathbf (\boldsymbol \eta_p(t))\,,
        \qquad [\Pi^0_{L^2, ijk} u_h](\boldsymbol \eta_p) = \frac 1N \sum_{p} w_p \boldsymbol \Lambda^0_{ijk}(\boldsymbol \eta_p)\,,

    where :math:`D>0` is a positive diffusion coefficient.

    Available algorithms:

    * Explicit from :class:`~struphy.ode.utils.ButcherTableau`
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
            assert new.space == "Particles3D"
            self._var = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
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

    @profile
    def allocate(self, verbose: bool = False):
        self._bc_type = self.options.bc_type
        self._diffusion = self.options.diff_coeff

        self._tmp = self.derham.V1.zeros()

        # choose algorithm
        self._butcher = self.options.butcher
        # temp fix due to refactoring of ButcherTableau:
        import cunumpy as xp

        self._butcher._a = xp.diag(self._butcher.a, k=-1)
        self._butcher._a = xp.array(list(self._butcher.a) + [0.0])

        particles = self.variables.var.particles

        self._u_on_grid = AccumulatorVector(
            particles,
            "H1",
            Pyccelkernel(accum_kernels.charge_density_0form),
            self.mass_ops,
            self.domain.args_domain,
        )

        # instantiate Pusher
        args_kernel = (
            self.derham.args_derham,
            self._u_on_grid.vectors[0]._data,
            self._tmp[0]._data,
            self._tmp[1]._data,
            self._tmp[2]._data,
            self._diffusion,
            self._butcher.a,
            self._butcher.b,
            self._butcher.c,
        )

        self._pusher = Pusher(
            particles,
            Pyccelkernel(pusher_kernels.push_deterministic_diffusion_stage),
            args_kernel,
            self.domain.args_domain,
            alpha_in_kernel=1.0,
            n_stages=self._butcher.n_stages,
            mpi_sort="each",
        )

    def __call__(self, dt):
        """
        TODO
        """
        particles = self.variables.var.particles

        # accumulate
        self._u_on_grid()

        # take gradient
        pi_u = self._u_on_grid.vectors[0]
        grad_pi_u = self.derham.grad.dot(pi_u, out=self._tmp)
        grad_pi_u.update_ghost_regions()

        # push markers
        self._pusher(dt)

        # update_weights
        if particles.control_variate:
            particles.update_weights()


