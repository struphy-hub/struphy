"Only particle variables are updated."

from dataclasses import dataclass
from typing import Literal

import cunumpy as xp
from line_profiler import profile

from struphy.io.options import LiteralOptions
from struphy.models.variables import SPHVariable
from struphy.pic.pushing import eval_kernels_gc, pusher_kernels
from struphy.pic.pushing.pusher import Pusher
from struphy.propagators.base import Propagator
from struphy.utils.pyccel import Pyccelkernel
from struphy.utils.utils import check_option
class PushVinSPHpressure(Propagator):
    r"""For each marker :math:`p`, solves

    .. math::

        \frac{\textnormal d \mathbf v_p(t)}{\textnormal d t} = \kappa_p \sum_{i=1}^N w_i \left( \frac{1}{\rho^{N,h}(\boldsymbol \eta_p)} + \frac{1}{\rho^{N,h}(\boldsymbol \eta_i)} \right) DF^{-\top}\nabla W_h(\boldsymbol \eta_p - \boldsymbol \eta_i) \,,

    where :math:`DF^{-\top}` denotes the inverse transpose Jacobian, and with the smoothed density

    .. math::

        \rho^{N,h}(\boldsymbol \eta) = \frac 1N \sum_{j=1}^N w_j \, W_h(\boldsymbol \eta - \boldsymbol \eta_j)\,,

    where :math:`W_h(\boldsymbol \eta)` is a smoothing kernel from :mod:`~struphy.pic.sph_smoothing_kernels`.
    Time stepping:

    * Explicit from :class:`~struphy.ode.utils.ButcherTableau`
    """

    class Variables:
        def __init__(self):
            self._fluid: SPHVariable = None

        @property
        def fluid(self) -> SPHVariable:
            return self._fluid

        @fluid.setter
        def fluid(self, new):
            assert isinstance(new, SPHVariable)
            assert new.space == "ParticlesSPH"
            self._fluid = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
        # specific literals
        OptsAlgo = Literal["forward_euler"]
        OptsThermo = Literal["isothermal", "polytropic"]
        # propagator options
        kernel_type: LiteralOptions.OptsKernel = "gaussian_2d"
        kernel_width: tuple = None
        algo: OptsAlgo = "forward_euler"
        gravity: tuple = (0.0, 0.0, 0.0)
        thermodynamics: OptsThermo = "isothermal"

        def __post_init__(self):
            # checks
            check_option(self.kernel_type, LiteralOptions.OptsKernel)
            check_option(self.algo, self.OptsAlgo)
            check_option(self.thermodynamics, self.OptsThermo)

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
        # init kernel for evaluating density etc. before each time step.
        init_kernel = eval_kernels_gc.sph_pressure_coeffs

        particles = self.variables.fluid.particles

        first_free_idx = particles.args_markers.first_free_idx
        comps = (0, 1, 2)

        boxes = particles.sorting_boxes.boxes
        neighbours = particles.sorting_boxes.neighbours
        holes = particles.holes
        periodic = [bci == "periodic" for bci in particles.bc]
        kernel_nr = particles.ker_dct()[self.options.kernel_type]

        if self.options.kernel_width is None:
            self.options.kernel_width = tuple([1 / ni for ni in particles.boxes_per_dim])
        else:
            assert all([hi <= 1 / ni for hi, ni in zip(self.options.kernel_width, particles.boxes_per_dim)])

        # init kernel
        args_init = (
            boxes,
            neighbours,
            holes,
            *periodic,
            kernel_nr,
            *self.options.kernel_width,
        )

        self.add_init_kernel(
            init_kernel,
            first_free_idx,
            comps,
            args_init,
        )

        # pusher kernel
        if self.options.thermodynamics == "isothermal":
            kernel = Pyccelkernel(pusher_kernels.push_v_sph_pressure)
        elif self.options.thermodynamics == "polytropic":
            kernel = Pyccelkernel(pusher_kernels.push_v_sph_pressure_ideal_gas)

        gravity = xp.array(self.options.gravity, dtype=float)

        args_kernel = (
            boxes,
            neighbours,
            holes,
            *periodic,
            kernel_nr,
            *self.options.kernel_width,
            gravity,
        )

        # the Pusher class wraps around all kernels
        self._pusher = Pusher(
            particles,
            kernel,
            args_kernel,
            self.domain.args_domain,
            alpha_in_kernel=0.0,
            init_kernels=self.init_kernels,
        )

    @profile
    def __call__(self, dt):
        self.variables.fluid.particles.put_particles_in_boxes()
        self._pusher(dt)


