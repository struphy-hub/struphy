"Only particle variables are updated."

import logging
from dataclasses import dataclass
from typing import Literal

from line_profiler import profile

from struphy.io.options import LiteralOptions, OptionsBase
from struphy.models.variables import SPHVariable
from struphy.pic.pushing import eval_kernels_sph, pusher_kernels
from struphy.pic.pushing.pusher import Pusher
from struphy.propagators.base import Propagator
from struphy.utils.pyccel import Pyccelkernel
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class PushVinViscousPotential(Propagator):
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
        """Container for variables advanced by :class:`PushVinViscousPotential`.

        Attributes
        ----------
        fluid : SPHVariable
            SPH particle variable in ``"ParticlesSPH"`` space.
        """

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

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`PushVinViscousPotential`.

        Parameters
        ----------
        kernel_type : LiteralOptions.OptsKernel, default="gaussian_2d"
            Smoothing kernel used for SPH evaluations.

        kernel_width : tuple, default=None
            Kernel widths per logical direction. If ``None``, defaults to
            ``(1 / n_i)`` based on sorting boxes.

        algo : {"forward_euler"}, default="forward_euler"
            Time stepping algorithm for the viscous potential push.

        mu : float, default=1.0
            Dynamic viscosity coefficient used by the viscosity tensor kernel.
            Must be non-negative.
        """

        # specific literals
        OptsAlgo = Literal["forward_euler"]
        # propagator options
        kernel_type: LiteralOptions.OptsKernel = "gaussian_2d"
        kernel_width: tuple = None
        algo: OptsAlgo = "forward_euler"
        mu: float = 1.0

        def __post_init__(self):
            # checks
            check_option(self.kernel_type, LiteralOptions.OptsKernel)
            check_option(self.algo, self.OptsAlgo)
            # validate mu
            if not isinstance(self.mu, (int, float)):
                raise TypeError("Options.mu must be a number")
            if self.mu < 0:
                raise ValueError("Options.mu must be non-negative")

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
    def allocate(self):  # ersetzt init
        particles = self.variables.fluid.particles

        # init kernel for evaluating density etc. before each time step.
        init_kernel_1 = eval_kernels_sph.sph_mean_velocity_coeffs
        first_free_idx = particles.args_markers.first_free_idx
        comps = (0, 1, 2)

        init_kernel_2 = eval_kernels_sph.sph_viscosity_tensor
        comps_tensor = (0, 1, 2, 3, 4, 5, 6, 7, 8)

        boxes = particles.sorting_boxes.boxes
        neighbours = particles.sorting_boxes.neighbours
        holes = particles.holes
        periodic = [bci == "periodic" for bci in particles.bc]
        kernel_nr = particles.ker_dct()[self.options.kernel_type]

        if self.options.kernel_width is None:
            self.options.kernel_width = tuple([1 / ni for ni in particles.boxes_per_dim])
        else:
            assert all([hi <= 1 / ni for hi, ni in zip(self.options.kernel_width, particles.boxes_per_dim)])

        # for sph_mean_velocity_coeffs
        args_init_mean = (
            boxes,
            neighbours,
            holes,
            *periodic,
            kernel_nr,
            *self.options.kernel_width,
        )

        # for sph_viscosity_tensor
        args_init_visc = (
            boxes,
            neighbours,
            holes,
            *periodic,
            kernel_nr,
            *self.options.kernel_width,
            self.options.mu,
        )

        self.add_init_kernel(
            init_kernel_1,
            first_free_idx,
            comps,
            args_init_mean,
        )

        self.add_init_kernel(
            init_kernel_2,
            first_free_idx + 3,
            comps_tensor,
            args_init_visc,
        )

        kernel = Pyccelkernel(pusher_kernels.push_v_viscosity)

        args_kernel = (
            boxes,
            neighbours,
            holes,
            *periodic,
            kernel_nr,
            *self.options.kernel_width,
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

    def __call__(self, dt):
        self.variables.fluid.particles.put_particles_in_boxes()
        self._pusher(dt)
