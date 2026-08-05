"Only particle variables are updated."

import logging
from dataclasses import dataclass
from typing import Literal

import cunumpy as xp
from line_profiler import profile

from struphy.io.options import LiteralOptions, OptionsBase
from struphy.models.variables import SPHVariable
from struphy.pic.pushing import eval_kernels_sph, pusher_kernels_sph
from struphy.pic.pushing.pusher import Pusher
from struphy.propagators.base import Propagator
from cunumpy import PyccelKernel
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class PushVinSPHpressure(Propagator):
    r"""For each marker :math:`p`, solves

    .. math::

        \frac{\textnormal d \mathbf v_p(t)}{\textnormal d t} = \mathbf g - \sum_{i=1}^N w_i \left( \frac{\partial \mathcal U}{\partial \rho}(\boldsymbol \eta_p) + \frac{\partial \mathcal U}{\partial \rho}(\boldsymbol \eta_i) \right) DF^{-\top}\nabla W_h(\boldsymbol \eta_p - \boldsymbol \eta_i) \,,

    where :math:`\mathbf g` is a constant acceleration and the second term corresponds to the pressure gradient.
    Here, :math:`\mathcal U(\rho)` denotes the internal energy per unit mass
    as a function of the mass density :math:`\rho` and :math:`DF^{-\top}` denotes the inverse transpose Jacobian
    arising in the pull back of the gradient of the smoothing kernel :math:`W_h`
    chosen from :mod:`~struphy.pic.sph_smoothing_kernels`.
    Two choices of the internal energy are implemented:

    * Isothermal closure: :math:`\mathcal U(\rho) = \kappa \, \ln(\rho)`, where :math:`\kappa` is constant.
    * Polytropic closure: :math:`\mathcal U(\rho) = \kappa \, \rho^{\gamma - 1} / (\gamma - 1)`, where :math:`\kappa` is the polytropic constant and :math:`\gamma = C_p / C_v` is the polytropic index.

    Time stepping is forward Euler.
    """

    class Variables:
        """Container for variables advanced by :class:`PushVinSPHpressure`.

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
        """Configuration options for :class:`PushVinSPHpressure`.

        Parameters
        ----------
        kernel_type : LiteralOptions.OptsKernel, default="gaussian_2d"
            Smoothing kernel used for density and pressure-force evaluation.

        kernel_width : tuple, default=None
            Kernel widths per logical direction. If ``None``, defaults to
            ``(1 / n_i)`` based on sorting boxes.

        algo : {"forward_euler"}, default="forward_euler"
            Time stepping algorithm for velocity pushing.

        gravity : tuple, default=(0.0, 0.0, 0.0)
            Constant gravity vector added in the SPH pressure push.

        kappa : float, default=1.0
            Coefficient in the internal energy function.

        thermodynamics : {"isothermal", "polytropic"}, default="isothermal"
            Thermodynamic closure selecting the SPH pressure kernel.
        """

        # specific literals
        OptsAlgo = Literal["forward_euler"]
        OptsThermo = Literal["isothermal", "polytropic"]
        # propagator options
        kernel_type: LiteralOptions.OptsKernel = "gaussian_2d"
        kernel_width: tuple = None
        algo: OptsAlgo = "forward_euler"
        gravity: tuple = (0.0, 0.0, 0.0)
        kappa: float = 1.0
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
        logger.info(f"\nNew options for propagator '{self.__class__.__name__}':\n{self._options}")

    @profile
    def allocate(self):
        # init kernel for evaluating density etc. before each time step.
        init_kernel = eval_kernels_sph.sph_pressure_coeffs

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
            kernel = PyccelKernel(pusher_kernels_sph.push_v_sph_pressure)
        elif self.options.thermodynamics == "polytropic":
            kernel = PyccelKernel(pusher_kernels_sph.push_v_sph_pressure_ideal_gas)

        gravity = xp.array(self.options.gravity, dtype=float)

        args_kernel = (
            boxes,
            neighbours,
            holes,
            *periodic,
            kernel_nr,
            *self.options.kernel_width,
            gravity,
            self.options.kappa,
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
        particles = self.variables.fluid.particles

        # The "noslip" BC reflects ghost particles and, when ``mean_velocity_index`` is set,
        # negates the three auxiliary columns starting at that index (see Particles._mirror_particles).
        # That negation is meant for the viscous mean-velocity coefficients (which are odd under
        # wall reflection). Here those same columns hold the SPH density rho, w/rho and w*rho^(gamma-2)
        # computed by sph_pressure_coeffs -- scalars that are EVEN under reflection and must NOT be
        # flipped. Aliasing the index (it defaults to first_free_idx, which is exactly where the
        # pressure coefficients live) would otherwise give the mirror particles a negative w/rho, so
        # the symmetric pressure-gradient sum no longer cancels at the wall and the near-wall markers
        # receive a spurious normal force into the wall. Suppress the flip for the duration of the
        # pressure push and restore it afterwards so the viscous propagator keeps its correct behaviour.
        saved_mean_velocity_index = particles._mean_velocity_index
        particles._mean_velocity_index = None
        try:
            particles.put_particles_in_boxes()
            self._pusher(dt)
        finally:
            particles._mean_velocity_index = saved_mean_velocity_index
