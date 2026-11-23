"Only particle variables are updated."

import logging
from dataclasses import dataclass
from typing import Literal

from line_profiler import profile

from struphy.io.options import LiteralOptions, OptionsBase
from struphy.models.variables import SPHVariable
from struphy.pic.pushing.kernel_setup import KernelSetup
from struphy.pic.pushing.kernels.push_v_viscosity import push_v_viscosity
from struphy.pic.pushing.kernels.sph_mean_velocity_coeffs import sph_mean_velocity_coeffs
from struphy.pic.pushing.kernels.sph_viscosity_tensor import sph_viscosity_tensor
from struphy.pic.pushing.pusher import Pusher
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class PushVinViscousPotential(Propagator):
    r"""Viscous force on SPH markers, see :ref:`sph_viscosity`.

    **Continuous problem:** the velocity is accelerated by the divergence of the viscous stress,

    .. math::

        \rho\, \partial_t \mathbf u = \nabla \cdot \boldsymbol\sigma\,, \qquad
        \boldsymbol\sigma = \mu \left( \nabla \mathbf u + (\nabla \mathbf u)^\top - \frac 23 (\nabla \cdot \mathbf u)\, \mathbb I \right)\,,

    with mass density :math:`\rho` and dynamic viscosity :math:`\mu \geq 0`.

    **Semi-discrete problem:** the kernel :math:`W_h` from :mod:`~struphy.pic.sph_smoothing_kernels` is evaluated in
    logical coordinates. With the density 3-form and the velocity interpolant

    .. math::

        \rho^\eta_i = \sum_{j=1}^N w_j\, W_h(\boldsymbol\eta_i - \boldsymbol\eta_j)\,, \qquad
        \mathbf u_h(\boldsymbol\eta) = \sum_{j=1}^N \frac{w_j}{\rho^\eta_j}\, \mathbf v_j\, W_h(\boldsymbol\eta - \boldsymbol\eta_j)\,,

    the stress :math:`\boldsymbol\sigma_i` is computed at each marker from the gradient of :math:`\mathbf u_h`
    (transformed to Cartesian derivatives with :math:`DF^{-1}`), and for each marker :math:`p`

    .. math::

        \frac{\textnormal d \mathbf v_p}{\textnormal d t}
        = \frac{1}{\rho_p\, \sqrt g(\boldsymbol\eta_p)} \sum_{i=1}^N \frac{w_i}{\rho^\eta_i}\, \sqrt g(\boldsymbol\eta_i)\,
        \boldsymbol\sigma_i\, DF^{-\top}(\boldsymbol\eta_i)\, \nabla W_h(\boldsymbol\eta_p - \boldsymbol\eta_i)\,,

    which is the SPH estimate of :math:`\frac 1\rho \nabla \cdot \boldsymbol\sigma` in Piola form, with the physical
    density :math:`\rho_p = \rho^\eta_p / \sqrt g(\boldsymbol\eta_p)` and :math:`\sqrt g = |\det DF|`. The
    coefficients :math:`w_i \mathbf v_i/\rho^\eta_i` and :math:`\sqrt g\, (w_i/\rho^\eta_i) \boldsymbol\sigma_i DF^{-\top}`
    (with a minus sign) are computed once per time step and stored in the marker array.

    **Time discretization:** forward Euler (``algo="forward_euler"``),

    .. math::

        \mathbf v_p^{n+1} = \mathbf v_p^n + \Delta t\, \frac{\textnormal d \mathbf v_p}{\textnormal d t}\Big|^n\,.
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
        init_kernel_1 = sph_mean_velocity_coeffs
        first_free_idx = particles.args_markers.first_free_idx

        init_kernel_2 = sph_viscosity_tensor

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
            KernelSetup(
                kernel=init_kernel_1,
                output_indices=(first_free_idx, first_free_idx + 1, first_free_idx + 2),
                args=args_init_mean,
            ),
        )

        self.add_init_kernel(
            KernelSetup(
                kernel=init_kernel_2,
                output_indices=(
                    first_free_idx + 3,
                    first_free_idx + 4,
                    first_free_idx + 5,
                    first_free_idx + 6,
                    first_free_idx + 7,
                    first_free_idx + 8,
                    first_free_idx + 9,
                    first_free_idx + 10,
                    first_free_idx + 11,
                ),
                args=args_init_visc,
            ),
        )

        kernel = push_v_viscosity

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
            pushes_eta=False,
        )

    def __call__(self, dt):
        self.variables.fluid.particles.put_particles_in_boxes()
        self._pusher(dt)
