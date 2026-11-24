"Only particle variables are updated."

import logging
from dataclasses import dataclass
from typing import Literal

import cunumpy as xp
from line_profiler import profile

from struphy.io.options import OptionsBase
from struphy.models.variables import SPHVariable
from struphy.pic.pushing.kernel_setup import KernelSetup
from struphy.pic.pushing.kernels.push_v_sph_pressure import push_v_sph_pressure
from struphy.pic.pushing.kernels.push_v_sph_pressure_ideal_gas import push_v_sph_pressure_ideal_gas
from struphy.pic.pushing.kernels.sph_pressure_coeffs import sph_pressure_coeffs
from struphy.pic.pushing.pusher import Pusher
from struphy.propagators.base import Propagator
from struphy.propagators.utils import check_removed_kernel_options
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class PushVinSPHpressure(Propagator):
    r"""Pressure force and constant acceleration on SPH markers, see :ref:`sph_pressure`.

    **Continuous problem:** the velocity is accelerated by the pressure gradient and a constant acceleration
    :math:`\mathbf g`,

    .. math::

        \partial_t \mathbf u = - \frac{1}{\rho} \nabla p(\rho) + \mathbf g\,, \qquad
        p(\rho) = \rho^2\, \frac{\partial \mathcal U}{\partial \rho}(\rho)\,,

    with the internal energy per unit mass :math:`\mathcal U(\rho)` of one of two closures (option ``thermodynamics``):

    * ``"isothermal"``: :math:`\mathcal U(\rho) = \kappa\, \ln \rho`, i.e. :math:`p = \kappa \rho`;
    * ``"polytropic"``: :math:`\mathcal U(\rho) = \kappa\, \rho^{\gamma - 1} / (\gamma - 1)`, i.e. :math:`p = \kappa \rho^\gamma`,
      with the adiabatic index :math:`\gamma > 1` (option ``gamma``).

    **Semi-discrete problem:** the Euler-Lagrange equations of the SPH Lagrangian
    :math:`L_N = \sum_p w_p \big(\frac 12 |\mathbf v_p|^2 - \mathcal U(\rho_p)\big)`. The radial kernel :math:`W_h`
    is defined and normalized in physical space (:class:`~struphy.particles.parameters.SPHParameters`) and evaluated
    at the physical marker positions :math:`\mathbf x_j = F(\boldsymbol\eta_j)`, so the SPH sum gives the physical
    density :math:`\rho_p = \sum_j w_j W_h(\mathbf x_p - \mathbf x_j)` directly, on any mapping. For each marker
    :math:`p`,

    .. math::

        \frac{\textnormal d \mathbf v_p}{\textnormal d t} = \mathbf g -
        \sum_{j=1}^N w_j \big( \mathcal U'(\rho_p) + \mathcal U'(\rho_j) \big)\, \nabla W_h(\mathbf x_p - \mathbf x_j)\,,

    with :math:`\mathcal U' = \partial_\rho \mathcal U`, i.e. :math:`\kappa/\rho` (isothermal) and
    :math:`\kappa \rho^{\gamma - 2}` (polytropic). The force is antisymmetric in :math:`p, j`, so momentum is
    conserved, and the coefficients :math:`w_j \mathcal U'(\rho_j)` are computed once per time step (kernel
    :func:`~struphy.pic.pushing.kernels.sph_pressure_coeffs.sph_pressure_coeffs_kernels.sph_pressure_coeffs`) and
    stored in the marker array.

    **Time discretization:** forward Euler (``algo="forward_euler"``),
    :math:`\mathbf v_p^{n+1} = \mathbf v_p^n + \Delta t\, \frac{\textnormal d \mathbf v_p}{\textnormal d t}\big|^n`.
    Since the positions are fixed during this step, the update is exact.
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
        algo : {"forward_euler"}, default="forward_euler"
            Time stepping algorithm for velocity pushing.

        gravity : tuple, default=(0.0, 0.0, 0.0)
            Constant gravity vector added in the SPH pressure push.

        kappa : float, default=1.0
            Coefficient in the internal energy function.

        thermodynamics : {"isothermal", "polytropic"}, default="isothermal"
            Thermodynamic closure selecting the SPH pressure kernel.

        gamma : float, default=5/3
            Adiabatic index (gamma > 1) of the polytropic closure (not used for ``"isothermal"``).

        kernel_type, kernel_width : None
            Removed: the smoothing kernel is set per species with
            :class:`~struphy.particles.parameters.SPHParameters` (``set_markers(sph_params=...)``).
        """

        # specific literals
        OptsAlgo = Literal["forward_euler"]
        OptsThermo = Literal["isothermal", "polytropic"]
        # propagator options
        algo: OptsAlgo = "forward_euler"
        gravity: tuple = (0.0, 0.0, 0.0)
        kappa: float = 1.0
        thermodynamics: OptsThermo = "isothermal"
        gamma: float = 5.0 / 3.0
        kernel_type: None = None
        kernel_width: None = None

        def __post_init__(self):
            # checks
            check_removed_kernel_options(self)
            check_option(self.algo, self.OptsAlgo)
            check_option(self.thermodynamics, self.OptsThermo)
            assert self.gamma > 1.0, f"The adiabatic index gamma must be > 1, got {self.gamma}."

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
        check_removed_kernel_options(self.options)
        # init kernel for evaluating density etc. before each time step.
        init_kernel = sph_pressure_coeffs

        particles = self.variables.fluid.particles
        first_free_idx = particles.args_markers.first_free_idx
        kernel_args = particles.sph_kernel_args

        # init kernel
        self.add_init_kernel(
            KernelSetup(
                kernel=init_kernel,
                output_indices=(first_free_idx, first_free_idx + 1, first_free_idx + 2),
                args=kernel_args + (float(self.options.gamma),),
            ),
        )

        gravity = xp.array(self.options.gravity, dtype=float)
        args_kernel = kernel_args + (gravity, self.options.kappa)

        # pusher kernel
        if self.options.thermodynamics == "isothermal":
            kernel = push_v_sph_pressure
        elif self.options.thermodynamics == "polytropic":
            kernel = push_v_sph_pressure_ideal_gas
            args_kernel += (float(self.options.gamma),)

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
