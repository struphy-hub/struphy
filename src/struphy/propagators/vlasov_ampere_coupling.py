"Particle and FEEC variables are updated."

import logging
from dataclasses import dataclass

import cunumpy as xp
from line_profiler import profile

from struphy.feec import preconditioner
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable, PICVariable
from struphy.pic.accumulation import accum_kernels
from struphy.pic.accumulation.particles_to_grid import Accumulator
from struphy.pic.pushing import pusher_kernels
from struphy.pic.pushing.pusher import Pusher
from struphy.propagators.base import Propagator
from struphy.utils.pyccel import Pyccelkernel
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class VlasovAmpereCoupling(Propagator):
    r"""PIC-FEEC discretization of the following equations:
    find :math:`\mathbf{E} \in H(\textnormal{curl})` and :math:`f` such that

    .. math::

        -\int_{\Omega} \frac{\partial \mathbf{E}}{\partial t} \cdot \mathbf{F}\,\textrm d \mathbf x &= \frac{\alpha^2}{\varepsilon} \int_{\Omega} \int_{\mathbb{R}^3} f \mathbf{v} \cdot \mathbf{F} \, \text{d}^3 \mathbf{v} \,\textrm d \mathbf x \qquad \forall \, \mathbf{F} \in H(\textnormal{curl})
        \\[2mm]
        \frac{\partial f}{\partial t} + \frac{1}{\varepsilon}\, \mathbf{E} \cdot \frac{\partial f}{\partial \mathbf{v}} &= 0 .

    Time discretization: Crank-Nicolson (implicit mid-point).

    System size reduction via :class:`~struphy.linear_algebra.schur_solver.SchurSolver`.
    """

    class Variables:
        """Container for variables advanced by :class:`VlasovAmpereCoupling`.

        Attributes
        ----------
        e : FEECVariable
            Electric-field variable in ``"Hcurl"`` space.
        ions : PICVariable
            Particle variable in ``"Particles6D"`` space.
        """

        def __init__(self):
            self._e: FEECVariable = None
            self._ions: PICVariable = None

        @property
        def e(self) -> FEECVariable:
            return self._e

        @e.setter
        def e(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hcurl"
            self._e = new

        @property
        def ions(self) -> PICVariable:
            return self._ions

        @ions.setter
        def ions(self, new):
            assert isinstance(new, PICVariable)
            assert new.space == "Particles6D"
            self._ions = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`VlasovAmpereCoupling`.

        Parameters
        ----------
        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Symmetric iterative solver used by :class:`SchurSolver`.
        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner for electric-field mass matrix block.
        solver_params : SolverParameters, default=None
            Solver controls.
        """

        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

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
        # scaling factors
        alpha = self.variables.ions.species.equation_params.alpha
        epsilon = self.variables.ions.species.equation_params.epsilon

        self._c1 = alpha**2 / epsilon
        self._c2 = 1.0 / epsilon

        self._info = self.options.solver_params.info

        # get accumulation kernel
        accum_kernel = Pyccelkernel(accum_kernels.vlasov_maxwell)

        # Initialize Accumulator object
        particles = self.variables.ions.particles

        self._accum = Accumulator(
            particles,
            "Hcurl",
            accum_kernel,
            self.mass_ops,
            self.domain.args_domain,
            add_vector=True,
            symmetry="symm",
        )

        # Create buffers to store temporarily e and its sum with old e
        self._e_tmp = self.derham.V1.zeros()
        self._e_scale = self.derham.V1.zeros()
        self._e_sum = self.derham.V1.zeros()

        # ================================
        # ========= Schur Solver =========
        # ================================

        # Preconditioner
        if self.options.precond is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
            pc = pc_class(self.mass_ops.M1)

        # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = self.mass_ops.M1
        _BC = -self._accum.operators[0]

        # Instantiate Schur solver
        self._schur_solver = SchurSolver(
            _A,
            _BC,
            self.options.solver,
            precond=pc,
            solver_params=self.options.solver_params,
        )

        # Instantiate particle pusher
        args_kernel = (
            self.derham.args_derham,
            self._e_sum.blocks[0]._data,
            self._e_sum.blocks[1]._data,
            self._e_sum.blocks[2]._data,
            self._c2,
        )

        self._pusher = Pusher(
            particles,
            Pyccelkernel(pusher_kernels.push_v_with_efield),
            args_kernel,
            self.domain.args_domain,
            alpha_in_kernel=1.0,
        )

    @profile
    def __call__(self, dt):
        # accumulate
        self._accum()

        # Update Schur solver
        self._schur_solver.BC = self._accum.operators[0]
        self._schur_solver.BC *= -self._c1 * self._c2 / 4.0

        # Vector for Schur solver
        self._e_scale *= 0.0
        self._e_scale += self._accum.vectors[0]
        self._e_scale *= self._c1 / 2.0

        # new e coeffs
        self._e_tmp, info = self._schur_solver(
            self.variables.e.spline.vector,
            self._e_scale,
            dt,
            out=self._e_tmp,
        )

        # mid-point e-field (no tmps created here)
        self._e_sum *= 0.0
        self._e_sum += self.variables.e.spline.vector
        self._e_sum += self._e_tmp
        self._e_sum *= 0.5

        # Update velocities
        self._pusher(dt)

        # update_weights
        if self.variables.ions.species.weights_params.control_variate:
            self.variables.ions.particles.update_weights()

        # write new coeffs into self.variables
        (max_de,) = self.update_feec_variables(e=self._e_tmp)

        # Print out max differences for weights and e-field
        if self._info:
            logger.info(f"Status      for VlasovMaxwell: {info['success']}")
            logger.info(f"Iterations  for VlasovMaxwell: {info['niter']}")
            logger.info(f"Maxdiff e1  for VlasovMaxwell: {max_de}")
            particles = self.variables.ions.particles
            buffer_idx = particles.bufferindex
            max_diff = xp.max(
                xp.abs(
                    xp.sqrt(
                        particles.markers_wo_holes[:, 3] ** 2
                        + particles.markers_wo_holes[:, 4] ** 2
                        + particles.markers_wo_holes[:, 5] ** 2,
                    )
                    - xp.sqrt(
                        particles.markers_wo_holes[:, buffer_idx + 3] ** 2
                        + particles.markers_wo_holes[:, buffer_idx + 4] ** 2
                        + particles.markers_wo_holes[:, buffer_idx + 5] ** 2,
                    ),
                ),
            )
            logger.info(f"Maxdiff |v| for VlasovMaxwell: {max_diff}")
            logger.info("")
