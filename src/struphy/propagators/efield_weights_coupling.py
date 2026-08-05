"Particle and FEEC variables are updated."

import logging
from dataclasses import dataclass

import cunumpy as xp
from line_profiler import profile

from struphy.feec import preconditioner
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.kinetic_background.maxwellians import Maxwellian3D
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable, PICVariable
from struphy.pic.accumulation import accum_kernels
from struphy.pic.accumulation.particles_to_grid import Accumulator
from struphy.pic.pushing import pusher_kernels
from struphy.pic.pushing.pusher import Pusher
from struphy.propagators.base import Propagator
from cunumpy import Pyccelkernel
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class EfieldWeightsCoupling(Propagator):
    r"""Solves the following substep

    .. math::

        \begin{align}
            & \frac{\partial \mathbf{E}}{\partial t} = - \frac{\alpha^2}{\varepsilon} \int \mathbf{v} f_1 \, \text{d} \mathbf{v} \, ,
            \\[2mm]
            & \frac{\partial f_1}{\partial t} = \frac{1}{v_{\text{th}}^2 \varepsilon} \, \mathbf{E} \cdot \mathbf{v} f_0 \, ,
        \end{align}

    which after discretization and in curvilinear coordinates reads

    .. math::

        \frac{\text{d}}{\text{d} t} w_p &= \frac{f_{0,p}}{s_{0, p}} \frac{1}{v_{\text{th}}^2 \varepsilon} \left[ DF^{-T} (\mathbb{\Lambda}^1)^T \mathbf{e} \right] \cdot \mathbf{v}_p \, ,
        \\[2mm]
        \frac{\text{d}}{\text{d} t} \mathbb{M}_1 \mathbf{e} &= - \frac{\alpha^2}{\varepsilon} \frac 1N \sum_p w_p \mathbb{\Lambda}^1 \cdot \left( DF^{-1} \mathbf{v}_p \right) \,.

    This is solved using the Crank-Nicolson method

    .. math::

        \begin{bmatrix}
            \mathbb{M}_1 \left( \mathbf{e}^{n+1} - \mathbf{e}^n \right)
            \mathbf{W}^{n+1} - \mathbf{W}^n
        \end{bmatrix}
        =
        \frac{\Delta t}{2}
        \begin{bmatrix}
            0 & - \mathbb{E} \\
            \mathbb{W} & 0
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf{e}^{n+1} + \mathbf{e}^n \\
            \mathbf{V}^{n+1} + \mathbf{V}^n
        \end{bmatrix} \, ,

    where

    .. math::

        \mathbb{E} &= \frac{\alpha^2}{\varepsilon} \frac 1N \mathbb{\Lambda}^1 \cdot \left( DF^{-1} \mathbf{v}_p \right)  \, ,
        \\[2mm]
        \mathbb{W} &= \frac{f_{0,p}}{s_{0,p}} \frac{1}{v_\text{th}^2 \varepsilon} \left( DF^{-1} \mathbf{v}_p \right) \cdot \left(\mathbb{\Lambda}^1\right)^T  \, ,

    based on the :class:`~struphy.linear_algebra.schur_solver.SchurSolver`.

    The accumulation matrix and vector assembled in :class:`~struphy.pic.accumulation.particles_to_grid.Accumulator` are

    .. math::

        BC = \mathbb{E} \mathbb{W} \, , \qquad By_n = \mathbb{E} \mathbf{W} \,.

    """

    class Variables:
        """Container for variables advanced by :class:`EfieldWeightsCoupling`.

        Attributes
        ----------
        e : FEECVariable
            Electric-field variable in ``"Hcurl"`` space.
        ions : PICVariable
            Particle variable in ``"Particles6D"`` or
            ``"DeltaFParticles6D"`` space.
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
            assert new.space in ("Particles6D", "DeltaFParticles6D")
            self._ions = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`EfieldWeightsCoupling`.

        Parameters
        ----------
        alpha : float, default=1.0
            Coupling scaling factor used in the electric-field equation.

        kappa : float, default=1.0
            Coupling scaling factor used in the particle-weight update.

        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Symmetric iterative solver used by :class:`SchurSolver`.

        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner applied to the electric-field mass matrix block.

        solver_params : SolverParameters, default=None
            Iterative-solver controls. If ``None``, defaults to
            ``SolverParameters()``.
        """

        alpha: float = 1.0
        kappa: float = 1.0
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)

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
        self._alpha = self.options.alpha
        self._kappa = self.options.kappa

        backgrounds = self.variables.ions.backgrounds
        # use single Maxwellian
        if isinstance(backgrounds, list):
            self._f0 = backgrounds[0]
        else:
            self._f0 = backgrounds
        assert isinstance(self._f0, Maxwellian3D), "The background distribution function must be a uniform Maxwellian!"
        self._vth = self._f0.params["vth1"][0]

        self._info = self.options.solver_params.info

        # Initialize Accumulator object
        e = self.variables.e.spline.vector
        particles = self.variables.ions.particles

        self._accum = Accumulator(
            particles,
            "Hcurl",
            Pyccelkernel(accum_kernels.linear_vlasov_ampere),
            self.mass_ops,
            self.domain.args_domain,
            add_vector=True,
            symmetry="symm",
        )

        # Create buffers to store temporarily e and its sum with old e
        self._e_tmp = e.space.zeros()
        self._e_scale = e.space.zeros()
        self._e_sum = e.space.zeros()

        # marker storage
        self._f0_values = xp.zeros(particles.markers.shape[0], dtype=float)
        self._old_weights = xp.empty(particles.markers.shape[0], dtype=float)

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
        _BC = self._alpha**2 * self._kappa**2 * self._accum.operators[0] / (4 * self._vth**2)

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
            self._f0_values,
            self._kappa,
            self._vth,
        )

        self._pusher = Pusher(
            particles,
            Pyccelkernel(pusher_kernels.push_weights_with_efield_lin_va),
            args_kernel,
            self.domain.args_domain,
            alpha_in_kernel=1.0,
        )

    def __call__(self, dt):
        en = self.variables.e.spline.vector
        particles = self.variables.ions.particles

        # evaluate f0 and accumulate
        self._f0_values[:] = self._f0(
            particles.markers[:, 0],
            particles.markers[:, 1],
            particles.markers[:, 2],
            particles.markers[:, 3],
            particles.markers[:, 4],
            particles.markers[:, 5],
        )

        self._accum(self._f0_values)

        # Update Schur solver
        self._schur_solver.BC = self._accum.operators[0]
        self._schur_solver.BC *= (-1) * self._alpha**2 * self._kappa**2 / (4 * self._vth**2)

        # Vector for schur solver
        self._e_scale *= 0.0
        self._e_scale += self._accum.vectors[0]
        self._e_scale *= self._alpha**2 * self._kappa / 2.0

        # new e-field (no tmps created here)
        self._e_tmp, info = self._schur_solver(
            xn=en,
            Byn=self._e_scale,
            dt=dt,
            out=self._e_tmp,
        )

        # Store old weights
        self._old_weights[~particles.holes] = particles.markers_wo_holes[:, 6]

        # Compute (e^{n+1} + e^n) (no tmps created here)
        self._e_sum *= 0.0
        self._e_sum += en
        self._e_sum += self._e_tmp

        # Update weights
        self._pusher(dt)

        # write new coeffs into self.variables
        max_de = self.update_feec_variables(e=self._e_tmp)

        # Print out max differences for weights and e-field
        if self._info:
            logger.info(f"Status          for StepEfieldWeights: {info['success']}")
            logger.info(f"Iterations      for StepEfieldWeights: {info['niter']}")
            logger.info(f"Maxdiff    e1   for StepEfieldWeights: {max_de}")
            max_diff = xp.max(
                xp.abs(
                    self._old_weights[~particles.holes] - particles.markers[~particles.holes, 6],
                ),
            )
            logger.info(f"Maxdiff weights for StepEfieldWeights: {max_diff}")
            logger.info("")
