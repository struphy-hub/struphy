import logging
from dataclasses import dataclass
from typing import Literal

from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.block import BlockVector
from feectools.linalg.solvers import inverse
from line_profiler import profile

from struphy.feec import preconditioner
from struphy.feec.basis_projection_ops import (
    BasisProjectionOperator,
    BasisProjectionOperatorLocal,
)
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable, PICVariable
from struphy.ode.solvers import ODEsolverFEEC
from struphy.ode.utils import ButcherTableau
from struphy.pic.accumulation import accum_kernels_gc
from struphy.pic.accumulation.filter import FilterParameters
from struphy.pic.accumulation.particles_to_grid import AccumulatorVector
from struphy.propagators.base import Propagator
from struphy.utils.pyccel import Pyccelkernel
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class ShearAlfvenCurrentCoupling5D(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations: 
    find :math:`\mathbf U \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}` and  :math:`\mathbf B \in H(\textnormal{div})` such that

    .. math::

        \left\{ 
            \begin{aligned} 
                \int \rho_0 &\frac{\partial \tilde{\mathbf U}}{\partial t} \cdot \mathbf V \, \textnormal{d} \mathbf{x} = \int \left(\tilde{\mathbf B} - \frac{A_\textnormal{h}}{A_b} \iint f^\text{vol} \mu \mathbf{b}_0\textnormal{d} \mu \textnormal{d} v_\parallel \right) \cdot \nabla \times (\tilde{\mathbf B} \times \mathbf V) \, \textnormal{d} \mathbf{x} \quad \forall \, \mathbf V \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}\,, \,,
                \\
                &\frac{\partial \tilde{\mathbf B}}{\partial t} = - \nabla \times (\tilde{\mathbf B} \times \tilde{\mathbf U}) \,.
            \end{aligned}
        \right.

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point). System size reduction via :class:`~struphy.linear_algebra.schur_solver.SchurSolver`:

    .. math::

        \begin{bmatrix} 
            \mathbf u^{n+1} - \mathbf u^n \\ \mathbf b^{n+1} - \mathbf b^n
        \end{bmatrix} 
        = \frac{\Delta t}{2} \,.
        \begin{bmatrix} 
            0 & (\mathbb M^{2,n})^{-1} \mathcal {T^2}^\top \mathbb C^\top \\ - \mathbb C \mathcal {T^2} (\mathbb M^{2,n})^{-1} & 0 
        \end{bmatrix} 
        \begin{bmatrix}
            {\mathbb M^{2,n}}(\mathbf u^{n+1} + \mathbf u^n) \\ \mathbb M_2(\mathbf b^{n+1} + \mathbf b^n) + \sum_k^{N_p} \omega_k \mu_k \hat{\mathbf b}¹_0 (\boldsymbol \eta_k) \cdot \left(\frac{1}{\sqrt{g(\boldsymbol \eta_k)}} \vec \Lambda² (\boldsymbol \eta_k) \right)
        \end{bmatrix} \,,

    where 
    :math:`\mathcal{T}^2 = \hat \Pi \left[\frac{\tilde{\mathbf B}^2}{\sqrt{g} \times \vec \Lambda^2\right]`  and
    :math:`\mathbb M^{2,n}` is a :class:`~struphy.feec.mass.WeightedMassOperators` being weighted with :math:`\rho_\text{eq}`, the MHD equilibirum density. 
    """

    class Variables:
        """Container for variables advanced by :class:`ShearAlfvenCurrentCoupling5D`.

        Attributes
        ----------
        u : FEECVariable
            Velocity variable in one of ``"Hcurl"``, ``"Hdiv"``, or
            ``"H1vec"``.
        b : FEECVariable
            Magnetic-field variable in ``"Hdiv"`` space.
        """

        def __init__(self):
            self._u: FEECVariable = None
            self._b: FEECVariable = None

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space in ("Hcurl", "Hdiv", "H1vec")
            self._u = new

        @property
        def b(self) -> FEECVariable:
            return self._b

        @b.setter
        def b(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hdiv"
            self._b = new

    def __init__(self, energetic_ions: PICVariable = None):
        """
        Parameters
        ----------
        energetic_ions : PICVariable, default=None
            Energetic-ion particle distribution (``"Particles5D"`` space) providing
            the current source for the shear Alfvén wave coupling.
        """
        self.variables = self.Variables()
        self.energetic_ions = energetic_ions

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`ShearAlfvenCurrentCoupling5D`.

        Parameters
        ----------
        ep_scale : float, default=1.0
            Scaling factor for particle accumulation terms.
        u_space : LiteralOptions.OptsVecSpace, default="Hdiv"
            FEEC space used for velocity variable ``u``.
        algo : {"implicit", "explicit"}, default="implicit"
            Time stepping algorithm.
        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Symmetric iterative solver.
        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixDiagonalPreconditioner"
            Preconditioner for mass-matrix block.
        solver_params : SolverParameters, default=None
            Solver controls.
        filter_params : FilterParameters, default=None
            Particle-to-grid filtering parameters.
        butcher : ButcherTableau, default=None
            Explicit Runge-Kutta tableau for ``algo="explicit"``.
        nonlinear : bool, default=True
            If ``True``, include nonlinear TB operator updates.
        """

        # specific literals
        OptsAlgo = Literal["implicit", "explicit"]
        # propagator options
        ep_scale: float = 1.0
        u_space: LiteralOptions.OptsVecSpace = "Hdiv"
        algo: OptsAlgo = "implicit"
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixDiagonalPreconditioner"
        solver_params: SolverParameters = None
        filter_params: FilterParameters = None
        butcher: ButcherTableau = None
        nonlinear: bool = True

        def __post_init__(self):
            # checks
            check_option(self.u_space, LiteralOptions.OptsVecSpace)
            check_option(self.algo, self.OptsAlgo)
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)
            assert isinstance(self.ep_scale, float)
            assert isinstance(self.nonlinear, bool)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

            if self.filter_params is None:
                self.filter_params = FilterParameters()

            if self.algo == "explicit" and self.butcher is None:
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
        logger.info(f"\nNew options for propagator '{self.__class__.__name__}':\n{self._options}")

    @profile
    def allocate(self):
        self._u_form = self.derham.space_to_form[self.options.u_space]

        # call operatros
        id_M = "M" + self._u_form + "n"
        id_T = "T" + self._u_form

        _A = getattr(self.mass_ops, id_M)
        _T = getattr(self.basis_ops, id_T)
        M2 = self.mass_ops.M2
        curl = self.derham.curl
        PB = getattr(self.basis_ops, "PB")

        # define Accumulator and arguments
        self._ACC = AccumulatorVector(
            self.energetic_ions.particles,
            "H1",
            Pyccelkernel(accum_kernels_gc.gc_mag_density_0form),
            self.mass_ops,
            self.domain.args_domain,
            filter_params=self.options.filter_params,
        )

        # Preconditioner
        if self.options.precond is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
            pc = pc_class(getattr(self.mass_ops, id_M))

        if self.options.nonlinear:
            # initialize operator TB
            self._initialize_projection_operator_TB()

            _T = _T + self._TB
            _TT = _T.T + self._TBT

        else:
            _TT = _T.T

        if self.options.algo == "implicit":
            self._info = self.options.solver_params.info

            # define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
            self._B = -1 / 2 * _TT @ curl.T @ M2
            self._B2 = -1 / 2 * _TT @ curl.T @ PB.T

            self._C = 1 / 2 * curl @ _T

            # Instantiate Schur solver (constant in this case)
            _BC = self._B @ self._C

            self._schur_solver = SchurSolver(
                _A,
                _BC,
                self.options.solver,
                precond=pc,
                solver_params=self.options.solver_params,
            )

            # allocate dummy vectors to avoid temporary array allocations
            self._u_tmp1 = self.variables.u.spline.vector.space.zeros()
            self._u_tmp2 = self.variables.u.spline.vector.space.zeros()
            self._b_tmp1 = self.variables.b.spline.vector.space.zeros()

            self._byn = self._B.codomain.zeros()
            self._tmp_acc = self._B2.codomain.zeros()

        else:
            self._info = False

            # define vector field
            A_inv = inverse(
                _A,
                self.options.solver,
                pc=pc,
                tol=self.options.solver_params.tol,
                maxiter=self.options.solver_params.maxiter,
                verbose=self.options.solver_params.verbose,
            )
            _f1 = A_inv @ _TT @ curl.T @ M2
            _f1_acc = A_inv @ _TT @ curl.T @ PB.T
            _f2 = curl @ _T

            # allocate output of vector field
            out_acc = self.variables.u.spline.vector.space.zeros()
            out1 = self.variables.u.spline.vector.space.zeros()
            out2 = self.variables.b.spline.vector.space.zeros()

            def f1(t, y1, y2, out: BlockVector = out1):
                _f1.dot(y2, out=out)
                _f1_acc.dot(self._ACC.vectors[0], out=out_acc)
                out += out_acc
                out.update_ghost_regions()
                return out

            def f2(t, y1, y2, out: BlockVector = out2):
                _f2.dot(y1, out=out)
                out *= -1.0
                out.update_ghost_regions()
                return out

            vector_field = {self.variables.u.spline.vector: f1, self.variables.b.spline.vector: f2}
            self._ode_solver = ODEsolverFEEC(vector_field, butcher=self.options.butcher)

    def __call__(self, dt):
        # update time-dependent operator TB
        if self.options.nonlinear:
            self._update_weights_TB()

        # current FE coeffs
        un = self.variables.u.spline.vector
        bn = self.variables.b.spline.vector

        # accumulate
        self._ACC(self.options.ep_scale)

        if self.options.algo == "implicit":
            # solve for new u coeffs (no tmps created here)
            byn = self._B.dot(bn, out=self._byn)
            b2acc = self._B2.dot(self._ACC.vectors[0], out=self._tmp_acc)
            byn += b2acc

            un1, info = self._schur_solver(un, byn, dt, out=self._u_tmp1)

            # new b coeffs (no tmps created here)
            _u = un.copy(out=self._u_tmp2)
            _u += un1
            bn1 = self._C.dot(_u, out=self._b_tmp1)
            bn1 *= -dt
            bn1 += bn

            diffs = self.update_feec_variables(u=un1, b=bn1)

        else:
            self._ode_solver(0.0, dt)

        # update_weights
        if self.energetic_ions.species.weights_params.control_variate:
            self.energetic_ions.particles.update_weights()

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            if self.options.algo == "implicit":
                logger.info(f"Status     for ShearAlfvenCurrentCoupling5D: {info['success']}")
                logger.info(f"Iterations for ShearAlfvenCurrentCoupling5D: {info['niter']}")
                logger.info(f"Maxdiff up for ShearAlfvenCurrentCoupling5D: {diffs['u']}")
                logger.info(f"Maxdiff b2 for ShearAlfvenCurrentCoupling5D: {diffs['b']}")
                logger.info("")

    def _initialize_projection_operator_TB(self):
        r"""Initialize BasisProjectionOperator TB with the time-varying weight.

        .. math::

            \mathcal{T}^B_{(\mu,ijk),(\nu,mno)} := \hat \Pi¹_{(\mu,ijk)} \left[ \epsilon_{\mu \alpha \nu} \frac{\tilde{B}²_\alpha}{\sqrt{g}} \Lambda²_{\nu,mno} \right] \,.

        """

        # Call the projector and the space
        P1 = self.derham.P1
        Vh = self.derham.fem_spaces[self._u_form]

        # Femfield for the field evaluation
        self._bf = self.derham.create_spline_function("bf", "Hdiv")

        # Initialize BasisProjectionOperator
        if self.derham._with_local_projectors:
            self._TB = BasisProjectionOperatorLocal(
                P1,
                Vh,
                [
                    [None, None, None],
                    [None, None, None],
                    [None, None, None],
                ],
                transposed=False,
                use_cache=True,
                polar_shift=True,
                V_extraction_op=self.derham.extraction_ops[self._u_form],
                V_boundary_op=self.derham.boundary_ops[self._u_form],
                P_boundary_op=self.derham.boundary_ops["1"],
            )
            self._TBT = BasisProjectionOperatorLocal(
                P1,
                Vh,
                [
                    [None, None, None],
                    [None, None, None],
                    [None, None, None],
                ],
                transposed=True,
                use_cache=True,
                polar_shift=True,
                V_extraction_op=self.derham.extraction_ops[self._u_form],
                V_boundary_op=self.derham.boundary_ops[self._u_form],
                P_boundary_op=self.derham.boundary_ops["1"],
            )
        else:
            self._TB = BasisProjectionOperator(
                P1,
                Vh,
                [
                    [None, None, None],
                    [None, None, None],
                    [None, None, None],
                ],
                transposed=False,
                use_cache=True,
                polar_shift=True,
                V_extraction_op=self.derham.extraction_ops[self._u_form],
                V_boundary_op=self.derham.boundary_ops[self._u_form],
                P_boundary_op=self.derham.boundary_ops["1"],
            )
            self._TBT = BasisProjectionOperator(
                P1,
                Vh,
                [
                    [None, None, None],
                    [None, None, None],
                    [None, None, None],
                ],
                transposed=True,
                use_cache=True,
                polar_shift=True,
                V_extraction_op=self.derham.extraction_ops[self._u_form],
                V_boundary_op=self.derham.boundary_ops[self._u_form],
                P_boundary_op=self.derham.boundary_ops["1"],
            )

    def _update_weights_TB(self):
        """Updats time-dependent weights of the BasisProjectionOperator TB"""

        # Update Femfield
        self.variables.b.spline.vector.copy(out=self._bf.vector)

        # define callable weights
        def bf1(x, y, z):
            return self._bf(x, y, z, local=True)[0]

        def bf2(x, y, z):
            return self._bf(x, y, z, local=True)[1]

        def bf3(x, y, z):
            return self._bf(x, y, z, local=True)[2]

        from struphy.feec.utilities import LocalRotationMatrix

        rot_B = LocalRotationMatrix(bf1, bf2, bf3)

        fun = []

        if self._u_form == "v":
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [
                        lambda e1, e2, e3, m=m, n=n: rot_B(e1, e2, e3)[:, :, :, m, n],
                    ]

        elif self._u_form == "1":
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [
                        lambda e1, e2, e3, m=m, n=n: (
                            rot_B(e1, e2, e3)
                            @ self.domain.metric_inv(
                                e1,
                                e2,
                                e3,
                                change_out_order=True,
                                squeeze_out=False,
                            )
                        )[:, :, :, m, n],
                    ]

        else:
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [
                        lambda e1, e2, e3, m=m, n=n: (
                            rot_B(e1, e2, e3)[:, :, :, m, n]
                            / abs(self.domain.jacobian_det(e1, e2, e3, squeeze_out=False))
                        ),
                    ]

        # update BasisProjectionOperator
        self._TB.update_weights(fun)
        self._TBT.update_weights(fun)
