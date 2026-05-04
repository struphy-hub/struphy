import logging
from dataclasses import dataclass
from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.solvers import inverse
from line_profiler import profile
from struphy.feec import preconditioner
from struphy.io.options import LiteralOptions
from struphy.linear_algebra.solver import NonlinearSolverParameters, SolverParameters
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable, Variable
from struphy.pic.accumulation import accum_kernels, accum_kernels_gc
from struphy.pic.accumulation.filter import FilterParameters
from struphy.pic.accumulation.particles_to_grid import Accumulator, AccumulatorVector
from struphy.propagators.base import Propagator
from struphy.utils.pyccel import Pyccelkernel
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class CurrentCoupling6DDensity(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\tilde{\mathbf{U}}  \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}` such that

    .. math::

        &\int_\Omega \rho_0 \frac{\partial \tilde{\mathbf{U}}}{\partial t} \cdot \mathbf V \,\textrm d \mathbf x = \frac{A_\textnormal{h}}{A_\textnormal{b}} \frac{1}{\varepsilon} \int_\Omega n_\textnormal{h}\tilde{\mathbf{U}} \times(\mathbf{B}_0+\tilde{\mathbf{B}}) \cdot \mathbf V \,\textrm d \mathbf x
        \qquad \forall \, \mathbf V \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}\,,
        \\[2mm]
        &n_\textnormal{h}=\int_{\mathbb{R}^3}f_\textnormal{h}\,\textnormal{d}^3 \mathbf v\,.

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point).
    """

    class Variables:
        """Container for variables advanced by :class:`CurrentCoupling6DDensity`.

        Attributes
        ----------
        u : FEECVariable
            Fluid-like FEEC variable in one of ``"Hcurl"``, ``"Hdiv"``, or
            ``"H1vec"``.
        """
        def __init__(self):
            self._u: FEECVariable = None

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space in ("Hcurl", "Hdiv", "H1vec")
            self._u = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
        """Configuration options for :class:`CurrentCoupling6DDensity`.

        Parameters
        ----------
        energetic_ions : PICVariable, default=None
            Energetic-ion particle variable providing marker data in
            ``"Particles6D"`` space.

        b_tilde : FEECVariable, default=None
            Perturbed magnetic field variable added to the equilibrium field.

        u_space : LiteralOptions.OptsVecSpace, default="Hdiv"
            FEEC space used for the unknown ``u`` variable.

        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Symmetric iterative solver used for the linear system.

        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner applied to the FEEC mass matrix.

        solver_params : SolverParameters, default=None
            Iterative-solver controls. If ``None``, defaults to
            ``SolverParameters()``.

        filter_params : FilterParameters, default=None
            Particle-to-grid filtering parameters used by the accumulator.

        boundary_cut : tuple, default=(0.0, 0.0, 0.0)
            Boundary clipping parameters forwarded to the accumulation kernel.
        """
        # propagator options
        energetic_ions: PICVariable = None
        b_tilde: FEECVariable = None
        u_space: LiteralOptions.OptsVecSpace = "Hdiv"
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        filter_params: FilterParameters = None
        boundary_cut: tuple = (0.0, 0.0, 0.0)

        def __post_init__(self):
            # checks
            check_option(self.u_space, LiteralOptions.OptsVecSpace)
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)
            assert self.energetic_ions.space == "Particles6D"
            assert self.b_tilde.space == "Hdiv"

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

    @profile
    def allocate(self, verbose: bool = False):
        self._space_key_int = int(self.derham.space_to_form[self.options.u_space])

        particles = self.options.energetic_ions.particles
        u = self.variables.u.spline.vector
        self._b_eq = self.projected_equil.b2
        self._b_tilde = self.options.b_tilde.spline.vector

        # if self._particles.control_variate:

        #     # control variate method is only valid with Maxwellian distributions
        #     assert isinstance(self._particles.f0, Maxwellian)
        #     assert u_space == 'Hdiv'

        #     # evaluate and save nh0/|det(DF)| (push-forward) at quadrature points for control variate
        #     quad_pts = [quad_grid[nquad].points.flatten()
        #                 for quad_grid, nquad in zip(self.derham.V0fem._quad_grids, self.derham.V0fem.nquads)]

        #     self._nh0_at_quad = self.domain.push(
        #         self._particles.f0.n, *quad_pts, kind='3', squeeze_out=False)

        #     # memory allocation of magnetic field at quadrature points
        #     self._b_quad1 = xp.zeros_like(self._nh0_at_quad)
        #     self._b_quad2 = xp.zeros_like(self._nh0_at_quad)
        #     self._b_quad3 = xp.zeros_like(self._nh0_at_quad)

        #     # memory allocation for self._b_quad x self._nh0_at_quad * self._coupling_const
        #     self._mat12 = xp.zeros_like(self._nh0_at_quad)
        #     self._mat13 = xp.zeros_like(self._nh0_at_quad)
        #     self._mat23 = xp.zeros_like(self._nh0_at_quad)

        #     self._mat21 = xp.zeros_like(self._nh0_at_quad)
        #     self._mat31 = xp.zeros_like(self._nh0_at_quad)
        #     self._mat32 = xp.zeros_like(self._nh0_at_quad)

        self._type = self.options.solver
        self._tol = self.options.solver_params.tol
        self._maxiter = self.options.solver_params.maxiter
        self._info = self.options.solver_params.info
        self._verbose = self.options.solver_params.verbose
        self._recycle = self.options.solver_params.recycle

        Ah = self.options.energetic_ions.species.mass_number
        Ab = self.variables.u.species.mass_number
        epsilon = self.options.energetic_ions.species.equation_params.epsilon

        self._coupling_const = Ah / Ab / epsilon

        self._boundary_cut_e1 = self.options.boundary_cut[0]

        # load accumulator
        self._accumulator = Accumulator(
            particles,
            self.options.u_space,
            Pyccelkernel(accum_kernels.cc_lin_mhd_6d_1),
            self.mass_ops,
            self.domain.args_domain,
            add_vector=False,
            symmetry="asym",
            filter_params=self.options.filter_params,
        )

        # transposed extraction operator PolarVector --> BlockVector (identity map in case of no polar splines)
        self._E2T = self.derham.extraction_ops["2"].transpose()

        # mass matrix in system (M - dt/2 * A)*u^(n + 1) = (M + dt/2 * A)*u^n
        u_id = self.derham.space_to_form[self.options.u_space]
        self._M = getattr(self.mass_ops, "M" + u_id + "n")

        # preconditioner
        if self.options.precond is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
            pc = pc_class(self._M)

        # linear solver
        self._solver = inverse(
            self._M,
            self.options.solver,
            pc=pc,
            x0=self.variables.u.spline.vector,
            tol=self._tol,
            maxiter=self._maxiter,
            verbose=self._verbose,
            recycle=self._recycle,
        )

        # temporary vectors to avoid memory allocation
        self._b_full1 = self._b_eq.space.zeros()
        self._b_full2 = self._E2T.codomain.zeros()

        self._rhs_v = u.space.zeros()
        self._u_new = u.space.zeros()

    def __call__(self, dt):
        # pointer to old coefficients
        un = self.variables.u.spline.vector

        # sum up total magnetic field b_full1 = b_eq + b_tilde (in-place)
        self._b_eq.copy(out=self._b_full1)

        if self._b_tilde is not None:
            self._b_full1 += self._b_tilde

        # extract coefficients to tensor product space (in-place)
        self._E2T.dot(self._b_full1, out=self._b_full2)

        # update ghost regions because of non-local access in accumulation kernel!
        self._b_full2.update_ghost_regions()

        # perform accumulation (either with or without control variate)
        # if self._particles.control_variate:

        #     # evaluate magnetic field at quadrature points (in-place)
        #     WeightedMassOperator.eval_quad(self.derham.V2fem, self._b_full2,
        #                                    out=[self._b_quad1, self._b_quad2, self._b_quad3])

        #     self._mat12[:, :, :] = self._coupling_const * \
        #         self._b_quad3 * self._nh0_at_quad
        #     self._mat13[:, :, :] = -self._coupling_const * \
        #         self._b_quad2 * self._nh0_at_quad
        #     self._mat23[:, :, :] = self._coupling_const * \
        #         self._b_quad1 * self._nh0_at_quad

        #     self._mat21[:, :, :] = -self._mat12
        #     self._mat31[:, :, :] = -self._mat13
        #     self._mat32[:, :, :] = -self._mat23

        #     self._accumulator(self._b_full2[0]._data,
        #                       self._b_full2[1]._data,
        #                       self._b_full2[2]._data,
        #                       self._space_key_int,
        #                       self._coupling_const,
        #                       control_mat=[[None, self._mat12, self._mat13],
        #                                    [self._mat21, None,
        #                                     self._mat23],
        #                                    [self._mat31, self._mat32, None]])
        # else:
        self._accumulator(
            self._b_full2[0]._data,
            self._b_full2[1]._data,
            self._b_full2[2]._data,
            self._space_key_int,
            self._coupling_const,
            self._boundary_cut_e1,
        )

        # define system (M - dt/2 * A)*u^(n + 1) = (M + dt/2 * A)*u^n
        lhs = self._M - dt / 2 * self._accumulator.operators[0]
        rhs = self._M + dt / 2 * self._accumulator.operators[0]

        # solve linear system for updated u coefficients (in-place)
        rhs = rhs.dot(un, out=self._rhs_v)
        self._solver.linop = lhs

        un1 = self._solver.solve(rhs, out=self._u_new)
        info = self._solver._info

        # write new coeffs into Propagator.variables
        max_du = self.update_feec_variables(u=un1)

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            logger.info(f"Status     for CurrentCoupling6DDensity: {info['success']}")
            logger.info(f"Iterations for CurrentCoupling6DDensity: {info['niter']}")
            logger.info(f"Maxdiff up for CurrentCoupling6DDensity: {max_du}")
            logger.info("")
