"Particle and FEEC variables are updated."

import logging
from dataclasses import dataclass

from feectools.ddm.mpi import mpi as MPI
from line_profiler import profile

from struphy.feec import preconditioner
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable, PICVariable
from struphy.pic.accumulation import accum_kernels
from struphy.pic.accumulation.filter import FilterParameters
from struphy.pic.accumulation.particles_to_grid import Accumulator
from struphy.pic.pushing import pusher_kernels
from struphy.pic.pushing.pusher import Pusher
from struphy.propagators.base import Propagator
from struphy.utils.pyccel import Pyccelkernel
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class CurrentCoupling6DCurrent(Propagator):
    r"""
    :ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\tilde{\mathbf{U}}  \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}` and :math:`f` such that

    .. math::

        \int_\Omega \rho_0 &\frac{\partial \tilde{\mathbf{U}}}{\partial t} \cdot \mathbf V \,\textrm d \mathbf x = - \frac{A_\textnormal{h}}{A_\textnormal{b}} \frac{1}{\varepsilon}  \int_\Omega n_\textnormal{h}\mathbf{u}_\textnormal{h} \times(\mathbf{B}_0+\tilde{\mathbf{B}}) \cdot \mathbf V \,\textrm d \mathbf x
        \qquad \forall \, \mathbf V \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}\,,
        \\[2mm]
        &\frac{\partial f_\textnormal{h}}{\partial t} + \frac{1}{\varepsilon} \Big[(\mathbf{B}_0+\tilde{\mathbf{B}})\times\tilde{\mathbf{U}} \Big] \cdot \frac{\partial f_\textnormal{h}}{\partial \mathbf{v}} =0\,,
        \\[2mm]
        &n_\textnormal{h}\mathbf{u}_\textnormal{h}=\int_{\mathbb{R}^3}f_\textnormal{h}\mathbf{v}\,\textnormal{d}^3 \mathbf v\,.

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point). System size reduction via :class:`~struphy.linear_algebra.schur_solver.SchurSolver`.
    """

    class Variables:
        """Container for variables advanced by :class:`CurrentCoupling6DCurrent`.

        Attributes
        ----------
        ions : PICVariable
            Energetic-ion particle variable in ``"Particles6D"`` space.
        u : FEECVariable
            Fluid-like FEEC variable in one of ``"Hcurl"``, ``"Hdiv"``, or
            ``"H1vec"``.
        """

        def __init__(self):
            self._ions: PICVariable = None
            self._u: FEECVariable = None

        @property
        def ions(self) -> PICVariable:
            return self._ions

        @ions.setter
        def ions(self, new):
            assert isinstance(new, PICVariable)
            assert new.space in ("Particles6D")
            self._ions = new

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space in ("Hcurl", "Hdiv", "H1vec")
            self._u = new

    def __init__(self, b_tilde: FEECVariable = None):
        self.variables = self.Variables()
        self.b_tilde = b_tilde

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`CurrentCoupling6DCurrent`.

        Parameters
        ----------
        u_space : LiteralOptions.OptsVecSpace, default="Hdiv"
            FEEC space used for the unknown ``u`` variable.

        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Symmetric iterative solver used by :class:`SchurSolver`.

        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner applied to the FEEC mass matrix block.

        solver_params : SolverParameters, default=None
            Iterative-solver controls. If ``None``, defaults to
            ``SolverParameters()``.

        filter_params : FilterParameters, default=None
            Particle-to-grid filtering parameters used by the accumulator.

        boundary_cut : tuple, default=(0.0, 0.0, 0.0)
            Boundary clipping parameters forwarded to accumulation/pushing
            kernels.
        """

        # propagator options
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
        self._space_key_int = int(self.derham.space_to_form[self.options.u_space])

        particles = self.variables.ions.particles
        u = self.variables.u.spline.vector
        self._b_eq = self.projected_equil.b2

        self._info = self.options.solver_params.info

        if self.derham.comm is None:
            self._rank = 0
        else:
            self._rank = self.derham.comm.Get_rank()

        Ah = self.variables.ions.species.mass_number
        Ab = self.variables.u.species.mass_number
        epsilon = self.variables.ions.species.equation_params.epsilon

        self._coupling_mat = Ah / Ab / epsilon**2
        self._coupling_vec = Ah / Ab / epsilon
        self._scale_push = 1.0 / epsilon

        self._boundary_cut_e1 = self.options.boundary_cut[0]

        # load accumulator
        self._accumulator = Accumulator(
            particles,
            self.options.u_space,
            Pyccelkernel(accum_kernels.cc_lin_mhd_6d_2),
            self.mass_ops,
            self.domain.args_domain,
            add_vector=True,
            symmetry="symm",
            filter_params=self.options.filter_params,
        )

        # FEM spaces and basis extraction operators for u and b
        u_id = self.derham.space_to_form[self.options.u_space]
        self._EuT = self.derham.extraction_ops[u_id].transpose()
        self._EbT = self.derham.extraction_ops["2"].transpose()

        # temporary vectors to avoid memory allocation
        self._b_full1 = self._b_eq.space.zeros()
        self._b_full2 = self._EbT.codomain.zeros()

        self._u_new = u.space.zeros()

        self._u_avg1 = u.space.zeros()
        self._u_avg2 = self._EuT.codomain.zeros()

        # load particle pusher kernel
        if self.options.u_space == "Hcurl":
            kernel = Pyccelkernel(pusher_kernels.push_bxu_Hcurl)
        elif self.options.u_space == "Hdiv":
            kernel = Pyccelkernel(pusher_kernels.push_bxu_Hdiv)
        elif self.options.u_space == "H1vec":
            kernel = Pyccelkernel(pusher_kernels.push_bxu_H1vec)
        else:
            raise ValueError(
                f'{self.options.u_space =} not valid, choose from "Hcurl", "Hdiv" or "H1vec.',
            )

        # instantiate Pusher
        args_kernel = (
            self.derham.args_derham,
            self._b_full2[0]._data,
            self._b_full2[1]._data,
            self._b_full2[2]._data,
            self._u_avg2[0]._data,
            self._u_avg2[1]._data,
            self._u_avg2[2]._data,
            self._boundary_cut_e1,
        )

        self._pusher = Pusher(
            particles,
            kernel,
            args_kernel,
            self.domain.args_domain,
            alpha_in_kernel=1.0,
        )

        # define system [[A B], [C I]] [u_new, v_new] = [[A -B], [-C I]] [u_old, v_old] (without time step size dt)
        _A = getattr(self.mass_ops, "M" + u_id + "n")

        # preconditioner
        if self.options.precond is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
            pc = pc_class(_A)

        _BC = -1 / 4 * self._accumulator.operators[0]

        self._schur_solver = SchurSolver(
            _A,
            _BC,
            self.options.solver,
            precond=pc,
            solver_params=self.options.solver_params,
        )

    def __call__(self, dt):
        # pointer to old coefficients
        particles = self.variables.ions.particles
        un = self.variables.u.spline.vector

        # sum up total magnetic field b_full1 = b_eq + b_tilde (in-place)
        self._b_eq.copy(out=self._b_full1)

        if self.b_tilde is not None:
            self._b_full1 += self.b_tilde.spline.vector

        # extract coefficients to tensor product space (in-place)
        self._EbT.dot(self._b_full1, out=self._b_full2)

        # update ghost regions because of non-local access in accumulation kernel!
        self._b_full2.update_ghost_regions()

        self._accumulator(
            self._b_full2[0]._data,
            self._b_full2[1]._data,
            self._b_full2[2]._data,
            self._space_key_int,
            self._coupling_mat,
            self._coupling_vec,
            self._boundary_cut_e1,
        )

        # solve linear system for updated u coefficients (in-place)
        un1, info = self._schur_solver(
            un,
            -self._accumulator.vectors[0] / 2,
            dt,
            out=self._u_new,
        )

        # call pusher kernel with average field (u_new + u_old)/2 and update ghost regions because of non-local access in kernel
        _u = un.copy(out=self._u_avg1)
        _u += un1
        _u *= 0.5

        _Eu = self._EuT.dot(_u, out=self._u_avg2)

        _Eu.update_ghost_regions()

        # push particles
        self._pusher(self._scale_push * dt)

        # write new coeffs into Propagator.variables
        max_du = self.update_feec_variables(u=un1)

        # update weights in case of control variate
        if particles.control_variate:
            particles.update_weights()

        if self._info and self._rank == 0:
            logger.info(f"Status     for CurrentCoupling6DCurrent: {info['success']}")
            logger.info(f"Iterations for CurrentCoupling6DCurrent: {info['niter']}")
            logger.info(f"Maxdiff up for CurrentCoupling6DCurrent: {max_du}")
            logger.info("")
