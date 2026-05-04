"Particle and FEEC variables are updated."

import logging
from dataclasses import dataclass

from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.block import BlockVector
from line_profiler import profile

from struphy.feec import preconditioner
from struphy.feec.linear_operators import LinOpWithTransp
from struphy.io.options import LiteralOptions
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


class PressureCoupling6D(Propagator):
    r"""
    :ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\tilde{\mathbf{U}}  \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}` and :math:`f` such that

    .. math::

        \int_\Omega \rho_0 &\frac{\partial \tilde{\mathbf{U}}}{\partial t} \cdot \mathbf V \,\textrm d \mathbf x = - \frac{A_\textnormal{h}}{A_\textnormal{b}} \nabla \cdot \tilde{\mathbb{P}}_{\textnormal{h},\perp} \cdot \mathbf V \,\textrm d \mathbf x
        \qquad \forall \, \mathbf V \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}\,,
        \\[2mm]
        &\frac{\partial f_\textnormal{h}}{\partial t} - \left(\nabla \tilde{\mathbf U}_\perp \cdot \mathbf{v} \right) \cdot \frac{\partial f_\textnormal{h}}{\partial \mathbf{v}} =0\,,
        \\[2mm]
        &\tilde{\mathbb{P}}_{\textnormal{h},\perp} = \int_{\mathbb{R}^3}f_\textnormal{h}\mathbf{v}_\perp \mathbf{v}_\perp^\top \,\textnormal{d}^3 \mathbf v\,.

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point). System size reduction via :class:`~struphy.linear_algebra.schur_solver.SchurSolver`:

    .. math::

        \begin{bmatrix} u^{n+1} - u^n \\ V^{n+1} - V^n \end{bmatrix}
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^n)^{-1} V^\top (\bar {\mathcal X})^\top \mathbb G^\top (\bar {\mathbf \Lambda}^1)^\top \bar {DF}^{-1} \\ - {DF}^{-\top} \bar {\mathbf \Lambda}^1 \mathbb G \bar {\mathcal X} V (\mathbb M^n)^{-1} & 0 \end{bmatrix}
        \begin{bmatrix} {\mathbb M^n}(u^{n+1} + u^n) \\ \bar W (V^{n+1} + V^{n} \end{bmatrix} \,.
    """

    class Variables:
        def __init__(self):
            self._u: FEECVariable = None
            self._energetic_ions: PICVariable = None

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space in ("Hcurl", "Hdiv", "H1vec")
            self._u = new

        @property
        def energetic_ions(self) -> PICVariable:
            return self._energetic_ions

        @energetic_ions.setter
        def energetic_ions(self, new):
            assert isinstance(new, PICVariable)
            assert new.space == "Particles6D"
            self._energetic_ions = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
        # propagator options
        ep_scale: float = 1.0
        u_space: LiteralOptions.OptsVecSpace = "Hdiv"
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        filter_params: FilterParameters = None
        use_perp_model: bool = True

        def __post_init__(self):
            # checks
            check_option(self.u_space, LiteralOptions.OptsVecSpace)
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)
            assert isinstance(self.ep_scale, float)
            assert isinstance(self.use_perp_model, bool)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

            if self.filter_params is None:
                self.filter_params = FilterParameters()

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
        if self.options.u_space == "H1vec":
            self._u_form_int = 0
        else:
            self._u_form_int = int(self.derham.space_to_form[self.options.u_space])

        if self.options.u_space == "Hcurl":
            id_Mn = "M1n"
            id_X = "X1"
        elif self.options.u_space == "Hdiv":
            id_Mn = "M2n"
            id_X = "X2"
        elif self.options.u_space == "H1vec":
            id_Mn = "Mvn"
            id_X = "Xv"

        # call operatros
        id_M = "M" + self.derham.space_to_form[self.options.u_space] + "n"
        _A = getattr(self.mass_ops, id_M)
        self._X = getattr(self.basis_ops, id_X)
        self._XT = self._X.transpose()
        grad = self.derham.grad
        gradT = grad.transpose()

        # Preconditioner
        if self.options.precond is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
            pc = pc_class(getattr(self.mass_ops, id_M))

        # Call the accumulation and Pusher class
        if self.options.use_perp_model:
            accum_ker = Pyccelkernel(accum_kernels.pc_lin_mhd_6d)
            pusher_ker = Pyccelkernel(pusher_kernels.push_pc_GXu)
        else:
            accum_ker = Pyccelkernel(accum_kernels.pc_lin_mhd_6d_full)
            pusher_ker = Pyccelkernel(pusher_kernels.push_pc_GXu_full)

        # define Accumulator and arguments
        self._ACC = Accumulator(
            self.variables.energetic_ions.particles,
            "Hcurl",  # TODO:check
            accum_ker,
            self.mass_ops,
            self.domain.args_domain,
            add_vector=True,
            symmetry="pressure",
            filter_params=self.options.filter_params,
        )

        self._tmp_g1 = grad.codomain.zeros()
        self._tmp_g2 = grad.codomain.zeros()
        self._tmp_g3 = grad.codomain.zeros()

        # instantiate Pusher
        args_pusher_kernel = (
            self.derham.args_derham,
            self._tmp_g1[0]._data,
            self._tmp_g1[1]._data,
            self._tmp_g1[2]._data,
            self._tmp_g2[0]._data,
            self._tmp_g2[1]._data,
            self._tmp_g2[2]._data,
            self._tmp_g3[0]._data,
            self._tmp_g3[1]._data,
            self._tmp_g3[2]._data,
        )

        self._pusher = Pusher(
            self.variables.energetic_ions.particles,
            pusher_ker,
            args_pusher_kernel,
            self.domain.args_domain,
            alpha_in_kernel=1.0,
        )

        self.u_temp = self.variables.u.spline.vector.space.zeros()
        self.u_temp2 = self.variables.u.spline.vector.space.zeros()
        self._tmp = self._X.codomain.zeros()
        self._BV = self.variables.u.spline.vector.space.zeros()

        self._MAT = [
            [self._ACC.operators[0], self._ACC.operators[1], self._ACC.operators[2]],
            [self._ACC.operators[1], self._ACC.operators[3], self._ACC.operators[4]],
            [self._ACC.operators[2], self._ACC.operators[4], self._ACC.operators[5]],
        ]

        self._GT_VEC = BlockVector(self.derham.Vv)

        _BC = -1 / 4 * self._XT @ self.GT_MAT_G(self.derham, self._MAT) @ self._X

        self._schur_solver = SchurSolver(
            _A,
            _BC,
            self.options.solver,
            precond=pc,
            solver_params=self.options.solver_params,
        )

    def __call__(self, dt):
        # current FE coeffs
        un = self.variables.u.spline.vector

        # operators
        grad = self.derham.grad
        gradT = grad.transpose()

        # acuumulate MAT and VEC
        self._ACC(
            self.options.ep_scale,
        )

        # update GT_VEC
        for i in range(3):
            self._GT_VEC[i] = gradT.dot(self._ACC.vectors[i])

        self._BV = self._XT.dot(self._GT_VEC) * (-1 / 2)

        # update u (no tmps created here)
        un1, info = self._schur_solver(un, self._BV, dt, out=self.u_temp)

        _u = un.copy(out=self.u_temp2)
        _u += un1

        # calculate GXu
        Xu = self._X.dot(_u, out=self._tmp)

        GXu_1 = grad.dot(Xu[0], out=self._tmp_g1)
        GXu_2 = grad.dot(Xu[1], out=self._tmp_g2)
        GXu_3 = grad.dot(Xu[2], out=self._tmp_g3)

        GXu_1.update_ghost_regions()
        GXu_2.update_ghost_regions()
        GXu_3.update_ghost_regions()

        # push particles
        self._pusher(dt)

        # write new coeffs into Propagator.variables
        diffs = self.update_feec_variables(u=un1)

        # update weights in case of control variate
        if self.variables.energetic_ions.species.weights_params.control_variate:
            self.variables.energetic_ions.particles.update_weights()

        if self.options.solver_params.info and MPI.COMM_WORLD.Get_rank() == 0:
            logger.info(f"Status     for StepPressurecoupling: {info['success']}")
            logger.info(f"Iterations for StepPressurecoupling: {info['niter']}")
            logger.info(f"Maxdiff u1 for StepPressurecoupling: {diffs['u']}")
            logger.info("")

    class GT_MAT_G(LinOpWithTransp):
        r"""
        Class for defining LinearOperator corresponding to :math:`G^\top (\text{MAT}) G \in \mathbb{R}^{3N^0 \times 3N^0}`
        where :math:`\text{MAT} = V^\top (\bar {\mathbf \Lambda}^1)^\top \bar{DF}^{-1} \bar{W} \bar{DF}^{-\top} \bar{\mathbf \Lambda}^1 V \in \mathbb{R}^{3N^1 \times 3N^1}`.

        Parameters
        ----------
            derham : struphy.feec.psydac_derham.Derham
                Discrete de Rham sequence on the logical unit cube.

            MAT : List of StencilMatrices
                List with six of accumulated pressure terms
        """

        def __init__(self, derham, MAT, transposed=False):
            self._derham = derham
            self._grad = derham.grad
            self._gradT = derham.grad.transpose()

            self._domain = derham.Vv
            self._codomain = derham.Vv
            self._MAT = MAT

            self._vector = BlockVector(derham.Vv)
            self._temp = BlockVector(derham.V1)

        @property
        def domain(self):
            return self._domain

        @property
        def codomain(self):
            return self._codomain

        @property
        def dtype(self):
            return self._derham.Vv.dtype

        @property
        def tosparse(self):
            raise NotImplementedError()

        @property
        def toarray(self):
            raise NotImplementedError()

        @property
        def transposed(self):
            return self._transposed

        def transpose(self):
            return self.GT_MAT_G(self._derham, self._MAT, True)

        def dot(self, v, out=None):
            """dot product between GT_MAT_G and v.

            Parameters
            ----------
                v : StencilVector or BlockVector
                    Input FE coefficients from V.coeff_space.

            Returns
            -------
                A StencilVector or BlockVector from W.coeff_space."""

            assert v.space == self.domain

            v.update_ghost_regions()

            for i in range(3):
                for j in range(3):
                    self._temp += self._MAT[i][j].dot(self._grad.dot(v[j]))

                self._vector[i] = self._gradT.dot(self._temp)
                self._temp *= 0.0

            self._vector.update_ghost_regions()

            if out is not None:
                self._vector.copy(out=out)

            assert self._vector.space == self.codomain

            return self._vector
