from feectools.linalg.basic import IdentityOperator, LinearOperator, Vector
from feectools.linalg.block import BlockLinearOperator, BlockVector
from feectools.linalg.solvers import inverse
from line_profiler import profile
from scope_profiler import ProfileManager

from struphy.feec.mass import WeightedMassOperator
from struphy.linear_algebra.solver import SolverParameters


class SchurOperator(LinearOperator):
    r"""The operator :math:`A + c\, \mathbb B_W\, BC\, \mathbb B_V^\top` of :class:`SchurSolver`, applied lazily
    (without forming the sum as a matrix): :math:`A` and :math:`BC` are applied one after the other and the results
    are added.

    :math:`\mathbb B_V` and :math:`\mathbb B_W` are the boundary operators of :math:`A` if :math:`A` is a
    :class:`~struphy.feec.mass.WeightedMassOperator` (identities otherwise). The operator is thus the same as the
    matrix sum :math:`A + c\, BC` assembled into :math:`A` (whose boundary operators are applied to the whole sum).
    Unlike the matrix sum, it also works if :math:`BC` is not a matrix of the tensor-product spline space, e.g. a
    product of discrete derivatives and mass matrices on polar splines, or a matrix-free operator.

    Parameters
    ----------
    A : LinearOperator
        First summand.

    BC : LinearOperator
        Second summand (without the factor :math:`c` and the boundary operators).

    scalar : float
        The factor :math:`c`, can be changed with :attr:`scalar` (e.g. :math:`-\Delta t^2` for a new time step).
    """

    def __init__(self, A: LinearOperator, BC: LinearOperator, scalar: float = 1.0):
        assert A.domain == BC.domain
        assert A.codomain == BC.codomain
        self._A = A
        self._BC = BC
        self._scalar = scalar

        # boundary operators of A (the matrix sum applies them to BC too); skipped if they are identities
        self._bv = None
        self._bw = None
        if isinstance(A, WeightedMassOperator):
            bv, bw = A.boundary_ops
            if A.transposed:  # the boundary operators of V act on the codomain side
                bv, bw = bw, bv
            if not isinstance(bv, IdentityOperator):
                self._bv = bv.T
                self._tmp_v = A.domain.zeros()
            if not isinstance(bw, IdentityOperator):
                self._bw = bw
                self._tmp_w = A.codomain.zeros()

        self._tmp = A.codomain.zeros()

    @property
    def A(self):
        """First summand :math:`A`."""
        return self._A

    @property
    def BC(self):
        """Second summand :math:`BC`."""
        return self._BC

    @property
    def scalar(self):
        """The factor :math:`c` of :math:`BC`."""
        return self._scalar

    @scalar.setter
    def scalar(self, c):
        self._scalar = c

    @property
    def domain(self):
        return self._A.domain

    @property
    def codomain(self):
        return self._A.codomain

    @property
    def dtype(self):
        return self._A.dtype

    def tosparse(self):
        raise NotImplementedError()

    def toarray(self):
        raise NotImplementedError()

    def transpose(self, conjugate=False):
        return SchurOperator(
            self._A.transpose(conjugate=conjugate), self._BC.transpose(conjugate=conjugate), self._scalar
        )

    def dot(self, v, out=None):
        """``out = A v + c B_W BC B_V^T v`` (no temporaries are allocated if ``out`` is given)."""
        assert isinstance(v, Vector)
        assert v.space == self.domain
        if out is None:
            out = self.codomain.zeros()
        else:
            assert isinstance(out, Vector)
            assert out.space == self.codomain
        if out is v:
            v = v.copy()

        x = v if self._bv is None else self._bv.dot(v, out=self._tmp_v)
        if self._bw is None:
            y = self._BC.dot(x, out=self._tmp)
        else:
            self._BC.dot(x, out=self._tmp_w)
            y = self._bw.dot(self._tmp_w, out=self._tmp)
        y *= self._scalar

        self._A.dot(v, out=out)
        out += y
        return out


class SchurSolver:
    r"""Solves for :math:`x^{n+1}` in the block system

    .. math::

        \begin{pmatrix} A & \Delta t\, B \\ \Delta t\, C & \mathbb I \end{pmatrix}
        \begin{pmatrix} x^{n+1} \\ y^{n+1} \end{pmatrix}
        =
        \begin{pmatrix} A & - \Delta t\, B \\ - \Delta t\, C & \mathbb I \end{pmatrix}
        \begin{pmatrix} x^n \\ y^n \end{pmatrix}

    using the Schur complement :math:`S = A - \Delta t^2 BC`, where :math:`\mathbb I` is the identity matrix
    and :math:`(x^n, y^n)` is given. The solution is

    .. math::

        x^{n+1} = S^{-1} \left[ (A + \Delta t^2 BC) \, x^n - 2 \Delta t\, B \, y^n \right] \,.

    The caller passes :math:`B y^n` and computes :math:`y^{n+1} = y^n - \Delta t\, C (x^{n+1} + x^n)` itself.

    If :math:`A` and :math:`BC` are both :class:`~struphy.feec.mass.WeightedMassOperator` of the same spaces (e.g.
    :math:`BC` is an accumulated particle matrix), :math:`S` is assembled as a matrix (in a copy of :math:`A`).
    Otherwise (e.g. :math:`BC` is a product of discrete derivatives and mass matrices, or a matrix-free operator),
    :math:`S` is applied lazily as a :class:`SchurOperator`, which also works with polar splines.

    Parameters
    ----------
    A : LinearOperator
        Upper left block :math:`A`.

    BC : LinearOperator
        Product :math:`BC` of the off-diagonal blocks (same domain and codomain as :math:`A`).

    solver_name : str
        Name of the iterative solver, passed to ``feectools.linalg.solvers.inverse`` (e.g. ``"pcg"``, ``"cg"``,
        ``"pbicgstab"``, ``"gmres"``).

    precond : LinearOperator, optional
        Preconditioner of the Schur complement :math:`S` (can be replaced later with :attr:`precond`).

    solver_params : SolverParameters, optional
        Solver controls (``tol``, ``maxiter``, ...); defaults to ``SolverParameters()``.
    """

    def __init__(
        self,
        A: LinearOperator,
        BC: LinearOperator,
        solver_name: str,
        precond=None,  # TODO: add Preconditioner base class
        solver_params: SolverParameters = None,
    ):
        assert isinstance(A, LinearOperator)
        assert isinstance(BC, LinearOperator)

        assert A.domain == BC.domain
        assert A.codomain == BC.codomain

        if solver_params is None:
            solver_params = SolverParameters()

        # linear operators
        self._A = A
        self._BC = BC

        # left- and right-hand side operators A - dt^2 BC and A + dt^2 BC (see update_operators): copies of A for the
        # matrix sum (allocated at the first solve), lazy SchurOperators otherwise (rebuilt when A or BC is replaced)
        self._schur = None
        self._rhs_m = None
        self._lazy_ops = None

        # initialize solver with dummy matrix A
        self._solver_name = solver_name

        kwargs = solver_params.__dict__.copy()
        kwargs.pop("info")
        if precond is not None:
            kwargs["pc"] = precond
        self._solver_kwargs = kwargs

        self._solver = inverse(A, solver_name, **kwargs)

        # right-hand side vector (avoids temporary memory allocation!)
        self._rhs = A.codomain.zeros()

    @property
    def A(self):
        """Upper left block :math:`A`."""
        return self._A

    @property
    def BC(self):
        """Product :math:`BC` of the off-diagonal blocks."""
        return self._BC

    @property
    def precond(self):
        """Preconditioner of the Schur system (None without preconditioner)."""
        return self._solver_kwargs.get("pc")

    @precond.setter
    def precond(self, pc):
        """Replace the preconditioner, e.g. by one of the Schur system for the current dt (built at the first solve)."""
        self._solver_kwargs.pop("pc", None)
        if pc is not None:
            self._solver_kwargs["pc"] = pc
        self._solver = inverse(self._A, self._solver_name, **self._solver_kwargs)

    @A.setter
    def A(self, a):
        """Upper left block :math:`A`."""
        self._A = a
        self._lazy_ops = None

    @BC.setter
    def BC(self, bc):
        """Product :math:`BC` of the off-diagonal blocks."""
        self._BC = bc
        self._lazy_ops = None

    @property
    def matrix_sum(self) -> bool:
        """Whether :math:`S` is assembled as a matrix (:math:`A` and :math:`BC` are weighted mass operators of the same
        spaces) or applied lazily as a :class:`SchurOperator`."""
        return (
            isinstance(self._A, WeightedMassOperator)
            and isinstance(self._BC, WeightedMassOperator)
            and self._BC.domain is self._A.domain
            and self._BC.codomain is self._A.codomain
        )

    def update_operators(self, dt: float):
        r"""Set the left- and right-hand side operators :math:`S = A - \Delta t^2 BC` (also the operator of the
        iterative solver) and :math:`A + \Delta t^2 BC` for the time step ``dt``.

        Returns
        -------
        schur, rhs_m : LinearOperator
            The left- and right-hand side operators.
        """
        if self.matrix_sum:
            if self._schur is None:
                self._schur = self._A.copy()
                self._rhs_m = self._A.copy()

            self._schur *= 0.0
            self._schur += self._BC
            self._schur *= -(dt**2)
            self._schur += self._A

            self._rhs_m *= 0.0
            self._rhs_m += self._BC
            self._rhs_m *= dt**2
            self._rhs_m += self._A
            schur, rhs_m = self._schur, self._rhs_m
        else:
            if self._lazy_ops is None:
                self._lazy_ops = (SchurOperator(self._A, self._BC), SchurOperator(self._A, self._BC))
            schur, rhs_m = self._lazy_ops
            schur.scalar = -(dt**2)
            rhs_m.scalar = dt**2

        # use setter to update lhs matrix
        self._solver.linop = schur
        return schur, rhs_m

    @profile
    @ProfileManager.profile("solve: SchurSolver")
    def __call__(self, xn, Byn, dt, out=None):
        """Solves the 2x2 block matrix linear system.

        Parameters
        ----------
        xn : feectools.linalg.basic.Vector
            Solution from previous time step.

        Byn : feectools.linalg.basic.Vector
            The product B*yn.

        dt : float
            Time step size.

        out : feectools.linalg.basic.Vector, optional
            If given, the converged solution will be written into this vector (in-place).

        Returns
        -------
        out : feectools.linalg.basic.Vector
            Converged solution.

        info : dict
            Convergence information.
        """

        assert isinstance(xn, Vector)
        assert isinstance(Byn, Vector)
        assert xn.space == self._A.domain
        assert Byn.space == self._A.codomain

        # left- and right-hand side operators
        _, rhs_m = self.update_operators(dt)

        # right-hand side vector rhs = 2*dt*[ rhs_m/(2*dt) @ xn - Byn ] (in-place!)
        rhs = rhs_m.dot(xn, out=self._rhs)
        rhs /= 2 * dt
        rhs -= Byn
        rhs *= 2 * dt

        # solve linear system (in-place if out is not None)
        x = self._solver.dot(rhs, out=out)

        return x, self._solver._info


class SchurSolverFull:
    r"""Solves the block system

    .. math::

        \begin{pmatrix} A & B \\ C & \mathbb I \end{pmatrix}
        \begin{pmatrix} x \\ y \end{pmatrix}
        =
        \begin{pmatrix} b_x \\ b_y \end{pmatrix}

    using the Schur complement :math:`S = A - BC`, where :math:`\mathbb I` is the identity matrix
    and :math:`(b_x, b_y)^\top` is given. The solution is

    .. math::

        x &= S^{-1} \, (b_x - B b_y ) \,,
        \\
        y &= b_y - C x \,.

    Parameters
    ----------
    M : BlockLinearOperator
        Block matrix :math:`[[A, B], [C, \mathbb I]]`.

    solver_name : str
        Name of the iterative solver, passed to ``feectools.linalg.solvers.inverse`` (e.g. ``"pcg"``, ``"cg"``,
        ``"pbicgstab"``, ``"gmres"``), or ``"rpbicgstab"``: BiCGStab (``"bicgstab"``) for :math:`S P` with the
        preconditioner :math:`P` (right preconditioning), then :math:`x = P\, x'`. It needs two products with
        :math:`P` per iteration (``"pbicgstab"`` three), with the same stopping criterion (residual of :math:`S x`).

    **solver_params :
        Keyword arguments of the chosen solver; must contain ``pc`` (the preconditioner, None for none).
    """

    def __init__(self, M, solver_name, **solver_params):
        assert isinstance(M, BlockLinearOperator)
        assert M.domain == M.codomain  # solve square system

        # initialize solver with dummy matrix A
        self._solver_name = solver_name

        if solver_params["pc"] is None:
            solver_params.pop("pc")

        self._M = M

        self._A = M[0, 0]
        self._B = M[0, 1]
        self._C = M[1, 0]
        assert isinstance(M[1, 1], IdentityOperator)

        self._S = self._A - self._B @ self._C

        self._solver_params = solver_params
        self._build_solver()

        # right-hand side vector (avoids temporary memory allocation!)
        self._rhs = self._A.codomain.zeros()
        self._tmp_x = self._A.domain.zeros()

    def _build_solver(self):
        """The solver of the Schur system, with right preconditioning for ``"rpbicgstab"``."""
        if self._solver_name == "rpbicgstab":
            params = {k: v for k, v in self._solver_params.items() if k != "pc"}
            self._right_pc = self._solver_params.get("pc")
            op = self._S if self._right_pc is None else self._S @ self._right_pc
            self._solver = inverse(op, "bicgstab", **params)
        else:
            self._right_pc = None
            self._solver = inverse(self._S, self._solver_name, **self._solver_params)

    @property
    def precond(self):
        """Preconditioner of the Schur system (None without preconditioner)."""
        return self._solver_params.get("pc")

    @precond.setter
    def precond(self, pc):
        """Replace the preconditioner, e.g. by one of the Schur system for the current dt."""
        self._solver_params.pop("pc", None)
        if pc is not None:
            self._solver_params["pc"] = pc
        self._build_solver()

    @profile
    @ProfileManager.profile("solve: SchurSolverFull")
    def dot(self, v, out=None):
        """Solves the 2x2 block matrix linear system.

        Parameters
        ----------
        v : feectools.linalg.basic.Vector
            Left hand side of the system.

        out : feectools.linalg.basic.Vector, optional
            If given, the converged solution will be written into this vector (in-place).

        Returns
        -------
        out : feectools.linalg.block.BLockVector
            Converged solution.

        info : dict
            Convergence information.
        """

        assert isinstance(v, BlockVector)
        assert v.space == self._M.domain

        if out is None:
            out = self._M.codomain.zeros()
        else:
            assert out.space == self._M.codomain

        bx = v[0]
        by = v[1]

        # right-hand side vector rhs bx - B by
        rhs = self._B.dot(by, out=self._rhs)
        rhs *= -1
        rhs += bx

        # solve linear system (in-place if out is not None)
        if self._right_pc is None:
            x = self._solver.dot(rhs, out=out[0])
        else:
            xp = self._solver.dot(rhs, out=self._tmp_x)
            x = self._right_pc.dot(xp, out=out[0])
        y = self._C.dot(x, out=out[1])
        y *= -1
        y += by

        return out


class SchurSolverFull3:
    r"""Solves the block system

    .. math::

        \begin{pmatrix} A & B & D \\ C & \mathbb I & 0 \\ E & 0 & \mathbb I \end{pmatrix}
        \begin{pmatrix} x \\ y \\ z \end{pmatrix}
        =
        \begin{pmatrix} b_x \\ b_y \\ b_z \end{pmatrix}

    using the Schur complement :math:`S = A - BC - DE`, where :math:`\mathbb I` is the identity matrix
    and :math:`(b_x, b_y, b_z)^\top` is given. The solution is

    .. math::

        x &= S^{-1} \, (b_x - B b_y - D b_z) \,,
        \\
        y &= b_y - C x \,,
        \\
        z &= b_z - E x \,.

    Parameters
    ----------
    M : BlockLinearOperator
        Block matrix :math:`[[A, B, D], [C, \mathbb I, 0], [E, 0, \mathbb I]]`.

    solver_name : str
        Name of the iterative solver, passed to ``feectools.linalg.solvers.inverse`` (e.g. ``"pcg"``, ``"cg"``,
        ``"pbicgstab"``, ``"gmres"``).

    **solver_params :
        Keyword arguments of the chosen solver; must contain ``pc`` (the preconditioner, None for none).
    """

    def __init__(self, M, solver_name, **solver_params):
        assert isinstance(M, BlockLinearOperator)
        assert M.domain == M.codomain  # solve square system

        # initialize solver with dummy matrix A
        self._solver_name = solver_name

        if solver_params["pc"] is None:
            solver_params.pop("pc")

        self._M = M

        self._A = M[0, 0]
        self._B = M[0, 1]
        self._C = M[1, 0]
        self._D = M[0, 2]
        self._E = M[2, 0]
        NoneType = type(None)
        assert isinstance(M[1, 1], IdentityOperator)
        assert isinstance(M[2, 2], IdentityOperator)
        assert isinstance(M[1, 2], NoneType)
        assert isinstance(M[2, 1], NoneType)

        self._S = self._A - self._B @ self._C - self._D @ self._E

        self._solver_params = solver_params
        self._solver = inverse(self._S, solver_name, **solver_params)

        # right-hand side vector (avoids temporary memory allocation!)
        self._rhs = self._A.codomain.zeros()
        self._rhs2 = self._A.codomain.zeros()

    @property
    def precond(self):
        """Preconditioner of the Schur system (None without preconditioner)."""
        return self._solver_params.get("pc")

    @precond.setter
    def precond(self, pc):
        """Replace the preconditioner, e.g. by one of the Schur system for the current dt."""
        self._solver_params.pop("pc", None)
        if pc is not None:
            self._solver_params["pc"] = pc
        self._solver = inverse(self._S, self._solver_name, **self._solver_params)

    @profile
    @ProfileManager.profile("solve: SchurSolverFull3")
    def dot(self, v, out=None):
        """Solves the 3x3 block matrix linear system.

        Parameters
        ----------
        v : feectools.linalg.basic.Vector
            Left hand side of the system.

        out : feectools.linalg.basic.Vector, optional
            If given, the converged solution will be written into this vector (in-place).

        Returns
        -------
        out : feectools.linalg.block.BLockVector
            Converged solution.

        info : dict
            Convergence information.
        """

        assert isinstance(v, BlockVector)
        assert v.space == self._M.domain

        if out is None:
            out = self._M.codomain.zeros()
        else:
            assert out.space == self._M.codomain

        bx = v[0]
        by = v[1]
        bz = v[2]

        # right-hand side vector rhs bx - B by
        rhs = self._B.dot(by, out=self._rhs)
        rhs *= -1
        rhs -= self._D.dot(bz, out=self._rhs2)
        rhs += bx

        # solve linear system (in-place if out is not None)
        x = self._solver.dot(rhs, out=out[0])
        y = self._C.dot(x, out=out[1])
        y *= -1
        y += by
        z = self._E.dot(x, out=out[2])
        z *= -1
        z += bz

        return out
