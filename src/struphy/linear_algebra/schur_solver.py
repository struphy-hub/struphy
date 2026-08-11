from feectools.linalg.basic import IdentityOperator, LinearOperator, Vector
from feectools.linalg.block import BlockLinearOperator, BlockVector
from feectools.linalg.solvers import inverse
from line_profiler import profile

from struphy.linear_algebra.solver import SolverParameters
from struphy.linear_algebra.solver import inverse as struphy_inverse


class SchurSolver:
    r"""Solves for :math:`x^{n+1}` in the block system

    .. math::

        \left( \matrix{
            A & \Delta t B \cr
            \Delta t C & \\text{Id}
        } \\right)
        \left( \matrix{
            x^{n+1} \cr y^{n+1}
        } \\right)
        =
        \left( \matrix{
            A & - \Delta t B \cr
            - \Delta t C & \\text{Id}
        } \\right)
        \left( \matrix{
            x^n \cr y^n
        } \\right)

    using the Schur complement :math:`S = A - \Delta t^2 BC`, where Id is the identity matrix
    and :math:`(x^n, y^n)` is given. The solution is given by

    .. math::

        x^{n+1} = S^{-1} \left[ (A + \Delta t^2 BC) \, x^n - 2 \Delta t B \, y^n \\right] \,.

    Parameters
    ----------
    A : LinearOperator
        Upper left block from [[A B], [C Id]].

    BC : LinearOperator
        Product from [[A B], [C Id]].

    solver_name : str
        See [feectools.linalg.solvers](https://github.com/pyccel/psydac/blob/535717c6f5ea328aacbbbbcc2d582a92b31c9377/psydac/linalg/solvers.py#L47) for possible names.

    **solver_params :
        Must correspond to the chosen solver.
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
        # Set by the A/BC property setters whenever a caller reassigns either (e.g.
        # VlasovAmpereCoupling/EfieldWeightsCoupling rebuild `.BC` from a fresh, particle-dependent
        # operator every call); only consulted for petsc, see below.
        self._schur_dirty = True

        self._is_petsc = solver_name == "petsc"

        if self._is_petsc:
            # PETScSolver caches its assembled PETSc.Mat by the *object identity* of `linop`
            # (see PETScSolver._get_ksp), rebuilding only when that identity changes -- exactly
            # the mechanism ImplicitDiffusion relies on (see its lhs-operator caching). The
            # in-place-mutated `self._schur` buffer below defeats that: it is the same Python
            # object on every call, so PETScSolver would keep the *first* call's matrix forever,
            # silently going stale if `dt`, `A` or `BC` ever change. So for petsc, `self._schur`
            # is instead a fresh composite operator, rebuilt only when `dt` changes or `.A`/`.BC`
            # were reassigned (`self._schur_dirty`) -- see __call__. Callers that mutate an
            # operator obtained via the `A`/`BC` getters in place, without reassigning it through
            # the setter, would not be detected -- no current caller does this.
            self._schur = None
            self._schur_dt = None
        else:
            # Allocate memory for matrices used in solving the Schur system
            self._schur = A.copy()

        self._rhs_m = A.copy()

        # initialize solver with dummy matrix A
        self._solver_name = solver_name

        kwargs = solver_params.__dict__.copy()
        kwargs.pop("info")
        if precond is not None:
            kwargs["pc"] = precond

        if self._is_petsc:
            # struphy's inverse() dispatches "petsc" to PETScSolver and forwards pc_type; the
            # dummy operator here is just to build the solver object -- __call__ always assigns
            # the real one via `self._solver.linop` before solving.
            self._solver = struphy_inverse(A, solver_name, **kwargs)
        else:
            # pc_type is petsc-only (see struphy.linear_algebra.solver.SolverParameters); this
            # branch goes straight to feectools' own `inverse` (imported directly above, not
            # struphy's petsc-aware wrapper, which would otherwise strip it), whose
            # InverseLinearOperator subclasses forward unknown kwargs straight to their
            # constructor and would raise on it.
            kwargs.pop("pc_type", None)
            self._solver = inverse(A, solver_name, **kwargs)

        # right-hand side vector (avoids temporary memory allocation!)
        self._rhs = A.codomain.zeros()

    @property
    def A(self):
        """Upper left block from [[A B], [C Id]]."""
        return self._A

    @property
    def BC(self):
        """Product from [[A B], [C Id]]."""
        return self._BC

    @A.setter
    def A(self, a):
        """Upper left block from [[A B], [C Id]]."""
        self._A = a
        # e.g. VlasovAmpereCoupling/EfieldWeightsCoupling reassign `.A`/`.BC` to a fresh
        # (possibly particle-dependent) operator every call; `x.A *= y`-style augmented
        # assignment also lands here (Python always re-invokes the setter). See the petsc
        # cache-invalidation note in __init__/__call__.
        self._schur_dirty = True

    @BC.setter
    def BC(self, bc):
        """Product from [[A B], [C Id]]."""
        self._BC = bc
        self._schur_dirty = True

    @profile
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

        # left-hand side operator
        if self._is_petsc:
            if self._schur is None or dt != self._schur_dt or self._schur_dirty:
                self._schur = self._A - (dt**2) * self._BC
                self._schur_dt = dt
                self._schur_dirty = False
        else:
            self._schur *= 0.0
            self._schur += self._BC
            self._schur *= -(dt**2)
            self._schur += self._A

        # right-hand side operator
        self._rhs_m *= 0.0
        self._rhs_m += self._BC
        self._rhs_m *= dt**2
        self._rhs_m += self._A

        # use setter to update lhs matrix
        self._solver.linop = self._schur

        # right-hand side vector rhs = 2*dt*[ rhs_m/(2*dt) @ xn - Byn ] (in-place!)
        rhs = self._rhs_m.dot(xn, out=self._rhs)
        rhs /= 2 * dt
        rhs -= Byn
        rhs *= 2 * dt

        # solve linear system (in-place if out is not None)
        x = self._solver.dot(rhs, out=out)

        return x, self._solver._info


class SchurSolverFull:
    r"""Solves the block system

    .. math::

        \left( \matrix{
            A & B \cr
            C & \\text{Id}
        } \\right)
        \left( \matrix{
            x \cr y
        } \\right)
        =
        \left( \matrix{
            b_x \cr b_y
        } \\right)

    using the Schur complement :math:`S = A - BC`, where Id is the identity matrix
    and :math:`(b_x, b_y)^T` is given. The solution is given by

    .. math::

        x &= S^{-1} \, (b_x - B b_y ) \,,

        y &= b_y - C x \,.

    Parameters
    ----------
    M : BlockLinearOperator
        Matrix [[A B], [C Id]].

    solver_name : str
        See [feectools.linalg.solvers](https://github.com/pyccel/psydac/blob/535717c6f5ea328aacbbbbcc2d582a92b31c9377/psydac/linalg/solvers.py#L47) for possible names.

    **solver_params :
        Must correspond to the chosen solver.
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

        self._solver = inverse(self._S, solver_name, **solver_params)

        # right-hand side vector (avoids temporary memory allocation!)
        self._rhs = self._A.codomain.zeros()

    @profile
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
        x = self._solver.dot(rhs, out=out[0])
        y = self._C.dot(x, out=out[1])
        y *= -1
        y += by

        return out


class SchurSolverFull3:
    r"""Solves the block system

    .. math::

        \left( \matrix{
            A & B & D \cr
            C & \\text{Id} & 0 \cr
            E & 0 & \\text{Id} \cr
        } \\right)
        \left( \matrix{
            x \cr y \cr z
        } \\right)
        =
        \left( \matrix{
            b_x \cr b_y \cr b_z
        } \\right)

    using the Schur complement :math:`S = A - BC - DE`, where Id is the identity matrix
    and :math:`(b_x, b_y, b_z)^T` is given. The solution is given by

    .. math::

        x &= S^{-1} \, (b_x - B b_y - D b_z) \,,

        y &= b_y - C x \,,

        z &= b_z - E x \,.

    Parameters
    ----------
    M : BlockLinearOperator
        Matrix [[A B D], [C Id 0], [E 0 Id]].

    solver_name : str
        See [feectools.linalg.solvers](https://github.com/pyccel/psydac/blob/535717c6f5ea328aacbbbbcc2d582a92b31c9377/psydac/linalg/solvers.py#L47) for possible names.

    **solver_params :
        Must correspond to the chosen solver.
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

        self._solver = inverse(self._S, solver_name, **solver_params)

        # right-hand side vector (avoids temporary memory allocation!)
        self._rhs = self._A.codomain.zeros()
        self._rhs2 = self._A.codomain.zeros()

    @profile
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
