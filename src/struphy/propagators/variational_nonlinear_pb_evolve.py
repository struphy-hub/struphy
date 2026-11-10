import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from feectools.linalg.solvers import inverse
from line_profiler import profile
from maybempi import MPI

from struphy.feec.preconditioner import MassMatrixPreconditioner
from struphy.feec.variational_utilities import Hdiv0_transport_operator, Pressure_transport_operator
from struphy.io.options import OptionsBase
from struphy.linear_algebra.solver import NonlinearSolverParameters, SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.propagators.utils import VelocityMass
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class VariationalNonlinearPBEvolve(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the terms of the perturbative (:math:`\delta f`) variational MHD model that
    are nonlinear in the perturbations of the magnetic field and the pressure. Together with the linear propagators
    (:class:`~struphy.propagators.variational_linear_alfven_evolve.VariationalLinearAlfvenEvolve` and
    :class:`~struphy.propagators.variational_linear_sound_evolve.VariationalLinearSoundEvolve`) it gives the
    coupling of the velocity, the magnetic field and the pressure of the perturbative model.

    **Continuous problem.** Find :math:`\mathbf u` (in :math:`(H^1)^3` or :math:`H_0(\textnormal{div})`) and the
    perturbations :math:`\tilde{\mathbf B} \in H_0(\textnormal{div})`, :math:`\tilde p \in L^2` such that, with the
    transport operators :math:`\mathcal T_{\bar{\mathbf B}}(\mathbf v) = \nabla \times (\bar{\mathbf B} \times \mathbf v)`
    and :math:`\mathcal S_{\bar p}(\mathbf v) = \nabla \cdot (\bar p\, \mathbf v) + (\gamma - 1) \bar p\, \nabla \cdot
    \mathbf v`,

    .. math::

        \int_\Omega \rho\, \partial_t \mathbf u \cdot \mathbf v\,\textrm d \mathbf x
        - \int_\Omega \tilde{\mathbf B} \cdot \mathcal T_{\tilde{\mathbf B}}(\mathbf v)\,\textrm d \mathbf x = 0
        \qquad \forall \, \mathbf v\,,
        \qquad
        \partial_t \tilde{\mathbf B} + \mathcal T_{\tilde{\mathbf B}}(\mathbf u) = 0 \,,
        \qquad
        \partial_t \tilde p + \mathcal S_{\tilde p}(\mathbf u) = 0 \,,

    with the total density :math:`\rho = \rho_0 + \tilde\rho` (frozen). These are the terms of the transport of the
    magnetic field and of the pressure with the perturbations as weights, and the force of the first one, the variation
    of :math:`\frac12 \int |\tilde{\mathbf B}|^2`. The pressure perturbation does not enter the energy of the
    perturbative model (only the second-order pressure :math:`\hat p` does, in the linear sound step), so its nonlinear
    transport has no force. The energy :math:`\frac12 \int \rho |\mathbf u|^2 + \frac12 \int |\tilde{\mathbf B}|^2` is
    conserved.

    **Semi-discrete problem.** With :math:`\mathbb T[\mathbf b] = \mathbb C \hat\Pi^1[\hat{\mathbf b} \times
    \vec{\boldsymbol\Lambda}^u]` (:class:`~struphy.feec.variational_utilities.Hdiv0_transport_operator`, for a
    velocity 2-form the product is divided by :math:`\sqrt g`) and the pressure transport :math:`\mathbb S[\mathbf p]`
    (:class:`~struphy.feec.variational_utilities.Pressure_transport_operator`), the operators of the linear propagators
    with the perturbations as weights,

    .. math::

        \mathbb M^u_\rho \dot{\mathbf u} - \mathbb T[\mathbf b]^\top \mathbb M^2 \mathbf b = 0 \,,
        \qquad
        \dot{\mathbf b} + \mathbb T[\mathbf b]\, \mathbf u = 0 \,,
        \qquad
        \dot{\mathbf p} + \mathbb S[\mathbf p]\, \mathbf u = 0 \,,

    with the velocity mass matrix of the total density, :math:`\mathbb M^u_\rho` = ``WMMnew`` for
    :math:`\mathbf u \in (H^1)^3` (assembled in :class:`~struphy.propagators.variational_density_evolve.VariationalDensityEvolve`)
    or :math:`\mathbb M^2_\rho` (:class:`~struphy.propagators.utils.VelocityMass`) for a velocity 2-form.

    **Time discretization.** Implicit midpoint rule, with the weights :math:`\mathbf b^{n+\frac12}`,
    :math:`\mathbf p^{n+\frac12}`:

    .. math::

        \mathbb M^u_\rho \frac{\mathbf u^{n+1} - \mathbf u^n}{\Delta t}
        = \mathbb T[\mathbf b^{n+\frac12}]^\top \mathbb M^2 \mathbf b^{n+\frac12} \,,
        \qquad
        \frac{\mathbf b^{n+1} - \mathbf b^n}{\Delta t} = -\mathbb T[\mathbf b^{n+\frac12}]\, \mathbf u^{n+\frac12} \,,
        \qquad
        \frac{\mathbf p^{n+1} - \mathbf p^n}{\Delta t} = -\mathbb S[\mathbf p^{n+\frac12}]\, \mathbf u^{n+\frac12} \,,

    which conserves the energy exactly. It is solved by a Picard iteration (one mass-matrix solve per iteration), which
    converges for :math:`\Delta t` small compared with the inverse frequency of the waves carried by the perturbation
    (:math:`\Delta t\, k\, |\tilde{\mathbf B}| / \sqrt{\rho} \lesssim 1`); a warning is logged when it does not
    converge within ``nonlin_solver.maxiter`` iterations.
    """

    class Variables:
        """Container for variables advanced by :class:`VariationalNonlinearPBEvolve`.

        Attributes
        ----------
        u : FEECVariable
            Velocity variable in ``"H1vec"`` or ``"Hdiv"`` space.
        b : FEECVariable
            Magnetic-field perturbation in ``"Hdiv"`` space.
        p : FEECVariable
            Pressure perturbation in ``"L2"`` space.
        """

        def __init__(self):
            self._u: FEECVariable = None
            self._b: FEECVariable = None
            self._p: FEECVariable = None

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space in ("H1vec", "Hdiv")
            self._u = new

        @property
        def b(self) -> FEECVariable:
            return self._b

        @b.setter
        def b(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hdiv"
            self._b = new

        @property
        def p(self) -> FEECVariable:
            return self._p

        @p.setter
        def p(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "L2"
            self._p = new

    def __init__(self, rho: FEECVariable = None):
        r"""
        Parameters
        ----------
        rho : FEECVariable, default=None
            Density perturbation :math:`\tilde\rho` (``"L2"`` space, not updated): with a velocity in ``"Hdiv"`` the
            velocity mass matrix is :math:`\mathbb M^2_\rho` of the total density :math:`\rho_0 + \tilde\rho`
            (re-assembled in every step); None for :math:`\mathbb M^2_{\rho_0}`.
        """
        assert rho is None or (isinstance(rho, FEECVariable) and rho.space == "L2")
        self.variables = self.Variables()
        self._rho_field = rho

    @dataclass(repr=False)
    class Options(OptionsBase):
        r"""Configuration options for :class:`VariationalNonlinearPBEvolve`.

        Parameters
        ----------
        gamma : float, default=5/3
            Adiabatic index.
        nonlin_solver : NonlinearSolverParameters, default=None
            Controls of the Picard iteration: it stops when the change of :math:`(\mathbf u, \mathbf b)` in the energy
            norm is below ``tol`` times the change in the step; at most ``maxiter`` iterations.
        solver : {"pcg", "cg"}, default="pcg"
            Solver of the mass-matrix systems.
        precond : {"MassMatrixPreconditioner", None}, default="MassMatrixPreconditioner"
            Preconditioner of the mass-matrix systems.
        solver_params : SolverParameters, default=None
            Controls of the mass-matrix solves.
        """

        # specific literals
        OptsSolver = Literal["pcg", "cg"]
        OptsPrecond = Literal["MassMatrixPreconditioner", None]
        # propagator options
        gamma: float = 5.0 / 3.0
        nonlin_solver: NonlinearSolverParameters = None
        solver: OptsSolver = "pcg"
        precond: OptsPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None

        def __post_init__(self):
            # checks
            assert self.gamma > 1.0, f"gamma must be larger than 1, got {self.gamma}."
            check_option(self.solver, self.OptsSolver)
            check_option(self.precond, self.OptsPrecond)

            # defaults
            if self.nonlin_solver is None:
                self.nonlin_solver = NonlinearSolverParameters(tol=1e-10, maxiter=100)
            if self.solver_params is None:
                self.solver_params = SolverParameters(tol=1e-14)

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
        self._u_space = self.variables.u.space
        self._info = self.options.nonlin_solver.info and (self.rank == 0)
        u_form = self.derham.space_to_form[self._u_space]
        self._bop_u = self.derham.boundary_ops[u_form]

        # velocity mass matrix of the total density
        if self._u_space == "Hdiv":
            self._velocity_mass = VelocityMass(
                self.derham, self.domain, self.mass_ops, self.projected_equil, rho_field=self._rho_field
            )
            self._Mrho = self._velocity_mass.M_rho
        else:
            # assembly of WMMnew (total density) happens in VariationalDensityEvolve
            self._velocity_mass = None
            self._Mrho = self.mass_ops.WMMnew
        pc = None if self.options.precond is None else MassMatrixPreconditioner(self._Mrho, dim_reduce=None)
        self._Mrho_pc = pc
        sp = self.options.solver_params
        self._inv_Mrho = inverse(
            self._Mrho, self.options.solver, pc=pc, tol=sp.tol, maxiter=sp.maxiter, verbose=sp.verbose, recycle=True
        )

        # transport operators with the perturbations as weights
        self._Tw = Hdiv0_transport_operator(self.derham, u_space=self._u_space, phys_domain=self.domain)
        self._TwT = self._Tw.T
        self._Sp = Pressure_transport_operator(
            self.derham, self.domain, self.basis_ops.Uv, self.options.gamma, u_space=self._u_space
        )

        # temporaries
        u = self.variables.u.spline.vector
        b = self.variables.b.spline.vector
        p = self.variables.p.spline.vector
        self._un1, self._un12, self._du, self._fu, self._u_new = (u.space.zeros() for _ in range(5))
        self._bn1, self._bn12, self._db, self._M2b, self._b_new = (b.space.zeros() for _ in range(5))
        self._pn1, self._pn12, self._p_new = (p.space.zeros() for _ in range(3))
        self._Mdu = u.space.zeros()
        self._fu0 = u.space.zeros()
        self._M2db = b.space.zeros()

    def _energy_norm2(self, du, db) -> float:
        """du^T M_rho du + db^T M2 db (collective)."""
        return du.inner(self._Mrho.dot(du, out=self._Mdu)) + db.inner(self.mass_ops.M2.dot(db, out=self._M2db))

    @profile
    def __call__(self, dt):
        if self._velocity_mass is not None:
            self._velocity_mass.update()
        if self._Mrho_pc is not None:
            self._Mrho_pc.refresh()

        un = self.variables.u.spline.vector
        bn = self.variables.b.spline.vector
        pn = self.variables.p.spline.vector
        un1 = un.copy(out=self._un1)
        bn1 = bn.copy(out=self._bn1)
        pn1 = pn.copy(out=self._pn1)
        M2 = self.mass_ops.M2
        tol = self.options.nonlin_solver.tol
        maxiter = self.options.nonlin_solver.maxiter

        for it in range(maxiter):
            # midpoint weights of the current iterate
            bn12 = bn.copy(out=self._bn12)
            bn12 += bn1
            bn12 *= 0.5
            pn12 = pn.copy(out=self._pn12)
            pn12 += pn1
            pn12 *= 0.5
            self._Tw.update_coeffs(bn12)
            self._TwT.update_coeffs(bn12)
            self._Sp.update_coeffs(pn12)

            # velocity: M (u1 - un) = dt T[b12]^T M2 b12
            # (the boundary operator not in place: its dot with out=v zeroes v)
            f = self._TwT.dot(M2.dot(bn12, out=self._M2b), out=self._fu0)
            f = self._bop_u.dot(f, out=self._fu)
            u_new = self._inv_Mrho.dot(f, out=self._u_new)
            u_new *= dt
            u_new += un
            # midpoint velocity, and the magnetic field and the pressure transported with it
            un12 = un.copy(out=self._un12)
            un12 += u_new
            un12 *= 0.5
            b_new = self._Tw.dot(un12, out=self._b_new)
            b_new *= -dt
            b_new += bn
            p_new = self._Sp.dot(un12, out=self._p_new)
            p_new *= -dt
            p_new += pn

            # change of the iterate in the energy norm, relative to the change in the step
            du = u_new.copy(out=self._du)
            du -= un1
            db = b_new.copy(out=self._db)
            db -= bn1
            change = self._energy_norm2(du, db)
            u_new.copy(out=un1)
            b_new.copy(out=bn1)
            p_new.copy(out=pn1)
            du = un1.copy(out=self._du)
            du -= un
            db = bn1.copy(out=self._db)
            db -= bn
            step = self._energy_norm2(du, db)
            err = np.sqrt(change / step) if step > 0.0 else 0.0
            if err < tol:
                break
        else:
            if MPI.COMM_WORLD.Get_rank() == 0:
                logger.warning(
                    f"{self.__class__.__name__}: Picard iteration not converged in {maxiter} iterations "
                    f"(relative change {err:.2e})."
                )
        if self._info:
            logger.info(f"{self.__class__.__name__}: {it + 1} Picard iterations, relative change {err:.2e}")

        self.update_feec_variables(u=un1, b=bn1, p=pn1)
