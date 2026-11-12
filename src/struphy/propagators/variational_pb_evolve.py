import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import cunumpy as xp
from feectools.linalg.basic import IdentityOperator, ScaledLinearOperator
from feectools.linalg.block import BlockLinearOperator, BlockVectorSpace
from feectools.linalg.solvers import inverse
from line_profiler import profile
from maybempi import MPI

from struphy.feec import preconditioner
from struphy.feec.mass import L2Projector
from struphy.feec.preconditioner import MassMatrixPreconditioner
from struphy.feec.variational_utilities import (
    Hdiv0_transport_operator,
    NewtonConvergence,
    Pressure_transport_operator,
    PressureTransportAdjointMatrix,
    TransportAdjointMatrix,
    alfven_schur_preconditioner,
    dot_normalized,
    inv_norm2,
)
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.solver import NonlinearSolverParameters, SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.propagators.utils import VelocityMass
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class VariationalPBEvolve(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\mathbf u \in (H^1)^3`, :math:`p \in L^2` and :math:`\mathbf B \in H(\textnormal{div})` such that

    .. math::

        &\int_\Omega \partial_t (\rho \mathbf u) \cdot \mathbf v\,\textrm d \mathbf x - \int_\Omega \mathbf B \cdot \nabla \times (\tilde{\mathbf B} \times \mathbf v) - \int_\Omega \frac{1}{\gamma -1} (\nabla \cdot (\tilde{p} \mathbf v))\,\textrm d \mathbf x = 0 \qquad \forall \, \mathbf v \in (H^1)^3\,,
        \\[4mm]
        &\partial_t \mathbf B + \nabla \cdot ( \tilde{\mathbf B} \times \mathbf u ) = 0 \,,
        \\[4mm]
        &\partial_t p + \nabla \cdot(\tilde{p} \mathbf u) + (\gamma - 1) \tilde{p} \nabla \cdot u = 0 \,.

    Where :math:`\tilde{\mathbf B}` (resp. :math:`\tilde{p}`) is either :math:`\mathbf B` (resp. :math:`p`) for full-f models, :math:`\mathbf B_0` (resp. :math:`p_0`) for linear models or :math:`\mathbf B_0+\mathbf B` (resp. :math:`p_0+p`) for :math:`\delta f` models.

    On the logical domain:

    .. math::

        \begin{align}
        &\int_{\hat{\Omega}} \partial_t ( \hat{\rho}^3  \hat{\mathbf{u}}) \cdot G \hat{\mathbf{v}} \, \textrm d \boldsymbol \eta
        - \int_{\hat{\Omega}} \hat{\mathbf{B}}^2 \cdot G \,\nabla \times (\hat{\mathbf{B}}^2 \times \hat{\mathbf{v}}) \,\frac{1}{\sqrt g}\,
        - \frac{g}{\gamma -1} \nabla \cdot (\hat{p} \hat{v})  \textrm d \boldsymbol \eta = 0 ~ ,
        \\[2mm]
        &\partial_t \hat{\mathbf{B}}^2 + \nabla \times (\hat{\mathbf{B}}^2 \times \hat{\mathbf{u}}) = 0 ~ ,
        \\[2mm]
        &\partial_t \hat{p} + \nabla \cdot (\hat{p} \hat{u}) + (\gamma - 1 ) \hat{p} \nabla \cdot G \hat{u} = 0 \,
        \end{align}

    It is discretized as

    .. math::

        \begin{align}
        &\mathbb M^v[\hat{\rho}_h^{n}] \frac{ \mathbf u^{n+1}-\mathbf u^n}{\Delta t}
        - (\mathbb C \hat{\Pi}^{1}[\hat{\mathbf B_h^{n}} \cdot \vec{\boldsymbol \Lambda}^v])^\top \mathbb M^2 \mathbf B^{n+\frac{1}{2}}
        - (\mathbb D \hat{\Pi}^{2}[\hat{p_h^{n}} \cdot \vec{\boldsymbol \Lambda}^v])^\top \hat{l}^3(\frac{g}{\gamma-1}) = 0 ~ ,
        \\[2mm]
        &\frac{\mathbf b^{n+1}- \mathbf b^n}{\Delta t} + \mathbb C \hat{\Pi}^{1}[\hat{\mathbf B_h^{n}} \cdot \vec{\boldsymbol \Lambda}^v]] \mathbf u^{n+1/2} = 0 ~ ,
        \\[2mm]
        &\frac{\mathbf p^{n+1}- \mathbf p^n}{\Delta t} + \big(\mathbb D \hat{\Pi}^{2}[\hat{p_h^{n}} \cdot \vec{\boldsymbol \Lambda}^v]]
        + (\gamma - 1)\hat{\Pi}^{3}[\hat{p_h^{n}} \cdot \vec{\boldsymbol \Lambda}^3] \mathbb D \mathcal{U}^v \big) \mathbf u^{n+1/2}= 0 ~ ,
        \\[2mm]
        \end{align}

    with

    .. math::

        \hat{l}^3(f)_{ijk}=\int_{\hat{\Omega}} f \Lambda^3_{ijk} \textrm d \boldsymbol \eta

    where weights in the the :class:`~struphy.feec.basis_projection_ops.BasisProjectionOperator` and the :class:`~struphy.feec.mass.WeightedMassOperator` are given by

    .. math::

        \hat{\mathbf{B}}_h^{n} = (\mathbf{b}^{n})^\top \vec{\boldsymbol \Lambda}^2 \in V_h^2 \,
        \qquad \hat{\rho}_h^{n} = (\boldsymbol \rho^{n})^\top \vec{\boldsymbol \Lambda}^3 \in V_h^3 \,
        \qquad \hat{p}_h^{n} = (\boldsymbol p^{n})^\top \vec{\boldsymbol \Lambda}^3 \in V_h^3 \,.

    and :math:`\mathcal{U}^v` is :class:`~struphy.feec.basis_projection_ops.BasisProjectionOperators`.

    The weights of the transport operators are taken at :math:`t^n` (in all equations, which conserves the
    energy); the step is therefore only first-order accurate in time.

    **Velocity in** :math:`H(\textnormal{div})` (``model="perturbative"``): :math:`\mathbf u` is a 2-form with vanishing
    normal component at the boundary. The scheme is the same, with the mass matrix of the total density
    :math:`\mathbb M^2_\rho` (:class:`~struphy.propagators.utils.VelocityMass` with the constructor argument ``rho``,
    assembled at the beginning of the step) instead of :math:`\mathbb M^v[\hat\rho_h^n]`, and the transport operators
    :class:`~struphy.feec.variational_utilities.Hdiv0_transport_operator` and
    :class:`~struphy.feec.variational_utilities.Pressure_transport_operator` with ``u_space="Hdiv"``
    (:math:`\hat\Pi^1[(\hat{\mathbf B}^2 \times \vec{\boldsymbol\Lambda}^2)/\sqrt g]`,
    :math:`\hat\Pi^2[\hat p^3 \vec{\boldsymbol\Lambda}^2/\sqrt g]` and :math:`\mathbb D` without
    :math:`\mathcal U^v`). Not available with polar splines.

    **Perturbative model** (``model="perturbative"``): the conserved energy is
    :math:`\frac12 \mathbf u^\top \mathbb M^v_\rho \mathbf u + \frac12 \mathbf b^\top \mathbb M^2 \mathbf b
    + \mathbf b_0^\top \mathbb M^2 \hat{\mathbf b} + \int \hat p / (\gamma - 1)`, the expansion of the energy of the full
    model (Galerkin Hodge star :math:`\mathbb M^2` throughout) without the terms linear in the perturbation, whose forces
    (the equilibrium force) are dropped. The auxiliary second-order variables :math:`\hat p` (``pt3``) and
    :math:`\hat{\mathbf b}` (``bt2``) are advanced with the part of the transport of :math:`\mathbf p` and :math:`\mathbf b`
    that has the perturbations :math:`\mathbf p^w`, :math:`\mathbf b^w` as weights; the forces are the adjoints of these
    transports. The equilibrium transport (weights :math:`\mathbf B_0`, :math:`p_0`) is as above, and

    .. math::

        \begin{align}
        &\mathbb M^v_\rho \frac{\mathbf u^{n+1}-\mathbf u^n}{\Delta t}
        - (\mathbb T[\mathbf B_0] + \mathbb T[\mathbf b^w])^\top \mathbb M^2 \mathbf b^{n+\frac12}
        - \mathbb T[\mathbf b^w]^\top \mathbb M^2 \mathbf b_0
        - \mathbb S[\mathbf p^w]^\top \hat l^3(\tfrac{g}{\gamma-1}) = 0 \,,
        \\[2mm]
        &\frac{\mathbf b^{n+1}-\mathbf b^n}{\Delta t} + (\mathbb T[\mathbf B_0] + \mathbb T[\mathbf b^w])
        \mathbf u^{n+\frac12} = 0 \,, \qquad
        \frac{\mathbf p^{n+1}-\mathbf p^n}{\Delta t} + (\mathbb S[p_0] + \mathbb S[\mathbf p^w])
        \mathbf u^{n+\frac12} = 0 \,,
        \\[2mm]
        &\frac{\hat{\mathbf b}^{n+1}-\hat{\mathbf b}^n}{\Delta t} + \mathbb T[\mathbf b^w] \mathbf u^{n+\frac12} = 0
        \,, \qquad
        \frac{\hat{\mathbf p}^{n+1}-\hat{\mathbf p}^n}{\Delta t} + \mathbb S[\mathbf p^w] \mathbf u^{n+\frac12} = 0 \,,
        \end{align}

    with :math:`\mathbb T[\bar{\mathbf B}] = \mathbb C \hat\Pi^1[\hat{\bar{\mathbf B}} \times
    \vec{\boldsymbol\Lambda}^v]` and :math:`\mathbb S[\bar p]` the pressure transport above. The second-order magnetic
    term :math:`\mathbf b_0^\top \mathbb M^2 \hat{\mathbf b}` has no wall defect: :math:`\hat{\mathbf b} = \mathbb C
    \hat{\mathbf a}` with :math:`\hat{\mathbf a}` the sum of the projections :math:`\hat\Pi^1[\hat{\mathbf b}^w \times
    \mathbf u]`, which have zero tangential trace (the degrees of freedom on a wall edge integrate the tangential
    component of :math:`\mathbf B^w \times \mathbf u`, zero for :math:`\mathbf B^w \cdot \mathbf n = \mathbf u \cdot
    \mathbf n = 0`), so :math:`\mathbf b_0^\top \mathbb M^2 \mathbb C \hat{\mathbf a} = \hat{\mathbf a}^\top
    \mathbb C^\top \mathbb M^2 \mathbf b_0` is the weak curl of :math:`\mathbf b_0` tested only against fields of zero
    tangential trace, on which it is exact; the current is never used as a field. The energy is conserved for any
    weights. The option ``weights`` sets them:

    - ``"midpoint"`` (default): :math:`\mathbf b^w = \mathbf b^{n+\frac12}`, :math:`\mathbf p^w =
      \mathbf p^{n+\frac12}`, the implicit midpoint rule. At small amplitude the step is that of the linear model with
      midpoint weights
      (:class:`~struphy.propagators.variational_linear_alfven_evolve.VariationalLinearAlfvenEvolve` and
      :class:`~struphy.propagators.variational_linear_sound_evolve.VariationalLinearSoundEvolve`, which use the same
      operators, up to the splitting): sound waves and second-order terms
      are implicit. The Newton matrix contains the derivatives of these linear terms,
      :math:`\mathbb M^v_\rho - \frac{\Delta t^2}{4} \mathbb A\, \mathbb S[p_0]` in the upper left block, with the
      matrix :math:`\mathbb A` of the pressure force :math:`\mathbf p \mapsto \mathbb S[\mathbf p]^\top \hat{\mathbf l}`
      (:class:`~struphy.feec.variational_utilities.PressureTransportAdjointMatrix`), and :math:`\frac{\Delta t}{2}
      \mathbb F` in the upper right block, the exact derivative of the second-order magnetic force
      :math:`\mathbb T[\mathbf b]^\top \mathbb M^2 \mathbf b_0 = -\mathbb F \mathbf b`, which is linear in
      :math:`\mathbf b` (:class:`~struphy.feec.variational_utilities.TransportAdjointMatrix`); it is not symmetric
      (``solver="rpbicgstab"``).
      The weights are updated in every iteration, which stops when also the pressure has converged.
    - ``"frozen"``: :math:`\mathbf b^w = \mathbf b^n`, :math:`\mathbf p^w = \mathbf p^n`. The pressure force and the
      force of the equilibrium current are then explicit, and sound waves grow for :math:`\Delta t\, \omega_s
      \gtrsim 1`.
    """

    class Variables:
        """Container for variables advanced by :class:`VariationalPBEvolve`.

        Attributes
        ----------
        p : FEECVariable
            Pressure variable in ``"L2"`` space.
        u : FEECVariable
            Velocity variable in ``"H1vec"`` space, or in ``"Hdiv"`` space for ``model="perturbative"``.
        b : FEECVariable
            Magnetic-field variable in ``"Hdiv"`` space.
        """

        def __init__(self):
            self._p: FEECVariable = None
            self._u: FEECVariable = None
            self._b: FEECVariable = None

        @property
        def p(self) -> FEECVariable:
            return self._p

        @p.setter
        def p(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "L2"
            self._p = new

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

    def __init__(
        self,
        div_u: FEECVariable = None,
        u2: FEECVariable = None,
        pt3: FEECVariable = None,
        bt2: FEECVariable = None,
        rho: FEECVariable = None,
    ):
        r"""
        Parameters
        ----------
        div_u : FEECVariable, default=None
            Divergence diagnostic variable (``"L2"`` space) updated by the velocity field.
        u2 : FEECVariable, default=None
            Velocity 2-form diagnostic (``"Hdiv"`` space).
        pt3 : FEECVariable, default=None
            Auxiliary second-order pressure :math:`p_2` (``"L2"`` space) of the linear and perturbative models,
            advected by the pressure perturbation. It enters only the energy, not the momentum equation.
        bt2 : FEECVariable, default=None
            Auxiliary second-order magnetic field :math:`\mathbf B_2` (``"Hdiv"`` space) of the linear and perturbative
            models, advected by the magnetic perturbation. It enters only the energy, not the momentum equation.
        rho : FEECVariable, default=None
            Density perturbation :math:`\tilde\rho` (``"L2"`` space, not updated) of the mass matrix
            :math:`\mathbb M^2_\rho` (total density :math:`\rho_0 + \tilde\rho`) of a velocity in ``"Hdiv"``;
            None for :math:`\mathbb M^2_{\rho_0}`. Not used for a velocity in ``"H1vec"``.
        """
        assert rho is None or (isinstance(rho, FEECVariable) and rho.space == "L2")
        self._rho = rho
        self.variables = self.Variables()
        self.div_u = div_u
        self.u2 = u2
        self.pt3 = pt3
        self.bt2 = bt2

    @dataclass(repr=False)
    class Options(OptionsBase):
        r"""Configuration options for :class:`VariationalPBEvolve`.

        Parameters
        ----------
        model : {"full_p", "linear", "perturbative"}, default="full_p"
            Pressure-magnetic coupled model variant.
        gamma : float, default=5/3
            Adiabatic index.
        weights : {"frozen", "midpoint"}, default=None
            Weights of the terms with the perturbations of ``model="perturbative"`` (see above): at :math:`t^n`
            (``"frozen"``) or at :math:`t^{n+1/2}` (``"midpoint"``). None gives ``"midpoint"`` for
            ``"perturbative"`` and ``"frozen"`` otherwise (the only choice for the other models).
        solver : {"pcg", "pcg_sr", "cg", "pbicgstab", "rpbicgstab"}, default=None
            Linear solver of the Newton system; None gives ``"rpbicgstab"`` for midpoint weights (not symmetric) and
            ``"pcg"`` otherwise.
        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner used in linear solves.
        solver_params : SolverParameters, default=None
            Linear-solver controls. The Newton systems are solved for a normalized right-hand side, so ``tol`` is
            relative to the norm of the residual.
        nonlin_solver : NonlinearSolverParameters, default=None
            Nonlinear iteration controls. The iteration stops when the norm of the residual is below ``tol`` times the
            norm of the state at :math:`t^n` if that is smaller than one (absolute otherwise), or when it
            stagnates (with a warning).
        schur_precond : {"MassMatrixPreconditioner", "StiffnessPreconditioner"}, default="MassMatrixPreconditioner"
            Preconditioner of the Schur system of the Newton step: the mass-matrix preconditioner of the velocity
            block, or the Kronecker (fast diagonalization) preconditioner of the whole shear-Alfven Schur operator
            :math:`\mathbb M^v[\rho_0] + \frac{\Delta t^2}{4} \mathbb T_0^\top \mathbb M^2 \mathbb T_0`
            (:func:`~struphy.feec.variational_utilities.alfven_schur_preconditioner`, built from the equilibrium for
            each time step in :meth:`prepare_time_step`; for ``model="perturbative"`` it approximates the Schur
            operator of the Newton step). The latter needs ``model="linear"`` or ``"perturbative"``, ``solver="pcg"``,
            tensor-product splines and the global commuting projectors; otherwise the mass-matrix preconditioner is
            used (with a warning).
        exb_correction : bool, default=True
            For ``schur_precond="StiffnessPreconditioner"``: add the auxiliary space correction of the
            :math:`\mathbf E \times \mathbf B` flows :math:`\mathbf B_0 \times \nabla \phi` (the incompressible shear
            Alfven modes with small parallel wave number, which block Jacobi does not reduce).
        """

        # specific literals
        OptsSchurPrecond = Literal["MassMatrixPreconditioner", "StiffnessPreconditioner"]
        OptsModel = Literal["full_p", "linear", "perturbative"]
        OptsWeights = Literal["frozen", "midpoint"]
        OptsSolver = Literal["pcg", "pcg_sr", "cg", "pbicgstab", "rpbicgstab"]
        # propagator options
        model: OptsModel = "full_p"
        gamma: float = 5.0 / 3.0
        weights: OptsWeights = None
        solver: OptsSolver = None
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        nonlin_solver: NonlinearSolverParameters = None
        schur_precond: OptsSchurPrecond = "MassMatrixPreconditioner"
        exb_correction: bool = True

        def __post_init__(self):
            # checks
            check_option(self.model, self.OptsModel)
            if self.weights is None:
                self.weights = "midpoint" if self.model == "perturbative" else "frozen"
            check_option(self.weights, self.OptsWeights)
            assert self.model == "perturbative" or self.weights == "frozen", (
                'weights="midpoint" is only available for model="perturbative".'
            )
            if self.solver is None:
                self.solver = "rpbicgstab" if self.weights == "midpoint" else "pcg"
            check_option(self.solver, self.OptsSolver)
            assert self.weights == "frozen" or self.solver in ("pbicgstab", "rpbicgstab"), (
                f'weights="midpoint" needs solver="rpbicgstab" or "pbicgstab" (not symmetric), got {self.solver!r}.'
            )
            check_option(self.precond, LiteralOptions.OptsMassPrecond)
            check_option(self.schur_precond, self.OptsSchurPrecond)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

            if self.nonlin_solver is None:
                self.nonlin_solver = NonlinearSolverParameters()

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
        self._model = self.options.model
        self._lin_solver = self.options.solver_params
        self._nonlin_solver = self.options.nonlin_solver
        self._linearize = self.options.nonlin_solver.linearize
        self._gamma = self.options.gamma
        self._midpoint = self.options.weights == "midpoint"

        if self.div_u is None:
            self._divu = None
        else:
            self._divu = self.div_u.spline.vector

        if self.u2 is None:
            self._u2 = None
        else:
            self._u2 = self.u2.spline.vector

        if self.pt3 is None:
            self._pt3 = None
        else:
            self._pt3 = self.pt3.spline.vector

        if self.bt2 is None:
            self._bt2 = None
        else:
            self._bt2 = self.bt2.spline.vector

        self._info = self._nonlin_solver.info and (MPI.COMM_WORLD.Get_rank() == 0)

        self._u_space = self.variables.u.space
        self._u_form = self.derham.space_to_form[self._u_space]
        if self._u_space == "Hdiv":
            assert self._model == "perturbative", (
                f'A velocity in "Hdiv" needs model="perturbative", got {self._model!r}.'
            )
            assert not self.derham.polar_splines, 'A velocity in "Hdiv" is not available with polar splines.'
            # mass matrix of the total density, assembled at the beginning of each step
            self._velocity_mass = VelocityMass(
                self.derham, self.domain, self.mass_ops, self.projected_equil, rho_field=self._rho
            )
            self._Mrho = self._velocity_mass.M_rho
        else:
            # assembly of WMMnew happens in VariationalDensityEvolve
            self._Mrho = self.mass_ops.WMMnew
        self._Mrho_pc = MassMatrixPreconditioner(self._Mrho, dim_reduce=None)

        # Projector
        self._initialize_projectors_and_mass()

        # Kronecker preconditioner of the Schur operator, built for the time step in prepare_time_step
        # built from the equilibrium: exact for the Schur operator of the linear model, an approximation (still a fixed
        # linear preconditioner) for the perturbative one
        unmet = [] if self._model in ("linear", "perturbative") else ['model="linear" or "perturbative"']
        if self.derham.polar_splines:
            unmet += ["tensor-product splines"]
        self._schur_pc = alfven_schur_preconditioner(
            self,
            unmet,
            u_space=self._u_space,
        )
        self._schur_pc_dt = None

        # bunch of temporaries to avoid allocating in the loop
        u = self.variables.u.spline.vector
        p = self.variables.p.spline.vector
        b = self.variables.b.spline.vector

        self._tmp_un1 = u.space.zeros()
        self._tmp_un2 = u.space.zeros()
        self._tmp_un12 = u.space.zeros()
        self._tmp_bn1 = b.space.zeros()
        self._tmp_bn12 = b.space.zeros()
        self._tmp_un_diff = u.space.zeros()
        self._tmp_bn_diff = b.space.zeros()
        self._tmp_un_weak_diff = u.space.zeros()
        self._tmp_bn_weak_diff = b.space.zeros()
        self._tmp_pn1 = p.space.zeros()
        self._tmp_pn2 = p.space.zeros()
        self._tmp_pn_diff = p.space.zeros()
        self._tmp_pn_weak_diff = p.space.zeros()
        self._linear_form_dl_dp = p.space.zeros()
        self._tmp_pn12 = p.space.zeros()
        self._tmp_mn = u.space.zeros()
        self._tmp_mn1 = u.space.zeros()
        self._tmp_mn_diff = u.space.zeros()
        self._tmp_advection = u.space.zeros()
        self._tmp_advection2 = u.space.zeros()
        self._tmp_b_advection = b.space.zeros()
        self._tmp_b_advection2 = b.space.zeros()
        self._linear_form_dl_db = b.space.zeros()

        self._tmp_p_advection = p.space.zeros()
        self._tmp_p_advection2 = p.space.zeros()
        self._tmp_bw = b.space.zeros()
        self._tmp_pw = p.space.zeros()
        self._tmp_pn1_old = p.space.zeros()

        self._create_linear_form_p()

        if self._linearize:
            self._extracted_b2 = self.projected_equil.b2

    def prepare_time_step(self, dt: float):
        """Kronecker preconditioner of the Schur operator for the time step ``dt`` (with
        ``schur_precond="StiffnessPreconditioner"``): built in the setup, rebuilt when dt changes."""
        if self._schur_pc is not None and dt != self._schur_pc_dt:
            self._inv_Jacobian.precond = self._schur_pc.build(dt)
            self._schur_pc_dt = dt

    def __call__(self, dt):
        # preconditioner for the current time step (no-op unless dt changed since the setup)
        self.prepare_time_step(dt)
        if self._u_space == "Hdiv":
            self._velocity_mass.update()
        # WMMnew is re-assembled in VariationalDensityEvolve: update the diagonal scaling of its preconditioner
        self._Mrho_pc.refresh()

        if self._nonlin_solver.type == "Picard":
            self.__call_picard(dt)
        else:
            raise ValueError("Only Picard solver is implemented for VariationalPBEvolve")

    def __call_picard(self, dt):
        """Solve the non linear system for updating the variables using Newton iteration method"""
        # In fact it is linear due to the explicit update, only one iteration will be done at each time step
        if self._info:
            logger.info("")
            logger.info("Newton iteration in VariationalPBEvolve")

        un = self.variables.u.spline.vector
        pn = self.variables.p.spline.vector
        bn = self.variables.b.spline.vector

        self._update_Pib(bn)
        self._update_Projp(pn)
        perturbative = self._model == "perturbative"
        # weights of the terms with the perturbation: b^n, p^n, or the midpoint values of the current iterate
        bw, pw = bn, pn

        mn = self._Mrho.dot(un, out=self._tmp_mn)
        bn1 = bn.copy(out=self._tmp_bn1)
        bn1 += self._tmp_bn_diff
        pn1 = pn.copy(out=self._tmp_pn1)
        pn1 += self._tmp_pn_diff
        un1 = un.copy(out=self._tmp_un1)
        un1 += self._tmp_un_diff
        mn1 = self._Mrho.dot(un1, out=self._tmp_mn1)
        tol = self._nonlin_solver.tol
        # stopping criterion relative to the state at t^n, with a check for stagnation
        conv = NewtonConvergence("VariationalPBEvolve", tol, self._get_error(mn, bn))

        for it in range(self._nonlin_solver.maxiter):
            # Picard iteration
            # half time step approximation

            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5

            # pn12 = pn.copy(out=self._tmp_pn12)
            # pn12 += pn1
            # pn12 *= 0.5

            if perturbative and self._midpoint:
                bw = bn.copy(out=self._tmp_bw)
                bw += bn1
                bw *= 0.5
                self._update_Pib(bw)
                # pressure at the current velocity iterate (the nonlinear transport with the weight of the previous
                # iterate), so that the pressure force follows the velocity as assumed in the Newton matrix
                p_advection = self._transop_p0.dot(un12, out=self._tmp_p_advection)
                p_advection += self._transop_p.dot(un12, out=self._tmp_p_advection2)
                p_advection *= dt
                pn1 = pn.copy(out=self._tmp_pn1)
                pn1 -= p_advection
                pw = pn.copy(out=self._tmp_pw)
                pw += pn1
                pw *= 0.5
                self._update_Projp(pw)
            # Update the linear form
            self._update_linear_form_dl_db(bn, bn1)

            # Compute the advection terms
            if self._model == "linear":
                advection = self.curlPibT0.dot(
                    self._linear_form_dl_db,
                    out=self._tmp_advection,
                )

                advection += self.curlPibT.dot(
                    self._linear_form_dl_db0,
                    out=self._tmp_advection2,
                )

                advection += self._transop_pT.dot(self._linear_form_dl_dp, out=self._tmp_advection2)

                b_advection = self.curlPib0.dot(
                    un12,
                    out=self._tmp_b_advection,
                )

                p_advection = self._transop_p0.dot(
                    un12,
                    out=self._tmp_p_advection,
                )

            elif self._model == "perturbative":
                advection = self.curlPibT0.dot(
                    self._linear_form_dl_db,
                    out=self._tmp_advection,
                )

                # second-order magnetic term b0^T M2 b-hat: the force is the adjoint of the transport of b-hat
                advection += self.curlPibT.dot(self._linear_form_dl_db0, out=self._tmp_advection2)

                advection += self.curlPibT.dot(
                    self._linear_form_dl_db,
                    out=self._tmp_advection2,
                )

                # pressure force: adjoint of the transport S[p^w] of the perturbation (energy of the auxiliary pressure)
                advection += self._transop_pT.dot(self._linear_form_dl_dp, out=self._tmp_advection2)

                b_advection = self.curlPib.dot(
                    un12,
                    out=self._tmp_b_advection,
                )

                b_advection += self.curlPib0.dot(
                    un12,
                    out=self._tmp_b_advection2,
                )

                p_advection = self._transop_p0.dot(
                    un12,
                    out=self._tmp_p_advection,
                )

                p_advection += self._transop_p.dot(
                    un12,
                    out=self._tmp_p_advection2,
                )

            else:
                advection = self.curlPibT.dot(
                    self._linear_form_dl_db,
                    out=self._tmp_advection,
                )

                advection2 = self._transop_pT.dot(self._linear_form_dl_dp, out=self._tmp_advection2)

                advection += advection2

                b_advection = self.curlPib.dot(
                    un12,
                    out=self._tmp_b_advection,
                )

                p_advection = self._transop_p.dot(
                    un12,
                    out=self._tmp_p_advection,
                )

            advection *= dt
            b_advection *= dt
            p_advection *= dt

            # Get diff
            bn_diff = bn1.copy(out=self._tmp_bn_diff)
            bn_diff -= bn
            bn_diff += b_advection

            # pn_diff = pn1.copy(out= self._tmp_pn_diff)
            # pn_diff -= pn
            # pn_diff += p_advection

            mn_diff = mn1.copy(out=self._tmp_mn_diff)
            mn_diff -= mn
            mn_diff += advection

            if perturbative and self._midpoint:
                pn1.copy(out=self._tmp_pn1_old)
            pn1 = pn.copy(out=self._tmp_pn1)
            pn1 -= p_advection

            # Get error (with midpoint weights also the change of the pressure, which is a weight)
            err = self._get_error(mn_diff, bn_diff)
            if perturbative and self._midpoint:
                dp = self._tmp_pn1_old
                dp -= pn1
                err = max(err, self.mass_ops.M3.dot_inner(dp, dp))
            stop = conv.stop(err, it)

            if self._info:
                logger.info(f"iteration : {it} relative error : {conv.err}")

            if stop:
                break

            # Derivative for Newton
            self._get_jacobian(dt)

            # Newton step
            self._tmp_f[0] = mn_diff
            self._tmp_f[1] = bn_diff

            # normalized right-hand side: the tolerance of the linear solver is relative to the residual
            incr = dot_normalized(self._inv_Jacobian, self._tmp_f, self._tmp_incr)
            if self._info:
                logger.info(f"information on the linear solver : {self._inv_Jacobian._solver._info}")
            un1 -= incr[0]
            bn1 -= incr[1]

            # Multiply by the mass matrix to get the momentum
            mn1 = self._Mrho.dot(un1, out=self._tmp_mn1)

        conv.warn_if_not_converged(it, self._nonlin_solver.maxiter)

        if not conv.converged:
            # the last iteration applied an update after the residual: midpoint velocity of the final iterate
            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5

        self._tmp_un_diff = un1 - un
        self._tmp_bn_diff = bn1 - bn
        self._tmp_pn_diff = pn1 - pn
        self.update_feec_variables(p=pn1, b=bn1, u=un1)

        self._transop_p.div.dot(un12, out=self._divu)
        self._transop_p._Uv.dot(un1, out=self._u2)

        # Update the 2nd order variables

        if self._pt3 is not None:
            # the part of the pressure transport with the perturbation as weight (p^n or p^{n+1/2})
            p_advection = self._transop_p.dot(
                un12,
                out=self._tmp_p_advection,
            )
            p_advection *= dt
            self._pt3 -= p_advection

        if self._bt2 is not None:
            b_advection = self.curlPib.dot(
                un12,
                out=self._tmp_b_advection,
            )
            b_advection *= dt
            self._bt2 -= b_advection

    def _initialize_projectors_and_mass(self):
        """Initialization of all the `BasisProjectionOperator` and needed to compute the bracket term"""

        self.curlPib = Hdiv0_transport_operator(self.derham, u_space=self._u_space, phys_domain=self.domain)
        self.curlPibT = self.curlPib.T
        self._transop_p = Pressure_transport_operator(
            self.derham, self.domain, self.basis_ops.Uv, self._gamma, u_space=self._u_space
        )
        self._transop_pT = self._transop_p.T

        integration_grid = [grid_1d.flatten() for grid_1d in self.derham.V3splines.quad_grid_pts[0]]

        self.integration_grid_spans, self.integration_grid_bn, self.integration_grid_bd = (
            self.derham.prepare_eval_tp_fixed(
                integration_grid,
            )
        )

        grid_shape = tuple([len(loc_grid) for loc_grid in integration_grid])

        self._tmp_int_grid = xp.zeros(grid_shape, dtype=float)

        # Inverse mass matrix of the velocity space (Mv or M2) needed to compute the error
        M_u = self.mass_ops.M2 if self._u_space == "Hdiv" else self.mass_ops.Mv
        self.pc_Mv = preconditioner.MassMatrixPreconditioner(
            M_u,
            dim_reduce=None,
        )
        self._inv_Mv = inverse(
            M_u,
            "pcg",
            pc=self.pc_Mv,
            tol=1e-10,
            maxiter=1000,
            verbose=False,
        )

        self._I2 = IdentityOperator(self.derham.V2pol)

        Jacs = BlockVectorSpace(
            getattr(self.derham, f"V{self._u_form}pol"),
            self.derham.V2pol,
        )

        self._tmp_f = Jacs.zeros()
        self._tmp_incr = Jacs.zeros()

        self._Jacobian = BlockLinearOperator(Jacs, Jacs)

        if self._model == "linear":
            # initialize the jacobian differently if linear model
            self._create_Pib0()
            self._create_transop0()

            self._linear_form_dl_db0 = -self.mass_ops.M2.dot(self.projected_equil.b2)

            self._mdt2_pc_curlPibT_M = 2 * (self.curlPibT0 @ self.mass_ops.M2)
            self._dt2_curlPib = 2 * self.curlPib0

        elif self._model == "perturbative":
            # initialize the jacobian differently if linear model
            self._create_Pib0()
            self._create_transop0()

            self._full_curlPib = self.curlPib0 + self.curlPib
            self._full_curlPibT = self.curlPibT0 + self.curlPibT

            # second-order magnetic term b0^T M2 b-hat (Galerkin Hodge star, as the full-f energy): its force is the adjoint
            # of the transport of b-hat applied to this linear form
            self._linear_form_dl_db0 = -self.mass_ops.M2.dot(self.projected_equil.b2)

            self._mdt2_pc_curlPibT_M = 2 * (self._full_curlPibT @ self.mass_ops.M2)
            self._dt2_curlPib = 2 * self._full_curlPib
            if self._midpoint:
                # exact derivatives of the linear terms with midpoint weights: the second-order magnetic force in the
                # upper right block (the matrix F of b -> T[b]^T l_b, linear in b) and the sound waves in the upper left
                # block, -A S[p0] with the matrix A of p -> S[p]^T l_p (the pressure force)
                F = TransportAdjointMatrix(
                    self.derham, self.domain, self.mass_ops, self._u_space, self._linear_form_dl_db0
                )
                self._dt2_R = ScaledLinearOperator(F.domain, F.codomain, 1.0, F)
                self._J01 = self._mdt2_pc_curlPibT_M + self._dt2_R
                A = PressureTransportAdjointMatrix(
                    self.derham,
                    self.domain,
                    self.mass_ops,
                    self.basis_ops,
                    self._u_space,
                    self._gamma,
                    # the linear form of int p-hat / (gamma - 1) (as _create_linear_form_p, which runs later)
                    L2Projector("L2", self.mass_ops).get_dofs(self.domain.jacobian_det) * (-1.0 / (self._gamma - 1.0)),
                )
                # the sign inside the composition: the outer ScaledLinearOperator absorbs an outer scalar, and its scalar
                # is set to dt^2/4 in _get_jacobian
                sound = (-1.0 * A) @ self._transop_p0
                self._dt2_sound = ScaledLinearOperator(sound.domain, sound.codomain, 1.0, sound)
                self._J00 = self._Mrho + self._dt2_sound

        else:
            self._mdt2_pc_curlPibT_M = 2 * (self.curlPibT @ self.mass_ops.M2)
            self._dt2_curlPib = 2 * self.curlPib

        self._get_L2dofs_V3 = L2Projector("L2", self.mass_ops).get_dofs
        metric = self.domain.jacobian_det(
            *integration_grid,
        )

        self._energy_metric_term = deepcopy(metric)

        # local version to avoid creating new version of LinearOperator every time

        midpoint = self._model == "perturbative" and self._midpoint
        self._Jacobian[0, 0] = self._J00 if midpoint else self._Mrho
        self._Jacobian[0, 1] = self._J01 if midpoint else self._mdt2_pc_curlPibT_M
        self._Jacobian[1, 0] = self._dt2_curlPib
        self._Jacobian[1, 1] = self._I2

        from struphy.linear_algebra.schur_solver import SchurSolverFull

        self._inv_Jacobian = SchurSolverFull(
            self._Jacobian,
            self.options.solver,
            # fixed (linear) preconditioner: an inner iterative solve is expensive and not a linear operator
            pc=self._Mrho_pc,
            tol=self._lin_solver.tol,
            maxiter=self._lin_solver.maxiter,
            verbose=self._lin_solver.verbose,
            recycle=True,
        )

        # self._inv_Jacobian = inverse(self._Jacobian,
        #                          'gmres',
        #                          tol=self._lin_solver['tol'],
        #                          maxiter=self._lin_solver['maxiter'],
        #                          verbose=self._lin_solver['verbose'],
        #
        #                     recycle=True)

    def _update_Pib(self, b):
        """Update the weights of the `BasisProjectionOperator`"""

        self.curlPib.update_coeffs(b)
        self.curlPibT.update_coeffs(b)

    def _create_Pib0(self):
        self.curlPib0 = Hdiv0_transport_operator(self.derham, u_space=self._u_space, phys_domain=self.domain)
        self.curlPibT0 = self.curlPib0.T
        self.curlPib0.update_coeffs(self.projected_equil.b2)
        self.curlPibT0.update_coeffs(self.projected_equil.b2)

    def _update_Projp(self, p):
        """Update the weights of the `BasisProjectionOperator`"""
        self._transop_p.update_coeffs(p)
        self._transop_pT.update_coeffs(p)

    def _create_transop0(self):
        """Update the weights of the `BasisProjectionOperator`"""

        self._transop_p0 = Pressure_transport_operator(
            self.derham, self.domain, self.basis_ops.Uv, self._gamma, u_space=self._u_space
        )
        self._transop_p0T = self._transop_p0.T
        self._transop_p0.update_coeffs(self.projected_equil.p3)
        self._transop_p0T.update_coeffs(self.projected_equil.p3)

    def _update_linear_form_dl_db(self, bn, bn1):
        """Update the linearform representing integration in V2 derivative of the lagrangian"""
        bn12 = bn.copy(out=self._tmp_bn12)
        bn12 += bn1
        bn12 *= 0.5
        if self._linearize:
            wb = self.mass_ops.M2.dot(bn12 - self._extracted_b2, out=self._linear_form_dl_db)
        else:
            wb = self.mass_ops.M2.dot(bn12, out=self._linear_form_dl_db)
        wb *= -1

    def _create_linear_form_p(self):
        """Update the linearform representing integration in V3 against pressure energy"""

        if self._model in ["full_p", "linear", "perturbative"]:
            self._tmp_int_grid *= 0.0
            self._tmp_int_grid -= 1.0 / (self._gamma - 1.0)
            self._tmp_int_grid *= self._energy_metric_term

        self._get_L2dofs_V3(self._tmp_int_grid, dofs=self._linear_form_dl_dp)

    def _get_error(self, mn_diff, bn_diff):  # , pn_diff):
        # the inverse mass matrix is applied to the normalized residual (relative solver tolerance)
        err_u = inv_norm2(self._inv_Mv, self.derham.boundary_ops[self._u_form].dot(mn_diff))
        err_b = self.mass_ops.M2.dot_inner(bn_diff, bn_diff)
        # weak_pn_diff = self.mass_ops.M3.dot(
        #     pn_diff,
        #     out=self._tmp_pn_weak_diff,
        # )
        # err_p = weak_pn_diff.dot(pn_diff)
        # logger.info("err_b :"+str(err_b))
        # logger.info("err_p :"+str(err_p))
        # logger.info("err_u :"+str(err_u))
        return max(err_b, err_u)
        # return max(max(err_b, err_u),err_p)

    def _get_jacobian(self, dt):
        self._mdt2_pc_curlPibT_M._scalar = -dt / 2
        self._dt2_curlPib._scalar = dt / 2
        if self._model == "perturbative" and self._midpoint:
            self._dt2_R._scalar = dt / 2
            self._dt2_sound._scalar = dt**2 / 4
