"""Utilities shared by several propagators (called explicitly from the propagators, no mixin classes).

Operators of a velocity 2-form weighted with the total density :math:`\\rho = \\rho_0 + \\tilde \\rho` of the
perturbative models (density 3-form evolving in time), on the quadrature grid of the mass matrices
(:class:`HdivDensityOperators`, :class:`VelocityMass`), and the second-order magnetic term of the linear and perturbative
variational models with the equilibrium current of the dual complex (:class:`EquilibriumCurrentTerm`) and their
second-order pressure term in weak form (:class:`WeakPressureTerm`)."""

import logging

import cunumpy as xp
from feectools.linalg.basic import Vector
from feectools.linalg.solvers import inverse
from maybempi import MPI

from struphy.feec.mass import L2Projector, WeightedMassOperators
from struphy.feec.preconditioner import MassMatrixPreconditioner
from struphy.feec.psydac_derham import Derham
from struphy.geometry.base import Domain
from struphy.polar.basic import PolarVector

logger = logging.getLogger("struphy")


class HdivDensityOperators:
    r"""Operators of a velocity :math:`\mathbf u \in H(\textnormal{div})` (2-form coefficients) weighted with a density
    3-form :math:`\hat \rho^3 = \sum_{ijk} \rho_{ijk} \Lambda^3_{ijk}` that changes in time:

    * the mass matrix :attr:`M_rho` (:meth:`update_M_rho`),

      .. math::

          \mathbb M^2_\rho = \int \hat \rho^3 \, \vec{\boldsymbol\Lambda}^2 \, G \, \vec{\boldsymbol\Lambda}^2 \frac{1}{g}\,\textnormal d \boldsymbol \eta\,,

      i.e. :math:`\mathbf v^\top \mathbb M^2_\rho \mathbf u = \int \rho\, \mathbf v \cdot \mathbf u \,\textnormal d \mathbf x`
      (as :attr:`~struphy.feec.mass.WeightedMassOperators.M2n` for :math:`\rho = \rho_0`);
    * the linear form :meth:`kinetic_form` of the kinetic energy in :math:`V^3_h`,
      :math:`k(\mathbf u)_{ijk} = \int \Lambda^3_{ijk} |\mathbf u|^2 \frac{1}{\sqrt g}\,\textnormal d \mathbf x`, such that
      :math:`\boldsymbol \rho^\top k(\mathbf u) = \mathbf u^\top \mathbb M^2_\rho \mathbf u` (same quadrature), the
      derivative of the kinetic energy :math:`\frac 12 \int \rho |\mathbf u|^2` with respect to the density
      coefficients;
    * the antisymmetric matrix :attr:`W` of the vorticity force (:meth:`update_W`),
      :math:`\mathbf v^\top \mathbb W \mathbf u = \int \rho\, \mathbf v \cdot (\boldsymbol \omega \times \mathbf u)
      \,\textnormal d \mathbf x` for a vorticity 1-form :math:`\boldsymbol \omega`.

    :meth:`update_M_rho` also checks the sign of the density: the discrete transport of the density is not
    positivity preserving, and where :math:`\rho \leq 0` at a quadrature point, :math:`\mathbb M^2_\rho` is no longer
    positive definite (its solves can fail, and the kinetic energy is no longer a norm). The minimum of the (physical)
    density at the quadrature points is :attr:`min_density`; a warning is logged when it becomes non-positive.

    Parameters
    ----------
    derham : Derham
        Discrete de Rham sequence.
    domain : Domain
        Mapping.
    mass_ops : WeightedMassOperators
        Weighted mass operators, used to create the weighted mass matrices.
    """

    def __init__(self, derham: Derham, domain: Domain, mass_ops: WeightedMassOperators):
        self._derham = derham
        grid = [grid_1d.flatten() for grid_1d in derham.V0splines.quad_grid_pts[0]]
        spans, bn, bd = derham.prepare_eval_tp_fixed(grid)
        self._spans = spans
        self._bases3 = bd
        self._bases2 = [tuple(bn[d] if d == c else bd[d] for d in range(3)) for c in range(3)]
        self._bases1 = [tuple(bd[d] if d == c else bn[d] for d in range(3)) for c in range(3)]

        sqrt_g = xp.abs(domain.jacobian_det(*grid))
        # push-forward of two 2-forms: G / g; of a 1-form and the cross product of two 2-forms: G^{-1} / sqrt(g)
        self._G_g = domain.metric(*grid) / sqrt_g**2
        self._Ginv_sqrt_g = domain.metric_inv(*grid) / sqrt_g
        self._inv_sqrt_g = 1.0 / sqrt_g

        shape = tuple(len(g) for g in grid)
        self._rho = xp.zeros(shape, dtype=float)
        self._u = [xp.zeros(shape, dtype=float) for _ in range(3)]
        self._omega = [xp.zeros(shape, dtype=float) for _ in range(3)]
        self._a = [xp.zeros(shape, dtype=float) for _ in range(3)]
        self._f = xp.zeros(shape, dtype=float)
        self._tmp = xp.zeros(shape, dtype=float)
        self._weights_M_rho = [[xp.zeros(shape, dtype=float) for _ in range(3)] for _ in range(3)]
        self._weights_W = [[xp.zeros(shape, dtype=float) for _ in range(3)] for _ in range(3)]

        self._rho_spline = derham.create_spline_function("rho_density_weighted", "L2")
        self._u_spline = derham.create_spline_function("u_density_weighted", "Hdiv")
        self._omega_spline = derham.create_spline_function("omega_density_weighted", "Hcurl")

        self._M_rho = mass_ops.create_weighted_mass("Hdiv", "Hdiv")
        self._W = mass_ops.create_weighted_mass("Hdiv", "Hdiv")
        self._get_dofs_V3 = L2Projector("L2", mass_ops).get_dofs

        self._min_density = None
        self._warned_non_positive = False

    @property
    def M_rho(self):
        r"""Density-weighted mass matrix :math:`\mathbb M^2_\rho` (assembled by :meth:`update_M_rho`)."""
        return self._M_rho

    @property
    def W(self):
        r"""Antisymmetric matrix of the vorticity force (assembled by :meth:`update_W`)."""
        return self._W

    def _eval_rho(self, rho: Vector):
        self._rho_spline.vector = rho
        return self._rho_spline.eval_tp_fixed_loc(self._spans, self._bases3, out=self._rho)

    @property
    def min_density(self) -> float:
        """Minimum of the physical density at the quadrature points (all MPI ranks) at the last :meth:`update_M_rho`."""
        return self._min_density

    def _check_density(self, rho_values):
        """Global minimum of the physical density rho^3 / sqrt(g) at the quadrature points, with a warning when it becomes
        non-positive (once, until it is positive again). Collective: call on all ranks."""
        self._tmp[:] = rho_values
        self._tmp *= self._inv_sqrt_g
        min_density = float(xp.min(self._tmp)) if self._tmp.size > 0 else float("inf")
        comm = MPI.COMM_WORLD
        if comm.Get_size() > 1:
            min_density = comm.allreduce(min_density, op=MPI.MIN)
        self._min_density = min_density
        if min_density <= 0.0:
            if not self._warned_non_positive and comm.Get_rank() == 0:
                logger.warning(
                    f"Total density rho0 + rho~ non-positive at quadrature points (minimum {min_density:.3e}): the "
                    "density-weighted mass matrix is not positive definite. The density transport is not positivity "
                    "preserving; reduce the time step or the amplitude, or refine the grid."
                )
            self._warned_non_positive = True
        else:
            self._warned_non_positive = False

    def update_M_rho(self, rho: Vector):
        r"""Assemble :attr:`M_rho` for the density 3-form with the coefficients ``rho`` (total density), and check its sign
        (:attr:`min_density`)."""
        rho_values = self._eval_rho(rho)
        self._check_density(rho_values)
        for i in range(3):
            for j in range(3):
                self._weights_M_rho[i][j][:] = rho_values
                self._weights_M_rho[i][j] *= self._G_g[i, j]
        self._M_rho.assemble(self._weights_M_rho)

    def kinetic_form(self, u: Vector, out: Vector) -> Vector:
        r"""The linear form :math:`k(\mathbf u)` in :math:`V^3_h` (see class docstring), written to ``out``."""
        self._u_spline.vector = u
        u_values = self._u_spline.eval_tp_fixed_loc(self._spans, self._bases2, out=self._u)
        # |u|^2 sqrt(g) = u^T G u / sqrt(g): the 3-form geometric weight 1/sqrt(g) of get_dofs gives u^T G u / g
        self._f *= 0.0
        for i in range(3):
            for j in range(3):
                self._tmp[:] = u_values[i]
                self._tmp *= self._G_g[i, j]
                self._tmp *= u_values[j]
                self._f += self._tmp
        self._f /= self._inv_sqrt_g
        return self._get_dofs_V3(self._f, dofs=out)

    def update_W(self, rho: Vector, omega: Vector):
        r"""Assemble :attr:`W` for the density 3-form ``rho`` and the vorticity 1-form ``omega``: with
        :math:`\mathbf a = \hat\rho^3 G^{-1} \hat{\boldsymbol\omega}^1 / \sqrt g`, the weight of
        :math:`\hat v_i \hat u_j` is :math:`\epsilon_{ikj} a_k` (:math:`\hat{\mathbf v} \cdot (\mathbf a \times \hat{\mathbf u})`)."""
        rho_values = self._eval_rho(rho)
        self._omega_spline.vector = omega
        omega_values = self._omega_spline.eval_tp_fixed_loc(self._spans, self._bases1, out=self._omega)
        for k in range(3):
            self._a[k] *= 0.0
            for m in range(3):
                self._tmp[:] = omega_values[m]
                self._tmp *= self._Ginv_sqrt_g[k, m]
                self._a[k] += self._tmp
            self._a[k] *= rho_values
        a0, a1, a2 = self._a
        w = self._weights_W
        for i in range(3):
            w[i][i][:] = 0.0
        w[0][1][:] = -a2
        w[0][2][:] = a1
        w[1][0][:] = a2
        w[1][2][:] = -a0
        w[2][0][:] = -a1
        w[2][1][:] = a0
        self._W.assemble(self._weights_W)


class VelocityMass:
    r"""Mass matrix of a velocity 2-form for the linear and perturbative models: :math:`\mathbb M^2_{\rho_0}`
    (:attr:`~struphy.feec.mass.WeightedMassOperators.M2n`) without a density variable, or the mass matrix
    :math:`\mathbb M^2_\rho` of the total density :math:`\rho = \rho_0 + \tilde \rho` (coefficients
    ``projected_equil.n3`` plus those of the density perturbation ``rho_field``), assembled by :meth:`update`.

    Parameters
    ----------
    derham : Derham
        Discrete de Rham sequence.
    domain : Domain
        Mapping.
    mass_ops : WeightedMassOperators
        Weighted mass operators.
    projected_equil : ProjectedFluidEquilibrium
        Projected equilibrium, for the coefficients of the equilibrium density 3-form.
    rho_field : FEECVariable, optional
        Density perturbation (``"L2"`` space, not updated); None for :math:`\mathbb M^2_{\rho_0}`.
    """

    def __init__(
        self, derham: Derham, domain: Domain, mass_ops: WeightedMassOperators, projected_equil, rho_field=None
    ):
        self._rho_field = rho_field
        if rho_field is None:
            self._M_rho = mass_ops.M2n
        else:
            self._ops = HdivDensityOperators(derham, domain, mass_ops)
            self._M_rho = self._ops.M_rho
            self._n3 = projected_equil.n3
            self._rho = self._n3.space.zeros()
            self.update()

    @property
    def M_rho(self):
        """The mass matrix (the same object after every :meth:`update`)."""
        return self._M_rho

    @property
    def weighted(self) -> bool:
        """Whether the mass matrix is weighted with the total density (``rho_field`` given)."""
        return self._rho_field is not None

    def update(self):
        """Assemble the mass matrix for the current density (nothing without ``rho_field``)."""
        if self._rho_field is None:
            return
        self._n3.copy(out=self._rho)
        self._rho += self._rho_field.spline.vector
        self._ops.update_M_rho(self._rho)


class EquilibriumCurrentTerm:
    r"""Second-order magnetic term of the linear and perturbative variational models, written with the equilibrium
    current: :math:`\int \mathbf B_0 \cdot \hat{\mathbf B} = \int \mathbf J_0 \cdot \hat{\mathbf A}`, with the
    auxiliary second-order field :math:`\hat{\mathbf B} = \nabla \times \hat{\mathbf A}` advanced by
    :math:`\partial_t \hat{\mathbf A} = - \mathbf B^w \times \mathbf u` (``B^w`` the perturbation, the weight of the
    term). Its variation gives the force :math:`\int \mathbf v \cdot (\mathbf J_0 \times \mathbf B^w)`.

    The current :math:`\mathbf j_0 \in V^1_h` is the one of the dual complex (no boundary conditions, see
    :meth:`equilibrium_current`), not the weak curl :math:`\mathbb M_1^{-1} \mathbb C^\top \mathbb M_2 \mathbf b_0`,
    whose tangential components vanish at a wall and cause a spurious wall instability. The force is the matrix
    :math:`\mathbb W` of :math:`\int \mathbf v \cdot (\mathbf j_0 \times \mathbf B)` (:attr:`R` is
    :math:`-\mathbb W`), and :meth:`advance` updates :math:`\hat{\mathbf a} \leftarrow \hat{\mathbf a} - \Delta t\,
    (\mathbb M^1_{\textnormal{free}})^{-1} \mathbf y(\mathbf b^w \times \mathbf u)` with
    :math:`\mathbf y(\mathbf w)_k = \int \mathbf w \cdot \vec{\boldsymbol\Lambda}^1_k` over all basis functions,
    so that the energy :meth:`energy` :math:`= \mathbf j_0^\top \mathbb M^1_{\textnormal{free}} \hat{\mathbf a}`
    changes by exactly the work :math:`-\Delta t\, \mathbf u^\top \mathbb W \mathbf b^w` of the force (both are
    evaluated at the same quadrature points).

    Parameters
    ----------
    derham : Derham
        Discrete de Rham sequence.
    domain : Domain
        Mapping.
    mass_ops : WeightedMassOperators
        Weighted mass operators.
    projected_equil : ProjectedFluidEquilibrium
        Projected equilibrium, for the coefficients of the equilibrium magnetic field 2-form.
    u_space : str
        Space of the velocity, ``"H1vec"`` or ``"Hdiv"``.
    aux : bool
        Whether the auxiliary vector potential is advanced (:meth:`advance`, :meth:`energy`).
    """

    def __init__(
        self,
        derham: Derham,
        domain: Domain,
        mass_ops: WeightedMassOperators,
        projected_equil,
        u_space: str,
        aux: bool = True,
    ):
        self._derham = derham
        self._mass_ops = mass_ops
        # equilibrium current through the dual complex (no boundary conditions): h0 = *_h b0 in V1, J0 = curl h0 in
        # V2 and its representation j0 in V1, all over the full spaces
        M1 = mass_ops.M1
        self._M1_free = M1.without_boundary_ops()
        pc = MassMatrixPreconditioner(M1, apply_bc=False)
        self._inv_M1 = inverse(self._M1_free, "pcg", pc=pc, tol=1e-14, maxiter=3000, recycle=True)
        j0 = self.equilibrium_current(projected_equil.b2)

        # spline evaluation at the quadrature points
        grid = [pts.flatten() for pts in derham.V3splines.quad_grid_pts[0]]
        self._spans, bn, bd = derham.prepare_eval_tp_fixed(grid)
        self._bases_1form = [[bd[0], bn[1], bn[2]], [bn[0], bd[1], bn[2]], [bn[0], bn[1], bd[2]]]
        self._bases_2form = [[bn[0], bd[1], bd[2]], [bd[0], bn[1], bd[2]], [bd[0], bd[1], bn[2]]]
        self._bases_u = self._bases_2form if u_space == "Hdiv" else [[bn[0], bn[1], bn[2]]] * 3
        sqrt_g = domain.jacobian_det(*grid)
        # u-tilde / u-hat
        self._u_scale = 1.0 / sqrt_g if u_space == "Hdiv" else None

        j0f = derham.create_spline_function("j0", "Hcurl")
        j0f.vector = j0
        j0_v = j0f.eval_tp_fixed_loc(self._spans, self._bases_1form)
        # c = G^{-1} j0 sqrt(g) times the factor of u-tilde: weights of W, u . (c x b) = u^T [c]_x b
        g_inv = domain.metric_inv(*grid)
        c = [sum(g_inv[a, e] * j0_v[e] for e in range(3)) for a in range(3)]
        if u_space == "H1vec":
            c = [ca * sqrt_g for ca in c]
        # R = -W, with the weights -[c]_x
        self._R = mass_ops.create_weighted_mass("Hdiv", u_space)
        self._R.assemble([[None, c[2], -c[1]], [-c[2], None, c[0]], [c[1], -c[0], None]])

        if aux:
            # the linear form j0^T M1 of the energy
            self._M1j0 = self._M1_free.dot(j0)
            self._bf = derham.create_spline_function("b_aux", "Hdiv")
            self._uf = derham.create_spline_function("u_aux", u_space)
            self._b_values = [xp.zeros_like(sqrt_g) for _ in range(3)]
            self._u_values = [xp.zeros_like(sqrt_g) for _ in range(3)]
            self._bxu_values = [xp.zeros_like(sqrt_g) for _ in range(3)]
            self._get_dofs_1 = L2Projector("Hcurl", mass_ops).get_dofs
            self._tmp_y = derham.V1pol.zeros()
            self._tmp_a = derham.V1pol.zeros()
            self._tmp_ca = derham.V2pol.zeros()

    @property
    def R(self):
        r""":math:`-\mathbb W`, the matrix (2-forms to the velocity space) of
        :math:`-\int \mathbf v \cdot (\mathbf j_0 \times \mathbf B)`."""
        return self._R

    def equilibrium_current(self, b0):
        r"""The equilibrium current :math:`\mathbf j_0 \in V^1_h` of the second-order term, through the dual complex:

        .. math::

            \mathbb M^1_{\textnormal{free}}\, \mathbf h_0 = \mathbb K \mathbf b_0 \,, \qquad
            \mathbf J_0 = \mathbb C\, \mathbf h_0 \,, \qquad
            \mathbb M^1_{\textnormal{free}}\, \mathbf j_0 = \mathbb K \mathbf J_0 \,,

        with the metric-free pairing :math:`\mathbb K_{ij} = \int_{\hat\Omega} \vec{\boldsymbol\Lambda}^1_i \cdot
        \vec{\boldsymbol\Lambda}^2_j \,\textrm d \boldsymbol\eta` of 1-forms and 2-forms, all operators over the full
        spaces (without boundary conditions). :math:`\mathbf h_0` is the field as a 1-form (Hodge star of the 2-form
        :math:`\mathbf b_0`) and :math:`\mathbf J_0` its curl, exactly divergence free. Collective: call on all
        ranks."""
        K = self._mass_ops.create_weighted_mass("Hdiv", "Hcurl", weights=("Identity",), assemble=True)
        K = K.without_boundary_ops()
        h0 = self._inv_M1.dot(K.dot(b0)).copy()
        J0 = self._derham.curl_bcfree.dot(h0)
        return self._inv_M1.dot(K.dot(J0)).copy()

    def energy(self, at1) -> float:
        r"""The second-order magnetic energy :math:`\int \mathbf J_0 \cdot \hat{\mathbf A} \approx \mathbf j_0^\top
        \mathbb M^1_{\textnormal{free}} \hat{\mathbf a}` of the auxiliary vector potential ``at1``. Collective."""
        return self._M1j0.inner(at1)

    def advance(self, dt, un12, bw, at1=None, bt2=None):
        r""":math:`\hat{\mathbf a} \leftarrow \hat{\mathbf a} - \Delta t\, (\mathbb M^1_{\textnormal{free}})^{-1}
        \mathbf y(\mathbf b^w \times \mathbf u^{n+1/2})` (``at1``) and the diagnostic :math:`\hat{\mathbf b} \leftarrow
        \hat{\mathbf b} + \mathbb C\, \delta \hat{\mathbf a}` (``bt2``); either may be None. Collective."""
        self._bf.vector = bw
        self._uf.vector = un12
        b_v = self._bf.eval_tp_fixed_loc(self._spans, self._bases_2form, out=self._b_values)
        u_v = self._uf.eval_tp_fixed_loc(self._spans, self._bases_u, out=self._u_values)
        if self._u_scale is not None:
            for ui in u_v:
                ui *= self._u_scale
        bxu = self._bxu_values
        for i in range(3):
            j, k = (i + 1) % 3, (i + 2) % 3
            bxu[i][:] = b_v[j] * u_v[k]
            bxu[i] -= b_v[k] * u_v[j]
        y = self._get_dofs_1(bxu, dofs=self._tmp_y, apply_bc=False)
        a = self._inv_M1.dot(y, out=self._tmp_a)
        a *= dt
        if at1 is not None:
            at1 -= a
            at1.update_ghost_regions()
        if bt2 is not None:
            ca = self._derham.curl.dot(a, out=self._tmp_ca)
            bt2 -= ca
            bt2.update_ghost_regions()


def coeff_arrays(v) -> list:
    """The coefficient arrays of a scalar Stencil- or PolarVector, for elementwise operations."""
    if isinstance(v, PolarVector):
        return [*v.pol, v.tp._data]
    return [v._data]


class WeakPressureTerm:
    r"""Second-order pressure term of the linear and perturbative variational models in :math:`p`, in weak form:
    only the integral of the auxiliary pressure :math:`\hat p` (``pt3``) enters the energy,
    :math:`\int \hat p / (\gamma - 1)`, and for :math:`\mathbf u \cdot \mathbf n = 0` on the boundary its transport
    :math:`\partial_t \hat p + \nabla \cdot (p^w \mathbf u) + (\gamma - 1) p^w \nabla \cdot \mathbf u = 0` gives the
    force :math:`\int p^w \nabla \cdot \mathbf v`. Discretely the force is :math:`-\mathbb R \mathbf p^w` with
    :math:`\mathbb R = -(\mathbb D \mathcal U^v)^\top \mathbb M^3` (:attr:`R`; :math:`\mathcal U^v` the identity for a
    velocity in :math:`H(\textnormal{div})`), and :meth:`advance` updates

    .. math::

        \frac{\hat{\mathbf p}^{n+1} - \hat{\mathbf p}^n}{\Delta t}
        + (\gamma - 1)\, \frac{(\mathbb D \mathcal U^v \mathbf u) \odot (\mathbb M^3 \mathbf p^w)}
        {\hat l^3(1)} = 0 \,,

    with :math:`\hat l^3(1)_{ijk} = \int_{\hat\Omega} \Lambda^3_{ijk} \,\textrm d \boldsymbol\eta` and elementwise
    product and division (lumped :math:`L^2` projection of :math:`(\gamma - 1) p^w \nabla \cdot \mathbf u`): the
    energy :meth:`energy` :math:`= \hat l^3(1) \cdot \hat{\mathbf p} / (\gamma - 1)` changes by exactly the work
    :math:`-\Delta t\, (\mathbb D \mathcal U^v \mathbf u)^\top \mathbb M^3 \mathbf p^w` of the force.

    Parameters
    ----------
    derham : Derham
        Discrete de Rham sequence.
    domain : Domain
        Mapping.
    mass_ops : WeightedMassOperators
        Weighted mass operators.
    basis_ops : BasisProjectionOperators
        Basis projection operators, for the projection :math:`\mathcal U^v` of a velocity in ``"H1vec"``.
    u_space : str
        Space of the velocity, ``"H1vec"`` or ``"Hdiv"``.
    gamma : float
        Adiabatic index.
    """

    def __init__(self, derham: Derham, domain: Domain, mass_ops: WeightedMassOperators, basis_ops, u_space, gamma):
        self._mass_ops = mass_ops
        self._gamma = gamma
        self._div_u = derham.div if u_space == "Hdiv" else derham.div @ basis_ops.Uv
        # (the sign inside the composition: a Schur solver rescales an outer ScaledLinearOperator with dt)
        self._R = self._div_u.T @ (-1.0 * mass_ops.M3)
        # 1 / l3(1), for the lumped L2 projection of the auxiliary update
        self._l3 = L2Projector("L2", mass_ops).get_dofs(domain.jacobian_det)
        self._inv_l3 = self._l3.space.zeros()
        for a, inv in zip(coeff_arrays(self._l3), coeff_arrays(self._inv_l3)):
            nonzero = a != 0.0
            inv[nonzero] = 1.0 / a[nonzero]
        self._tmp_a = derham.V3pol.zeros()
        self._tmp_M3p = derham.V3pol.zeros()

    @property
    def div_u(self):
        r"""The divergence :math:`\mathbb D \mathcal U^v` of the velocity."""
        return self._div_u

    @property
    def R(self):
        r""":math:`-(\mathbb D \mathcal U^v)^\top \mathbb M^3`, the matrix of :math:`-\int p \nabla \cdot \mathbf v`."""
        return self._R

    def energy(self, pt3) -> float:
        r""":math:`\hat l^3(1) \cdot \hat{\mathbf p} / (\gamma - 1)`. Collective."""
        return self._l3.inner(pt3) / (self._gamma - 1.0)

    def advance(self, dt, un12, pw, pt3):
        r""":math:`\hat{\mathbf p} \leftarrow \hat{\mathbf p} - \Delta t\, (\gamma - 1) (\mathbb D \mathcal U^v
        \mathbf u) \odot (\mathbb M^3 \mathbf p^w) / \hat l^3(1)`. Collective."""
        a = self._div_u.dot(un12, out=self._tmp_a)
        M3p = self._mass_ops.M3.dot(pw, out=self._tmp_M3p)
        for ai, m, inv in zip(coeff_arrays(a), coeff_arrays(M3p), coeff_arrays(self._inv_l3)):
            ai *= m
            ai *= inv
        a *= (self._gamma - 1.0) * dt
        pt3 -= a
        pt3.update_ghost_regions()


def extracted_ones(derham: Derham, form: str) -> Vector:
    """The vector of ones in the tensor-product space of the ``form`` (``"0"``, ``"1"``, ``"2"``, ``"3"`` or ``"v"``),
    mapped by the extraction operator, i.e. a vector of the coefficient space of the variables (polar with polar
    splines). Used for density averages such as ``ones^T A ones / ones^T M ones`` (e.g. the reference density of a
    stiffness preconditioner). With polar splines it is not the coefficient vector of a constant (the polar
    coefficients near the pole are sums over the angle), so such an average weights the region near the axis more;
    it is exact for a uniform density."""
    extraction = derham.extraction_ops[form]
    ones = extraction.domain.zeros()
    for block in getattr(ones, "blocks", (ones,)):
        block._data[:] = 1.0
    return extraction.dot(ones)
