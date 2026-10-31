"""Operators for a velocity 2-form weighted with the total density :math:`\\rho = \\rho_0 + \\tilde \\rho` of the
perturbative models (density 3-form evolving in time), on the quadrature grid of the mass matrices."""

import logging

import cunumpy as xp
from feectools.linalg.basic import Vector
from maybempi import MPI

from struphy.feec.mass import L2Projector, WeightedMassOperators
from struphy.feec.psydac_derham import Derham
from struphy.geometry.base import Domain

logger = logging.getLogger("struphy")


class HdivDensityOperators:
    r"""Operators of a velocity :math:`\mathbf u \in H(\textnormal{div})` (2-form coefficients) weighted with a density
    3-form :math:`\hat \rho^3 = \sum_{ijk} \rho_{ijk} \Lambda^3_{ijk}` that changes in time:

    * the mass matrix :attr:`M` (:meth:`update_density`),

      .. math::

          \mathbb M^2_\rho = \int \hat \rho^3 \, \vec{\boldsymbol\Lambda}^2 \, G \, \vec{\boldsymbol\Lambda}^2 \frac{1}{g}\,\textnormal d \boldsymbol \eta\,,

      i.e. :math:`\mathbf v^\top \mathbb M^2_\rho \mathbf u = \int \rho\, \mathbf v \cdot \mathbf u \,\textnormal d \mathbf x`
      (as :attr:`~struphy.feec.mass.WeightedMassOperators.M2n` for :math:`\rho = \rho_0`);
    * the linear form :meth:`kinetic_form` of the kinetic energy in :math:`V^3_h`,
      :math:`k(\mathbf u)_{ijk} = \int \Lambda^3_{ijk} |\mathbf u|^2 \frac{1}{\sqrt g}\,\textnormal d \mathbf x`, such that
      :math:`\boldsymbol \rho^\top k(\mathbf u) = \mathbf u^\top \mathbb M^2_\rho \mathbf u` (same quadrature), the
      derivative of the kinetic energy :math:`\frac 12 \int \rho |\mathbf u|^2` with respect to the density
      coefficients;
    * the antisymmetric matrix :attr:`W` of the vorticity force (:meth:`update_vorticity`),
      :math:`\mathbf v^\top \mathbb W \mathbf u = \int \rho\, \mathbf v \cdot (\boldsymbol \omega \times \mathbf u)
      \,\textnormal d \mathbf x` for a vorticity 1-form :math:`\boldsymbol \omega`.

    :meth:`update_density` also checks the sign of the density: the discrete transport of the density is not
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
        self._weights_M = [[xp.zeros(shape, dtype=float) for _ in range(3)] for _ in range(3)]
        self._weights_W = [[xp.zeros(shape, dtype=float) for _ in range(3)] for _ in range(3)]

        self._rho_spline = derham.create_spline_function("rho_density_weighted", "L2")
        self._u_spline = derham.create_spline_function("u_density_weighted", "Hdiv")
        self._omega_spline = derham.create_spline_function("omega_density_weighted", "Hcurl")

        self._M = mass_ops.create_weighted_mass("Hdiv", "Hdiv")
        self._W = mass_ops.create_weighted_mass("Hdiv", "Hdiv")
        self._get_dofs_V3 = L2Projector("L2", mass_ops).get_dofs

        self._min_density = None
        self._warned_non_positive = False

    @property
    def M(self):
        r"""Density-weighted mass matrix :math:`\mathbb M^2_\rho` (assembled by :meth:`update_density`)."""
        return self._M

    @property
    def W(self):
        r"""Antisymmetric matrix of the vorticity force (assembled by :meth:`update_vorticity`)."""
        return self._W

    def _eval_rho(self, rho: Vector):
        self._rho_spline.vector = rho
        return self._rho_spline.eval_tp_fixed_loc(self._spans, self._bases3, out=self._rho)

    @property
    def min_density(self) -> float:
        """Minimum of the physical density at the quadrature points (all MPI ranks) at the last :meth:`update_density`."""
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

    def update_density(self, rho: Vector):
        r"""Assemble :attr:`M` for the density 3-form with the coefficients ``rho`` (total density), and check its sign
        (:attr:`min_density`)."""
        rho_values = self._eval_rho(rho)
        self._check_density(rho_values)
        for i in range(3):
            for j in range(3):
                self._weights_M[i][j][:] = rho_values
                self._weights_M[i][j] *= self._G_g[i, j]
        self._M.assemble(self._weights_M)

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

    def update_vorticity(self, rho: Vector, omega: Vector):
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
            self._M = mass_ops.M2n
        else:
            self._ops = HdivDensityOperators(derham, domain, mass_ops)
            self._M = self._ops.M
            self._n3 = projected_equil.n3
            self._rho = self._n3.space.zeros()
            self.update()

    @property
    def M(self):
        """The mass matrix (the same object after every :meth:`update`)."""
        return self._M

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
        self._ops.update_density(self._rho)
