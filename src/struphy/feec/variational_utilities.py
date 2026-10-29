import logging
from copy import deepcopy

import cunumpy as xp
from feectools.linalg.basic import IdentityOperator, LinearOperator, Vector
from feectools.linalg.block import BlockVector
from feectools.linalg.solvers import inverse

from struphy.feec import preconditioner
from struphy.feec.basis_projection_ops import (
    BasisProjectionOperator,
    BasisProjectionOperatorLocal,
    CoordinateProjector,
)
from struphy.feec.psydac_derham import Derham

logger = logging.getLogger("struphy")


class BracketOperator(LinearOperator):
    r"""The linear map :math:`\mathbb R^{3N_0} \to \mathbb R^{3N_0}`,

    .. math::

        \mathbf v \in \mathbb R^{3N_0} \mapsto \mathbf w = (w_{\mu,ijk})_{\mu,ijk} \in \mathbb R^{3N_0}\,,

    defined by

    .. math::

        w_{\mu,ijk} = \int \hat{\mathbf m}(\boldsymbol \eta)\, G\, [\mathbf v^\top \vec{\boldsymbol \Lambda}^v, \vec{\Lambda}^v_{\mu,ijk}] \,\sqrt g\, \textnormal d\boldsymbol \eta\,,

    where :math:`\hat{\mathbf m}(\boldsymbol \eta)` is a given vector-field, and with the usual vector-field bracket

    .. math::

        [\mathbf v^\top \vec{\boldsymbol \Lambda}^v, \vec{\Lambda}^v_{\mu,ijk}] = \mathbf v^\top \vec{\boldsymbol \Lambda}^v \cdot \nabla \vec{\Lambda}^v_{\mu,ijk} - \vec{\Lambda}^v_{\mu,ijk} \cdot \nabla (\mathbf v^\top \vec{\boldsymbol \Lambda}^v)\,.

    This is discretized as

    .. math::

        \mathbf w = \sum_{\mu = 1}^3 I_\mu \Big(\hat{\Pi}^{0}[\hat{\mathbf v}_h \cdot \vec{\boldsymbol \Lambda}^1 ] \mathbb G P_\mu - \hat{\Pi}^0[\hat{\mathbf A}^1_{\mu,h} \cdot \vec{\boldsymbol \Lambda}^v] \Big)^\top \mathbf u  \,,

    where :math:`I_\mu` and :math:`P_\mu` stand for the :class:`~struphy.feec.basis_projection_ops.CoordinateInclusion`
    and :class:`~struphy.feec.basis_projection_ops.CoordinateProjector`, respectively,
    and the vector :math:`\mathbf u = (\hat{\mathbf m}, \vec{\boldsymbol \Lambda}^v)_{L^2} = \mathbb M^v \mathbf m` is provided as input.
    The weights in the the two :class:`~struphy.feec.basis_projection_ops.BasisProjectionOperator` are given by

    .. math::

        \hat{\mathbf v}_h = \mathbf v^\top \vec{\boldsymbol \Lambda}^v \in (V_h^0)^3 \,, \qquad \hat{\mathbf A}^1_{\mu,h} = \nabla P_\mu(\mathbf v^\top \vec{\boldsymbol \Lambda}^v)] \in V_h^1\,.

    Initialized and used in :class:`~struphy.propagators.variational_momentum_advection.VariationalMomentumAdvection` propagator.

    Parameters
    ----------
    derham : Derham
        Discrete de Rham sequence.

    u : BlockVector
        Coefficient of a field belonging to the H1vec space of the de Rahm sequence,
        representing the mass matrix applie to the m factor in the above integral.

    """

    def __init__(
        self,
        derham: Derham,
        u: BlockVector,
    ):
        Xh = derham.Vvfem
        V1h = derham.V1fem
        self._domain = derham.Vvpol
        self._codomain = derham.Vvpol
        self._dtype = Xh.coeff_space.dtype
        self._u = u

        # tmp for evaluating u
        self.vf = derham.create_spline_function("uf", "H1vec")
        self.gv1f = derham.create_spline_function("gu1f", "Hcurl")  # grad(u[0])
        self.gv2f = derham.create_spline_function("gu2f", "Hcurl")  # grad(u[1])
        self.gv3f = derham.create_spline_function("gu3f", "Hcurl")  # grad(u[2])

        self.gp1v = derham.V1pol.zeros()
        self.gp2v = derham.V1pol.zeros()
        self.gp3v = derham.V1pol.zeros()

        P0 = derham.P0
        # Initialize the CoordinateProjectors
        # self.Pcoord1 = CoordinateProjector(0, Xh, V0h)
        # self.Pcoord2 = CoordinateProjector(1, Xh, V0h)
        # self.Pcoord3 = CoordinateProjector(2, Xh, V0h)
        self.Pcoord1 = CoordinateProjector(0, derham.Vvpol, derham.V0pol) @ derham.boundary_ops["v"]
        self.Pcoord2 = CoordinateProjector(1, derham.Vvpol, derham.V0pol) @ derham.boundary_ops["v"]
        self.Pcoord3 = CoordinateProjector(2, derham.Vvpol, derham.V0pol) @ derham.boundary_ops["v"]

        # Initialize the BasisProjectionOperators
        if derham._with_local_projectors:
            self.PiuT = BasisProjectionOperatorLocal(
                P0,
                V1h,
                [[None, None, None]],
                transposed=True,
                V_extraction_op=derham.extraction_ops["1"],
                V_boundary_op=IdentityOperator(derham.V1pol),
                P_boundary_op=IdentityOperator(derham.V0pol),
            )

            self.PigvT_1 = BasisProjectionOperatorLocal(
                P0,
                Xh,
                [[None, None, None]],
                transposed=True,
                V_extraction_op=derham.extraction_ops["v"],
                V_boundary_op=derham.boundary_ops["v"],
                P_boundary_op=IdentityOperator(derham.V0pol),
            )

            self.PigvT_2 = BasisProjectionOperatorLocal(
                P0,
                Xh,
                [[None, None, None]],
                transposed=True,
                V_extraction_op=derham.extraction_ops["v"],
                V_boundary_op=derham.boundary_ops["v"],
                P_boundary_op=IdentityOperator(derham.V0pol),
            )

            self.PigvT_3 = BasisProjectionOperatorLocal(
                P0,
                Xh,
                [[None, None, None]],
                transposed=True,
                V_extraction_op=derham.extraction_ops["v"],
                V_boundary_op=derham.boundary_ops["v"],
                P_boundary_op=IdentityOperator(derham.V0pol),
            )
        else:
            self.PiuT = BasisProjectionOperator(
                P0,
                V1h,
                [[None, None, None]],
                transposed=True,
                use_cache=True,
                V_extraction_op=derham.extraction_ops["1"],
                V_boundary_op=IdentityOperator(derham.V1pol),
                P_boundary_op=IdentityOperator(derham.V0pol),
            )

            self.PigvT_1 = BasisProjectionOperator(
                P0,
                Xh,
                [[None, None, None]],
                transposed=True,
                use_cache=True,
                V_extraction_op=derham.extraction_ops["v"],
                V_boundary_op=derham.boundary_ops["v"],
                P_boundary_op=IdentityOperator(derham.V0pol),
            )
            self.PigvT_2 = BasisProjectionOperator(
                P0,
                Xh,
                [[None, None, None]],
                transposed=True,
                use_cache=True,
                V_extraction_op=derham.extraction_ops["v"],
                V_boundary_op=derham.boundary_ops["v"],
                P_boundary_op=IdentityOperator(derham.V0pol),
            )
            self.PigvT_3 = BasisProjectionOperator(
                P0,
                Xh,
                [[None, None, None]],
                transposed=True,
                use_cache=True,
                V_extraction_op=derham.extraction_ops["v"],
                V_boundary_op=derham.boundary_ops["v"],
                P_boundary_op=IdentityOperator(derham.V0pol),
            )

        # Store the interpolation grid for later use in _update_all_weights
        interpolation_grid = [pts.flatten() for pts in derham.V0splines.proj_grid_pts[0]]

        self.interpolation_grid_spans, self.interpolation_grid_bn, self.interpolation_grid_bd = (
            derham.prepare_eval_tp_fixed(interpolation_grid)
        )

        self.interpolation_grid_gradient = [
            [self.interpolation_grid_bd[0], self.interpolation_grid_bn[1], self.interpolation_grid_bn[2]],
            [self.interpolation_grid_bn[0], self.interpolation_grid_bd[1], self.interpolation_grid_bn[2]],
            [self.interpolation_grid_bn[0], self.interpolation_grid_bn[1], self.interpolation_grid_bd[2]],
        ]

        # Create tmps for later use in evaluating on the grid
        grid_shape = tuple([len(loc_grid) for loc_grid in interpolation_grid])
        self._vf_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]
        self._gvf1_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]
        self._gvf2_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]
        self._gvf3_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]

        # gradient of the component of the vector field
        grad = derham.grad_bcfree
        self.gp1 = grad @ self.Pcoord1
        self.gp2 = grad @ self.Pcoord2
        self.gp3 = grad @ self.Pcoord3

        # v-> int(Pi(grad w_i . v)m_i)
        m1vgw1 = self.gp1.T @ self.PiuT @ self.Pcoord1
        m2vgw2 = self.gp2.T @ self.PiuT @ self.Pcoord2
        m3vgw3 = self.gp3.T @ self.PiuT @ self.Pcoord3

        # v-> int(Pi(grad v_i . w)m_i)
        m1wgv1 = self.PigvT_1 @ self.Pcoord1
        m2wgv2 = self.PigvT_2 @ self.Pcoord2
        m3wgv3 = self.PigvT_3 @ self.Pcoord3

        # v-> int(Pi([v,w]) . m)
        self.mbrackvw = m1wgv1 + m2wgv2 + m3wgv3 - m1vgw1 - m2vgw2 - m3vgw3

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def dtype(self):
        return self._dtype

    def update_u(self, newu):
        assert isinstance(newu, Vector)
        assert newu.space == self.domain
        self._u = newu

    def transpose(self, conjugate=False):
        return -self

    def dot(self, v, out=None):
        assert isinstance(v, Vector)
        assert v.space == self.domain

        if out is not None:
            assert isinstance(out, Vector)
            assert out.space == self.codomain

        self.vf.vector = v

        grad_1_v = self.gp1.dot(v, out=self.gp1v)
        grad_2_v = self.gp2.dot(v, out=self.gp2v)
        grad_3_v = self.gp3.dot(v, out=self.gp3v)

        # To avoid tmp we need to update the fields we created.
        self.gv1f.vector = grad_1_v
        self.gv2f.vector = grad_2_v
        self.gv3f.vector = grad_3_v

        vf_values = self.vf.eval_tp_fixed_loc(
            self.interpolation_grid_spans,
            [self.interpolation_grid_bn] * 3,
            out=self._vf_values,
        )

        gvf1_values = self.gv1f.eval_tp_fixed_loc(
            self.interpolation_grid_spans,
            self.interpolation_grid_gradient,
            out=self._gvf1_values,
        )

        gvf2_values = self.gv2f.eval_tp_fixed_loc(
            self.interpolation_grid_spans,
            self.interpolation_grid_gradient,
            out=self._gvf2_values,
        )

        gvf3_values = self.gv3f.eval_tp_fixed_loc(
            self.interpolation_grid_spans,
            self.interpolation_grid_gradient,
            out=self._gvf3_values,
        )

        self.PiuT.update_weights([[vf_values[0], vf_values[1], vf_values[2]]])

        self.PigvT_1.update_weights([[gvf1_values[0], gvf1_values[1], gvf1_values[2]]])
        self.PigvT_2.update_weights([[gvf2_values[0], gvf2_values[1], gvf2_values[2]]])
        self.PigvT_3.update_weights([[gvf3_values[0], gvf3_values[1], gvf3_values[2]]])

        if out is not None:
            self.mbrackvw.dot(self._u, out=out)
        else:
            out = self.mbrackvw.dot(self._u)

        return out


class L2_transport_operator(LinearOperator):
    r"""
    Operator

    .. math::
        \mathbf u \mapsto \nabla \cdot(\Pi^2(\rho \mathbf u)) \,
    from H1vec (or H(div)) to L2, where :math:`\rho` is a discrete 3-form which can be updated.

    For a velocity 2-form (``u_space="Hdiv"``) the projected 2-form is
    :math:`\hat\Pi^2[\hat\rho^3 \vec{\boldsymbol\Lambda}^2/\sqrt g]`, as in
    :attr:`~struphy.feec.basis_projection_ops.BasisProjectionOperators.Q2`.

    Parameters
    ----------
    derham : Derham
        Discrete de Rham sequence.

    transposed : Bool
        Assemble the transposed operator

    u_space : str
        Space of the velocity, ``"H1vec"`` or ``"Hdiv"``.

    phys_domain : Domain
        The mapping (needed for the metric term with ``u_space="Hdiv"``).
    """

    def __init__(self, derham, transposed=False, weights=None, u_space="H1vec", phys_domain=None):
        assert u_space in ("H1vec", "Hdiv"), f'u_space must be "H1vec" or "Hdiv", got {u_space!r}.'
        assert u_space == "H1vec" or phys_domain is not None, 'u_space="Hdiv" needs phys_domain.'
        # Get the projector and the spaces
        self._derham = derham
        self._transposed = transposed
        self._u_space = u_space
        self._phys_domain = phys_domain
        if weights is None:
            weights = [[None] * 3] * 3
        self._weights = weights
        u_form = self._derham.space_to_form[u_space]
        Vu = getattr(self._derham, f"V{u_form}pol")
        if self._transposed:
            self._codomain = Vu
            self._domain = self._derham.V3pol
        else:
            self._domain = Vu
            self._codomain = self._derham.V3pol
        P2 = self._derham.P2
        Xh = getattr(self._derham, f"V{u_form}fem")
        self._dtype = Xh.coeff_space.dtype
        self.field = self._derham.create_spline_function("rhof", "L2")

        # Initialize the BasisProjectionOperator
        if self._derham._with_local_projectors:
            self.Proj = BasisProjectionOperatorLocal(
                P2,
                Xh,
                self._weights,
                transposed=transposed,
                V_extraction_op=self._derham.extraction_ops[u_form],
                V_boundary_op=self._derham.boundary_ops[u_form],
                P_boundary_op=IdentityOperator(self._derham.V2pol),
            )

        else:
            self.Proj = BasisProjectionOperator(
                P2,
                Xh,
                self._weights,
                transposed=transposed,
                use_cache=True,
                V_extraction_op=self._derham.extraction_ops[u_form],
                V_boundary_op=self._derham.boundary_ops[u_form],
                P_boundary_op=IdentityOperator(self._derham.V2pol),
            )

        # divergence
        self.div = self._derham.div_bcfree

        # Initialize the transport operator and transposed
        if self._transposed:
            self._op = self.Proj @ self.div.T
        else:
            self._op = self.div @ self.Proj

        hist_grid = self._derham.V2splines.proj_grid_pts

        hist_grid_0 = [pts.flatten() for pts in hist_grid[0]]
        hist_grid_1 = [pts.flatten() for pts in hist_grid[1]]
        hist_grid_2 = [pts.flatten() for pts in hist_grid[2]]

        self.hist_grid_0_spans, self.hist_grid_0_bn, self.hist_grid_0_bd = self._derham.prepare_eval_tp_fixed(
            hist_grid_0,
        )
        self.hist_grid_1_spans, self.hist_grid_1_bn, self.hist_grid_1_bd = self._derham.prepare_eval_tp_fixed(
            hist_grid_1,
        )
        self.hist_grid_2_spans, self.hist_grid_2_bn, self.hist_grid_2_bd = self._derham.prepare_eval_tp_fixed(
            hist_grid_2,
        )

        grid_shape = tuple([len(loc_grid) for loc_grid in hist_grid_0])
        self._f_0_values = xp.zeros(grid_shape, dtype=float)

        grid_shape = tuple([len(loc_grid) for loc_grid in hist_grid_1])
        self._f_1_values = xp.zeros(grid_shape, dtype=float)

        grid_shape = tuple([len(loc_grid) for loc_grid in hist_grid_2])
        self._f_2_values = xp.zeros(grid_shape, dtype=float)

        # 1/sqrt(g) on the histopolation grids of the three components, for a velocity 2-form
        if u_space == "Hdiv":
            self._inv_sqrt_g = [
                1.0 / phys_domain.jacobian_det(*grid) for grid in (hist_grid_0, hist_grid_1, hist_grid_2)
            ]

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def dtype(self):
        return self._dtype

    def transpose(self, conjugate=False):
        return L2_transport_operator(
            self._derham,
            not self._transposed,
            weights=self._weights,
            u_space=self._u_space,
            phys_domain=self._phys_domain,
        )

    def dot(self, v, out=None):
        out = self._op.dot(v, out=out)
        return out

    def update_coeffs(self, coeff):
        r"""Update the coefficient of the projection operator.

        Parameters
        ----------
        coeffs : StencilVector
            coefficient of the discrete 3 form to update the projection operator.
        """
        self.field.vector = coeff

        f0_values = self.field.eval_tp_fixed_loc(
            self.hist_grid_0_spans,
            self.hist_grid_0_bd,
            out=self._f_0_values,
        )
        f1_values = self.field.eval_tp_fixed_loc(
            self.hist_grid_1_spans,
            self.hist_grid_1_bd,
            out=self._f_1_values,
        )
        f2_values = self.field.eval_tp_fixed_loc(
            self.hist_grid_2_spans,
            self.hist_grid_2_bd,
            out=self._f_2_values,
        )

        if self._u_space == "Hdiv":
            for values, inv_sqrt_g in zip((f0_values, f1_values, f2_values), self._inv_sqrt_g):
                values *= inv_sqrt_g

        self._weights = [
            [f0_values, None, None],
            [None, f1_values, None],
            [None, None, f2_values],
        ]

        self.Proj.update_weights(self._weights)


class Hdiv0_transport_operator(LinearOperator):
    r"""
    Operator

    .. math::
        u \mapsto \nabla \times (\Pi^1(\mathbf B \times \mathbf u)) \,
    from H1vec (or H(div)) to H(div), where :math:`\mathbf B` is a discrete 2-form which can be updated.

    For a velocity 2-form :math:`\hat{\mathbf u}^2` (``u_space="Hdiv"``) the vector field is
    :math:`\hat{\mathbf u}^2/\sqrt g`, i.e. the projected 1-form is
    :math:`\hat\Pi^1[(\hat{\mathbf B}^2 \times \vec{\boldsymbol\Lambda}^2)/\sqrt g]`, as in
    :attr:`~struphy.feec.basis_projection_ops.BasisProjectionOperators.T2`.

    Parameters
    ----------
    derham : Derham
        Discrete de Rham sequence.

    transposed : Bool
        Assemble the transposed operator

    u_space : str
        Space of the velocity, ``"H1vec"`` or ``"Hdiv"``.

    phys_domain : Domain
        The mapping (needed for the metric term with ``u_space="Hdiv"``).
    """

    def __init__(self, derham, transposed=False, weights=None, u_space="H1vec", phys_domain=None):
        assert u_space in ("H1vec", "Hdiv"), f'u_space must be "H1vec" or "Hdiv", got {u_space!r}.'
        assert u_space == "H1vec" or phys_domain is not None, 'u_space="Hdiv" needs phys_domain.'
        # Get the projector and the spaces
        self._derham = derham
        self._transposed = transposed
        self._u_space = u_space
        self._phys_domain = phys_domain
        if weights is None:
            weights = [[None] * 3] * 3
        self._weights = weights
        u_form = self._derham.space_to_form[u_space]
        Vu = getattr(self._derham, f"V{u_form}pol")
        if self._transposed:
            self._codomain = Vu
            self._domain = self._derham.V2pol
        else:
            self._domain = Vu
            self._codomain = self._derham.V2pol
        P1 = self._derham.P1
        Xh = getattr(self._derham, f"V{u_form}fem")
        self._dtype = Xh.coeff_space.dtype
        self.field = self._derham.create_spline_function("Bf", "Hdiv")

        # Initialize the BasisProjectionOperators
        if self._derham._with_local_projectors:
            self.Proj = BasisProjectionOperatorLocal(
                P1,
                Xh,
                self._weights,
                transposed=transposed,
                V_extraction_op=self._derham.extraction_ops[u_form],
                V_boundary_op=self._derham.boundary_ops[u_form],
                P_boundary_op=IdentityOperator(self._derham.V1pol),
            )

        else:
            self.Proj = BasisProjectionOperator(
                P1,
                Xh,
                self._weights,
                transposed=transposed,
                use_cache=True,
                V_extraction_op=self._derham.extraction_ops[u_form],
                V_boundary_op=self._derham.boundary_ops[u_form],
                P_boundary_op=IdentityOperator(self._derham.V1pol),
            )

        # gradient of the component of the vector field
        self.curl = self._derham.curl_bcfree

        # Initialize the transport operator and transposed
        if self._transposed:
            self._op = self.Proj @ self.curl.T
        else:
            self._op = self.curl @ self.Proj

        hist_grid = self._derham.V1splines.proj_grid_pts

        hist_grid_0 = [pts.flatten() for pts in hist_grid[0]]
        hist_grid_1 = [pts.flatten() for pts in hist_grid[1]]
        hist_grid_2 = [pts.flatten() for pts in hist_grid[2]]

        self.hist_grid_0_spans, self.hist_grid_0_bn, self.hist_grid_0_bd = self._derham.prepare_eval_tp_fixed(
            hist_grid_0,
        )
        self.hist_grid_1_spans, self.hist_grid_1_bn, self.hist_grid_1_bd = self._derham.prepare_eval_tp_fixed(
            hist_grid_1,
        )
        self.hist_grid_2_spans, self.hist_grid_2_bn, self.hist_grid_2_bd = self._derham.prepare_eval_tp_fixed(
            hist_grid_2,
        )

        grid_shape = tuple([len(loc_grid) for loc_grid in hist_grid_0])
        self._bf0_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]
        self.hist_grid_0_b = [
            [self.hist_grid_0_bn[0], self.hist_grid_0_bd[1], self.hist_grid_0_bd[2]],
            [
                self.hist_grid_0_bd[0],
                self.hist_grid_0_bn[1],
                self.hist_grid_0_bd[2],
            ],
            [self.hist_grid_0_bd[0], self.hist_grid_0_bd[1], self.hist_grid_0_bn[2]],
        ]
        grid_shape = tuple([len(loc_grid) for loc_grid in hist_grid_1])
        self._bf1_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]
        self.hist_grid_1_b = [
            [self.hist_grid_1_bn[0], self.hist_grid_1_bd[1], self.hist_grid_1_bd[2]],
            [
                self.hist_grid_1_bd[0],
                self.hist_grid_1_bn[1],
                self.hist_grid_1_bd[2],
            ],
            [self.hist_grid_1_bd[0], self.hist_grid_1_bd[1], self.hist_grid_1_bn[2]],
        ]

        grid_shape = tuple([len(loc_grid) for loc_grid in hist_grid_2])
        self._bf2_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]

        # 1/sqrt(g) on the histopolation grids of the three components, for a velocity 2-form
        if u_space == "Hdiv":
            self._inv_sqrt_g = [
                1.0 / phys_domain.jacobian_det(*grid) for grid in (hist_grid_0, hist_grid_1, hist_grid_2)
            ]
        self.hist_grid_2_b = [
            [self.hist_grid_2_bn[0], self.hist_grid_2_bd[1], self.hist_grid_2_bd[2]],
            [
                self.hist_grid_2_bd[0],
                self.hist_grid_2_bn[1],
                self.hist_grid_2_bd[2],
            ],
            [self.hist_grid_2_bd[0], self.hist_grid_2_bd[1], self.hist_grid_2_bn[2]],
        ]

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def dtype(self):
        return self._dtype

    def transpose(self, conjugate=False):
        return Hdiv0_transport_operator(
            self._derham,
            not self._transposed,
            weights=self._weights,
            u_space=self._u_space,
            phys_domain=self._phys_domain,
        )

    def dot(self, v, out=None):
        out = self._op.dot(v, out=out)
        return out

    def update_coeffs(self, coeff):
        r"""
        Update the coefficient of the projection operator.

        Parameters
        ----------
        coeffs : BlockVector
            coefficient of the discrete 2 form to update the projection operator.
        """
        self.field.vector = coeff

        bf0_values = self.field.eval_tp_fixed_loc(
            self.hist_grid_0_spans,
            self.hist_grid_0_b,
            out=self._bf0_values,
        )
        bf1_values = self.field.eval_tp_fixed_loc(
            self.hist_grid_1_spans,
            self.hist_grid_1_b,
            out=self._bf1_values,
        )
        bf2_values = self.field.eval_tp_fixed_loc(
            self.hist_grid_2_spans,
            self.hist_grid_2_b,
            out=self._bf2_values,
        )

        if self._u_space == "Hdiv":
            for values, inv_sqrt_g in zip((bf0_values, bf1_values, bf2_values), self._inv_sqrt_g):
                for v in values:
                    v *= inv_sqrt_g

        self._weights = [
            [None, -bf0_values[2], bf0_values[1]],
            [bf1_values[2], None, -bf1_values[0]],
            [-bf2_values[1], bf2_values[0], None],
        ]

        self.Proj.update_weights(self._weights)


class Pressure_transport_operator(LinearOperator):
    r"""
    Operator

    .. math::
        \mathbf u \mapsto \nabla \cdot (\Pi^2(p \mathbf u)) + (\gamma -1) \Pi^3(p \nabla \cdot \Pi^2(\mathbf u))  \,
    from H1vec (or H(div)) to L2, where :math:`p` is a discrete 3-form which can be updated.

    For a velocity 2-form (``u_space="Hdiv"``) the operator is
    :math:`\mathbb D \hat\Pi^2[\hat p^3 \vec{\boldsymbol\Lambda}^2/\sqrt g]
    + (\gamma - 1) \hat\Pi^3[\hat p^3 \Lambda^3/\sqrt g] \mathbb D`, without :math:`\mathbb U^v`, as built from
    :attr:`~struphy.feec.basis_projection_ops.BasisProjectionOperators.S2` and
    :attr:`~struphy.feec.basis_projection_ops.BasisProjectionOperators.K3` in
    :class:`~struphy.propagators.magnetosonic.Magnetosonic`.

    Parameters
    ----------
    derham : Derham
        Discrete de Rham sequence.

    phys_domain : Domain
        The domain in which the problem is discretized (needed for metric terms)

    Uv : BasisProjectionOperator
        The projection from H1vec to H(div) (not used with ``u_space="Hdiv"``, where it is the identity)

    gamma : Float
        Thermodynamical constant

    transposed : Bool
        Assemble the transposed operator

    u_space : str
        Space of the velocity, ``"H1vec"`` or ``"Hdiv"``.
    """

    def __init__(self, derham, phys_domain, Uv, gamma, transposed=False, weights1=None, weights2=None, u_space="H1vec"):
        assert u_space in ("H1vec", "Hdiv"), f'u_space must be "H1vec" or "Hdiv", got {u_space!r}.'
        # Get the projector and the spaces
        self._derham = derham
        self._phys_domain = phys_domain
        self._u_space = u_space
        u_form = self._derham.space_to_form[u_space]
        Vu = getattr(self._derham, f"V{u_form}pol")
        # the velocity as a 2-form
        self._Uv = Uv if u_space == "H1vec" else IdentityOperator(Vu)
        self._transposed = transposed
        self._gamma = gamma
        if weights1 is None:
            weights1 = [[None] * 3] * 3
        self._weights1 = weights1
        if weights2 is None:
            weights2 = [[lambda eta1, eta2, eta3: 0 * eta1]]
        self._weights2 = weights2
        if self._transposed:
            self._codomain = Vu
            self._domain = self._derham.V3pol
        else:
            self._domain = Vu
            self._codomain = self._derham.V3pol
        P2 = self._derham.P2
        P3 = self._derham.P3
        Xh = getattr(self._derham, f"V{u_form}fem")
        V3h = self._derham.V3fem
        self._dtype = Xh.coeff_space.dtype
        self.field = self._derham.create_spline_function("pf", "L2")

        self.Pip = BasisProjectionOperator(
            P2,
            Xh,
            self._weights1,
            transposed=transposed,
            use_cache=True,
            V_extraction_op=self._derham.extraction_ops[u_form],
            V_boundary_op=self._derham.boundary_ops[u_form],
            P_boundary_op=IdentityOperator(self._derham.V2pol),
        )

        self.Pip_div = BasisProjectionOperator(
            P3,
            V3h,
            self._weights2,
            transposed=transposed,
            use_cache=True,
            V_extraction_op=self._derham.extraction_ops["3"],
            V_boundary_op=self._derham.boundary_ops["3"],
            P_boundary_op=IdentityOperator(self._derham.V3pol),
        )

        # BC?

        div = self._derham.div

        self.div = div @ Uv if u_space == "H1vec" else div

        # Initialize the transport operator and transposed
        if self._transposed:
            self._op = self.Pip @ div.T + self.div.T @ self.Pip_div

        else:
            self._op = div @ self.Pip + self.Pip_div @ self.div

        int_grid = [pts.flatten() for pts in self._derham.V3splines.proj_grid_pts[0]]

        self.int_grid_spans, self.int_grid_bn, self.int_grid_bd = self._derham.prepare_eval_tp_fixed(
            int_grid,
        )

        metric = 1.0 / phys_domain.jacobian_det(*int_grid)
        self._proj_p_metric = deepcopy(metric)

        grid_shape = tuple([len(loc_grid) for loc_grid in int_grid])
        self._pf_values = xp.zeros(grid_shape, dtype=float)
        self._mapped_pf_values = xp.zeros(grid_shape, dtype=float)

        # gradient of the component of the vector field

        hist_grid_P2 = self._derham.V2splines.proj_grid_pts

        hist_grid_20 = [pts.flatten() for pts in hist_grid_P2[0]]
        hist_grid_21 = [pts.flatten() for pts in hist_grid_P2[1]]
        hist_grid_22 = [pts.flatten() for pts in hist_grid_P2[2]]

        self.hist_grid_20_spans, self.hist_grid_20_bn, self.hist_grid_20_bd = self._derham.prepare_eval_tp_fixed(
            hist_grid_20,
        )
        self.hist_grid_21_spans, self.hist_grid_21_bn, self.hist_grid_21_bd = self._derham.prepare_eval_tp_fixed(
            hist_grid_21,
        )
        self.hist_grid_22_spans, self.hist_grid_22_bn, self.hist_grid_22_bd = self._derham.prepare_eval_tp_fixed(
            hist_grid_22,
        )

        grid_shape = tuple([len(loc_grid) for loc_grid in hist_grid_20])
        self._pf_0_values = xp.zeros(grid_shape, dtype=float)

        grid_shape = tuple([len(loc_grid) for loc_grid in hist_grid_21])
        self._pf_1_values = xp.zeros(grid_shape, dtype=float)

        grid_shape = tuple([len(loc_grid) for loc_grid in hist_grid_22])
        self._pf_2_values = xp.zeros(grid_shape, dtype=float)

        # 1/sqrt(g) on the histopolation grids of the three components, for a velocity 2-form
        if u_space == "Hdiv":
            self._inv_sqrt_g = [
                1.0 / phys_domain.jacobian_det(*grid) for grid in (hist_grid_20, hist_grid_21, hist_grid_22)
            ]

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def dtype(self):
        return self._dtype

    def transpose(self, conjugate=False):
        return Pressure_transport_operator(
            self._derham,
            self._phys_domain,
            self._Uv,
            self._gamma,
            not self._transposed,
            weights1=self._weights1,
            weights2=self._weights2,
            u_space=self._u_space,
        )

    def dot(self, v, out=None):
        out = self._op.dot(v, out=out)
        return out

    def update_coeffs(self, coeff):
        r"""Update the coefficient of the projection operator.

        Parameters
        ----------
        coeffs : StencilVector
            coefficient of the discrete 3 form to update the projection operator.
        """
        self.field.vector = coeff

        pf_values = self.field.eval_tp_fixed_loc(
            self.int_grid_spans,
            self.int_grid_bd,
            out=self._pf_values,
        )

        self._mapped_pf_values *= 0.0
        self._mapped_pf_values += pf_values
        self._mapped_pf_values *= self._proj_p_metric
        self._mapped_pf_values *= self._gamma - 1.0

        self._weights2 = [[self._mapped_pf_values]]

        self.Pip_div.update_weights(self._weights2)

        # logger.info(self.Pip_divT._dof_mat._data)

        pf0_values = self.field.eval_tp_fixed_loc(
            self.hist_grid_20_spans,
            self.hist_grid_20_bd,
            out=self._pf_0_values,
        )
        pf1_values = self.field.eval_tp_fixed_loc(
            self.hist_grid_21_spans,
            self.hist_grid_21_bd,
            out=self._pf_1_values,
        )
        pf2_values = self.field.eval_tp_fixed_loc(
            self.hist_grid_22_spans,
            self.hist_grid_22_bd,
            out=self._pf_2_values,
        )

        if self._u_space == "Hdiv":
            for values, inv_sqrt_g in zip((pf0_values, pf1_values, pf2_values), self._inv_sqrt_g):
                values *= inv_sqrt_g

        self._weights1 = [
            [pf0_values, None, None],
            [None, pf1_values, None],
            [None, None, pf2_values],
        ]

        self.Pip.update_weights(self._weights1)


class InternalEnergyEvaluator:
    r"""Helper class for the evaluation of the internal energy or its partial derivative/discrete partial derivatives on an integration grid

    This class only contains a lot of array corresponding to the integration grid to avoid the allocation of temporaries,
    and method that can be called to evaluate the energy and derivatives on the grid.

    Parameters
    ----------
    derham : Derham
        Discrete de Rham sequence.

    gamma : Float
        Thermodynamical constant
    """

    def __init__(self, derham, gamma):
        self._derham = derham
        self._gamma = gamma
        integration_grid = [grid_1d.flatten() for grid_1d in self._derham.V0splines.quad_grid_pts[0]]

        self.integration_grid_spans, self.integration_grid_bn, self.integration_grid_bd = (
            self._derham.prepare_eval_tp_fixed(
                integration_grid,
            )
        )

        self._density_field = self._derham.create_spline_function("f3", "L2")
        self.sf = self._derham.create_spline_function("sf", "L2")
        self.sf1 = self._derham.create_spline_function("sf", "L2")
        self.rhof = self._derham.create_spline_function("rhof", "L2")
        self.rhof1 = self._derham.create_spline_function("rhof1", "L2")

        grid_shape = tuple([len(loc_grid) for loc_grid in integration_grid])
        self._rhof_values = xp.zeros(grid_shape, dtype=float)
        self._rhof1_values = xp.zeros(grid_shape, dtype=float)
        self._sf_values = xp.zeros(grid_shape, dtype=float)
        self._sf1_values = xp.zeros(grid_shape, dtype=float)
        self._delta_values = xp.zeros(grid_shape, dtype=float)
        self._rhof_mid_values = xp.zeros(grid_shape, dtype=float)
        self._sf_mid_values = xp.zeros(grid_shape, dtype=float)
        self._eta_values = xp.zeros(grid_shape, dtype=float)
        self._en_values = xp.zeros(grid_shape, dtype=float)
        self._en1_values = xp.zeros(grid_shape, dtype=float)
        self._de_values = xp.zeros(grid_shape, dtype=float)
        self._d2e_values = xp.zeros(grid_shape, dtype=float)
        self._tmp_int_grid = xp.zeros(grid_shape, dtype=float)
        self._tmp_int_grid2 = xp.zeros(grid_shape, dtype=float)
        self._DG_values = xp.zeros(grid_shape, dtype=float)

    def ener(self, rho, s, out=None):
        r"""Themodynamical energy as a function of rho and s, usign the perfect gaz hypothesis.

        .. math::
            E(\rho, s) = \rho^\gamma \text{exp}(s/\rho) \,.
        """
        gam = self._gamma
        if out is None:
            out = xp.power(rho, gam) * xp.exp(s / rho)
        else:
            out *= 0.0
            out += s
            out /= rho
            xp.exp(out, out=out)
            xp.power(rho, gam, out=self._tmp_int_grid)
            out *= self._tmp_int_grid
        return out

    def dener_drho(self, rho, s, out=None):
        r"""Derivative with respect to rho of the thermodynamical energy as a function of rho and s, usign the perfect gaz hypothesis.

        .. math::
            \frac{\partial E}{\partial \rho}(\rho, s) = (\gamma \rho^{\gamma-1} - s \rho^{\gamma-2})*\text{exp}(s/\rho) \,.
        """
        gam = self._gamma
        if out is None:
            out = (gam * xp.power(rho, gam - 1) - s * xp.power(rho, gam - 2)) * xp.exp(s / rho)
        else:
            out *= 0.0
            out += s
            out /= rho
            xp.exp(out, out=out)

            xp.power(rho, gam - 1, out=self._tmp_int_grid)
            self._tmp_int_grid *= gam

            xp.power(rho, gam - 2, out=self._tmp_int_grid2)
            self._tmp_int_grid2 *= s

            self._tmp_int_grid -= self._tmp_int_grid2
            out *= self._tmp_int_grid
        return out

    def dener_ds(self, rho, s, out=None):
        r"""Derivative with respect to s of the thermodynamical energy as a function of rho and s, usign the perfect gaz hypothesis.

        .. math::
            \frac{\partial E}{\partial s}(\rho, s) = \rho^{\gamma-1} \text{exp}(s/\rho) \,.
        """
        gam = self._gamma
        if out is None:
            out = xp.power(rho, gam - 1) * xp.exp(s / rho)
        else:
            out *= 0.0
            out += s
            out /= rho
            xp.exp(out, out=out)
            xp.power(rho, gam - 1, out=self._tmp_int_grid)
            out *= self._tmp_int_grid
        return out

    def d2ener_drho2(self, rho, s, out=None):
        r"""Second derivative with respect to (rho, rho) of the thermodynamical energy as a function of rho and s, usign the perfect gaz hypothesis.

        .. math::
            \frac{\partial^2 E}{\partial \rho^2}(\rho, s) = (\gamma*(\gamma-1) \rho^{\gamma-2}- 2 s (\gamma-1) rho^{\gamma-3}+ s^2 \rho^{\gamma-4}) \text{exp}(s/\rho) \,.
        """
        gam = self._gamma
        if out is None:
            out = (
                gam * (gam - 1) * xp.power(rho, gam - 2)
                - s * 2 * (gam - 1) * xp.power(rho, gam - 3)
                + s**2 * xp.power(rho, gam - 4)
            ) * xp.exp(s / rho)
        else:
            out *= 0.0
            out += s
            out /= rho
            xp.exp(out, out=out)

            xp.power(rho, gam - 2, out=self._tmp_int_grid)
            self._tmp_int_grid *= gam * (gam - 1)

            xp.power(rho, gam - 3, out=self._tmp_int_grid2)
            self._tmp_int_grid2 *= s
            self._tmp_int_grid2 *= 2 * (gam - 1)
            self._tmp_int_grid -= self._tmp_int_grid2

            xp.power(rho, gam - 4, out=self._tmp_int_grid2)
            self._tmp_int_grid2 *= s
            self._tmp_int_grid2 *= s
            self._tmp_int_grid += self._tmp_int_grid2
            out *= self._tmp_int_grid
        return out

    def d2ener_ds2(self, rho, s, out=None):
        r"""Second derivative with respect to (s, s) of the thermodynamical energy as a function of rho and s, usign the perfect gaz hypothesis.

        .. math::
            \frac{\partial^2 E}{\partial s^2}(\rho, s) = \rho^{\gamma-2} \text{exp}(s/ \rho) \,.
        """
        gam = self._gamma
        if out is None:
            out = xp.power(rho, gam - 2) * xp.exp(s / rho)
        else:
            out *= 0.0
            out += s
            out /= rho
            xp.exp(out, out=out)
            xp.power(rho, gam - 2, out=self._tmp_int_grid)
            out *= self._tmp_int_grid
        return out

    def eta(self, delta_x, out=None):
        r"""Switch function :math:`\eta(\delta) = 1- \text{exp}((-\delta/10^{-5})^2)`."""
        if out is None:
            out = 1.0 - xp.exp(-((delta_x / 1e-5) ** 2))
        else:
            out *= 0.0
            out += delta_x
            out /= 1e-5
            out **= 2
            out *= -1
            xp.exp(out, out=out)
            out *= -1
            out += 1.0
        return out

    def evaluate_discrete_de_drho_grid(self, rhon, rhon1, sn, out=None):
        r"""Evaluate the discrete gradient of the internal energy with respect to the :math:`\rho` variable

        .. math::
            \eta(\delta \rho)\frac{e(\rho^{n+1},s^n)-e(\rho^{n},s^n)}{\rho^{n+1}-\rho^n}+(1-\eta(\delta \rho))\frac{\partial e}{\partial \rho}(\rho^{n+\frac{1}{2}}, s^n) \,,

        """

        # Get the value of the fields on the grid
        rhof_values = self.eval_3form(rhon, out=self._rhof_values)
        rhof1_values = self.eval_3form(rhon1, out=self._rhof1_values)
        sf_values = self.eval_3form(sn, out=self._sf_values)

        if out is None:
            out = xp.zeros_like(rhof_values)

        # delta_rho_values = rhof1_values-rhof_values
        delta_rho_values = self._delta_values
        delta_rho_values *= 0.0
        delta_rho_values += rhof1_values
        delta_rho_values -= rhof_values

        # rho_mid_values = (rhof1_values+rhof_values)/2
        rho_mid_values = self._rhof_mid_values
        rho_mid_values *= 0
        rho_mid_values += rhof1_values
        rho_mid_values += rhof_values
        rho_mid_values /= 2

        eta = self.eta(delta_rho_values, out=self._eta_values)

        e_rho1_s = self.ener(
            rhof1_values,
            sf_values,
            out=self._en1_values,
        )
        e_rho_s = self.ener(
            rhof_values,
            sf_values,
            out=self._en_values,
        )

        de_rhom_s = self.dener_drho(
            rho_mid_values,
            sf_values,
            out=self._de_values,
        )

        # eta*delta_rho_values*(e_rho1_s-e_rho_s)*delta_rho_values/(delta_rho_values**2+1e-40)
        self._tmp_int_grid *= 0.0
        self._tmp_int_grid += e_rho1_s
        self._tmp_int_grid -= e_rho_s
        self._tmp_int_grid *= delta_rho_values
        delta_rho_values **= 2
        delta_rho_values += 1e-40
        self._tmp_int_grid /= delta_rho_values
        self._tmp_int_grid *= eta

        # (1-eta)*de_rhom_s
        eta -= 1.0
        eta *= -1.0
        de_rhom_s *= eta

        out *= 0.0
        out += self._tmp_int_grid
        out += de_rhom_s

        return out

    def evaluate_exact_de_drho_grid(self, rhon, sn, out=None):
        r"""
        Evaluation of the derivative of :math:`E` with respect to :math:`\rho` on the grid.
        """

        rhof_values = self.eval_3form(rhon, out=self._rhof_values)
        sf_values = self.eval_3form(sn, out=self._sf_values)

        out = self.dener_drho(rhof_values, sf_values, out=out)
        return out

    def evaluate_discrete_de_ds_grid(self, rhon, sn, sn1, out=None):
        r"""Evaluate the discrete gradient of the internal energy with respect to the :math:`s` variable

        .. math::
            \eta(\delta \rho)\frac{e(\rho^{n},s^{n+1})-e(\rho^{n},s^n)}{s^{n+1}-s^n}+(1-\eta(\delta s))\frac{\partial e}{\partial s}(\rho^n, s^{n+\frac{1}{2}}) \,,

        """
        # Get the value of the fields on the grid
        sf_values = self.eval_3form(sn, out=self._sf_values)
        sf1_values = self.eval_3form(sn1, out=self._sf1_values)
        rhof_values = self.eval_3form(rhon, out=self._rhof_values)

        if out is None:
            out = xp.zeros_like(rhof_values)

        # delta_s_values = s1_values-sf_values
        delta_s_values = self._delta_values
        delta_s_values *= 0.0
        delta_s_values += sf1_values
        delta_s_values -= sf_values

        # rho_mid_values = (rhof1_values+rhof_values)/2
        s_mid_values = self._sf_mid_values
        s_mid_values *= 0.0
        s_mid_values += sf1_values
        s_mid_values += sf_values
        s_mid_values /= 2.0

        eta = self.eta(delta_s_values, out=self._eta_values)

        e_rho_s1 = self.ener(
            rhof_values,
            sf1_values,
            out=self._en1_values,
        )
        e_rho_s = self.ener(
            rhof_values,
            sf_values,
            out=self._en_values,
        )

        de_rho_sm = self.dener_ds(
            rhof_values,
            s_mid_values,
            out=self._de_values,
        )

        # (eta*delta_s_values*(e_rho_s1-e_rho_s) / (delta_s_values**2+1e-40)+(1-eta)*de_rho_sm)

        # eta*delta_s_values*(e_rho_s1-e_rho_s) /(delta_s_values**2+1e-40)
        self._tmp_int_grid *= 0.0
        self._tmp_int_grid += e_rho_s1
        self._tmp_int_grid -= e_rho_s
        self._tmp_int_grid *= delta_s_values
        self._tmp_int_grid *= eta

        # delta_s_values**2+1e-40
        delta_s_values **= 2
        delta_s_values += 1e-40
        self._tmp_int_grid /= delta_s_values

        # (1-eta)
        eta -= 1.0
        eta *= -1.0

        # (1-eta)*de_rho_sm
        de_rho_sm *= eta

        out *= 0.0
        out += self._tmp_int_grid
        out += de_rho_sm

        return out

    def evaluate_exact_de_ds_grid(self, rhon, sn, out=None):
        r"""
        Evaluation of the derivative of :math:`E` with respect to :math:`s` on the grid.
        """
        rhof_values = self.eval_3form(rhon, out=self._rhof_values)
        sf_values = self.eval_3form(sn, out=self._sf_values)

        out = self.dener_ds(rhof_values, sf_values, out=out)
        return out

    def evaluate_discrete_d2e_drho2_grid(self, rhon, rhon1, sn, out=None):
        "Evaluate the derivative of the discrete derivative with respect to rhon1"
        # Get the value of the fields on the grid
        rhof_values = self.eval_3form(rhon, out=self._rhof_values)
        rhof1_values = self.eval_3form(rhon1, out=self._rhof1_values)
        sf_values = self.eval_3form(sn, out=self._sf_values)

        if out is None:
            out = xp.zeros_like(rhof_values)

        # delta_rho_values = rhof1_values-rhof_values
        delta_rho_values = self._delta_values
        delta_rho_values *= 0.0
        delta_rho_values += rhof1_values
        delta_rho_values -= rhof_values

        eta = self.eta(delta_rho_values)

        e_rho1_s = self.ener(
            rhof1_values,
            sf_values,
            out=self._en1_values,
        )
        e_rho_s = self.ener(
            rhof_values,
            sf_values,
            out=self._en_values,
        )

        de_rho1_s = self.dener_drho(
            rhof1_values,
            sf_values,
            out=self._de_values,
        )

        d2e_rho1_s = self.d2ener_drho2(
            rhof1_values,
            sf_values,
            out=self._d2e_values,
        )

        # eta*(de_rho1_s*delta_rho_values-e_rho1_s+e_rho_s)/(delta_rho_values**2+1e-40)
        self._DG_values *= 0.0
        self._DG_values += de_rho1_s
        self._DG_values *= delta_rho_values
        self._DG_values -= e_rho1_s
        self._DG_values += e_rho_s
        delta_rho_values **= 2
        delta_rho_values += 1e-40
        self._DG_values /= delta_rho_values
        self._DG_values *= eta

        # (1-eta)*d2e_rho1_s
        eta -= 1.0
        eta *= -1.0
        d2e_rho1_s *= eta

        # -metric_term * (DG_values + d2e_rho1_s)
        out *= 0.0
        out -= self._DG_values
        out -= d2e_rho1_s

        return out

    def evaluate_discrete_d2e_ds2_grid(self, rhon, sn, sn1, out=None):
        "Evaluate the derivative of the discrete derivative with respect to sn1"
        # Get the value of the fields on the grid
        rhof_values = self.eval_3form(rhon, out=self._rhof_values)
        sf_values = self.eval_3form(sn, out=self._sf_values)
        sf1_values = self.eval_3form(sn1, out=self._sf1_values)

        if out is None:
            out = xp.zeros_like(rhof_values)

        # delta_s_values = s1_values-sf_values
        delta_s_values = self._delta_values
        delta_s_values *= 0.0
        delta_s_values += sf1_values
        delta_s_values -= sf_values

        eta = self.eta(delta_s_values, out=self._eta_values)

        e_rho_s1 = self.ener(
            rhof_values,
            sf1_values,
            out=self._en1_values,
        )
        e_rho_s = self.ener(
            rhof_values,
            sf_values,
            out=self._en_values,
        )

        de_rho_s1 = self.dener_ds(
            rhof_values,
            sf1_values,
            out=self._de_values,
        )

        d2e_rho_s1 = self.d2ener_ds2(
            rhof_values,
            sf1_values,
            out=self._d2e_values,
        )

        # de_rho_s1*delta_s_values-e_rho_s1+e_rho_s
        out *= 0.0
        out += de_rho_s1
        out *= delta_s_values
        out -= e_rho_s1
        out += e_rho_s

        # (delta_s_values**2+1e-40)
        delta_s_values **= 2
        delta_s_values += 1e-40

        # eta*(de_rho_s1*delta_s_values-e_rho_s1+e_rho_s)/(delta_s_values**2+1e-40)
        out /= delta_s_values
        out *= eta

        # (1-eta)*d2e_rho_s1
        eta -= 1.0
        eta *= -1.0
        d2e_rho_s1 *= eta

        # -metric *(eta*(de_rho_s1*delta_s_values-e_rho_s1+e_rho_s)/(delta_s_values**2+1e-40) + (1-eta)*d2e_rho_s1)
        out += d2e_rho_s1
        out *= -1.0

        return out

    def eval_3form(self, coeffs, out=None):
        """Evaluate the 3 form with FE coefficient coeffs on the grid"""
        self._density_field.vector = coeffs
        f_values = self._density_field.eval_tp_fixed_loc(
            self.integration_grid_spans,
            self.integration_grid_bd,
            out=out,
        )

        return f_values


class H1vecMassMatrix_density:
    """Wrapper around a Weighted mass operator from H1vec to H1vec whose weights are given by a 3 form"""

    def __init__(self, derham, mass_ops, domain):
        self._massop = mass_ops.create_weighted_mass("H1vec", "H1vec")
        self.field = derham.create_spline_function("field", "L2")

        integration_grid = [grid_1d.flatten() for grid_1d in derham.V0splines.quad_grid_pts[0]]

        self.integration_grid_spans, self.integration_grid_bn, self.integration_grid_bd = derham.prepare_eval_tp_fixed(
            integration_grid,
        )

        grid_shape = tuple([len(loc_grid) for loc_grid in integration_grid])
        self._f_values = xp.zeros(grid_shape, dtype=float)

        metric = domain.metric(*integration_grid)
        self._mass_metric_term = deepcopy(metric)
        self._full_term_mass = deepcopy(metric)

    @property
    def massop(
        self,
    ):
        """The WeightedMassOperator"""
        return self._massop

    @property
    def inv(
        self,
    ):
        """The inverse WeightedMassOperator"""
        if not hasattr(self, "_inv"):
            self._create_inv()
        return self._inv

    def update_weight(self, coeffs):
        """Update the weighted mass matrix operator"""

        self.field.vector = coeffs
        f_values = self.field.eval_tp_fixed_loc(
            self.integration_grid_spans,
            self.integration_grid_bd,
            out=self._f_values,
        )
        for i in range(3):
            for j in range(3):
                self._full_term_mass[i, j] = f_values * self._mass_metric_term[i, j]

        self._massop.assemble(
            [
                [self._full_term_mass[0, 0], self._full_term_mass[0, 1], self._full_term_mass[0, 2]],
                [
                    self._full_term_mass[1, 0],
                    self._full_term_mass[
                        1,
                        1,
                    ],
                    self._full_term_mass[1, 2],
                ],
                [self._full_term_mass[2, 0], self._full_term_mass[2, 1], self._full_term_mass[2, 2]],
            ],
        )

        if hasattr(self, "_inv") and self._pc is not None:
            self._pc.update_mass_operator(self._massop)

    def _create_inv(
        self,
        type="pcg",
        pc_type="MassMatrixPreconditioner",
        tol=1e-16,
        maxiter=500,
        verbose=False,
    ):
        """Inverse the  weighted mass matrix"""
        if pc_type is None:
            self._pc = None
        else:
            pc_class = getattr(
                preconditioner,
                pc_type,
            )
            # dim_reduce=None: the Kronecker factors do not depend on the weights, cheap to update
            self._pc = pc_class(self.massop, dim_reduce=None)

        self._inv = inverse(
            self.massop,
            type,
            pc=self._pc,
            tol=tol,
            maxiter=maxiter,
            verbose=verbose,
            recycle=True,
        )


def kinetic_metric(derham: Derham, mass_ops, basis_ops, divdiv_alpha: float = 0.0) -> LinearOperator:
    r"""Metric :math:`\mathbb A[\hat\rho]` of the (regularized) kinetic energy :math:`\frac 12 \mathbf u^\top \mathbb A \mathbf u`
    of the variational fluid models,

    .. math::

        \frac 12 \mathbf u^\top \mathbb A[\hat\rho] \mathbf u = \int \rho \frac{|\mathbf u|^2}{2} \,\textnormal d \mathbf x
        + \alpha \int \rho\, (\nabla \cdot \mathbf u)^2 \,\textnormal d \mathbf x \,, \qquad
        \mathbb A[\hat\rho] = \mathbb M^v[\hat\rho] + 2 \alpha\, (\mathbb D \mathbb U^v)^\top \mathbb M^{3,\rho} \,\mathbb D \mathbb U^v \,,

    with :math:`\mathbb M^v[\hat\rho]` = ``mass_ops.WMMnew``, :math:`\mathbb M^{3,\rho}` = ``mass_ops.WMM3new`` and
    :math:`\mathbb U^v` = ``basis_ops.Uv``. The divergence of :math:`\mathbf u = DF \hat{\mathbf u}` is discretized as
    :math:`\sqrt g \, \nabla \cdot \mathbf u \approx \mathbb D \hat\Pi^2[\sqrt g \, \hat{\mathbf u}]`, the discrete divergence that
    is also used for the transport of pressure. The div-div term penalizes compressions (Hamiltonian regularization);
    both mass matrices depend linearly on the density and are re-assembled in
    :class:`~struphy.propagators.variational_density_evolve.VariationalDensityEvolve`.

    With :math:`\alpha > 0` the momentum :math:`\mathbb A \mathbf u` contains second derivatives of :math:`\mathbf u`,
    which makes the Picard iteration of
    :class:`~struphy.propagators.variational_momentum_advection.VariationalMomentumAdvection` converge only for
    small time steps (it diverges for larger :math:`\alpha \Delta t`); use a smaller time step or its Newton iteration.

    Parameters
    ----------
    derham : Derham
        Discrete de Rham sequence.

    mass_ops : WeightedMassOperators
        Weighted mass operators (provides ``WMMnew`` and ``WMM3new``).

    basis_ops : BasisProjectionOperators
        Basis projection operators (provides ``Uv``).

    divdiv_alpha : float
        Coefficient :math:`\alpha \geq 0` of the div-div term; for 0, the metric is ``mass_ops.WMMnew``.

    Returns
    -------
    LinearOperator
        The metric :math:`\mathbb A[\hat\rho]` (symmetric, positive definite).
    """
    assert divdiv_alpha >= 0.0, f"divdiv_alpha must be non-negative, got {divdiv_alpha}."
    if divdiv_alpha == 0.0:
        return mass_ops.WMMnew

    divUv = derham.div @ basis_ops.Uv
    return mass_ops.WMMnew + (2.0 * divdiv_alpha) * (divUv.T @ mass_ops.WMM3new @ divUv)


class KineticEnergyEvaluator:
    r"""Helper class to evaluate the different Kinetic energy terms appearing in VariationalDensityEvolve.

    This class only contains arrays corresponding to the integration grid to avoid the allocation of temporaries,
    methods that can be called to evaluate the energy and derivatives on the grid and weighted mass operators corresponding to integration against a vector field.

    Parameters
    ----------
    derham : Derham
        Discrete de Rham sequence.

    domain : Domain
        The domain in which the problem is discretized (needed for metric terms)

    mass_ops : WeightedMassOperators
        The weighted mass operators needed to create new mass matrices

    basis_ops : BasisProjectionOperators, optional
        Basis projection operators, needed for ``divdiv_alpha > 0`` (discrete divergence, see :func:`kinetic_metric`).

    divdiv_alpha : float
        Coefficient of the div-div term of the kinetic energy, see :func:`kinetic_metric`.
    """

    def __init__(self, derham, domain, mass_ops, basis_ops=None, divdiv_alpha: float = 0.0):
        integration_grid = [grid_1d.flatten() for grid_1d in derham.V0splines.quad_grid_pts[0]]

        self.integration_grid_spans, self.integration_grid_bn, self.integration_grid_bd = derham.prepare_eval_tp_fixed(
            integration_grid,
        )

        # tmps
        grid_shape = tuple([len(loc_grid) for loc_grid in integration_grid])

        self.uf = derham.create_spline_function("uf", "H1vec")
        self.uf1 = derham.create_spline_function("uf1", "H1vec")

        self._uf_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]
        self._uf1_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]
        self._Guf_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]
        self._tmp_int_grid = xp.zeros(grid_shape, dtype=float)

        metric = domain.metric(
            *integration_grid,
        ) * domain.jacobian_det(*integration_grid)
        self._proj_u2_metric_term = deepcopy(metric)

        metric = domain.metric(*integration_grid)
        self._mass_u_metric_term = deepcopy(metric)

        self._M_un = mass_ops.create_weighted_mass("H1vec", "L2")
        self._M_un1 = mass_ops.create_weighted_mass("L2", "H1vec")

        # div-div term alpha int rho (div u)^2 dx of the kinetic energy, with sqrt(g) div u = D Uv u (a 3-form)
        assert divdiv_alpha >= 0.0, f"divdiv_alpha must be non-negative, got {divdiv_alpha}."
        self._divdiv_alpha = divdiv_alpha
        if divdiv_alpha > 0.0:
            assert basis_ops is not None, "basis_ops are needed for divdiv_alpha > 0."
            self._divUv = derham.div @ basis_ops.Uv
            self._divuf = derham.create_spline_function("divuf", "L2")
            self._divuf1 = derham.create_spline_function("divuf1", "L2")
            self._divu = self._divUv.codomain.zeros()
            self._divu_values = xp.zeros(grid_shape, dtype=float)
            self._divu1_values = xp.zeros(grid_shape, dtype=float)
            self._divu_weight = xp.zeros(grid_shape, dtype=float)
            # 1/g, the push-forward of two 3-forms (see mass_ops.WMM3new)
            self._inv_g = 1.0 / domain.jacobian_det(*integration_grid) ** 2
            # 1/sqrt(g): get_u2_grid values are integrated with the 3-form geometric weight 1/sqrt(g)
            self._inv_sqrt_g = deepcopy(self._inv_g * domain.jacobian_det(*integration_grid))
            # V3 -> V3 parts of the derivatives of the div-div term, composed with D Uv below
            self._M3_divun = mass_ops.create_weighted_mass("L2", "L2")
            self._M3_divun1 = mass_ops.create_weighted_mass("L2", "L2")
            self._M_divun = self._M3_divun @ self._divUv
            self._M_divun1 = self._divUv.T @ self._M3_divun1

    @property
    def divdiv_alpha(self) -> float:
        """Coefficient of the div-div term of the kinetic energy (0: no regularization)."""
        return self._divdiv_alpha

    @property
    def M_divun(self) -> LinearOperator:
        r"""Operator H1vec -> L2, :math:`\mathbf v \mapsto \int \Lambda^3_{ijk} \frac{d^n \, (\mathbb D \mathbb U^v \mathbf v)^\top \vec{\boldsymbol\Lambda}^3}{g} \textnormal d \boldsymbol\eta`
        with :math:`d^n = (\mathbb D \mathbb U^v \mathbf u^n)^\top \vec{\boldsymbol\Lambda}^3` (see :meth:`assemble_M_divun`)."""
        return self._M_divun

    @property
    def M_divun1(self) -> LinearOperator:
        r"""Operator L2 -> H1vec, :math:`\delta\rho \mapsto (\mathbb D \mathbb U^v)^\top \mathbb M^3[d^{n+1}/g] \, \delta\rho`,
        such that :math:`2\alpha` times it is the derivative of the div-div part of :math:`\mathbb A[\hat\rho]\mathbf u^{n+1}`
        in :math:`\hat\rho` (see :meth:`assemble_M_divun1`)."""
        return self._M_divun1

    def _eval_divu(self, u, spline, out):
        """Values of the 3-form D Uv u on the integration grid."""
        self._divUv.dot(u, out=self._divu)
        spline.vector = self._divu
        return spline.eval_tp_fixed_loc(self.integration_grid_spans, self.integration_grid_bd, out=out)

    def add_divdiv_grid(self, un, un1, out):
        r"""Add :math:`\alpha \frac{d^n d^{n+1}}{\sqrt g}` to ``out`` (values on the integration grid), the div-div part of the
        :math:`\hat\rho`-derivative of the kinetic energy (to be integrated with the 3-form weight :math:`1/\sqrt g`, as
        :meth:`get_u2_grid`); :math:`d = (\mathbb D \mathbb U^v \mathbf u)^\top \vec{\boldsymbol\Lambda}^3`."""
        if self._divdiv_alpha == 0.0:
            return out
        d = self._eval_divu(un, self._divuf, self._divu_values)
        d1 = self._eval_divu(un1, self._divuf1, self._divu1_values)
        self._tmp_int_grid *= 0.0
        self._tmp_int_grid += d
        self._tmp_int_grid *= d1
        self._tmp_int_grid *= self._inv_sqrt_g
        self._tmp_int_grid *= self._divdiv_alpha
        out += self._tmp_int_grid
        return out

    def assemble_M_divun(self, un):
        """Assemble the weight :math:`d^n/g` of :attr:`M_divun`."""
        d = self._eval_divu(un, self._divuf, self._divu_values)
        self._divu_weight *= 0.0
        self._divu_weight += d
        self._divu_weight *= self._inv_g
        self._M3_divun.assemble([[self._divu_weight]])

    def assemble_M_divun1(self, un1):
        """Assemble the weight :math:`d^{n+1}/g` of :attr:`M_divun1`."""
        d1 = self._eval_divu(un1, self._divuf1, self._divu1_values)
        self._divu_weight *= 0.0
        self._divu_weight += d1
        self._divu_weight *= self._inv_g
        self._M3_divun1.assemble([[self._divu_weight]])

    @property
    def M_un(
        self,
    ):
        """Weighted mass matrix with domain H1vec et codomain L2
        represented the integration against a vector field in H1vec"""
        return self._M_un

    @property
    def M_un1(
        self,
    ):
        """Weighted mass matrix with domain L2 et codomain H1vec
        represented the integration against a vector field in H1vec"""
        return self._M_un1

    def get_u2_grid(self, un, un1, out):
        r"""Values of :math:`u_n \cdot u_{n+1}` represented by the coefficient un and un1, on the integration grid"""
        self.uf.vector = un
        self.uf1.vector = un1

        uf_values = self.uf.eval_tp_fixed_loc(
            self.integration_grid_spans,
            [
                self.integration_grid_bn,
            ]
            * 3,
            out=self._uf_values,
        )
        uf1_values = self.uf1.eval_tp_fixed_loc(
            self.integration_grid_spans,
            [
                self.integration_grid_bn,
            ]
            * 3,
            out=self._uf1_values,
        )

        out *= 0.0
        for i in range(3):
            for j in range(3):
                self._tmp_int_grid *= 0
                self._tmp_int_grid += uf_values[i]
                self._tmp_int_grid *= self._proj_u2_metric_term[i, j]
                self._tmp_int_grid *= uf1_values[j]
                out += self._tmp_int_grid

        out *= 0.5
        return out

    def assemble_M_un(self, un):
        """Update the weights of the matrix M_un with the vector fields given by the coeficient un"""
        self.uf.vector = un

        uf_values = self.uf.eval_tp_fixed_loc(
            self.integration_grid_spans,
            [
                self.integration_grid_bn,
            ]
            * 3,
            out=self._uf_values,
        )

        for i in range(3):
            self._Guf_values[i] *= 0.0
            for j in range(3):
                self._tmp_int_grid *= 0.0
                self._tmp_int_grid += self._mass_u_metric_term[i, j]
                self._tmp_int_grid *= uf_values[j]
                self._Guf_values[i] += self._tmp_int_grid

        self._M_un.assemble(
            [[self._Guf_values[0], self._Guf_values[1], self._Guf_values[2]]],
        )

    def assemble_M_un1(self, un1):
        """Update the weights of the matrix M_un1 with the vector fields given by the coeficient un1"""
        self.uf1.vector = un1

        uf1_values = self.uf1.eval_tp_fixed_loc(
            self.integration_grid_spans,
            [
                self.integration_grid_bn,
            ]
            * 3,
            out=self._uf1_values,
        )

        for i in range(3):
            self._Guf_values[i] *= 0.0
            for j in range(3):
                self._tmp_int_grid *= 0.0
                self._tmp_int_grid += self._mass_u_metric_term[i, j]
                self._tmp_int_grid *= uf1_values[j]
                self._Guf_values[i] += self._tmp_int_grid

        self._M_un1.assemble(
            [[self._Guf_values[0]], [self._Guf_values[1]], [self._Guf_values[2]]],
        )


def project_div_free_velocity(
    u: Vector,
    derham: Derham,
    mass_ops,
    basis_ops,
    tol: float = 1e-8,
    maxiter: int = 1000,
    inner_tol: float = 1e-12,
) -> dict:
    r"""Project an H1vec velocity in place onto the discretely divergence-free fields of the variational schemes.

    The variational propagators compute the divergence of :math:`\mathbf u \in V^v_h` as
    :math:`\mathbb G \mathbf u` with :math:`\mathbb G = \mathbb D\, \mathcal U^v`, where
    :math:`\mathcal U^v = \hat\Pi^2[\sqrt g\, \vec\Lambda^v]` maps to H(div) and :math:`\mathbb D` is the discrete
    divergence (see :class:`Pressure_transport_operator`). The projection of a continuous divergence-free field is
    divergence-free only up to the interpolation error. This function replaces :math:`\mathbf u` by its
    :math:`\mathbb M^v`-orthogonal projection onto :math:`\ker \mathbb G`,

    .. math::

        (\mathbb G (\mathbb M^v)^{-1} \mathbb G^\top) \boldsymbol\lambda = \mathbb G \mathbf u\,,\qquad
        \mathbf u \leftarrow \mathbf u - (\mathbb M^v)^{-1} \mathbb G^\top \boldsymbol\lambda\,,

    so that :math:`\mathbb G \mathbf u = 0` up to the solver tolerance, with the smallest change of
    :math:`\mathbf u` in the :math:`\mathbb M^v` norm. The Schur system is singular (with :math:`u \cdot n = 0` the
    total flux vanishes) but consistent, and is solved with CG; the inner :math:`\mathbb M^v` solves use PCG with the
    mass-matrix preconditioner. The solvers take absolute tolerances (Euclidean norms of the coefficient residuals),
    so they are scaled with the right-hand side: ``tol`` relative to :math:`\mathbb G \mathbf u` and ``inner_tol``
    relative to :math:`\mathbb G^\top \mathbb G \mathbf u`. Must be called on all MPI ranks.

    Parameters
    ----------
    u : Vector
        Coefficients of the velocity in :math:`V^v_h` (H1vec), modified in place.
    derham : Derham
        The discrete de Rham sequence.
    mass_ops : WeightedMassOperators
        Mass matrices (``Mv`` and ``M3`` are used).
    basis_ops : BasisProjectionOperators
        Basis projection operators (``Uv`` is used).
    tol : float
        Tolerance of the CG solve of the Schur system, relative to the (Euclidean) norm of its right-hand side.
    maxiter : int
        Maximum number of CG iterations of the Schur system.
    inner_tol : float
        Tolerance of the inner PCG solves with ``Mv``, relative to the (Euclidean) norm of
        :math:`\mathbb G^\top \mathbb G \mathbf u`.

    Returns
    -------
    dict
        ``div_before`` and ``div_after``: the :math:`L^2` norms of :math:`\mathbb G \mathbf u` before and after;
        ``delta_rel``: the relative change :math:`\|\delta \mathbf u\|_{\mathbb M^v} / \|\mathbf u\|_{\mathbb M^v}`;
        ``niter``, ``success`` and ``res_norm`` (relative) of the Schur solve.
    """
    Mv, M3 = mass_ops.Mv, mass_ops.M3
    G = derham.div @ basis_ops.Uv
    # built once: the transpose of the basis projection operator is assembled anew at each call of .T
    GT = G.T

    out = {"div_before": 0.0, "div_after": 0.0, "delta_rel": 0.0, "niter": 0, "success": True, "res_norm": 0.0}
    norm_u = xp.sqrt(Mv.dot_inner(u, u))
    if norm_u == 0.0:
        return out

    div_u = G.dot(u)
    out["div_before"] = float(xp.sqrt(M3.dot_inner(div_u, div_u)))
    norm_rhs = float(xp.sqrt(div_u.inner(div_u)))
    if norm_rhs == 0.0:
        return out
    norm_inner_rhs = float(xp.sqrt(GT.dot(div_u).inner(GT.dot(div_u))))

    Mv_inv = inverse(
        Mv,
        "pcg",
        pc=preconditioner.MassMatrixPreconditioner(Mv),
        tol=inner_tol * norm_inner_rhs,
        maxiter=1000,
        verbose=False,
    )
    schur_inv = inverse(G @ Mv_inv @ GT, "cg", tol=tol * norm_rhs, maxiter=maxiter, verbose=False)

    lam = schur_inv.dot(div_u)
    delta = Mv_inv.dot(GT.dot(lam))
    u -= delta
    u.update_ghost_regions()

    div_u = G.dot(u)
    info = schur_inv.get_info()
    out["div_after"] = float(xp.sqrt(M3.dot_inner(div_u, div_u)))
    out["delta_rel"] = float(xp.sqrt(Mv.dot_inner(delta, delta)) / norm_u)
    out["niter"] = info["niter"]
    out["success"] = bool(info["success"])
    out["res_norm"] = float(info["res_norm"]) / norm_rhs
    if not out["success"]:
        logger.warning(
            f"project_div_free_velocity: CG not converged after {out['niter']} iterations "
            f"(relative residual {out['res_norm']:.2e}, tolerance {tol:.1e})."
        )
    return out


class NewtonConvergence:
    r"""Stopping criterion of the Newton and Picard iterations of the variational propagators.

    The squared residual norm of an iteration is compared with ``tol**2`` times ``min(err_ref, 1)``, where ``err_ref``
    is the squared norm of the state at :math:`t^n`: relative for small states, such as the perturbations of the
    linear and perturbative models, and absolute (as for the full models) for states of norm one or larger, or zero.
    The iteration also stops, with a warning, when the error has not decreased in ``patience`` consecutive iterations
    (stagnation at the accuracy of the linear solves).

    Parameters
    ----------
    name : str
        Name of the propagator, for the warnings.
    tol : float
        Tolerance of the relative residual norm.
    err_ref : float
        Squared norm of the state at :math:`t^n`, measured as the residual.
    patience : int
        Number of consecutive iterations without decrease of the error after which the iteration stops.
    """

    def __init__(self, name: str, tol: float, err_ref: float, patience: int = 3):
        self._name = name
        self._tol = tol
        self._err_ref = min(float(err_ref), 1.0) if err_ref > 0.0 else 1.0
        self._patience = patience
        self._err_min = xp.inf
        self._n_no_decrease = 0
        self.err = xp.inf
        self.converged = False
        self.stagnated = False

    def stop(self, err: float, it: int, min_iter: int = 0) -> bool:
        """Whether to stop at iteration ``it``, given the squared (absolute) residual norm ``err``.

        No stop before iteration ``min_iter``, except for NaN."""
        self.err = float(err) / self._err_ref
        if xp.isnan(self.err):
            return True
        if it >= min_iter and self.err < self._tol**2:
            self.converged = True
            return True
        if self.err < self._err_min:
            self._err_min = self.err
            self._n_no_decrease = 0
        else:
            self._n_no_decrease += 1
        if it >= min_iter and self._n_no_decrease >= self._patience:
            self.stagnated = True
            logger.warning(
                f"{self._name}: Newton iteration stagnated at iteration {it}, relative error {self.err:.3e} "
                f"(smallest {self._err_min:.3e}, tol**2 = {self._tol**2:.1e}).",
            )
            return True
        return False

    def warn_if_not_converged(self, it: int, maxiter: int):
        """Warning if the iteration ended at ``maxiter`` or with NaN (stagnation is reported by :meth:`stop`)."""
        if not self.converged and not self.stagnated and (it == maxiter - 1 or xp.isnan(self.err)):
            logger.warning(
                f"Maximum iteration in {self._name} reached - not converged: relative error {self.err:.3e}, "
                f"tol**2 = {self._tol**2:.1e}",
            )


def dot_normalized(op: LinearOperator, v: Vector, out: Vector) -> Vector:
    """``op.dot(v, out=out)`` for a linear operator with iterative solves of absolute tolerance (e.g. an inverse).

    The operator is applied to :math:`v / |v|` and the result scaled back, so that the solver tolerances are relative
    to the Euclidean norm of ``v``. ``v`` is restored (up to round-off); ``out`` must not be ``v``."""
    norm = xp.sqrt(v.inner(v))
    if not norm > 0.0:
        out *= 0.0
        return out
    v *= 1.0 / norm
    op.dot(v, out=out)
    v *= norm
    out *= norm
    return out


def inv_norm2(inv: LinearOperator, v: Vector) -> float:
    """The squared norm ``v . inv(v)`` with an inverse (mass matrix) of absolute tolerance, applied to
    :math:`v / |v|`, so that the tolerance is relative. ``v`` is restored (up to round-off)."""
    norm = xp.sqrt(v.inner(v))
    if not norm > 0.0:
        return 0.0
    v *= 1.0 / norm
    err = inv.dot_inner(v, v)
    v *= norm
    return norm**2 * err


class LinearPressureHeating:
    r"""Heating of the auxiliary pressure :math:`\hat p` (3-form) of the linear p model, whose thermal energy
    :math:`\frac{1}{\gamma - 1} \int \hat p^3 \,\textnormal d \boldsymbol \eta` is linear in :math:`\hat p`, by a
    dissipated energy density :math:`e` (per unit logical volume, given at the quadrature points of :math:`V^3_h`):

    .. math::

        \hat{\mathbf p} \leftarrow \hat{\mathbf p} + (\gamma - 1)\, \tilde{\mathbb M}^{-1} \mathbf e \,, \qquad
        \tilde{\mathbb M}_{ij} = \int \Lambda^3_i \Lambda^3_j \,\textnormal d \boldsymbol \eta \,, \qquad
        \mathbf e_i = \int e\, \Lambda^3_i \,\textnormal d \boldsymbol \eta \,.

    This is the :math:`L^2` projection in the logical inner product. Since the constants are in :math:`V^3_h`, it
    preserves the integral, so the thermal energy increases by exactly :math:`\int e \,\textnormal d \boldsymbol \eta`
    (up to the solver tolerance) on any mapping. The projection with the weight :math:`1/\sqrt g` of the mass matrix
    :math:`\mathbb M^3` would do so only if :math:`\sqrt g \in V^3_h`.

    Parameters
    ----------
    derham : Derham
        Discrete de Rham sequence.

    mass_ops : WeightedMassOperators
        Weighted mass operators.

    domain : Domain
        The mapping (for :math:`\sqrt g` at the quadrature points).

    gamma : float
        Adiabatic index.

    tol : float
        Relative tolerance of the mass solve.
    """

    def __init__(self, derham: Derham, mass_ops, domain, gamma: float, tol: float = 1e-12):
        from struphy.feec.mass import L2Projector

        self._gamma = gamma
        grid = [pts.flatten() for pts in derham.V3splines.quad_grid_pts[0]]
        self._sqrt_g = domain.jacobian_det(*grid)
        self._mass = mass_ops.create_weighted_mass("L2", "L2")
        self._mass.assemble([[xp.ones_like(self._sqrt_g)]])
        self._inv = inverse(
            self._mass,
            "pcg",
            pc=preconditioner.MassMatrixPreconditioner(self._mass, dim_reduce=None),
            tol=tol,
            maxiter=1000,
        )
        self._get_dofs = L2Projector("L2", mass_ops).get_dofs
        self._e = derham.V3pol.zeros()
        self._incr = derham.V3pol.zeros()
        self._tmp = xp.zeros_like(self._sqrt_g)

    @property
    def grid_shape(self) -> tuple:
        """Shape of the arrays of values at the quadrature points of :math:`V^3_h`."""
        return self._sqrt_g.shape

    def __call__(self, e: xp.ndarray, pt3: Vector):
        """Add the heating by the energy density ``e`` (values at the quadrature points) to ``pt3`` (in place)."""
        # get_dofs integrates with the weight 1/sqrt(g) of 3-forms
        tmp = self._tmp
        tmp[:] = e
        tmp *= self._sqrt_g
        self._get_dofs(tmp, dofs=self._e)
        incr = dot_normalized(self._inv, self._e, self._incr)
        incr *= self._gamma - 1.0
        pt3 += incr


def alfven_schur_preconditioner(propagator, unmet: list[str], u_space: str = "H1vec"):
    r"""The Kronecker preconditioner of the shear-Alfven Schur operator of a linear variational propagator,

    .. math::

        \mathcal S_A = \mathbb M^v[\rho_0] + \frac{\Delta t^2}{4}\, \mathbb T_0^\top \mathbb M^2\, \mathbb T_0 \,,
        \qquad \mathbb T_0 = \mathbb C\, \hat\Pi^1[\hat{\mathbf B}_0 \times \vec\Lambda^v] \,,

    as :class:`~struphy.feec.preconditioner.ShearAlfvenSchurPreconditioner` with ``u_space="H1vec"``: the
    :class:`~struphy.feec.preconditioner.StiffnessPreconditioner` of the Kronecker approximation of
    :math:`\mathbb C\, \hat\Pi^1[\mathbf B_0 \times \cdot\,]` (``basis_ops.Tv``, analytic equilibrium) with the
    stabilization :math:`\mathbb M^v[\rho_0]` with the projected density of the Schur operator (``WMMnew``), plus the
    auxiliary space correction of
    the :math:`\mathbf E \times \mathbf B` flows if ``options.exb_correction``. Its :meth:`build` gives the
    preconditioner for a time step. For a velocity in :math:`H(\textnormal{div})` (``u_space="Hdiv"``), the operators
    are ``basis_ops.T2`` and ``mass_ops.M2n``, as in
    :class:`~struphy.propagators.shear_alfven_propagator.ShearAlfvenPropagator`.

    Parameters
    ----------
    propagator : Propagator
        The propagator, with the options ``schur_precond``, ``solver`` and ``exb_correction``.
    unmet : list[str]
        Requirements of the propagator that do not hold (e.g. ``'model="linear"'``).
    u_space : str
        Space of the velocity, ``"H1vec"`` or ``"Hdiv"``.

    Returns
    -------
    ShearAlfvenSchurPreconditioner | None
        None if ``options.schur_precond`` is not ``"StiffnessPreconditioner"`` or if a requirement does not hold
        (then with a warning: the mass-matrix preconditioner is used).
    """
    from struphy.feec.preconditioner import ShearAlfvenSchurPreconditioner

    options = propagator.options
    if options.schur_precond != "StiffnessPreconditioner":
        return None
    unmet = list(unmet)
    if options.solver not in ("pcg", "pbicgstab", "rpbicgstab"):
        unmet += ['solver="pcg", "pbicgstab" or "rpbicgstab"']
    if propagator.derham.polar_splines:
        unmet += ["tensor-product splines"]
    if propagator.derham.with_local_projectors:
        unmet += ["the global commuting projectors"]
    if unmet:
        logger.warning(
            f'{propagator.__class__.__name__}: schur_precond="StiffnessPreconditioner" needs {", ".join(unmet)}; '
            "falling back to the mass-matrix preconditioner."
        )
        return None
    if u_space == "Hdiv":
        return ShearAlfvenSchurPreconditioner(
            propagator.derham,
            propagator.mass_ops,
            propagator.basis_ops,
            propagator.basis_ops.T2,
            "M2n",
            exb_correction=options.exb_correction,
            u_space="Hdiv",
        )
    # the velocity mass matrix of the Schur operator, G weighted with the projected density: WMMnew stores only G as
    # weight and evaluates its density spline during the assembly, so the density is passed here as a function, which
    # is evaluated into the weights seen by the separable approximation of the preconditioner
    rho = propagator.mass_ops.WMMnew.spline_functions["l2_field"]
    stab = propagator.mass_ops.create_weighted_mass(
        "H1vec",
        "H1vec",
        weights=("G", lambda e1, e2, e3: rho(e1, e2, e3)),
        name="WMM_alfven_precond",
        assemble=False,
    )
    return ShearAlfvenSchurPreconditioner(
        propagator.derham,
        propagator.mass_ops,
        propagator.basis_ops,
        propagator.basis_ops.Tv,
        stab,
        exb_correction=options.exb_correction,
        u_space="H1vec",
    )
