from copy import deepcopy

from psydac.linalg.basic import IdentityOperator, Vector
from psydac.linalg.block import BlockVector
from psydac.linalg.solvers import inverse

from struphy.feec import preconditioner
from struphy.feec.basis_projection_ops import (
    BasisProjectionOperator,
    BasisProjectionOperatorLocal,
    CoordinateProjector,
)
from struphy.feec.linear_operators import LinOpWithTransp
from struphy.feec.psydac_derham import Derham
from struphy.utils.arrays import xp as np


class BracketOperator(LinOpWithTransp):
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

    Initialized and used in :class:`~struphy.propagators.propagators_fields.VariationalMomentumAdvection` propagator.

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
        Xh = derham.Vh_fem["v"]
        V1h = derham.Vh_fem["1"]
        self._domain = derham.Vh_pol["v"]
        self._codomain = derham.Vh_pol["v"]
        self._dtype = Xh.coeff_space.dtype
        self._u = u

        # tmp for evaluating u
        self.vf = derham.create_spline_function("uf", "H1vec")
        self.gv1f = derham.create_spline_function("gu1f", "Hcurl")  # grad(u[0])
        self.gv2f = derham.create_spline_function("gu2f", "Hcurl")  # grad(u[1])
        self.gv3f = derham.create_spline_function("gu3f", "Hcurl")  # grad(u[2])

        self.gp1v = derham.Vh_pol["1"].zeros()
        self.gp2v = derham.Vh_pol["1"].zeros()
        self.gp3v = derham.Vh_pol["1"].zeros()

        P0 = derham.P["0"]
        # Initialize the CoordinateProjectors
        # self.Pcoord1 = CoordinateProjector(0, Xh, V0h)
        # self.Pcoord2 = CoordinateProjector(1, Xh, V0h)
        # self.Pcoord3 = CoordinateProjector(2, Xh, V0h)
        self.Pcoord1 = CoordinateProjector(0, derham.Vh_pol["v"], derham.Vh_pol["0"]) @ derham.boundary_ops["v"]
        self.Pcoord2 = CoordinateProjector(1, derham.Vh_pol["v"], derham.Vh_pol["0"]) @ derham.boundary_ops["v"]
        self.Pcoord3 = CoordinateProjector(2, derham.Vh_pol["v"], derham.Vh_pol["0"]) @ derham.boundary_ops["v"]

        # Initialize the BasisProjectionOperators
        if derham._with_local_projectors == True:
            self.PiuT = BasisProjectionOperatorLocal(
                P0,
                V1h,
                [[None, None, None]],
                transposed=True,
                V_extraction_op=derham.extraction_ops["1"],
                V_boundary_op=IdentityOperator(derham.Vh_pol["1"]),
                P_boundary_op=IdentityOperator(derham.Vh_pol["0"]),
            )

            self.PigvT_1 = BasisProjectionOperatorLocal(
                P0,
                Xh,
                [[None, None, None]],
                transposed=True,
                V_extraction_op=derham.extraction_ops["v"],
                V_boundary_op=derham.boundary_ops["v"],
                P_boundary_op=IdentityOperator(derham.Vh_pol["0"]),
            )

            self.PigvT_2 = BasisProjectionOperatorLocal(
                P0,
                Xh,
                [[None, None, None]],
                transposed=True,
                V_extraction_op=derham.extraction_ops["v"],
                V_boundary_op=derham.boundary_ops["v"],
                P_boundary_op=IdentityOperator(derham.Vh_pol["0"]),
            )

            self.PigvT_3 = BasisProjectionOperatorLocal(
                P0,
                Xh,
                [[None, None, None]],
                transposed=True,
                V_extraction_op=derham.extraction_ops["v"],
                V_boundary_op=derham.boundary_ops["v"],
                P_boundary_op=IdentityOperator(derham.Vh_pol["0"]),
            )
        else:
            self.PiuT = BasisProjectionOperator(
                P0,
                V1h,
                [[None, None, None]],
                transposed=True,
                use_cache=True,
                V_extraction_op=derham.extraction_ops["1"],
                V_boundary_op=IdentityOperator(derham.Vh_pol["1"]),
                P_boundary_op=IdentityOperator(derham.Vh_pol["0"]),
            )

            self.PigvT_1 = BasisProjectionOperator(
                P0,
                Xh,
                [[None, None, None]],
                transposed=True,
                use_cache=True,
                V_extraction_op=derham.extraction_ops["v"],
                V_boundary_op=derham.boundary_ops["v"],
                P_boundary_op=IdentityOperator(derham.Vh_pol["0"]),
            )
            self.PigvT_2 = BasisProjectionOperator(
                P0,
                Xh,
                [[None, None, None]],
                transposed=True,
                use_cache=True,
                V_extraction_op=derham.extraction_ops["v"],
                V_boundary_op=derham.boundary_ops["v"],
                P_boundary_op=IdentityOperator(derham.Vh_pol["0"]),
            )
            self.PigvT_3 = BasisProjectionOperator(
                P0,
                Xh,
                [[None, None, None]],
                transposed=True,
                use_cache=True,
                V_extraction_op=derham.extraction_ops["v"],
                V_boundary_op=derham.boundary_ops["v"],
                P_boundary_op=IdentityOperator(derham.Vh_pol["0"]),
            )

        # Store the interpolation grid for later use in _update_all_weights
        interpolation_grid = [pts.flatten() for pts in derham.proj_grid_pts["0"]]

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
        self._vf_values = [np.zeros(grid_shape, dtype=float) for i in range(3)]
        self._gvf1_values = [np.zeros(grid_shape, dtype=float) for i in range(3)]
        self._gvf2_values = [np.zeros(grid_shape, dtype=float) for i in range(3)]
        self._gvf3_values = [np.zeros(grid_shape, dtype=float) for i in range(3)]

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

    @property
    def tosparse(self):
        raise NotImplementedError()

    @property
    def toarray(self):
        raise NotImplementedError()

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
            self.interpolation_grid_spans, [self.interpolation_grid_bn] * 3, out=self._vf_values
        )

        gvf1_values = self.gv1f.eval_tp_fixed_loc(
            self.interpolation_grid_spans, self.interpolation_grid_gradient, out=self._gvf1_values
        )

        gvf2_values = self.gv2f.eval_tp_fixed_loc(
            self.interpolation_grid_spans, self.interpolation_grid_gradient, out=self._gvf2_values
        )

        gvf3_values = self.gv3f.eval_tp_fixed_loc(
            self.interpolation_grid_spans, self.interpolation_grid_gradient, out=self._gvf3_values
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


class L2_transport_operator(LinOpWithTransp):
    r"""
    Operator

    .. math::
        \mathbf u \mapsto \nabla \cdot(\Pi^2(\rho \mathbf u)) \,
    from H1vec to L2, where :math:`\rho` is a discrete 3-form which can be updated.

    Parameters
    ----------
    derham : Derham
        Discrete de Rham sequence.

    transposed : Bool
        Assemble the transposed operator
    """

    def __init__(self, derham, transposed=False, weights=None):
        # Get the projector and the spaces
        self._derham = derham
        self._transposed = transposed
        if weights == None:
            weights = [[None] * 3] * 3
        self._weights = weights
        if self._transposed:
            self._codomain = self._derham.Vh_pol["v"]
            self._domain = self._derham.Vh_pol["3"]
        else:
            self._domain = self._derham.Vh_pol["v"]
            self._codomain = self._derham.Vh_pol["3"]
        P2 = self._derham.P["2"]
        Xh = self._derham.Vh_fem["v"]
        self._dtype = Xh.coeff_space.dtype
        self.field = self._derham.create_spline_function("rhof", "L2")

        # Initialize the BasisProjectionOperator
        if self._derham._with_local_projectors:
            self.Proj = BasisProjectionOperatorLocal(
                P2,
                Xh,
                self._weights,
                transposed=transposed,
                V_extraction_op=self._derham.extraction_ops["v"],
                V_boundary_op=self._derham.boundary_ops["v"],
                P_boundary_op=IdentityOperator(self._derham.Vh_pol["2"]),
            )

        else:
            self.Proj = BasisProjectionOperator(
                P2,
                Xh,
                self._weights,
                transposed=transposed,
                use_cache=True,
                V_extraction_op=self._derham.extraction_ops["v"],
                V_boundary_op=self._derham.boundary_ops["v"],
                P_boundary_op=IdentityOperator(self._derham.Vh_pol["2"]),
            )

        # divergence
        self.div = self._derham.div_bcfree

        # Initialize the transport operator and transposed
        if self._transposed:
            self._op = self.Proj @ self.div.T
        else:
            self._op = self.div @ self.Proj

        hist_grid = self._derham.proj_grid_pts["2"]

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
        self._f_0_values = np.zeros(grid_shape, dtype=float)

        grid_shape = tuple([len(loc_grid) for loc_grid in hist_grid_1])
        self._f_1_values = np.zeros(grid_shape, dtype=float)

        grid_shape = tuple([len(loc_grid) for loc_grid in hist_grid_2])
        self._f_2_values = np.zeros(grid_shape, dtype=float)

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def dtype(self):
        return self._dtype

    @property
    def tosparse(self):
        raise NotImplementedError()

    @property
    def toarray(self):
        raise NotImplementedError()

    def transpose(self, conjugate=False):
        return L2_transport_operator(self._derham, not self._transposed, weights=self._weights)

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

        self._weights = [
            [f0_values, None, None],
            [None, f1_values, None],
            [None, None, f2_values],
        ]

        self.Proj.update_weights(self._weights)


class Hdiv0_transport_operator(LinOpWithTransp):
    r"""
    Operator

    .. math::
        u \mapsto \nabla \times (\Pi^1(\mathbf B \times \mathbf u)) \,
    from H1vec to H(div), where :math:`\mathbf B` is a discrete 2-form which can be updated.

    Parameters
    ----------
    derham : Derham
        Discrete de Rham sequence.

    transposed : Bool
        Assemble the transposed operator
    """

    def __init__(self, derham, transposed=False, weights=None):
        # Get the projector and the spaces
        self._derham = derham
        self._transposed = transposed
        if weights == None:
            weights = [[None] * 3] * 3
        self._weights = weights
        if self._transposed:
            self._codomain = self._derham.Vh_pol["v"]
            self._domain = self._derham.Vh_pol["2"]
        else:
            self._domain = self._derham.Vh_pol["v"]
            self._codomain = self._derham.Vh_pol["2"]
        P1 = self._derham.P["1"]
        Xh = self._derham.Vh_fem["v"]
        self._dtype = Xh.coeff_space.dtype
        self.field = self._derham.create_spline_function("Bf", "Hdiv")

        # Initialize the BasisProjectionOperators
        if self._derham._with_local_projectors:
            self.Proj = BasisProjectionOperatorLocal(
                P1,
                Xh,
                self._weights,
                transposed=transposed,
                V_extraction_op=self._derham.extraction_ops["v"],
                V_boundary_op=self._derham.boundary_ops["v"],
                P_boundary_op=IdentityOperator(self._derham.Vh_pol["1"]),
            )

        else:
            self.Proj = BasisProjectionOperator(
                P1,
                Xh,
                self._weights,
                transposed=transposed,
                use_cache=True,
                V_extraction_op=self._derham.extraction_ops["v"],
                V_boundary_op=self._derham.boundary_ops["v"],
                P_boundary_op=IdentityOperator(self._derham.Vh_pol["1"]),
            )

        # gradient of the component of the vector field
        self.curl = self._derham.curl_bcfree

        # Initialize the transport operator and transposed
        if self._transposed:
            self._op = self.Proj @ self.curl.T
        else:
            self._op = self.curl @ self.Proj

        hist_grid = self._derham.proj_grid_pts["1"]

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
        self._bf0_values = [np.zeros(grid_shape, dtype=float) for i in range(3)]
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
        self._bf1_values = [np.zeros(grid_shape, dtype=float) for i in range(3)]
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
        self._bf2_values = [np.zeros(grid_shape, dtype=float) for i in range(3)]
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

    @property
    def tosparse(self):
        raise NotImplementedError()

    @property
    def toarray(self):
        raise NotImplementedError()

    def transpose(self, conjugate=False):
        return Hdiv0_transport_operator(self._derham, not self._transposed, weights=self._weights)

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

        self._weights = [
            [None, -bf0_values[2], bf0_values[1]],
            [bf1_values[2], None, -bf1_values[0]],
            [-bf2_values[1], bf2_values[0], None],
        ]

        self.Proj.update_weights(self._weights)


class Pressure_transport_operator(LinOpWithTransp):
    r"""
    Operator

    .. math::
        \mathbf u \mapsto \nabla \cdot (\Pi^2(p \mathbf u)) + (\gamma -1) \Pi^3(p \nabla \cdot \Pi^2(\mathbf u))  \,
    from H1vec to L2, where :math:`p` is a discrete 3-form which can be updated.

    Parameters
    ----------
    derham : Derham
        Discrete de Rham sequence.

    phys_domain : Domain
        The domain in which the problem is discretized (needed for metric terms)

    Uv : BasisProjectionOperator
        The projection from H1vec to H(div)

    gamma : Float
        Thermodynamical constant

    transposed : Bool
        Assemble the transposed operator
    """

    def __init__(self, derham, phys_domain, Uv, gamma, transposed=False, weights1=None, weights2=None):
        # Get the projector and the spaces
        self._derham = derham
        self._phys_domain = phys_domain
        self._Uv = Uv
        self._transposed = transposed
        self._gamma = gamma
        if weights1 is None:
            weights1 = [[None] * 3] * 3
        self._weights1 = weights1
        if weights2 is None:
            weights2 = [[lambda eta1, eta2, eta3: 0 * eta1]]
        self._weights2 = weights2
        if self._transposed:
            self._codomain = self._derham.Vh_pol["v"]
            self._domain = self._derham.Vh_pol["3"]
        else:
            self._domain = self._derham.Vh_pol["v"]
            self._codomain = self._derham.Vh_pol["3"]
        P2 = self._derham.P["2"]
        P3 = self._derham.P["3"]
        Xh = self._derham.Vh_fem["v"]
        V3h = self._derham.Vh_fem["3"]
        self._dtype = Xh.coeff_space.dtype
        self.field = self._derham.create_spline_function("pf", "L2")

        self.Pip = BasisProjectionOperator(
            P2,
            Xh,
            self._weights1,
            transposed=transposed,
            use_cache=True,
            V_extraction_op=self._derham.extraction_ops["v"],
            V_boundary_op=self._derham.boundary_ops["v"],
            P_boundary_op=IdentityOperator(self._derham.Vh_pol["2"]),
        )

        self.Pip_div = BasisProjectionOperator(
            P3,
            V3h,
            self._weights2,
            transposed=transposed,
            use_cache=True,
            V_extraction_op=self._derham.extraction_ops["3"],
            V_boundary_op=self._derham.boundary_ops["3"],
            P_boundary_op=IdentityOperator(self._derham.Vh_pol["3"]),
        )

        # BC?

        div = self._derham.div

        self.div = div @ Uv

        # Initialize the transport operator and transposed
        if self._transposed:
            self._op = self.Pip @ div.T + self.div.T @ self.Pip_div

        else:
            self._op = div @ self.Pip + self.Pip_div @ self.div

        int_grid = [pts.flatten() for pts in self._derham.proj_grid_pts["3"]]

        self.int_grid_spans, self.int_grid_bn, self.int_grid_bd = self._derham.prepare_eval_tp_fixed(
            int_grid,
        )

        metric = 1.0 / phys_domain.jacobian_det(*int_grid)
        self._proj_p_metric = deepcopy(metric)

        grid_shape = tuple([len(loc_grid) for loc_grid in int_grid])
        self._pf_values = np.zeros(grid_shape, dtype=float)
        self._mapped_pf_values = np.zeros(grid_shape, dtype=float)

        # gradient of the component of the vector field

        hist_grid_P2 = self._derham.proj_grid_pts["2"]

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
        self._pf_0_values = np.zeros(grid_shape, dtype=float)

        grid_shape = tuple([len(loc_grid) for loc_grid in hist_grid_21])
        self._pf_1_values = np.zeros(grid_shape, dtype=float)

        grid_shape = tuple([len(loc_grid) for loc_grid in hist_grid_22])
        self._pf_2_values = np.zeros(grid_shape, dtype=float)

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def dtype(self):
        return self._dtype

    @property
    def tosparse(self):
        raise NotImplementedError()

    @property
    def toarray(self):
        raise NotImplementedError()

    def transpose(self, conjugate=False):
        return Pressure_transport_operator(
            self._derham,
            self._phys_domain,
            self._Uv,
            self._gamma,
            not self._transposed,
            weights1=self._weights1,
            weights2=self._weights2,
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

        # print(self.Pip_divT._dof_mat._data)

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
        integration_grid = [grid_1d.flatten() for grid_1d in self._derham.quad_grid_pts["0"]]

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
        self._rhof_values = np.zeros(grid_shape, dtype=float)
        self._rhof1_values = np.zeros(grid_shape, dtype=float)
        self._sf_values = np.zeros(grid_shape, dtype=float)
        self._sf1_values = np.zeros(grid_shape, dtype=float)
        self._delta_values = np.zeros(grid_shape, dtype=float)
        self._rhof_mid_values = np.zeros(grid_shape, dtype=float)
        self._sf_mid_values = np.zeros(grid_shape, dtype=float)
        self._eta_values = np.zeros(grid_shape, dtype=float)
        self._en_values = np.zeros(grid_shape, dtype=float)
        self._en1_values = np.zeros(grid_shape, dtype=float)
        self._de_values = np.zeros(grid_shape, dtype=float)
        self._d2e_values = np.zeros(grid_shape, dtype=float)
        self._tmp_int_grid = np.zeros(grid_shape, dtype=float)
        self._tmp_int_grid2 = np.zeros(grid_shape, dtype=float)
        self._DG_values = np.zeros(grid_shape, dtype=float)

    def ener(self, rho, s, out=None):
        r"""Themodynamical energy as a function of rho and s, usign the perfect gaz hypothesis.

        .. math::
            E(\rho, s) = \rho^\gamma \text{exp}(s/\rho) \,.
        """
        gam = self._gamma
        if out is None:
            out = np.power(rho, gam) * np.exp(s / rho)
        else:
            out *= 0.0
            out += s
            out /= rho
            np.exp(out, out=out)
            np.power(rho, gam, out=self._tmp_int_grid)
            out *= self._tmp_int_grid
        return out

    def dener_drho(self, rho, s, out=None):
        r"""Derivative with respect to rho of the thermodynamical energy as a function of rho and s, usign the perfect gaz hypothesis.

        .. math::
            \frac{\partial E}{\partial \rho}(\rho, s) = (\gamma \rho^{\gamma-1} - s \rho^{\gamma-2})*\text{exp}(s/\rho) \,.
        """
        gam = self._gamma
        if out is None:
            out = (gam * np.power(rho, gam - 1) - s * np.power(rho, gam - 2)) * np.exp(s / rho)
        else:
            out *= 0.0
            out += s
            out /= rho
            np.exp(out, out=out)

            np.power(rho, gam - 1, out=self._tmp_int_grid)
            self._tmp_int_grid *= gam

            np.power(rho, gam - 2, out=self._tmp_int_grid2)
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
            out = np.power(rho, gam - 1) * np.exp(s / rho)
        else:
            out *= 0.0
            out += s
            out /= rho
            np.exp(out, out=out)
            np.power(rho, gam - 1, out=self._tmp_int_grid)
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
                gam * (gam - 1) * np.power(rho, gam - 2)
                - s * 2 * (gam - 1) * np.power(rho, gam - 3)
                + s**2 * np.power(rho, gam - 4)
            ) * np.exp(s / rho)
        else:
            out *= 0.0
            out += s
            out /= rho
            np.exp(out, out=out)

            np.power(rho, gam - 2, out=self._tmp_int_grid)
            self._tmp_int_grid *= gam * (gam - 1)

            np.power(rho, gam - 3, out=self._tmp_int_grid2)
            self._tmp_int_grid2 *= s
            self._tmp_int_grid2 *= 2 * (gam - 1)
            self._tmp_int_grid -= self._tmp_int_grid2

            np.power(rho, gam - 4, out=self._tmp_int_grid2)
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
            out = np.power(rho, gam - 2) * np.exp(s / rho)
        else:
            out *= 0.0
            out += s
            out /= rho
            np.exp(out, out=out)
            np.power(rho, gam - 2, out=self._tmp_int_grid)
            out *= self._tmp_int_grid
        return out

    def eta(self, delta_x, out=None):
        r"""Switch function :math:`\eta(\delta) = 1- \text{exp}((-\delta/10^{-5})^2)`."""
        if out is None:
            out = 1.0 - np.exp(-((delta_x / 1e-5) ** 2))
        else:
            out *= 0.0
            out += delta_x
            out /= 1e-5
            out **= 2
            out *= -1
            np.exp(out, out=out)
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

        integration_grid = [grid_1d.flatten() for grid_1d in derham.quad_grid_pts["0"]]

        self.integration_grid_spans, self.integration_grid_bn, self.integration_grid_bd = derham.prepare_eval_tp_fixed(
            integration_grid,
        )

        grid_shape = tuple([len(loc_grid) for loc_grid in integration_grid])
        self._f_values = np.zeros(grid_shape, dtype=float)

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
            verbose=False,
        )

        if hasattr(self, "_inv") and self._pc is not None:
            self._pc.update_mass_operator(self._massop)

    def _create_inv(
        self, type="pcg", pc_type="MassMatrixDiagonalPreconditioner", tol=1e-16, maxiter=500, verbose=False
    ):
        """Inverse the  weighted mass matrix"""
        if pc_type is None:
            self._pc = None
        else:
            pc_class = getattr(
                preconditioner,
                pc_type,
            )
            self._pc = pc_class(self.massop)

        self._inv = inverse(
            self.massop,
            type,
            pc=self._pc,
            tol=tol,
            maxiter=maxiter,
            verbose=verbose,
            recycle=True,
        )


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
    """

    def __init__(self, derham, domain, mass_ops):
        integration_grid = [grid_1d.flatten() for grid_1d in derham.quad_grid_pts["0"]]

        self.integration_grid_spans, self.integration_grid_bn, self.integration_grid_bd = derham.prepare_eval_tp_fixed(
            integration_grid,
        )

        # tmps
        grid_shape = tuple([len(loc_grid) for loc_grid in integration_grid])

        self.uf = derham.create_spline_function("uf", "H1vec")
        self.uf1 = derham.create_spline_function("uf1", "H1vec")

        self._uf_values = [np.zeros(grid_shape, dtype=float) for i in range(3)]
        self._uf1_values = [np.zeros(grid_shape, dtype=float) for i in range(3)]
        self._Guf_values = [np.zeros(grid_shape, dtype=float) for i in range(3)]
        self._tmp_int_grid = np.zeros(grid_shape, dtype=float)

        metric = domain.metric(
            *integration_grid,
        ) * domain.jacobian_det(*integration_grid)
        self._proj_u2_metric_term = deepcopy(metric)

        metric = domain.metric(*integration_grid)
        self._mass_u_metric_term = deepcopy(metric)

        self._M_un = mass_ops.create_weighted_mass("H1vec", "L2")
        self._M_un1 = mass_ops.create_weighted_mass("L2", "H1vec")

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
            verbose=False,
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
            verbose=False,
        )
