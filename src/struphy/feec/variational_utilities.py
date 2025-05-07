import numpy as np
from psydac.linalg.basic import IdentityOperator, Vector

from struphy.feec.linear_operators import LinOpWithTransp
from struphy.feec.basis_projection_ops import BasisProjectionOperator, BasisProjectionOperatorLocal, CoordinateProjector


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

    def __init__(self, derham, u):

        Xh = derham.Vh_fem["v"]
        V1h = derham.Vh_fem["1"]
        self._domain = derham.Vh_pol["v"]
        self._codomain = derham.Vh_pol["v"]
        self._dtype = Xh.coeff_space.dtype
        self._u = u

        # tmp for evaluating u
        self.vf = derham.create_field("uf", "H1vec")
        self.gv1f = derham.create_field("gu1f", "Hcurl")  # grad(u[0])
        self.gv2f = derham.create_field("gu2f", "Hcurl")  # grad(u[1])
        self.gv3f = derham.create_field("gu3f", "Hcurl")  # grad(u[2])

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
    """
    Operator u -> div(Pi(rho u)) from H1vec to L2.
    """
    def __init__(self, derham, transposed = False):
        # Get the projector and the spaces
        self._derham = derham
        self._transposed = transposed
        if self._transposed:
            self._codomain = self._derham.Vh_pol["v"]
            self._domain = self._derham.Vh_pol["3"]
        else:
            self._domain = self._derham.Vh_pol["v"]
            self._codomain = self._derham.Vh_pol["3"]
        P2 = self._derham.P["2"]
        Xh = self._derham.Vh_fem["v"]
        self._dtype = Xh.coeff_space.dtype
        self.field = self._derham.create_field("rhof", "L2")

        # Initialize the BasisProjectionOperator
        if self._derham._with_local_projectors:
            self.Proj = BasisProjectionOperatorLocal(
                P2,
                Xh,
                [
                    [None, None, None],
                    [None, None, None],
                    [None, None, None],
                ],
                transposed=transposed,
                V_extraction_op=self._derham.extraction_ops["v"],
                V_boundary_op=self._derham.boundary_ops["v"],
                P_boundary_op=IdentityOperator(self._derham.Vh_pol["2"]),
            )

        else:
            self.Proj = BasisProjectionOperator(
                P2,
                Xh,
                [
                    [None, None, None],
                    [None, None, None],
                    [None, None, None],
                ],
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
        return L2_transport_operator(self._derham, not self._transposed)

    def dot(self, v, out=None):
        out = self._op.dot(v, out=out)
        return out
    
    def update_coeffs(self, coeff):
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

        self.Proj.update_weights(
            [
                [f0_values, None, None],
                [None, f1_values, None],
                [None, None, f2_values],
            ]
        )