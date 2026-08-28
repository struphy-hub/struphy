import logging
from copy import deepcopy
from typing import Callable

import cunumpy as xp
from cunumpy import PyccelKernel
from feectools.api.settings import PSYDAC_BACKEND_GPYCCEL
from feectools.linalg.basic import ComposedLinearOperator, IdentityOperator, Vector
from feectools.linalg.block import BlockLinearOperator, BlockVector
from feectools.linalg.solvers import inverse
from feectools.linalg.stencil import StencilMatrix
from line_profiler import profile
from scope_profiler import ProfileManager

from struphy.feec import mass_kernels, preconditioner
from struphy.feec.basis_projection_ops import (
    BasisProjectionOperator,
    BasisProjectionOperatorLocal,
    CoordinateProjector,
)
from struphy.feec.linear_operators import LinOpWithTransp
from struphy.feec.mass import WeightedMassOperators
from struphy.feec.psydac_derham import Derham
from struphy.geometry.base import Domain

logger = logging.getLogger("struphy")


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

        with ProfileManager.profile_region("momentum bracket: gradients"):
            grad_1_v = self.gp1.dot(v, out=self.gp1v)
            grad_2_v = self.gp2.dot(v, out=self.gp2v)
            grad_3_v = self.gp3.dot(v, out=self.gp3v)

        # To avoid tmp we need to update the fields we created.
        self.gv1f.vector = grad_1_v
        self.gv2f.vector = grad_2_v
        self.gv3f.vector = grad_3_v

        with ProfileManager.profile_region("momentum bracket: spline evaluation"):
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

        with ProfileManager.profile_region("momentum bracket: projector weights"):
            self.PiuT.update_weights([[vf_values[0], vf_values[1], vf_values[2]]])
            self.PigvT_1.update_weights([[gvf1_values[0], gvf1_values[1], gvf1_values[2]]])
            self.PigvT_2.update_weights([[gvf2_values[0], gvf2_values[1], gvf2_values[2]]])
            self.PigvT_3.update_weights([[gvf3_values[0], gvf3_values[1], gvf3_values[2]]])

        with ProfileManager.profile_region("momentum bracket: operator application"):
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
        if weights is None:
            weights = [[None] * 3] * 3
        self._weights = weights
        if self._transposed:
            self._codomain = self._derham.Vvpol
            self._domain = self._derham.V3pol
        else:
            self._domain = self._derham.Vvpol
            self._codomain = self._derham.V3pol
        P2 = self._derham.P2
        Xh = self._derham.Vvfem
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
                P_boundary_op=IdentityOperator(self._derham.V2pol),
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
                P_boundary_op=IdentityOperator(self._derham.V2pol),
            )

        # divergence
        self.div = self._derham.div_bcfree

        # Initialize the transport operator and transposed
        if self._transposed:
            self._op = self.Proj @ self.div.T
        else:
            self._op = self.div @ self.Proj
        self._dot_tmp = self._op.tmp_vectors[0]

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
        direction = "transpose" if self._transposed else "forward"
        if self._transposed:
            with ProfileManager.profile_region(
                f"L2 transport {direction}: divergence"
            ):
                self.div.T.dot(v, out=self._dot_tmp)
            with ProfileManager.profile_region(
                f"L2 transport {direction}: projection"
            ):
                out = self.Proj.dot(self._dot_tmp, out=out)
        else:
            with ProfileManager.profile_region(
                f"L2 transport {direction}: projection"
            ):
                self.Proj.dot(v, out=self._dot_tmp)
            with ProfileManager.profile_region(
                f"L2 transport {direction}: divergence"
            ):
                out = self.div.dot(self._dot_tmp, out=out)
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
        if weights is None:
            weights = [[None] * 3] * 3
        self._weights = weights
        if self._transposed:
            self._codomain = self._derham.Vvpol
            self._domain = self._derham.V2pol
        else:
            self._domain = self._derham.Vvpol
            self._codomain = self._derham.V2pol
        P1 = self._derham.P1
        Xh = self._derham.Vvfem
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
                P_boundary_op=IdentityOperator(self._derham.V1pol),
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
            self._codomain = self._derham.Vvpol
            self._domain = self._derham.V3pol
        else:
            self._domain = self._derham.Vvpol
            self._codomain = self._derham.V3pol
        P2 = self._derham.P2
        P3 = self._derham.P3
        Xh = self._derham.Vvfem
        V3h = self._derham.V3fem
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

        self.div = div @ Uv

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
        pc_type="MassMatrixDiagonalPreconditioner",
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


class H1vecKineticMetric(LinOpWithTransp):
    def __init__(
        self,
        mass_rho,
        divdiv_rho,
        *,
        alpha: float = 1.0,
    ):
        assert alpha >= 0.0
        assert mass_rho.domain == divdiv_rho.domain
        assert mass_rho.codomain == divdiv_rho.codomain

        self._mass_rho = mass_rho
        self._divdiv_rho = divdiv_rho
        self._alpha = alpha

        self._domain = mass_rho.domain
        self._codomain = mass_rho.codomain
        self._dtype = mass_rho.dtype

        self._tmp_div = self.codomain.zeros()
        self._tmp_apply = self.codomain.zeros()

        first_matrix_block = self.divdiv_operator.matrix[0, 0]
        self._use_fused_device_metric = xp.is_gpu(first_matrix_block._data)
        if self._use_fused_device_metric:
            # Both terms use the same H1vec tensor space, extraction and
            # boundary maps. Combine their assembled stencil coefficients so
            # a device application traverses the input only once. Div-div has
            # the full 3x3 block pattern whereas mass may be block diagonal,
            # so use div-div as the structural template.
            self._fused_matrix = self.divdiv_operator.matrix.copy()
            divdiv_chain = self.divdiv_operator.M0
            if isinstance(divdiv_chain, ComposedLinearOperator):
                fused_chain = tuple(
                    self._fused_matrix if op is self.divdiv_operator.matrix else op
                    for op in divdiv_chain.multiplicants
                )
                self._fused_operator = ComposedLinearOperator(
                    self.domain,
                    self.codomain,
                    *fused_chain,
                )
            else:
                self._fused_operator = self._fused_matrix
            self._refresh_fused_matrix()

    def _refresh_fused_matrix(self):
        """Refresh the assembled matrix ``M_rho + alpha K_rho`` in place."""
        if not self._use_fused_device_metric:
            return

        mass_matrix = self.mass_operator.matrix
        divdiv_matrix = self.divdiv_operator.matrix
        for row in range(3):
            for col in range(3):
                fused = self._fused_matrix[row, col]
                mass = mass_matrix[row, col]
                divdiv = divdiv_matrix[row, col]
                fused._data[:] = 0.0
                if mass is not None:
                    fused._data += mass._data
                fused._data += self.alpha * divdiv._data
                fused.ghost_regions_in_sync = (
                    (mass is None or mass.ghost_regions_in_sync)
                    and divdiv.ghost_regions_in_sync
                )

    @staticmethod
    def _accumulate_scaled_device(out, value, alpha):
        """Apply ``out += alpha * value`` with one kernel per leaf vector."""
        if hasattr(out, "blocks"):
            for out_block, value_block in zip(out.blocks, value.blocks):
                H1vecKineticMetric._accumulate_scaled_device(
                    out_block,
                    value_block,
                    alpha,
                )
            out.ghost_regions_in_sync = False
            return

        import cupy as cp

        kernel = getattr(H1vecKineticMetric, "_scaled_add_kernel", None)
        if kernel is None:
            kernel = cp.ElementwiseKernel(
                "T x, T alpha",
                "T y",
                "y += alpha * x",
                "struphy_h1_metric_scaled_add",
            )
            H1vecKineticMetric._scaled_add_kernel = kernel
        kernel(value._data, alpha, out._data)
        out.ghost_regions_in_sync = False

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
    def mass_operator(self):
        return self._mass_rho

    @property
    def divdiv_operator(self):
        return self._divdiv_rho

    @property
    def alpha(self):
        return self._alpha

    def update_mass_weight(self, rho):
        """Reassemble only the density-weighted H1vec mass operator."""
        self.mass_operator.spline_functions["l2_field"].vector = rho
        self.mass_operator.assemble()

    def update_divdiv_weight(self, rho):
        """Reassemble only the density-weighted div-div operator."""
        self.divdiv_operator.update_weight(rho)

    def update_weight(
        self,
        rho,
        *,
        update_mass=True,
        update_divdiv=True,
    ):
        """Update selected density-dependent parts of the metric."""

        if update_mass:
            self.update_mass_weight(rho)

        if update_divdiv:
            self.update_divdiv_weight(rho)

        self._refresh_fused_matrix()

    def update_weight_if_needed(self, rho, generation):
        if getattr(self, "_rho_generation", None) == generation:
            return False

        self.update_weight(rho)
        self._rho_generation = generation
        return True

    def mark_weight_generation(self, generation):
        """Record the generation represented by the currently assembled weight."""
        self._rho_generation = generation

    def dot(self, v, out=None):
        assert isinstance(v, Vector)
        assert v.space == self.domain

        if out is None:
            out = self.codomain.zeros()
        else:
            assert isinstance(out, Vector)
            assert out.space == self.codomain

        first_data = v.blocks[0]._data if hasattr(v, "blocks") else v._data
        if self._use_fused_device_metric and xp.is_gpu(first_data):
            self._fused_operator.dot(v, out=out)
            return out

        self.mass_operator.dot(v, out=out)

        if self.alpha != 0.0:
            self.divdiv_operator.dot(
                v,
                out=self._tmp_div,
            )
            first_data = out.blocks[0]._data if hasattr(out, "blocks") else out._data
            if xp.is_gpu(first_data):
                self._accumulate_scaled_device(out, self._tmp_div, self.alpha)
            else:
                self._tmp_div *= self.alpha
                out += self._tmp_div

        return out

    def dot_inner(self, v, w):
        self.dot(v, out=self._tmp_apply)
        return w.inner(self._tmp_apply)

    def transpose(self, conjugate=False):
        return self

    def tosparse(self):
        r"""
        Return the sparse matrix representation of

            M_rho + alpha * K_rho.

        This requires the constituent operators to support sparse
        conversion for the active extraction and boundary operators.
        """
        mass_sparse = self.mass_operator.tosparse()

        if self.alpha == 0.0:
            return mass_sparse

        divdiv_sparse = self.divdiv_operator.tosparse()

        if mass_sparse.shape != divdiv_sparse.shape:
            raise ValueError(
                "The mass and div-div sparse matrices have "
                "incompatible shapes: "
                f"{mass_sparse.shape} and {divdiv_sparse.shape}."
            )

        return mass_sparse + self.alpha * divdiv_sparse

    def toarray(self):
        r"""
        Return the dense matrix representation of

            M_rho + alpha * K_rho.
        """
        return self.tosparse().toarray()


class H1vecDivergenceEvaluator:
    r"""
    Evaluate the physical divergence of an H1vec field on a fixed
    tensor-product grid.

    The H1vec proxy is pushed forward according to

    .. math::

        \mathbf u = DF\,\widehat{\mathbf u},

    and therefore

    .. math::

        \nabla_{\mathbf x}\cdot\mathbf u
        =
        \sum_{\mu=1}^3
        \left(
            \partial_{\eta_\mu}\widehat u_\mu
            +
            \widehat u_\mu
            \partial_{\eta_\mu}\log|\det DF|
        \right).

    Parameters
    ----------
    derham : Derham
        Discrete de Rham sequence.

    domain : Domain
        Mapping from logical to physical coordinates.

    evaluation_grid : tuple | list
        Three one-dimensional logical evaluation grids.

    dlog_jacobian_det : callable, optional
        Callable evaluating

        ``grad_eta(log(abs(det(DF))))``.

        If omitted, ``domain.log_jacobian_det_gradient`` is used.
    """

    def __init__(
        self,
        derham,
        domain,
        evaluation_grid,
    ):
        self._derham = derham
        self._domain = domain
        self._evaluation_grid = tuple(grid.flatten() for grid in evaluation_grid)

        if len(self._evaluation_grid) != 3:
            raise ValueError("evaluation_grid must contain three one-dimensional grids.")

        (
            self._spans,
            self._bn,
            self._bd,
        ) = derham.prepare_eval_tp_fixed(
            self._evaluation_grid,
        )

        self._grid_shape = tuple(grid.size for grid in self._evaluation_grid)

        # Basis arrays for evaluating an Hcurl field. These correspond to
        # the three components of grad(V0):
        #
        #   (DNN, NDN, NND).
        self._hcurl_bases = (
            (self._bd[0], self._bn[1], self._bn[2]),
            (self._bn[0], self._bd[1], self._bn[2]),
            (self._bn[0], self._bn[1], self._bd[2]),
        )

        # H1vec field used to evaluate the logical vector proxy.
        self._velocity_field = derham.create_spline_function(
            "h1vec_divergence_velocity",
            "H1vec",
        )

        # Extract each H1vec component as a scalar H1 field, then apply
        # the exact discrete gradient H1 -> Hcurl.
        self._coordinate_projectors = tuple(
            CoordinateProjector(
                component,
                derham.Vvpol,
                derham.V0pol,
            )
            @ derham.boundary_ops["v"]
            for component in range(3)
        )

        self._gradient_operators = tuple(derham.grad_bcfree @ projector for projector in self._coordinate_projectors)
        self._gradient_vectors = tuple(derham.V1pol.zeros() for _ in range(3))
        self._gradient_fields = tuple(
            derham.create_spline_function(
                f"h1vec_divergence_gradient_{component}",
                "Hcurl",
            )
            for component in range(3)
        )

        # Logical velocity values at the evaluation points.
        self._velocity_values = [xp.zeros(self._grid_shape, dtype=float) for _ in range(3)]

        # gradient_values[a][b] contains partial_{eta_b} uhat_a.
        self._gradient_values = [[xp.zeros(self._grid_shape, dtype=float) for _ in range(3)] for _ in range(3)]

        # Evaluate grad_eta(log(abs(det(DF)))) once, since the geometry
        # does not change during the simulation.
        dlogj = domain.log_jacobian_det_gradient(
            *self._evaluation_grid,
            squeeze_out=False,
            remove_outside=False,
        )
        expected_shape = (3, *self._grid_shape)

        if isinstance(dlogj, (list, tuple)):
            if len(dlogj) != 3:
                raise ValueError("dlog_jacobian_det must return three components.")

            self._dlogj = tuple(xp.ascontiguousarray(component).copy() for component in dlogj)

            for component in self._dlogj:
                if component.shape != self._grid_shape:
                    raise ValueError(
                        "Invalid dlog_jacobian_det component shape: "
                        f"expected {self._grid_shape}, "
                        f"got {component.shape}."
                    )

        elif isinstance(dlogj, xp.ndarray):
            if dlogj.shape != expected_shape:
                raise ValueError(f"Invalid dlog_jacobian_det shape: expected {expected_shape}, got {dlogj.shape}.")

            self._dlogj = tuple(xp.ascontiguousarray(dlogj[component]).copy() for component in range(3))

        else:
            raise TypeError("dlog_jacobian_det must return a list, tuple, or xp.ndarray.")

        self._output = xp.zeros(
            self._grid_shape,
            dtype=float,
        )

    @property
    def derham(self):
        """Discrete de Rham sequence."""
        return self._derham

    @property
    def domain(self):
        """Mapping from logical to physical coordinates."""
        return self._domain

    @property
    def evaluation_grid(self):
        """Logical tensor-product evaluation grid."""
        return self._evaluation_grid

    @property
    def grid_shape(self):
        """Shape of scalar fields evaluated on the grid."""
        return self._grid_shape

    @property
    def dlog_jacobian_det(self):
        """Logical gradient of log(abs(det(DF))) on the grid."""
        return self._dlogj

    @property
    def velocity_values(self):
        """Logical H1vec proxy values on the grid."""
        return self._velocity_values

    @property
    def gradient_values(self):
        """Logical component gradients on the grid."""
        return self._gradient_values

    def evaluate(self, coeffs, out=None):
        r"""
        Evaluate the physical divergence of an H1vec field.

        Parameters
        ----------
        coeffs : BlockVector | PolarVector
            H1vec coefficients.

        out : xp.ndarray, optional
            Output array of shape :attr:`grid_shape`.

        Returns
        -------
        out : xp.ndarray
            Physical divergence evaluated on the configured grid.
        """
        if coeffs.space != self.derham.Vvpol:
            raise ValueError("coeffs must belong to the H1vec polar coefficient space.")

        if out is None:
            out = self._output
        else:
            if not isinstance(out, xp.ndarray):
                raise TypeError("out must be an xp.ndarray.")

            if out.shape != self.grid_shape:
                raise ValueError(f"Expected out shape {self.grid_shape}, got {out.shape}.")

        # Evaluate the logical H1vec proxy.
        self._velocity_field.vector = coeffs

        velocity_values = self._velocity_field.eval_tp_fixed_loc(
            self._spans,
            [self._bn] * 3,
            out=self._velocity_values,
        )

        # Evaluate grad(uhat_a) for each component a.
        for component in range(3):
            self._gradient_operators[component].dot(
                coeffs,
                out=self._gradient_vectors[component],
            )
            self._gradient_fields[component].vector = self._gradient_vectors[component]
            self._gradient_fields[component].eval_tp_fixed_loc(
                self._spans,
                self._hcurl_bases,
                out=self._gradient_values[component],
            )

        out[:] = 0.0

        # div_x(u) =
        #
        #   sum_a [
        #       partial_{eta_a}(uhat_a)
        #       + uhat_a * partial_{eta_a}(log(abs(J)))
        #   ].
        for component in range(3):
            out += self._gradient_values[component][component]
            out += velocity_values[component] * self._dlogj[component]

        return out


class H1vecWeakDivergenceMultiplicationOperator(LinOpWithTransp):
    r"""
    Weak H1vec divergence multiplication operator

    .. math::

        B_w : V_h^v \longrightarrow (V_h^3)^*,

    defined by

    .. math::

        \mathbf q^\top B_w\mathbf v
        =
        \int_{\widehat\Omega}
        q_h\,w\,
        \nabla_{\mathbf x}\cdot\mathbf v_h
        \,\mathrm d\boldsymbol\eta.

    Here :math:`w` is a scalar function evaluated at quadrature points.
    The H1vec push-forward is

    .. math::

        \mathbf v = DF\,\widehat{\mathbf v},

    and therefore

    .. math::

        \nabla_{\mathbf x}\cdot\mathbf v
        =
        \sum_{a=1}^3
        \left(
            \partial_{\eta_a}\widehat v_a
            +
            \widehat v_a
            \partial_{\eta_a}\log|\det DF|
        \right).

    Parameters
    ----------
    derham : Derham
        Discrete de Rham sequence.

    domain : Domain
        Mapping from logical to physical coordinates.

    name : str
        Name of the operator.

    dlog_jacobian_det : Callable, optional
        Callable returning
        ``grad_eta(log(abs(det(DF))))``. If omitted,
        :meth:`Domain.log_jacobian_det_gradient` is used.
    """

    def __init__(
        self,
        derham: Derham,
        domain: Domain,
        *,
        name: str = "H1vecWeakDivergenceMultiplication",
    ):
        self._derham = derham
        self._physical_domain = domain
        self._name = name

        V = derham.Vvfem
        W = derham.V3fem

        assert len(V.spaces) == 3

        reference_space = V.spaces[0]

        assert all(space.coeff_space == reference_space.coeff_space for space in V.spaces)
        assert all(space.degree == reference_space.degree for space in V.spaces)

        self._dtype = V.coeff_space.dtype

        # One L2 row and three H1vec columns.
        blocks = [
            [
                StencilMatrix(
                    trial_space.coeff_space,
                    W.coeff_space,
                    backend=PSYDAC_BACKEND_GPYCCEL,
                    precompiled=True,
                )
                for trial_space in V.spaces
            ],
        ]

        self._mat = BlockLinearOperator(
            V.coeff_space,
            W.coeff_space,
            blocks=blocks,
        )

        # Extraction and boundary operators.
        self._V_extraction_op = derham.extraction_ops["v"]
        self._W_extraction_op = derham.extraction_ops["3"]

        self._V_boundary_op = derham.boundary_ops["v"]
        self._W_boundary_op = derham.boundary_ops["3"]

        self._M = self._W_extraction_op @ self._mat @ self._V_extraction_op.T

        self._M0 = self._W_boundary_op @ self._M @ self._V_boundary_op.T

        self._domain = self._M0.domain
        self._codomain = self._M0.codomain

        # Use the L2 quadrature grid, since L2 is the test/codomain space.
        test_attr = derham.V3splines
        trial_attr = derham.V0splines

        self._quad_pts = tuple(points.flatten() for points in test_attr.quad_grid_pts[0])

        self._test_spans = test_attr.quad_grid_spans[0]
        self._test_weights = test_attr.quad_grid_wts[0]
        self._test_bases = test_attr.quad_grid_bases[0]

        self._trial_spans = trial_attr.quad_grid_spans[0]
        self._trial_bases = trial_attr.quad_grid_bases[0]

        # Both spaces must use the same local elements and quadrature grid.
        trial_quad_pts = tuple(points.flatten() for points in trial_attr.quad_grid_pts[0])

        for test_points, trial_points in zip(
            self._quad_pts,
            trial_quad_pts,
        ):
            if test_points.shape != trial_points.shape:
                raise ValueError(
                    "The H1 and L2 quadrature grids must have the same shape for weak-divergence assembly."
                )

            if not bool(
                xp.all(
                    xp.abs(test_points - trial_points) < 1e-14,
                ),
            ):
                raise ValueError("The H1 and L2 quadrature points must coincide for weak-divergence assembly.")

        for basis in self._trial_bases:
            if basis.shape[2] < 2:
                raise ValueError(
                    "H1vecWeakDivergenceMultiplicationOperator requires first derivatives of the H1 basis."
                )

        self._quad_shape = tuple(points.size for points in self._quad_pts)

        self._weight_values = xp.zeros(
            self._quad_shape,
            dtype=float,
        )

        # Geometry coefficient.
        self._dlogj = [xp.zeros(self._quad_shape, dtype=float) for _ in range(3)]

        values = domain.log_jacobian_det_gradient(
            *self._quad_pts,
            squeeze_out=False,
            remove_outside=False,
        )

        expected_shape = (3, *self._quad_shape)

        if not isinstance(values, xp.ndarray):
            raise TypeError("Domain.log_jacobian_det_gradient must return an xp.ndarray.")

        if values.shape != expected_shape:
            raise ValueError(f"Invalid log-Jacobian gradient shape: expected {expected_shape}, got {values.shape}.")

        for component in range(3):
            self._dlogj[component][:] = values[component]

        self._assembly_kernel = PyccelKernel(
            mass_kernels.kernel_3d_h1vec_weak_divergence,
        )

    @property
    def derham(self):
        return self._derham

    @property
    def physical_domain(self):
        return self._physical_domain

    @property
    def name(self):
        return self._name

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
    def matrix(self):
        """Tensor-product matrix before extraction and boundary operators."""
        return self._mat

    @property
    def M(self):
        """Operator including extraction but excluding boundary operators."""
        return self._M

    @property
    def M0(self):
        """Operator including extraction and boundary operators."""
        return self._M0

    @property
    def weight_values(self):
        """Scalar weight evaluated at quadrature points."""
        return self._weight_values

    def assemble(
        self,
        weight_values: xp.ndarray,
        clear: bool = True,
    ):
        r"""
        Assemble the operator with scalar quadrature weight ``w``.

        Parameters
        ----------
        weight_values : xp.ndarray
            Values of ``w`` at the L2 quadrature points.

        clear : bool
            Whether to clear existing matrix entries before assembly.
        """
        if not isinstance(weight_values, xp.ndarray):
            raise TypeError("weight_values must be an xp.ndarray.")

        if weight_values.shape != self._quad_shape:
            raise ValueError(f"Expected weight shape {self._quad_shape}, got {weight_values.shape}.")

        self._weight_values[:] = weight_values

        if clear:
            for block in self._mat.blocks[0]:
                block._data[:] = 0.0

        W = self.derham.V3fem
        V = self.derham.Vvfem

        starts = tuple(int(start) for start in W.coeff_space.starts)
        pads = W.coeff_space.pads

        for component_trial in range(3):
            trial_space = V.spaces[component_trial]
            mat = self._mat[0, component_trial]

            if xp.is_gpu(mat._data):
                from struphy.feec.mass_kernels_cuda import weak_divergence_assemble_gpu

                weak_divergence_assemble_gpu(
                    self._test_spans,
                    W.degree,
                    trial_space.degree,
                    starts,
                    pads,
                    self._test_weights,
                    self._test_bases,
                    self._trial_bases,
                    self._weight_values,
                    self._dlogj,
                    component_trial,
                    mat._data,
                )
            else:
                self._assembly_kernel(
                    *self._test_spans,
                    *W.degree,
                    *trial_space.degree,
                    *starts,
                    *pads,
                    *self._test_weights,
                    *self._test_bases,
                    *self._trial_bases,
                    self._weight_values,
                    self._dlogj[0],
                    self._dlogj[1],
                    self._dlogj[2],
                    component_trial,
                    mat._data,
                )

        self._mat.exchange_assembly_data()

    def dot(self, v, out=None):
        """Apply the weak-divergence multiplication operator."""
        assert isinstance(v, Vector)
        assert v.space == self.domain

        if out is not None:
            assert isinstance(out, Vector)
            assert out.space == self.codomain

        return self._M0.dot(v, out=out)

    def transpose(self, conjugate=False):
        """
        Return the transposed L2-to-H1vec operator.

        The returned composite operator references the same underlying
        stencil matrix, so later calls to :meth:`assemble` update both
        orientations.
        """
        return self._M0.T

    def tosparse(self):
        raise NotImplementedError()

    def toarray(self):
        raise NotImplementedError()


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

    def __init__(
        self,
        derham,
        domain,
        mass_ops,
        *,
        with_regularization: bool = False,
        divergence_operator=None,
    ):
        self._derham = derham
        self._domain = domain
        self._mass_ops = mass_ops
        self._with_regularization = with_regularization
        self._divergence_operator = divergence_operator

        integration_grid = [grid_1d.flatten() for grid_1d in derham.V0splines.quad_grid_pts[0]]

        grid_shape = tuple(len(loc_grid) for loc_grid in integration_grid)

        (
            self.integration_grid_spans,
            self.integration_grid_bn,
            self.integration_grid_bd,
        ) = derham.prepare_eval_tp_fixed(integration_grid)

        self.uf = derham.create_spline_function("uf", "H1vec")
        self.uf1 = derham.create_spline_function("uf1", "H1vec")

        self._uf_values = [xp.zeros(grid_shape, dtype=float) for _ in range(3)]
        self._uf1_values = [xp.zeros(grid_shape, dtype=float) for _ in range(3)]
        self._Guf_values = [xp.zeros(grid_shape, dtype=float) for _ in range(3)]
        self._tmp_int_grid = xp.zeros(grid_shape, dtype=float)

        if xp.cupy_backend:
            from struphy.feec.variational_kernels_cuda import prepare_kinetic_energy_kernel

            prepare_kinetic_energy_kernel()

        metric = domain.metric(*integration_grid) * domain.jacobian_det(*integration_grid)
        self._proj_u2_metric_term = deepcopy(metric)

        metric = domain.metric(*integration_grid)
        self._mass_u_metric_term = deepcopy(metric)

        self._M_un = mass_ops.create_weighted_mass(
            "H1vec",
            "L2",
        )
        self._M_un1 = mass_ops.create_weighted_mass(
            "L2",
            "H1vec",
        )
        self._jacobian_det = deepcopy(domain.jacobian_det(*integration_grid))
        if self._with_regularization:
            self._divergence_evaluator = H1vecDivergenceEvaluator(
                derham,
                domain,
                integration_grid,
            )

            self._div_u_values = xp.zeros(
                grid_shape,
                dtype=float,
            )
            self._div_u1_values = xp.zeros(
                grid_shape,
                dtype=float,
            )

            self._M_div_un = H1vecWeakDivergenceMultiplicationOperator(
                derham,
                domain,
                name="M_div_un",
            )

            # A separate mutable matrix is required because both operators
            # occur simultaneously in the Newton Jacobian with different
            # weights.
            self._M_div_un1_base = H1vecWeakDivergenceMultiplicationOperator(
                derham,
                domain,
                name="M_div_un1",
            )
        else:
            self._divergence_evaluator = None
            self._div_u_values = None
            self._div_u1_values = None
            self._M_div_un = None
            self._M_div_un1_base = None

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

    @property
    def M_div_un(self):
        """H1vec-to-L2 weak divergence operator weighted by div(un)."""
        return self._M_div_un

    @property
    def M_div_un1(self):
        """L2-to-H1vec transpose weighted by div(un1)."""
        if self._M_div_un1_base is None:
            return None
        return self._M_div_un1_base.T

    def cache_div_u_grid(self, un):
        """Evaluate and retain the beginning-of-step velocity divergence."""
        if not self._with_regularization:
            raise RuntimeError("The divergence evaluator was not allocated.")
        return self._evaluate_divergence(un, out=self._div_u_values)

    def _evaluate_divergence(self, coefficients, *, out):
        """Evaluate physical divergence, using the exact CUDA Q operator."""
        tensor_coefficients = coefficients.tp if hasattr(coefficients, "tp") else coefficients
        first_data = (
            tensor_coefficients.blocks[0]._data
            if hasattr(tensor_coefficients, "blocks")
            else tensor_coefficients._data
        )
        if self._divergence_operator is not None and xp.is_gpu(first_data):
            return self._divergence_operator.apply_Q(coefficients, out=out)
        return self._divergence_evaluator.evaluate(coefficients, out=out)

    def get_div_u_product_grid(self, un, un1, out, *, div_un_is_cached=False):
        r"""
        Evaluate

        .. math::

            |\det DF|\,
            \operatorname{div}(u^n)
            \operatorname{div}(u^{n+1})

        on the integration grid.

        The Jacobian factor compensates for the ``1/sqrt_g`` geometric
        weight applied by ``L2Projector.get_dofs``.
        """
        if not self._with_regularization:
            raise RuntimeError("The divergence evaluator was not allocated.")

        if div_un_is_cached:
            div_un = self._div_u_values
        else:
            div_un = self.cache_div_u_grid(un)
        div_un1 = self._evaluate_divergence(
            un1,
            out=self._div_u1_values,
        )

        out[:] = div_un
        out *= div_un1
        out *= self._jacobian_det

        return out

    def get_u2_grid(self, un, un1, out):
        r"""Values of :math:`u_n \cdot u_{n+1}` represented by the coefficient un and un1, on the integration grid"""
        self.uf.vector = un
        self.uf1.vector = un1

        tensor_u = self.uf.vector.tp if hasattr(self.uf.vector, "tp") else self.uf.vector
        tensor_u1 = self.uf1.vector.tp if hasattr(self.uf1.vector, "tp") else self.uf1.vector
        first_data = tensor_u.blocks[0]._data if hasattr(tensor_u, "blocks") else tensor_u._data
        if xp.is_gpu(first_data):
            from struphy.feec.variational_kernels_cuda import kinetic_energy_grid_gpu

            coefficients = tuple(block._data for block in tensor_u.blocks)
            coefficients1 = tuple(block._data for block in tensor_u1.blocks)
            return kinetic_energy_grid_gpu(
                self.integration_grid_spans,
                self.integration_grid_bn,
                self._derham.degree,
                self.uf.starts[0],
                coefficients,
                coefficients1,
                self._proj_u2_metric_term,
                out,
            )

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

    def assemble_M_div_un(self, un):
        if not self._with_regularization:
            return

        div_un = self._divergence_evaluator.evaluate(
            un,
            out=self._div_u_values,
        )
        self._M_div_un.assemble(div_un)

    def assemble_M_div_un_cached(self):
        """Assemble from the divergence cached by ``get_div_u_product_grid``."""
        if not self._with_regularization:
            return
        self._M_div_un.assemble(self._div_u_values)

    def assemble_M_div_un1(self, un1):
        if not self._with_regularization:
            return

        div_un1 = self._divergence_evaluator.evaluate(
            un1,
            out=self._div_u1_values,
        )
        self._M_div_un1_base.assemble(div_un1)

    def assemble_M_div_un1_cached(self):
        """Assemble from the divergence cached by ``get_div_u_product_grid``."""
        if not self._with_regularization:
            return
        self._M_div_un1_base.assemble(self._div_u1_values)
