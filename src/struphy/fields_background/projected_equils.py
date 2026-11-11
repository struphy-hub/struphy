import logging

from feectools.linalg.basic import Vector
from feectools.linalg.block import BlockVector
from feectools.linalg.stencil import StencilVector

from struphy.feec.projectors import CommutingProjector
from struphy.feec.psydac_derham import Derham
from struphy.fields_background.base import (
    FluidEquilibrium,
    FluidEquilibriumWithB,
    MHDequilibrium,
)
from struphy.polar.basic import PolarVector

logger = logging.getLogger("struphy")


class ProjectorNoBC:
    """Wraps a projector such that essential (Dirichlet) boundary conditions are NOT applied."""

    def __init__(self, projector):
        self._projector = projector

    def __call__(self, fun, *args, **kwargs):
        if isinstance(self._projector, CommutingProjector):
            kwargs.setdefault("apply_bc", False)
        return self._projector(fun, *args, **kwargs)


class TensorProductCoefficients:
    """The tensor-product coefficients of the projections of a :class:`ProjectedFluidEquilibrium`:
    ``projected_equil.tp.<name>`` is ``projected_equil.to_tp(projected_equil.<name>)``, e.g. ``projected_equil.tp.b2``
    (a BlockVector also with polar splines, for kernels and other tensor-product code)."""

    def __init__(self, projected_equil: "ProjectedFluidEquilibrium"):
        self._projected_equil = projected_equil

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        value = getattr(self._projected_equil, name)
        if not isinstance(value, Vector):
            raise AttributeError(f"{name!r} is not a projection of the equilibrium (no FE coefficients).")
        return self._projected_equil.to_tp(value)

    def __dir__(self):
        return [n for n in dir(type(self._projected_equil)) if not n.startswith("_") and n not in ("tp", "to_tp")]


class ProjectedFluidEquilibrium:
    r"""Commuting projections of :class:`~struphy.fields_background.base.FluidEquilibrium` into Derham spaces.

    The properties return the FE coefficients in the space of the Derham (``derham.Vh_pol``-like spaces): PolarVectors
    with polar splines, Stencil-/BlockVectors otherwise. These are the vectors for the FEEC operators (mass matrices,
    derivatives, inner products, ``SplineFunction.vector``, sums with FEEC variables). The tensor-product coefficients
    (the polar extraction transposed, :math:`\mathbb E^\top`, applied; the same vectors without polar splines), e.g. for
    kernels, are ``self.tp.<name>`` (see :meth:`to_tp`). The projections are computed on each access."""

    def __init__(self, equil: FluidEquilibrium, derham: Derham):
        self._equil = equil
        self._derham = derham

        logger.debug(f"Projecting equilibrium '{equil.__class__.__name__}' into Derham spaces ...")
        logger.debug(f"{self.derham = }")

        # commuting projectors
        self._P0 = ProjectorNoBC(derham.P0)
        self._P1 = ProjectorNoBC(derham.P1)
        self._P2 = ProjectorNoBC(derham.P2)
        self._P3 = ProjectorNoBC(derham.P3)
        self._Pv = ProjectorNoBC(derham.Pv)

        # access to the tensor-product coefficients: self.tp.<name> (transposed extraction operators built on demand)
        self._tp = TensorProductCoefficients(self)
        self._ET = {}

        logger.debug("... Done.")

    @property
    def equil(self):
        return self._equil

    @property
    def tp(self) -> TensorProductCoefficients:
        """The tensor-product coefficients of the projections, e.g. ``self.tp.b2`` (see :meth:`to_tp`)."""
        return self._tp

    def to_tp(self, coeffs: Vector) -> StencilVector | BlockVector:
        r"""The tensor-product coefficients of FE coefficients in a space of the Derham.

        Parameters
        ----------
        coeffs : Vector
            FE coefficients in a space of the Derham (a PolarVector with polar splines).

        Returns
        -------
        StencilVector | BlockVector
            :math:`\mathbb E^\top` ``coeffs`` (new vector, ghost regions updated) for a PolarVector; other vectors are
            returned as they are.
        """
        if isinstance(coeffs, PolarVector):
            space_id = coeffs.space.space_id
            if space_id not in self._ET:
                self._ET[space_id] = self.derham.extraction_ops[space_id].transpose()
            coeffs = self._ET[space_id].dot(coeffs)
            coeffs.update_ghost_regions()
        return coeffs

    @property
    def derham(self):
        return self._derham

    # ---------#
    # 0-forms #
    # ---------#
    @property
    def p0(self):
        tmp = self._P0(self.equil.p0)
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def q0(self):
        tmp = self._P0(self.equil.q0)
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def n0(self):
        tmp = self._P0(self.equil.n0)
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def t0(self):
        tmp = self._P0(self.equil.t0)
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def vth0(self):
        tmp = self._P0(self.equil.vth0)
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def s0_monoatomic(self):
        tmp = self._P0(self.equil.s0_monoatomic)
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def s0_diatomic(self):
        tmp = self._P0(self.equil.s0_diatomic)
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    # ---------#
    # 3-forms #
    # ---------#
    @property
    def absB3(self):
        tmp = self._P3(self.equil.absB3)
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def p3(self):
        tmp = self._P3(self.equil.p3)
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def q3(self):
        tmp = self._P3(self.equil.q3)
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def n3(self):
        tmp = self._P3(self.equil.n3)
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def t3(self):
        tmp = self._P3(self.equil.t3)
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def vth3(self):
        tmp = self._P3(self.equil.vth3)
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def s3_monoatomic(self):
        tmp = self._P3(self.equil.s3_monoatomic)
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def s3_diatomic(self):
        tmp = self._P3(self.equil.s3_diatomic)
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    # ---------#
    # 1-forms #
    # ---------#
    @property
    def u1(self):
        tmp = self._P1([self.equil.u1_1, self.equil.u1_2, self.equil.u1_3])
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    # ---------#
    # 2-forms #
    # ---------#
    @property
    def u2(self):
        tmp = self._P2([self.equil.u2_1, self.equil.u2_2, self.equil.u2_3])
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    # -----------------------#
    # vector fields (H^1)^3 #
    # -----------------------#
    @property
    def uv(self):
        tmp = self._Pv([self.equil.uv_1, self.equil.uv_2, self.equil.uv_3])
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs


class ProjectedFluidEquilibriumWithB(ProjectedFluidEquilibrium):
    """Commuting projections of
    :class:`~struphy.fields_background.base.FluidEquilibriumWithB` into Derham spaces.
    Return coefficients."""

    def __init__(self, equil: FluidEquilibriumWithB, derham: Derham):
        super().__init__(equil, derham)

    # ---------#
    # 0-forms #
    # ---------#
    @property
    def absB0(self):
        tmp = self._P0(self.equil.absB0)
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def u_para0(self):
        tmp = self._P0(self.equil.u_para0)
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    # ---------#
    # 3-forms #
    # ---------#
    @property
    def absB3(self):
        tmp = self._P3(self.equil.absB3)
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def u_para3(self):
        tmp = self._P3(self.equil.u_para3)
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    # ---------#
    # 1-forms #
    # ---------#
    @property
    def b1(self):
        tmp = self._P1([self.equil.b1_1, self.equil.b1_2, self.equil.b1_3])
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def unit_b1(self):
        tmp = self._P1([self.equil.unit_b1_1, self.equil.unit_b1_2, self.equil.unit_b1_3])
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def gradB1(self):
        tmp = self._P1([self.equil.gradB1_1, self.equil.gradB1_2, self.equil.gradB1_3])
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def a1(self):
        tmp = self._P1([self.equil.a1_1, self.equil.a1_2, self.equil.a1_3])
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    # ---------#
    # 2-forms #
    # ---------#
    @property
    def b2(self):
        tmp = self._P2([self.equil.b2_1, self.equil.b2_2, self.equil.b2_3])
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def unit_b2(self):
        tmp = self._P2([self.equil.unit_b2_1, self.equil.unit_b2_2, self.equil.unit_b2_3])
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def gradB2(self):
        tmp = self._P2([self.equil.gradB2_1, self.equil.gradB2_2, self.equil.gradB2_3])
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def a2(self):
        tmp = self._P2([self.equil.a2_1, self.equil.a2_2, self.equil.a2_3])
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    # -----------------------#
    # vector fields (H^1)^3 #
    # -----------------------#
    @property
    def bv(self):
        tmp = self._Pv([self.equil.bv_1, self.equil.bv_2, self.equil.bv_3])
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def unit_bv(self):
        tmp = self._Pv([self.equil.unit_bv_1, self.equil.unit_bv_2, self.equil.unit_bv_3])
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def gradBv(self):
        tmp = self._Pv([self.equil.gradBv_1, self.equil.gradBv_2, self.equil.gradBv_3])
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def av(self):
        tmp = self._Pv([self.equil.av_1, self.equil.av_2, self.equil.av_3])
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs


class ProjectedMHDequilibrium(ProjectedFluidEquilibriumWithB):
    """Commuting projections of
    :class:`~struphy.fields_background.base.MHDequilibrium` into Derham spaces.
    Return coefficients."""

    def __init__(self, equil: MHDequilibrium, derham: Derham):
        super().__init__(equil, derham)

    # ---------#
    # 0-forms #
    # ---------#
    @property
    def curl_unit_b_dot_b0(self):
        tmp = self._P0(self.equil.curl_unit_b_dot_b0)
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    # ---------#
    # 3-forms #
    # ---------#

    # ---------#
    # 1-forms #
    # ---------#
    @property
    def j1(self):
        tmp = self._P1([self.equil.j1_1, self.equil.j1_2, self.equil.j1_3])
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def curl_unit_b1(self):
        tmp = self._P1([self.equil.curl_unit_b1_1, self.equil.curl_unit_b1_2, self.equil.curl_unit_b1_3])
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    # ---------#
    # 2-forms #
    # ---------#
    @property
    def j2(self):
        tmp = self._P2([self.equil.j2_1, self.equil.j2_2, self.equil.j2_3])
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def curl_unit_b2(self):
        tmp = self._P2([self.equil.curl_unit_b2_1, self.equil.curl_unit_b2_2, self.equil.curl_unit_b2_3])
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    # -----------------------#
    # vector fields (H^1)^3 #
    # -----------------------#
    @property
    def jv(self):
        tmp = self._Pv([self.equil.jv_1, self.equil.jv_2, self.equil.jv_3])
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def curl_unit_bv(self):
        tmp = self._Pv([self.equil.curl_unit_bv_1, self.equil.curl_unit_bv_2, self.equil.curl_unit_bv_3])
        coeffs = tmp
        coeffs.update_ghost_regions()
        return coeffs
