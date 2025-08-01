from struphy.feec.psydac_derham import Derham
from struphy.fields_background.base import (
    FluidEquilibrium,
    FluidEquilibriumWithB,
    MHDequilibrium,
)
from psydac.linalg.stencil import StencilVector
from psydac.linalg.block import BlockVector


class ProjectedFluidEquilibrium:
    """Commuting projections of
    :class:`~struphy.fields_background.base.FluidEquilibrium` into Derham spaces.
    Return coefficients."""

    def __init__(self, equil: FluidEquilibrium, derham: Derham):
        self._equil = equil
        self._derham = derham

        # commuting projectors
        self._P0 = derham.P["0"]
        self._P1 = derham.P["1"]
        self._P2 = derham.P["2"]
        self._P3 = derham.P["3"]
        self._Pv = derham.P["v"]

        # transposed extraction operator PolarVector --> BlockVector (identity map in case of no polar splines)
        self._E0T = derham.extraction_ops["0"].transpose()
        self._E1T = derham.extraction_ops["1"].transpose()
        self._E2T = derham.extraction_ops["2"].transpose()
        self._E3T = derham.extraction_ops["3"].transpose()
        self._EvT = derham.extraction_ops["v"].transpose()

    @property
    def equil(self):
        return self._equil

    @property
    def derham(self):
        return self._derham

    # ---------#
    # 0-forms #
    # ---------#
    @property
    def p0(self):
        tmp = self._P0(self.equil.p0)
        coeffs = self._E0T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def q0(self):
        tmp = self._P0(self.equil.q0)
        coeffs = self._E0T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def n0(self):
        tmp = self._P0(self.equil.n0)
        coeffs = self._E0T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def t0(self):
        tmp = self._P0(self.equil.t0)
        coeffs = self._E0T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def vth0(self):
        tmp = self._P0(self.equil.vth0)
        coeffs = self._E0T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def s0_monoatomic(self):
        tmp = self._P0(self.equil.s0_monoatomic)
        coeffs = self._E0T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def s0_diatomic(self):
        tmp = self._P0(self.equil.s0_diatomic)
        coeffs = self._E0T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    # ---------#
    # 3-forms #
    # ---------#
    @property
    def absB3(self):
        tmp = self._P3(self.equil.absB3)
        coeffs = self._E3T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def p3(self) -> StencilVector:
        tmp = self._P3(self.equil.p3)
        coeffs = self._E3T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def q3(self):
        tmp = self._P3(self.equil.q3)
        coeffs = self._E3T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def n3(self):
        tmp = self._P3(self.equil.n3)
        coeffs = self._E3T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def t3(self):
        tmp = self._P3(self.equil.t3)
        coeffs = self._E3T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def vth3(self):
        tmp = self._P3(self.equil.vth3)
        coeffs = self._E3T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def s3_monoatomic(self):
        tmp = self._P3(self.equil.s3_monoatomic)
        coeffs = self._E3T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def s3_diatomic(self):
        tmp = self._P3(self.equil.s3_diatomic)
        coeffs = self._E3T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    # ---------#
    # 1-forms #
    # ---------#
    @property
    def u1(self):
        tmp = self._P1([self.equil.u1_1, self.equil.u1_2, self.equil.u1_3])
        coeffs = self._E1T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    # ---------#
    # 2-forms #
    # ---------#
    @property
    def u2(self):
        tmp = self._P2([self.equil.u2_1, self.equil.u2_2, self.equil.u2_3])
        coeffs = self._E2T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    # -----------------------#
    # vector fields (H^1)^3 #
    # -----------------------#
    @property
    def uv(self):
        tmp = self._Pv([self.equil.uv_1, self.equil.uv_2, self.equil.uv_3])
        coeffs = self._EvT.dot(tmp)
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
        coeffs = self._E0T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def u_para0(self):
        tmp = self._P0(self.equil.u_para0)
        coeffs = self._E0T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    # ---------#
    # 3-forms #
    # ---------#
    @property
    def absB3(self):
        tmp = self._P3(self.equil.absB3)
        coeffs = self._E3T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def u_para3(self):
        tmp = self._P3(self.equil.u_para3)
        coeffs = self._E3T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    # ---------#
    # 1-forms #
    # ---------#
    @property
    def b1(self):
        tmp = self._P1([self.equil.b1_1, self.equil.b1_2, self.equil.b1_3])
        coeffs = self._E1T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def unit_b1(self):
        tmp = self._P1([self.equil.unit_b1_1, self.equil.unit_b1_2, self.equil.unit_b1_3])
        coeffs = self._E1T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def gradB1(self):
        tmp = self._P1([self.equil.gradB1_1, self.equil.gradB1_2, self.equil.gradB1_3])
        coeffs = self._E1T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def a1(self):
        tmp = self._P1([self.equil.a1_1, self.equil.a1_2, self.equil.a1_3])
        coeffs = self._E1T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    # ---------#
    # 2-forms #
    # ---------#
    @property
    def b2(self) -> BlockVector:
        tmp = self._P2([self.equil.b2_1, self.equil.b2_2, self.equil.b2_3])
        coeffs = self._E2T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def unit_b2(self):
        tmp = self._P2([self.equil.unit_b2_1, self.equil.unit_b2_2, self.equil.unit_b2_3])
        coeffs = self._E2T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def gradB2(self):
        tmp = self._P2([self.equil.gradB2_1, self.equil.gradB2_2, self.equil.gradB2_3])
        coeffs = self._E2T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def a2(self):
        tmp = self._P2([self.equil.a2_1, self.equil.a2_2, self.equil.a2_3])
        coeffs = self._E2T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    # -----------------------#
    # vector fields (H^1)^3 #
    # -----------------------#
    @property
    def bv(self):
        tmp = self._Pv([self.equil.bv_1, self.equil.bv_2, self.equil.bv_3])
        coeffs = self._EvT.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def unit_bv(self):
        tmp = self._Pv([self.equil.unit_bv_1, self.equil.unit_bv_2, self.equil.unit_bv_3])
        coeffs = self._EvT.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def gradBv(self):
        tmp = self._Pv([self.equil.gradBv_1, self.equil.gradBv_2, self.equil.gradBv_3])
        coeffs = self._EvT.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def av(self):
        tmp = self._Pv([self.equil.av_1, self.equil.av_2, self.equil.av_3])
        coeffs = self._EvT.dot(tmp)
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
        coeffs = self._E0T.dot(tmp)
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
        coeffs = self._E1T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def curl_unit_b1(self):
        tmp = self._P1([self.equil.curl_unit_b1_1, self.equil.curl_unit_b1_2, self.equil.curl_unit_b1_3])
        coeffs = self._E1T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    # ---------#
    # 2-forms #
    # ---------#
    @property
    def j2(self):
        tmp = self._P2([self.equil.j2_1, self.equil.j2_2, self.equil.j2_3])
        coeffs = self._E2T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def curl_unit_b2(self):
        tmp = self._P2([self.equil.curl_unit_b2_1, self.equil.curl_unit_b2_2, self.equil.curl_unit_b2_3])
        coeffs = self._E2T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    # -----------------------#
    # vector fields (H^1)^3 #
    # -----------------------#
    @property
    def jv(self):
        tmp = self._Pv([self.equil.jv_1, self.equil.jv_2, self.equil.jv_3])
        coeffs = self._EvT.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs

    @property
    def curl_unit_bv(self):
        tmp = self._Pv([self.equil.curl_unit_bv_1, self.equil.curl_unit_bv_2, self.equil.curl_unit_bv_3])
        coeffs = self._EvT.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs
