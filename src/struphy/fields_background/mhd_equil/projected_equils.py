from struphy.fields_background.mhd_equil.base import MHDequilibrium
from struphy.feec.psydac_derham import Derham

class ProjectedMHDequilibrium():
    '''Commuting projections of MHD equilibrium into Derham spaces.
    Return coefficients.'''
    
    def __init__(self, 
                 mhd_equil: MHDequilibrium,
                 derham: Derham):
        
        self._mhd_equil = mhd_equil
        self._derham = derham
        
        # commuting projectors
        self._P0 = derham.P['0']
        self._P1 = derham.P['1']
        self._P2 = derham.P['2']
        self._P3 = derham.P['3']
        self._Pv = derham.P['v']
        
        # transposed extraction operator PolarVector --> BlockVector (identity map in case of no polar splines)
        self._E0T = derham.extraction_ops['0'].transpose()
        self._E1T = derham.extraction_ops['1'].transpose()
        self._E2T = derham.extraction_ops['2'].transpose()
        self._E3T = derham.extraction_ops['3'].transpose()
        self._EvT = derham.extraction_ops['v'].transpose()
    
    @property
    def mhd_equil(self):
        return self._mhd_equil
    
    @property
    def derham(self):
        return self._derham
    
    #---------#
    # 0-forms #
    #---------#
    @property
    def absB0(self):
        tmp = self._P0(self.mhd_equil.absB0)
        coeffs = self._E0T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs
    
    @property
    def curl_unit_b_dot_b0(self):
        tmp = self._P0(self.mhd_equil.curl_unit_b_dot_b0)
        coeffs = self._E0T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs
    
    @property
    def p0(self):
        tmp = self._P0(self.mhd_equil.p0)
        coeffs = self._E0T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs
    
    @property
    def n0(self):
        tmp = self._P0(self.mhd_equil.n0)
        coeffs = self._E0T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs
    
    @property
    def s0_monoatomic(self):
        tmp = self._P0(self.mhd_equil.s0_monoatomic)
        coeffs = self._E0T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs
    
    @property
    def s0_diatomic(self):
        tmp = self._P0(self.mhd_equil.s0_diatomic)
        coeffs = self._E0T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs
    
    #---------#
    # 3-forms #
    #---------#
    @property
    def absB3(self):
        tmp = self._P3(self.mhd_equil.absB3)
        coeffs = self._E3T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs
    
    @property
    def p3(self):
        tmp = self._P3(self.mhd_equil.p3)
        coeffs = self._E3T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs
    
    @property
    def n3(self):
        tmp = self._P3(self.mhd_equil.n3)
        coeffs = self._E3T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs
    
    @property
    def s3_monoatomic(self):
        tmp = self._P3(self.mhd_equil.s3_monoatomic)
        coeffs = self._E3T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs
    
    @property
    def s3_diatomic(self):
        tmp = self._P3(self.mhd_equil.s3_diatomic)
        coeffs = self._E3T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs
    
    #---------#
    # 1-forms #
    #---------#
    @property
    def b1(self):
        tmp = self._P1([self.mhd_equil.b1_1,
                        self.mhd_equil.b1_2,
                        self.mhd_equil.b1_3])
        coeffs = self._E1T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs
    
    @property
    def unit_b1(self):
        tmp = self._P1([self.mhd_equil.unit_b1_1,
                        self.mhd_equil.unit_b1_2,
                        self.mhd_equil.unit_b1_3])
        coeffs = self._E1T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs
    
    @property
    def curl_unit_b1(self):
        tmp = self._P1([self.mhd_equil.curl_unit_b1_1,
                        self.mhd_equil.curl_unit_b1_2,
                        self.mhd_equil.curl_unit_b1_3])
        coeffs = self._E1T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs
    
    @property
    def a1(self):
        tmp = self._P1([self.mhd_equil.a1_1,
                        self.mhd_equil.a1_2,
                        self.mhd_equil.a1_3])
        coeffs = self._E1T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs
    
    @property
    def j1(self):
        tmp = self._P1([self.mhd_equil.j1_1,
                        self.mhd_equil.j1_2,
                        self.mhd_equil.j1_3])
        coeffs = self._E1T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs
    
    @property
    def gradB1(self):
        tmp = self._P1([self.mhd_equil.gradB1_1,
                        self.mhd_equil.gradB1_2,
                        self.mhd_equil.gradB1_3])
        coeffs = self._E1T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs
    
    #---------#
    # 2-forms #
    #---------#
    @property
    def b2(self):
        tmp = self._P2([self.mhd_equil.b2_1,
                        self.mhd_equil.b2_2,
                        self.mhd_equil.b2_3])
        coeffs = self._E2T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs
    
    @property
    def unit_b2(self):
        tmp = self._P2([self.mhd_equil.unit_b2_1,
                        self.mhd_equil.unit_b2_2,
                        self.mhd_equil.unit_b2_3])
        coeffs = self._E2T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs
    
    @property
    def curl_unit_b2(self):
        tmp = self._P2([self.mhd_equil.curl_unit_b2_1,
                        self.mhd_equil.curl_unit_b2_2,
                        self.mhd_equil.curl_unit_b2_3])
        coeffs = self._E2T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs
    
    @property
    def a2(self):
        tmp = self._P2([self.mhd_equil.a2_1,
                        self.mhd_equil.a2_2,
                        self.mhd_equil.a2_3])
        coeffs = self._E2T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs
    
    @property
    def j2(self):
        tmp = self._P2([self.mhd_equil.j2_1,
                        self.mhd_equil.j2_2,
                        self.mhd_equil.j2_3])
        coeffs = self._E2T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs
    
    @property
    def gradB2(self):
        tmp = self._P2([self.mhd_equil.gradB2_1,
                        self.mhd_equil.gradB2_2,
                        self.mhd_equil.gradB2_3])
        coeffs = self._E2T.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs
    
    #-----------------------#
    # vector fields (H^1)^3 #
    #-----------------------#
    @property
    def bv(self):
        tmp = self._Pv([self.mhd_equil.bv_1,
                        self.mhd_equil.bv_2,
                        self.mhd_equil.bv_3])
        coeffs = self._EvT.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs
    
    @property
    def unit_bv(self):
        tmp = self._Pv([self.mhd_equil.unit_bv_1,
                        self.mhd_equil.unit_bv_2,
                        self.mhd_equil.unit_bv_3])
        coeffs = self._EvT.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs
    
    @property
    def curl_unit_bv(self):
        tmp = self._Pv([self.mhd_equil.curl_unit_bv_1,
                        self.mhd_equil.curl_unit_bv_2,
                        self.mhd_equil.curl_unit_bv_3])
        coeffs = self._EvT.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs
    
    @property
    def av(self):
        tmp = self._Pv([self.mhd_equil.av_1,
                        self.mhd_equil.av_2,
                        self.mhd_equil.av_3])
        coeffs = self._EvT.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs
    
    @property
    def jv(self):
        tmp = self._Pv([self.mhd_equil.jv_1,
                        self.mhd_equil.jv_2,
                        self.mhd_equil.jv_3])
        coeffs = self._EvT.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs
    
    @property
    def gradBv(self):
        tmp = self._Pv([self.mhd_equil.gradBv_1,
                        self.mhd_equil.gradBv_2,
                        self.mhd_equil.gradBv_3])
        coeffs = self._EvT.dot(tmp)
        coeffs.update_ghost_regions()
        return coeffs
    
    