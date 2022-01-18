class Equilibrium_mhd_logical:
    """
    TODO
    """
    
    def __init__(self, DOMAIN, MHD):
        
        self.DOMAIN = DOMAIN
        self.MHD    = MHD
        
    
    # equilibrium bulk pressure (0-form on logical domain)
    def p0_eq(self, eta1, eta2, eta3):
        return self.DOMAIN.pull(self.MHD.p_eq, eta1, eta2, eta3, '0_form')
           
    
    # equilibrium bulk pressure (3-form on logical domain)
    def p3_eq(self, eta1, eta2, eta3):
        return self.DOMAIN.pull(self.MHD.p_eq, eta1, eta2, eta3, '3_form')        
    
    
    # equilibrium bulk density (0-form on logical domain)
    def r0_eq(self, eta1, eta2, eta3):
        return self.DOMAIN.pull(self.MHD.r_eq, eta1, eta2, eta3, '0_form')
            

    # equilibrium bulk density (3-form on logical domain)
    def r3_eq(self, eta1, eta2, eta3):
        return self.DOMAIN.pull(self.MHD.r_eq, eta1, eta2, eta3, '3_form')
    
    
    # equilibrium magnetic field (0-form absolute value on logical domain)
    def b0_eq(self, eta1, eta2, eta3):
        return self.DOMAIN.pull(self.MHD.b_eq, eta1, eta2, eta3, '0_form')
      

    # equilibrium magnetic field (1-form on logical domain, 1-component)
    def b1_eq_1(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.MHD.b_eq_x, self.MHD.b_eq_y, self.MHD.b_eq_z], eta1, eta2, eta3, '1_form_1')
        
 
    # equilibrium magnetic field (1-form on logical domain, 2-component)
    def b1_eq_2(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.MHD.b_eq_x, self.MHD.b_eq_y, self.MHD.b_eq_z], eta1, eta2, eta3, '1_form_2')
        
    
    # equilibrium magnetic field (1-form on logical domain, 3-component)
    def b1_eq_3(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.MHD.b_eq_x, self.MHD.b_eq_y, self.MHD.b_eq_z], eta1, eta2, eta3, '1_form_3')
    
    
    # equilibrium magnetic field (2-form on logical domain, 1-component)
    def b2_eq_1(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.MHD.b_eq_x, self.MHD.b_eq_y, self.MHD.b_eq_z], eta1, eta2, eta3, '2_form_1')
 
    
    # equilibrium magnetic field (2-form on logical domain, 2-component)
    def b2_eq_2(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.MHD.b_eq_x, self.MHD.b_eq_y, self.MHD.b_eq_z], eta1, eta2, eta3, '2_form_2')
        
    
    # equilibrium magnetic field (2-form on logical domain, 3-component)
    def b2_eq_3(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.MHD.b_eq_x, self.MHD.b_eq_y, self.MHD.b_eq_z], eta1, eta2, eta3, '2_form_3')
    
    
    # equilibrium magnetic field (vector on logical domain, 1-component)
    def bv_eq_1(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.MHD.b_eq_x, self.MHD.b_eq_y, self.MHD.b_eq_z], eta1, eta2, eta3, 'vector_1')
 
    
    # equilibrium magnetic field (vector on logical domain, 2-component)
    def bv_eq_2(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.MHD.b_eq_x, self.MHD.b_eq_y, self.MHD.b_eq_z], eta1, eta2, eta3, 'vector_2')
        
    
    # equilibrium magnetic field (vector on logical domain, 3-component)
    def bv_eq_3(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.MHD.b_eq_x, self.MHD.b_eq_y, self.MHD.b_eq_z], eta1, eta2, eta3, 'vector_3')
        
            
    # equilibrium current (2-form on logical domain, 1-component)
    def j2_eq_1(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.MHD.j_eq_x, self.MHD.j_eq_y, self.MHD.j_eq_z], eta1, eta2, eta3, '2_form_1')
        
        
    # equilibrium current (2-form on logical domain, 2-component)
    def j2_eq_2(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.MHD.j_eq_x, self.MHD.j_eq_y, self.MHD.j_eq_z], eta1, eta2, eta3, '2_form_2')
        
        
    # equilibrium current (2-form on logical domain, 3-component)
    def j2_eq_3(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.MHD.j_eq_x, self.MHD.j_eq_y, self.MHD.j_eq_z], eta1, eta2, eta3, '2_form_3')
    
    
    # equilibrium current (vector on logical domain, 1-component)
    def jv_eq_1(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.MHD.j_eq_x, self.MHD.j_eq_y, self.MHD.j_eq_z], eta1, eta2, eta3, 'vector_1')
        
        
    # equilibrium current (vector on logical domain, 2-component)
    def jv_eq_2(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.MHD.j_eq_x, self.MHD.j_eq_y, self.MHD.j_eq_z], eta1, eta2, eta3, 'vector_2')
        
        
    # equilibrium current (vector on logical domain, 3-component)
    def jv_eq_3(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.MHD.j_eq_x, self.MHD.j_eq_y, self.MHD.j_eq_z], eta1, eta2, eta3, 'vector_3')
        