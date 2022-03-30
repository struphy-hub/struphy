class Equilibrium_fields_logical:
    """
    TODO
    """
    
    def __init__(self, DOMAIN, FIELDS):
        
        self.DOMAIN = DOMAIN
        self.FIELDS = FIELDS
    


    # equilibrium electric field (0-form absolute value on logical domain)
    def e0_eq(self, eta1, eta2, eta3):
        return self.DOMAIN.pull(self.FIELDS.e_eq, eta1, eta2, eta3, '0_form')
      

    # equilibrium electric field (1-form on logical domain, 1-component)
    def e1_eq_1(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.FIELDS.e_eq_x, self.FIELDS.e_eq_y, self.FIELDS.e_eq_z], eta1, eta2, eta3, '1_form_1')
        
 
    # equilibrium electric field (1-form on logical domain, 2-component)
    def e1_eq_2(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.FIELDS.e_eq_x, self.FIELDS.e_eq_y, self.FIELDS.e_eq_z], eta1, eta2, eta3, '1_form_2')
        
    
    # equilibrium electric field (1-form on logical domain, 3-component)
    def e1_eq_3(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.FIELDS.e_eq_x, self.FIELDS.e_eq_y, self.FIELDS.e_eq_z], eta1, eta2, eta3, '1_form_3')
    
    
    # equilibrium electric field (2-form on logical domain, 1-component)
    def e2_eq_1(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.FIELDS.e_eq_x, self.FIELDS.e_eq_y, self.FIELDS.e_eq_z], eta1, eta2, eta3, '2_form_1')
 
    
    # equilibrium electric field (2-form on logical domain, 2-component)
    def e2_eq_2(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.FIELDS.e_eq_x, self.FIELDS.e_eq_y, self.FIELDS.e_eq_z], eta1, eta2, eta3, '2_form_2')
        
    
    # equilibrium electric field (2-form on logical domain, 3-component)
    def e2_eq_3(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.FIELDS.e_eq_x, self.FIELDS.e_eq_y, self.FIELDS.e_eq_z], eta1, eta2, eta3, '2_form_3')
    
    
    # equilibrium electric field (vector on logical domain, 1-component)
    def ev_eq_1(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.FIELDS.e_eq_x, self.FIELDS.e_eq_y, self.FIELDS.e_eq_z], eta1, eta2, eta3, 'vector_1')
 
    
    # equilibrium electric field (vector on logical domain, 2-component)
    def ev_eq_2(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.FIELDS.e_eq_x, self.FIELDS.e_eq_y, self.FIELDS.e_eq_z], eta1, eta2, eta3, 'vector_2')
        
    
    # equilibrium electric field (vector on logical domain, 3-component)
    def ev_eq_3(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.FIELDS.e_eq_x, self.FIELDS.e_eq_y, self.FIELDS.e_eq_z], eta1, eta2, eta3, 'vector_3')
        




    
    # equilibrium magnetic field (0-form absolute value on logical domain)
    def b0_eq(self, eta1, eta2, eta3):
        return self.DOMAIN.pull(self.FIELDS.b_eq, eta1, eta2, eta3, '0_form')
      

    # equilibrium magnetic field (1-form on logical domain, 1-component)
    def b1_eq_1(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.FIELDS.b_eq_x, self.FIELDS.b_eq_y, self.FIELDS.b_eq_z], eta1, eta2, eta3, '1_form_1')
        
 
    # equilibrium magnetic field (1-form on logical domain, 2-component)
    def b1_eq_2(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.FIELDS.b_eq_x, self.FIELDS.b_eq_y, self.FIELDS.b_eq_z], eta1, eta2, eta3, '1_form_2')
        
    
    # equilibrium magnetic field (1-form on logical domain, 3-component)
    def b1_eq_3(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.FIELDS.b_eq_x, self.FIELDS.b_eq_y, self.FIELDS.b_eq_z], eta1, eta2, eta3, '1_form_3')
    
    
    # equilibrium magnetic field (2-form on logical domain, 1-component)
    def b2_eq_1(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.FIELDS.b_eq_x, self.FIELDS.b_eq_y, self.FIELDS.b_eq_z], eta1, eta2, eta3, '2_form_1')
 
    
    # equilibrium magnetic field (2-form on logical domain, 2-component)
    def b2_eq_2(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.FIELDS.b_eq_x, self.FIELDS.b_eq_y, self.FIELDS.b_eq_z], eta1, eta2, eta3, '2_form_2')
        
    
    # equilibrium magnetic field (2-form on logical domain, 3-component)
    def b2_eq_3(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.FIELDS.b_eq_x, self.FIELDS.b_eq_y, self.FIELDS.b_eq_z], eta1, eta2, eta3, '2_form_3')
    
    
    # equilibrium magnetic field (vector on logical domain, 1-component)
    def bv_eq_1(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.FIELDS.b_eq_x, self.FIELDS.b_eq_y, self.FIELDS.b_eq_z], eta1, eta2, eta3, 'vector_1')
 
    
    # equilibrium magnetic field (vector on logical domain, 2-component)
    def bv_eq_2(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.FIELDS.b_eq_x, self.FIELDS.b_eq_y, self.FIELDS.b_eq_z], eta1, eta2, eta3, 'vector_2')
        
    
    # equilibrium magnetic field (vector on logical domain, 3-component)
    def bv_eq_3(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.FIELDS.b_eq_x, self.FIELDS.b_eq_y, self.FIELDS.b_eq_z], eta1, eta2, eta3, 'vector_3')
        



            
    # equilibrium current (2-form on logical domain, 1-component)
    def j2_eq_1(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.FIELDS.j_eq_x, self.FIELDS.j_eq_y, self.FIELDS.j_eq_z], eta1, eta2, eta3, '2_form_1')
        
        
    # equilibrium current (2-form on logical domain, 2-component)
    def j2_eq_2(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.FIELDS.j_eq_x, self.FIELDS.j_eq_y, self.FIELDS.j_eq_z], eta1, eta2, eta3, '2_form_2')
        
        
    # equilibrium current (2-form on logical domain, 3-component)
    def j2_eq_3(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.FIELDS.j_eq_x, self.FIELDS.j_eq_y, self.FIELDS.j_eq_z], eta1, eta2, eta3, '2_form_3')
    
    
    # equilibrium current (vector on logical domain, 1-component)
    def jv_eq_1(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.FIELDS.j_eq_x, self.FIELDS.j_eq_y, self.FIELDS.j_eq_z], eta1, eta2, eta3, 'vector_1')
        
        
    # equilibrium current (vector on logical domain, 2-component)
    def jv_eq_2(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.FIELDS.j_eq_x, self.FIELDS.j_eq_y, self.FIELDS.j_eq_z], eta1, eta2, eta3, 'vector_2')
        
        
    # equilibrium current (vector on logical domain, 3-component)
    def jv_eq_3(self, eta1, eta2, eta3):
        return self.DOMAIN.pull([self.FIELDS.j_eq_x, self.FIELDS.j_eq_y, self.FIELDS.j_eq_z], eta1, eta2, eta3, 'vector_3')
        