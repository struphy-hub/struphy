import numpy as np

class Initialize_emw:
    '''
    Initialize Maxwell variables E (as 1-form) and B (as 2-form)

    Parameters
    ----------
    DOMAIN : class
        mapped domain
    SPACES : class
        2d or 3d tensor-product B-spline space
    general_init : dict
        Keys are "type", "coords", "basis_u" and "basis_p" (see parameters.yml)
    params_init : dict
        The parameters needed to define the initial conditions (see Notes).

    Attributes
    ----------
    e1 : np.array
        Flattened initial coefficients of electric field as 1-form.

    b2 : np.array
        Flattened initial coefficients of magnetic field as 2-form.

    init_type : string
        Types of initialization
        {only noise} (right now, will be changed)

    init_coords : string
        Coordinates of initial condition
        {physical, norm_logical}

    target : list of strings
        maxwell variables to be initialize
        {e1, e2, e3, b1, b2, b3}

    modes_k : 2d list of floats
        wavelength of modes_k
        kx = modes_k[0] : list of float
        ky = modes_k[1] : list of float
        kz = modes_k[2] : list of float
    
    modes_mn : 2d list of floats
        mode number of modes_mn
        m = modes_mn[0] : list of float
        n = modes_mn[1] : list of float

    amp : list of floats
        amplitude of modes

    n_tor : int
        toroidal mode number

    profiles : boolean
        project equilibrium profiles

    eig_kind : int
        real {11} or imag {12} part

    eig_freq : float
        squared eigenfreq

    Notes
    -----
    Currently only 'modes_k' and 'noise' types are available.

    For both cases, multiple targets are available.
    ex) target = [e2, e3, b2, b3]

    For 'modes_k', multiple modes are available
    ex) modes_k[0] = [0.7  , 0.8 ]
        modes_k[1] = [0.   , 0.1 ]
        modes_k[2] = [0.3  , 0.  ]
        amp        = [0.001, 0.01]
    '''
    
    def __init__(self, DOMAIN, SPACES, general_init, params_init):

        self.DOMAIN = DOMAIN
        self.SPACES = SPACES
        self.init_type   = general_init['type']
        self.init_coords = general_init['coords']


        if self.init_type == 'modes_k':
            self.target   =   params_init['target']
            self.modes_k  = [ params_init['kx'], params_init['ky'], params_init['kz'] ]
            self.amp      =   params_init['amp']
            assert np.all(len(self.amp) == len(self.modes_k[0]) == len(self.modes_k[1]) == len(self.modes_k[2]))

        elif self.init_type == 'modes_mn':
            self.target     =   params_init['target']
            self.modes_mn   = [ params_init['modes_m'],params_init['modes_n'] ]
            self.amp        =   params_init['amp']
            
        elif self.init_type == 'noise':
            self.target     =   params_init['target']
            self.type       =   params_init['type']
            self.direction  =   params_init['direction']
            self.plane      =   params_init['plane']
            self.amp        =   params_init['amp']


        N_dof_1form = self.SPACES.E1.shape[0]
        N_dof_2form = self.SPACES.E2.shape[0]


        self.e1 = np.zeros(N_dof_1form, dtype=float)
        self.b2 = np.zeros(N_dof_2form, dtype=float)



        if self.init_type == 'modes_k':
            self.e1[:] = self.SPACES.projectors.pi_1([self.e1_ini_1, self.e1_ini_2, self.e1_ini_3])
            self.b2[:] = self.SPACES.projectors.pi_2([self.b2_ini_1, self.b2_ini_2, self.b2_ini_3])


            
        elif self.init_type == 'noise':
            np.random.seed(1607)
            dim_list = self.SPACES.Nbase_1form[0]  # assuming periodic boundary conditions in all directions 
            e1_temp_1 = np.empty(dim_list, dtype=float)
            e1_temp_2 = np.empty(dim_list, dtype=float)
            e1_temp_3 = np.empty(dim_list, dtype=float)
            b2_temp_1 = np.empty(dim_list, dtype=float)
            b2_temp_2 = np.empty(dim_list, dtype=float)
            b2_temp_3 = np.empty(dim_list, dtype=float)


            if self.type == "direction":
                if self.direction == 'x':
                    amps = np.random.rand(9, dim_list[0])
                    for j in range(dim_list[1]):
                        for k in range(dim_list[2]):
                            e1_temp_1[:, j, k] = amps[0]
                            e1_temp_2[:, j, k] = amps[1]
                            e1_temp_3[:, j, k] = amps[2]
                            b2_temp_1[:, j, k] = amps[3]
                            b2_temp_2[:, j, k] = amps[4]
                            b2_temp_3[:, j, k] = amps[5]


                elif self.direction == 'y':
                    amps = np.random.rand(9, dim_list[1])
                    for j in range(dim_list[0]):
                        for k in range(dim_list[2]):
                            e1_temp_1[j, :, k] = amps[0]
                            e1_temp_2[j, :, k] = amps[1]
                            e1_temp_3[j, :, k] = amps[2]
                            b2_temp_1[j, :, k] = amps[3]
                            b2_temp_2[j, :, k] = amps[4]
                            b2_temp_3[j, :, k] = amps[5]


                elif self.direction == 'z':
                    amps = np.random.rand(9, dim_list[2])
                    for j in range(dim_list[0]):
                        for k in range(dim_list[1]):
                            e1_temp_1[j, k, :] = amps[0]
                            e1_temp_2[j, k, :] = amps[1]
                            e1_temp_3[j, k, :] = amps[2]
                            b2_temp_1[j, k, :] = amps[3]
                            b2_temp_2[j, k, :] = amps[4]
                            b2_temp_3[j, k, :] = amps[5]



            elif self.type == "plane":
                if self.plane == 'xy':
                    amps = np.random.rand(9, self.SPACES.NbaseN[0], self.SPACES.NbaseN[1])
                    for k in range(self.SPACES.NbaseN[2]):
                        e1_temp_1[:, :, k] = amps[0]
                        e1_temp_2[:, :, k] = amps[1]
                        e1_temp_3[:, :, k] = amps[2]
                        b2_temp_1[:, :, k] = amps[3]
                        b2_temp_2[:, :, k] = amps[4]
                        b2_temp_3[:, :, k] = amps[5]


                elif self.plane == 'yz':
                    amps = np.random.rand(9, self.SPACES.NbaseN[1], self.SPACES.NbaseN[2])
                    for k in range(self.SPACES.NbaseN[0]):
                        e1_temp_1[k, :, :] = amps[0]
                        e1_temp_2[k, :, :] = amps[1]
                        e1_temp_3[k, :, :] = amps[2]
                        b2_temp_1[k, :, :] = amps[3]
                        b2_temp_2[k, :, :] = amps[4]
                        b2_temp_3[k, :, :] = amps[5]


                elif self.plane == 'xz':
                    amps = np.random.rand(9, self.SPACES.NbaseN[0], self.SPACES.NbaseN[2])
                    for k in range(self.SPACES.NbaseN[1]):
                        e1_temp_1[:, k, :] = amps[0]
                        e1_temp_2[:, k, :] = amps[1]
                        e1_temp_3[:, k, :] = amps[2]
                        b2_temp_1[:, k, :] = amps[3]
                        b2_temp_2[:, k, :] = amps[4]
                        b2_temp_3[:, k, :] = amps[5]


            elif self.type == "volume":
                amps = np.random.rand(9, dim_list[0], dim_list[1], dim_list[2])
                e1_temp_1[:, :, :] = amps[0]
                e1_temp_2[:, :, :] = amps[1]
                e1_temp_3[:, :, :] = amps[2]
                b2_temp_1[:, :, :] = amps[3]
                b2_temp_2[:, :, :] = amps[4]
                b2_temp_3[:, :, :] = amps[5]



            if not 'e1' in self.target: e1_temp_1[:, :, :] = 0.
            if not 'e2' in self.target: e1_temp_2[:, :, :] = 0.
            if not 'e3' in self.target: e1_temp_3[:, :, :] = 0.
            if not 'b1' in self.target: b2_temp_1[:, :, :] = 0.
            if not 'b2' in self.target: b2_temp_2[:, :, :] = 0.
            if not 'b3' in self.target: b2_temp_3[:, :, :] = 0.

            self.e1[:] = self.amp*np.concatenate((e1_temp_1.flatten(), e1_temp_2.flatten(), e1_temp_3.flatten()))
            self.b2[:] = self.amp*np.concatenate((b2_temp_1.flatten(), b2_temp_2.flatten(), b2_temp_3.flatten()))


        elif self.init_type == 'modes_mn' or 'eigfun':
            print('modes_mn and eigfun mode are not implemented yet')

        print('electric field'.ljust(16) + 'initialized as 1-form of size', self.e1.size)
        print('magnetic field'.ljust(16) + 'initialized as 2-form of size', self.b2.size)

            
            
    # ===============================================================
    #                     functions for modes_k
    # ===============================================================

    # initial electric field (x - component)
    def fun_e_x(self, x, y, z):

        e1 = 0*x
        
        if 'e1' in self.target :
            if self.init_type == 'modes_k' :
                for i in range(len(self.amp)):
                    e1 += self.amp[i]*np.sin(self.modes_k[0][i]*x + self.modes_k[1][i]*y + self.modes_k[2][i]*z)

        return e1

    # initial electricfield (y - component)
    def fun_e_y(self, x, y, z):
        
        e2 = 0*x

        if 'e2' in self.target :
            if self.init_type == 'modes_k' :
                for i in range(len(self.amp)):
                    e2 += self.amp[i]*np.sin(self.modes_k[0][i]*x + self.modes_k[1][i]*y + self.modes_k[2][i]*z)

        return e2

    # initial electric field (z - component)
    def fun_e_z(self, x, y, z):

        e3 = 0*x

        if 'e3' in self.target :
            if self.init_type == 'modes_k' :
                for i in range(len(self.amp)):
                    e3 += self.amp[i]*np.sin(self.modes_k[0][i]*x + self.modes_k[1][i]*y + self.modes_k[2][i]*z)

        return e3

    # initial magnetic field (x - component)
    def fun_b_x(self, x, y, z):

        b1 = 0*x
        
        if 'b1' in self.target :
            if self.init_type == 'modes_k' :
                for i in range(len(self.amp)):
                    b1 += self.amp[i]*np.sin(self.modes_k[0][i]*x + self.modes_k[1][i]*y + self.modes_k[2][i]*z)

        return b1

    # initial magnetic field (y - component)
    def fun_b_y(self, x, y, z):
        
        b2 = 0*x

        if 'b2' in self.target :
            if self.init_type == 'modes_k' :
                for i in range(len(self.amp)):
                    b2 += self.amp[i]*np.sin(self.modes_k[0][i]*x + self.modes_k[1][i]*y + self.modes_k[2][i]*z)

        return b2

    # initial magnetic field (z - component)
    def fun_b_z(self, x, y, z):

        b3 = 0*x

        if 'b3' in self.target :
            if self.init_type == 'modes_k' :
                for i in range(len(self.amp)):
                    b3 += self.amp[i]*np.sin(self.modes_k[0][i]*x + self.modes_k[1][i]*y + self.modes_k[2][i]*z)

        return b3

 
    # ===============================================================
    #               pullback or transform to the form
    # ===============================================================

    # initial electric field (1-form on logical domain, 1-component)
    def e1_ini_1(self, eta1, eta2, eta3=None):
        
        if self.init_coords == 'physical':
            return self.DOMAIN.pull([self.fun_e_x, self.fun_e_y, self.fun_e_z], eta1, eta2, eta3, '1_form_1')
        elif self.init_coords == 'norm_logical':
            return self.DOMAIN.transformation([self.fun_e_x, self.fun_e_y, self.fun_e_z], eta1, eta2, eta3, 'norm_to_1_1')
  
    # initial electric field (1-form on logical domain, 2-component)
    def e1_ini_2(self, eta1, eta2, eta3=None):
        
        if self.init_coords == 'physical':
            return self.DOMAIN.pull([self.fun_e_x, self.fun_e_y, self.fun_e_z], eta1, eta2, eta3, '1_form_2')
        elif self.init_coords == 'norm_logical':
            return self.DOMAIN.transformation([self.fun_e_x, self.fun_e_y, self.fun_e_z], eta1, eta2, eta3, 'norm_to_1_2')

    # initial electric field (1-form on logical domain, 3-component)
    def e1_ini_3(self, eta1, eta2, eta3=None):
        
        if self.init_coords == 'physical':
            return self.DOMAIN.pull([self.fun_e_x, self.fun_e_y, self.fun_e_z], eta1, eta2, eta3, '1_form_3')
        elif self.init_coords == 'norm_logical':
            return self.DOMAIN.transformation([self.fun_e_x, self.fun_e_y, self.fun_e_z], eta1, eta2, eta3, 'norm_to_1_3')
            
    # initial magnetic field (2-form on logical domain, 1-component)
    def b2_ini_1(self, eta1, eta2, eta3=None):
        
        if self.init_coords == 'physical':
            return self.DOMAIN.pull([self.fun_b_x, self.fun_b_y, self.fun_b_z], eta1, eta2, eta3, '2_form_1')
        elif self.init_coords == 'norm_logical':
            return self.DOMAIN.transformation([self.fun_b_x, self.fun_b_y, self.fun_b_z], eta1, eta2, eta3, 'norm_to_2_1')
 
    # initial magnetic field (2-form on logical domain, 2-component)
    def b2_ini_2(self, eta1, eta2, eta3=None):
        
        if self.init_coords == 'physical':
            return self.DOMAIN.pull([self.fun_b_x, self.fun_b_y, self.fun_b_z], eta1, eta2, eta3, '2_form_2')
        elif self.init_coords == 'norm_logical':
            return self.DOMAIN.transformation([self.fun_b_x, self.fun_b_y, self.fun_b_z], eta1, eta2, eta3, 'norm_to_2_2')

    # initial magnetic field (2-form on logical domain, 3-component)
    def b2_ini_3(self, eta1, eta2, eta3=None):
        
        if self.init_coords == 'physical':
            return self.DOMAIN.pull([self.fun_b_x, self.fun_b_y, self.fun_b_z], eta1, eta2, eta3, '2_form_3')
        elif self.init_coords == 'norm_logical':
            return self.DOMAIN.transformation([self.fun_b_x, self.fun_b_y, self.fun_b_z], eta1, eta2, eta3, 'norm_to_2_3')

 