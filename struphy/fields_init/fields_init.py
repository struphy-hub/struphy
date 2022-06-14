import numpy as np

class Initialize_fields:
    '''
    Initialize fields E (as 1-form) and B (as 2-form).

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
        {unperturbed, modes_k, modes_mn, eigfun, noise}

    init_coords : string
        Coordinates of initial condition
        {physical, norm_logical}

    target : list of strings
        mhd variables to be initialize
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

    '''
    
    def __init__(self, DOMAIN, SPACES, general_init, params_init):

        self.DOMAIN = DOMAIN
        self.SPACES = SPACES
        self.init_type   = general_init['type']
        self.init_coords = general_init['coords']

        if self.init_type == 'modes_k':
            self.target   =   params_init['target']
            self.modes_k  = [ params_init['kx'], 
                              params_init['ky'], 
                              params_init['kz'] ]
            self.amp      =   params_init['amp']

            assert np.all(len(self.amp) == len(self.modes_k[0]) == len(self.modes_k[1]) == len(self.modes_k[2]))

        elif self.init_type == 'modes_mn':
            self.target   =   params_init['target']
            self.modes_mn = [ params_init['modes_m'],
                              params_init['modes_n'] ]
            self.amp      =   params_init['amp']
            
        elif self.init_type == 'eigenfun':
            self.n_tor    = params_init['n_tor']
            self.profiles = params_init['profiles']
            self.eig_kind = params_init['eig_kind']
            self.eig_freq = params_init['eig_freq']

        elif self.init_type == 'noise':
            self.target   =   params_init['target']
            self.plane    =   params_init['plane']
            self.amp      =   params_init['amp']

        N_dof_0form = self.SPACES.E0.shape[0]
        N_dof_1form = self.SPACES.E1.shape[0]
        N_dof_2form = self.SPACES.E2.shape[0]
        N_dof_3form = self.SPACES.E3.shape[0]

        self.e1 = np.zeros(N_dof_1form, dtype=float)
        self.b2 = np.zeros(N_dof_2form, dtype=float)

        if self.init_type == 'modes_k':
            self.e1[:] = self.SPACES.projectors.pi_1([self.e1_ini_1, self.e1_ini_2, self.e1_ini_3])
            self.b2[:] = self.SPACES.projectors.pi_2([self.b2_ini_1, self.b2_ini_2, self.b2_ini_3])

        elif self.init_type == 'noise':
            np.random.seed(1607)

            e1_temp_1 = np.empty(self.SPACES.Nbase_1form[0], dtype=float)
            e1_temp_2 = np.empty(self.SPACES.Nbase_1form[1], dtype=float)
            e1_temp_3 = np.empty(self.SPACES.Nbase_1form[2], dtype=float)
            b2_temp_1 = np.empty(self.SPACES.Nbase_2form[0], dtype=float)
            b2_temp_2 = np.empty(self.SPACES.Nbase_2form[1], dtype=float)
            b2_temp_3 = np.empty(self.SPACES.Nbase_2form[2], dtype=float)

            if self.plane == 'xy':
                amps = np.random.rand(8, self.SPACES.NbaseN[0], self.SPACES.NbaseN[1])
                for k in range(self.SPACES.NbaseN[2]):
                    e1_temp_1[:, :, k] = amps[0]
                    e1_temp_2[:, :, k] = amps[1]
                    e1_temp_3[:, :, k] = amps[2]
                    b2_temp_1[:, :, k] = amps[3]
                    b2_temp_2[:, :, k] = amps[4]
                    b2_temp_3[:, :, k] = amps[5]

            elif self.plane == 'yz':
                amps = np.random.rand(8, self.SPACES.NbaseN[1], self.SPACES.NbaseN[2])
                for k in range(self.SPACES.NbaseN[0]):
                    e1_temp_1[k, :, :] = amps[0]
                    e1_temp_2[k, :, :] = amps[1]
                    e1_temp_3[k, :, :] = amps[2]
                    b2_temp_1[k, :, :] = amps[3]
                    b2_temp_2[k, :, :] = amps[4]
                    b2_temp_3[k, :, :] = amps[5]

            elif self.plane == 'xz':
                amps = np.random.rand(8, self.SPACES.NbaseN[0], self.SPACES.NbaseN[2])
                for k in range(self.SPACES.NbaseN[1]):
                    e1_temp_1[:, k, :] = amps[0]
                    e1_temp_2[:, k, :] = amps[1]
                    e1_temp_3[:, k, :] = amps[2]
                    b2_temp_1[:, k, :] = amps[3]
                    b2_temp_2[:, k, :] = amps[4]
                    b2_temp_3[:, k, :] = amps[5]

            self.e1[:] = self.amp*np.concatenate((e1_temp_1.flatten(), e1_temp_2.flatten(), e1_temp_3.flatten()))
            self.b2[:] = self.amp*np.concatenate((b2_temp_1.flatten(), b2_temp_2.flatten(), b2_temp_3.flatten()))

        elif self.init_type == 'modes_mn' or 'eigfun':
            print('modes_mn and eigfun mode are not implemented yet')

        print('electric field'.ljust(16) + 'initialized as 1-form of size', self.e1.size)
        print('magnetic field'.ljust(16) + 'initialized as 2-form of size', self.b2.size)
       


    # ===============================================================
    #                     functions for modes_k
    # ===============================================================
    
    def fun_e_x(self, x, y, z):
        '''Initial electric field (x - component) in physical space.'''

        b1 = 0*x
        
        if 'e1' in self.target :
            if self.init_type == 'modes_k' :
                for i in range(len(self.amp)):
                    e1 += self.amp[i]*np.sin(self.modes_k[0][i]*x + self.modes_k[1][i]*y + self.modes_k[2][i]*z)

        return e1

    def fun_e_y(self, x, y, z):
        '''Initial electric field (y - component) in physical space.'''

        e2 = 0*x

        if 'e2' in self.target :
            if self.init_type == 'modes_k' :
                for i in range(len(self.amp)):
                    e2 += self.amp[i]*np.sin(self.modes_k[0][i]*x + self.modes_k[1][i]*y + self.modes_k[2][i]*z)

        return e2

    def fun_e_z(self, x, y, z):
        '''Initial electric field (z - component) in physical space.'''

        e3 = 0*x

        if 'e3' in self.target :
            if self.init_type == 'modes_k' :
                for i in range(len(self.amp)):
                    e3 += self.amp[i]*np.sin(self.modes_k[0][i]*x + self.modes_k[1][i]*y + self.modes_k[2][i]*z)

        return e3


    def fun_b_x(self, x, y, z):
        '''Initial magnetic field (x - component) in physical space.'''

        b1 = 0*x
        
        if 'b1' in self.target :
            if self.init_type == 'modes_k' :
                for i in range(len(self.amp)):
                    b1 += self.amp[i]*np.cos(self.modes_k[0][i]*x + self.modes_k[1][i]*y + self.modes_k[2][i]*z)

        return b1

    def fun_b_y(self, x, y, z):
        '''Initial magnetic field (y - component) in physical space.'''

        b2 = 0*x

        if 'b2' in self.target :
            if self.init_type == 'modes_k' :
                for i in range(len(self.amp)):
                    b2 += self.amp[i]*np.cos(self.modes_k[0][i]*x + self.modes_k[1][i]*y + self.modes_k[2][i]*z)

        return b2

    def fun_b_z(self, x, y, z):
        '''Initial magnetic field (z - component) in physical space.'''

        b3 = 0*x

        if 'b3' in self.target :
            if self.init_type == 'modes_k' :
                for i in range(len(self.amp)):
                    b3 += self.amp[i]*np.cos(self.modes_k[0][i]*x + self.modes_k[1][i]*y + self.modes_k[2][i]*z)

        return b3



    # ===============================================================
    #               pullback or transform to the form
    # ===============================================================


    def e1_ini_1(self, eta1, eta2, eta3=None):
        '''Initial electric field (1-form on logical domain, 1-component).'''
        
        if self.init_coords == 'physical':
            return self.DOMAIN.pull([self.fun_e_x, self.fun_e_y, self.fun_e_z], eta1, eta2, eta3, '2_form_1')
        elif self.init_coords == 'norm_logical':
            return self.DOMAIN.transform([self.fun_e_x, self.fun_e_y, self.fun_e_z], eta1, eta2, eta3, 'norm_to_2_1')
 
    def e1_ini_2(self, eta1, eta2, eta3=None):
        '''Initial electric field (1-form on logical domain, 2-component).'''
        
        if self.init_coords == 'physical':
            return self.DOMAIN.pull([self.fun_e_x, self.fun_e_y, self.fun_e_z], eta1, eta2, eta3, '2_form_2')
        elif self.init_coords == 'norm_logical':
            return self.DOMAIN.transform([self.fun_e_x, self.fun_e_y, self.fun_e_z], eta1, eta2, eta3, 'norm_to_2_2')

    def e1_ini_3(self, eta1, eta2, eta3=None):
        '''Initial electric field (1-form on logical domain, 3-component).'''
        
        if self.init_coords == 'physical':
            return self.DOMAIN.pull([self.fun_e_x, self.fun_e_y, self.fun_e_z], eta1, eta2, eta3, '2_form_3')
        elif self.init_coords == 'norm_logical':
            return self.DOMAIN.transform([self.fun_e_x, self.fun_e_y, self.fun_e_z], eta1, eta2, eta3, 'norm_to_2_3')
    

    def b2_ini_1(self, eta1, eta2, eta3=None):
        '''Initial magnetic field (2-form on logical domain, 1-component).'''
        
        if self.init_coords == 'physical':
            return self.DOMAIN.pull([self.fun_b_x, self.fun_b_y, self.fun_b_z], eta1, eta2, eta3, '2_form_1')
        elif self.init_coords == 'norm_logical':
            return self.DOMAIN.transform([self.fun_b_x, self.fun_b_y, self.fun_b_z], eta1, eta2, eta3, 'norm_to_2_1')

    def b2_ini_2(self, eta1, eta2, eta3=None):
        '''Initial magnetic field (2-form on logical domain, 2-component).'''
        
        if self.init_coords == 'physical':
            return self.DOMAIN.pull([self.fun_b_x, self.fun_b_y, self.fun_b_z], eta1, eta2, eta3, '2_form_2')
        elif self.init_coords == 'norm_logical':
            return self.DOMAIN.transform([self.fun_b_x, self.fun_b_y, self.fun_b_z], eta1, eta2, eta3, 'norm_to_2_2')

    def b2_ini_3(self, eta1, eta2, eta3=None):
        '''Initial magnetic field (2-form on logical domain, 3-component).'''
        
        if self.init_coords == 'physical':
            return self.DOMAIN.pull([self.fun_b_x, self.fun_b_y, self.fun_b_z], eta1, eta2, eta3, '2_form_3')
        elif self.init_coords == 'norm_logical':
            return self.DOMAIN.transform([self.fun_b_x, self.fun_b_y, self.fun_b_z], eta1, eta2, eta3, 'norm_to_2_3')
    