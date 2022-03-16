import numpy as np

class Initialize_mhd:
    '''
    Initialize MHD variables u or n (as 0-form) and B (as 1-form).

    Parameters
    ----------
    DOMAIN : obj
        mapped domain
    SPACES : obj
        2d or 3d tensor-product B-spline space
    general_init : dict
        Keys are "type", "coords", "basis_u" and "basis_p" (see parameters.yml)
    params_init : dict
        The parameters needed to define the initial conditions (see Notes).

    Attributes
    ----------
    u0 : np.array 
        Flattened initial coefficients of log(density) as 0-form.
                
    n0 : np.array
        Flattened initial coefficients of density as 0-form.
        
    b1 : np.array
        Flattened initial coefficients of magnetic field as 1-form.
        
    init_type : string
        Types of initialization
        {unperturbed, modes_k, modes_mn, eigfun, noise}

    init_coords : string
        Coordinates of initial condition
        {physical, norm_logical}

    target : list of strings
        mhd variables to be initialize
        {b1, b2, b3, u, n}

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
    Currently only 'modes_k' type is available.

    For this case, multiple targets are available.
    ex) target = [b1, b2, b3, n, u]

    For 'modes_k', multiple modes can be initialized via lists.
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

        self.n0 = np.zeros(N_dof_0form, dtype=float)
        self.u0 = np.zeros(N_dof_0form, dtype=float)
        temp_b  = np.zeros(N_dof_1form, dtype=float)
        self.b1 = np.zeros(self.SPACES.Nbase_1form[0], dtype=float)
        self.b2 = np.zeros(self.SPACES.Nbase_1form[1], dtype=float)
        self.b3 = np.zeros(self.SPACES.Nbase_1form[2], dtype=float)

        if self.init_type == 'modes_k':
            self.n0[:] = self.SPACES.projectors.pi_0(self.n0_ini)
            self.u0[:] = self.SPACES.projectors.pi_0(self.u0_ini)
            temp_b[:] = self.SPACES.projectors.pi_1([self.b1_ini_1, self.b1_ini_2, self.b1_ini_3])
            temp_b1, temp_b2, temp_b3 = np.split(temp_b, [self.SPACES.Ntot_1form[0], self.SPACES.Ntot_1form[0] + self.SPACES.Ntot_1form[1]]   )
            self.b1[:] = temp_b1.reshape(self.SPACES.Nbase_1form[0])
            self.b2[:] = temp_b2.reshape(self.SPACES.Nbase_1form[1])
            self.b3[:] = temp_b3.reshape(self.SPACES.Nbase_1form[2])


        elif self.init_type == 'noise' or 'modes_mn' or 'eigfun':
            print('noise, modes_mn and eigfun mode are not implemented yet')

        print('density'.ljust(16) + 'initialized as 0-form of size', self.n0.size)
        print('magnetic field'.ljust(16) + 'initialized as 1-form of size', self.b1.size)
            
    # ===============================================================
    #                     functions for modes_k
    # ===============================================================
    def fun_n(self, x, y, z):
        '''Initial bulk pressure in physical space.'''

        n = 0*x + 1.0

        if 'n' in self.target :
            if self.init_type == 'modes_k' :
                for i in range(len(self.amp)):
                    n += self.amp[i]*np.sin(self.modes_k[0][i]*x + self.modes_k[1][i]*y + self.modes_k[2][i]*z)

        return n

    def fun_u(self, x, y, z):
        '''Initial bulk pressure in physical space.'''

        u = 0*x

        if 'u' in self.target :
            if self.init_type == 'modes_k' :
                for i in range(len(self.amp)):
                    u += self.amp[i]*np.sin(self.modes_k[0][i]*x + self.modes_k[1][i]*y + self.modes_k[2][i]*z)

        return u
    
    def fun_b_x(self, x, y, z):
        '''Initial magnetic field (x - component) in physical space.'''

        b1 = 0*x

        if 'b1' in self.target :
            if self.init_type == 'modes_k' :
                for i in range(len(self.amp)):
                    b1 += self.amp[i]*np.sin(self.modes_k[0][i]*x + self.modes_k[1][i]*y + self.modes_k[2][i]*z)


        return b1
 
    def fun_b_y(self, x, y, z):
        '''Initial magnetic field (y - component) in physical space.'''

        b2 = 0*x

        if 'b2' in self.target :
            if self.init_type == 'modes_k' :
                for i in range(len(self.amp)):
                    b2 += self.amp[i]*np.sin(self.modes_k[0][i]*x + self.modes_k[1][i]*y + self.modes_k[2][i]*z)

        return b2

    def fun_b_z(self, x, y, z):
        '''Initial magnetic field (z - component) in physical space.'''

        b3 = 0*x

        if 'b3' in self.target :
            if self.init_type == 'modes_k' :
                for i in range(len(self.amp)):
                    b3 += self.amp[i]*np.sin(self.modes_k[0][i]*x + self.modes_k[1][i]*y + self.modes_k[2][i]*z)

        return b3

    # ===============================================================
    #               pullback or transform to the form
    # ===============================================================

    def n0_ini(self, eta1, eta2, eta3=None):
        '''Initial density (0-form on logical domain).'''

        if self.init_coords == 'physical':
            return self.DOMAIN.pull(self.fun_n, eta1, eta2, eta3, '0_form')
        
        elif self.init_coords == 'norm_logical':
            return self.DOMAIN.transformation(self.fun_n, eta1, eta2, eta3, 'norm_to_0')


    def u0_ini(self, eta1, eta2, eta3=None):
        '''Initial log of density (0-form on logical domain).'''
        
        if self.init_coords == 'physical':
            return self.DOMAIN.pull(self.fun_u, eta1, eta2, eta3, '0_form')
        
        elif self.init_coords == 'norm_logical':
            return self.DOMAIN.transformation(self.fun_u, eta1, eta2, eta3, 'norm_to_0')


    
    def b1_ini_1(self, eta1, eta2, eta3=None):
        '''Initial magnetic field (1-form on logical domain, 1-component).'''
        
        if self.init_coords == 'physical':
            return self.DOMAIN.pull([self.fun_b_x, self.fun_b_y, self.fun_b_z], eta1, eta2, eta3, '1_form_1')
        elif self.init_coords == 'norm_logical':
            return self.DOMAIN.transformation([self.fun_b_x, self.fun_b_y, self.fun_b_z], eta1, eta2, eta3, 'norm_to_1_1')

 
    def b1_ini_2(self, eta1, eta2, eta3=None):
        '''Initial flow velocity (1-form on logical domain, 2-component).'''
        
        if self.init_coords == 'physical':
            return self.DOMAIN.pull([self.fun_b_x, self.fun_b_y, self.fun_b_z], eta1, eta2, eta3, '1_form_2')
        elif self.init_coords == 'norm_logical':
            return self.DOMAIN.transformation([self.fun_b_x, self.fun_b_y, self.fun_b_z], eta1, eta2, eta3, 'norm_to_1_2')
 

    def b1_ini_3(self, eta1, eta2, eta3=None):
        '''Initial flow velocity (1-form on logical domain, 3-component).'''
        
        if self.init_coords == 'physical':
            return self.DOMAIN.pull([self.fun_b_x, self.fun_b_y, self.fun_b_z], eta1, eta2, eta3, '1_form_3')
        elif self.init_coords == 'norm_logical':
            return self.DOMAIN.transformation([self.fun_b_x, self.fun_b_y, self.fun_b_z], eta1, eta2, eta3, 'norm_to_1_3')