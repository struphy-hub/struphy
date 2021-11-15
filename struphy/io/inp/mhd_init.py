import yaml
import numpy as np

'''
Module to initialize all mhd variables.
'''

class Initialize_mhd:
    '''
    Read "parameters.yaml" file and then initialize mhd variables.

    Parameters
    ----------
    DOMAIN : class
        mapped domain
    SPACES : class
        2d or 3d tensor-product B-spline space
    file_in : yml file
        parameters.yml

    Attributes
    ----------
    basis_p : int
        form of the basis function for pressure(pp) 
        {0, 3}

    basis_u : int
        form of the basis function for flow velocity(up) 
        {0, 1, 2}

    init_type : string
        Types of initialization
        {unperturbed, modes_k, modes_mn, eigfun, noise}

    init_coords : string
        Coordinates of initial condition
        {physical, norm_logical}

    target : list of strings
        mhd variables to be initialize
        {b1, b2, b3, u1, u2, u3, p, r}

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
    ex) target = [b1, b2, b3, r, u1]

    For 'modes_k', multiple modes are available
    ex) modes_k[0] = [0.7  , 0.8 ]
        modes_k[1] = [0.   , 0.1 ]
        modes_k[2] = [0.3  , 0.  ]
        amp        = [0.001, 0.01]
    '''
    
    def __init__(self, DOMAIN, SPACES, file_in):

        with open(file_in) as file:
            params = yaml.load(file, Loader=yaml.FullLoader)

        self.DOMAIN = DOMAIN
        self.SPACES = SPACES
        self.init_type   = params['mhd_init']['type']
        self.init_coords = params['mhd_init']['coords']
        self.basis_p = params['forms']['basis_p']
        self.basis_u = params['forms']['basis_u']

        if self.init_type == 'modes_k':
            self.target   =   params['mhd_init']['params_modes_k']['target']
            self.modes_k  = [ params['mhd_init']['params_modes_k']['kx'], 
                               params['mhd_init']['params_modes_k']['ky'], 
                               params['mhd_init']['params_modes_k']['kz'] ]
            self.amp      =   params['mhd_init']['params_modes_k']['amp']

            assert np.all(len(self.amp) == len(self.modes_k[0]) == len(self.modes_k[1]) == len(self.modes_k[2]))

        elif self.init_type == 'modes_mn':
            self.target   =   params['mhd_init']['params_modes_mn']['target']
            self.modes_mn = [ params['mhd_init']['params_modes_mn']['modes_m'],
                               params['mhd_init']['params_modes_mn']['modes_n'] ]
            self.amp      =   params['mhd_init']['params_modes_mn']['amp']
            
        elif self.init_type == 'eigenfun':
            self.n_tor    = params['mhd_init']['params_eigfun']['n_tor']
            self.profiles = params['mhd_init']['params_eigfun']['profiles']
            self.eig_kind = params['mhd_init']['params_eigfun']['eig_kind']
            self.eig_freq = params['mhd_init']['params_eigfun']['eig_freq']

        elif self.init_type == 'noise':
            self.target   =   params['mhd_init']['params_noise']['target']
            self.plane    =   params['mhd_init']['params_noise']['plane']


    # =================================================
    def initialize(self, r3, pp, b2, up):

        if self.init_type == 'modes_k':
            r3[:] = self.SPACES.projectors.pi_3(self.r3_ini)
            b2[:] = self.SPACES.projectors.pi_2([self.b2_ini_1, self.b2_ini_2, self.b2_ini_3])

            if   self.basis_p == 0:
                pp[:] = self.SPACES.projectors.pi_3(self.r3_ini)

            elif self.basis_p == 3:
                pp[:] = self.SPACES.projectors.pi_3(self.r3_ini)

            if   self.basis_u == 0:
                up_1  = self.SPACES.projectors.pi_0(self.uv_ini_1)
                up_2  = self.SPACES.projectors.pi_0(self.uv_ini_2)
                up_3  = self.SPACES.projectors.pi_0(self.uv_ini_3)

                up[:] = np.concatenate((up_1, up_2, up_3))

            elif self.basis_u == 1:
                up[:]     = self.SPACES.projectors.pi_1([self.u1_ini_1, self.u1_ini_2, self.u1_ini_3])

            elif self.basis_u == 2:
                up[:]     = self.SPACES.projectors.pi_2([self.u1_ini_1, self.u1_ini_2, self.u1_ini_3])


        elif self.init_type == 'noise':
            np.random.seed(1607)

            b2_temp_1 = np.empty(self.SPACES.Nbase_2form[0], dtype=float)
            b2_temp_2 = np.empty(self.SPACES.Nbase_2form[1], dtype=float)
            b2_temp_3 = np.empty(self.SPACES.Nbase_2form[2], dtype=float)
            r3_temp = np.empty(self.SPACES.Nbase_3form   , dtype=float)

            if   self.basis_p == 0:
                pp_temp = np.empty(self.SPACES.Nbase_0form, dtype=float)

            elif self.basis_p == 3:
                pp_temp = np.empty(self.SPACES.Nbase_3form, dtype=float)

            if   self.basis_u == 0:
                up_temp_1 = np.empty(self.SPACES.Nbase_0form, dtype=float)
                up_temp_2 = np.empty(self.SPACES.Nbase_0form, dtype=float)
                up_temp_3 = np.empty(self.SPACES.Nbase_0form, dtype=float)

            elif self.basis_u == 1:
                up_temp_1 = np.empty(self.SPACES.Nbase_1form[0], dtype=float)
                up_temp_2 = np.empty(self.SPACES.Nbase_1form[1], dtype=float)
                up_temp_3 = np.empty(self.SPACES.Nbase_1form[2], dtype=float)

            elif self.basis_u == 2:
                up_temp_1 = np.empty(self.SPACES.Nbase_2form[0], dtype=float)
                up_temp_2 = np.empty(self.SPACES.Nbase_2form[1], dtype=float)
                up_temp_3 = np.empty(self.SPACES.Nbase_2form[2], dtype=float)

            if self.plane == 'xy':
                amps = np.random.rand(8, self.SPACES.NbaseN[0], self.SPACES.NbaseN[1])
                for k in range(self.SPACES.NbaseN[2]):
                    pp_temp[:, :, k]   = amps[0]
                    up_temp_1[:, :, k] = amps[1]
                    up_temp_2[:, :, k] = amps[2]
                    up_temp_3[:, :, k] = amps[3]
                    b2_temp_1[:, :, k] = amps[4]
                    b2_temp_2[:, :, k] = amps[5]
                    b2_temp_3[:, :, k] = amps[6]
                    r3_temp[:, :, k]   = amps[7]

            elif self.plane == 'yz':
                amps = np.random.rand(8, self.SPACES.NbaseN[1], self.SPACES.NbaseN[2])
                for k in range(self.SPACES.NbaseN[0]):
                    pp_temp[k, :, :]   = amps[0]
                    up_temp_1[k, :, :] = amps[1]
                    up_temp_2[k, :, :] = amps[2]
                    up_temp_3[k, :, :] = amps[3]
                    b2_temp_1[k, :, :] = amps[4]
                    b2_temp_2[k, :, :] = amps[5]
                    b2_temp_3[k, :, :] = amps[6]
                    r3_temp[k, :, :]   = amps[7]

            elif self.plane == 'xz':
                amps = np.random.rand(8, self.SPACES.NbaseN[0], self.SPACES.NbaseN[2])
                for k in range(self.SPACES.NbaseN[1]):
                    pp_temp[:, k, :]   = amps[0]
                    up_temp_1[:, k, :] = amps[1]
                    up_temp_2[:, k, :] = amps[2]
                    up_temp_3[:, k, :] = amps[3]
                    b2_temp_1[:, k, :] = amps[4]
                    b2_temp_2[:, k, :] = amps[5]
                    b2_temp_3[:, k, :] = amps[6]
                    r3_temp[:, k, :]   = amps[7]

            if not 'p' in self.target:  pp_temp[:, :, :] = 0.
            if not 'r' in self.target:  r3_temp[:, :, :] = 0.
            if not 'u1' in self.target: up_temp_1[:, :, :] = 0.
            if not 'u2' in self.target: up_temp_2[:, :, :] = 0.
            if not 'u3' in self.target: up_temp_3[:, :, :] = 0.
            if not 'b1' in self.target: up_temp_1[:, :, :] = 0.
            if not 'b2' in self.target: up_temp_2[:, :, :] = 0.
            if not 'b3' in self.target: up_temp_3[:, :, :] = 0.

            pp[:] = pp_temp.flatten()
            r3[:] = r3_temp.flatten()
            b2[:] = np.concatenate((b2_temp_1.flatten(), b2_temp_2.flatten(), b2_temp_3.flatten()))
            up[:] = np.concatenate((up_temp_1.flatten(), up_temp_2.flatten(), up_temp_3.flatten()))

        elif self.init_type == 'modes_mn' or 'eigfun':
            print('modes_mn and eigfun mode are not implemented yet')

            
    # ===============================================================
    #                     functions for modes_k
    # ===============================================================
    # initial bulk pressure
    def fun_p(self, x, y, z):

        p = 0*x

        if 'p' in self.target :
            if self.init_type == 'modes_k' :
                for i in range(len(self.amp)):
                    p += self.amp[i]*np.sin(self.modes_k[0][i]*x + self.modes_k[1][i]*y + self.modes_k[2][i]*z)

        return p
    
    # initial velocity (x - component)
    def fun_u_x(self, x, y, z):

        u1 = 0*x

        if 'u1' in self.target :
            if self.init_type == 'modes_k' :
                for i in range(len(self.amp)):
                    u1 += self.amp[i]*np.sin(self.modes_k[0][i]*x + self.modes_k[1][i]*y + self.modes_k[2][i]*z)


        return u1

    # initial velocity (y - component)
    def fun_u_y(self, x, y, z):

        u2 = 0*x

        if 'u2' in self.target :
            if self.init_type == 'modes_k' :
                for i in range(len(self.amp)):
                    u2 += self.amp[i]*np.sin(self.modes_k[0][i]*x + self.modes_k[1][i]*y + self.modes_k[2][i]*z)

        return u2

    # initial velocity (z - component)
    def fun_u_z(self, x, y, z):

        u3 = 0*x

        if 'u3' in self.target :
            if self.init_type == 'modes_k' :
                for i in range(len(self.amp)):
                    u3 += self.amp[i]*np.sin(self.modes_k[0][i]*x + self.modes_k[1][i]*y + self.modes_k[2][i]*z)

        return u3
    
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

    # initial bulk density
    def fun_r(self, x, y, z):
        
        r = 0*x

        if 'r' in self.target :
            if self.init_type == 'modes_k' :
                for i in range(len(self.amp)):
                    r += self.amp[i]*np.sin(self.modes_k[0][i]*x + self.modes_k[1][i]*y + self.modes_k[2][i]*z)

        return r


    # ===============================================================
    #               pullback or transform to the form
    # ===============================================================

    # initial bulk pressure (0-form on logical domain)
    def p0_ini(self, eta1, eta2, eta3=None):

        if self.init_coords == 'physical':
            return self.DOMAIN.pull(self.fun_p, eta1, eta2, eta3, '0_form')
        
        elif self.init_coords == 'norm_logical':
            return self.DOMAIN.transformation(self.fun_p, eta1, eta2, eta3, 'norm_to_0')

    # initial bulk density (0-form on logical domain)
    def r0_ini(self, eta1, eta2, eta3=None):
        
        if self.init_coords == 'physical':
            return self.DOMAIN.pull(self.fun_r, eta1, eta2, eta3, '0_form')
        
        elif self.init_coords == 'norm_logical':
            return self.DOMAIN.transformation(self.fun_r, eta1, eta2, eta3, 'norm_to_0')

    # initial bulk pressure (3-form on logical domain)
    def p3_ini(self, eta1, eta2, eta3=None):

        if self.init_coords == 'physical':
            return self.DOMAIN.pull(self.fun_p, eta1, eta2, eta3, '3_form')
        
        elif self.init_coords == 'norm_logical':
            return self.DOMAIN.transformation(self.fun_p, eta1, eta2, eta3, 'norm_to_3')

    # initial bulk density (3-form on logical domain)
    def r3_ini(self, eta1, eta2, eta3=None):
        
        if self.init_coords == 'physical':
            return self.DOMAIN.pull(self.fun_r, eta1, eta2, eta3, '3_form')
        
        elif self.init_coords == 'norm_logical':
            return self.DOMAIN.transformation(self.fun_r, eta1, eta2, eta3, 'norm_to_3')


    # initial flow velocity (1-form on logical domain, 1-component)
    def u1_ini_1(self, eta1, eta2, eta3=None):
        
        if self.init_coords == 'physical':
            return self.DOMAIN.pull([self.fun_u_x, self.fun_u_y, self.fun_u_z], eta1, eta2, eta3, '1_form_1')
        elif self.init_coords == 'norm_logical':
            return self.DOMAIN.transformation([self.fun_u_x, self.fun_u_y, self.fun_u_z], eta1, eta2, eta3, 'norm_to_1_1')

        
    # initial flow velocity (1-form on logical domain, 2-component)
    def u1_ini_2(self, eta1, eta2, eta3=None):
        
        if self.init_coords == 'physical':
            return self.DOMAIN.pull([self.fun_u_x, self.fun_u_y, self.fun_u_z], eta1, eta2, eta3, '1_form_2')
        elif self.init_coords == 'norm_logical':
            return self.DOMAIN.transformation([self.fun_u_x, self.fun_u_y, self.fun_u_z], eta1, eta2, eta3, 'norm_to_1_2')

        
    # initial flow velocity (1-form on logical domain, 3-component)
    def u1_ini_3(self, eta1, eta2, eta3=None):
        
        if self.init_coords == 'physical':
            return self.DOMAIN.pull([self.fun_u_x, self.fun_u_y, self.fun_u_z], eta1, eta2, eta3, '1_form_3')
        elif self.init_coords == 'norm_logical':
            return self.DOMAIN.transformation([self.fun_u_x, self.fun_u_y, self.fun_u_z], eta1, eta2, eta3, 'norm_to_1_3')


    # initial flow velocity (2-form on logical domain, 1-component)
    def u2_ini_1(self, eta1, eta2, eta3=None):
        
        if self.init_coords == 'physical':
            return self.DOMAIN.pull([self.fun_u_x, self.fun_u_y, self.fun_u_z], eta1, eta2, eta3, '2_form_1')
        elif self.init_coords == 'norm_logical':
            return self.DOMAIN.transformation([self.fun_u_x, self.fun_u_y, self.fun_u_z], eta1, eta2, eta3, 'norm_to_2_1')

        
    # initial flow velocity (2-form on logical domain, 2-component)
    def u2_ini_2(self, eta1, eta2, eta3=None):
        
        if self.init_coords == 'physical':
            return self.DOMAIN.pull([self.fun_u_x, self.fun_u_y, self.fun_u_z], eta1, eta2, eta3, '2_form_2')
        elif self.init_coords == 'norm_logical':
            return self.DOMAIN.transformation([self.fun_u_x, self.fun_u_y, self.fun_u_z], eta1, eta2, eta3, 'norm_to_2_2')

        
    # initial flow velocity (2-form on logical domain, 3-component)
    def u2_ini_3(self, eta1, eta2, eta3=None):
        
        if self.init_coords == 'physical':
            return self.DOMAIN.pull([self.fun_u_x, self.fun_u_y, self.fun_u_z], eta1, eta2, eta3, '2_form_3')
        elif self.init_coords == 'norm_logical':
            return self.DOMAIN.transformation([self.fun_u_x, self.fun_u_y, self.fun_u_z], eta1, eta2, eta3, 'norm_to_2_3')

    
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
            
    
    # initial flow velocity (vector on logical domain, 1-component)
    def uv_ini_1(self, eta1, eta2, eta3=None):

        if self.init_coords == 'physical':
            return self.DOMAIN.pull([self.fun_u_x, self.fun_u_y, self.fun_u_z], eta1, eta2, eta3, 'vector_1')
        elif self.init_coords == 'norm_logical':
            return self.DOMAIN.transformation([self.fun_u_x, self.fun_u_y, self.fun_u_z], eta1, eta2, eta3, 'norm_to_vector_1')

        
    # initial flow velocity (vector on logical domain, 2-component)
    def uv_ini_2(self, eta1, eta2, eta3=None):

        if self.init_coords == 'physical':
            return self.DOMAIN.pull([self.fun_u_x, self.fun_u_y, self.fun_u_z], eta1, eta2, eta3, 'vector_2')
        elif self.init_coords == 'norm_logical':
            return self.DOMAIN.transformation([self.fun_u_x, self.fun_u_y, self.fun_u_z], eta1, eta2, eta3, 'norm_to_vector_2')

        
    # initial flow velocity (vector on logical domain, 3-component)
    def uv_ini_3(self, eta1, eta2, eta3=None):
        
        if self.init_coords == 'physical':
            return self.DOMAIN.pull([self.fun_u_x, self.fun_u_y, self.fun_u_z], eta1, eta2, eta3, 'vector_3')
        elif self.init_coords == 'norm_logical':
            return self.DOMAIN.transformation([self.fun_u_x, self.fun_u_y, self.fun_u_z], eta1, eta2, eta3, 'norm_to_vector_3')
