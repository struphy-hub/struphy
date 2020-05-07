import numpy as np
import hylife.geometry.mappings_analytical as mapping


class parameters():
    
    def __init__(self):
        
        self.Nel          = [20, 20, 2]             # mesh generation on logical domain
        self.bc           = [True, True, True]     # boundary conditions (True: periodic, False: else)
        self.p            = [2, 2, 1]              # spline degrees  
        
        self.nq_el        = [6, 6, 6] # number of quadrature points per element for integrations over whole domain
        self.nq_pr        = [6, 6, 6] # number of quadrature points per integration interval of projectors

        self.time_int     = True     # do time integration?
        self.dt           = 0.05     # time step
        self.Tend         = 5.0      # simulation time
        self.max_time     = 60*60    # maximum runtime of program in minutes
        self.add_pressure = False    # add non-Hamiltonian terms to simulation?

        # geometry
        self.kind_map     = 3                         # 1 : slab, 2 : hollow cylinder, 3 : colella
        
        #self.params_map   = [2*np.pi/0.75, 2*np.pi, 1.]     # parameters for mapping  
        self.params_map   = [2*np.pi/0.75, 2*np.pi, 0.05, 1.]     # parameters for mapping  
        
        # physical constants
        self.gamma        = 5/3                        # adiabatic exponent

        # particle parameters
        self.add_PIC      = True                # add kinetic terms to simulation?
        self.Np           = 128000              # total number of particles
        self.control      = True                # control variate for noise resuction? (delta-f method)

        self.v0x = 2.5                          # shift of Maxwellian in vx-direction
        self.v0y = 0.                           # shift of Maxwellian in vy-direction
        self.v0z = 0.                           # shift of Maxwellian in vz-direction
        
        self.vth = 1.                           # hot ion thermal velocity

        
        # particle loading
        """
        1. pseudo-random: particles[:, :6] = np.random.rand(Np, 6)
            particles logical coordinates and physical velocities are drawn randomly.
        
        2. sobol_standard: particles[:, :6] = sobol.i4_sobol_generate(6, Np, 1000)
            particles logical coordinates and physical velocities are drawn from a Sobol sequence, where the first 1000 numbers             are skipped.
        
        3. sobol_antithetic: sobol.i4_sobol_generate(6, int(Np/64), 1000) --> 64 symmetric particles
            particles logical coordinates and physical velocities are drawn from a Sobol sequence, where the first 1000 numbers             are skipped. Additionally, for every particle which is drawn, the 63 next particles are drawn mirrored in all other             coordinates: e.g. (1 - xi1, xi2, xi3, vx, vy, vz), (xi1, 1 - xi2, xi3, vx, vy, vz), ...
        
        4. pr_space_uni_velocity: pseudo-random in space, uniform in velocity space
        
        5. external: particles[:, :6] = np.load('name_of_file.npy') 
        """
        
        self.loading    = 'sobol_antithetic'


        # Is this run a restart? If yes, select restart data with num_restart
        self.restart = False 
        self.num_restart = 0

        # Create restart files at the end of the simulation?
        self.create_restart = False
        
        # ======================== initial conditions ===========================================
        self.ic_from_params = True
        
        
        # some parameters
        self.nmodes = 64
        self.modes  = np.linspace(0, self.nmodes, self.nmodes + 1) - self.nmodes/2
        self.modes  = np.delete(self.modes, int(self.nmodes/2))
        
    
    # ========== physical space ================
    # initial bulk pressure
    def p_ini_phys(self, x, y, z):

        p_phys = 0.

        return p_phys

    # initial bulk velocity (x - component)
    def ux_ini(self, x, y, z):

        ux = 0.

        return ux

    # initial bulk velocity (y - component)
    def uy_ini(self, x, y, z):
        
        kx = 0.75
        ky = 1.

        uy = np.cos(kx * x + ky * y)

        return uy

    # initial bulk velocity (z - component)
    def uz_ini(self, x, y, z):

        uz = 0.

        return uz

    # initial magnetic field (x - component)
    def bx_ini(self, x, y, z):

        bx = 0.

        return bx

    # initial magnetic field (y - component)
    def by_ini(self, x, y, z):

        amp = 1e-4

        kx  = 0.75
        ky  = 1.
        kz  = 0.
        
        by  = amp * np.cos(kx * x + ky * y)*0

        return by

    # initial magnetic field (z - component)
    def bz_ini(self, x, y, z):

        amp = 1e-4

        kx  = 0.75
        ky  = 1.
        kz  = 0.

        bz  = amp * np.sin(kx * x + ky * y)

        return bz

    # initial bulk density
    def rho_ini_phys(self, x, y, z):

        rho_phys = 0.

        return rho_phys
    
    
    # ========== logical space ================ 
    def p_ini(self, xi1, xi2, xi3):
    
        values = np.zeros(xi1.shape, dtype=float)

        for i1 in range(xi1.shape[0]):
            for i2 in range(xi2.shape[1]):
                for i3 in range(xi3.shape[2]):

                    i = i1, i2, i3

                    x = mapping.f(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 1)
                    y = mapping.f(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 2)
                    z = mapping.f(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 3)

                    values[i] = self.p_ini_phys(x, y, z)

        return values

    
    
    def u1_ini(self, xi1, xi2, xi3):
    
        values = np.zeros(xi1.shape, dtype=float)

        for i1 in range(xi1.shape[0]):
            for i2 in range(xi2.shape[1]):
                for i3 in range(xi3.shape[2]):

                    i = i1, i2, i3

                    x = mapping.f(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 1)
                    y = mapping.f(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 2)
                    z = mapping.f(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 3)

                    df_11 = mapping.df(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 11)
                    df_21 = mapping.df(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 21)
                    df_31 = mapping.df(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 31)

                    ux = self.ux_ini(x, y, z)
                    uy = self.uy_ini(x, y, z)
                    uz = self.uz_ini(x, y, z)

                    values[i] = df_11 * ux + df_21 * uy + df_31 * uz

        return values


    def u2_ini(self, xi1, xi2, xi3):
    
        values = np.zeros(xi1.shape, dtype=float)

        for i1 in range(xi1.shape[0]):
            for i2 in range(xi2.shape[1]):
                for i3 in range(xi3.shape[2]):

                    i = i1, i2, i3

                    x = mapping.f(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 1)
                    y = mapping.f(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 2)
                    z = mapping.f(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 3)

                    df_12 = mapping.df(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 12)
                    df_22 = mapping.df(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 22)
                    df_32 = mapping.df(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 32)

                    ux = self.ux_ini(x, y, z)
                    uy = self.uy_ini(x, y, z)
                    uz = self.uz_ini(x, y, z)

                    values[i] = df_12 * ux + df_22 * uy + df_32 * uz

        return values


    def u3_ini(self, xi1, xi2, xi3):

        values = np.zeros(xi1.shape, dtype=float)

        for i1 in range(xi1.shape[0]):
            for i2 in range(xi2.shape[1]):
                for i3 in range(xi3.shape[2]):

                    i = i1, i2, i3

                    x = mapping.f(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 1)
                    y = mapping.f(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 2)
                    z = mapping.f(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 3)

                    df_13 = mapping.df(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 13)
                    df_23 = mapping.df(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 23)
                    df_33 = mapping.df(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 33)

                    ux = self.ux_ini(x, y, z)
                    uy = self.uy_ini(x, y, z)
                    uz = self.uz_ini(x, y, z)

                    values[i] = df_13 * ux + df_23 * uy + df_33 * uz

        return values
    
    
    def b1_ini(self, xi1, xi2, xi3):
    
        values = np.zeros(xi1.shape, dtype=float)

        for i1 in range(xi1.shape[0]):
            for i2 in range(xi2.shape[1]):
                for i3 in range(xi3.shape[2]):

                    i = i1, i2, i3

                    x = mapping.f(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 1)
                    y = mapping.f(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 2)
                    z = mapping.f(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 3)

                    dfinv_11 = mapping.df_inv(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 11)
                    dfinv_12 = mapping.df_inv(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 12)
                    dfinv_13 = mapping.df_inv(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 13)

                    det_df   = mapping.det_df(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map)

                    bx = self.bx_ini(x, y, z)
                    by = self.by_ini(x, y, z)
                    bz = self.bz_ini(x, y, z)

                    values[i] = (dfinv_11 * bx + dfinv_12 * by + dfinv_13 * bz) * det_df

        return values


    def b2_ini(self, xi1, xi2, xi3):
    
        values = np.zeros(xi1.shape, dtype=float)

        for i1 in range(xi1.shape[0]):
            for i2 in range(xi2.shape[1]):
                for i3 in range(xi3.shape[2]):

                    i = i1, i2, i3

                    x = mapping.f(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 1)
                    y = mapping.f(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 2)
                    z = mapping.f(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 3)

                    dfinv_21 = mapping.df_inv(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 21)
                    dfinv_22 = mapping.df_inv(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 22)
                    dfinv_23 = mapping.df_inv(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 23)

                    det_df   = mapping.det_df(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map)

                    bx = self.bx_ini(x, y, z)
                    by = self.by_ini(x, y, z)
                    bz = self.bz_ini(x, y, z)

                    values[i] = (dfinv_21 * bx + dfinv_22 * by + dfinv_23 * bz) * det_df

        return values


    def b3_ini(self, xi1, xi2, xi3):

        values = np.zeros(xi1.shape, dtype=float)

        for i1 in range(xi1.shape[0]):
            for i2 in range(xi2.shape[1]):
                for i3 in range(xi3.shape[2]):

                    i = i1, i2, i3

                    x = mapping.f(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 1)
                    y = mapping.f(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 2)
                    z = mapping.f(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 3)

                    dfinv_31 = mapping.df_inv(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 31)
                    dfinv_32 = mapping.df_inv(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 32)
                    dfinv_33 = mapping.df_inv(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 33)

                    det_df   = mapping.det_df(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map)

                    bx = self.bx_ini(x, y, z)
                    by = self.by_ini(x, y, z)
                    bz = self.bz_ini(x, y, z)

                    values[i] = (dfinv_31 * bx + dfinv_32 * by + dfinv_33 * bz) * det_df

        return values
    
    
    def rho_ini(self, xi1, xi2, xi3):
    
        values = np.zeros(xi1.shape, dtype=float)

        for i1 in range(xi1.shape[0]):
            for i2 in range(xi2.shape[1]):
                for i3 in range(xi3.shape[2]):

                    i = i1, i2, i3

                    x = mapping.f(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 1)
                    y = mapping.f(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 2)
                    z = mapping.f(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map, 3)
                    
                    det_df = mapping.det_df(xi1[i], xi2[i], xi3[i], self.kind_map, self.params_map)

                    values[i] = self.rho_ini_phys(x, y, z) * det_df 

        return values