import numpy as np
import os

class parameters():
    
    def __init__(self):
        
        self.Nel          = [16, 2, 2]             # mesh generation on logical domain
        self.bc           = [True, True, True]     # boundary conditions (True: periodic, False: else)
        self.p            = [2, 1, 1]              # spline degrees  
        
        self.nq_el        = [3, 2, 2] # number of quadrature points per element for integrations over whole domain
        self.nq_pr        = [6, 6, 6] # number of quadrature points per integration interval of projectors

        self.time_int     = True     # do time integration?
        self.dt           = 0.05     # time step
        self.Tend         = 5.0      # simulation time
        self.max_time     = 60*60    # maximum runtime of program in minutes
        self.add_pressure = False    # add non-Hamiltonian terms to simulation?

        # geometry
        self.kind_map     =  1                         # 1 : slab, 2 : hollow cylinder, 3 : colella
        
        self.params_map   = [2*np.pi/0.75, 1., 1.]     # parameters for mapping  
        #self.params_map   = [2*np.pi/0.75, 1., 0.05, 1.]     # parameters for mapping  
        
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
        
        # initial conditions 
        self.ic_from_params = False
        
        self.nmodes = 64
        self.modes  = np.linspace(0, self.nmodes, self.nmodes + 1) - self.nmodes/2
        self.modes  = np.delete(self.modes, int(self.nmodes/2))
        
         
    def u1_ini(self, xi1, xi2, xi3):
    
        values = np.zeros(xi1.shape)

        for i in range(self.nmodes):
            values += np.random.rand()*np.sin(2*np.pi*self.modes[i]*xi1)

        return values

    def u2_ini(self, xi1, xi2, xi3):

        values = np.zeros(xi1.shape)

        for i in range(self.nmodes):
            values += np.random.rand()*np.sin(2*np.pi*self.modes[i]*xi1)

        return values

    def u3_ini(self, xi1, xi2, xi3):

        values = np.zeros(xi1.shape)

        for i in range(self.nmodes):
            values += np.random.rand()*np.sin(2*np.pi*self.modes[i]*xi1)

        return values

    def b1_ini(self, xi1, xi2, xi3):

        values = np.zeros(xi1.shape)

        for i in range(self.nmodes):
            values += 0*np.sin(2*np.pi*self.modes[i]*xi1)

        return values

    def b2_ini(self, xi1, xi2, xi3):

        values = np.zeros(xi1.shape)

        for i in range(self.nmodes):
            values += np.random.rand()*np.sin(2*np.pi*self.modes[i]*xi1)

        return values

    def b3_ini(self, xi1, xi2, xi3):

        values = np.zeros(xi1.shape)

        for i in range(self.nmodes):
            values += np.random.rand()*np.sin(2*np.pi*self.modes[i]*xi1)

        return values


    def rho_ini(self, xi1, xi2, xi3):

        values = np.zeros(xi1.shape)

        for i in range(self.nmodes):
            values += np.random.rand()*np.sin(2*np.pi*self.modes[i]*xi1)

        return values

    def p_ini(self, xi1, xi2, xi3):

        values = np.zeros(xi1.shape)

        for i in range(self.nmodes):
            values += np.random.rand()*np.sin(2*np.pi*self.modes[i]*xi1)

        return values