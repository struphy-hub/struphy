import numpy as np
import os

class parameters():
    
    def __init__(self):
        
        self.Nel          = [64, 4, 4]            # mesh generation on logical domain
        self.bc           = [True, True, True]    # boundary conditions (True: periodic, False: else)
        self.p            = [3, 2, 2]             # spline degrees  

        self.time_int     = True     # do time integration?
        self.dt           = 0.05     # time step
        self.Tend         = 20.      # simulation time
        self.max_time     = 60*60    # maximum runtime of program in minutes
        self.add_pressure = True     # add non-Hamiltonian terms to simulation?

        # geometry
        self.kind_map     =  1                  # 1 : slab, 2 : hollow cylinder, 3 : colella
        self.params_map   = [20., 1., 1.]       # parameters for mapping  
        
        # physical constants
        self.gamma        = 5/3                 # adiabatic exponent

        # particle parameters
        self.add_PIC      = False           # add kinetic terms to simulation?
        self.Np           = 10              # total number of particles
        self.control      = False           # control variate? (0: no, 1: yes)

        self.v0x = 2.5                      # shift of Maxwellian in vx-direction
        self.v0y = 0.                       # shift of Maxwellian in vy-direction
        self.v0z = 0.                       # shift of Maxwellian in vz-direction
        
        self.vth = 1.                       # hot ion thermal velocity

        # particle loading
        '''
        pseudo-random: particles[:, :6] = np.random.rand(Np, 6)
        sobol_standard: particles[:, :6] = sobol.i4_sobol_generate(6, Np, 1000)
        sobol_antithetic: sobol.i4_sobol_generate(6, int(Np/64), 1000) --> 64 symmetric particles
        pr_space_uni_velocity: pseudo-random in space, uniform in velocity space
        external: particles[:, :6] = np.load('name_of_file.npy') 
        '''
        self.loading    = 'pseudo-random'


        # Is this run a restart?
        self.restart = False  

        # Create restart files at the end of the simulation? If True, name full directory where to save them
        self.create_restart = False
        
        
        
        # initial conditions 
        self.ic_from_params = True
        
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