import numpy as np
import os

class parameters():
    
    def __init__(self):
        
        self.Nel          = [16, 4, 4]             # mesh generation on logical domain
        self.bc           = [True, True, True]     # boundary conditions (True: periodic, False: else)
        self.p            = [3, 2, 2]              # spline degrees  

        self.time_int     = True     # do time integration?
        self.dt           = 0.06     # time step
        self.Tend         = 201.     # simulation time
        self.max_time     = 60*60    # maximum runtime of program in minutes
        self.add_pressure = False    # add non-Hamiltonian terms to simulation?

        # geometry
        self.kind_map     =  1                         # 1 : slab, 2 : hollow cylinder, 3 : colella
        self.params_map   = [2*np.pi/0.75, 1., 1.]     # parameters for mapping  
        
        # physical constants
        self.gamma        = 5/3                       # adiabatic exponent

        # particle parameters
        self.Np           = 128000              # total number of particles
        self.control      = 1                   # control variate? (0: no, 1: yes)

        self.v0x = 2.5
        self.v0y = 0.
        self.v0z = 0.
        
        self.vth = 1.

        # particle loading
        '''
        pseudo-random: particles[:, :6] = np.random.rand(Np, 6)
        sobol_standard: particles[:, :6] = sobol.i4_sobol_generate(6, Np, 1000)
        sobol_antithetic: sobol.i4_sobol_generate(6, int(Np/64), 1000) --> 64 symmetric particles
        pr_space_uni_velocity: pseudo-random in space, uniform in velocity space
        external: particles[:, :6] = np.load('name_of_file.npy') 
        '''
        self.loading    = 'sobol_antithetic'


        # Is this run a restart?
        self.restart = False  

        # Create restart files at the end of the simulation? If True, name full directory where to save them
        self.create_restart = False