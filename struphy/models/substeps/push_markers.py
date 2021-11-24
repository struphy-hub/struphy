import numpy as np

from struphy.pic import pusher_pos
from struphy.pic import pusher_vel_3d

class Push:
    '''Split steps of pure particle pushing.
    
    Parameters
    ----------
        dts : list
            Time steps, one for each split step.
        
        DOMAIN : obj
            Domain object from geometry/domain_3d.

        SPACES : obj
            FEEC self.SPACES.

        Np_loc : int
            Number of particles per rank.

        params_kin : dict
            Parameters of kinetic_equilibrium/general.
    '''

    def __init__(self, dts, DOMAIN, SPACES, Np_loc, params_kin):

        self.dts     = dts
        self.DOMAIN  = DOMAIN
        self.SPACES  = SPACES
        self.Np_loc  = Np_loc
        self.params  = params_kin


    def step_eta_RK4(self, particles, print_info=False):
        '''RK4 method to update marker positions via d(eta)/dt = DF^(-1) v.
        
        Parameters
        ---------
            particles : np.array
                Shape (6, Np), where the rows hold the positions [:3] and velocities [3:].

            print_info : boolean
                Print to screen max difference of abs(input-output).
        '''

        # store initial values
        temp = particles.copy()

        pusher_pos.pusher_rk4(temp, self.dts[0], self.Np_loc,
                              self.DOMAIN.kind_map, self.DOMAIN.params_map, 
                              self.DOMAIN.T[0], self.DOMAIN.T[1], self.DOMAIN.T[2],
                              self.DOMAIN.p, self.DOMAIN.Nel, self.DOMAIN.NbaseN,
                              self.DOMAIN.cx, self.DOMAIN.cy, self.DOMAIN.cz
                              )

        # update global variable
        particles[:, :] = temp

        if print_info: 
            print('Maxdiff eta for step_eta_RK4:', np.max(np.abs(particles - temp)))
            print()

    
    def step_v_cyclotron_ana(self, particles, b2_eq, b2, print_info=False):
        '''Analytical update d(v)/dt = v x (B0 + dB).
        
        Parameters
        ---------
            particles : np.array
                Shape (6, Np), where the rows hold the positions [:3] and velocities [3:].

            b2_eq : np.array
                FE coefficients (flattened) of the equilibirum magnetic field.

            b2 : np.array
                FE coefficients (flattened) of the perturbed magnetic field.

            print_info : boolean
                Print to screen max difference of abs(input-output).
        '''

        # store initial values
        temp = particles.copy()

        b2_ten_1, b2_ten_2, b2_ten_3 = self.SPACES.extract_2(b2 + b2_eq)
        
        pusher_vel_3d.pusher_vxb(temp, 
                                 self.dts[1] * self.params['alpha'] * self.params['particle_charge'] / self.params['particle_mass'], 
                                 self.SPACES.T[0], self.SPACES.T[1], self.SPACES.T[2], 
                                 self.SPACES.p, self.SPACES.Nel, self.SPACES.NbaseN, self.SPACES.NbaseD, self.Np_loc, 
                                 b2_ten_1, b2_ten_2, b2_ten_3, 
                                 self.DOMAIN.kind_map, self.DOMAIN.params_map, 
                                 self.DOMAIN.T[0], self.DOMAIN.T[1], self.DOMAIN.T[2], 
                                 self.DOMAIN.p, self.DOMAIN.Nel, self.DOMAIN.NbaseN, 
                                 self.DOMAIN.cx, self.DOMAIN.cy, self.DOMAIN.cz
                                 )

        # update global variable
        particles[:, :] = temp

        if print_info: 
            print('Maxdiff v for step_v_cyclotron_ana:', np.max(np.abs(particles - temp)))
            print()
        
