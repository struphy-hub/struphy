import numpy as np

from struphy.pic import pusher_pos
from struphy.pic import pusher_vel_3d
from struphy.pic import pusher_pos_vel_3d
from struphy.pic import pusher_weights


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

        self.dts = dts
        self.DOMAIN = DOMAIN
        self.SPACES = SPACES
        self.Np_loc = Np_loc
        self.params = params_kin

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

        if print_info:
            print('Maxdiff eta for step_eta_RK4:',
                  np.max(np.abs(particles - temp)))
            print()

        # update global variable
        particles[:, :] = temp

    def step_eta_RK4_pc_full(self, particles, up, basis_u, print_info=False):
        '''RK4 method to update marker positions via d(eta)/dt = DF^(-1) v + G^(-1)   up    1form
                                                                           + 1/g_sqrt up    2form

        Parameters
        ---------
            particles : np.array
                Shape (6, Np), where the rows hold the positions [:3] and velocities [3:].

            print_info : boolean
                Print to screen max difference of abs(input-output).
        '''

        # store initial values
        temp = particles.copy()

        if basis_u == 1:
            up_ten_1, up_ten_2, up_ten_3 = self.SPACES.extract_1(up)

        elif basis_u == 2:
            up_ten_1, up_ten_2, up_ten_3 = self.SPACES.extract_2(up)

        pusher_pos.pusher_rk4_pc_full(temp, self.dts[0],
                                      self.SPACES.T[0], self.SPACES.T[1], self.SPACES.T[2],
                                      self.SPACES.p, self.SPACES.Nel, self.SPACES.NbaseN, self.SPACES.NbaseD,
                                      self.Np_loc, up_ten_1, up_ten_2, up_ten_3, basis_u,
                                      self.DOMAIN.kind_map, self.DOMAIN.params_map,
                                      self.DOMAIN.T[0], self.DOMAIN.T[1], self.DOMAIN.T[2],
                                      self.DOMAIN.p, self.DOMAIN.Nel, self.DOMAIN.NbaseN,
                                      self.DOMAIN.cx, self.DOMAIN.cy, self.DOMAIN.cz
                                      )

        if print_info:
            print('Maxdiff eta for step_eta_RK4:',
                  np.max(np.abs(particles - temp)))
            print()

        # update global variable
        particles[:, :] = temp

    def step_eta_RK4_pc_perp(self, particles, up, basis_u, print_info=False):
        '''RK4 method to update marker positions via d(eta)/dt = DF^(-1) v + G^(-1)   up_perp   1form
                                                                           + 1/g_sqrt up_perp   2form

        Parameters
        ---------
            particles : np.array
                Shape (6, Np), where the rows hold the positions [:3] and velocities [3:].

            print_info : boolean
                Print to screen max difference of abs(input-output).
        '''

        # store initial values
        temp = particles.copy()

        if basis_u == 1:
            up_ten_1, up_ten_2, up_ten_3 = self.SPACES.extract_1(up)

        elif basis_u == 2:
            up_ten_1, up_ten_2, up_ten_3 = self.SPACES.extract_2(up)

        pusher_pos.pusher_rk4_pc_perp(temp, self.dts[0],
                                      self.SPACES.T[0], self.SPACES.T[1], self.SPACES.T[2],
                                      self.SPACES.p, self.SPACES.Nel, self.SPACES.NbaseN, self.SPACES.NbaseD,
                                      self.Np_loc, up_ten_1, up_ten_2, up_ten_3, basis_u,
                                      self.DOMAIN.kind_map, self.DOMAIN.params_map,
                                      self.DOMAIN.T[0], self.DOMAIN.T[1], self.DOMAIN.T[2],
                                      self.DOMAIN.p, self.DOMAIN.Nel, self.DOMAIN.NbaseN,
                                      self.DOMAIN.cx, self.DOMAIN.cy, self.DOMAIN.cz
                                      )

        if print_info:
            print('Maxdiff eta for step_eta_RK4:',
                  np.max(np.abs(particles - temp)))
            print()

        # update global variable
        particles[:, :] = temp

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
                                 self.dts[1] * self.params['alpha'] *
                                 self.params['particle_charge'] /
                                 self.params['particle_mass'],
                                 self.SPACES.T[0], self.SPACES.T[1], self.SPACES.T[2],
                                 self.SPACES.p, self.SPACES.Nel, self.SPACES.NbaseN, self.SPACES.NbaseD, self.Np_loc,
                                 b2_ten_1, b2_ten_2, b2_ten_3,
                                 self.DOMAIN.kind_map, self.DOMAIN.params_map,
                                 self.DOMAIN.T[0], self.DOMAIN.T[1], self.DOMAIN.T[2],
                                 self.DOMAIN.p, self.DOMAIN.Nel, self.DOMAIN.NbaseN,
                                 self.DOMAIN.cx, self.DOMAIN.cy, self.DOMAIN.cz
                                 )

        if print_info:
            print('Maxdiff v for step_v_cyclotron_ana:',
                  np.max(np.abs(particles - temp)))
            print()

        # update global variable
        particles[:, :] = temp

    def step_in_const_efield(self, particles, efield, accuracy, maxiter, print_info=False):
        """
        updates the system dx/dt = v ; dv/dt = q/m * e_0(x)

        Parameters : 
        ------------
            particles : array
                shape(6,) contains the positions [0:3,], the velocities [3:6,], and the weights [6,]

            efield : array
                contains the coefficient vector of the electric field

            accuracy : array
                sets the accuracy for the position (in [0]) and velocity (in [1]) with which the iterative scheme of the pusher_x_v_static_efield should work

            maxiter : integer
                sets the maximum number of iterations for the iterative scheme in pusher_x_v_static_efield
            
            print_info : Boolean
                if true then max step sizes for x and v will be displayed
        """
        from numpy import polynomial
        temp = particles.copy()

        # total number of basis functions : B-splines (pn) and D-splines(pd)
        pn1 = self.SPACES.p[0]
        pn2 = self.SPACES.p[1]
        pn3 = self.SPACES.p[2]

        pd1 = pn1 - 1
        pd2 = pn2 - 1
        pd3 = pn3 - 1

        # number of quadrature points in direction 1
        n_quad1 = int(pd1*pn2*pn3/2.) + 2
        # number of quadrature points in direction 2
        n_quad2 = int(pn1*pd2*pn3/2.) + 2
        # number of quadrature points in direction 3
        n_quad3 = int(pn1*pn2*pd3/2.) + 2

        # get quadrature weights and locations
        loc1, weight1 = polynomial.legendre.leggauss(n_quad1)
        loc2, weight2 = polynomial.legendre.leggauss(n_quad2)
        loc3, weight3 = polynomial.legendre.leggauss(n_quad3)

        pusher_pos_vel_3d.pusher_x_v_static_efield( temp,
                                                    self.dts[1],
                                                    self.SPACES.p,
                                                    self.SPACES.T[0],    self.SPACES.T[1],    self.SPACES.T[2],
                                                    self.SPACES.indN[0], self.SPACES.indN[1], self.SPACES.indN[2],
                                                    self.SPACES.indD[0], self.SPACES.indD[1], self.SPACES.indD[2],
                                                    loc1,    loc2,    loc3,
                                                    weight1, weight2, weight3,
                                                    self.Np_loc,
                                                    self.SPACES.NbaseN, self.SPACES.NbaseD,
                                                    efield,
                                                    accuracy,
                                                    maxiter
                                                    )

        if np.isnan(temp).any():
            print('position of nan', np.where(np.isnan(temp)))
            print()

        if print_info:
            print('max step size for positions  in step_in_const_efield: ', np.max(np.abs(particles[0:3, :] - temp[0:3, :])))
            print('max step size for velocities in step_in_const_efield: ', np.max(np.abs(particles[3:6, :] - temp[3:6, :])))
            print()

        # update particles
        particles[:, :] = temp[:, :]
