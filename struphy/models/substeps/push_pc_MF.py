import scipy.sparse as spa
import numpy as np

from struphy.pic import pusher_vel_3d

class Pressure_coupling:
    '''The substeps of the Poisson splitting algorithm for pressure coupling terms.
    
    Parameters
    ----------
        dts : list
            Time steps, one for each split step.
            
        SPACES : obj
            FEEC spaces.
            
        MHD_OPS_MF : obj
            MHD operators from "struphy/feec/projectors/pro_global/mhd_operators_MF".

        ACCUM : obj
            Accumulation routines from "struphy/pic/accumulation".

        MPI_COMM : obj
            MPI communicator.

        Np : int
            Number of particles (all ranks).

        Np_loc : int
            Number of particles per rank.
            
        params_solver : dict
            Parameters for the linear "solvers" from parameters.yml.

        params_kin : dict
            Parameters of kinetic_equilibrium/general.

        basis_u : int
            Which FE basis is used for up (1 or 2).
    '''

    
    def __init__(self, dts, DOMAIN, SPACES, MHD_OPS_MF, ACCUM, MPI_COMM, Np, Np_loc, params_solver, params_kin, basis_u):

        self.dts        = dts
        self.DOMAIN     = DOMAIN
        self.SPACES     = SPACES
        self.MHD_OPS_MF = MHD_OPS_MF
        self.ACCUM      = ACCUM
        self.MPI_COMM   = MPI_COMM
        self.Np         = Np
        self.Np_loc     = Np_loc
        self.params_sol = params_solver
        self.params_kin = params_kin
        self.basis_u    = basis_u


    def step_ph_full(self, up, particles, print_info=False):
        '''Coupling term involving full hot pressure tensor, updates velocity (up) and marker velocity (v).
        
        Parameters
        ----------
            up : np.array
                FE coefficients, flattened.
                
            particles : np.array
                Shape (6, Np), where the rows hold the positions [:3] and velocities [3:].

            print_info : boolean
                Print to screen a) solver info b) number of iterations and c) max difference of abs(input-output).
        '''

        # store initial values 
        up_old = up.copy()
        temp   = particles.copy()

        # charge over mass coefficient
        charge_over_mass = self.params_kin['particle_charge'] / self.params_kin['particle_mass']

        # counter for number of interation steps in iterative solvers
        num_iters = 0
        def count_iters(xk):
            nonlocal num_iters
            num_iters += 1

        # pressure tensor accumulation (all processes) 
        self.ACCUM.accumulate_step_ph_full(particles, self.MPI_COMM)
        self.ACCUM.assemble_step_ph_full(self.Np, self.params_kin['nuh'], self.params_kin['alpha'], charge_over_mass)

        # RHS and LHS of linear system
        RHS = self.MHD_OPS_MF.A.matvec(up) - self.dts[0]**2/4*self.ACCUM.assemble_mat_X_step_ph_full(self.MHD_OPS_MF, up) + self.dts[0]*self.ACCUM.assemble_vec_X_step_ph_full(self.MHD_OPS_MF)
        LHS = spa.linalg.LinearOperator(self.MHD_OPS_MF.A_mat.shape, lambda x : self.MHD_OPS_MF.A.matvec(x) + self.dts[0]**2/4*self.ACCUM.assemble_mat_X_step_ph_full(self.MHD_OPS_MF, x))
        
        # solve linear system with gmres method and values from last time step as initial guess
        if   self.params_sol['solver_type_3'] == 'gmres':
            up[:], info = spa.linalg.gmres(LHS, RHS, x0=up, tol=self.params_sol['tol3'], 
                                                         maxiter=self.params_sol['maxiter3'], M=self.MHD_OPS_MF.A_inv, callback=count_iters
                                                         )
        elif self.params_sol['solver_type_3'] == 'cg':
            up[:], info = spa.linalg.cg(   LHS, RHS, x0=up, tol=self.params_sol['tol3'], 
                                                         maxiter=self.params_sol['maxiter3'], M=self.MHD_OPS_MF.A_inv, callback=count_iters
                                                         )
        elif self.params_sol['solver_type_3'] == 'cgs':
            up[:], info = spa.linalg.cgs(  LHS, RHS, x0=up, tol=self.params_sol['tol3'], 
                                                         maxiter=self.params_sol['maxiter3'], M=self.MHD_OPS_MF.A_inv, callback=count_iters
                                                         )
        
        # update velocities 
        if self.basis_u == 1:
            X_dot_u = self.MHD_OPS_MF.X1_dot((up + up_old))

            up_ten_11, up_ten_12, up_ten_13 = self.SPACES.extract_1(self.SPACES.G.dot(X_dot_u[0]))
            up_ten_21, up_ten_22, up_ten_23 = self.SPACES.extract_1(self.SPACES.G.dot(X_dot_u[1]))
            up_ten_31, up_ten_32, up_ten_33 = self.SPACES.extract_1(self.SPACES.G.dot(X_dot_u[2]))

        elif self.basis_u == 2:
            X_dot_u = self.MHD_OPS_MF.X2_dot((up + up_old))
    
            up_ten_11, up_ten_12, up_ten_13 = self.SPACES.extract_2(self.SPACES.G.dot(X_dot_u[0]))
            up_ten_21, up_ten_22, up_ten_23 = self.SPACES.extract_2(self.SPACES.G.dot(X_dot_u[1]))
            up_ten_31, up_ten_32, up_ten_33 = self.SPACES.extract_2(self.SPACES.G.dot(X_dot_u[2]))
        
        pusher_vel_3d.pusher_v_pressure_full(temp, charge_over_mass*self.dts[0], 
                                self.SPACES.T[0], self.SPACES.T[1], self.SPACES.T[2], 
                                self.SPACES.p, self.SPACES.Nel, self.SPACES.NbaseN, self.SPACES.NbaseD, 
                                self.Np_loc, 
                                up_ten_11, up_ten_12, up_ten_13,
                                up_ten_21, up_ten_22, up_ten_23,
                                up_ten_31, up_ten_32, up_ten_33, 
                                self.basis_u, 
                                self.DOMAIN.kind_map, self.DOMAIN.params_map, 
                                self.DOMAIN.T[0], self.DOMAIN.T[1], self.DOMAIN.T[2], 
                                self.DOMAIN.p, self.DOMAIN.Nel, self.DOMAIN.NbaseN, 
                                self.DOMAIN.cx, self.DOMAIN.cy, self.DOMAIN.cz
                                )

        if print_info: 
            print('Status     for step_ph:', info)
            print('Iterations for step_ph:', num_iters)
            print('Maxdiff up for step_ph:', np.max(np.abs(up - up_old)))
            print('Maxdiff  v for step_ph:', np.max(np.abs(particles - temp)))
            print()

        # update global variable
        particles[:, :] = temp


    def step_ph_perp(self, up, particles, print_info=False):
        '''Coupling term involving hot pressure tensor perpendicular to equilibrium magnetic field, updates velocity (up) and marker velocity (v).
        
        Parameters
        ----------
            up : np.array
                FE coefficients, flattened.
                
            particles : np.array
                Shape (6, Np), where the rows hold the positions [:3] and velocities [3:].

            print_info : boolean
                Print to screen a) solver info b) number of iterations and c) max difference of abs(input-output).
        '''

        # store initial values 
        up_old = up.copy()
        temp   = particles.copy()

        # charge over mass coefficient
        charge_over_mass = self.params_kin['particle_charge'] / self.params_kin['particle_mass']

        # counter for number of interation steps in iterative solvers
        num_iters = 0
        def count_iters(xk):
            nonlocal num_iters
            num_iters += 1

        # pressure tensor accumulation (all processes) 
        self.ACCUM.accumulate_step_ph_perp(particles, self.MPI_COMM)
        self.ACCUM.assemble_step_ph_perp(self.Np, self.params_kin['nuh'], self.params_kin['alpha'], charge_over_mass)

        # RHS and LHS of linear system
        RHS = self.MHD_OPS_MF.A.matvec(up) - self.dts[0]**2/4*self.ACCUM.assemble_mat_X_step_ph_perp(self.MHD_OPS_MF, up) + self.dts[0]*self.ACCUM.assemble_vec_X_step_ph_perp(self.MHD_OPS_MF)
        LHS = spa.linalg.LinearOperator(self.MHD_OPS_MF.A_mat.shape, lambda x : self.MHD_OPS_MF.A.matvec(x) + self.dts[0]**2/4*self.ACCUM.assemble_mat_X_step_ph_perp(self.MHD_OPS_MF, x))
        
        # solve linear system with gmres method and values from last time step as initial guess
        if   self.params_sol['solver_type_3'] == 'gmres':
            up[:], info = spa.linalg.gmres(LHS, RHS, x0=up, tol=self.params_sol['tol3'], 
                                                         maxiter=self.params_sol['maxiter3'], M=self.MHD_OPS_MF.A_inv, callback=count_iters
                                                         )
        elif self.params_sol['solver_type_3'] == 'cg':
            up[:], info = spa.linalg.cg(   LHS, RHS, x0=up, tol=self.params_sol['tol3'], 
                                                         maxiter=self.params_sol['maxiter3'], M=self.MHD_OPS_MF.A_inv, callback=count_iters
                                                         )
        elif self.params_sol['solver_type_3'] == 'cgs':
            up[:], info = spa.linalg.cgs(  LHS, RHS, x0=up, tol=self.params_sol['tol3'], 
                                                         maxiter=self.params_sol['maxiter3'], M=self.MHD_OPS_MF.A_inv, callback=count_iters
                                                         )
        
        # update velocities 
        if self.basis_u == 1:
            X_dot_u = self.MHD_OPS_MF.X1_dot((up + up_old))

            up_ten_11, up_ten_12, up_ten_13 = self.SPACES.extract_1(self.SPACES.G.dot(X_dot_u[0]))
            up_ten_21, up_ten_22, up_ten_23 = self.SPACES.extract_1(self.SPACES.G.dot(X_dot_u[1]))
            up_ten_31, up_ten_32, up_ten_33 = self.SPACES.extract_1(self.SPACES.G.dot(X_dot_u[2]))

        elif self.basis_u == 2:
            X_dot_u = self.MHD_OPS_MF.X2_dot((up + up_old))
    
            up_ten_11, up_ten_12, up_ten_13 = self.SPACES.extract_2(self.SPACES.G.dot(X_dot_u[0]))
            up_ten_21, up_ten_22, up_ten_23 = self.SPACES.extract_2(self.SPACES.G.dot(X_dot_u[1]))
            up_ten_31, up_ten_32, up_ten_33 = self.SPACES.extract_2(self.SPACES.G.dot(X_dot_u[2]))
        
        pusher_vel_3d.pusher_v_pressure_perp(temp, charge_over_mass*self.dts[0], 
                                self.SPACES.T[0], self.SPACES.T[1], self.SPACES.T[2], 
                                self.SPACES.p, self.SPACES.Nel, self.SPACES.NbaseN, self.SPACES.NbaseD, 
                                self.Np_loc, 
                                up_ten_11, up_ten_12, up_ten_13,
                                up_ten_21, up_ten_22, up_ten_23,
                                up_ten_31, up_ten_32, up_ten_33, 
                                self.basis_u, 
                                self.DOMAIN.kind_map, self.DOMAIN.params_map, 
                                self.DOMAIN.T[0], self.DOMAIN.T[1], self.DOMAIN.T[2], 
                                self.DOMAIN.p, self.DOMAIN.Nel, self.DOMAIN.NbaseN, 
                                self.DOMAIN.cx, self.DOMAIN.cy, self.DOMAIN.cz
                                )

        if print_info: 
            print('Status     for step_ph:', info)
            print('Iterations for step_ph:', num_iters)
            print('Maxdiff up for step_ph:', np.max(np.abs(up - up_old)))
            print('Maxdiff  v for step_ph:', np.max(np.abs(particles - temp)))
            print()

        # update global variable
        particles[:, :] = temp