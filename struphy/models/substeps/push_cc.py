import scipy.sparse as spa
import numpy as np

from struphy.pic import pusher_vel_3d

class Current_coupling:
    '''The substeps of the Poisson splitting algorithm for current coupling terms.
    
    [Ref] F. Holderied, S. Possanner and X. Wang, "MHD-kinetic hybrid code based on structure-preserving finite elements with particles-in-cell",
            J. Comp. Phys. 433 (2021) 110143.
    
    Parameters
    ----------
        dts : list
            Time steps, one for each split step.
            
        SPACES : obj
            FEEC spaces.
            
        MHD_OPS : obj
            MHD operators from "struphy/feec/projectors/pro_global/mhd_operators_cc_lin_6d".

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
            Which FE basis is used for up (1, 2 or 0).
    '''

    
    def __init__(self, dts, DOMAIN, SPACES, MHD_OPS, ACCUM, MPI_COMM, Np, Np_loc, params_solver, params_kin, basis_u):

        self.dts        = dts
        self.DOMAIN     = DOMAIN
        self.SPACES     = SPACES
        self.MHD_OPS    = MHD_OPS
        self.ACCUM      = ACCUM
        self.MPI_COMM   = MPI_COMM
        self.Np         = Np
        self.Np_loc     = Np_loc
        self.params_sol = params_solver
        self.params_kin = params_kin
        self.basis_u    = basis_u


    def step_jh(self, up, particles, b2_eq, b2, print_info=False):
        '''Coupling term involving hot current density, updates velocity (up) and marker velocity (v).
        
        Parameters
        ----------
            up : np.array
                FE coefficients, flattened.
                
            particles : np.array
                Shape (6, Np), where the rows hold the positions [:3] and velocities [3:].

            b2_eq : np.array
                FE coefficients (flattened) of the equilibirum magnetic field.

            b2 : np.array
                FE coefficients (flattened) of the perturbed magnetic field.

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

        # current accumulation (all processes) 
        self.ACCUM.accumulate_step3(particles, b2, b2_eq, self.MPI_COMM)
        
        # build global sparse matrix and vector
        mat, vec = self.ACCUM.assemble_step3(self.Np, b2, b2_eq)
        mat     *= self.params_kin['nuh'] * self.params_kin['alpha'] * charge_over_mass
        vec     *= self.params_kin['nuh'] * self.params_kin['alpha'] * charge_over_mass
        
        # rhs of linear system
        rhs = self.MHD_OPS.A(up) - self.dts[0]**2/4 * mat.dot(up) + self.dts[0]*vec
        
        # LHS of linear system
        LHS = spa.linalg.LinearOperator(self.MHD_OPS.A.shape, lambda x : self.MHD_OPS.A(x) + self.dts[0]**2/4 * mat.dot(x))
        
        # solve linear system with gmres method and values from last time step as initial guess
        if   self.params_sol['solver_type_3'] == 'gmres':
            up[:], info = spa.linalg.gmres(LHS, rhs, x0=up, tol=self.params_sol['tol3'], 
                                                         maxiter=self.params_sol['maxiter3'], M=self.MHD_OPS.A_inv, callback=count_iters
                                                         )
        elif self.params_sol['solver_type_3'] == 'cg':
            up[:], info = spa.linalg.cg(   LHS, rhs, x0=up, tol=self.params_sol['tol3'], 
                                                         maxiter=self.params_sol['maxiter3'], M=self.MHD_OPS.A_inv, callback=count_iters
                                                         )
        elif self.params_sol['solver_type_3'] == 'cgs':
            up[:], info = spa.linalg.cgs(  LHS, rhs, x0=up, tol=self.params_sol['tol3'], 
                                                         maxiter=self.params_sol['maxiter3'], M=self.MHD_OPS.A_inv, callback=count_iters
                                                         )
        
        # update velocities 
        b2_ten_1, b2_ten_2, b2_ten_3 = self.SPACES.extract_2(b2 + b2_eq)
        
        if self.basis_u == 0:
            up_ten_1, up_ten_2, up_ten_3 = self.SPACES.extract_0((up + up_old)/2)
        else:
            up_ten_1, up_ten_2, up_ten_3 = self.SPACES.extract_2((up + up_old)/2)
        
        pusher_vel_3d.pusher_v_mhd_electric(temp, charge_over_mass*self.dts[0], 
                                self.SPACES.T[0], self.SPACES.T[1], self.SPACES.T[2], 
                                self.SPACES.p, self.SPACES.Nel, self.SPACES.NbaseN, self.SPACES.NbaseD, 
                                self.Np_loc, b2_ten_1, b2_ten_2, b2_ten_3, 
                                np.zeros(self.SPACES.Nbase_0form, dtype=float), up_ten_1, up_ten_2, up_ten_3, self.basis_u, 
                                self.DOMAIN.kind_map, self.DOMAIN.params_map, 
                                self.DOMAIN.T[0], self.DOMAIN.T[1], self.DOMAIN.T[2], 
                                self.DOMAIN.p, self.DOMAIN.Nel, self.DOMAIN.NbaseN, 
                                self.DOMAIN.cx, self.DOMAIN.cy, self.DOMAIN.cz, np.zeros(self.Np_loc, dtype=float)
                                )
        
        # update global variable
        particles[:, :] = temp

        if print_info: 
            print('Status     for step_jh:', info)
            print('Iterations for step_jh:', num_iters)
            print('Maxdiff up for step_jh:', np.max(np.abs(up - up_old)))
            print('Maxdiff  v for step_jh:', np.max(np.abs(particles - temp)))
            print()


    def step_rhoh(self, up, particles, b2_eq, b2, print_info=False):
        '''Coupling term involving hot charge density, updates velocity (up).
        
        Parameters
        ----------
            up : np.array
                FE coefficients, flattened.
                
            particles : np.array
                Shape (6, Np), where the rows hold the positions [:3] and velocities [3:].

            b2_eq : np.array
                FE coefficients (flattened) of the equilibirum magnetic field.

            b2 : np.array
                FE coefficients (flattened) of the perturbed magnetic field.

            print_info : boolean
                Print to screen a) solver info b) number of iterations and c) max difference of abs(input-output).
                
        Returns
        -------
            up : np.array
                FE coefficients, flattened.
                
            num_iters : int
                Number of iterations of linear solver.
        '''

        # store initial values 
        up_old = up.copy()

        # charge over mass coefficient
        charge_over_mass = self.params_kin['particle_charge'] / self.params_kin['particle_mass']

        # counter for number of interation steps in iterative solvers
        num_iters = 0
        def count_iters(xk):
            nonlocal num_iters
            num_iters += 1

        # charge accumulation (all processes) 
        self.ACCUM.accumulate_step1(particles, b2, b2_eq, self.MPI_COMM)
            
        # build global sparse matrix 
        mat  = self.ACCUM.assemble_step1(self.Np, b2, b2_eq)
        mat *= self.params_kin['nuh'] * self.params_kin['alpha'] * charge_over_mass
        
        # rhs of linear system
        rhs = self.MHD_OPS.A(up) + self.dts[1]/2 * mat.dot(up)
        
        # LHS of linear system
        LHS = spa.linalg.LinearOperator(self.MHD_OPS.A.shape, lambda x : self.MHD_OPS.A(x) - self.dts[1]/2 * mat.dot(x))
            
        # solve linear system with gmres method and values from last time step as initial guess 
        up[:], info = spa.linalg.gmres(LHS, rhs, x0=up, tol=self.params_sol['tol1'], 
                                                     maxiter=self.params_sol['maxiter1'], M=self.MHD_OPS.A_inv, callback=count_iters
                                                     )
        
        if print_info: 
            print('Status     for step_rhoh:', info)
            print('Iterations for step_rhoh:', num_iters)
            print('Maxdiff up for step_rhoh:', np.max(np.abs(up - up_old)))
            print()