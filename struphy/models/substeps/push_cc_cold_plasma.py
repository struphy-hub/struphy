import scipy.sparse as spa
import numpy as np

from struphy.pic import pusher_vel_3d

class Current_coupling:
    '''The substeps of the Poisson splitting algorithm for current coupling terms.
    
    [Ref] F. Holderied, S. Possanner and X. Wang, "MHD-kinetic hybrid code based on structure-preserving finite elements with particles-in-cell", J. Comp. Phys. 433 (2021) 110143.
    
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


    def step_jh(self, e1, particles, b2_eq, b2, print_info=False):
        '''Coupling term involving hot current density, updates velocity (up) and marker velocity (v).
        
        Parameters
        ----------
            e1 : np.array
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
        e1_old = e1.copy()
        temp   = particles.copy()

        # charge over mass coefficient
        charge_over_mass = self.params_kin['particle_charge'] / self.params_kin['particle_mass']

        # counter for number of interation steps in iterative solvers
        num_iters = 0
        def count_iters(xk):
            nonlocal num_iters
            num_iters += 1

        # current accumulation (all processes) 
        self.ACCUM.accumulate_current(particles, self.MPI_COMM)
        # dummy = np.sum(particles)
        # print('particels: step_j,  mpi_rank:', self.MPI_COMM.Get_rank(), 'nans:', np.isnan(dummy))
        # build global sparse matrix and vector
        mat, vec = self.ACCUM.assemble_current(self.Np, b2, b2_eq)
        mat     *= self.params_kin['nuh'] * self.params_kin['alpha'] * charge_over_mass
        vec     *= self.params_kin['nuh'] * self.params_kin['alpha'] * charge_over_mass
        
        # rhs of linear system
        rhs = self.SPACES.M1(e1) - self.dts[0]**2/4 * mat.dot(e1) + self.dts[0]*vec
        
        # LHS of linear system
        LHS = spa.linalg.LinearOperator(self.SPACES.M1.shape, lambda x : self.SPACES.M1.dot(x) + self.dts[0]**2/4 * mat.dot(x))
        
        # solve linear system with gmres method and values from last time step as initial guess
        if   self.params_sol['solver_type_3'] == 'gmres':
            e1[:], info = spa.linalg.gmres(LHS, rhs, x0=e1, tol=self.params_sol['tol3'], 
                                                         maxiter=self.params_sol['maxiter3'], M=self.MHD_OPS.M1_inv, callback=count_iters
                                                         )
        elif self.params_sol['solver_type_3'] == 'cg':
            e1[:], info = spa.linalg.cg(   LHS, rhs, x0=e1, tol=self.params_sol['tol3'], 
                                                         maxiter=self.params_sol['maxiter3'], M=self.MHD_OPS.M1_inv, callback=count_iters
                                                         )
        elif self.params_sol['solver_type_3'] == 'cgs':
            e1[:], info = spa.linalg.cgs(  LHS, rhs, x0=e1, tol=self.params_sol['tol3'], 
                                                         maxiter=self.params_sol['maxiter3'], M=self.MHD_OPS.M1_inv, callback=count_iters
                                                         )
        
        self.MPI_COMM.Bcast(e1, root=0)
        self.MPI_COMM.Barrier()


        e1_ten_1, e1_ten_2, e1_ten_3 = self.SPACES.extract_1((e1 + e1_old)/2)
        
        pusher_vel_3d.pusher_v_cold_plasma(temp, charge_over_mass*self.dts[0],                                          # particle_pos and q_e/m_e * dts
                                self.SPACES.T[0], self.SPACES.T[1], self.SPACES.T[2],                                   # knot poinst     
                                self.SPACES.p, self.SPACES.Nel, self.SPACES.NbaseN, self.SPACES.NbaseD, self.Np_loc,    # degree, elements, N_tot_Bsple. N_tot_Msple
                                e1_ten_1, e1_ten_2, e1_ten_3,                                                           # vel field
                                self.basis_u,                                                                           # basis
                                self.DOMAIN.kind_map, self.DOMAIN.params_map,                                           # maps                 
                                self.DOMAIN.T[0], self.DOMAIN.T[1], self.DOMAIN.T[2],                                   # doamin vec                                      
                                self.DOMAIN.p, self.DOMAIN.Nel, self.DOMAIN.NbaseN,                                     #                              
                                self.DOMAIN.cx, self.DOMAIN.cy, self.DOMAIN.cz                            
                                )
        # dummy = np.sum(temp)
        # print('particels: step_v,  mpi_rank:', self.MPI_COMM.Get_rank(), 'nans:', np.isnan(dummy))
        # update global variable

        if print_info: 
            print('Status     for step_jh:', info)
            print('Iterations for step_jh:', num_iters)
            print('Maxdiff  e for step_jh:', np.max(np.abs(e1 - e1_old)))
            print('Maxdiff  v for step_jh:', np.max(np.abs(particles - temp)))
            print()
        particles[:, :] = temp


   