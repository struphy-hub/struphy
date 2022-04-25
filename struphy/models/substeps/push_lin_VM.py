import numpy as np
import scipy.sparse as spa

import struphy.feec.basics.mass_matrices_3d_pre as mass_3d_pre

from struphy.pic.lin_Vlasov_Maxwell import pusher_weights


class Push_lVM:
    '''Split step of the linearized Vlasov Maxwell system where E-field and weights are updated
    
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
            Parameters.
    '''

    def __init__(self, particles, dts, DOMAIN, SPACES, Np_loc, ACCUM, MPI_COMM, v_shift, v_th, n0, params):

        self.__particles    = particles
        self.__DOMAIN       = DOMAIN
        self.__SPACES       = SPACES
        self.__Np_loc       = Np_loc
        self.__ACCUM        = ACCUM
        self.__MPI_COMM     = MPI_COMM

        self.v_shift = v_shift
        self.v_th    = v_th
        self.n0      = n0

        # Define Dimension
        self.__dim_V1     = self.__SPACES.Ntot_1form_cum[-1]

        # Define parameters
        self.__solver     = params['solvers']
        self.__num_iters  = int(0)
        self.__dts        = dts

        # accumulate and assemble accumulation matrix and vector
        self.__ACCUM.accumulate_e_W_step(self.__particles, self.__MPI_COMM, self.__Np_loc, self.v_shift, self.v_th, self.n0)
        self.__Accum_mat, self.__Accum_vec = self.__ACCUM.assemble_step_e_W(self.__Np_loc)

        # define the necessary linear operator for the maxwell step
        self.__Schur_e_W        = spa.linalg.LinearOperator( (self.__dim_V1, self.__dim_V1), matvec=self.__Schur_e_W_mat)
        self.__RHS_e            = spa.linalg.LinearOperator( (self.__dim_V1, self.__dim_V1), matvec=self.__RHS_e_mat)

        self.__M1_inv           = mass_3d_pre.get_M1_PRE(self.__SPACES, self.__DOMAIN)

    # Counter function
    # ======================================
    def __counter(self):
        self.__num_iters += int(1)


    # Update operators for the e_W step
    # ======================================
    def __Schur_e_W_mat(self, u):
        return self.__SPACES.M1(u) + (self.__dts[0]**2/4.) * self.__Accum_mat.dot(u)
    
    def __RHS_e_mat(self, u):
        return self.__SPACES.M1(u) - (self.__dts[0]**2/4.) * self.__Accum_mat.dot(u)

    def step_e_W(self, particles, efield, print_info=False):
        """
        E_W substep (subsystem 3) in the linearized Vlasov-Maxwell system

        Parameters:
        -----------
            particles : np.array
                Shape (7, Np), where the rows hold the positions [:3] and velocities [3:6] and weights [6]

            efield : np.array
                Shape (Nel[0]*Nel[1]*Nel[2]*3, ), contains efield coefficients

            print_info : boolean
                Print the maximal absolute difference between input and output
        """

        # store initial values
        old_e         = efield.copy()
        old_particles = particles.copy()

        # accumulate and assemble accumulation matrix and vector
        self.__ACCUM.accumulate_e_W_step(particles, self.__MPI_COMM, self.__Np_loc, self.v_shift, self.v_th, self.n0)
        self.__Accum_mat, self.__Accum_vec = self.__ACCUM.assemble_step_e_W(self.__Np_loc)

        self.__update_e(efield, print_info=print_info)

        pusher_weights.push_weights(particles,
                                    self.__Np_loc,
                                    self.__SPACES.p,
                                    self.__SPACES.T[0],    self.__SPACES.T[1],    self.__SPACES.T[2],
                                    self.__SPACES.indN[0], self.__SPACES.indN[1], self.__SPACES.indN[2],
                                    self.__SPACES.indD[0], self.__SPACES.indD[1], self.__SPACES.indD[2],
                                    self.__SPACES.NbaseN, self.__SPACES.NbaseD,
                                    efield + old_e,
                                    self.__dts[0],
                                    self.v_shift, self.v_th, self.n0
                                    )
        
        # print info
        if print_info:
            print('Iterations           for step_e_W:', self.__num_iters)
            print('Maxdiff    e1        for step_e_W:', np.max(np.abs(efield - old_e)))
            print('Maxdiff    weights   for step_e_W:', np.max(np.abs(particles[6,:] - old_particles[6,:])))
            print()

    def __update_e(self, efield, print_info=False):
        """
        updates the e-field by inverting the Schur matrix in substep 3

        Parameters :
        ------------
        efield : array
            contains the values for the spline coefficients of the electric field
        
        print_info : Boolean
            if true then success status for solver will be displayed
        """
        
        old_e = efield.copy()

        ## calculate the RHS
        rhs = self.__RHS_e(old_e) - self.__dts[0] * self.__Accum_vec.flatten()

        ## pick solver
        self.__num_iters = int(0)
        if   self.__solver['solver_type_2'] == 'gmres':
            efield[:], info = spa.linalg.gmres( self.__Schur_e_W, rhs, x0=efield, tol=self.__solver['tol2'], maxiter=self.__solver['maxiter2'], M=self.__M1_inv, callback=self.__counter() )
        elif self.__solver['solver_type_2'] == 'cg':
            efield[:], info = spa.linalg.cg(    self.__Schur_e_W, rhs, x0=efield, tol=self.__solver['tol2'], maxiter=self.__solver['maxiter2'], M=self.__M1_inv, callback=self.__counter() )
        elif self.__solver['solver_type_2'] == 'cgs':
            efield[:], info = spa.linalg.cgs(   self.__Schur_e_W, rhs, x0=efield, tol=self.__solver['tol2'], maxiter=self.__solver['maxiter2'], M=self.__M1_inv, callback=self.__counter() )
        else:
            raise ValueError('only gmres and cg solvers available')
        
        # print info
        if print_info: 
            print('Status     for step_e_W:', info)
