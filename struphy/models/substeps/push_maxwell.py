import struphy.feec.basics.mass_matrices_3d_pre as mass_3d_pre
import scipy.sparse as spa
import numpy as np


class Push_maxwell:
    '''The substeps of the Poisson splitting algorithm for linear MHD equations
    
    [Ref] F. Holderied, S. Possanner and X. Wang, "MHD-kinetic hybrid code based on structure-preserving finite elements with particles-in-cell",
            J. Comp. Phys. 433 (2021) 110143.
    
    Parameters
    ----------            
        SPACES : obj
            FEEC spaces.
            
        OPERATORS : obj
            operators from emw_operators.EMW_operators.
        
        dts : list
            Time steps, one for each split step.

        params : dict
            parameters.yml.

    '''

    
    def __init__(self, DOMAIN, SPACES, time_steps, params):

        # Set objects
        self.__DOMAIN     = DOMAIN
        self.__SPACES     = SPACES

        # Define Dimensions 
        self.__dim_V1     = self.__SPACES.Ntot_1form_cum[-1]
        self.__dim_V2     = self.__SPACES.Ntot_2form_cum[-1]


        # Define parameter
        self.__solver     = params['solvers']
        self.__num_iters  = int(0)
        self.__dts        = time_steps

        # define the necessary linear operator for the maxwell step
        self.__Schur_maxwell   = spa.linalg.LinearOperator( (self.__dim_V1, self.__dim_V1), matvec=self.__Schur_maxwell_mat)
        self.__RHS_maxwell_e   = spa.linalg.LinearOperator( (self.__dim_V1, self.__dim_V1), matvec=self.__RHS_maxwell_e_mat)
        self.__RHS_maxwell_b   = spa.linalg.LinearOperator( (self.__dim_V1, self.__dim_V2), matvec=self.__RHS_maxwell_b_mat)
        self.__M1_inv          = mass_3d_pre.get_M1_PRE(self.__SPACES, self.__DOMAIN)

    # Counter function
    # ======================================
    def __counter(self):
        self.__num_iters += int(1)


    # Update operators for the maxwell step
    # ======================================
    def __Schur_maxwell_mat(self, u):
        return self.__SPACES.M1(u) + (self.__dts[0]**2/4.) * self.__SPACES.C0.T.dot(self.__SPACES.M2(self.__SPACES.C0.dot(u)))
    
    def __RHS_maxwell_e_mat(self, e):        
        return self.__SPACES.M1(e)- (self.__dts[0]**2/4.) * self.__SPACES.C0.T.dot(self.__SPACES.M2(self.__SPACES.C0.dot(e)))

    def __RHS_maxwell_b_mat(self, b):
        return self.__dts[0] * self.__SPACES.C0.T.dot(self.__SPACES.M2(b))

    # Executable steps
    # ======================================  
    def step_maxwell(self, e1, b2, print_info=False):
        '''Substep for the maxwell case.
           updates electric field (e1) and magnetic field (b2).
        
        Parameters
        ----------
            e1 : np.array
                FE coefficients, flattened.
                
            b2 : np.array
                FE coefficients, flattened.

            print_info : boolean
                Print to screen a) solver info b) number of iterations and c) max difference of abs(input-output).
        '''

        ## store initial values 
        e1_old  = e1.copy()
        b2_old  = b2.copy()
        
        ## calculate the 
        rhs     = self.__RHS_maxwell_e(e1) + self.__RHS_maxwell_b(b2)

        ## pick solver
        self.__num_iters = int(0)
        if   self.__solver['solver_type_2'] == 'gmres':
            e1[:], info = spa.linalg.gmres( self.__Schur_maxwell, rhs, x0=e1, tol=self.__solver['tol2'], maxiter=self.__solver['maxiter2'], M=self.__M1_inv, callback=self.__counter() )
        elif self.__solver['solver_type_2'] == 'cg':
            e1[:], info = spa.linalg.cg(    self.__Schur_maxwell, rhs, x0=e1, tol=self.__solver['tol2'], maxiter=self.__solver['maxiter2'], M=self.__M1_inv, callback=self.__counter() )
        elif self.__solver['solver_type_2'] == 'cgs':
            e1[:], info = spa.linalg.cgs(   self.__Schur_maxwell, rhs, x0=e1, tol=self.__solver['tol2'], maxiter=self.__solver['maxiter2'], M=self.__M1_inv, callback=self.__counter() )
        else:
            raise ValueError('only gmres and cg solvers available')
        
        ## update magnetic field (strong)
        b2[:] = b2 - self.__dts[0]*self.__SPACES.C0.dot((e1 + e1_old)/2.)

        ## print info
        if print_info: 
            print('Status     for step_maxwell:', info)
            print('Iterations for step_maxwell:', self.__num_iters)
            print('Maxdiff e1 for step_maxwell:', np.max(np.abs(e1 - e1_old)))
            print('Maxdiff b2 for step_maxwell:', np.max(np.abs(b2 - b2_old)))
            print()

