import struphy.feec.basics.mass_matrices_3d_pre as mass_3d_pre
import scipy.sparse as spa
import numpy as np


class Push_cold_plasma:
    '''The substeps of the Poisson splitting algorithm for cold plasma equations
    

    Parameters
    ----------        
        DOMAIN : onj
            domain space

        SPACES : obj
            FEEC spaces.
            
        OPERATORS : obj
            operators from emw_operators.EMW_operators.
        
        dts : list
            Time steps, one for each split step.

        params : dict
            parameters.yml.

    '''
    def __init__(self, DOMAIN, SPACES, OPERATORS, time_steps, params):

        # Set objects
        self.__DOMAIN     = DOMAIN
        self.__SPACES     = SPACES
        self.__OPERATORS  = OPERATORS

        # Define Dimensions 
        self.__dim_V1     = self.__SPACES.Ntot_1form_cum[-1]
        self.__dim_V2     = self.__SPACES.Ntot_2form_cum[-1]


        # Define parameter
        self.__solver     = params['solvers']
        self.__num_iters  = int(0)
        self.__dts        = time_steps
        self.__ratio_w    = params['equilibrium']['params_slab']['ratio_omega']

        # define the necessary linear operator for the maxwell step
        self.__Schur_maxwell   = spa.linalg.LinearOperator( (self.__dim_V1, self.__dim_V1), matvec=self.__Schur_maxwell_fun)
        self.__RHS_maxwell_e   = spa.linalg.LinearOperator( (self.__dim_V1, self.__dim_V1), matvec=self.__RHS_maxwell_e_fun)
        self.__RHS_maxwell_b   = spa.linalg.LinearOperator( (self.__dim_V1, self.__dim_V2), matvec=self.__RHS_maxwell_b_fun)
        self.__M1_inv          = mass_3d_pre.get_M1_PRE(self.__SPACES, self.__DOMAIN)

        # define the necessary linear operator for the maxwell step
        self.__CrNi_rot       = spa.linalg.LinearOperator( (self.__dim_V1, self.__dim_V1), matvec=self.__CrNi_rot_fun)
        self.__RHS_rot        = spa.linalg.LinearOperator( (self.__dim_V1, self.__dim_V1), matvec=self.__RHS_rot_fun)


    # Counter function
    # ======================================
    def __counter(self):
        self.__num_iters += int(1)

    # Update operators for the maxwell step
    # ======================================
    def __Schur_maxwell_fun(self, u):
        return self.__SPACES.M1(u) + (self.__dts[0]**2/4.) * self.__SPACES.C0.T.dot(self.__SPACES.M2(self.__SPACES.C0.dot(u)))
    
    def __RHS_maxwell_e_fun(self, e):        
        return self.__SPACES.M1(e)- (self.__dts[0]**2/4.) * self.__SPACES.C0.T.dot(self.__SPACES.M2(self.__SPACES.C0.dot(e)))

    def __RHS_maxwell_b_fun(self, b):
        return self.__dts[0] * self.__SPACES.C0.T.dot(self.__SPACES.M2(b))

    # Update operator for the rotation step
    # ======================================    
    def __CrNi_rot_fun(self, j):
        return self.__SPACES.M1(j) - (self.__dts[0] /2.)*self.__OPERATORS.R1(j)    

    def __RHS_rot_fun(self, j):
        return self.__SPACES.M1(j) + (self.__dts[0]/2.)*self.__OPERATORS.R1(j)


        
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
            print('Iterations for step_maxwell:', self.num_iters)
            print('Maxdiff e1 for step_maxwell:', np.max(np.abs(e1 - e1_old)))
            print('Maxdiff b2 for step_maxwell:', np.max(np.abs(b2 - b2_old)))
            print()

    def step_analytic(self, e1, j1, print_info=False):
        '''Rotation substep, updates electric field (e1) and current field (j1) analytical
        
        Parameters
        ----------
            e1 : np.array
                FE coefficients, flattened.
                
            j1 : np.array
                FE coefficients, flattened.

            print_info : boolean
                Print to screen a) solver info b) number of iterations and c) max difference of abs(input-output).
        '''

        # store initial values 
        e1_old = e1.copy()
        j1_old = j1.copy()

        # update electric and current field (analytical)
        e1[:] = np.cos(self.__ratio_w*self.__dts[0])*e1_old - np.sin(self.__ratio_w*self.__dts[0])*j1_old
        j1[:] = np.sin(self.__ratio_w*self.__dts[0])*e1_old + np.cos(self.__ratio_w*self.__dts[0])*j1_old
        

        if print_info: 
            print('Maxdiff e1 for step_rotation:', np.max(np.abs(e1 - e1_old)))
            print('Maxdiff b2 for step_rotation:', np.max(np.abs(j1 - j1_old)))
            print()

    def step_rotation(self, j1, print_info=False):
        '''current substep,  ubdated current.
        
        Parameters
        ----------
            j1 : np.array
                FE coefficients, flattened.
                
            print_info : boolean
                Print to screen a) solver info b) number of iterations and c) max difference of abs(input-output).
        '''

        # store initial values 
        j1_old = j1.copy()
        
        # rhs of linear system
        rhs = self.__RHS_rot(j1)
        
        # pick solver
        self.num_iters = int(0)
        if   self.__solver['solver_type_2'] == 'gmres':     
            j1[:], info = spa.linalg.gmres( self.__CrNi_rot, rhs, x0=j1, tol=self.__solver['tol2'], maxiter=self.__solver['maxiter2'], M=self.__M1_inv, callback=self.__counter() )
        elif self.__solver['solver_type_2'] == 'cg':
            j1[:], info = spa.linalg.cg(    self.__CrNi_rot, rhs, x0=j1, tol=self.__solver['tol2'], maxiter=self.__solver['maxiter2'], M=self.__M1_inv, callback=self.__counter() )
        elif self.__solver['solver_type_2'] == 'cgs':
            j1[:], info = spa.linalg.cgs(   self.__CrNi_rot, rhs, x0=j1, tol=self.__solver['tol2'], maxiter=self.__solver['maxiter2'], M=self.__M1_inv, callback=self.__counter() )
        else:
            raise ValueError('only gmres and cg solvers available')


        if print_info: 
            print('Status     for step_cross_product:', info)
            print('Iterations for step_cross_product:', self.num_iters)
            print('Maxdiff j1 for step_cross_product:', np.max(np.abs(j1 - j1_old)))
            print()


