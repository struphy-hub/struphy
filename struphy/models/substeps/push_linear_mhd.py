import scipy.sparse as spa
import numpy as np

class Linear_mhd:
    '''The substeps of the Poisson splitting algorithm for linear MHD equations
    
    [Ref] F. Holderied, S. Possanner and X. Wang, "MHD-kinetic hybrid code based on structure-preserving finite elements with particles-in-cell",
            J. Comp. Phys. 433 (2021) 110143.
    
    Parameters
    ----------
        dts : list
            Time steps, one for each split step.
            
        SPACES : obj
            FEEC spaces.
            
        MHD_OPS : obj
            MHD operators from mhd_ops.MHD_operators.
            
        params_solver : dict
            Parameters for the linear "solvers" from parameters.yml.

        basis_u : int
            Which FE basis is used for up (1, 2 or 0).

        basis_ : int
            Which FE basis is used for pp (0 or 3).
    '''

    
    def __init__(self, dts, SPACES, MHD_OPS, params_solver, basis_u, basis_p):

        self.dts     = dts
        self.SPACES  = SPACES
        self.MHD_OPS = MHD_OPS
        self.params  = params_solver
        self.basis_u = basis_u
        self.basis_p = basis_p


    def step_alfven(self, up, b2, print_info=False):
        '''Alfven substep, updates velocity (up) and magnetic field (b2).
        
        Parameters
        ----------
            up : np.array
                FE coefficients, flattened.
                
            b2 : np.array
                FE coefficients, flattened.

            print_info : boolean
                Print to screen a) solver info b) number of iterations and c) max difference of abs(input-output).
        '''

        # store initial values 
        up_old = up.copy()
        b2_old = b2.copy()

        # counter for number of interation steps in iterative solvers
        num_iters = 0
        def count_iters(xk):
            nonlocal num_iters
            num_iters += 1
        
        if self.basis_u == 2:

            # rhs of linear system
            rhs = self.MHD_OPS.RHS2(up, b2)
            # pick solver
            if   self.params['solver_type_2'] == 'gmres':
                up[:], info = spa.linalg.gmres(self.MHD_OPS.S2, rhs, x0=up, tol=self.params['tol2'], maxiter=self.params['maxiter2'], M=self.MHD_OPS.S2_PRE, callback=count_iters)
            elif self.params['solver_type_2'] == 'cg':
                up[:], info = spa.linalg.cg(   self.MHD_OPS.S2, rhs, x0=up, tol=self.params['tol2'], maxiter=self.params['maxiter2'], M=self.MHD_OPS.S2_PRE, callback=count_iters)
            elif self.params['solver_type_2'] == 'cgs':
                up[:], info = spa.linalg.cgs(  self.MHD_OPS.S2, rhs, x0=up, tol=self.params['tol2'], maxiter=self.params['maxiter2'], M=self.MHD_OPS.S2_PRE, callback=count_iters)
            else:
                raise ValueError('only gmres and cg solvers available')
            # update magnetic field (strong)
            b2[:] = b2 - self.dts[0]*self.SPACES.C.dot(self.MHD_OPS.EF((up + up_old)/2))

        else:
            raise ValueError('basis_u not implemented.')

        if print_info: 
            print('Status     for step_alfven:', info)
            print('Iterations for step_alfven:', num_iters)
            print('Maxdiff up for step_alfven:', np.max(np.abs(up - up_old)))
            print('Maxdiff b2 for step_alfven:', np.max(np.abs(b2 - b2_old)))
            print()
        
        
    def step_magnetosonic(self, r3, up, b2, pp, print_info=False):
        '''Non-Hamiltonian substep, updates density (r3), velocity (up) and pressure (pp).
        
        Parameters
        ----------
            r3 : np.array
                FE coefficients, flattened.

            up : np.array
                FE coefficients, flattened.

            b2_old : np.array
                FE coefficients, flattened.
                
            pp : np.array
                FE coefficients, flattened.

            print_info : boolean
                Print to screen a) solver info b) number of iterations and c) max difference of abs(input-output).
        '''

        # store initial values 
        r3_old = r3.copy()
        up_old = up.copy()
        pp_old = pp.copy()

        # counter for number of interation steps in iterative solvers
        num_iters = 0
        def count_iters(xk):
            nonlocal num_iters
            num_iters += 1
        
        if self.basis_u == 2 and self.basis_p == 3:

            # rhs of linear system
            rhs = self.MHD_OPS.RHS6(up, pp, b2)
            # solve linear system with conjugate gradient squared method and values from last time step as initial guess
            up[:], info = spa.linalg.gmres(self.MHD_OPS.S6, rhs, x0=up, tol=self.params['tol6'], 
                                            maxiter=self.params['maxiter6'], M=self.MHD_OPS.S6_PRE, callback=count_iters)
            # update pressure
            pp[:] = pp + self.dts[1]*self.MHD_OPS.L((up + up_old)/2)
            # update density
            r3[:] = r3 - self.dts[1]*self.SPACES.D.dot(self.MHD_OPS.MF((up + up_old)/2))

        else:
            raise ValueError('basis_u/basis_p not implemented.')

        if print_info: 
            print('Status     for step_magnetosonic:', info)
            print('Iterations for step_magnetosonic:', num_iters)
            print('Maxdiff r3 for step_magnetosonic:', np.max(np.abs(r3 - r3_old)))
            print('Maxdiff up for step_magnetosonic:', np.max(np.abs(up - up_old)))
            print('Maxdiff pp for step_magnetosonic:', np.max(np.abs(pp - pp_old)))
            print()

        