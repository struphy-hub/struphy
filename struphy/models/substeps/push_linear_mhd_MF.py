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
            
        MHD_OPS_MF : obj
            MHD operators from mhd_operators_MF.py
            
        params_solver : dict
            Parameters for the linear "solvers" from parameters.yml.

        basis_u : int
            Which FE basis is used for up (1, 2 or 0).

        basis_ : int
            Which FE basis is used for pp (0 or 3).
    '''

    
    def __init__(self, dts, SPACES, MHD_OPS_MF, params_solver, basis_u, basis_p):

        self.dts        = dts[0]
        self.SPACES     = SPACES
        self.MHD_OPS_MF = MHD_OPS_MF
        self.params     = params_solver
        self.basis_u    = basis_u
        self.basis_p    = basis_p

        self.M0     = self.SPACES.M0
        self.M0_inv = spa.linalg.inv(self.SPACES.M0_mat.tocsc())
        self.M1     = self.SPACES.M1
        self.M2     = self.SPACES.M2
        self.M3     = self.SPACES.M3

        self.Grad   = self.SPACES.G
        self.Curl   = self.SPACES.C
        self.Div    = self.SPACES.D

        self.A      = MHD_OPS_MF.A

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

        if self.basis_u == 1:
            B = spa.linalg.LinearOperator((self.MHD_OPS_MF.dim_1, self.MHD_OPS_MF.dim_2), matvec = lambda x : -1 * self.dts   / 2 * self.MHD_OPS_MF.transpose_T1_dot(self.Curl.T.dot(self.M2.dot(x))) )
            C = spa.linalg.LinearOperator((self.MHD_OPS_MF.dim_2, self.MHD_OPS_MF.dim_1), matvec = lambda x :  self.dts   / 2 * self.Curl.dot(self.MHD_OPS_MF.T1_dot(x)) )
            S = spa.linalg.LinearOperator((self.MHD_OPS_MF.dim_1, self.MHD_OPS_MF.dim_1), matvec = lambda x :  (self.A - B * C).matvec(x) ) 
        
        elif self.basis_u ==2:
            B = spa.linalg.LinearOperator((self.MHD_OPS_MF.dim_2, self.MHD_OPS_MF.dim_2), matvec = lambda x : -1 * self.dts  / 2 * self.MHD_OPS_MF.transpose_T2_dot(self.Curl.T.dot(self.M2.dot(x))) )
            C = spa.linalg.LinearOperator((self.MHD_OPS_MF.dim_2, self.MHD_OPS_MF.dim_2), matvec = lambda x :  self.dts  / 2 * self.Curl.dot(self.MHD_OPS_MF.T2_dot(x)) )
            S = spa.linalg.LinearOperator((self.MHD_OPS_MF.dim_2, self.MHD_OPS_MF.dim_2), matvec = lambda x : (self.A - B * C).matvec(x) )
        
        else:
            raise ValueError('basis_u not implemented.')

        # store initial values 
        up_old = up.copy()
        b2_old = b2.copy()

        # counter for number of interation steps in iterative solvers
        num_iters = 0
        def count_iters(xk):
            nonlocal num_iters
            num_iters += 1
        
        # rhs of linear system
        rhs = (self.A + B * C).matvec(up) - 2 * B.matvec(b2)

        # pick solver
        if   self.params['solver_type_2'] == 'gmres':
            up[:], info = spa.linalg.gmres(S, rhs, x0=up, tol=self.params['tol2'], maxiter=self.params['maxiter2'], M=self.MHD_OPS_MF.A_inv, callback=count_iters)
        elif self.params['solver_type_2'] == 'cg':
            up[:], info = spa.linalg.cg(   S, rhs, x0=up, tol=self.params['tol2'], maxiter=self.params['maxiter2'], M=self.MHD_OPS_MF.A_inv, callback=count_iters)
        elif self.params['solver_type_2'] == 'cgs':
            up[:], info = spa.linalg.cgs(  S, rhs, x0=up, tol=self.params['tol2'], maxiter=self.params['maxiter2'], M=self.MHD_OPS_MF.A_inv, callback=count_iters)
        else:
            raise ValueError('only gmres and cg solvers available')

        # update magnetic field (strong)
        b2[:] = b2 - C.matvec(up + up_old)

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

        if self.basis_u == 1:
            if self.basis_p == 0:
                B     = spa.linalg.LinearOperator((self.MHD_OPS_MF.dim_1, self.MHD_OPS_MF.dim_0), matvec = lambda x : self.dts / 2 * self.M1.dot(self.Grad.dot(x)))
                C     = spa.linalg.LinearOperator((self.MHD_OPS_MF.dim_0, self.MHD_OPS_MF.dim_1), matvec = lambda x : -1 * self.dts / 2 * (self.M0_inv.dot(self.Grad.T.dot(self.M1.dot(self.MHD_OPS_MF.S10_dot(x)))) + (5./3. - 1)*self.M0_inv.dot(self.MHD_OPS_MF.transpose_K10_dot(self.Grad.T.dot(self.M1.dot(x))))) )
                S     = spa.linalg.LinearOperator((self.MHD_OPS_MF.dim_1, self.MHD_OPS_MF.dim_1), matvec = lambda x :  (self.A - B * C).matvec(x) )
            
            elif self.basis_p == 3:
                B     = spa.linalg.LinearOperator((self.MHD_OPS_MF.dim_1, self.MHD_OPS_MF.dim_3), matvec = lambda x : -1 * self.dts / 2 * self.MHD_OPS_MF.transpose_U1_dot(self.Div.T.dot(self.M3.dot(x))) )
                C     = spa.linalg.LinearOperator((self.MHD_OPS_MF.dim_3, self.MHD_OPS_MF.dim_1), matvec = lambda x :  self.dts / 2 * (self.Div.dot(self.MHD_OPS_MF.S1_dot(x)) + (5./3. - 1) * self.MHD_OPS_MF.K1_dot(self.Div.dot(self.MHD_OPS_MF.U1_dot(x)))) )
                S     = spa.linalg.LinearOperator((self.MHD_OPS_MF.dim_1, self.MHD_OPS_MF.dim_1), matvec = lambda x : (self.A - B * C).matvec(x) )

            else:
                raise ValueError('basis_u/basis_p not implemented.')

        elif self.basis_u ==2:
            if self.basis_p == 0:
                B = spa.linalg.LinearOperator((self.MHD_OPS_MF.dim_2, self.MHD_OPS_MF.dim_0), matvec = lambda x : -1 * self.dts / 2 * self.Div.T.dot(self.M3.dot(self.MHD_OPS_MF.Y20_dot(x))) )
                # slower 
                C = spa.linalg.LinearOperator((self.MHD_OPS_MF.dim_0, self.MHD_OPS_MF.dim_2), matvec = lambda x : -1 * self.dts / 2 * (self.M0_inv.dot(self.Grad.T.dot(self.M1.dot(self.MHD_OPS_MF.S20_dot(x)))) + (5./3. - 1) * self.M0_inv.dot(self.MHD_OPS_MF.transpose_K10_dot(self.Grad.T.dot(self.MHD_OPS_MF.transpose_U1_dot(self.M2.dot(x))))) ) )
                # faster
                C = spa.linalg.LinearOperator((self.MHD_OPS_MF.dim_0, self.MHD_OPS_MF.dim_2), matvec = lambda x : -1 * self.dts / 2 * (self.M0_inv.dot(self.Grad.T.dot(self.M1.dot(self.MHD_OPS_MF.S20_dot(x)))) + (5./3. - 1) * self.M0_inv.dot(self.MHD_OPS_MF.transpose_K10_dot(self.Grad.T.dot(self.M1.dot(self.MHD_OPS_MF.Z20_dot(x))))) ) )
                S = spa.linalg.LinearOperator((self.MHD_OPS_MF.dim_2, self.MHD_OPS_MF.dim_2), matvec = lambda x : (self.A - B * C).matvec(x) )
            
            elif self.basis_p == 3:
                B = spa.linalg.LinearOperator((self.MHD_OPS_MF.dim_1, self.MHD_OPS_MF.dim_3), matvec = lambda x : -1 * self.dts / 2 * self.Div.T.dot(self.M3.dot(x)) )
                C = spa.linalg.LinearOperator((self.MHD_OPS_MF.dim_3, self.MHD_OPS_MF.dim_2), matvec = lambda x :  self.dts / 2 * (self.Div.dot(self.MHD_OPS_MF.S2_dot(x)) + (5./3. - 1) * self.MHD_OPS_MF.K2_dot(self.Div.dot(x))) )
                S = spa.linalg.LinearOperator((self.MHD_OPS_MF.dim_2, self.MHD_OPS_MF.dim_2), matvec = lambda x : (self.A - B * C).matvec(x) )
            
            else:
                raise ValueError('basis_u/basis_p not implemented.')
        else:
            raise ValueError('basis_u/basis_p not implemented.')

        # store initial values 
        r3_old = r3.copy()
        up_old = up.copy()
        pp_old = pp.copy()

        # counter for number of interation steps in iterative solvers
        num_iters = 0
        def count_iters(xk):
            nonlocal num_iters
            num_iters += 1
        
        # rhs of linear system
        if self.basis_u == 1:
            rhs = (self.A + B * C).matvec(up) - 2 * B.matvec(pp) + self.dts * self.M1.dot(self.MHD_OPS_MF.P1_dot(b2))
        
        elif self.basis_u ==2:
            rhs = (self.A + B * C).matvec(up) - 2 * B.matvec(pp) + self.dts * self.M2.dot(self.MHD_OPS_MF.P2_dot(b2))

        # solve linear system with conjugate gradient squared method and values from last time step as initial guess
        if   self.params['solver_type_2'] == 'gmres':
            up[:], info = spa.linalg.gmres(S, rhs, x0=up, tol=self.params['tol6'], maxiter=self.params['maxiter6'], M=self.MHD_OPS_MF.A_inv, callback=count_iters)
        elif self.params['solver_type_2'] == 'cg':
            up[:], info = spa.linalg.cg(   S, rhs, x0=up, tol=self.params['tol6'], maxiter=self.params['maxiter6'], M=self.MHD_OPS_MF.A_inv, callback=count_iters)
        elif self.params['solver_type_2'] == 'cgs':
            up[:], info = spa.linalg.cgs(  S, rhs, x0=up, tol=self.params['tol6'], maxiter=self.params['maxiter6'], M=self.MHD_OPS_MF.A_inv, callback=count_iters)
        else:
            raise ValueError('only gmres and cg solvers available')
            
        # update pressure
        pp[:] = pp - C.matvec(up + up_old)
        # update density
        r3[:] = r3 - self.dts / 2 * self.Div.dot(self.MHD_OPS_MF.Q2_dot(up + up_old))

        if print_info: 
            print('Status     for step_magnetosonic:', info)
            print('Iterations for step_magnetosonic:', num_iters)
            print('Maxdiff r3 for step_magnetosonic:', np.max(np.abs(r3 - r3_old)))
            print('Maxdiff up for step_magnetosonic:', np.max(np.abs(up - up_old)))
            print('Maxdiff pp for step_magnetosonic:', np.max(np.abs(pp - pp_old)))
            print()

        