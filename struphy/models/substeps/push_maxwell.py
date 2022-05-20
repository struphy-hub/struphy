import struphy.feec.basics.mass_matrices_3d_pre as mass_3d_pre
import scipy.sparse as spa
import numpy as np

from struphy.linear_algebra.schur_solver import Schur_solver
from struphy.psydac_api.linear_operators import CompositeLinearOperator as Compose
from struphy.psydac_api.linear_operators import SumLinearOperator as Sum
from struphy.psydac_api.linear_operators import ScalarTimesLinearOperator as Multiply
from struphy.psydac_api.linear_operators import InverseLinearOperator as Invert
from struphy.psydac_api.preconditioner import MassMatrixPreConditioner as MassPre

from psydac.linalg.stencil import StencilVector
from psydac.linalg.block import BlockVector


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
        self.__DOMAIN = DOMAIN
        self.__SPACES = SPACES

        # Define Dimensions
        self.__dim_V1 = self.__SPACES.Ntot_1form_cum[-1]
        self.__dim_V2 = self.__SPACES.Ntot_2form_cum[-1]

        # Define parameter
        self.__solver = params['solvers']
        self.__num_iters = int(0)
        self.__dts = time_steps

        # define the necessary linear operator for the maxwell step
        self.__Schur_maxwell = spa.linalg.LinearOperator(
            (self.__dim_V1, self.__dim_V1), matvec=self.__Schur_maxwell_mat)
        self.__RHS_maxwell_e = spa.linalg.LinearOperator(
            (self.__dim_V1, self.__dim_V1), matvec=self.__RHS_maxwell_e_mat)
        self.__RHS_maxwell_b = spa.linalg.LinearOperator(
            (self.__dim_V1, self.__dim_V2), matvec=self.__RHS_maxwell_b_mat)
        self.__M1_inv = mass_3d_pre.get_M1_PRE(self.__SPACES, self.__DOMAIN)

    # Counter function
    # ======================================
    def __counter(self):
        self.__num_iters += int(1)

    # Update operators for the maxwell step
    # ======================================

    def __Schur_maxwell_mat(self, u):
        return self.__SPACES.M1(u) + (self.__dts[0]**2/4.) * self.__SPACES.C0.T.dot(self.__SPACES.M2(self.__SPACES.C0.dot(u)))

    def __RHS_maxwell_e_mat(self, e):
        return self.__SPACES.M1(e) - (self.__dts[0]**2/4.) * self.__SPACES.C0.T.dot(self.__SPACES.M2(self.__SPACES.C0.dot(e)))

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

        # store initial values
        e1_old = e1.copy()
        b2_old = b2.copy()

        # calculate the
        rhs = self.__RHS_maxwell_e(e1) + self.__RHS_maxwell_b(b2)

        # pick solver
        self.__num_iters = int(0)
        if self.__solver['solver_type_2'] == 'gmres':
            e1[:], info = spa.linalg.gmres(self.__Schur_maxwell, rhs, x0=e1, tol=self.__solver['tol2'],
                                           maxiter=self.__solver['maxiter2'], M=self.__M1_inv, callback=self.__counter())
        elif self.__solver['solver_type_2'] == 'cg':
            e1[:], info = spa.linalg.cg(self.__Schur_maxwell, rhs, x0=e1, tol=self.__solver['tol2'],
                                        maxiter=self.__solver['maxiter2'], M=self.__M1_inv, callback=self.__counter())
        elif self.__solver['solver_type_2'] == 'cgs':
            e1[:], info = spa.linalg.cgs(self.__Schur_maxwell, rhs, x0=e1, tol=self.__solver['tol2'],
                                         maxiter=self.__solver['maxiter2'], M=self.__M1_inv, callback=self.__counter())
        else:
            raise ValueError('only gmres and cg solvers available')

        # update magnetic field (strong)
        b2[:] = b2 - self.__dts[0]*self.__SPACES.C0.dot((e1 + e1_old)/2.)

        # print info
        if print_info:
            print('Status     for step_maxwell:', info)
            print('Iterations for step_maxwell:', self.__num_iters)
            print('Maxdiff e1 for step_maxwell:', np.max(np.abs(e1 - e1_old)))
            print('Maxdiff b2 for step_maxwell:', np.max(np.abs(b2 - b2_old)))
            print()


class Push_maxwell_psydac:
    '''Crank-Nicolson step

    [e - en, b - bn] = dt/2* [[0 M1^{-1}*C^T], [-C*M1^{-1} 0]] [M1*(e + en), M2(b + bn)] ,

    based on the Schur complement.

    Parameters
    ----------    
        DR: obj
            From struphy/psydac_api/fields.Field_init.

        params: dict
            Solver parameters for this splitting step. 

    Arguments
    ---------
        en: StencilVector
            Electric field coefficients.

        bn: StencilVector
            Magnetic field coefficients.

        dt : float
            Time step size.

    Returns
    -------
        Nothing. The coefficients are updated in place (overwritten).
    '''

    def __init__(self, DR, params):

        self._DR = DR
        self._info = params['info']

        # Preconditioner
        if params['pc'] == None:
            pc = None
        elif params['pc'] == 'fft':
            pc = MassPre(DR.V1)
        else:
            raise ValueError(f'Preconditioner "{params["pc"]}" not implemented.')

        # Define block matrix (without time step size dt in the diangonals)
        _A = DR.M1
        self._B = Multiply(-1./2., Compose(DR.curl.transpose(), DR.M2)) # no dt
        self._C = Multiply(1./2., DR.curl) # no dt
        _BC = Compose(self._B, self._C)

        # Instantiate Schur solver
        self._schur_solver = Schur_solver(_A, _BC, pc=pc, tol=params['tol'], maxiter=params['maxiter'], verbose=params['verbose'])

    def __call__(self, en, bn, dt):

        assert isinstance(en, BlockVector)
        assert isinstance(bn, BlockVector)

        _e, info = self._schur_solver(en, self._B.dot(bn), dt)
        _b = bn - dt*self._C.dot(_e + en)

        # in place update of e
        _diff_e = []
        for old, new in zip(en, _e):
            _diff_e += [np.max(np.abs(new._data - old._data))]
            old[:] = new[:]
            old.update_ghost_regions() # important: sync processes!
            
        # in place update of b
        _diff_b = []
        for old, new in zip(bn, _b):
            _diff_b += [np.max(np.abs(new._data - old._data))]
            old[:] = new[:]
            old.update_ghost_regions() # important: sync processes!

        if self._info:
            print('Status     for Push_maxwell_psydac:', info['success'])
            print('Iterations for Push_maxwell_psydac:', info['niter'])
            print('Maxdiff e1 for Push_maxwell_psydac:', max(_diff_e))
            print('Maxdiff b2 for Push_maxwell_psydac:', max(_diff_b))
            print()

