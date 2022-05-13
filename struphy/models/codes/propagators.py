from abc import ABCMeta, abstractmethod  
import numpy as np
from psydac.linalg.stencil import StencilVector 

from struphy.linear_algebra.schur_solver import Schur_solver
from struphy.psydac_linear_operators.linear_operators import CompositeLinearOperator as Compose
from struphy.psydac_linear_operators.linear_operators import SumLinearOperator as Sum
from struphy.psydac_linear_operators.linear_operators import ScalarTimesLinearOperator as Multiply
from struphy.psydac_linear_operators.linear_operators import InverseLinearOperator as Invert
from struphy.psydac_linear_operators.preconditioner import MassMatrixPreConditioner as MassPre

from psydac.linalg.block import BlockVector

__all__ = ['Propagator',
            'StepMaxwell',]


class Propagator( metaclass=ABCMeta ):
    '''Base class for Struphy propagators used in Struphy models.'''

    @property
    @abstractmethod
    def variables(self):
        '''List of variabels to be updated by the propagator. Can be FE coefficients or particle arrays.'''
        pass

    @abstractmethod
    def push(self, dt):
        '''Push variables from t -> t + dt.
        
        Parameters
        ----------
            dt : float
                Time step size.

        Returns
        -------
            A list of updated variables.
        '''
        pass

    def in_place_update(self, *args):
        '''Updates variables in place (no new copy created).
        
        Parameters
        ----------
            args : list
                Updated variables. The sequence must be the same as in variables, 
                ie. for variables = [e, b] we must have args = [e_updated, b_updated].
                
        Returns
        -------
            A list of max(abs(e - e_updated)) for all variables.'''

        _diffs = []
        for i, arg in enumerate(args):

            assert type(arg) is type(self.variables[i])

            if isinstance(arg, StencilVector):

                new = arg
                old = self.variables[i]
                _diff = [np.max(np.abs(new._data - old._data))]
                old[:] = new[:]
                old.update_ghost_regions() # important: sync processes!

            elif isinstance(arg, BlockVector):

                _diff = []
                for new, old in zip(arg, self.variables[i]):
                    _diff += [np.max(np.abs(new._data - old._data))]
                    old[:] = new[:]
                    old.update_ghost_regions() # important: sync processes!

            else:
                raise NotImplementedError(f'Update of variable type {type(arg)} not implemented.')

            _diffs += [_diff]

        return _diffs


class StepMaxwell( Propagator ):
    '''Crank-Nicolson step

    [e - en, b - bn] = dt/2* [[0 M1^{-1}*C^T], [-C*M1^{-1} 0]] [M1*(e + en), M2(b + bn)] ,

    based on the Schur complement.

    Parameters
    ---------- 
        e : BlockVector
            FE coefficients of the electric field.

        b : BlockVector
            FE coefficients of the magnetic field.

        DR: obj
            From struphy/psydac_api/fields.Field_init.

        params: dict
            Solver parameters for this splitting step. 
    '''

    def __init__(self, e, b, DR, params):

        assert isinstance(e, BlockVector)
        assert isinstance(b, BlockVector)

        self._e = e
        self._b = b
        self._DR = DR
        self._info = params['info']

        # Preconditioner
        if params['pc'] == None:
            pc = None
        elif params['pc'] == 'fft':
            pc = MassPre(DR.V1)
        else:
            raise ValueError(f'Preconditioner "{params["pc"]}" not implemented.')

        # Define block matrix [[A B], [C I]] (without time step size dt in the diangonals)
        _A = DR.M1
        self._B = Multiply(-1./2., Compose(DR.curl.transpose(), DR.M2)) # no dt
        self._C = Multiply(1./2., DR.curl) # no dt
        _BC = Compose(self._B, self._C)

        # Instantiate Schur solver (constant in this case)
        self._schur_solver = Schur_solver(_A, _BC, pc=pc, tol=params['tol'], maxiter=params['maxiter'], verbose=params['verbose'])

    @property
    def variables(self):
        return [self._e, self._b]

    def push(self, dt):

        en = self.variables[0]
        bn = self.variables[1]

        _e, info = self._schur_solver(en, self._B.dot(bn), dt)
        _b = bn - dt*self._C.dot(_e + en)

        _de, _db = self.in_place_update(_e, _b)

        if self._info:
            print('Status     for Push_maxwell_psydac:', info['success'])
            print('Iterations for Push_maxwell_psydac:', info['niter'])
            print('Maxdiff e1 for Push_maxwell_psydac:', max(_de))
            print('Maxdiff b2 for Push_maxwell_psydac:', max(_db))
            print()