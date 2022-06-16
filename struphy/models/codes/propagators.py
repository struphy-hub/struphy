from abc import ABCMeta, abstractmethod  
import numpy as np
from psydac.linalg.stencil import StencilVector 

from struphy.linear_algebra.schur_solver import Schur_solver
from struphy.psydac_api.linear_operators import CompositeLinearOperator as Compose
from struphy.psydac_api.linear_operators import SumLinearOperator as Sum
from struphy.psydac_api.linear_operators import ScalarTimesLinearOperator as Multiply
from struphy.psydac_api.linear_operators import InverseLinearOperator as Invert
from struphy.psydac_api.preconditioner import MassMatrixPreConditioner as MassPre

from psydac.linalg.block import BlockVector

__all__ = ['Propagator',
            'StepMaxwell',]


class Propagator( metaclass=ABCMeta ):
    '''Base class for Struphy propagators used in Struphy models.
    
    Note
    ---- 
        All Struphy propagators are subclasses of ``Propagator`` and should be added to ``struphy/models/codes/propagators.py``'''

    @property
    @abstractmethod
    def variables(self):
        '''List of variabels to be updated by the propagator. Can contain
        
            * FE coefficients from the ``Field.vector`` property of :ref:`fields`.
            * Marker arrays from the ``Particles6D.markers`` (or ``Particles5D.markers``) property of :ref:`particles`.    
        '''
        pass

    @abstractmethod
    def push(self, dt):
        '''Push entries in ``Propagator.variables`` from t -> t + dt.
        Use ``Propagators.in_place_update`` to write to ``Propagator.variables``.
        
        Parameters
        ----------
            dt : float
                Time step size.
        '''
        pass

    def in_place_update(self, *variables_new):
        '''Writes new entries into ``Propagator.variables``.
        
        Parameters
        ----------
            variables_new : list
                Same sequence as in ``Propagator.variables`` but with the updated variables, 
                ie. for variables = [e, b] we must have variables_new = [e_updated, b_updated].
                
        Returns
        -------
            A list of max(abs(e - e_updated)) for all variables.'''

        diffs = []
        for i, arg in enumerate(variables_new):

            assert type(arg) is type(self.variables[i])

            if isinstance(arg, StencilVector):

                new = arg
                old = self.variables[i]
                diff = [np.max(np.abs(new._data - old._data))]
                old[:] = new[:]
                old.update_ghost_regions() # important: sync processes!

            elif isinstance(arg, BlockVector):

                diff = []
                for new, old in zip(arg, self.variables[i]):
                    diff += [np.max(np.abs(new._data - old._data))]
                    old[:] = new[:]
                    old.update_ghost_regions() # important: sync processes!

            else:
                raise NotImplementedError(f'Update of variable type {type(arg)} not implemented.')

            diffs += [diff]

        return diffs


class StepMaxwell( Propagator ):
    '''Crank-Nicolson step

    .. math::

        \\begin{bmatrix} e^{n+1} - e^n \\\ b^{n+1} - b^n \end{bmatrix} 
        = \\frac{\Delta t}{2} \\begin{bmatrix} 0 & \mathbb M_1^{-1} \mathbb C^\\top \\\ - \mathbb C \mathbb M_1^{-1} & 0 \end{bmatrix} 
        \\begin{bmatrix} \mathbb M_1(e^{n+1} + e^n) \\\ \mathbb M_2(b^{n+1} + b^n) \end{bmatrix} ,

    based on the :ref:`Schur complement <schur_solver>`.

    Parameters
    ---------- 
        e : BlockVector
            FE coefficients of a 1-form.

        b : BlockVector
            FE coefficients of a 2-form.

        DR: struphy.psydac_api.psydac_derham.DerhamBuild
            Discrete Derham complex.

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

        # current variables
        en = self.variables[0]
        bn = self.variables[1]

        # allocate temporary FemFields _e, _b during solution
        _e, info = self._schur_solver(en, self._B.dot(bn), dt)
        _b = bn - dt*self._C.dot(_e + en)

        # write new coeffs into Propagator.variables
        de, db = self.in_place_update(_e, _b)

        if self._info:
            print('Status     for Push_maxwell_psydac:', info['success'])
            print('Iterations for Push_maxwell_psydac:', info['niter'])
            print('Maxdiff e1 for Push_maxwell_psydac:', max(de))
            print('Maxdiff b2 for Push_maxwell_psydac:', max(db))
            print()

