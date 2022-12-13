from abc import ABCMeta, abstractmethod  
import numpy as np

from psydac.linalg.stencil import StencilVector
from psydac.linalg.block import BlockVector


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
    def __call__(self, dt):
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
                    
            elif isinstance(arg, PolarVector):
                
                new = arg
                old = self.variables[i]
                # TODO: diff for PolarVectors
                old.set_vector(new)
                old.update_ghost_regions() # important: sync processes!

            else:
                raise NotImplementedError(f'Update of variable type {type(arg)} not implemented.')

            diffs += [diff]

        return diffs