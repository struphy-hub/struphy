from abc import ABCMeta, abstractmethod
import numpy as np

from psydac.linalg.stencil import StencilVector
from psydac.linalg.block import BlockVector
from struphy.polar.basic import PolarVector


class Propagator(metaclass=ABCMeta):
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
                i.e. for variables = [e, b] we must have variables_new = [e_updated, b_updated].

        Returns
        -------
            diffs : list
                A list [max(abs(self.variables - variables_new)), ...] for all variables in self.variables and variables_new.'''

        diffs = []
        
        for i, new in enumerate(variables_new):
            
            assert type(new) is type(self.variables[i])
            
            # calculate maximum of difference abs(old - new)
            diffs += [np.max(np.abs(self.variables[i].toarray() - new.toarray()))]
            
            # in-place update
            if isinstance(new, StencilVector):
                self.variables[i][:] = new[:]
                
            elif isinstance(new, BlockVector):
                for n in range(3):
                    self.variables[i][n][:] = new[n][:]
                    
            elif isinstance(new, PolarVector):
                self.variables[i].set_vector(new)
                
            else:
                raise NotImplementedError(
                    f'Update of variable type {type(arg)} not implemented.')
                
            # important: sync processes!
            self.variables[i].update_ghost_regions()

        return diffs