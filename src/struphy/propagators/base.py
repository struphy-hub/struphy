from abc import ABCMeta, abstractmethod
import numpy as np


class Propagator(metaclass=ABCMeta):
    '''Base class for Struphy propagators used in Struphy models.

    Note
    ---- 
        All Struphy propagators are subclasses of ``Propagator``.
        
        The ``__init__`` of child classes must take as first arguments the variables to be updated.
        All additional arguments MUST be passed as **keyword arguments**.
    '''

    @property
    @abstractmethod
    def variables(self):
        '''List of FEEC variables (not particles) to be updated by the propagator. 
        Contains FE coefficients from the ``Field.vector`` property of :ref:`fields`.
        '''
        pass

    @abstractmethod
    def __call__(self, dt):
        '''Update from t -> t + dt.
        Use ``Propagators.in_place_update`` to write to FE variables to ``Propagator.variables``.

        Parameters
        ----------
            dt : float
                Time step size.
        '''
        pass

    @property
    def derham(self):
        """ Derham spaces and projectors.
        """
        assert hasattr(
            self, '_derham'), 'Derham not set. Please do obj.deram = ...'
        return self._derham

    @derham.setter
    def derham(self, derham):
        self._derham = derham

    @property
    def domain(self):
        """ Domain object that characterizes the mapping from the logical to the physical domain.
        """
        assert hasattr(self, '_domain'), \
            'Domain for analytical MHD equilibrium not set. Please do obj.domain = ...'
        return self._domain

    @domain.setter
    def domain(self, domain):
        self._domain = domain

    @property
    def mass_ops(self):
        """ Weighted mass operators.
        """
        assert hasattr(self, '_mass_ops'), \
            'Weighted mass operators not set. Please do obj.mass_ops = ...'
        return self._mass_ops

    @mass_ops.setter
    def mass_ops(self, mass_ops):
        self._mass_ops = mass_ops

    @property
    def basis_ops(self):
        """ Basis projection operators.
        """
        assert hasattr(self, '_basis_ops'), \
            'Basis projection operators not set. Please do obj.basis_ops = ...'
        return self._basis_ops

    @basis_ops.setter
    def basis_ops(self, basis_ops):
        self._basis_ops = basis_ops

    def in_place_update(self, *variables_new):
        '''Writes new entries into the FEEC variables in ``Propagator.variables``.

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

            # copy new variables into self.variables
            new.copy(out=self.variables[i])

            # important: sync processes!
            self.variables[i].update_ghost_regions()

        return diffs
