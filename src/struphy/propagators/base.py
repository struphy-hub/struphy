'Propagator base class.'


from abc import ABCMeta, abstractmethod
import numpy as np


class Propagator(metaclass=ABCMeta):
    """ Base class for Struphy propagators used in Struphy models. 

    Note
    ---- 
    All Struphy propagators are subclasses of ``Propagator`` and must be added to ``struphy/propagators``
    in one of the modules ``propagators_fields.py``, ``propagators_markers.py`` or ``propagators_coupling.py``.
    Only propagators that update both a FEEC and a PIC species go into ``propagators_coupling.py``.
    """

    def __init__(self, *vars):
        """ Create an instance of a Propagator.

        Parameters
        ----------
        vars : Vector or Particles
            :attr:`struphy.models.base.StruphyModel.pointer` of variables to be updated.
        """
        from psydac.linalg.basic import Vector
        from struphy.pic.particles import Particles

        self._feec_vars = []
        self._particles = []

        for var in vars:
            if isinstance(var, Vector):
                self._feec_vars += [var]
            elif isinstance(var, Particles):
                self._particles += [var]
            else:
                ValueError(
                    f'Variable {var} must be of type "Vector" or "Particles".')

    @property
    def feec_vars(self):
        """ List of FEEC variables (not particles) to be updated by the propagator. 
        Contains FE coefficients from :attr:`struphy.feec.Derham.Field.vector`.
        """
        return self._feec_vars

    @property
    def particles(self):
        """ List of kinetic variables (not FEEC) to be updated by the propagator. 
        Contains :class:`struphy.pic.particles.Particles`.
        """
        return self._particles

    @abstractmethod
    def __call__(self, dt):
        """ Update from t -> t + dt.
        Use ``Propagators.feec_vars_update`` to write to FEEC variables to ``Propagator.feec_vars``.

        Parameters
        ----------
        dt : float
            Time step size.
        """
        pass

    @classmethod
    @abstractmethod
    def options(cls):
        '''Dictionary of available propagator options, as appearing under species/options in the parameter file.'''
        pass

    @property
    def derham(self):
        """ Derham spaces and projectors.
        """
        assert hasattr(
            self, '_derham'), 'Derham not set. Please do obj.derham = ...'
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

    def feec_vars_update(self, *variables_new):
        """ Writes new entries into the FEEC variables in ``Propagator.feec_vars``.

        Parameters
        ----------
        variables_new : list
            Same sequence as in ``Propagator.feec_vars`` but with the updated variables, 
            i.e. for feec_vars = [e, b] we must have variables_new = [e_updated, b_updated].

        Returns
        -------
        diffs : list
            A list [max(abs(self.feec_vars - variables_new)), ...] for all variables in self.feec_vars and variables_new.
        """

        diffs = []

        for i, new in enumerate(variables_new):

            assert type(new) is type(self.feec_vars[i])

            # calculate maximum of difference abs(old - new)
            diffs += [np.max(np.abs(self.feec_vars[i].toarray() - new.toarray()))]

            # copy new variables into self.feec_vars
            new.copy(out=self.feec_vars[i])

            # important: sync processes!
            self.feec_vars[i].update_ghost_regions()

        return diffs
