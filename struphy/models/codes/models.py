from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.special as sp

from struphy.psydac_linear_operators.fields import Field_init
from struphy.diagnostics.data_module import Data_container_psydac as Data_container   


class StruphyModels( metaclass=ABCMeta ):
    '''The base class for Struphy models.
    
    Parameters
    ..........
        names : list of str
            Names of FE fields.
        
        spaces : list of str
            One of "H1", "Hcurl", "Hdiv" or "L2"; len(spaces)==len(names), one space for each name.

        DR: obj
            From struphy/feec/psydac_derham.Derham_build.

        DOMAIN: obj
            From struphy/geometry/domain_3d.Domain.

        solver_params : list
            Each entry corresponds to one linear solver used in the model; 
            an entry is a dict with the solver parameters correpsonding to one solver used, obtained from paramaters.yml.
    '''

    def __init__(self, names, spaces, DR, DOMAIN, *solver_params, verbose=False):

        self._names = names
        self._spaces = spaces
        self._DR = DR
        self._DOMAIN = DOMAIN
        self._solver_params = solver_params

        self._fields = []
        for name, space in zip(names, spaces):
            self._fields += [Field_init(name, space, self.DR)]

            if verbose:
                print(f'field      : {self._fields[-1].name}')
                print(f'space_id   : {self._fields[-1].space_id}')
                print(f'starts     : {self._fields[-1].starts}')
                print(f'ends       : {self._fields[-1].ends}')
                print(f'pads       : {self._fields[-1].pads}')

        self._propagators = []
        self._substep_vars = []

    @property
    def names( self ):
        '''List of FE variable names.'''
        return self._names

    @property
    def spaces( self ):
        '''List of 3d Derham spaces corresponding to names.'''
        return self._spaces

    @property
    def DR( self ):
        '''3d Derham sequence, see struphy/feec/psydac_derham.'''
        return self._DR

    @property
    def DOMAIN( self ):
        '''Domain object, see struphy/geometry/domain_3d.'''
        return self._DOMAIN

    @property
    def fields( self ):
        '''List of Struphy fields, see struphy/psydac_linear_operators/fields.'''
        return self._fields

    def set_initial_conditions(self, comps_li, init_type, init_coords, init_params):
        '''For FE coefficients.
        
        Parameters
        ----------
            comps_li : list
                From parameters.yml, ['fields']['general']['init_comps']. Specifies which components(s) of each field should be initialized.'''

        for field, comps in zip(self.fields, comps_li):
            field.set_initial_conditions(self.DOMAIN, comps=comps, init_type=init_type, init_coords=init_coords, init_params=init_params)

    @property
    @abstractmethod
    def propagators(self):
        '''Must return list of callable propagators/integrators/substeps used in the time stepping of the model.'''

    @property
    @abstractmethod
    def substep_vars(self):
        '''Must return list of lists, where substep_vars[i][j] points to the j-th variable updated in the i-th substep
        specified in propagators; len(substep_vars)==len(propagators).'''


class Maxwell( StruphyModels ):
    '''Maxwell's equations in vacuum, in Struphy normalization (c=1).'''

    def __init__(self, names, spaces, DR, DOMAIN, *solver_params, verbose=False):

        from struphy.models.substeps.push_maxwell import Push_maxwell_psydac

        super().__init__(names, spaces, DR, DOMAIN, *solver_params, verbose=verbose)

        # Assemble necessary mass matrices 
        self.DR.assemble_M1()
        self.DR.assemble_M2()

        # Pointers to Stencil-/Blockvectors
        e = self.fields[0].vector
        b = self.fields[1].vector

        # Initialize propagators/integrators used in splitting substeps
        self._propagators += [Push_maxwell_psydac(DR, self._solver_params[0])]
        self._substep_vars +=[[e, b]]   

    def propagators(self):
        return self._propagators

    def substep_vars(self):
        return self._substep_vars

    

    