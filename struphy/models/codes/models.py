from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.special as sp

from struphy.psydac_linear_operators.fields import Field_init
from struphy.diagnostics.data_module import Data_container_psydac as Data_container   

__all__ = ['StruphyModels',
            'Maxwell',]


class StruphyModel( metaclass=ABCMeta ):
    '''The base class for Struphy models.
    
    Parameters
    ..........
        DR: Derham obj
            From struphy/feec/psydac_derham.Derham_build.

        DOMAIN: Domain obj
            From struphy/geometry/domain_3d.Domain.

        solver_params : list
            Each entry corresponds to one linear solver used in the model. 
            An entry is a dict with the solver parameters correpsonding to one solver, obtained from paramaters.yml.

        fields_def : dict
            Keys are the field names (str), values are the space_ids ("H1", "Hcurl", "Hdiv" or "L2"). 
    '''

    def __init__(self, DR, DOMAIN, *solver_params, **fields_def):

        self._names = []
        self._space_ids = []
        for key, val in fields_def.items():
            self._names += [key]
            self._space_ids += [val]

        self._DR = DR
        self._DOMAIN = DOMAIN
        self._solver_params = solver_params

        self._fields = []
        for name, space_id in zip(self._names, self._space_ids):
            self._fields += [Field_init(name, space_id, self.DR)]

        # These need to be filled in each model class
        self._propagators = []
        self._substep_vars = []
        self._scalar_quantities = {}

    @property
    def names( self ):
        '''List of FE variable names (str).'''
        return self._names

    @property
    def space_ids( self ):
        '''List of 3d Derham space identifiers (str) corresponding to names.'''
        return self._space_ids

    @property
    def fields( self ):
        '''List of Struphy fields, see struphy/psydac_linear_operators/fields.'''
        return self._fields

    @property
    def DR( self ):
        '''3d Derham sequence, see struphy/feec/psydac_derham.'''
        return self._DR

    @property
    def DOMAIN( self ):
        '''Domain object, see struphy/geometry/domain_3d.'''
        return self._DOMAIN

    @property
    def solver_params( self ):
        '''List of dicts holding the solver parameters.'''
        return self._solver_params

    @property
    def propagators( self ):
        '''Must return list of callable propagators/integrators/substeps used in the time stepping of the model.'''
        return self._propagators

    @property
    def substep_vars( self ):
        '''Must return list of lists, where substep_vars[i][j] points to the j-th variable updated in the i-th substep
        specified in propagators; len(substep_vars)==len(propagators).'''
        return self._substep_vars

    @property
    def scalar_quantities( self ):
        '''Dictionary of scalar quantities to be saved during simulation.'''
        return self._scalar_quantities

    @abstractmethod
    def update_scalar_quantities( self, time ):
        '''
        Parameters
        ----------
            time : float
                Time at which to update.
        '''
        pass

    def print_scalar_quantities( self ):
        '''
        Print quantities saved in scalar_quantities to screen.
        '''
        sq_str = ''
        for key, val in self.scalar_quantities.items():
            sq_str += key + ': {:16.12f}'.format(val[0]) + '     '
        print(sq_str)

    def set_initial_conditions(self, comps_li, init_type, init_coords, init_params):
        '''For FE coefficients.
        
        Parameters
        ----------
            comps_li : list
                From parameters.yml ['fields']['init']['comps']. Specifies which components(s) of each field should be initialized.
        '''

        # initialize all field components
        if comps_li == 'all':
            comps_li = []
            for space_id in self.space_ids:
                if space_id in {'H1', 'L2'}:
                    comps_li += [[True]]
                elif space_id in {'Hcurl', 'Hdiv'}:
                    comps_li += [[True] * 3]

        for field, comps in zip(self.fields, comps_li):
            field.set_initial_conditions(self.DOMAIN, comps=comps, init_type=init_type, init_coords=init_coords, init_params=init_params)

    
class Maxwell( StruphyModel ):
    '''Maxwell's equations in vacuum, in Struphy normalization (c=1).
    
    Parameters
    ..........
        DR: Derham obj
            From struphy/feec/psydac_derham.Derham_build.

        DOMAIN: Domain obj
            From struphy/geometry/domain_3d.Domain.

        solver_params : list
            Each entry corresponds to one linear solver used in the model. 
            An entry is a dict with the solver parameters correpsonding to one solver, obtained from paramaters.yml.
    '''

    def __init__(self, DR, DOMAIN, *solver_params):

        from struphy.models.codes.propagators import StepMaxwell

        super().__init__(DR, DOMAIN, *solver_params, e_field='Hcurl', b_field='Hdiv')

        # Assemble necessary mass matrices 
        self.DR.assemble_M1()
        self.DR.assemble_M2()

        # Pointers to Stencil-/Blockvectors
        self._e = self.fields[0].vector
        self._b = self.fields[1].vector

        # Initialize propagators/integrators used in splitting substeps
        self._propagators += [StepMaxwell(self._e, self._b, DR, self.solver_params[0])]  

        # Scalar variables to be saved during simulation
        self._scalar_quantities['time'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_E'] = np.empty(1, dtype=float) 
        self._scalar_quantities['en_B'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_tot'] = np.empty(1, dtype=float)
    
    def update_scalar_quantities(self, time):
        self._scalar_quantities['time'][0] = time
        self._scalar_quantities['en_E'][0] = 1/2*self._e.dot(self.DR.M1.dot(self._e))
        self._scalar_quantities['en_B'][0] = 1/2*self._b.dot(self.DR.M2.dot(self._b))
        self._scalar_quantities['en_tot'][0] = self._scalar_quantities['en_E'][0] + self._scalar_quantities['en_B'][0]




        
    

    