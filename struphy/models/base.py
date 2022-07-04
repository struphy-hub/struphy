from abc import ABCMeta, abstractmethod
import scipy.special as sp

from struphy.psydac_api.fields import Field
from struphy.pic.particles import Particles6D, Particles5D
from struphy.diagnostics.data_module import Data_container_psydac as Data_container


class StruphyModel( metaclass=ABCMeta ):
    '''Base class for all Struphy models.

    Parameters
    ----------
        DR : struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.

        DOMAIN : struphy.geometry.domain_3d.Domain
            All things mapping.

        solver_params : list[dict]
            Each entry corresponds to one linear solver used in the model. 
            An entry is a dict with the solver parameters correpsonding to one solver, obtained from paramaters.yml.

        kwargs : dict
            Keys are either a) the field names, then values are the space_ids ("H1", "Hcurl", "Hdiv", "L2" or "H1vec"), or
            b) the names of the kinetic species, then values are the marker parameters (dict).

    Note
    ----
        All Struphy models are subclasses of ``StruphyModel`` and should be added to ``struphy/models/codes/models.py``.  
    '''

    def __init__(self, DR, DOMAIN, *solver_params, **kwargs):

        self._field_names = []
        self._space_ids = []
        self._kinetic_names = []
        self._marker_params = []
        self._KIN = []

        for key, val in kwargs.items():

            # Kinetic species
            if isinstance(val, dict):
                self._kinetic_names += [key]
                self._marker_params += [val]

            # Field variables
            else:
                self._field_names += [key]
                self._space_ids += [val]

        self._DR = DR
        self._DOMAIN = DOMAIN
        self._solver_params = solver_params

        self._fields = []
        for name, space_id in zip(self._field_names, self._space_ids):
            self._fields += [Field(name, space_id, self.DR)]

        self._kinetic_species = []
        for name, params in zip(self._kinetic_names, self._marker_params):
            if params['type'] == 'fullorbit':
                self._kinetic_species += [Particles6D(name,
                                                      self._DOMAIN, params, self._DR.comm)]
            elif params['type'] == 'driftkinetic':
                self._kinetic_species += [Particles5D(name,
                                                      self._DOMAIN, params, self._DR.comm)]
            else:
                raise NotImplementedError('Marker type not implemented!')

    @property
    def names(self):
        '''List of FE variable names (str).'''
        return self._field_names

    @property
    def space_ids(self):
        '''List of 3d Derham space identifiers (str) corresponding to names.'''
        return self._space_ids

    @property
    def fields(self):
        '''List of Struphy fields, see :ref:`fields`.'''
        return self._fields

    @property
    def DR(self):
        '''3d Derham sequence, see :ref:`derham`.'''
        return self._DR

    @property
    def DOMAIN(self):
        '''Domain object, see :ref:`domains`.'''
        return self._DOMAIN

    @property
    def solver_params(self):
        '''List of dicts holding the solver parameters.'''
        return self._solver_params

    @property
    def kinetic_species(self):
        '''List of Struphy kinetic species, see :ref:`particles`.'''
        return self._kinetic_species

    @property
    def kinetic_params(self):
        '''List of kinetic parameters for the kinetic species.'''
        return self._marker_params

    @property
    @abstractmethod
    def propagators(self):
        '''List of :ref:`propagators` used in the time stepping of the model.'''
        pass

    @property
    @abstractmethod
    def scalar_quantities(self):
        '''Dictionary of scalar quantities to be saved during simulation. 
        Must be initialized as empty np.array of size 1, e.g.

        self._scalar_quantities['time'] = np.empty(1, dtype=float)'''
        pass

    @abstractmethod
    def update_scalar_quantities(self, time):
        '''
        Specify an update rule for each item in scalar_quantities.

        Parameters
        ----------
            time : float
                Time at which to update.
        '''
        pass

    def print_scalar_quantities(self):
        '''
        Print quantities saved in scalar_quantities to screen.
        '''
        sq_str = ''
        for key, val in self.scalar_quantities.items():
            sq_str += key + ': {:16.12f}'.format(val[0]) + '     '
        print(sq_str)

    def set_initial_conditions(self, fields_init, fields_params, particles_init, particles_params):
        '''For FE coefficients and marker weights.

        Parameters
        ----------
            fields_init : dict
                Basic info on field initial conditions, from parameters['fields']['init].

            fields_params : dict
                Parameters of field initial condition specified in field_init.

            kinetic_init : dict
                Basic info on kinetic initial conditions, from parameters['kinetic']['species_name']['init].

            kinetic_params : dict
                Parameters of kinetic initial conditions specified in kinetic_init.
        '''
        if fields_init is not None:
            init_type = fields_init['type']
            init_coords = fields_init['coords']
            comps_li = fields_init['comps']

            # initialize all field components
            if comps_li == 'all':
                comps_li = []
                for space_id in self.space_ids:
                    if space_id in {'H1', 'L2'}:
                        comps_li += [[True]]
                    elif space_id in {'Hcurl', 'Hdiv', 'H1vec'}:
                        comps_li += [[True] * 3]

            for field, comps in zip(self.fields, comps_li):
                field.set_initial_conditions(
                    self.DOMAIN, comps=comps, init_type=init_type, init_coords=init_coords, init_params=fields_params)

        if particles_init is not None:
            for species, init, param in zip(self.kinetic_species, particles_init, particles_params):
                species.set_initial_conditions(init, param)