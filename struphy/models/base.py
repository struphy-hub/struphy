from abc import ABCMeta, abstractmethod
import scipy.special as sp
import numpy as np

from struphy.geometry import domains
from struphy.psydac_api.psydac_derham import Derham
from struphy.psydac_api.fields import Field
from struphy.pic import particles


class StruphyModel(metaclass=ABCMeta):
    '''Base class for all Struphy models.

    Parameters
    ----------
        params : dict
            Simulation parameters, see from :ref:`params_yml`.

        mpi_comm : mpi4py.MPI.Intracomm
            MPI communicator for parallel runs (=None for serial runs).

        kwargs : dict
            The dynamical fields and kinetic species of the model. Keys are either 
                * a) the field names, then values are the space IDs ("H1", "Hcurl", "Hdiv", "L2" or "H1vec"), OR
                * b) the names of the kinetic species, then values are the type of particles ("Particles6D", "Particles5D", ...).

    Note
    ----
        All Struphy models are subclasses of ``StruphyModel`` and should be added to ``struphy/models/models.py``.  
    '''

    def __init__(self, params, mpi_comm=None, **kwargs):

        # create domain (mapping from logical unit cube to physical domain)
        dom_type = params['geometry']['type']
        dom_params = params['geometry'][dom_type]

        domain_class = getattr(domains, dom_type)
        self._domain = domain_class(dom_params)

        # create 3d derham sequence
        Nel = params['grid']['Nel']             # Number of grid cells
        p = params['grid']['p']                 # spline degrees
        # Spline types (clamped vs. periodic)
        spl_kind = params['grid']['spl_kind']
        # Boundary conditions (Homogeneous Dirichlet or None)
        bc = params['grid']['bc']
        # Number of quadrature points per histopolation cell
        nq_pr = params['grid']['nq_pr']
        # Number of quadrature points per grid cell
        nq_el = params['grid']['nq_el']

        quad_order = [nq_el[0] - 1,
                      nq_el[1] - 1,
                      nq_el[2] - 1]

        self._derham = Derham(Nel, p, spl_kind, bc,
                              quad_order=quad_order,
                              nq_pr=nq_pr,
                              comm=mpi_comm)

        
        # create fields and kinetic species
        self._field_names = []
        self._field_ids = []
        self._field_params = params['fields']

        self._kinetic_names = []
        self._kinetic_ids = []
        self._kinetic_params = []

        for key, val in kwargs.items():

            # field variables
            if val in {'H1', 'Hcurl', 'Hdiv', 'L2', 'H1vec'}:
                self._field_names += [key]
                self._field_ids += [val]

            # kinetic variables
            elif val in {'Particles6D'}:
                self._kinetic_names += [key]
                self._kinetic_ids += [val]
                self._kinetic_params += [params['kinetic'][key]]
  
            else:
                raise ValueError('The given value string ' + str(val) + ' is not supported!')
                
        # create fields
        self._fields = []
        for name, ID in zip(self._field_names, self._field_ids):
            self._fields += [Field(name, ID, self._derham)]

        # create kinetic species
        self._kinetic_species = []
        for name, ID, k_params in zip(self._kinetic_names, self._kinetic_ids, self._kinetic_params):
            kinetic_class = getattr(particles, ID)

            self._kinetic_species += [kinetic_class(
                name, k_params['markers'], self._domain, self._derham.domain_array, self._derham.comm)]
                            
        # create time propagators list
        self._propagators = []
        
        # create dictionary for scalar quantities
        self._scalar_quantities = {}
        
        # create list for markers, distribution function and other kinetic data dicts
        self._kinetic_data = []
        self._bin_edges = []
        
        for species, k_params in zip(self._kinetic_species, self._kinetic_params):
            self._kinetic_data += [{}]
            self._bin_edges += [{}]
            
            # markers
            if 'n_markers' in k_params['save_data']:
                n_markers = k_params['save_data']['n_markers']
                if n_markers > 0:
                    self._kinetic_data[-1]['markers'] = np.zeros((n_markers, species.markers.shape[1]), dtype=float)
                
            # distribution function
            if 'f' in k_params['save_data']:
                slices = k_params['save_data']['f']['slices']
                n_bins = k_params['save_data']['f']['n_bins']
                ranges = k_params['save_data']['f']['ranges']
                
                if len(slices) > 0:
                    self.kinetic_data[-1]['f'] = {}
                    for i, key in enumerate(slices):
                        
                        assert ((len(key) - 2)/3).is_integer()
    
                        self._bin_edges[-1][key] = []
    
                        dims = (len(key) - 2)//3 + 1
    
                        for j in range(dims):
                            self._bin_edges[-1][key] += [np.linspace(ranges[i][j][0], ranges[i][j][1], n_bins[i][j] + 1)]
        
                        self._kinetic_data[-1]['f'][key] = np.zeros(n_bins[i], dtype=float)
            
            # other data (wave-particle power exchange, etc.)
            # TODO
                

    @property
    def derham(self):
        '''3d Derham sequence, see :ref:`derham`.'''
        return self._derham

    @property
    def domain(self):
        '''Domain object, see :ref:`avail_mappings`.'''
        return self._domain

    @property
    def field_names(self):
        '''List of FE variable names (str).'''
        return self._field_names

    @property
    def field_ids(self):
        '''List of 3d Derham space identifiers (str) corresponding to names.'''
        return self._field_ids

    @property
    def field_params(self):
        '''Field simulation parameters (dict), see section "fields" from :ref:`params_yml`.'''
        return self._field_params

    @property
    def fields(self):
        '''List of Struphy fields, see :ref:`fields`.'''
        return self._fields

    @property
    def kinetic_names(self):
        '''List of kinetic species names (str).'''
        return self._kinetic_names

    @property
    def kinetic_ids(self):
        '''List of kinetic identifiers (str) corresponding to classes in struphy.pic.particles.py'''
        return self._kinetic_ids

    @property
    def kinetic_params(self):
        '''List of parameters (dict) for each kinetic species, see section "kinetic" from :ref:`params_yml`.'''
        return self._kinetic_params

    @property
    def kinetic_species(self):
        '''List of Struphy kinetic species, see :ref:`particles`.'''
        return self._kinetic_species

    @property
    @abstractmethod
    def propagators(self):
        '''List of :ref:`propagators` used in the time stepping of the model.'''
        return self._propagators

    @property
    def scalar_quantities(self):
        '''Dictionary of scalar quantities to be saved during simulation. 
        Must be initialized as empty np.array of size 1::

            self._scalar_quantities['time'] = np.empty(1, dtype=float)'''
        return self._scalar_quantities
    
    @property
    def kinetic_data(self):
        '''Dictionary of kinetic data to be saved during simulation. Must contain numpy arrays as values.'''
        return self._kinetic_data
    
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
    
    def update_markers_to_be_saved(self):
        '''
        Writes markers with IDs that are supposed to be saved into corresponding array.
        '''
        
        for n, (species, k_params) in enumerate(zip(self._kinetic_species, self._kinetic_params)):
            
            if 'n_markers' in k_params['save_data']:
                n_mks_save = k_params['save_data']['n_markers']
                if n_mks_save > 0:
            
                    markers_on_proc = np.logical_and(species.markers[:, -1] >= 0., 
                                                     species.markers[:, -1] < n_mks_save)

                    n_markers_on_proc = np.count_nonzero(markers_on_proc)

                    self._kinetic_data[n]['markers'][:] = -1.
                    self._kinetic_data[n]['markers'][:n_markers_on_proc] = species.markers[markers_on_proc]
            
    def update_distr_function(self):
        '''
        Writes distribution function slices that are supposed to be saved into corresponding array.
        '''
        
        dim_to_int = {'e1' : 0, 'e2' : 1, 'e3' : 2, 'vx' : 3, 'vy' : 4, 'vz' : 5}
        
        for n, species in enumerate(self._kinetic_species):
            
            for slic, edges in self._bin_edges[n].items():
                
                dims = (len(slic) - 2)//3 + 1
                comps = [slic[3*i:3*i + 2] for i in range(dims)]
                
                components = [False]*6
                
                for comp in comps:
                    components[dim_to_int[comp]] = True
                
                self._kinetic_data[n]['f'][slic][:] = species.binning(components, edges)
        
    def print_scalar_quantities(self):
        '''
        Print quantities saved in scalar_quantities to screen.
        '''
        sq_str = ''
        for key, val in self._scalar_quantities.items():
            sq_str += key + ': {:16.12f}'.format(val[0]) + '     '
        print(sq_str)

    def set_initial_conditions(self):
        '''
        Set initial conditions for FE coefficients and marker weights.
        '''

        # initialize fields
        if len(self._fields) > 0:

            comps_li = self._field_params['init']['comps']

            if self._field_params['init']['coords'] == 'physical':
                dom_arg = self._domain
            else:
                dom_arg = None

            for field, comps in zip(self._fields, comps_li):
                field.initialize_coeffs(comps, self._field_params['init'], domain=dom_arg)

        # initialize particles
        if len(self._kinetic_species) > 0:
            
            for species, params in zip(self._kinetic_species, self._kinetic_params):

                # set specific initial condition for some particles
                if 'initial' in params['markers']['loading']:
                    specific_markers = params['markers']['loading']['initial']

                    for i in range(len(specific_markers)):
                        for j in range(6):
                            if specific_markers[i][j] is not None:
                                self._kinetic_species[-1]._markers[i, j] = specific_markers[i][j]
                
                # do MPI sort
                species.mpi_sort_markers(do_test=True)
                
                # compute weights
                species.initialize_weights(params['background'], params['perturbations'])
                
        # initialize scalar quantities
        self.update_scalar_quantities(0.)
        
        # initialize markers to be saved
        self.update_markers_to_be_saved()
        
        # initialize binned distribution function
        self.update_distr_function()
