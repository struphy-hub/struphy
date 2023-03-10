from abc import ABCMeta, abstractmethod
import numpy as np

from struphy.geometry import domains
from struphy.psydac_api.psydac_derham import Derham
from struphy.psydac_api.fields import Field
from struphy.pic import particles
from struphy.fields_background.mhd_equil.base import CartesianMHDequilibrium, LogicalMHDequilibrium
from struphy.fields_background.mhd_equil import equils
from struphy.fields_background.electric_equil import analytical as analytical_electric
from struphy.psydac_api.mass import WeightedMassOperators


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
            * a) the electromagnetic field/potential names, then values are the space IDs ("H1", "Hcurl", "Hdiv", "L2" or "H1vec"), OR
            * b) the fluid species names, then the value is a dict with key=var_name (n, U, p, ...) and value=space ID ("H1", "Hcurl", "Hdiv", "L2" or "H1vec"), OR
            * c) the names of the kinetic species, then values are the type of particles ("Particles6D", "Particles5D", ...).

    Note
    ----
    All Struphy models are subclasses of ``StruphyModel`` and should be added to ``struphy/models/models.py``.  
    '''

    def __init__(self, params, mpi_comm=None, **kwargs):

        # domain and MHD equilibrium (latter is None if there is no MHD equilibrium)
        self._domain, self._mhd_equil = setup_domain_mhd(params)

        # electric equilibrium
        if 'electric_equilibrium' in params:
            equil_params = params['electric_equilibrium']
            electric_equil_class = getattr(
                analytical_electric, equil_params['type'])
            self._electric_equil = electric_equil_class(
                equil_params[equil_params['type']], self.domain)
        else:
            self._electric_equil = None

        # plasma size
        self._size_params = {}
        h = 1/20
        eta1 = np.linspace(h/2., 1.-h/2., 20)
        eta2 = np.linspace(h/2., 1.-h/2., 20)
        eta3 = np.linspace(h/2., 1.-h/2., 20)
        self._size_params['plasma volume [x\u0302³]'] = np.mean(
            np.abs(self.domain.jacobian_det(eta1, eta2, eta3)))
        self._size_params['minor radius [x\u0302]'] = 'no minor radius'
        self._size_params['transit length [x\u0302]'] = self.size_params['plasma volume [x\u0302³]']**(
            1/3)
        self._size_params['transit k [x\u0302⁻¹]'] = 2*np.pi / \
            self._size_params['transit length [x\u0302]']
        self._size_params['eps_key'] = 'rho*k'

        # minor radius
        if self.mhd_equil is not None:
            if 'a' in self.mhd_equil.params:
                self._size_params['minor radius [x\u0302]'] = self.mhd_equil.params['a']
                self._size_params['transit length [x\u0302]'] = self._size_params['minor radius [x\u0302]']
                self._size_params['transit k [x\u0302⁻¹]'] = 2 * \
                    np.pi / self._size_params['transit length [x\u0302]']
                self._size_params['eps_key'] = 'rhostar'

            # average B-field strength (Tesla)
            eta1 = np.linspace(0., 1., 20)
            eta2 = np.linspace(0., 1., 20)
            eta3 = np.linspace(0., 1., 20)

            # shift away point from pole!
            if self.mhd_equil.domain.pole:
                eta1[0] += 1e-10

            self._size_params['B_abs [B\u0302]'] = np.mean(
                self.mhd_equil.absB0(eta1, eta2, eta3))

        # 3d Derham sequence
        Nel = params['grid']['Nel']  # Number of grid cells
        p = params['grid']['p']  # spline degrees
        # spline types (clamped vs. periodic)
        spl_kind = params['grid']['spl_kind']
        # boundary conditions (Homogeneous Dirichlet or None)
        bc = params['grid']['bc']
        # Number of quadrature points per histopolation cell
        nq_pr = params['grid']['nq_pr']
        # Number of quadrature points per grid cell for L^2
        nq_el = params['grid']['nq_el']
        # C^k smoothness at eta_1=0 for polar domains
        polar_ck = params['grid']['polar_ck']

        quad_order = [nq_el[0] - 1,
                      nq_el[1] - 1,
                      nq_el[2] - 1]

        self._derham = Derham(Nel, p, spl_kind, bc,
                              quad_order=quad_order,
                              nq_pr=nq_pr,
                              comm=mpi_comm,
                              with_projectors=True,
                              polar_ck=polar_ck,
                              domain=self.domain)

        # weighted mass operators
        self._mass_ops = WeightedMassOperators(
            self.derham, self.domain, eq_mhd=self.mhd_equil)

        # electromagnetic fields, fluid and/or kinetic species
        self._em_fields = {}
        self._fluid = {}
        self._kinetic = {}

        nem = 0  # number of electromagnetic fields/potentials
        nf = []  # numbers of variables for each fluid species
        nk = 0  # number of kinetic species
        for key, val in kwargs.items():

            if isinstance(val, str):

                if val in {'H1', 'Hcurl', 'Hdiv', 'L2', 'H1vec'}:

                    assert 'em_fields' in params, 'Top-level key "em_fields" is missing in parameter file.'
                    self._em_fields[key] = {}
                    self._em_fields[key]['space'] = val
                    nem += 1

                elif val in {'Particles6D', 'Particles5D'}:

                    assert 'kinetic' in params, 'Top-level key "kinetic" is missing in parameter file.'
                    self._kinetic[key] = {}
                    self._kinetic[key]['space'] = val
                    nk += 1

                else:
                    raise ValueError('The given value string ' +
                                     str(val) + ' is not supported!')

            elif isinstance(val, dict):

                assert 'fluid' in params, 'Top-level key "fluid" is missing in parameter file.'
                self._fluid[key] = {}
                nf += [0]
                for variable, space in val.items():
                    self._fluid[key][variable] = {}
                    self._fluid[key][variable]['space'] = space
                    nf[-1] += 1

            else:
                raise ValueError(f'Type {type(val)} not supported as value.')

        # FE coeffs of electromagnetic fields/potentials
        if 'em_fields' in params:

            self._em_fields['params'] = params['em_fields']

            for n, (key, val) in enumerate(self.em_fields.items()):

                if 'params' not in key:
                    field = Field(key, val['space'], self.derham)
                    val['obj'] = field

        # FE coeffs and plasma parameters of fluid variables
        if 'fluid' in params:

            for nfi, (species, val) in zip(nf, self.fluid.items()):

                assert species in params['fluid']
                val['params'] = params['fluid'][species]

                # Z, M, kBT, beta = params['fluid'][species]['attributes'].values(
                # )

                # val['plasma_params'] = plasma_params(Z, M,
                #                                     kBT, beta,
                #                                     self.size_params)

                for n, (variable, subval) in enumerate(val.items()):

                    if 'params' not in variable:
                        field = Field(variable, subval['space'], self.derham)
                        subval['obj'] = field

        # marker arrays and plasma parameters of kinetic species
        if 'kinetic' in params:

            for species, val in self.kinetic.items():

                assert species in params['kinetic']
                val['params'] = params['kinetic'][species]

                if params['kinetic'][species]['markers']['type'] in ['control_variate', 'delta_f']:
                    assert 'background' in params['kinetic'][species], \
                        f'If a control variate or delta-f method is used, a analytical background must be given!'

                kinetic_class = getattr(particles, val['space'])

                val['obj'] = kinetic_class(species,
                                           **val['params']['markers'],
                                           comm=self.derham.comm,
                                           domain_array=self.derham.domain_array,
                                           domain=self.domain)

                #Z, M, kBT, beta = val['params']['attributes'].values()
                # val['plasma_params'] = plasma_params(
                #    Z, M, kBT, beta, self.size_params)

                # for storing markers
                n_markers = val['params']['save_data']['n_markers']
                assert n_markers <= val['obj'].n_mks
                if n_markers > 0:
                    val['kinetic_data'] = {}
                    val['kinetic_data']['markers'] = np.zeros(
                        (n_markers, val['obj'].markers.shape[1]), dtype=float)

                # for storing the distribution function
                if 'f' in val['params']['save_data']:
                    slices = val['params']['save_data']['f']['slices']
                    n_bins = val['params']['save_data']['f']['n_bins']
                    ranges = val['params']['save_data']['f']['ranges']

                    val['kinetic_data']['f'] = {}
                    val['bin_edges'] = {}
                    if len(slices) > 0:
                        for i, sli in enumerate(slices):

                            assert ((len(sli) - 2)/3).is_integer()
                            val['bin_edges'][sli] = []
                            dims = (len(sli) - 2)//3 + 1
                            for j in range(dims):
                                val['bin_edges'][sli] += [np.linspace(
                                    ranges[i][j][0], ranges[i][j][1], n_bins[i][j] + 1)]
                            val['kinetic_data']['f'][sli] = np.zeros(
                                n_bins[i], dtype=float)

                # other data (wave-particle power exchange, etc.)
                # TODO

        # create time propagators list
        self._propagators = []

        # create dictionary for scalar quantities
        self._scalar_quantities = {}

        # print info to screen
        if mpi_comm.Get_rank() == 0:
            print('GRID parameters:')
            print(f'number of elements      : {self.derham.Nel}')
            print(f'spline degrees          : {self.derham.p}')
            print(f'periodic bcs            : {self.derham.spl_kind}')
            print(f'hom. Dirichlet bc       : {self.derham.bc}')
            _gl_quad_pts_l2 = [self.derham.quad_order[0] + 1,
                               self.derham.quad_order[1] + 1,
                               self.derham.quad_order[2] + 1]
            print(f'GL quad pts (L2)        : {_gl_quad_pts_l2}')
            print(f'GL quad pts (hist)      : {self.derham.nq_pr}')
            print(
                f'N-spline indices rank 0 : {self.derham.index_array_N[0]}\n')

            print('DOMAIN parameters:')
            print(f'domain type       : {self.domain.__class__.__name__}')
            print(f'domain parameters :')
            for key, val in self.domain.params_map.items():
                if key not in {'cx', 'cy', 'cz'}:
                    print(key, ': ', val)
            print('')

            n_longest_key = len(max(self.size_params.keys(), key=len))
            for key, val in self.size_params.items():
                if key != 'eps_key':
                    diff = n_longest_key - len(key)

                    key_str = key
                    for i in range(diff):
                        key_str += ' '

                    if isinstance(val, float):
                        print(key_str + ' :', '{:16.13f}'.format(val))
                    else:
                        print(key_str + ' :', val)

            if hasattr(self, 'print_units') and 'model_units' in params:
                self.print_units(params['model_units'])

            #print('PLASMA parameters:')
            # print('size:')
            # print('-----')
            # for key, val in self.size_params.items():
            #    if key != 'eps_key':
            #        print(key + ': ', val)
            #print('\nelectromagnetic fields/potentials:')
            # print('----------------------------------')
            # for key, val in self.em_fields.items():
            #    if 'params' not in key:
            #        print(key + ': ' + val['space'])
            #print('\nfluid species:')
            # print('--------------')
            # for species, val in self.fluid.items():
            #    print(species + ':')
            #    for variable, subval in val.items():
            #        if 'params' not in variable:
            #            print(variable + ': ' + subval['space'])
            #    for p, pv in val['plasma_params'].items():
            #        print(p + ': ', pv)
            #print('\nkinetic species:')
            # print('----------------')
            # for species, val in self.kinetic.items():
            #    print(species + ': ' + val['space'] + ' with '
            #          + str(val['obj'].n_mks) + ' markers initialized, shape='
            #          + str(val['obj'].markers.shape) + ' on rank 0.')
            #    for p, pv in val['plasma_params'].items():
            #        print(p + ': ', pv)
            print('')

    @property
    def derham(self):
        '''3d Derham sequence, see :ref:`derham`.'''
        return self._derham

    @property
    def domain(self):
        '''Domain object, see :ref:`avail_mappings`.'''
        return self._domain

    @property
    def mhd_equil(self):
        '''MHD equilibrium object, see :ref:`mhd_equil`.'''
        return self._mhd_equil

    @property
    def mass_ops(self):
        '''WeighteMassOperators object, see :ref:`mass_ops`.'''
        return self._mass_ops

    @property
    def electric_equil(self):
        '''Eelctric equilibrium object, see :ref:`electric_equil`.'''
        return self._electric_equil

    @property
    def em_fields(self):
        '''Dictionary of electromagnetic field/potential variables.'''
        return self._em_fields

    @property
    def fluid(self):
        '''Dictionary of fluid species.'''
        return self._fluid

    @property
    def kinetic(self):
        '''Dictionary of kinetic species.'''
        return self._kinetic

    @property
    def size_params(self):
        '''Dictionary of plasma size and magnetic field strength.'''
        return self._size_params

    @property
    @abstractmethod
    def propagators(self):
        '''List of :ref:`propagators` used in the time stepping of the model.'''
        return self._propagators

    @property
    def scalar_quantities(self):
        '''Dictionary of scalar quantities to be saved during simulation. 
        Must be initialized as empty np.array of size 1::

            The time series self._scalar_quantities['time'] = np.empty(1, dtype=float) must be contained.'''
        return self._scalar_quantities

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

        for val in self.kinetic.values():

            n_mks_save = val['params']['save_data']['n_markers']
            if n_mks_save > 0:
                markers_on_proc = np.logical_and(
                    val['obj'].markers[:, -1] >= 0., val['obj'].markers[:, -1] < n_mks_save)
                n_markers_on_proc = np.count_nonzero(markers_on_proc)
                val['kinetic_data']['markers'][:] = -1.
                val['kinetic_data']['markers'][:n_markers_on_proc] = val['obj'].markers[markers_on_proc]

    def update_distr_function(self):
        '''
        Writes distribution function slices that are supposed to be saved into corresponding array.
        '''

        dim_to_int = {'e1': 0, 'e2': 1, 'e3': 2, 'vx': 3, 'vy': 4, 'vz': 5}

        for val in self.kinetic.values():

            if 'f' in val['params']['save_data']:

                for slice_i, edges in val['bin_edges'].items():

                    dims = (len(slice_i) - 2)//3 + 1
                    comps = [slice_i[3*i:3*i + 2] for i in range(dims)]
                    components = [False]*6

                    for comp in comps:
                        components[dim_to_int[comp]] = True

                    val['kinetic_data']['f'][slice_i][:] = val['obj'].binning(
                        components, edges, self.domain)

    def print_scalar_quantities(self):
        '''
        Print quantities saved in scalar_quantities to screen.
        '''
        sq_str = ''
        for key, val in self._scalar_quantities.items():
            sq_str += key + ': {:16.12f}'.format(val[0]) + '     '
        print(sq_str)

    def initialize_from_params(self):
        '''
        Set initial conditions for FE coefficients (electromagnetic and fluid) and markers.
        '''

        # initialize em fields
        if len(self.em_fields) > 0:

            for key, val in self.em_fields.items():
                if 'params' not in key:
                    val['obj'].initialize_coeffs(
                        self.em_fields['params']['init'], domain=self.domain)

        # initialize fields
        if len(self.fluid) > 0:

            for val in self.fluid.values():

                for variable, subval in val.items():
                    if 'params' not in variable:
                        subval['obj'].initialize_coeffs(
                            val['params']['init'], domain=self.domain)

        # initialize particles
        if len(self.kinetic) > 0:

            for val in self.kinetic.values():
                val['obj'].mpi_sort_markers(do_test=True)

                if val['params']['markers']['type'] == 'full_f':
                    val['obj'].initialize_weights(val['params']['init'],
                                                  self.domain)
                elif val['params']['markers']['type'] == 'delta_f':
                    val['obj'].initialize_weights(val['params']['init'],
                                                  self.domain)
                elif val['params']['markers']['type'] == 'control_variate':
                    val['obj'].initialize_weights(val['params']['init'], self.domain,
                                                  val['params']['background'])
                else:
                    typ = val['params']['markers']['type']
                    raise NotImplementedError(
                        f'Type {typ} for distribution function is not known!')

                #if val['space'] == 'Particles5D':
                #    val['obj'].save_magnetic_moment(
                #        self.derham, self.derham.P['0'](self.mhd_equil.absB0))
            
    def initialize_from_restart(self, file):
        '''
        Load restart data for FE coefficients (electromagnetic and fluid) and markers from restart group in hdf5 output files.
        '''
        
        # initialize em fields
        if len(self.em_fields) > 0:

            for key, val in self.em_fields.items():
                if 'params' not in key:
                    val['obj'].initialize_coeffs_from_restart_file(file)
                    
        # initialize fields
        if len(self.fluid) > 0:

            for species, val in self.fluid.items():

                for variable, subval in val.items():
                    if 'params' not in variable:
                        subval['obj'].initialize_coeffs_from_restart_file(file, species)
                        
        # initialize particles
        if len(self.kinetic) > 0:
            
            for key, val in self.kinetic.items():
                val['obj']._markers[:, :] = file['restart/' + key][-1, :, :]
                
                # important: sets holes attribute of markers!
                val['obj'].mpi_sort_markers(do_test=True)

            
###########################
#  Helper functions
###########################
def setup_domain_mhd(params):
    """
    Creates the domain object and MHD equilibrium for a given parameter file.
    
    Parameters
    ----------
    params : dict
        The full simulation parameter dictionary.
        
    Returns
    -------
    domain : struphy.geometry.base.Domain
        The Struphy domain object for evaluating the mapping F : [0, 1]^3 --> R^3 and the corresponding metric coefficients.
        
    mhd : struphy.fields_background.base.MHDequilibrium
        The ideal MHD equilibrium object.
    """
    
    # MHD equilibrium given (load equilibrium first, then set domain)
    if 'mhd_equilibrium' in params:
        
        mhd_type = params['mhd_equilibrium']['type']
        mhd_class = getattr(equils, mhd_type)
        mhd = mhd_class(**params['mhd_equilibrium'][mhd_type])

        # for logical MHD equilibria, the domain comes with the equilibrium
        if isinstance(mhd, LogicalMHDequilibrium):
            domain = mhd.domain
        
        # for cartesian MHD equilibria, the domain can be chosen idependently
        else:
            dom_type = params['geometry']['type']
            dom_class = getattr(domains, dom_type)

            if dom_type == 'Tokamak':
                domain = dom_class(**params['geometry'][dom_type], equilibrium=mhd)
            else:
                domain = dom_class(**params['geometry'][dom_type])
                
            # set domain attribute in mhd object
            mhd.domain = domain

    # no MHD equilibrium (load domain)
    else:

        dom_type = params['geometry']['type']
        dom_class = getattr(domains, dom_type)
        domain = dom_class(**params['geometry'][dom_type])
        
        mhd = None
            
    return domain, mhd
    