from abc import ABCMeta, abstractmethod
import scipy.special as sp
import numpy as np

from struphy.geometry import domains
from struphy.psydac_api.psydac_derham import Derham
from struphy.psydac_api.fields import Field
from struphy.pic import particles
from struphy.models.pre_processing import plasma_params
from struphy.fields_background.mhd_equil import analytical as analytical_mhd
from struphy.fields_background.mhd_equil import numerical as numerical_mhd
from struphy.fields_background.electric_equil import analytical as analytical_electric


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

        # domain (mapping from logical unit cube to physical domain)
        dom_type = params['geometry']['type']
        dom_params = params['geometry'][dom_type]

        domain_class = getattr(domains, dom_type)
        self._domain = domain_class(dom_params)

        # mhd equilibrium
        if 'mhd_equilibrium' in params:
            equil_params = params['mhd_equilibrium']

            if equil_params['type'] == 'analytical':
                mhd_equil_class = getattr(analytical_mhd, equil_params['name'])
                self._mhd_equil = mhd_equil_class(
                    equil_params[equil_params['name']])

                # set mapping for equilibrium object
                self._mhd_equil.domain = self.domain

            elif equil_params['type'] == 'numerical':
                mhd_equil_class = getattr(numerical_mhd, equil_params['name'])
                self._mhd_equil = mhd_equil_class(
                    equil_params[equil_params['name']])

                # reset domain
                self._domain = self.mhd_equil.domain

            else:
                raise TypeError('type parameter must be either "analytical" or "numerical", is {0}'.format(
                    equil_params['type']))
        else:
            self._mhd_equil = None

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
        h = 1/100
        eta1 = np.linspace(h/2., 1.-h/2., 100)
        eta2 = np.linspace(h/2., 1.-h/2., 100)
        eta3 = np.linspace(h/2., 1.-h/2., 100)
        self._size_params['plasma volume [m^3]'] = np.mean(
            np.abs(self.domain.jacobian_det(eta1, eta2, eta3)))
        self._size_params['minor radius [m]'] = 'No minor radius.'
        self._size_params['transit length [m]'] = self.size_params['plasma volume [m^3]']**(
            1/3)
        self._size_params['transit k [1/m]'] = 2*np.pi / \
            self._size_params['transit length [m]']
        self._size_params['eps_key'] = 'rho*k'

        # minor radius
        if self.mhd_equil is not None:
            if 'a' in self.mhd_equil.params:
                self._size_params['minor radius [m]'] = self.mhd_equil.params['a']
                self._size_params['transit length [m]'] = self._size_params['minor radius [m]']
                self._size_params['transit k [1/m]'] = 2 * \
                    np.pi / self._size_params['transit length [m]']
                self._size_params['eps_key'] = 'rhostar'

            # average B-field strength (Tesla)
            eta1 = np.linspace(0., 1., 100)
            eta2 = np.linspace(0., 1., 100)
            eta3 = np.linspace(0., 1., 100)
            self._size_params['B_abs [T]'] = np.mean(
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

            comps = params['em_fields']['init']['comps']
            assert len(
                comps) == nem, 'Lengths of ["em_fields"]["init"]["comps"] lists do not correspond to number of fields.'

            for n, (key, val) in enumerate(self.em_fields.items()):

                if 'params' not in key:
                    field = Field(key, val['space'], self.derham)
                    val['obj'] = field

                    assert len(comps[n]) == isinstance(field.nbasis, tuple)*1 + isinstance(
                        field.nbasis, list)*3, f'Wrong length of ["init"]["comps"] list for {key}.'
                    val['init_comps'] = comps[n]

        # FE coeffs and plasma parameters of fluid variables
        if 'fluid' in params:

            for nfi, (species, val) in zip(nf, self.fluid.items()):

                assert species in params['fluid']
                val['params'] = params['fluid'][species]

                Z, M, kBT, beta = params['fluid'][species]['attributes'].values(
                )
                comps = params['fluid'][species]['init']['comps']
                assert len(comps) == nfi, \
                    f'Lengths of ["fluid"]["species"]["attributes"] lists do not correspond to number of fluid variables of species {species}.'
                val['plasma_params'] = plasma_params(Z, M,
                                                     kBT, beta,
                                                     self.size_params)

                for n, (variable, subval) in enumerate(val.items()):

                    if 'params' not in variable:
                        field = Field(variable, subval['space'], self.derham)
                        subval['obj'] = field

                        assert len(comps[n]) == isinstance(field.nbasis, tuple)*1 + isinstance(field.nbasis, list)*3, \
                            f'Wrong length of ["init"]["comps"] list for {variable}.'
                        subval['init_comps'] = comps[n]

        # marker arrays and plasma parameters of kinetic species
        if 'kinetic' in params:

            for species, val in self.kinetic.items():

                assert species in params['kinetic']
                val['params'] = params['kinetic'][species]

                kinetic_class = getattr(particles, val['space'])
                val['obj'] = kinetic_class(species, val['params']['markers'],
                                           self.domain, self.derham.domain_array, self.derham.comm)

                Z, M, kBT, beta = val['params']['attributes'].values()
                val['plasma_params'] = plasma_params(
                    Z, M, kBT, beta, self.size_params)

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
            print(f'number of elements : {self.derham.Nel}')
            print(f'spline degrees     : {self.derham.p}')
            print(f'periodic bcs       : {self.derham.spl_kind}')
            print(f'hom. Dirichlet bc  : {self.derham.bc}')
            print(f'GL quad pts (L2)   : {self.derham.quad_order}')
            print(f'GL quad pts (hist) : {self.derham.nq_pr}')
            print(
                f'MPI indices for N-splines on rank 0: {self.derham.index_array_N[0]}\n')

            print('DOMAIN parameters:')
            print(f'domain type: {dom_type}')
            print(f'domain parameters: {dom_params}\n')

            print('PLASMA parameters:')
            print('size:')
            print('-----')
            for key, val in self.size_params.items():
                if key != 'eps_key':
                    print(key + ': ', val)
            print('\nelectromagnetic fields/potentials:')
            print('----------------------------------')
            for key, val in self.em_fields.items():
                if 'params' not in key:
                    print(key + ': ' + val['space'])
            print('\nfluid species:')
            print('--------------')
            for species, val in self.fluid.items():
                print(species + ':')
                for variable, subval in val.items():
                    if 'params' not in variable:
                        print(variable + ': ' + subval['space'])
                for p, pv in val['plasma_params'].items():
                    print(p + ': ', pv)
            print('\nkinetic species:')
            print('----------------')
            for species, val in self.kinetic.items():
                print(species + ': ' + val['space'] + ' with '
                      + str(val['obj'].n_mks) + ' markers initialized, shape='
                      + str(val['obj'].markers.shape) + ' on rank 0.')
                for p, pv in val['plasma_params'].items():
                    print(p + ': ', pv)
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

            self._scalar_quantities['time'] = np.empty(1, dtype=float)'''
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
                for slic, edges in val['bin_edges'].items():

                    dims = (len(slic) - 2)//3 + 1
                    comps = [slic[3*i:3*i + 2] for i in range(dims)]
                    components = [False]*6

                    for comp in comps:
                        components[dim_to_int[comp]] = True

                    val['kinetic_data']['f'][slic][:] = val['obj'].binning(
                        components, edges)

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

        # initialize em fields
        if len(self.em_fields) > 0:

            if self.em_fields['params']['init']['coords'] == 'physical':
                dom_arg = self.domain
            else:
                dom_arg = None

            for key, val in self.em_fields.items():
                if 'params' not in key:
                    val['obj'].initialize_coeffs(
                        val['init_comps'], self.em_fields['params']['init'], domain=dom_arg)

        # initialize fields
        if len(self.fluid) > 0:

            for val in self.fluid.values():

                if val['params']['init']['coords'] == 'physical':
                    dom_arg = self.domain
                else:
                    dom_arg = None

                for variable, subval in val.items():
                    if 'params' not in variable:
                        subval['obj'].initialize_coeffs(
                            subval['init_comps'], val['params']['init'], domain=dom_arg)

        # initialize particles
        if len(self.kinetic) > 0:

            for val in self.kinetic.values():
                val['obj'].mpi_sort_markers(do_test=True)
                if val['params']['markers']['type'] == 'full_f':
                    val['obj'].initialize_weights(val['params']['init'])
                elif val['params']['markers']['type'] == 'delta_f':
                    val['obj'].initialize_weights_delta_f(
                        val['params']['init'])
                else:
                    typ = val['params']['markers']['type']
                    raise NotImplementedError(
                        f'Type {typ} for distribution function is not known!')

                if val['space'] == 'Particles5D':
                    val['obj'].initialize_magnetic_moments(
                        self.derham, self.mhd_equil)

            self.update_markers_to_be_saved()
            self.update_distr_function()

        self.update_scalar_quantities(0.)
