from abc import ABCMeta, abstractmethod
import numpy as np


class StruphyModel(metaclass=ABCMeta):
    """
    Base class for all Struphy models.

    Parameters
    ----------
    params : dict
        Simulation parameters, see from :ref:`params_yml`.

    comm : mpi4py.MPI.Intracomm
        MPI communicator for parallel runs.

    species : dict
        The dynamical fields and kinetic species of the model. 

        Keys are either:

        a) the electromagnetic field/potential names (b_field=, e_field=) 
        b) the fluid species names (e.g. mhd=)
        c) the names of the kinetic species (e.g. electrons=, energetic_ions=)

        Corresponding values are:

        a) a space ID ("H1", "Hcurl", "Hdiv", "L2" or "H1vec"),
        b) a dict with key=variable_name (e.g. n, U, p, ...) and value=space ID ("H1", "Hcurl", "Hdiv", "L2" or "H1vec"),
        c) the type of particles ("Particles6D", "Particles5D", ...).

    Note
    ----
    All Struphy models are subclasses of ``StruphyModel`` and should be added to ``struphy/models/models.py``.  
    """

    def __init__(self, params, comm, **species):

        from struphy.models.setup import setup_domain_mhd, setup_electric_background, setup_derham

        from struphy.polar.basic import PolarVector
        from struphy.propagators.base import Propagator
        from struphy.propagators import propagators_fields, propagators_coupling, propagators_markers
        from struphy.psydac_api.basis_projection_ops import BasisProjectionOperators
        from struphy.psydac_api.mass import WeightedMassOperators

        self._params = params
        self._comm = comm
        self._species = species

        # initialize model variable dictionaries
        self._init_variable_dicts()

        # compute model units
        self._units, self._eq_params = self.model_units(
            self.params, verbose=True, comm=self._comm)

        # create domain, MHD equilibrium, background electric field
        self._domain, self._mhd_equil = setup_domain_mhd(
            params, units=self.units)
        self._electric_equil = setup_electric_background(params, self.domain)

        if comm.Get_rank() == 0:
            if 'mhd_equilibrium' in params:
                print('\nMHD EQUILIBRIUM:')
                print('type:'.ljust(25), self.mhd_equil.__class__.__name__)
                for key, val in self.mhd_equil.params.items():
                    print((key + ':').ljust(25), val)
            if 'electric_equilibrium' in params:
                print('\nELECTRIC EQUILIBRIUM:')
                print('type:', self.electric_equil.__class__.__name__)
                for key, val in self.electric_equil.params.items():
                    print((key + ':').ljust(25), val)

        # create discrete derham sequence
        dims_mask = params['grid']['dims_mask']
        if dims_mask is None:
            dims_mask = [True]*3

        self._derham = setup_derham(
            params['grid'], comm=comm, domain=self.domain, mpi_dims_mask=dims_mask)

        # create weighted mass operators
        self._mass_ops = WeightedMassOperators(
            self.derham, self.domain, eq_mhd=self.mhd_equil)

        # allocate memory for variables
        self._pointer = {}
        self._allocate_variables()

        # store plasma parameters
        if comm.Get_rank() == 0:
            self._pparams = self.print_plasma_params()
            print('\nOPERATOR ASSEMBLY:')
        else:
            self._pparams = self.print_plasma_params(verbose=False)

        # expose propagator modules
        self._prop = Propagator
        self._prop_fields = propagators_fields
        self._prop_coupling = propagators_coupling
        self._prop_markers = propagators_markers

        # set propagators base class attributes (available to all propagators)
        self.prop.derham = self.derham
        self.prop.domain = self.domain
        self.prop.mass_ops = self.mass_ops
        self.prop.basis_ops = BasisProjectionOperators(
            self.derham, self.domain, eq_mhd=self.mhd_equil)

        self._propagators = []
        self._scalar_quantities = {}

    @classmethod
    @abstractmethod
    def bulk_species(cls):
        '''Name of the bulk species of the plasma. Must be a key of self.fluid or self.kinetic, or None.'''
        pass

    @classmethod
    @abstractmethod
    def velocity_scale(cls):
        '''String that sets the velocity scale unit of the model. 
        Must be one of "alfvén", "cyclotron" or "light".'''
        pass

    @abstractmethod
    def update_scalar_quantities(self):
        ''' Specify an update rule for each item in scalar_quantities using :meth:`update_scalar`.
        '''
        pass

    @property
    def params(self):
        '''Model parameters from :code:`parameters.yml`.'''
        return self._params

    @property
    def pparams(self):
        '''Plasma parameters for each species.'''
        return self._pparams

    @property
    def eq_params(self):
        '''Parameters appearing in model equation due to Struphy normalization.
        '''
        return self._eq_params

    @property
    def comm(self):
        '''MPI communicator.'''
        return self._comm

    @property
    def species(self):
        '''Species dictionary.'''
        return self._species
    
    @property
    def pointer(self):
        '''Dictionary pointing to the data structures of the species (Stencil/BlockVector or "Particle" class).
        
        The keys are the keys from the "species" property. 
        In case of a fluid species, the keys are like "species_variable".'''
        return self._pointer

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
    def derham(self):
        '''3d Derham sequence, see :ref:`derham`.'''
        return self._derham

    @property
    def units(self):
        '''All Struphy units.
        '''
        return self._units

    @property
    def mass_ops(self):
        '''WeighteMassOperators object, see :ref:`mass_ops`.'''
        return self._mass_ops

    @property
    def prop(self):
        '''Class :class:`struphy.propagators.base.Propagator`.'''
        return self._prop

    @property
    def prop_fields(self):
        '''Module :mod:`struphy.propagators.propagators_fields`.'''
        return self._prop_fields

    @property
    def prop_coupling(self):
        '''Module :mod:`struphy.propagators.propagators_coupling`.'''
        return self._prop_coupling

    @property
    def prop_markers(self):
        '''Module :mod:`struphy.propagators.propagators_markers`.'''
        return self._prop_markers

    @property
    def propagators(self):
        '''A list of propagator instances for the model.'''
        return self._propagators

    @property
    def scalar_quantities(self):
        '''A dictionary of scalar quantities to be saved during the simulation.'''
        return self._scalar_quantities

    def add_propagator(self, prop_instance):
        '''Add a propagator to a Struphy model.

        Parameters
        ----------
            prop_instance : obj
                An instance of :class:`struphy.propagator.base.Propagator`.
        '''
        assert isinstance(prop_instance, self.prop)
        self._propagators += [prop_instance]

    def add_scalar(self, name):
        '''Add a scalar that should be saved during the simulation.

        Parameters
        ----------
            name : str
                Dictionary key of the scalar.
        '''
        assert isinstance(name, str)
        self._scalar_quantities[name] = np.empty(1, dtype=float)

    def update_scalar(self, name, value):
        '''Add a scalar that should be saved during the simulation.

        Parameters
        ----------
            name : str
                Dictionary key of the scalar.

            value : float
                Value to be saved.
        '''
        assert isinstance(name, str)
        assert isinstance(value, float)
        self._scalar_quantities[name][0] = value

    def integrate(self, dt, split_algo='LieTrotter'):
        """
        Advance the model by a time step dt.

        Parameters
        ----------
        dt : float
            Time step of time integration.

        split_algo : str
            Splitting algorithm. Currently available: "LieTrotter" and "Strang".
        """

        # first order in time
        if split_algo == 'LieTrotter':

            for propagator in self.propagators:
                propagator(dt)

        # second order in time
        elif split_algo == 'Strang':

            assert len(self.propagators) > 1

            for propagator in self.propagators:
                propagator(dt/2)

            for propagator in self.propagators[::-1]:
                propagator(dt/2)

        else:
            raise NotImplementedError(
                f'Splitting scheme {split_algo} not available.')

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

        dim_to_int = {'e1': 0, 'e2': 1, 'e3': 2, 'v1': 3, 'v2': 4, 'v3': 5}

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
            sq_str += key + ': {:14.11f}'.format(val[0]) + '   '
        print(sq_str)

    def initialize_from_params(self):
        """
        Set initial conditions for FE coefficients (electromagnetic and fluid) and markers according to parameter file.
        """

        # initialize em fields
        if len(self.em_fields) > 0:

            for key, val in self.em_fields.items():
                if 'params' not in key:
                    val['obj'].initialize_coeffs(
                        self.em_fields['params']['init'], domain=self.domain)

                    if self.comm.Get_rank() == 0:
                        init_type = self.em_fields['params']['init']['type']
                        print(f'EM field "{key}" was initialized with:')
                        print('type:'.ljust(25), init_type)

                        if init_type is None:
                            pass

                        elif type(init_type) == str:
                            init_types = [init_type]

                        elif type(init_type) == list:
                            init_types = init_type

                        else:
                            raise NotImplemented(
                                f'The type of initial condition must be null or str or list.')

                        if init_type is not None:

                            for _type in init_types:
                                print(_type, ':')
                                for key, val2 in self.em_fields['params']['init'][_type].items():
                                    print((key + ':').ljust(25), val2)

        # initialize fields
        if len(self.fluid) > 0:

            for species, val in self.fluid.items():

                for variable, subval in val.items():
                    if 'params' not in variable:
                        subval['obj'].initialize_coeffs(
                            val['params']['init'], domain=self.domain)

                if self.comm.Get_rank() == 0:
                    init_type = val['params']['init']['type']
                    print(f'Fluid species "{species}" was initialized with:')
                    print('type:'.ljust(25), init_type)

                    if init_type is None:
                        pass

                    elif type(init_type) == str:
                        init_types = [init_type]

                    elif type(init_type) == list:
                        init_types = init_type

                    else:
                        raise NotImplemented(
                            f'The type of initial condition must be null or str or list.')

                    if init_type is not None:

                        for _type in init_types:
                            print(_type, ':')
                            for key, val2 in val['params']['init'][_type].items():
                                print((key + ':').ljust(25), val2)

        # initialize particles
        if len(self.kinetic) > 0:

            for species, val in self.kinetic.items():

                if self.comm.Get_rank() == 0:
                    _type = val['params']['init']['type']
                    print(f'Kinetic species "{species}" was initialized with:')
                    print('type:'.ljust(25), _type)
                    if _type is not None:
                        for key, par in val['params']['init'][_type].items():
                            print((key + ':').ljust(25), par)

                val['obj'].draw_markers()
                val['obj'].mpi_sort_markers(do_test=True)

                typ = val['params']['markers']['type']
                if typ == 'full_f':
                    val['obj'].initialize_weights(val['params']['init'])
                elif typ == 'delta_f':
                    val['obj'].initialize_weights(val['params']['init'])
                elif typ == 'control_variate':
                    val['obj'].initialize_weights(
                        val['params']['init'], val['params']['background'])
                else:
                    raise NotImplementedError(
                        f'Type {typ} for distribution function is not known!')

                if val['space'] == 'Particles5D':
                    val['obj'].save_magnetic_moment(self.derham)

    def initialize_from_restart(self, data):
        """
        Set initial conditions for FE coefficients (electromagnetic and fluid) and markers from restart group in hdf5 files.

        Parameters
        ----------
        data : struphy.models.output_handling.DataContainer
            The data object that links to the hdf5 files.
        """

        # initialize em fields
        if len(self.em_fields) > 0:

            for key, val in self.em_fields.items():
                if 'params' not in key:
                    val['obj'].initialize_coeffs_from_restart_file(data.file)

        # initialize fields
        if len(self.fluid) > 0:

            for species, val in self.fluid.items():

                for variable, subval in val.items():
                    if 'params' not in variable:
                        subval['obj'].initialize_coeffs_from_restart_file(
                            data.file, species)

        # initialize particles
        if len(self.kinetic) > 0:

            for key, val in self.kinetic.items():
                val['obj'].draw_markers()
                val['obj']._markers[:, :] = data.file['restart/' + key][-1, :, :]

                # important: sets holes attribute of markers!
                val['obj'].mpi_sort_markers(do_test=True)

    def initialize_data_output(self, data, size):
        """
        Create datasets in hdf5 files according to model unknowns and diagnostics data.

        Parameters
        ----------
        data : struphy.models.output_handling.DataContainer
            The data object that links to the hdf5 files.

        size : int
            Number of MPI processes of the model run.

        Returns
        -------
        save_keys_all : list
            Keys of datasets which are saved during the simulation.

        save_keys_end : list
            Keys of datasets which are saved at the end of a simulation to enable restarts.
        """

        from psydac.linalg.stencil import StencilVector

        # save scalar quantities in group 'scalar/'
        for key, val in self.scalar_quantities.items():
            key_scalar = 'scalar/' + key
            data.add_data({key_scalar: val})

        # store grid_info only for runs with 512 ranks or smaller
        if size <= 512:
            data.file['scalar'].attrs['grid_info'] = self.derham.domain_array
        else:
            data.file['scalar'].attrs['grid_info'] = self.derham.domain_array[0]

        # save electromagentic fields/potentials data in group 'feec/'
        for key, val in self.em_fields.items():
            if 'params' not in key:
                key_field = 'feec/' + key
                key_field_restart = 'restart/' + key

                # in-place extraction of FEM coefficients from field.vector --> field.vector_stencil!
                val['obj'].extract_coeffs(update_ghost_regions=False)

                # save numpy array to be updated each time step.
                if isinstance(val['obj'].vector_stencil, StencilVector):
                    data.add_data({key_field: val['obj'].vector_stencil._data})
                    data.add_data(
                        {key_field_restart: val['obj'].vector_stencil._data})
                else:
                    for n in range(3):
                        key_component = key_field + '/' + str(n + 1)
                        key_component_restart = key_field_restart + \
                            '/' + str(n + 1)
                        data.add_data(
                            {key_component: val['obj'].vector_stencil[n]._data})
                        data.add_data(
                            {key_component_restart: val['obj'].vector_stencil[n]._data})

                # save field meta data
                data.file[key_field].attrs['space_id'] = val['obj'].space_id
                data.file[key_field].attrs['starts'] = val['obj'].starts
                data.file[key_field].attrs['ends'] = val['obj'].ends
                data.file[key_field].attrs['pads'] = val['obj'].pads

        # save fluid data in group 'feec/'
        for species, val in self.fluid.items():

            species_path = 'feec/' + species + '_'
            species_path_restart = 'restart/' + species + '_'

            for variable, subval in val.items():
                if 'params' not in variable:
                    key_field = species_path + variable
                    key_field_restart = species_path_restart + variable

                    # in-place extraction of FEM coefficients from field.vector --> field.vector_stencil!
                    subval['obj'].extract_coeffs(update_ghost_regions=False)

                    # save numpy array to be updated each time step.
                    if isinstance(subval['obj'].vector_stencil, StencilVector):
                        data.add_data(
                            {key_field: subval['obj'].vector_stencil._data})
                        data.add_data(
                            {key_field_restart: subval['obj'].vector_stencil._data})
                    else:
                        for n in range(3):
                            key_component = key_field + '/' + str(n + 1)
                            key_component_restart = key_field_restart + \
                                '/' + str(n + 1)
                            data.add_data(
                                {key_component: subval['obj'].vector_stencil[n]._data})
                            data.add_data(
                                {key_component_restart: subval['obj'].vector_stencil[n]._data})

                    # save field meta data
                    data.file[key_field].attrs['space_id'] = subval['obj'].space_id
                    data.file[key_field].attrs['starts'] = subval['obj'].starts
                    data.file[key_field].attrs['ends'] = subval['obj'].ends
                    data.file[key_field].attrs['pads'] = subval['obj'].pads

        # save kinetic data in group 'kinetic/'
        for key, val in self.kinetic.items():
            key_spec = 'kinetic/' + key
            key_spec_restart = 'restart/' + key

            data.add_data({key_spec_restart: val['obj']._markers})

            for key1, val1 in val['kinetic_data'].items():
                key_dat = key_spec + '/' + key1

                if isinstance(val1, dict):
                    for key2, val2 in val1.items():
                        key_f = key_dat + '/' + key2
                        data.add_data({key_f: val2})

                        dims = (len(key2) - 2)//3 + 1
                        for dim in range(dims):
                            data.file[key_f].attrs['bin_centers' + '_' + str(dim + 1)] = val['bin_edges'][key2][dim][:-1] + (
                                val['bin_edges'][key2][dim][1] - val['bin_edges'][key2][dim][0])/2

                else:
                    data.add_data({key_dat: val1})

        # keys to be saved at each time step and only at end (restart)
        save_keys_all = []
        save_keys_end = []

        for key in data.dset_dict:
            if 'restart' in key:
                save_keys_end.append(key)
            else:
                save_keys_all.append(key)

        return save_keys_all, save_keys_end

    ###################
    # Class methods :
    ###################

    @classmethod
    def model_units(cls, params, verbose=False, comm=None):
        """
        Return model units and print them to screen.

        Parameters
        ----------
        params : dict
            model parameters.

        verbose : bool, optional
            print model units to screen.

        comm : obj
            MPI communicator.

        Returns
        -------
        units_basic : dict
            Basic units for time, length, mass and magnetic field.

        units_der : dict
            Derived units for velocity, pressure, mass density and particle density.
        """

        from struphy.models.setup import derive_units

        # physics constants
        e = 1.602176634e-19  # elementary charge (C)
        mH = 1.67262192369e-27  # proton mass (kg)
        eps0 = 8.8541878128e-12  # vacuum permittivity (F/m)

        x_unit = params['units']['x']
        B_unit = params['units']['B']

        if comm is None:
            rank = 0
        else:
            rank = comm.Get_rank()

        if verbose and rank == 0:
            print('\nUNITS:')
            print(f'Unit of length:'.ljust(25),
                  '{:4.3e}'.format(x_unit) + ' m')

        # special case for model Maxwell (no plasma species)
        if cls.bulk_species() is None:
            if verbose and rank == 0:
                print(f'Unit of time:'.ljust(25),
                      '{:4.3e}'.format(x_unit / 299792458) + ' s')
                print(f'Unit of velocity:'.ljust(25),
                      '{:4.3e}'.format(299792458) + ' m/s')
                print(f'Unit of magnetic field:'.ljust(
                    25), '{:4.3e}'.format(B_unit) + ' T')
                print(f'Unit of electric field:'.ljust(25),
                      '{:4.3e}'.format(299792458*B_unit) + ' V/m')

            units = {}
            units['t'] = x_unit / 299792458
            units['x'] = x_unit
            units['B'] = B_unit

            eq_params = {}
        else:

            # look for bulk species in fluid OR kinetic parameter dictionaries
            if 'fluid' in params:
                if cls.bulk_species() in params['fluid']:
                    Z_bulk = params['fluid'][cls.bulk_species()
                                             ]['phys_params']['Z']
                    A_bulk = params['fluid'][cls.bulk_species()
                                             ]['phys_params']['A']
            else:
                Z_bulk = params['kinetic'][cls.bulk_species()
                                           ]['phys_params']['Z']
                A_bulk = params['kinetic'][cls.bulk_species()
                                           ]['phys_params']['A']

            # compute units
            units = derive_units(
                Z_bulk, A_bulk, params['units']['x'], params['units']['B'], params['units']['n'], cls.velocity_scale())

            if verbose and rank == 0:
                print(f'Unit of time:'.ljust(25),
                      '{:4.3e}'.format(units['t']) + ' s')
                print(f'Unit of velocity:'.ljust(25),
                      '{:4.3e}'.format(units['v']) + ' m/s')
                print(f'Unit of magnetic field:'.ljust(25),
                      '{:4.3e}'.format(units['B']) + ' T')
                print(f'Unit of particle density:'.ljust(25),
                      '{:4.3e}'.format(units['n']) + ' m⁻³')
                print(f'Unit of mass density:'.ljust(25),
                      '{:4.3e}'.format(units['rho']) + ' kg/m³')
                print(f'Unit of pressure:'.ljust(25),
                      '{:4.3e}'.format(units['p'] * 1e-5) + ' bar')

            # compute equation parameters arising from Struphy normalization
            eq_params = {}
            if 'fluid' in params:
                for species in params['fluid']:

                    Z = params['fluid'][species]['phys_params']['Z']
                    A = params['fluid'][species]['phys_params']['A']

                    # compute equation parameters
                    om_p = np.sqrt(units['n'] * (Z*e)**2 / (eps0 * A*mH))
                    om_c = Z*e * units['B'] / (A*mH)
                    eq_params[species] = {}
                    eq_params[species]['alpha_unit'] = om_p / om_c
                    eq_params[species]['epsilon_unit'] = 1. / \
                        (om_c * units['t'])

                    if verbose and rank == 0:
                        print('- ' + species + ':')
                        for key, val in eq_params[species].items():
                            print((key + ':').ljust(25), '{:4.3e}'.format(val))

            if 'kinetic' in params:
                for species in params['kinetic']:

                    Z = params['kinetic'][species]['phys_params']['Z']
                    A = params['kinetic'][species]['phys_params']['A']

                    # compute equation parameters
                    om_p = np.sqrt(units['n'] * (Z*e)**2 / (eps0 * A*mH))
                    om_c = Z*e * units['B'] / (A*mH)
                    eq_params[species] = {}
                    eq_params[species]['alpha_unit'] = om_p / om_c
                    eq_params[species]['epsilon_unit'] = 1. / \
                        (om_c * units['t'])

                    if verbose and rank == 0:
                        print('- ' + species + ':')
                        for key, val in eq_params[species].items():
                            print((key + ':').ljust(25), '{:4.3e}'.format(val))

        return units, eq_params

    ###################
    # Private methods :
    ###################

    def _init_variable_dicts(self):
        """
        Initialize em-fields, fluid and kinetic dictionaries for information on the model variables.
        """

        # electromagnetic fields, fluid and/or kinetic species
        self._em_fields = {}
        self._fluid = {}
        self._kinetic = {}

        # create dictionaries for each em-field/species and fill in space/class name and parameters
        for var_name, space in self.species.items():

            if isinstance(space, str):

                if space in {'H1', 'Hcurl', 'Hdiv', 'L2', 'H1vec'}:

                    assert 'em_fields' in self.params, 'Top-level key "em_fields" is missing in parameter file.'
                    self._em_fields[var_name] = {}
                    self._em_fields[var_name]['space'] = space
                    self._em_fields['params'] = self.params['em_fields']

                elif space in {'Particles6D', 'Particles5D'}:

                    assert 'kinetic' in self.params, 'Top-level key "kinetic" is missing in parameter file.'
                    assert var_name in self.params[
                        'kinetic'], f'Kinetic species {var_name} is missing in parameter file.'

                    self._kinetic[var_name] = {}
                    self._kinetic[var_name]['space'] = space
                    self._kinetic[var_name]['params'] = self.params['kinetic'][var_name]

                else:
                    raise ValueError(
                        f'The given value string {space} is not supported!')

            elif isinstance(space, dict):

                assert 'fluid' in self.params, 'Top-level key "fluid" is missing in parameter file.'
                assert var_name in self.params[
                    'fluid'], f'Fluid species {var_name} is missing in parameter file.'

                self._fluid[var_name] = {}

                for sub_var_name, sub_space in space.items():
                    self._fluid[var_name][sub_var_name] = {}
                    self._fluid[var_name][sub_var_name]['space'] = sub_space

                self._fluid[var_name]['params'] = self.params['fluid'][var_name]

            else:
                raise ValueError(f'Type {type(space)} not supported as value.')

    def _allocate_variables(self):
        """
        Allocate memory for model variables. 
        Creates FEM fields for em-fields and fluid variables and a particle class for kinetic species.
        """

        from struphy.psydac_api.fields import Field
        from struphy.pic import particles

        # allocate memory for FE coeffs of electromagnetic fields/potentials
        if 'em_fields' in self.params:

            for key, val in self.em_fields.items():

                if 'params' not in key:
                    val['obj'] = Field(key, val['space'], self.derham)
                    
                    self._pointer[key] = val['obj'].vector

        # allocate memory for FE coeffs of fluid variables
        if 'fluid' in self.params:

            for species, val in self.fluid.items():

                for variable, subval in val.items():

                    if 'params' not in variable:
                        subval['obj'] = Field(
                            variable, subval['space'], self.derham)
                        
                        self._pointer[species + '_' + variable] = subval['obj'].vector

        # marker arrays and plasma parameters of kinetic species
        if 'kinetic' in self.params:

            for species, val in self.kinetic.items():

                if self.params['kinetic'][species]['markers']['type'] in ['control_variate', 'delta_f']:
                    assert 'background' in self.params['kinetic'][species], \
                        f'If a control variate or delta-f method is used, a maxwellians background must be given!'

                kinetic_class = getattr(particles, val['space'])

                val['obj'] = kinetic_class(species,
                                           **val['params']['phys_params'],
                                           **val['params']['markers'],
                                           derham=self.derham,
                                           domain=self.domain,
                                           mhd_equil=self.mhd_equil,
                                           units_basic=self.units)
                
                self._pointer[species] = val['obj']

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

    def print_plasma_params(self, verbose=True):
        """
        Compute and print volume averaged plasma parameters for each species of the model.

        Global parameters:
        - plasma volume
        - transit length
        - magnetic field

        Species dependent parameters:
        - mass
        - charge
        - density
        - pressure
        - thermal energy kBT
        - Alfvén speed v_A
        - thermal speed v_th
        - thermal frequency Omega_th
        - cyclotron frequency Omega_c
        - plasma frequency Omega_p
        - Alfvèn frequency Omega_A
        - thermal Larmor radius rho_th
        - MHD length scale v_a/Omega_c
        - rho/L
        - alpha = Omega_p/Omega_c
        - epsilon = 1/(t*Omega_c)

        Returns
        -------
            pparams : dict
                Plasma parameters for each species.
        """

        from struphy.kinetic_background import maxwellians

        pparams = {}

        # physics constants
        e = 1.602176634e-19  # elementary charge (C)
        m_p = 1.67262192369e-27  # proton mass (kg)
        mu0 = 1.25663706212e-6  # magnetic constant (N*A^-2)
        eps0 = 8.8541878128e-12  # vacuum permittivity (F*m^-1)
        kB = 1.380649e-23  # Boltzmann constant (J*K^-1)

        # exit when there is not any plasma species
        if len(self.fluid) == 0 and len(self.kinetic) == 0:
            return

        # compute model units
        units, eq_params = self.model_units(
            self.params, verbose=True, comm=self.comm)

        # units affices for printing
        units_affix = {}
        units_affix['plasma volume'] = ' m³'
        units_affix['transit length'] = ' m'
        units_affix['magnetic field'] = ' T'
        units_affix['mass'] = ' kg'
        units_affix['charge'] = ' C'
        units_affix['density'] = ' m⁻³'
        units_affix['pressure'] = ' bar'
        units_affix['kBT'] = ' keV'
        units_affix['v_A'] = ' m/s'
        units_affix['v_th'] = ' m/s'
        units_affix['vth1'] = ' m/s'
        units_affix['vth2'] = ' m/s'
        units_affix['vth3'] = ' m/s'
        units_affix['Omega_th'] = ' Mrad/s'
        units_affix['Omega_c'] = ' Mrad/s'
        units_affix['Omega_p'] = ' Mrad/s'
        units_affix['Omega_A'] = ' Mrad/s'
        units_affix['rho_th'] = ' m'
        units_affix['v_A/Omega_c'] = ' m'
        units_affix['rho_th/L'] = ''
        units_affix['alpha'] = ''
        units_affix['epsilon'] = ''

        h = 1/20
        eta1 = np.linspace(h/2., 1.-h/2., 20)
        eta2 = np.linspace(h/2., 1.-h/2., 20)
        eta3 = np.linspace(h/2., 1.-h/2., 20)

        # global parameters
        # plasma volume (hat x^3)
        det_tmp = self.domain.jacobian_det(eta1, eta2, eta3)
        vol1 = np.mean(np.abs(det_tmp))
        # plasma volume (m⁻³)
        plasma_volume = vol1 * units['x']**3
        # transit length (m)
        transit_length = plasma_volume**(1/3)
        # magnetic field (T)
        B_tmp = self.mhd_equil.absB0(eta1, eta2, eta3)
        magnetic_field = np.mean(B_tmp * np.abs(det_tmp)) \
            / vol1 * units['B']
        B_max = np.max(B_tmp) * units['B']
        B_min = np.min(B_tmp) * units['B']

        if magnetic_field < 1e-14:
            magnetic_field = np.nan
            print('\n+++++++ WARNING +++++++ magnetic field is zero - set to nan !!')

        if verbose:
            print('\nPLASMA PARAMETERS:')
            print(f'Plasma volume:'.ljust(25),
                  '{:4.3e}'.format(plasma_volume) + units_affix['plasma volume'])
            print(f'Transit length:'.ljust(25),
                  '{:4.3e}'.format(transit_length) + units_affix['transit length'])
            print(f'Avg. magnetic field:'.ljust(25),
                  '{:4.3e}'.format(magnetic_field) + units_affix['magnetic field'])
            print(f'Max magnetic field:'.ljust(25),
                  '{:4.3e}'.format(B_max) + units_affix['magnetic field'])
            print(f'Min magnetic field:'.ljust(25),
                  '{:4.3e}'.format(B_min) + units_affix['magnetic field'])

        # species dependent parameters
        pparams = {}

        if len(self.fluid) > 0:

            for species, val in self.fluid.items():
                pparams[species] = {}
                # type
                pparams[species]['type'] = 'fluid'
                # mass (kg)
                pparams[species]['mass'] = val['params']['phys_params']['A'] * m_p
                # charge (C)
                pparams[species]['charge'] = val['params']['phys_params']['Z'] * e
                # density (m⁻³)
                pparams[species]['density'] = np.mean(self.mhd_equil.n0(
                    eta1, eta2, eta3) * np.abs(det_tmp)) * units['x']**3 / plasma_volume * units['n']
                # pressure (bar)
                pparams[species]['pressure'] = np.mean(self.mhd_equil.p0(
                    eta1, eta2, eta3) * np.abs(det_tmp)) * units['x']**3 / plasma_volume * units['p'] * 1e-5
                # thermal energy (keV)
                pparams[species]['kBT'] = pparams[species]['pressure'] * \
                    1e5 / pparams[species]['density'] / e * 1e-3

        if len(self.kinetic) > 0:

            eta1mg, eta2mg, eta3mg = np.meshgrid(
                eta1, eta2, eta3, indexing='ij')

            for species, val in self.kinetic.items():
                pparams[species] = {}
                # type
                pparams[species]['type'] = 'kinetic'
                # mass (kg)
                pparams[species]['mass'] = val['params']['phys_params']['A'] * m_p
                # charge (C)
                pparams[species]['charge'] = val['params']['phys_params']['Z'] * e

                # create temp kinetic object for (default) parameter extraction
                if 'background' in val['params']:
                    tmp_str = 'background'
                else:
                    tmp_str = 'init'
                tmp_type = val['params'][tmp_str]['type']
                tmp_params = val['params'][tmp_str][tmp_type]
                tmp = getattr(maxwellians, tmp_type)(**tmp_params)

                # density (m⁻³)
                pparams[species]['density'] = np.mean(tmp.n(
                    eta1mg, eta2mg, eta3mg) * np.abs(det_tmp)) * units['x']**3 / plasma_volume * units['n']
                # thermal speeds (m/s)
                vth = tmp.vth(eta1mg, eta2mg, eta3mg) * \
                    np.abs(det_tmp) * units['x']**3 / \
                    plasma_volume * units['v']
                thermal_speed = 0.
                for dir in range(val['obj'].vdim):
                    pparams[species]['vth' + str(dir + 1)] = np.mean(vth[dir])
                    thermal_speed += pparams[species]['vth' + str(dir + 1)]
                # TODO: here it is assumed that background density parameter is called "n",
                # and that background thermal speeds are called "vthn"; make this a convention?
                pparams[species]['v_th'] = thermal_speed / \
                    val['obj'].vdim
                # thermal energy (keV)
                pparams[species]['kBT'] = pparams[species]['mass'] * \
                    pparams[species]['v_th']**2 / e * 1e-3
                # pressure (bar)
                pparams[species]['pressure'] = pparams[species]['kBT'] * \
                    e * 1e3 * pparams[species]['density'] * 1e-5

        for species in pparams:
            # alfvén speed (m/s)
            pparams[species]['v_A'] = magnetic_field / np.sqrt(
                mu0 * pparams[species]['mass'] * pparams[species]['density'])
            # thermal speed (m/s)
            pparams[species]['v_th'] = np.sqrt(
                pparams[species]['kBT'] * 1e3 * e / pparams[species]['mass'])
            # thermal frequency (Mrad/s)
            pparams[species]['Omega_th'] = pparams[species]['v_th'] / \
                transit_length * 1e-6
            # cyclotron frequency (Mrad/s)
            pparams[species]['Omega_c'] = pparams[species]['charge'] * \
                magnetic_field / pparams[species]['mass'] * 1e-6
            # plasma frequency (Mrad/s)
            pparams[species]['Omega_p'] = np.sqrt(pparams[species]['density'] * (
                pparams[species]['charge'])**2 / eps0 / pparams[species]['mass']) * 1e-6
            # alfvén frequency (Mrad/s)
            pparams[species]['Omega_A'] = pparams[species]['v_A'] / \
                transit_length * 1e-6
            # Larmor radius (m)
            pparams[species]['rho_th'] = pparams[species]['v_th'] / \
                (pparams[species]['Omega_c'] * 1e6)
            # MHD length scale (m)
            pparams[species]['v_A/Omega_c'] = pparams[species]['v_A'] / \
                (np.abs(pparams[species]['Omega_c']) * 1e6)
            # dim-less ratios
            pparams[species]['rho_th/L'] = pparams[species]['rho_th'] / \
                transit_length

        if verbose:
            print('\nSPECIES PARAMETERS:')
            for species, ch in pparams.items():
                print(f'name:'.ljust(25), species)
                print(f'type:'.ljust(25), ch['type'])
                ch.pop('type')
                print(f'is bulk:'.ljust(25), species == self.bulk_species())
                for kinds, vals in ch.items():
                    print(kinds.ljust(25), '{:+4.3e}'.format(
                        vals), units_affix[kinds])
                print('------------------------------------')

        return pparams
