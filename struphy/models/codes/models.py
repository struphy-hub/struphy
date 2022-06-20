from abc import ABCMeta, abstractmethod
import numpy as np
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
            Keys are either a) the field names, then values are the space_ids ("H1", "Hcurl", "Hdiv", "L2" or "H1^3"), or
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
                    elif space_id in {'Hcurl', 'Hdiv'}:
                        comps_li += [[True] * 3]

            for field, comps in zip(self.fields, comps_li):
                field.set_initial_conditions(
                    self.DOMAIN, comps=comps, init_type=init_type, init_coords=init_coords, init_params=fields_params)

        if particles_init is not None:
            for species, init, param in zip(self.kinetic_species, particles_init, particles_params):
                species.set_initial_conditions(init, param)


class Maxwell( StruphyModel ):
    '''Maxwell's equations in vacuum. 
    
    Normalization:

    .. math::

        c = \\frac{\hat \omega}{\hat k} = \\frac{\hat E}{\hat B}\,,

    where :math:`c` is the vacuum speed of light. Implemented equations:

    .. math::
    
        &\\frac{\partial \mathbf E}{\partial t} - \\nabla\\times\mathbf B = 0\,, 
        
        &\\frac{\partial \mathbf B}{\partial t} + \\nabla\\times\mathbf E = 0\,.

    Parameters
    ----------
        DR: Derham obj
            From struphy/psydac_api/psydac_derham.Derham_build.

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
        self._propagators = []
        self._propagators += [StepMaxwell(self._e,
                                          self._b, DR, self.solver_params[0])]

        # Scalar variables to be saved during simulation
        self._scalar_quantities = {}
        self._scalar_quantities['time'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_E'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_tot'] = np.empty(1, dtype=float)

    @property
    def propagators(self):
        return self._propagators

    @property
    def scalar_quantities(self):
        return self._scalar_quantities

    def update_scalar_quantities(self, time):
        self._scalar_quantities['time'][0] = time
        self._scalar_quantities['en_E'][0] = .5*self._e.dot(self.DR.M1.dot(self._e))
        self._scalar_quantities['en_B'][0] = .5*self._b.dot(self.DR.M2.dot(self._b))
        self._scalar_quantities['en_tot'][0] = self._scalar_quantities['en_E'][0] + \
                                               self._scalar_quantities['en_B'][0]


class LinearMHD( StruphyModel ):
    '''Linear ideal MHD with zero-flow equilibrium (:math:`\mathbf U_0 = 0`). 
    
    Normalization:

    .. math::

        TODO.

    Implemented equations:

    .. math::

        &\\frac{\partial \\tilde \\rho}{\partial t}+\\nabla\cdot(\\rho_0 \\tilde{\mathbf{U}})=0\,, 

        \\rho_0&\\frac{\partial \\tilde{\mathbf{U}}}{\partial t} + \\nabla \\tilde p
        =(\\nabla\\times \\tilde{\mathbf{B}})\\times\mathbf{B}_0 + \mathbf{J}_0\\times \\tilde{\mathbf{B}}
        \,, \qquad
        \mathbf{J}_0 = \\nabla\\times\mathbf{B}_0\,,

        &\\frac{\partial \\tilde p}{\partial t} + \\nabla\cdot(p_0 \\tilde{\mathbf{U}}) 
        + (\gamma-1)p_0\\nabla\cdot \\tilde{\mathbf{U}}=0\,,
        
        &\\frac{\partial \\tilde{\mathbf{B}}}{\partial t} - \\nabla\\times(\\tilde{\mathbf{U}} \\times \mathbf{B}_0)
        = 0\,.

    Parameters
    ----------
        DR: Derham obj
            From struphy/psydac_api/psydac_derham.Derham_build.

        DOMAIN: Domain obj
            From struphy/geometry/domain_3d.Domain.

        solver_params : list
            Each entry corresponds to one linear solver used in the model. 
            An entry is a dict with the solver parameters correpsonding to one solver, obtained from paramaters.yml.
    '''

    def __init__(self, DR, DOMAIN, *solver_params):
        pass


class LinearVlasovMaxwell( StruphyModel ):
    """
    Linearized Vlasov Maxwell model, has electric and magnetic fields, and electrons as particles

    Parameters
    ----------
        DR : Derham obj
            From struphy/feec/psydac_derham.Derham_build.

        DOMAIN : Domain obj
            From struphy/geometry/domain_3d.Domain.

        solver_params : list
            Each entry corresponds to one linear solver used in the model. 
            An entry is a dict with the solver parameters corresponding to one solver, obtained from parameters.yml.
    """

    def __init__(self, DR, DOMAIN, *solver_params, electron_markers, f_0_params):
        from struphy.kinetic_equil.kinetic_equil_6d import MaxwellHomogenSlab
        from struphy.models.codes.propagators import StepMaxwell
        from struphy.models.codes.lin_Vlasov_Maxwell import StepEWLinVlasovMaxwell

        super().__init__(DR, DOMAIN, *solver_params, efield='Hcurl',
                         bfield='Hdiv', electrons=electron_markers)

        # set kinetic equilibrium/background distribution function and set it for the electrons
        if f_0_params['type'] == 'Maxwell_homogen_slab':
            EQ_KINETIC = MaxwellHomogenSlab(f_0_params, DOMAIN)
            self.EQ_Kinetic = EQ_KINETIC
            self._kinetic_species[0].set_kinetic_equil(self.EQ_Kinetic)
        else:
            raise ValueError('Equilibrium not implemented!')

        # Assemble necessary mass matrices
        self.DR.assemble_M1()
        self.DR.assemble_M2()

        # Pointers to Stencil-/Blockvectors
        self._e = self.fields[0].vector
        self._b = self.fields[1].vector

        # Initialize propagators/integrators used in splitting substeps
        self._propagators = []
        self._propagators += [StepMaxwell(self._e,
                                          self._b, DR, self.solver_params[0])]
        self._propagators += [StepEWLinVlasovMaxwell(self.KIN.particles_loc, self._e, self.KIN.Np_loc, self.DR, f_0_params, self.solver_params[0])]

        # Scalar variables to be saved during simulation
        self._scalar_quantities = {}
        self._scalar_quantities['time'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_E'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_tot'] = np.empty(1, dtype=float)

    @property
    def propagators(self):
        return self._propagators

    @property
    def scalar_quantities(self):
        return self._scalar_quantities

    @property
    def KIN(self):
        """Dictionary with all the kinetic objects in them, keys are the names"""
        return self._KIN
