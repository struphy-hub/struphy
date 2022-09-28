#!/usr/bin/env python3

from psydac.linalg.stencil import StencilVector
from psydac.fem.tensor import FemField

from struphy.initial import perturbations
from struphy.initial import analytic

from struphy.psydac_api.utilities import apply_essential_bc_to_array

import numpy as np


class Field:
    """
    Initializes a field variable (i.e. its FE coefficients) in memory and creates a method for assigning initial condition.

    Parameters
    ----------
        name : str
            Field's key to be used for saving in the hdf5 file.

        space_id : str
            Space identifier for the field (H1, Hcurl, Hdiv, L2 or H1vec).

        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.
    """
    
    def __init__(self, name, space_id, derham):

        self._name = name
        self._space_id = space_id
        self._derham = derham

        # Initialize field in memory
        self._space = getattr(derham, derham.spaces_dict[space_id])
        
        self._field = FemField(self._space)
        self._vector = self._field.coeffs

        # Global indices of each process, and paddings
        if isinstance(self._vector, StencilVector):
            self._gl_s = [self._vector.starts]
            self._gl_e = [self._vector.ends]
            self._pads = [self._vector.pads]
        else:
            self._gl_s = [comp.starts for comp in self._vector]
            self._gl_e = [comp.ends for comp in self._vector]
            self._pads = [comp.pads for comp in self._vector]

    @property
    def name(self):
        """ Name of the field in DATA container.
        """
        return self._name

    @property
    def space(self):
        """ Discrete space of the field (Psydac object).
        """ 
        return self._space

    @property
    def space_id(self):
        """ Continuous space fo the field (string).
        """
        return self._space_id

    @property
    def derham(self):
        """ 3d Derham complex.
        """
        return self._derham

    @property
    def field(self):
        """ Psydac Femfield.
        """
        return self._field

    @property
    def vector(self):
        """ Local finite element coefficients (Stencil- or Blockvector).
        """
        return self._vector

    @vector.setter
    def vector(self, value):
        self._vector = value

    @property
    def starts(self):
        """ Global start indices.
        """
        return self._gl_s

    @property
    def ends(self):
        """ Global end indices (add +1 when indexing).
        """
        return self._gl_e

    @property
    def pads(self):
        """ Paddings for ghost regions.
        """
        return self._pads

    def set_initial_conditions(self, domain, comps, init_params):
        """
        Sets the initial conditions for self.vector.

        Parameters
        ----------
            domain: struphy.geometry.domains
                All things mapping.

            comps: list
                Booleans that specify whether field component has non-zero initial conditions (True).

            init_params: dict
                Parameters of initial condition, see from :ref:`params_yml`.
        """

        rank = self._derham.comm.Get_rank()
        
        if rank == 0: print(f'Setting initial conditions for {self.name} in {self.space_id} ...')

        # Set initial conditions for each component
        assert isinstance(comps, list)

        self._init_type = init_params['type']
        self._init_params = init_params[self.init_type]
        init_coords = init_params['coords']

        if self.init_type == 'noise':

            # Set white noise FE coefficients
            if self.space_id in {'H1', 'L2'}:
                if comps[0]:
                    self._add_noise()

            elif self.space_id in {'Hcurl', 'Hdiv', 'H1vec'}:
                for n, comp in enumerate(comps):
                    if comp:
                        self._add_noise(n=n)
        
        elif 'ModesSin' in self.init_type or 'ModesCos' in self.init_type:

            # Get callable(s) for specified init type
            _fun_tmp = [None] * len(comps)
            for n, comp in enumerate(comps):
                if comp:
                    fun_class = getattr(perturbations, self.init_type)
                    _fun_tmp[n] = fun_class(*list(self._init_params.values()))

            # Pullback callable and project
            self._fun = []
            if self.space_id == 'H1':
                self._fun += [PulledPform(init_coords,
                                           _fun_tmp, domain, '0_form')]
                self._vector[:] = self.derham.P0(self._fun[0]).coeffs[:]

            elif self.space_id == 'Hcurl':
                self._fun += [PulledPform(init_coords,
                                           _fun_tmp, domain, '1_form_1')]
                self._fun += [PulledPform(init_coords,
                                           _fun_tmp, domain, '1_form_2')]
                self._fun += [PulledPform(init_coords,
                                           _fun_tmp, domain, '1_form_3')]
                _coeffs = self.derham.P1(self._fun).coeffs
                self._vector[0][:] = _coeffs[0][:]
                self._vector[1][:] = _coeffs[1][:]
                self._vector[2][:] = _coeffs[2][:]

            elif self.space_id == 'Hdiv':
                self._fun += [PulledPform(init_coords,
                                           _fun_tmp, domain, '2_form_1')]
                self._fun += [PulledPform(init_coords,
                                           _fun_tmp, domain, '2_form_2')]
                self._fun += [PulledPform(init_coords,
                                           _fun_tmp, domain, '2_form_3')]
                _coeffs = self.derham.P2(self._fun).coeffs
                self._vector[0][:] = _coeffs[0][:]
                self._vector[1][:] = _coeffs[1][:]
                self._vector[2][:] = _coeffs[2][:]

            elif self.space_id == 'L2':
                self._fun += [PulledPform(init_coords,
                                           _fun_tmp, domain, '3_form')]
                self._vector[:] = self.derham.P3(self._fun[0]).coeffs[:]
                
            elif self.space_id == 'H1vec':
                self._fun += [PulledPform(init_coords,
                                           _fun_tmp, domain, 'vector_1')]
                self._fun += [PulledPform(init_coords,
                                           _fun_tmp, domain, 'vector_2')]
                self._fun += [PulledPform(init_coords,
                                           _fun_tmp, domain, 'vector_3')]
                _coeffs = self.derham.P0vec(self._fun).coeffs
                self._vector[0][:] = _coeffs[0][:]
                self._vector[1][:] = _coeffs[1][:]
                self._vector[2][:] = _coeffs[2][:]
                
        else:
            
            fun_class = getattr(analytic, self.init_type)
            funs = fun_class(self.init_params, domain)
            
            if self.space_id == 'H1':
                self._vector[:] = self.derham.P0(getattr(funs, self.name)).coeffs[:]
                
            elif self.space_id == 'Hcurl':
                _coeffs = self.derham.P1(getattr(funs, self.name)).coeffs
                self._vector[0][:] = _coeffs[0][:]
                self._vector[1][:] = _coeffs[1][:]
                self._vector[2][:] = _coeffs[2][:]
                
            elif self.space_id == 'Hdiv':
                _coeffs = self.derham.P2(getattr(funs, self.name)).coeffs
                self._vector[0][:] = _coeffs[0][:]
                self._vector[1][:] = _coeffs[1][:]
                self._vector[2][:] = _coeffs[2][:]
                
            elif self.space_id == 'L2':
                self._vector[:] = self.derham.P3(getattr(funs, self.name)).coeffs[:]
                
            elif self.space_id == 'H1vec':
                _coeffs = self.derham.P0vec(getattr(funs, self.name)).coeffs
                self._vector[0][:] = _coeffs[0][:]
                self._vector[1][:] = _coeffs[1][:]
                self._vector[2][:] = _coeffs[2][:]
                
        # apply boundary conditions
        apply_essential_bc_to_array(self.space_id, self._vector, self.derham.bc)
        
        self._vector.update_ghost_regions()

        if rank == 0: print('Done.')

    @property
    def init_type(self):
        """ Type of initial condition.
        """
        return self._init_type

    @property
    def init_params(self):
        """ Parameters of initial condition.
        """
        return self._init_params

    def _add_noise(self, n=None):
        """ Add noise to a vector component where init_comps==True, otherwise leave at zero.
        """

        self._direction = self._init_params['direction']
        self._ampsize = self._init_params['amp']

        if n == None:
            _shape_w_pads = self._vector[:].shape
            _shape = (self._gl_e[0][0] + 1 - self._gl_s[0][0], self._gl_e[0]
                      [1] + 1 - self._gl_s[0][1], self._gl_e[0][2] + 1 - self._gl_s[0][2])
        else:
            _shape_w_pads = self._vector[n][:].shape
            _shape = (self._gl_e[n][0] + 1 - self._gl_s[n][0], self._gl_e[n]
                      [1] + 1 - self._gl_s[n][1], self._gl_e[n][2] + 1 - self._gl_s[n][2])

        if self._direction == 'x':
            _amps = np.random.rand(_shape_w_pads[0]) * self._ampsize
            for j in range(_shape[1]):
                for k in range(_shape[2]):
                    if n == None:
                        self._vector[:, self._gl_s[0][1] +
                                     j, self._gl_s[0][2] + k] = _amps
                    else:
                        self._vector[n][:, self._gl_s[n][1] +
                                        j, self._gl_s[n][2] + k] = _amps

        elif self._direction == 'y':
            _amps = np.random.rand(_shape_w_pads[1]) * self._ampsize
            for j in range(_shape[0]):
                for k in range(_shape[2]):
                    if n == None:
                        self._vector[self._gl_s[0][0] + j,
                                     :, self._gl_s[0][2] + k] = _amps
                    else:
                        self._vector[n][self._gl_s[n][0] + j,
                                        :, self._gl_s[n][2] + k] = _amps

        elif self._direction == 'z':
            _amps = np.random.rand(_shape_w_pads[2]) * self._ampsize
            for j in range(_shape[0]):
                for k in range(_shape[1]):
                    if n == None:
                        self._vector[self._gl_s[0][0] + j,
                                     self._gl_s[0][1] + k, :] = _amps
                    else:
                        self._vector[n][self._gl_s[n][0] + j,
                                        self._gl_s[n][1] + k, :] = _amps

        elif self._direction == 'xy':
            _amps = np.random.rand(
                _shape_w_pads[0], _shape_w_pads[1]) * self._ampsize
            for j in range(_shape[2]):
                if n == None:
                    self._vector[:, :, self._gl_s[0][2] + j] = _amps
                else:
                    self._vector[n][:, :, self._gl_s[n][2] + j] = _amps

        elif self._direction == 'xz':
            _amps = np.random.rand(
                _shape_w_pads[0], _shape_w_pads[2]) * self._ampsize
            for j in range(_shape[1]):
                if n == None:
                    self._vector[:, self._gl_s[0][1] + j, :] = _amps
                else:
                    self._vector[n][:, self._gl_s[n][1] + j, :] = _amps

        elif self._direction == 'yz':
            _amps = np.random.rand(
                _shape_w_pads[1], _shape_w_pads[2]) * self._ampsize
            for j in range(_shape[0]):
                if n == None:
                    self._vector[self._gl_s[0][0] + j, :, :] = _amps
                else:
                    self._vector[n][self._gl_s[n][0] + j, :, :] = _amps

        elif self._direction == 'xyz':
            _amps = np.random.rand(
                _shape_w_pads[0], _shape_w_pads[1], _shape_w_pads[2]) * self._ampsize
            if n == None:
                self._vector[:, :, :] = _amps
            else:
                self._vector[n][:, :, :] = _amps

        else:
            raise ValueError('Invalid direction for noise.')


class PulledPform:
    """
    Construct callable (component of) p-form on logical domain (unit cube).

    Depending on the dimension of eta1 either point-wise, tensor-product, slice plane or general (see :ref:`struphy.geometry.map_eval.prepare_args`).
    
    Parameters
    ----------
        coords : str
            From which coordinate representation to pull, either 'logical' or 'physical'.

        fun : list
            Callable function components. Has to be length 3 for 1- and 2-forms, length 1 otherwise.

        domain: struphy.geometry.domains
            All things mapping.

        form : str
            Which form to pull: '0_form', '1_form_1', '1_form_2', '1_form_3', '2_form_1', '2_form_2', '2_form_3', '3_form'.

    Returns
    -------
        f : array[float]
            Array holding the values.
    """

    def __init__(self, coords, fun, domain, form):

        assert len(fun) == 1 or len(fun) == 3

        self._fun = []
        for f in fun:
            if f is None:
                def f_zero(x, y, z): return 0*x
                self._fun += [f_zero]
            else:
                assert callable(f)
                self._fun += [f]

        self._coords = coords
        self._domain = domain
        self._form = form

        # define which component of the field is evaluated (=0 for scalar fields)
        if len(self._fun) == 1:
            self._comp = 0
        else:
            self._comp = int(self._form[-1]) - 1
            
        assert isinstance(self._fun, list)

    def __call__(self, eta1, eta2, eta3):
        """ Evaluate the component of the p-form specified in self._form.
        """

        if self._coords == 'logical':
            f = self._fun[self._comp](eta1, eta2, eta3)
        elif self._coords == 'physical':
            self._domain.pull(self._fun, eta1, eta2, eta3, self._form)
        else:
            raise ValueError(
                'Coordinates to be used for p-form pullback not properly specified.')

        return f
