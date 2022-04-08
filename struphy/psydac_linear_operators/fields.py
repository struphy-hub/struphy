#!/usr/bin/env python3

from psydac.linalg.stencil import StencilVector
from psydac.linalg.block import BlockVector

from struphy.geometry.domain_3d import prepare_args
from struphy.analytic_funcs.fourier import Modes_sin, Modes_cos

import numpy as np


class Field_init:
    '''Initializes a field variable (i.e. its FE coefficients) in memory and assigns the initial condition.'''

    def __init__(self, name, space, comps, init_type, init_coords, init_params, DR, DOMAIN):
        '''
        Parameters
        ----------
            name: str
                Key to be used in the hdf5 file, specified in the parameters.yml file by the user.

            space: str
                Space identifier for the field (H1, Hcurl, Hdiv or L2), specified in the parameters.yml file by the user.

            comps: 3-list
                Booleans that specify whether field component has non-zero initial conditions (True).

            init_type: str
                Type of initial condition, specified in the parameters.yml file by the user.

            init_coords: str
                In which coordinate system the initial condition is given (logical, physical or norm_logical), specified in the parameters.yml file by the user.

            init_params: dict
                Parameters of initial condition, specified in the parameters.yml file by the user.

            DR: obj
                From struphy/psydac_api/fields.Field_init.

            DOMAIN: obj
                From struphy/geometry/domain_3d.Domain.
        '''

        self._name = name
        self._space_cont = space
        self._init_type = init_type
        self._init_params = init_params

        # Initialize field in memory
        if space == 'H1':
            self._space = DR.V0
            self._vector = StencilVector(self._space.vector_space)
        elif space == 'Hcurl':
            self._space = DR.V1
            self._vector = BlockVector(self._space.vector_space, [
                StencilVector(comp) for comp in self._space.vector_space])
        elif space == 'Hdiv':
            self._space = DR.V2
            self._vector = BlockVector(self._space.vector_space, [
                StencilVector(comp) for comp in self._space.vector_space])
        elif space == 'L2':
            self._space = DR.V3
            self._vector = StencilVector(self._space.vector_space)
        else:
            raise ValueError('Space for field not properly defined.')

        # Global indices of each process, and paddings
        if isinstance(self._vector, StencilVector):
            self._gl_s = [self._vector.starts]
            self._gl_e = [self._vector.ends]
            self._pads = [self._vector.pads]
        else:
            self._gl_s = [comp.starts for comp in self._vector]
            self._gl_e = [comp.ends for comp in self._vector]
            self._pads = [comp.pads for comp in self._vector]

        # Set initial conditions for each component
        assert isinstance(comps, list)

        if init_type == 'noise':

            # Set white noise FE coefficients
            if space in {'H1', 'L2'}:
                if comps[0]:
                    self._add_noise()

            elif space in {'Hcurl', 'Hdiv'}:
                for n, comp in enumerate(comps):
                    if comp:
                        self._add_noise(n=n)

            self._vector.update_ghost_regions()

        else:

            # Contruct callable
            _fun_tmp = [None] * len(comps)
            #_key_tmp = []
            for n, comp in enumerate(comps):
                # _key_tmp += [self._space_cont + '_' + str(n)] # string to identtify comp in pullback/transformation
                if comp:
                    _fun_tmp[n] = self._get_callable_from_params(n)

            # Pullback callable and project
            if space == 'H1':
                self._fun = Pulled_0form(init_coords, _fun_tmp, DOMAIN)
                self._vector[:] = DR.P0(self._fun[0]).coeffs[:]

            elif space == 'Hcurl':
                self._fun = Pulled_1form(init_coords, _fun_tmp, DOMAIN)
                _coeffs = DR.P1(self._fun).coeffs
                self._vector[0][:] = _coeffs[0][:]
                self._vector[1][:] = _coeffs[1][:]
                self._vector[2][:] = _coeffs[2][:]

            elif space == 'Hdiv':
                self._fun = Pulled_2form(init_coords, _fun_tmp, DOMAIN)
                _coeffs = DR.P2(self._fun).coeffs
                self._vector[0][:] = _coeffs[0][:]
                self._vector[1][:] = _coeffs[1][:]
                self._vector[2][:] = _coeffs[2][:]

            elif space == 'L2':
                self._fun = Pulled_3form(init_coords, _fun_tmp, DOMAIN)
                self._vector[:] = DR.P3(self._fun[0]).coeffs[:]

            self._vector.update_ghost_regions()

        print(f'Field "{self._name}" initialized in space {self._space_cont}.')

    @property
    def name(self):
        '''Name of the field in DATA container.'''
        return self._name

    @property
    def space(self):
        '''Discrete space of the field.'''
        return self._space

    @property
    def space_cont(self):
        '''Continuous space fo the field.'''
        return self._space_cont

    @property
    def vector(self):
        '''Local finite element coefficients (Stencil- or Blockvector).'''
        return self._vector

    @vector.setter
    def vector(self, value):
        self._vector = value

    @property
    def starts(self):
        '''Global start indices.'''
        return self._gl_s

    @property
    def ends(self):
        '''Global end indices (add +1 when indexing).'''
        return self._gl_e

    @property
    def pads(self):
        '''Paddings for ghost regions.'''
        return self._pads

    @property
    def init_type(self):
        '''Type of initial condition.'''
        return self._init_type

    @property
    def init_params(self):
        '''Parameters of initial condition-'''
        return self._init_params

    def _add_noise(self, n=None):
        '''Add noise to a vector component where init_comps==True, otherwise leave at zero.'''

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

    def _get_callable_from_params(self, n):
        '''Construct callable initial condition from input parameters.

        Parameters
        ----------
            n : int
                Is 1, 2, or 3, the component of the vector (1 for scalar fields). 
                This parameter can be used to give different init conds to different components.'''

        if self._init_type == 'modes_k':

            kind = self._init_params['kind']
            k1s = self._init_params['k1']
            k2s = self._init_params['k2']
            k3s = self._init_params['k3']
            amps = self._init_params['amp']

            # Instantiate callable
            if kind == 'sin':
                fun = Modes_sin(k1s, k2s, k3s, amps)
            elif kind == 'cos':
                fun = Modes_cos(k1s, k2s, k3s, amps)
            else:
                raise ValueError('Invalid type for modes_k.')

        return fun


class Pulled_0form:
    '''Construct callable 0-form on logical domain (unit cube).

    Depending on the dimension of eta1 either point-wise, tensor-product, slice plane or general (see struphy/geometry/domain_3d.prepare_args).

    Returns a list of one np.array holding the values.'''

    def __init__(self, coords, fun, DOMAIN):
        '''
        Parameters
        ----------
            coords : str
                From which coordinate representation to pull, either 'logical', 'physical' or 'norm_logical'.

            fun : 1-list
                Callable function.

            DOMAIN : obj
                From struphy/geometry/domain_3d.Domain.
        '''

        assert len(fun) == 1

        self._fun = []
        for f in fun:
            if f == None:
                def f(x, y, z): return 0.
            self._fun += [f]

        self._coords = coords
        self._DOMAIN = DOMAIN

    def __call__(self, eta1, eta2, eta3):

        if self._coords == 'logical':
            E1, E2, E3, is_sparse_meshgrid = prepare_args(eta1, eta2, eta3)
            f = np.array(self._fun[0](E1, E2, E3))

        elif self._coords == 'physical':
            f = self._DOMAIN.pull(self._fun, eta1, eta2, eta3, '0_form')

        elif self._coords == 'norm_logical':
            f = self._DOMAIN.transformation(
                self._fun, eta1, eta2, eta3, 'norm_to_0')

        else:
            raise ValueError(
                'Coordinates to be used for 0-form pullback not properly specified.')

        return [f]


class Pulled_1form:
    '''Construct callable 1-form on logical domain (unit cube).

    Depending on the dimension of eta1 either point-wise, tensor-product, slice plane or general (see struphy/geometry/domain_3d.prepare_args).

    Returns a list of three np.arrays holding the values.'''

    def __init__(self, coords, fun, DOMAIN):
        '''
        Parameters
        ----------
            coords : str
                From which coordinate representation to pull: 'logical', 'physical' or 'norm_logical'.

            fun : 3-list
                Callable components.

            DOMAIN : obj
                From struphy/geometry/domain_3d.Domain.
        '''

        assert len(fun) == 3

        self._fun = []
        for f in fun:
            if f == None:
                def f(x, y, z): return 0.
            self._fun += [f]

        self._coords = coords
        self._DOMAIN = DOMAIN

    def __call__(self, eta1, eta2, eta3):

        if self._coords == 'logical':
            E1, E2, E3, is_sparse_meshgrid = prepare_args(eta1, eta2, eta3)
            f1 = np.array(self._fun[0](E1, E2, E3))
            f2 = np.array(self._fun[1](E1, E2, E3))
            f3 = np.array(self._fun[2](E1, E2, E3))

        elif self._coords == 'physical':
            f1 = self._DOMAIN.pull(self._fun, eta1, eta2, eta3, '1_form_1')
            f2 = self._DOMAIN.pull(self._fun, eta1, eta2, eta3, '1_form_2')
            f3 = self._DOMAIN.pull(self._fun, eta1, eta2, eta3, '1_form_3')

        elif self._coords == 'norm_logical':
            f1 = self._DOMAIN.transformation(
                self._fun, eta1, eta2, eta3, 'norm_to_1_1')
            f2 = self._DOMAIN.transformation(
                self._fun, eta1, eta2, eta3, 'norm_to_1_2')
            f3 = self._DOMAIN.transformation(
                self._fun, eta1, eta2, eta3, 'norm_to_1_3')

        else:
            raise ValueError(
                'Coordinates to be used for 1-form pullback not properly specified.')

        return [f1, f2, f3]


class Pulled_2form:
    '''Construct callable 2-form on logical domain (unit cube).

    Depending on the dimension of eta1 either point-wise, tensor-product, slice plane or general (see struphy/geometry/domain_3d.prepare_args).

    Returns a list of three np.arrays holding the values.'''

    def __init__(self, coords, fun, DOMAIN):
        '''
        Parameters
        ----------
            coords : str
                From which coordinate representation to pull: 'logical', 'physical' or 'norm_logical'.

            fun : 3-list
                Callable components.

            DOMAIN : obj
                From struphy/geometry/domain_3d.Domain.
        '''

        assert len(fun) == 3

        self._fun = []
        for f in fun:
            if f == None:
                def f(x, y, z): return 0.
            self._fun += [f]

        self._coords = coords
        self._DOMAIN = DOMAIN

    def __call__(self, eta1, eta2, eta3):

        if self._coords == 'logical':
            E1, E2, E3, is_sparse_meshgrid = prepare_args(eta1, eta2, eta3)
            f1 = np.array(self._fun[0](E1, E2, E3))
            f2 = np.array(self._fun[1](E1, E2, E3))
            f3 = np.array(self._fun[2](E1, E2, E3))

        elif self._coords == 'physical':
            f1 = self._DOMAIN.pull(self._fun, eta1, eta2, eta3, '2_form_1')
            f2 = self._DOMAIN.pull(self._fun, eta1, eta2, eta3, '2_form_2')
            f3 = self._DOMAIN.pull(self._fun, eta1, eta2, eta3, '2_form_3')

        elif self._coords == 'norm_logical':
            f1 = self._DOMAIN.transformation(
                self._fun, eta1, eta2, eta3, 'norm_to_2_1')
            f2 = self._DOMAIN.transformation(
                self._fun, eta1, eta2, eta3, 'norm_to_2_2')
            f3 = self._DOMAIN.transformation(
                self._fun, eta1, eta2, eta3, 'norm_to_2_3')

        else:
            raise ValueError(
                'Coordinates to be used for 2-form pullback not properly specified.')

        return [f1, f2, f3]


class Pulled_3form:
    '''Construct callable 3-form on logical domain (unit cube).

    Depending on the dimension of eta1 either point-wise, tensor-product, slice plane or general (see struphy/geometry/domain_3d.prepare_args).

    Returns a list of one np.array holding the values.'''

    def __init__(self, coords, fun, DOMAIN):
        '''
        Parameters
        ----------
            coords : str
                From which coordinate representation to pull: 'logical', 'physical' or 'norm_logical'.

            fun : 1-list
                Callable function.

            DOMAIN : obj
                From struphy/geometry/domain_3d.Domain.
        '''

        assert len(fun) == 1

        self._fun = []
        for f in fun:
            if f == None:
                def f(x, y, z): return 0.
            self._fun += [f]

        self._coords = coords
        self._DOMAIN = DOMAIN

    def __call__(self, eta1, eta2, eta3):

        if self._coords == 'logical':
            E1, E2, E3, is_sparse_meshgrid = prepare_args(eta1, eta2, eta3)
            f = np.array(self._fun[0](E1, E2, E3))

        elif self._coords == 'physical':
            f = self._DOMAIN.pull(self._fun, eta1, eta2, eta3, '3_form')

        elif self._coords == 'norm_logical':
            f = self._DOMAIN.transformation(
                self._fun, eta1, eta2, eta3, 'norm_to_3')

        else:
            raise ValueError(
                'Coordinates to be used for 3-form pullback not properly specified.')

        return [f]
