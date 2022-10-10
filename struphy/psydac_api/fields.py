#!/usr/bin/env python3

from xml import dom
from psydac.linalg.stencil import StencilVector
from psydac.fem.tensor import FemField

from struphy.initial import perturbations
from struphy.initial import analytic

from struphy.psydac_api.utilities import apply_essential_bc_to_array
from struphy.geometry.base import Domain
from struphy.feec.basics import spline_evaluation_3d as eval_3d

import numpy as np
from mpi4py import MPI


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
            self._gl_s = self._vector.starts
            self._gl_e = self._vector.ends
            self._pads = self._vector.pads
        else:
            self._gl_s = [comp.starts for comp in self._vector]
            self._gl_e = [comp.ends for comp in self._vector]
            self._pads = [comp.pads for comp in self._vector]

        # 1d spline types in each direction
        if isinstance(self._vector, StencilVector):
            self._spline_types = [space.basis for space in self._space.spaces]
            self._spline_types_pyccel = np.array([int(space.basis == 'M') for space in self._space.spaces])
        else:
            self._spline_types = [[space.basis for space in tensor_femspace.spaces]
                                  for tensor_femspace in self._space._spaces]
            self._spline_types_pyccel = [
                np.array([int(space.basis == 'M') for space in tensor_femspace.spaces]) for tensor_femspace in self._space._spaces]

        # dimensions in each direction
        if isinstance(self._vector, StencilVector):
            self._nbasis = tuple([space.nbasis for space in self._space.spaces])
        else:
            self._nbasis = [tuple([space.nbasis for space in vec_space.spaces]) for vec_space in self._space.spaces]

    @property
    def name(self):
        """ Name of the field in data container (string).
        """
        return self._name

    @property
    def space(self):
        """ Discrete space of the field, either psydac.fem.tensor.TensorFemSpace or psydac.fem.vector.ProductFemSpace.
        """
        return self._space

    @property
    def space_id(self):
        """ String identifying the continuous space of the field: 'H1', 'Hcurl', 'Hdiv', 'L2' or 'H1vec'.
        """
        return self._space_id

    @property
    def derham(self):
        """ 3d Derham complex struphy.psydac_api.psydac_derham.Derham.
        """
        return self._derham

    @property
    def field(self):
        """ psydac.fem.tensor.FemField.
        """
        return self._field

    @property
    def vector(self):
        """ psydac.linalg.stencil.StencilVector or psydac.linalg.block.BlockVector.
        """
        return self._vector

    @vector.setter
    def vector(self, value):
        self._vector = value

    @property
    def starts(self):
        """ Global indices of the first FE coefficient on the process, in each direction.
        """
        return self._gl_s

    @property
    def ends(self):
        """ Global indices of the last FE coefficient on the process, in each direction.
        """
        return self._gl_e

    @property
    def pads(self):
        """ Paddings for ghost regions, in each direction.
        """
        return self._pads

    @property
    def spline_types(self):
        """ List holding holding 1d spline types in each direction, entries either 'B' or 'M'.
        """
        return self._spline_types

    @property
    def spline_types_pyccel(self):
        """ List holding holding 1d spline types in each direction, entries either 0 (='B') or 1 (='M').
        """
        return self._spline_types_pyccel

    @property
    def nbasis(self):
        """ Tuple(s) of 1d dimensions for each direction.
        """
        return self._nbasis

    def initialize_coeffs(self, comps, init_params, domain=None):
        """
        Sets the initial conditions for self.vector.

        Parameters
        ----------
            comps: list
                Booleans that specify whether field component has non-zero initial conditions (True).

            init_params: dict
                Parameters of initial condition, see from :ref:`params_yml`.

            domain: struphy.geometry.domains
                Optional: all things mapping. Needed when init_params['coords'] == 'physical'.
        """

        rank = self._derham.comm.Get_rank()
        
        if rank == 0: print(f'Setting initial conditions for {self.name} in {self.space_id} ...')

        # Set initial conditions for each component
        assert isinstance(comps, list)

        init_coords = init_params['coords']
        if init_coords == 'physical':
            assert domain is not None

        fun_params = init_params[init_params['type']]

        if init_params['type'] == 'noise':

            # Set white noise FE coefficients
            if self.space_id in {'H1', 'L2'}:
                if comps[0]:
                    self._add_noise(fun_params)

            elif self.space_id in {'Hcurl', 'Hdiv', 'H1vec'}:
                for n, comp in enumerate(comps):
                    if comp:
                        self._add_noise(fun_params, n=n)

        elif 'ModesSin' in init_params['type'] or 'ModesCos' in init_params['type']:

            # Get callable(s) for specified init type
            _fun_tmp = [None] * len(comps)
            for n, comp in enumerate(comps):
                assert isinstance(comp, bool)
                if comp:
                    fun_class = getattr(perturbations, init_params['type'])
                    _fun_tmp[n] = fun_class(*list(fun_params.values()))

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

            fun_class = getattr(analytic, init_params['type'])
            funs = fun_class(self.init_params, domain)

            if self.space_id == 'H1':
                self._vector[:] = self.derham.P0(
                    getattr(funs, self.name)).coeffs[:]

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
                self._vector[:] = self.derham.P3(
                    getattr(funs, self.name)).coeffs[:]

            elif self.space_id == 'H1vec':
                _coeffs = self.derham.P0vec(getattr(funs, self.name)).coeffs
                self._vector[0][:] = _coeffs[0][:]
                self._vector[1][:] = _coeffs[1][:]
                self._vector[2][:] = _coeffs[2][:]

        # apply boundary conditions
        apply_essential_bc_to_array(
            self.space_id, self._vector, self.derham.bc)

        self._vector.update_ghost_regions()

        if rank == 0:
            print('Done.')

    def __call__(self, eta1, eta2, eta3, squeeze_output=False):
        """
        Evaluates the spline function on the local domain.

        Parameters
        ----------
            eta1, eta2, eta3 : array-like
                Logical coordinates at which to evaluate.

            flat_eval : bool
                Whether to do a flat evaluation, i.e. f([e11, e12], [e21, e22]) = [f(e11, e21), f(e12, e22)].

            squeeze_output : bool
                Whether to remove singleton dimensions in output "values".

        Returns
        -------
            values : array[float] or list
                The values of the spline function at the given point set (list in case of vector-valued spaces).
        """

        # all eval points
        E1, E2, E3, is_sparse_meshgrid = Domain.prepare_eval_pts(
            eta1, eta2, eta3)

        # check if eval points are "interior points" in domain_array; if so, add small offset
        dom_arr = self.derham.domain_array
        if self.derham.comm is not None:
            rank = self.derham.comm.Get_rank()
        else:
            rank = 0

        if dom_arr[rank, 0] != 0.:
            E1[E1==dom_arr[rank, 0]] += 1e-8
        if dom_arr[rank, 1] != 1.:
            E1[E1==dom_arr[rank, 1]] += 1e-8

        if dom_arr[rank, 3] != 0.:
            E2[E2==dom_arr[rank, 3]] += 1e-8
        if dom_arr[rank, 4] != 1.:
            E2[E2==dom_arr[rank, 4]] += 1e-8

        if dom_arr[rank, 6] != 0.:
            E3[E3==dom_arr[rank, 6]] += 1e-8
        if dom_arr[rank, 7] != 1.:
            E3[E3==dom_arr[rank, 7]] += 1e-8

        # True for eval points on current process
        E1_on_proc = np.logical_and(E1 >= dom_arr[rank, 0], E1 <= dom_arr[rank, 1])
        E2_on_proc = np.logical_and(E2 >= dom_arr[rank, 3], E2 <= dom_arr[rank, 4])
        E3_on_proc = np.logical_and(E3 >= dom_arr[rank, 6], E3 <= dom_arr[rank, 7])

        # flag eval points not on current process
        E1[~E1_on_proc] = -1.
        E2[~E2_on_proc] = -1.
        E3[~E3_on_proc] = -1.

        # prepare arrays for AllReduce
        tmp = np.zeros((E1.shape[0], E2.shape[1], E3.shape[2]), dtype=float)
        tmp_global = tmp.copy()

        # call pyccel kernels
        if isinstance(self.vector, StencilVector):

            kind = self.spline_types_pyccel
            if is_sparse_meshgrid:
                # eval_mpi needs flagged arrays E1, E2, E3 as input
                eval_3d.eval_spline_mpi_sparse_meshgrid(E1, E2, E3, self.vector._data, kind, np.array(self.derham.p),
                                                        self.derham.V0.knots[0], self.derham.V0.knots[1], self.derham.V0.knots[2],
                                                        np.array(self.starts), tmp)
            else:
                # eval_mpi needs flagged arrays E1, E2, E3 as input
                eval_3d.eval_spline_mpi_matrix(E1, E2, E3, self.vector._data, kind, np.array(self.derham.p),
                                               self.derham.V0.knots[0], self.derham.V0.knots[1], self.derham.V0.knots[2],
                                               np.array(self.starts), tmp)

            if self.derham.comm is not None:
                self.derham.comm.Allreduce(tmp, tmp_global, op=MPI.SUM)
            else:
                tmp_global = tmp

            # all processes have all values
            values = tmp_global

            if squeeze_output:
                values = np.squeeze(values)

        else:

            values = []
            for n, kind in enumerate(self.spline_types_pyccel):
                if is_sparse_meshgrid:
                    eval_3d.eval_spline_mpi_sparse_meshgrid(E1, E2, E3, self.vector[n]._data, kind, np.array(self.derham.p),
                                                            self.derham.V0.knots[0], self.derham.V0.knots[1], self.derham.V0.knots[2],
                                                            np.array(self.starts[n]), tmp)
                else:
                    eval_3d.eval_spline_mpi_matrix(E1, E2, E3, self.vector[n]._data, kind, np.array(self.derham.p),
                                                   self.derham.V0.knots[0], self.derham.V0.knots[1], self.derham.V0.knots[2],
                                                   np.array(self.starts[n]), tmp)
            
                if self.derham.comm is not None:
                    self.derham.comm.Allreduce(tmp, tmp_global, op=MPI.SUM)
                else:
                    tmp_global = tmp.copy()
                tmp[:] = 0.

                # all processes have all values
                values += [tmp_global.copy()]
                tmp_global[:] = 0.

                if squeeze_output:
                    values[-1] = np.squeeze(values[-1])
            
        return values

    def _add_noise(self, fun_params, n=None):
        """ Add noise to a vector component where init_comps==True, otherwise leave at zero.

        Parameters
        ----------
            fun_params : dict
                From parameters/fields/init/noise.
        """

        _direction = fun_params['direction']
        _ampsize = fun_params['amp']

        if n == None:
            _shape_w_pads = self._vector[:].shape
            _shape = (self._gl_e[0] + 1 - self._gl_s[0], self._gl_e
                      [1] + 1 - self._gl_s[1], self._gl_e[2] + 1 - self._gl_s[2])
        else:
            _shape_w_pads = self._vector[n][:].shape
            _shape = (self._gl_e[n][0] + 1 - self._gl_s[n][0], self._gl_e[n]
                      [1] + 1 - self._gl_s[n][1], self._gl_e[n][2] + 1 - self._gl_s[n][2])

        if _direction == 'x':
            _amps = (np.random.rand(_shape_w_pads[0]) - .5) * 2. * _ampsize
            for j in range(_shape[1]):
                for k in range(_shape[2]):
                    if n == None:
                        self._vector[:, self._gl_s[1] +
                                     j, self._gl_s[2] + k] = _amps
                    else:
                        self._vector[n][:, self._gl_s[n][1] +
                                        j, self._gl_s[n][2] + k] = _amps

        elif _direction == 'y':
            _amps = (np.random.rand(_shape_w_pads[1]) - .5) * 2. * _ampsize
            for j in range(_shape[0]):
                for k in range(_shape[2]):
                    if n == None:
                        self._vector[self._gl_s[0] + j,
                                     :, self._gl_s[2] + k] = _amps
                    else:
                        self._vector[n][self._gl_s[n][0] + j,
                                        :, self._gl_s[n][2] + k] = _amps

        elif _direction == 'z':
            _amps = (np.random.rand(_shape_w_pads[2]) - .5) * 2. * _ampsize
            for j in range(_shape[0]):
                for k in range(_shape[1]):
                    if n == None:
                        self._vector[self._gl_s[0] + j,
                                     self._gl_s[1] + k, :] = _amps
                    else:
                        self._vector[n][self._gl_s[n][0] + j,
                                        self._gl_s[n][1] + k, :] = _amps

        elif _direction == 'xy':
            _amps = (np.random.rand(
                _shape_w_pads[0], _shape_w_pads[1]) - .5) * 2. * _ampsize
            for j in range(_shape[2]):
                if n == None:
                    self._vector[:, :, self._gl_s[2] + j] = _amps
                else:
                    self._vector[n][:, :, self._gl_s[n][2] + j] = _amps

        elif _direction == 'xz':
            _amps = (np.random.rand(
                _shape_w_pads[0], _shape_w_pads[2]) - .5) * 2. * _ampsize
            for j in range(_shape[1]):
                if n == None:
                    self._vector[:, self._gl_s[1] + j, :] = _amps
                else:
                    self._vector[n][:, self._gl_s[n][1] + j, :] = _amps

        elif _direction == 'yz':
            _amps = (np.random.rand(
                _shape_w_pads[1], _shape_w_pads[2]) - .5) * 2. * _ampsize
            for j in range(_shape[0]):
                if n == None:
                    self._vector[self._gl_s[0] + j, :, :] = _amps
                else:
                    self._vector[n][self._gl_s[n][0] + j, :, :] = _amps

        elif _direction == 'xyz':
            _amps = (np.random.rand(
                _shape_w_pads[0], _shape_w_pads[1], _shape_w_pads[2]) - .5) * 2. * _ampsize
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
