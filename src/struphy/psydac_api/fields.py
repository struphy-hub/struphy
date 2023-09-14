#!/usr/bin/env python3

from xml import dom
from psydac.linalg.stencil import StencilVector
from psydac.linalg.block import BlockVector

from struphy.initial import perturbations
from struphy.initial import eigenfunctions

from struphy.polar.basic import PolarVector
from struphy.geometry.base import Domain
from struphy.b_splines import bspline_evaluation_3d as eval_3d
from struphy.fields_background.mhd_equil.equils import set_defaults

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
            Space identifier for the field ("H1", "Hcurl", "Hdiv", "L2" or "H1vec").

        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.
    """

    def __init__(self, name, space_id, derham):

        self._name = name
        self._space_id = space_id
        self._derham = derham

        # initialize field in memory (FEM space, vector and tensor product (stencil) vector)
        self._space_key = derham.spaces_dict[space_id]
        self._space = derham.Vh_fem[self._space_key]

        self._vector = derham.Vh_pol[self._space_key].zeros()

        self._vector_stencil = self._space.vector_space.zeros()

        # transposed basis extraction operator for PolarVector --> Stencil-/BlockVector
        self._ET = derham.E[self._space_key].transpose()

        # global indices of each process, and paddings
        if self._space_id in {'H1', 'L2'}:
            self._gl_s = self._space.vector_space.starts
            self._gl_e = self._space.vector_space.ends
            self._pads = self._space.vector_space.pads
        else:
            self._gl_s = [
                comp.starts for comp in self._space.vector_space.spaces]
            self._gl_e = [
                comp.ends for comp in self._space.vector_space.spaces]
            self._pads = [
                comp.pads for comp in self._space.vector_space.spaces]

        # dimensions in each direction
        # self._nbasis = derham.nbasis[self._space_key]

        if self._space_id in {'H1', 'L2'}:
            self._nbasis = tuple(
                [space.nbasis for space in self._space.spaces])
        else:
            self._nbasis = [tuple([space.nbasis for space in vec_space.spaces])
                            for vec_space in self._space.spaces]

    @property
    def name(self):
        """ Name of the field in data container (string).
        """
        return self._name

    @property
    def space_id(self):
        """ String identifying the continuous space of the field: 'H1', 'Hcurl', 'Hdiv', 'L2' or 'H1vec'.
        """
        return self._space_id

    @property
    def space_key(self):
        """ String identifying the discrete space of the field: '0', '1', '2', '3' or 'v'.
        """
        return self._space_key

    @property
    def derham(self):
        """ 3d Derham complex struphy.psydac_api.psydac_derham.Derham.
        """
        return self._derham

    @property
    def space(self):
        """ Discrete space of the field, either psydac.fem.tensor.TensorFemSpace or psydac.fem.vector.VectorFemSpace.
        """
        return self._space

    @property
    def ET(self):
        """ Transposed PolarExtractionOperator (or IdentityOperator) for mapping polar coeffs to polar tensor product rings.
        """
        return self._ET

    @property
    def vector(self):
        """ psydac.linalg.stencil.StencilVector or psydac.linalg.block.BlockVector or struphy.polar.basic.PolarVector.
        """
        return self._vector

    @vector.setter
    def vector(self, value):
        """ In-place setter for Stencil-/Block-/PolarVector.
        """

        if isinstance(self._vector, StencilVector):

            assert isinstance(value, (StencilVector, np.ndarray))

            s1, s2, s3 = self.starts
            e1, e2, e3 = self.ends

            self._vector[s1:e1 + 1, s2:e2 + 1, s3:e3 + 1] = \
                value[s1:e1 + 1, s2:e2 + 1, s3:e3 + 1]

        elif isinstance(self._vector, BlockVector):

            assert isinstance(value, (BlockVector, list, tuple))

            for n in range(3):

                s1, s2, s3 = self.starts[n]
                e1, e2, e3 = self.ends[n]

                self._vector[n][s1:e1 + 1, s2:e2 + 1, s3:e3 + 1] = \
                    value[n][s1:e1 + 1, s2:e2 + 1, s3:e3 + 1]

        elif isinstance(self._vector, PolarVector):

            assert isinstance(value, (PolarVector, list, tuple))

            if isinstance(value, PolarVector):
                self._vector.set_vector(value)
            else:

                if isinstance(self._vector.tp, StencilVector):

                    assert isinstance(value[0], np.ndarray)
                    assert isinstance(value[1], (StencilVector, np.ndarray))

                    self._vector.pol[0][:] = value[0][:]

                    s1, s2, s3 = self.starts
                    e1, e2, e3 = self.ends

                    self._vector.tp[s1:e1 + 1, s2:e2 + 1, s3:e3 + 1] = \
                        value[1][s1:e1 + 1, s2:e2 + 1, s3:e3 + 1]
                else:
                    for n in range(3):

                        assert isinstance(value[n][0], np.ndarray)
                        assert isinstance(
                            value[n][1], (StencilVector, np.ndarray))

                        self._vector.pol[n][:] = value[n][0][:]

                        s1, s2, s3 = self.starts[n]
                        e1, e2, e3 = self.ends[n]

                        self._vector.tp[n][s1:e1 + 1, s2:e2 + 1, s3:e3 + 1] = \
                            value[n][1][s1:e1 + 1, s2:e2 + 1, s3:e3 + 1]

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
    def nbasis(self):
        """ Tuple(s) of 1d dimensions for each direction.
        """
        return self._nbasis

    @property
    def vector_stencil(self):
        """ Tensor-product Stencil-/BlockVector corresponding to a copy of self.vector in case of Stencil-/Blockvector 

            OR 

            the extracted coefficients in case of PolarVector. Call self.extract_coeffs() beforehand.
        """
        return self._vector_stencil

    def extract_coeffs(self, update_ghost_regions=True):
        """
        Maps polar coeffs to polar tensor product rings in case of PolarVector (written in-place to self.vector_stencil) and updates ghost regions.

        Parameters
        ----------
            update_ghost_regions : bool
                If the ghost regions shall be updated (needed in case of non-local acccess, e.g. in field evaluation).
        """
        self._ET.dot(self._vector, out=self._vector_stencil)

        if update_ghost_regions:
            self._vector_stencil.update_ghost_regions()

    def initialize_coeffs(self, init_params, domain=None):
        """
        Sets the initial conditions for self.vector.

        Parameters
        ----------
        init_params : dict
            Parameters of initial condition, see from :ref:`params_yml`.

        domain : struphy.geometry.domains (optional)
            Domain object for metric coefficients. Needed if init_params[init_params['type']]['coords'] == 'physical'.
        """

        init_types = []
        fun_params = []

        # identifying initial conditions of self.vector
        if init_params['type'] is None:
            pass

        elif type(init_params['type']) == str:

            if np.any(init_params[init_params['type']]['comps'][self.name]):

                init_types += [init_params['type']]
                fun_params += [init_params[init_types[0]].copy()]

        elif type(init_params['type']) == list:

            for n, _type in enumerate(init_params['type']):

                if np.any(init_params[_type]['comps'][self.name]):

                    init_types += [_type]
                    fun_params += [init_params[_type].copy()]

        else:
            raise NotImplemented(
                f'The type of initial condition must be null or str or list.')

        ntypes = len(init_types)

        if ntypes != 0:

            # white noise in logical space for different components
            if any(_type == 'noise' for _type in init_types):

                assert ntypes == 1, \
                    AssertionError (
                        "The init type 'noise' cannot be applied with other init types")

                params_default = {'comps': {'b2': [True, False, False]},
                                  'variation_in': 'e3',
                                  'amp': 0.0001,
                                  'seed': 1234
                                  }

                self._params = set_defaults(fun_params[0], params_default)

                # component(s) to perturb
                if isinstance(fun_params[0]['comps'][self.name], bool):
                    comps = [fun_params[0]['comps'][self.name]]
                else:
                    comps = fun_params[0]['comps'][self.name]

                # set white noise FE coefficients
                if self.space_id in {'H1', 'L2'}:
                    if comps[0]:
                        self._add_noise(fun_params[0])

                elif self.space_id in {'Hcurl', 'Hdiv', 'H1vec'}:
                    for n, comp in enumerate(comps):
                        if comp:
                            self._add_noise(fun_params[0], n=n)

            # loading of eigenfunction
            elif any(_type[-6:] == 'EigFun' for _type in init_types):

                assert ntypes == 1, \
                    AssertionError(
                        "The init type 'EigFun' cannot be applied with other init types")

                # select class
                funs = getattr(eigenfunctions, init_types[0])(
                    self.derham, **fun_params[0])

                # select eigenvector and set coefficients
                if hasattr(funs, self.name):

                    eig_vec = getattr(funs, self.name)

                    self.vector = eig_vec

            # Fourier modes
            elif any(_type in ['ModesSin', 'ModesCos', 'TorusModesSin', 'TorusModesCos'] for _type in init_types):
                
                form_str = self.derham.forms_dict[self.space_id]

                if self.space_id in {'H1', 'L2'}:

                    assert ntypes == 1, \
                        AssertionError(
                            f'Only one init type can be applied to the variables in space {self.space_id}.')
                    
                    coord_tmp = 'logical'
                    fun_tmp = [None]

                    # coordinates: logical (default) or physical
                    if 'coords' in fun_params[0]:
                        coord_tmp = fun_params[0]['coords']
                        fun_params[0].pop('coords')

                    # get callable(s) for specified init type
                    fun_class = getattr(perturbations, init_types[0])
                    fun_params[0].pop('comps')
                    fun_tmp[0] = fun_class(**fun_params[0])

                    # pullback callable
                    fun = PulledPform(coord_tmp, fun_tmp, domain, form_str)

                elif self.space_id in {'Hcurl', 'Hdiv', 'H1vec'}:

                    assert ntypes < 4, \
                        AssertionError(
                            f'Maximum 3 init types can be applied to the variables in space {self.space_id}.')

                    coord_tmp = 'logical'
                    coords_tmp = ['logical', 'logical', 'logical']
                    fun_tmp = [None, None, None]

                    for n, _type in enumerate(init_types):

                        fun_class = getattr(perturbations, _type)

                        comps = fun_params[n]['comps'][self.name]
                        fun_params[n].pop('comps')

                        if 'coords' in fun_params[n]:
                            coord_tmp = fun_params[n]['coords']
                            fun_params[n].pop('coords')

                        for axis, comp in enumerate(comps):

                            if comp:
                                fun_tmp[axis] = fun_class(**fun_params[n])
                                coords_tmp[axis] = coord_tmp

                    # pullback callable
                    fun = []

                    fun += [PulledPform(coords_tmp[0], fun_tmp, domain,
                                        form_str + '_1')]
                    fun += [PulledPform(coords_tmp[1], fun_tmp, domain,
                                        form_str + '_2')]
                    fun += [PulledPform(coords_tmp[2], fun_tmp, domain,
                                        form_str + '_3')]

                # peform projection
                self.vector = self.derham.P[self.space_key](fun)

            else:
                raise NotImplemented(
                    f'Initial condition {init_types} not available.')

        # apply boundary operator (in-place)
        self.derham.B[self.space_key].dot(
            self._vector.copy(), out=self._vector)

        # update ghost regions
        self._vector.update_ghost_regions()


    def initialize_coeffs_from_restart_file(self, file, species=None):
        """
        TODO
        """

        if species is None:
            key = 'restart/' + self.name
        else:
            key = 'restart/' + species + '_' + self.name

        if isinstance(self.vector, StencilVector):
            self.vector._data[:] = file[key][-1]
        else:
            for n in range(3):
                self.vector[n]._data[:] = file[key + '/' + str(n + 1)][-1]

        self._vector.update_ghost_regions()

    def __call__(self, eta1, eta2, eta3, squeeze_output=False, local=False):
        """
        Evaluates the spline function on the global domain, unless local is given to True (in which case the spline function is evaluated only on the local domain).

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
            E1[E1 == dom_arr[rank, 0]] += 1e-8
        if dom_arr[rank, 1] != 1.:
            E1[E1 == dom_arr[rank, 1]] += 1e-8

        if dom_arr[rank, 3] != 0.:
            E2[E2 == dom_arr[rank, 3]] += 1e-8
        if dom_arr[rank, 4] != 1.:
            E2[E2 == dom_arr[rank, 4]] += 1e-8

        if dom_arr[rank, 6] != 0.:
            E3[E3 == dom_arr[rank, 6]] += 1e-8
        if dom_arr[rank, 7] != 1.:
            E3[E3 == dom_arr[rank, 7]] += 1e-8

        # True for eval points on current process
        E1_on_proc = np.logical_and(
            E1 >= dom_arr[rank, 0], E1 <= dom_arr[rank, 1])
        E2_on_proc = np.logical_and(
            E2 >= dom_arr[rank, 3], E2 <= dom_arr[rank, 4])
        E3_on_proc = np.logical_and(
            E3 >= dom_arr[rank, 6], E3 <= dom_arr[rank, 7])

        # flag eval points not on current process
        E1[~E1_on_proc] = -1.
        E2[~E2_on_proc] = -1.
        E3[~E3_on_proc] = -1.

        # prepare arrays for AllReduce
        tmp = np.zeros((E1.shape[0], E2.shape[1], E3.shape[2]), dtype=float)

        # extract coefficients and update ghost regions
        self.extract_coeffs(update_ghost_regions=True)

        # call pyccel kernels
        T1, T2, T3 = self.derham.Vh_fem['0'].knots

        if isinstance(self._vector_stencil, StencilVector):

            kind = self.derham.spline_types_pyccel[self.space_key]

            if is_sparse_meshgrid:
                # eval_mpi needs flagged arrays E1, E2, E3 as input
                eval_3d.eval_spline_mpi_sparse_meshgrid(E1, E2, E3, self._vector_stencil._data, kind,
                                                        np.array(self.derham.p), T1, T2, T3, np.array(self.starts), tmp)
            else:
                # eval_mpi needs flagged arrays E1, E2, E3 as input
                eval_3d.eval_spline_mpi_matrix(E1, E2, E3, self._vector_stencil._data, kind,
                                               np.array(self.derham.p), T1, T2, T3, np.array(self.starts), tmp)

            if self.derham.comm is not None:
                if local == False:
                    self.derham.comm.Allreduce(MPI.IN_PLACE, tmp, op=MPI.SUM)

            # all processes have all values
            values = tmp

            if squeeze_output:
                values = np.squeeze(values)

            if values.ndim == 0:
                values = values.item()

        else:

            values = []
            for n, kind in enumerate(self.derham.spline_types_pyccel[self.space_key]):

                if is_sparse_meshgrid:
                    eval_3d.eval_spline_mpi_sparse_meshgrid(E1, E2, E3, self._vector_stencil[n]._data, kind,
                                                            np.array(self.derham.p), T1, T2, T3, np.array(self.starts[n]), tmp)
                else:
                    eval_3d.eval_spline_mpi_matrix(E1, E2, E3, self._vector_stencil[n]._data, kind,
                                                   np.array(self.derham.p), T1, T2, T3, np.array(self.starts[n]), tmp)

                if self.derham.comm is not None:
                    if local == False:
                        self.derham.comm.Allreduce(
                            MPI.IN_PLACE, tmp, op=MPI.SUM)

                # all processes have all values
                values += [tmp.copy()]
                tmp[:] = 0.

                if squeeze_output:
                    values[-1] = np.squeeze(values[-1])

                if values[-1].ndim == 0:
                    values[-1] = values[-1].item()

        return values

    def _add_noise(self, fun_params, n=None):
        """ Add noise to a vector component where init_comps==True, otherwise leave at zero.

        Parameters
        ----------
            fun_params : dict
                From parameter file under init/noise.

            n : int
                Vector component (0, 1 or 2) to be initialized.
        """

        _direction = fun_params['variation_in']
        _ampsize = fun_params['amp']
        _seed = fun_params['seed']
        
        # index slices from global start to end in all directions
        sli = []
        gl_s = []
        for d in range(3):
            if n == None:
                sli += [slice(self._gl_s[d], self._gl_e[d] + 1)]
                gl_s += [self._gl_s[d]]
                vec = self._vector
            else:
                sli += [slice(self._gl_s[n][d], self._gl_e[n][d] + 1)]
                gl_s += [self._gl_s[n][d]]
                vec = self._vector[n]

        # local shape without ghost regions
        if n == None:
            _shape = (self._gl_e[0] + 1 - self._gl_s[0], self._gl_e
                      [1] + 1 - self._gl_s[1], self._gl_e[2] + 1 - self._gl_s[2])
        else:
            _shape = (self._gl_e[n][0] + 1 - self._gl_s[n][0], self._gl_e[n]
                      [1] + 1 - self._gl_s[n][1], self._gl_e[n][2] + 1 - self._gl_s[n][2])

        if _direction == 'e1':
            _amps = self._tmp_noise_for_mpi(
                _shape[0], direction=_direction, amp_size=_ampsize, seed=_seed)
            for j in range(_shape[1]):
                for k in range(_shape[2]):
                    vec[sli[0], gl_s[1] + j, gl_s[2] + k] = _amps
            del _amps

        elif _direction == 'e2':
            _amps = self._tmp_noise_for_mpi(
                _shape[1], direction=_direction, amp_size=_ampsize, seed=_seed)
            for j in range(_shape[0]):
                for k in range(_shape[2]):
                    vec[gl_s[0] + j, sli[1], gl_s[2] + k] = _amps

        elif _direction == 'e3':
            _amps = self._tmp_noise_for_mpi(
                _shape[2], direction=_direction, amp_size=_ampsize, seed=_seed)
            for j in range(_shape[0]):
                for k in range(_shape[1]):
                    vec[gl_s[0] + j, gl_s[1] + k, sli[2]] = _amps

        elif _direction == 'e1e2':
            _amps = self._tmp_noise_for_mpi(
                _shape[0], _shape[1], direction=_direction, amp_size=_ampsize, seed=_seed)
            for j in range(_shape[2]):
                vec[sli[0], sli[1], gl_s[2] + j] = _amps

        elif _direction == 'e1e3':
            _amps = self._tmp_noise_for_mpi(
                _shape[0], _shape[2], direction=_direction, amp_size=_ampsize, seed=_seed)
            for j in range(_shape[1]):
                vec[sli[0], gl_s[1] + j, sli[2]] = _amps

        elif _direction == 'e2e3':
            _amps = self._tmp_noise_for_mpi(
                _shape[1], _shape[2], direction=_direction, amp_size=_ampsize, seed=_seed)
            for j in range(_shape[0]):
                vec[gl_s[0] + j, sli[1], sli[2]] = _amps

        elif _direction == 'e1e2e3':
            _amps = self._tmp_noise_for_mpi(
                _shape[0], _shape[1], _shape[2], direction=_direction, amp_size=_ampsize, seed=_seed)
            vec[sli[0], sli[1], sli[2]] = _amps

        else:
            raise ValueError('Invalid direction for noise.')

    def _tmp_noise_for_mpi(self, *shapes, direction='e3', amp_size=0.0001, seed=None):
        '''Initialize same FEEC noise regardless of number of MPI processes.
        
        Parameters
        ----------
        shapes : int
            Length of local array size in each direction where noise is to be initialized.
        
        direction : str
            Noise direction ('e1', 'e2' or 'e3'). Multi-dim. not yet correct.
            
        amp_size : float
            Noise amplitude
            
        seed : int
            Seed for random number generator.
            
        Returns
        -------
        _amps : np.array
            The noisy FE coefficients in the desired direction (1d, 2d or 3d array).'''

        if self.derham.comm is not None:
            comm_size = self.derham.comm.Get_size()
            rank = self.derham.comm.Get_rank()
            nprocs = self.derham.domain_decomposition.nprocs
        else:
            comm_size = 1
            rank = 0
            nprocs = [1, 1, 1]

        domain_array = self.derham.domain_array

        if seed is not None:
            np.random.seed(seed)

        # temporary
        _amps = np.zeros(shapes)

        # no process has been drawn for yet
        already_drawn = np.zeros(nprocs) == 1.

        # 1d mid point arrays in each direction
        mid_points = []
        for npr in nprocs:
            delta = 1./npr
            mid_points_i = np.zeros(npr)
            for n in range(npr):
                mid_points_i[n] = delta*(n + 1/2)
            mid_points += [mid_points_i]

        if direction == 'e1':
            tmp_arrays = np.zeros(nprocs[0]).tolist()
        elif direction == 'e2':
            tmp_arrays = np.zeros(nprocs[1]).tolist()
        elif direction == 'e3':
            tmp_arrays = np.zeros(nprocs[2]).tolist()
        elif direction == 'e1e2':
            tmp_arrays = np.zeros((nprocs[0], nprocs[1])).tolist()
            Warning, f'2d noise in the directions {direction} is not correctly initilaized for MPI !!'
        elif direction == 'e1e3':
            tmp_arrays = np.zeros((nprocs[0], nprocs[2])).tolist()
            Warning, f'2d noise in the directions {direction} is not correctly initilaized for MPI !!'
        elif direction == 'e2e3':
            tmp_arrays = np.zeros((nprocs[1], nprocs[2])).tolist()
            Warning, f'2d noise in the directions {direction} is not correctly initilaized for MPI !!'
        elif direction == 'e1e2e3':
            Warning, f'3d noise in the directions {direction} is not correctly initilaized for MPI !!'
            pass
        else:
            raise ValueError('Invalid direction for tmp_arrays.')

        # 3d index of current process from mid points
        inds_current = []
        for n in range(3):
            mid_pt_current = (
                domain_array[rank, 3*n] + domain_array[rank, 3*n + 1]) / 2.
            inds_current += [np.argmin(np.abs(mid_points[n] - mid_pt_current))]

        # loop over processes
        for i in range(comm_size):

            # 3d index of process i from mid points
            inds = []
            for n in range(3):
                mid_pt = (domain_array[i, 3*n] + domain_array[i, 3*n + 1]) / 2.
                inds += [np.argmin(np.abs(mid_points[n] - mid_pt))]

            if already_drawn[inds[0], inds[1], inds[2]]:

                if direction == 'e1':
                    _amps[:] = tmp_arrays[inds[0]]
                elif direction == 'e2':
                    _amps[:] = tmp_arrays[inds[1]]
                elif direction == 'e3':
                    _amps[:] = tmp_arrays[inds[2]]
                elif direction == 'e1e2':
                    _amps[:] = tmp_arrays[inds[0]][inds[1]]
                elif direction == 'e1e3':
                    _amps[:] = tmp_arrays[inds[0]][inds[2]]
                elif direction == 'e2e3':
                    _amps[:] = tmp_arrays[inds[1]][inds[2]]
                elif direction == 'e1e2e3':
                    _amps[:] = (np.random.rand(*shapes) - .5) * 2. * amp_size

            else:

                if direction == 'e1':
                    tmp_arrays[inds[0]] = (np.random.rand(
                        *shapes) - .5) * 2. * amp_size
                    already_drawn[inds[0], :, :] = True
                    _amps[:] = tmp_arrays[inds[0]]
                elif direction == 'e2':
                    tmp_arrays[inds[1]] = (np.random.rand(
                        *shapes) - .5) * 2. * amp_size
                    already_drawn[:, inds[1], :] = True
                    _amps[:] = tmp_arrays[inds[1]]
                elif direction == 'e3':
                    tmp_arrays[inds[2]] = (np.random.rand(
                        *shapes) - .5) * 2. * amp_size
                    already_drawn[:, :, inds[2]] = True
                    _amps[:] = tmp_arrays[inds[2]]
                elif direction == 'e1e2':
                    tmp_arrays[inds[0]][inds[1]] = (
                        np.random.rand(*shapes) - .5) * 2. * amp_size
                    already_drawn[inds[0], inds[1], :] = True
                    _amps[:] = tmp_arrays[inds[0]][inds[1]]
                elif direction == 'e1e3':
                    tmp_arrays[inds[0]][inds[2]] = (
                        np.random.rand(*shapes) - .5) * 2. * amp_size
                    already_drawn[inds[0], :, inds[2]] = True
                    _amps[:] = tmp_arrays[inds[0]][inds[2]]
                elif direction == 'e2e3':
                    tmp_arrays[inds[1]][inds[2]] = (
                        np.random.rand(*shapes) - .5) * 2. * amp_size
                    already_drawn[:, inds[1], inds[2]] = True
                    _amps[:] = tmp_arrays[inds[1]][inds[2]]

            if np.all(np.array([ind_c == ind for ind_c, ind in zip(inds_current, inds)])):
                return _amps


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
            if self._form[0] == '0' or self._form[0] == '3':
                f = self._domain.pull(
                    self._fun, eta1, eta2, eta3, kind=self._form)
            else:
                f = self._domain.pull(
                    self._fun, eta1, eta2, eta3, kind=self._form[:-2])[self._comp]
        else:
            raise ValueError(
                'Coordinates to be used for p-form pullback not properly specified.')

        return f
