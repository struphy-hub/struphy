import numpy as np

from psydac.linalg.stencil import StencilVector, StencilMatrix
from psydac.linalg.block import BlockVector, BlockMatrix

import struphy.pic.accum_kernels as accums


class Accumulator():
    '''Struphy accumulation matrices and vectors of the form

    .. math::

        M^{\mu,\\nu}_{ijk,mno} &= \sum_p \Lambda^\mu_{ijk}(\eta_p) * A^{\mu,\\nu}_p * (\Lambda^\\nu_{mno})^\\top(\eta_p)  \qquad  (\mu,\\nu = 1,2,3)

        V^\mu_{ijk} &= \sum_p \Lambda^\mu_{ijk}(\eta_p) * B^\mu_p

    where :math:`p` runs over the particles, :math:`\Lambda^\mu_{ijk}(\eta_p)` denotes the :math:`ijk`-th basis function
    of the :math:`\mu`-th component of a Derham space (V0, V1, V2, V3) evaluated at the particle position :math:`\eta_p`.

    :math:`A^{\mu,\\nu}_p` and :math:`B^\mu_p` are particle-dependent "filling functions",
    to be defined in the module **struphy.pic.accum_kernels**.

    Parameters
    ----------
        DOMAIN : struphy.geometry.domain_3d.Domain
            Domain object for mapping evaluations.

        DR : struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.

        space_id : str
            Space identifier for the matrix/vector (H1, Hcurl, Hdiv, L2 or H1^3) to be accumulated into.

        accumulator_name : str
            Name of accumulator function to be loaded from struphy/pic/accum_kernels.py.

        args_add : list
            Additional arguments to be passed to the accumulator function, besides the mandatory arguments
            which are prepared automatically (spline bases info, mapping info, data arrays).
            Examples would be parameters for a background kinetic distribution or spline coefficients of a background magnetic field.
            Entries must be pyccel-conform types.

        do_vector : bool
            True if, additionally to a matrix, a vector in the same space is to be accumulated. Default=False.

        symmetry : str
            In case of space_id=Hcurl/Hdiv, the symmetry property of the block matrix: diag, asym, symm or None (=full matrix, default)

    Note
    ----
        Struphy accumulation kernels called by ``Accumulator`` objects should be added to ``struphy/pic/accum_kernels.py``. 
        Please follow the docstring in `struphy.pic.accum_kernels._docstring`.
    '''

    def __init__(self, DOMAIN, DR, space_id, accumulator_name, *args_add, do_vector=False, symmetry=None):

        self._domain = DOMAIN
        self._derham = DR
        self._space_id = space_id
        self._accumulator_name = accumulator_name
        self._args_add = args_add
        self._do_vector = do_vector
        self._symmetry = symmetry

        if space_id == 'H1':
            self._space = DR.V0
        elif space_id == 'Hcurl':
            self._space = DR.V1
        elif space_id == 'Hdiv':
            self._space = DR.V2
        elif space_id == 'L2':
            self._space = DR.V3
        elif space_id == 'H1^3':
            self._space = DR.V0vec
        else:
            raise ValueError('Space not properly defined.')

        # Initialize accumulation matrix/vector in memory and create pointers to _data attribute
        self._vector = None
        if space_id in {'H1', 'L2'}:

            self._matrix = StencilMatrix(
                self._space.vector_space, self._space.vector_space)

            self._args_space = [self.space.vector_space.starts,
                                self.space.vector_space.ends,
                                self.space.vector_space.pads]

            self._args_data = [self.matrix._data]

            if do_vector:
                self._vector = StencilVector(self._space.vector_space)
                self._args_data += [self.vector._data]

        else:

            self._args_space = [self.space.vector_space.starts[0],
                                self.space.vector_space.starts[1],
                                self.space.vector_space.starts[2],
                                self.space.vector_space.ends[0],
                                self.space.vector_space.ends[1],
                                self.space.vector_space.ends[2],
                                self.space.vector_space.pads[0],
                                self.space.vector_space.pads[1],
                                self.space.vector_space.pads[2]]

            if symmetry is None:

                A11 = StencilMatrix(
                    self.space.vector_space.spaces[0], self.space.vector_space.spaces[0])
                A12 = StencilMatrix(
                    self.space.vector_space.spaces[1], self.space.vector_space.spaces[0])
                A13 = StencilMatrix(
                    self.space.vector_space.spaces[2], self.space.vector_space.spaces[0])
                A21 = StencilMatrix(
                    self.space.vector_space.spaces[0], self.space.vector_space.spaces[1])
                A22 = StencilMatrix(
                    self.space.vector_space.spaces[1], self.space.vector_space.spaces[1])
                A23 = StencilMatrix(
                    self.space.vector_space.spaces[2], self.space.vector_space.spaces[1])
                A31 = StencilMatrix(
                    self.space.vector_space.spaces[0], self.space.vector_space.spaces[2])
                A32 = StencilMatrix(
                    self.space.vector_space.spaces[1], self.space.vector_space.spaces[2])
                A33 = StencilMatrix(
                    self.space.vector_space.spaces[2], self.space.vector_space.spaces[2])
                dict_blocks = {(0, 0): A11, (0, 1): A12, (0, 2): A13, (1, 0): A21,
                               (1, 1): A22, (1, 2): A23, (2, 0): A31, (2, 1): A32, (2, 2): A33}

                self._matrix = BlockMatrix(self._space.vector_space,
                                           self._space.vector_space, blocks=dict_blocks)

                self._args_data = [self.matrix[0, 0]._data,
                                   self.matrix[0, 1]._data,
                                   self.matrix[0, 2]._data,
                                   self.matrix[1, 0]._data,
                                   self.matrix[1, 1]._data,
                                   self.matrix[1, 2]._data,
                                   self.matrix[2, 0]._data,
                                   self.matrix[2, 1]._data,
                                   self.matrix[2, 2]._data]

            elif symmetry == 'symm':

                A11 = StencilMatrix(
                    self.space.vector_space.spaces[0], self.space.vector_space.spaces[0])
                A12 = StencilMatrix(
                    self.space.vector_space.spaces[1], self.space.vector_space.spaces[0])
                A13 = StencilMatrix(
                    self.space.vector_space.spaces[2], self.space.vector_space.spaces[0])
                A22 = StencilMatrix(
                    self.space.vector_space.spaces[1], self.space.vector_space.spaces[1])
                A23 = StencilMatrix(
                    self.space.vector_space.spaces[2], self.space.vector_space.spaces[1])
                A33 = StencilMatrix(
                    self.space.vector_space.spaces[2], self.space.vector_space.spaces[2])
                dict_blocks = {(0, 0): A11, (0, 1): A12, (0, 2): A13,
                               (1, 1): A22, (1, 2): A23, (2, 2): A33}

                self._matrix = BlockMatrix(self._space.vector_space,
                                           self._space.vector_space, blocks=dict_blocks)

                self._args_data = [self.matrix[0, 0]._data,
                                   self.matrix[0, 1]._data,
                                   self.matrix[0, 2]._data,
                                   self.matrix[1, 1]._data,
                                   self.matrix[1, 2]._data,
                                   self.matrix[2, 2]._data]

            elif symmetry == 'asym':

                A12 = StencilMatrix(
                    self.space.vector_space.spaces[1], self.space.vector_space.spaces[0])
                A13 = StencilMatrix(
                    self.space.vector_space.spaces[2], self.space.vector_space.spaces[0])
                A23 = StencilMatrix(
                    self.space.vector_space.spaces[2], self.space.vector_space.spaces[1])
                dict_blocks = {(0, 1): A12, (0, 2): A13, (1, 2): A23}

                self._matrix = BlockMatrix(self._space.vector_space,
                                           self._space.vector_space, blocks=dict_blocks)

                self._args_data = [self.matrix[0, 1]._data,
                                   self.matrix[0, 2]._data,
                                   self.matrix[1, 2]._data]

            elif symmetry == 'diag':

                A11 = StencilMatrix(
                    self.space.vector_space.spaces[0], self.space.vector_space.spaces[0])
                A22 = StencilMatrix(
                    self.space.vector_space.spaces[1], self.space.vector_space.spaces[1])
                A33 = StencilMatrix(
                    self.space.vector_space.spaces[2], self.space.vector_space.spaces[2])
                dict_blocks = {(0, 0): A11, (1, 1): A22, (2, 2): A33}

                self._matrix = BlockMatrix(self._space.vector_space,
                                           self._space.vector_space, blocks=dict_blocks)

                self._args_data = [self.matrix[0, 0]._data,
                                   self.matrix[1, 1]._data,
                                   self.matrix[2, 2]._data]
            else:
                raise ValueError(
                    f'Symmetry attribute {symmetry} is not defined.')

            if do_vector:
                v1 = StencilVector(self.space.vector_space.spaces[0])
                v2 = StencilVector(self.space.vector_space.spaces[1])
                v3 = StencilVector(self.space.vector_space.spaces[2])
                list_blocks = [v1, v2, v3]

                self._vector = BlockVector(
                    self._space.vector_space, blocks=list_blocks)

                self._args_data += [self._vector[0]._data,
                                    self._vector[1]._data,
                                    self._vector[2]._data]

        # fixed arguments for the accumulator function
        self._args_fixed = [np.array(DR.p),
                            DR.V0.spaces[0].knots,
                            DR.V0.spaces[1].knots,
                            DR.V0.spaces[2].knots,
                            DOMAIN.kind_map,
                            np.array(DOMAIN.params_map),
                            np.array(DOMAIN.p),
                            DOMAIN.T[0], DOMAIN.T[1], DOMAIN.T[2],
                            DOMAIN.cx, DOMAIN.cy, DOMAIN.cz, ]

        # combine all arguments
        self._args = self.args_fixed + self.args_space + \
            self.args_data + list(self.args_add)

        # load the appropriate accumulation routine (pyccelized)
        self._accumulator = getattr(accums, self._accumulator_name)

        self._send_types, self._recv_types = self._create_buffer_types()

    def _create_buffer_types(self):
        """
        Creates the buffer types for the ghost region sender. Send types are only the slicing information;
        receving has to be saved in a temporary array and then added to the _data object with the correct indices.
        """
        from mpi4py import MPI

        send_types = []
        recv_buf = []

        pads = self.space.vector_space.pads

        for k, arg in enumerate(self.args_data):

            send_types.append([])
            recv_buf.append([])

            for di in range(3):
                send_types_temp = self._create_send_buffer_1dir(pads, arg.shape, di)
                recv_buf_temp = self._create_recv_buffer_1dir(pads, arg.shape, di)

                send_types[k] += [send_types_temp]
                recv_buf[k] += [recv_buf_temp]

        return send_types, recv_buf

    def _create_send_buffer_1dir(self, pads, arg_shape, di):
        """
        creates the send buffer in direction for stencil matrices and vectors. Send buffer is the indexing (MPI.Create_subarray)

        Parameters
        ----------
            starts : list
                contains the start indices in each direction

            ends : list
                contains the end indices in each direction

            pads : list
                contains the paddings in each direction

            arg_shape : tuple
                called by arg.shape

            di : int
                direction for which the send buffer is to be created
        """
        from mpi4py import MPI

        assert di in (0, 1, 2)

        subsizes_sub = list(arg_shape)
        
        if len(arg_shape) == 6:
            starts_sub_l = [pads[0], pads[1], pads[2], 0, 0, 0]
            starts_sub_r = [pads[0], pads[1], pads[2], 0, 0, 0]

        elif len(arg_shape) == 3:
            starts_sub_l = [pads[0], pads[1], pads[2]]
            starts_sub_r = [pads[0], pads[1], pads[2]]

        else:
            raise NotImplementedError('Unknown shape of argument!')
        
        for k in range(3):
            subsizes_sub[k] -= 2*pads[k]

        subsizes_sub[di] = pads[di]
        starts_sub_l[di] = 0
        starts_sub_r[di] = arg_shape[di] - pads[di]

        temp = {'l': MPI.DOUBLE.Create_subarray(
            sizes=list(arg_shape),
            subsizes=subsizes_sub,
            starts=starts_sub_l
        ).Commit(),
            'r': MPI.DOUBLE.Create_subarray(
            sizes=list(arg_shape),
            subsizes=subsizes_sub,
            starts=starts_sub_r
        ).Commit()
        }

        return temp

    def _create_recv_buffer_1dir(self, pads, arg_shape, di):
        """
        creates the receive buffer in direction for stencil matrices. The receive buffer is an empty numpy array
        and the indices where the ghost regions will have to be added to.

        Parameters
        ----------
            starts : list
                contains the start indices in each direction

            ends : list
                contains the end indices in each direction

            pads : list
                contains the paddings in each direction

            dir : int
                direction for which the send buffer is to be created
        """
        assert di in (0, 1, 2)

        subsizes_sub = [arg_shape[k] for k in range(len(arg_shape))]

        if len(arg_shape) == 6:
            inds_l = [slice(pads[0], -pads[0])] + [slice(pads[1], -pads[1])
                                                   ] + [slice(pads[2], -pads[2])] + [slice(None)]*3
            inds_r = [slice(pads[0], -pads[0])] + [slice(pads[1], -pads[1])
                                                   ] + [slice(pads[2], -pads[2])] + [slice(None)]*3

        elif len(arg_shape) == 3:
            inds_l = [slice(pads[0], -pads[0])] + \
                [slice(pads[1], -pads[1])] + [slice(pads[2], -pads[2])]
            inds_r = [slice(pads[0], -pads[0])] + \
                [slice(pads[1], -pads[1])] + [slice(pads[2], -pads[2])]

        else:
            raise NotImplementedError('Unknown shape of argument!')
        
        for k in range(3):
            subsizes_sub[k] -= 2*pads[k]

        subsizes_sub[di] = pads[di]
        inds_l[di] = slice(pads[di], 2*pads[di])
        inds_r[di] = slice(-2*pads[di], -pads[di])

        temp = {'l': {
            'buf': np.zeros(tuple(subsizes_sub), dtype=float),
            'inds': tuple(inds_l)},
            'r': {
            'buf': np.zeros(tuple(subsizes_sub), dtype=float),
            'inds': tuple(inds_r)}
        }

        return temp

    def accumulate(self, markers):
        '''Perform accumulation.

        Parameters
        ----------
            markers : array[float]
                Particle information in format (7, n_markers), including holes.'''

        # remove holes
        markers_wo_holes = markers[:, np.nonzero(markers[0] != -1.)[0]]

        # reset arrays
        for dat in self._args_data:
            dat[:] = 0.

        # accumulate
        self.accumulator(markers_wo_holes, *self.args)

        # use mpi
        self._send_ghost_regions()

        # update ghost regions
        self.update_ghost_regions()

    def _send_ghost_regions(self):
        '''Communicates the entries of the ghost regions between all the processes.'''

        mpi_comm = self._derham.comm

        for (dat, send_type, recv_type) in zip(self.args_data, self._send_types, self._recv_types):

            for di in range(3):
                left_neighbour = self._derham.neighbours[2*di]
                right_neighbour = self._derham.neighbours[2*di + 1]
                spl_kind = self._derham.spl_kind[di]

                if left_neighbour == -1 and right_neighbour == -1 and spl_kind == True:
                    self._send_ghost_regions_1dir_to_self(dat, di)

                else:
                    send_type_t = send_type[di]
                    recv_type_t = recv_type[di]

                    if len(dat.shape) == 6:
                        recv_type_t['l']['buf'][:, :, :, :, :, :] == 0.
                        recv_type_t['r']['buf'][:, :, :, :, :, :] == 0.
                    elif len(dat.shape) == 3:
                        recv_type_t['l']['buf'][:, :, :] == 0.
                        recv_type_t['r']['buf'][:, :, :] == 0.
                    else:
                        raise NotImplementedError(
                            'Unknown shape of data object!')

                    self._send_ghost_regions_1dir(
                        dat, left_neighbour, right_neighbour, send_type_t, recv_type_t)

                    mpi_comm.Barrier()

    def _send_ghost_regions_1dir_to_self(self, dat, di):
        """
        'Sends' the ghost regions from one process to itself in case the domain is periodic in this direction but
        the domain is not split, i.e. the process is its own neighbour.

        Parameters
        ----------
            dat : array
                Stencil ._data object; numpy array
            
            dir : int
                direction in which the ghost 'sending' has to be done
        """

        pads = self.space.vector_space.pads

        # Make indices for reading out and writing to the _data object
        if len(dat.shape) == 6:
            inds_read_l = [slice(pads[0], -pads[0])] + [slice(pads[1], -pads[1])
                                                        ] + [slice(pads[2], -pads[2])] + [slice(None)]*3
            inds_read_r = [slice(pads[0], -pads[0])] + [slice(pads[1], -pads[1])
                                                        ] + [slice(pads[2], -pads[2])] + [slice(None)]*3

            inds_write_l = [slice(pads[0], -pads[0])] + [slice(pads[1], -pads[1])
                                                         ] + [slice(pads[2], -pads[2])] + [slice(None)]*3
            inds_write_r = [slice(pads[0], -pads[0])] + [slice(pads[1], -pads[1])
                                                         ] + [slice(pads[2], -pads[2])] + [slice(None)]*3

        elif len(dat.shape) == 3:
            inds_read_l = [slice(pads[0], -pads[0])] + [slice(pads[1], -pads[1])
                                                        ] + [slice(pads[2], -pads[2])]
            inds_read_r = [slice(pads[0], -pads[0])] + [slice(pads[1], -pads[1])
                                                        ] + [slice(pads[2], -pads[2])]

            inds_write_l = [slice(pads[0], -pads[0])] + [slice(pads[1], -pads[1])
                                                         ] + [slice(pads[2], -pads[2])]
            inds_write_r = [slice(pads[0], -pads[0])] + [slice(pads[1], -pads[1])
                                                         ] + [slice(pads[2], -pads[2])]

        else:
            raise NotImplementedError('Unknown object of data object!')

        inds_read_l[di] = slice(0, pads[di]+1)
        inds_read_r[di] = slice(-pads[di], dat.shape[di] - pads[di] + 1)

        inds_write_l[di] = slice(pads[di], 2*pads[di] + 1)
        inds_write_r[di] = slice(
            dat.shape[di] - 2*pads[di], dat.shape[di] - pads[di] + 1)

        # have to convert to tuple to use as indices with slicing
        inds_read_l = tuple(inds_read_l)
        inds_read_r = tuple(inds_read_r)

        inds_write_l = tuple(inds_write_l)
        inds_write_r = tuple(inds_write_r)

        dat[inds_write_l] += dat[inds_read_r]
        dat[inds_write_r] += dat[inds_read_l]

    def _send_ghost_regions_1dir(self, dat, left_neighbour, right_neighbour, send_type, recv_type):
        """
        Communicates the entries of the ghost regions between all the processes using non-blocking communication for one direction

        Parameters
        ----------
            dat : array
                Stencil ._data object; numpy array

            left_neighbour : int
                tag of the left neighbour

            right_neighbour : int
                tag of the right neighbour

            send_type : dict
                Dictionary with keys 'l' and 'r', values are MPI.Create_subarrays object; created by _create_buffer_types()

            recv_type : dict
                Dictionary with keys 'l' and 'r', values are dictionaries with keys 'buf' and 'ind' and values are numpy arrays; created by _create_buffer_types()
        """
        from mpi4py import MPI

        mpi_comm = self._derham.comm
        rank = mpi_comm.Get_rank()

        if left_neighbour != -1:
            mpi_comm.Isend(
                (dat, 1, send_type['l']), dest=left_neighbour, tag=rank + 100)

        if right_neighbour != -1:
            mpi_comm.Isend(
                (dat, 1, send_type['r']), dest=right_neighbour, tag=rank + 300)

        if left_neighbour != -1:
            req_l = mpi_comm.Irecv(
                recv_type['l']['buf'], source=left_neighbour, tag=left_neighbour + 300)
            re_l = False
            while not re_l:
                re_l = MPI.Request.Test(req_l)

            dat[recv_type['l']['inds']] += recv_type['l']['buf']

        if right_neighbour != -1:
            req_r = mpi_comm.Irecv(
                recv_type['r']['buf'], source=right_neighbour, tag=right_neighbour + 100)
            re_r = False
            while not re_r:
                re_r = MPI.Request.Test(req_r)

            dat[recv_type['r']['inds']] += recv_type['r']['buf']

        mpi_comm.Barrier()

    def update_ghost_regions(self):
        "updates ghost regions of all attributes"
        self.matrix.update_ghost_regions()
        if self._do_vector:
            self.vector.update_ghost_regions()

    @property
    def space(self):
        '''Discrete space of the matrix/vector (Psydac object).'''
        return self._space

    @property
    def matrix(self):
        '''Accumulation matrix (Stencil- or BlockMatrix)'''
        return self._matrix

    @property
    def vector(self):
        '''Accumulation vector, optional (Stencil- or BlockMatrix)'''
        return self._vector

    @property
    def space_id(self):
        '''Space identifier for the matrix/vector (H1, Hcurl, Hdiv or L2) to be accumulated into.'''
        return self._space_id

    @property
    def accumulator_name(self):
        '''String that identifies which function to load from the module struphy.pic.accum_kernels.'''
        return self._accumulator_name

    @property
    def accumulator(self):
        '''The function loaded from the module struphy.pic.accum_kernels.'''
        return self._accumulator

    @property
    def args(self):
        '''List of arguments passed to the accumulator, composed of 
        mandatory arguments, data arguments and additional arguments.'''
        return self._args

    @property
    def args_fixed(self):
        '''List of mandatory arguments for the accumulator.'''
        return self._args_fixed

    @property
    def args_space(self):
        '''List of space-dependent arguments (starts, ends, pads) for the accumulator.'''
        return self._args_space

    @property
    def args_data(self):
        '''List of data arguments for the accumulator.'''
        return self._args_data

    @property
    def args_add(self):
        '''List of additional arguments for the accumulator.'''
        return self._args_add

    @property
    def symmetry(self):
        '''In case of space_id=Hcurl/Hdiv, the symmetry property of the block matrix: diag, asym, symm or None (=full matrix, default)'''
        return self._symmetry
