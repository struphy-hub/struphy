import numpy as np

from psydac.linalg.stencil import StencilVector, StencilMatrix
from psydac.linalg.block import BlockVector, BlockMatrix
from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL

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

        # only for M1 Mac users
        PSYDAC_BACKEND_GPYCCEL['flags'] = '-O3 -march=native -mtune=native -ffast-math -ffree-line-length-none'

        # Initialize accumulation matrix/vector in memory and create pointers to _data attribute
        self._vector = None
        if space_id in {'H1', 'L2'}:

            self._matrix = StencilMatrix(
                self._space.vector_space, self._space.vector_space, backend=PSYDAC_BACKEND_GPYCCEL)

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
                    self.space.vector_space.spaces[0], self.space.vector_space.spaces[0], backend=PSYDAC_BACKEND_GPYCCEL)
                A12 = StencilMatrix(
                    self.space.vector_space.spaces[1], self.space.vector_space.spaces[0], backend=PSYDAC_BACKEND_GPYCCEL)
                A13 = StencilMatrix(
                    self.space.vector_space.spaces[2], self.space.vector_space.spaces[0], backend=PSYDAC_BACKEND_GPYCCEL)
                A21 = StencilMatrix(
                    self.space.vector_space.spaces[0], self.space.vector_space.spaces[1], backend=PSYDAC_BACKEND_GPYCCEL)
                A22 = StencilMatrix(
                    self.space.vector_space.spaces[1], self.space.vector_space.spaces[1], backend=PSYDAC_BACKEND_GPYCCEL)
                A23 = StencilMatrix(
                    self.space.vector_space.spaces[2], self.space.vector_space.spaces[1], backend=PSYDAC_BACKEND_GPYCCEL)
                A31 = StencilMatrix(
                    self.space.vector_space.spaces[0], self.space.vector_space.spaces[2], backend=PSYDAC_BACKEND_GPYCCEL)
                A32 = StencilMatrix(
                    self.space.vector_space.spaces[1], self.space.vector_space.spaces[2], backend=PSYDAC_BACKEND_GPYCCEL)
                A33 = StencilMatrix(
                    self.space.vector_space.spaces[2], self.space.vector_space.spaces[2], backend=PSYDAC_BACKEND_GPYCCEL)
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
                    self.space.vector_space.spaces[0], self.space.vector_space.spaces[0], backend=PSYDAC_BACKEND_GPYCCEL)
                A12 = StencilMatrix(
                    self.space.vector_space.spaces[1], self.space.vector_space.spaces[0], backend=PSYDAC_BACKEND_GPYCCEL)
                A13 = StencilMatrix(
                    self.space.vector_space.spaces[2], self.space.vector_space.spaces[0], backend=PSYDAC_BACKEND_GPYCCEL)
                A22 = StencilMatrix(
                    self.space.vector_space.spaces[1], self.space.vector_space.spaces[1], backend=PSYDAC_BACKEND_GPYCCEL)
                A23 = StencilMatrix(
                    self.space.vector_space.spaces[2], self.space.vector_space.spaces[1], backend=PSYDAC_BACKEND_GPYCCEL)
                A33 = StencilMatrix(
                    self.space.vector_space.spaces[2], self.space.vector_space.spaces[2], backend=PSYDAC_BACKEND_GPYCCEL)
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
                    self.space.vector_space.spaces[1], self.space.vector_space.spaces[0], backend=PSYDAC_BACKEND_GPYCCEL)
                A13 = StencilMatrix(
                    self.space.vector_space.spaces[2], self.space.vector_space.spaces[0], backend=PSYDAC_BACKEND_GPYCCEL)
                A23 = StencilMatrix(
                    self.space.vector_space.spaces[2], self.space.vector_space.spaces[1], backend=PSYDAC_BACKEND_GPYCCEL)
                dict_blocks = {(0, 1): A12, (0, 2): A13, (1, 2): A23}

                self._matrix = BlockMatrix(self._space.vector_space,
                                           self._space.vector_space, blocks=dict_blocks)

                self._args_data = [self.matrix[0, 1]._data,
                                   self.matrix[0, 2]._data,
                                   self.matrix[1, 2]._data]

            elif symmetry == 'diag':

                A11 = StencilMatrix(
                    self.space.vector_space.spaces[0], self.space.vector_space.spaces[0], backend=PSYDAC_BACKEND_GPYCCEL)
                A22 = StencilMatrix(
                    self.space.vector_space.spaces[1], self.space.vector_space.spaces[1], backend=PSYDAC_BACKEND_GPYCCEL)
                A33 = StencilMatrix(
                    self.space.vector_space.spaces[2], self.space.vector_space.spaces[2], backend=PSYDAC_BACKEND_GPYCCEL)
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
        Buffers have the same structure as struphy.psydac_api.psydac_derham.Derham.neighbours, i.e. a 3d array with shape (3,3,3)
        and are initialized with None. If the process has a neighbour, the send/recv information is filled in.
        """

        send_types = []
        recv_buf = []

        neighbours = self._derham.neighbours

        pads = self.space.vector_space.pads

        for k, arg in enumerate(self.args_data):
            for comp, neigh in np.ndenumerate(neighbours):

                send_types.append(np.array([[[None]*3]*3]*3))
                recv_buf.append(np.array([[[None]*3]*3]*3))

                if neigh != -1:
                    send_types[k][comp] = self._create_send_buffer_1_comp(
                        pads, arg.shape, comp)
                    recv_buf[k][comp] = self._create_recv_buffer_1_comp(
                        pads, arg.shape, comp)

        return send_types, recv_buf

    def _create_send_buffer_1_comp(self, pads, arg_shape, comp):
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

            comp : tuple
                component for which the send buffer is to be created; entries are in {0,1,2}
        """
        from mpi4py import MPI

        subsizes_sub = list(arg_shape)

        if len(arg_shape) == 6:
            starts_sub = [pads[0], pads[1], pads[2], 0, 0, 0]

        elif len(arg_shape) == 3:
            starts_sub = [pads[0], pads[1], pads[2]]

        else:
            raise NotImplementedError('Unknown shape of argument!')

        for k in range(3):
            subsizes_sub[k] -= 2*pads[k]

        for k, co in enumerate(comp):
            # if left neighbour
            if co == 0:
                subsizes_sub[k] = pads[k]
                starts_sub[k] = 0

            # if middle neighbour
            elif co == 1:
                continue

            # if right neighbour
            elif co == 2:
                subsizes_sub[k] = pads[k]
                starts_sub[k] = arg_shape[k] - pads[k]

            else:
                raise ValueError('Unknown value for component!')

        temp = MPI.DOUBLE.Create_subarray(
            sizes=list(arg_shape),
            subsizes=subsizes_sub,
            starts=starts_sub
        ).Commit()

        return temp

    def _create_recv_buffer_1_comp(self, pads, arg_shape, comp):
        """
        creates the receive buffer in direction for stencil matrices. The receive buffer is an empty numpy array
        and the indices where the ghost regions will have to be added to. Left and right are swapped compared to
        send-types since _send_ghost_regions() does the sending component-wise. Sending to the left means 

        Parameters
        ----------
            starts : list
                contains the start indices in each direction

            ends : list
                contains the end indices in each direction

            pads : list
                contains the paddings in each direction

            comp : tuple
                component for which the receive buffer is to be created; entries are in {0,1,2}
        """

        subsizes_sub = [arg_shape[k] for k in range(len(arg_shape))]

        if len(arg_shape) == 6:
            inds = [slice(pads[0], -pads[0])] + [slice(pads[1], -pads[1])] + [slice(pads[2], -pads[2])] \
                + [slice(None)]*3

        elif len(arg_shape) == 3:
            inds = [slice(pads[0], -pads[0])] + \
                [slice(pads[1], -pads[1])] + [slice(pads[2], -pads[2])]

        else:
            raise NotImplementedError('Unknown shape of argument!')

        for k in range(3):
            subsizes_sub[k] -= 2*pads[k]

        for k, co in enumerate(comp):
            # if left neighbour
            if co == 0:
                subsizes_sub[k] = pads[k]
                inds[k] = slice(pads[k], 2*pads[k])

            # if middle neighbour
            elif co == 1:
                continue

            # if right neighbour
            elif co == 2:
                subsizes_sub[k] = pads[k]
                inds[k] = slice(-2*pads[k], -pads[k])

            else:
                raise ValueError('Unknown value for component!')

        temp = {
            'buf': np.zeros(tuple(subsizes_sub), dtype=float),
            'inds': tuple(inds)
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
        """
        Communicates the ghost regions between all processes using non-blocking communication.
        In order to avoid communication overhead a sending in one direction component is always accompanied
        by a receiving (if neighbour is not -1) in the inverted direction. This guarantees that every send signal
        is received in the same comp iteration.
        """

        comm = self._derham.comm
        neighbours = self._derham.neighbours

        for dat, send_type, recv_type in zip(self.args_data, self._send_types, self._recv_types):

            for comp, send_neigh in np.ndenumerate(neighbours):
                inv_comp = self._invert_component(comp)
                recv_neigh = neighbours[inv_comp]
                
                if send_neigh != -1:
                    send_type_comp = send_type[comp]
                    # sending to component direction.
                    self._send_ghost_regions_1_comp(dat, send_neigh, send_type_comp, comp)
                
                if recv_neigh != -1:
                    recv_type_comp = recv_type[inv_comp]
                    # Receiving from the inverted component direction if there is a neighbour
                    self._recv_ghost_regions_1_comp(dat, recv_neigh, recv_type_comp, comp)

                    if len(dat.shape) == 6:
                        recv_type_comp['buf'][:, :, :, :, :, :] == 0.
                        recv_type_comp['buf'][:, :, :, :, :, :] == 0.
                    elif len(dat.shape) == 3:
                        recv_type_comp['buf'][:, :, :] == 0.
                        recv_type_comp['buf'][:, :, :] == 0.
                    else:
                        raise NotImplementedError('Unknown shape of data object!')

                comm.Barrier()

    def _send_ghost_regions_1_comp(self, dat, neighbour, send_type, comp):
        """
        Does the sending for one direction component using non-blocking communication.

        Parameters
        ----------
            dat : array
                Stencil ._data object; numpy array

            neighbour : int
                tag of the neighbour or -1 if no neighbour

            send_type : MPI.Create_subarrays object
                MPI.Create_subarrays object; created by _create_buffer_types()

            comp : tuple
                component direction into which the ghost region is to be sent; entries are in {0,1,2}
        """

        comm = self._derham.comm
        rank = comm.Get_rank()

        send_tag = rank + 1000*comp[0] + 100*comp[1] + 10*comp[2]

        comm.Isend(
            (dat, 1, send_type), dest=neighbour, tag=send_tag)

    def _recv_ghost_regions_1_comp(self, dat, neighbour, recv_type, comp):
        """
        Does the receving for one direction component using non-blocking communication.

        Parameters
        ----------
            dat : array
                Stencil ._data object; numpy array

            neighbour : int
                tag of the neighbour or -1 if no neighbour

            recv_type : dict
                dictionary with keys 'buf' and 'inds' and values are numpy arrays; created by _create_buffer_types()

            comp : tuple
                component direction from which the ghost region was sent (is only used for computing the tag); entries are in {0,1,2}
        """
        from mpi4py import MPI

        comm = self._derham.comm

        recv_tag = neighbour + 1000*comp[0] + 100*comp[1] + 10*comp[2]

        req_l = comm.Irecv(
            recv_type['buf'], source=neighbour, tag=recv_tag)

        re_l = False
        while not re_l:
            re_l = MPI.Request.Test(req_l)

        dat[recv_type['inds']] += recv_type['buf']

    def _invert_component(self, comp):
        """
        Given a component in the 3x3x3 cube this function 'inverts' it, i.e. reflects
        it on the central component (1,1,1)
        
        Parameters
        ----------
            comp : tuple
                component index in the 3x3x3 cube; entries are in {0,1,2}
        
        Returns
        -------
            res : tuple
                inverse component to input
        """
        res = [-1,-1,-1]

        for k,co in enumerate(comp):
            if co == 1:
                res[k] = 1
            elif co == 0:
                res[k] = 2
            elif co == 2:
                res[k] = 0
            else:
                raise ValueError('Unknown component value!')
        
        return tuple(res)

    def update_ghost_regions(self):
        """updates ghost regions of all attributes"""
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
