import numpy as np

from psydac.linalg.stencil import StencilVector, StencilMatrix
from psydac.linalg.block import BlockVector, BlockMatrix

import struphy.pic.accumulators as accums


class Accumulator():
    '''Struphy accumulation matrices and vectors of the form

    .. math::

        M^{\mu,\\nu}_{ijk,mno} &= \sum_p P^\mu_{ijk}(\eta_p) * A^{\mu,\\nu}_p * (P^\\nu_{mno})^\\top(\eta_p)  \qquad  (\mu,\\nu = 1,2,3)

        V^\mu_{ijk} &= \sum_p P^\mu_{ijk}(\eta_p) * B^\mu_p

    where :math:`p` runs over the particles, :math:`P^\mu_{ijk}(\eta_p)` denotes the :math:`ijk`-th basis function 
    of the :math:`\mu`-th component of a Derham space (V0, V1, V2, V3) evaluated at the particle position :math:`\eta_p`.
    
    :math:`A^{\mu,\\nu}_p` and :math:`B^\mu_p` are particle-dependent "filling functions", 
    to be defined in the module **struphy.pic.accumulators**.

    Parameters
    ----------
        DOMAIN : struphy.geometry.domain_3d.Domain
            Domain object for mapping evaluations.

        DR : struphy.psydac_api.psydac_derham.DerhamBuild
            Discrete Derham complex. 

        space_id : str
            Space identifier for the matrix/vector (H1, Hcurl, Hdiv, L2 or H1^3) to be accumulated into.

        accumulator_name : str 
            Name of accumulator function to be loaded from struphy/pic/accumulators.py.

        args_add : list
            Additional arguments to be passed to the accumulator function, besides the mandatory arguments
            which are prepared automatically (spline bases info, mapping info, data arrays). 
            Examples would be parameters for a background kinetic distribution or spline coefficients of a background magnetic field. 
            Entries must be pyccel-conform types.

        do_vector : bool 
            True if, additionally to a matrix, a vector in the same space is to be accumulated. Default=False.

        symmetry : str
            In case of space_id=Hcurl/Hdiv, the symmetry property of the block matrix: diag, asym, symm or None (=full matrix, default)
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

            self._matrix = BlockMatrix(
                self._space.vector_space, self._space.vector_space)

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
                self._args_data = [self.matrix[0, 0]._data,
                                   self.matrix[0, 1]._data,
                                   self.matrix[0, 2]._data,
                                   self.matrix[1, 1]._data,
                                   self.matrix[1, 2]._data,
                                   self.matrix[2, 2]._data]
            elif symmetry == 'asym':
                self._args_data = [self.matrix[0, 1]._data,
                                   self.matrix[0, 2]._data,
                                   self.matrix[1, 2]._data]
            elif symmetry == 'diag':
                self._args_data = [self.matrix[0, 0]._data,
                                   self.matrix[1, 1]._data,
                                   self.matrix[2, 2]._data]
            else:
                raise ValueError(
                    f'Symmetry attribute {symmetry} is not defined.')

            if do_vector:
                self._vector = BlockVector(self._space.vector_space)
                self._args_data += [self._vector[0]._data,
                                    self._vector[1]._data,
                                    self._vector[2]._data]

        # fixed arguments for the accumulator function
        self._args_fixed = [np.array(DR.p),
                            DR.V0.spaces[0].knots,
                            DR.V0.spaces[1].knots,
                            DR.V0.spaces[2].knots,
                            DOMAIN.keys_map[DOMAIN.kind_map],
                            np.array(DOMAIN.params_map),
                            np.array(DOMAIN.p),
                            DOMAIN.T[0],
                            DOMAIN.T[1],
                            DOMAIN.T[2],
                            DOMAIN.cx,
                            DOMAIN.cy,
                            DOMAIN.cz, ]

        # combine all arguments
        self._args = self.args_fixed + self.args_space + self.args_data + self.args_add

        # load the appropriate accumulation routine (pyccelized)
        self._accumulator = getattr(accums, self._accumulator_name)

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

    def _send_ghost_regions():
        '''TODO'''
        pass

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
        '''String that identifies which function to load from the module struphy.pic.accumulators.'''
        return self._accumulator_name

    @property
    def accumulator(self):
        '''The function loaded from the module struphy.pic.accumulators.'''
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
