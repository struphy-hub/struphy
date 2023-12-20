'Base classes for particle deposition (accumulation) on the grid.'


import numpy as np

from psydac.linalg.stencil import StencilVector, StencilMatrix
from psydac.linalg.block import BlockVector

from struphy.feec.mass import WeightedMassOperator

import struphy.pic.accumulation.accum_kernels as accums
import struphy.pic.accumulation.accum_kernels_gc as accums_gc


class Accumulator:
    r"""
    Struphy accumulation matrices and vectors of the form

    .. math::

        M^{\mu,\nu}_{ijk,mno} &= \sum_p \Lambda^\mu_{ijk}(\eta_p) * A^{\mu,\nu}_p * (\Lambda^\nu_{mno})^\top(\eta_p)  \qquad  (\mu,\nu = 1,2,3)

        V^\mu_{ijk} &= \sum_p \Lambda^\mu_{ijk}(\eta_p) * B^\mu_p

    where :math:`p` runs over the particles, :math:`\Lambda^\mu_{ijk}(\eta_p)` denotes the :math:`ijk`-th basis function
    of the :math:`\mu`-th component of a Derham space (V0, V1, V2, V3) evaluated at the particle position :math:`\eta_p`.

    :math:`A^{\mu,\nu}_p` and :math:`B^\mu_p` are particle-dependent "filling functions",
    to be defined in the module **struphy.pic.accumulation.accum_kernels**.

    Parameters
    ----------
    derham : struphy.feec.psydac_derham.Derham
        Discrete Derham complex.

    domain : struphy.geometry.domains
        Mapping info for evaluating metric coefficients.

    space_id : str
        Space identifier for the matrix/vector (H1, Hcurl, Hdiv, L2 or H1vec) to be accumulated into.

    kernel_name : str
        Name of accumulation kernel.

    add_vector : bool
        True if, additionally to a matrix, a vector in the same space is to be accumulated. Default=False.

    symmetry : str
        In case of space_id=Hcurl/Hdiv, the symmetry property of the block matrix: diag, asym, symm, pressure or None (=full matrix, default)

    Note
    ----
        Struphy accumulation kernels called by ``Accumulator`` objects must be added to ``struphy/pic/accumulation/accum_kernels.py`` 
        (6D particles) or ``struphy/pic/accumulation/accum_kernels_gc.py`` (5D particles), see :ref:`accum_kernels`
        and :ref:`accum_kernels_gc` for details.
    """

    def __init__(self, derham, domain, space_id, kernel_name, add_vector=False, symmetry=None):

        self._derham = derham
        self._domain = domain

        self._space_id = space_id
        self._symmetry = symmetry

        self._form = derham.space_to_form[space_id]

        # initialize matrices (instances of WeightedMassOperator)
        self._operators = []

        # special treatment in model LinearMHDVlasovPC (symmetry=pressure, six symmetric BlockMatrices are needed)
        if symmetry == 'pressure':
            for _ in range(6):
                self._operators += [WeightedMassOperator(
                    derham.Vh_fem[self.form], 
                    derham.Vh_fem[self.form],
                    V_extraction_op=derham.extraction_ops[self.form],
                    W_extraction_op=derham.extraction_ops[self.form],
                    V_boundary_op=derham.boundary_ops[self.form],
                    W_boundary_op=derham.boundary_ops[self.form],
                    weights_info='symm', transposed=False)]

        # "normal" treatment (just one matrix)
        else:
            self._operators += [WeightedMassOperator(
                derham.Vh_fem[self.form], 
                derham.Vh_fem[self.form],
                V_extraction_op=derham.extraction_ops[self.form],
                W_extraction_op=derham.extraction_ops[self.form],
                V_boundary_op=derham.boundary_ops[self.form],
                W_boundary_op=derham.boundary_ops[self.form],
                weights_info=symmetry, transposed=False)]

        # collect all _data attributes needed in accumulation kernel
        self._args_data = ()

        for op in self._operators:
            if isinstance(op.matrix, StencilMatrix):
                self._args_data += (op.matrix._data,)
            else:
                for a, row in enumerate(op.matrix.blocks):
                    for b, bl in enumerate(row):
                        if symmetry in ['pressure', 'symm', 'asym', 'diag']:
                            if b >= a and bl is not None:
                                self._args_data += (bl._data,)
                        else:
                            if bl is not None:
                                self._args_data += (bl._data,)

        # initialize vectors
        self._vectors = []

        if add_vector:
            # special treatment in model LinearMHDVlasovPC (symmetry=pressure, three BlockVectors are needed)
            if symmetry == 'pressure':
                for i in range(3):
                    self._vectors += [BlockVector(derham.Vh[self.form])]

            # normal treatment (just one vector)
            else:
                for op in self._operators:
                    if isinstance(op.matrix, StencilMatrix):
                        self._vectors += [StencilVector(op.matrix.domain)]
                    else:
                        self._vectors += [BlockVector(op.matrix.domain)]

            for vec in self._vectors:
                if isinstance(vec, StencilVector):
                    self._args_data += (vec._data,)
                else:
                    for bl in vec.blocks:
                        self._args_data += (bl._data,)

        # fixed FEM arguments for the accumulator kernel
        self._args_fem = (np.array(derham.p),
                          derham.Vh_fem['0'].knots[0],
                          derham.Vh_fem['0'].knots[1],
                          derham.Vh_fem['0'].knots[2],
                          np.array(derham.Vh['0'].starts))

        # load the appropriate accumulation kernel (pyccelized, fast)
        self._kernel_name = kernel_name
        self._kernel = None

        objs = [accums, accums_gc]
        for obj in objs:
            try:
                self._kernel = getattr(obj, self.kernel_name)
            except AttributeError:
                pass
        assert self.kernel is not None

    @property
    def derham(self):
        """ Discrete Derham complex on the logical unit cube.
        """
        return self._derham

    @property
    def domain(self):
        """ Mapping info for evaluating metric coefficients.
        """
        return self._domain

    @property
    def space_id(self):
        """ Space identifier for the matrix/vector (H1, Hcurl, Hdiv, L2 or H1vec) to be accumulated into.
        """
        return self._space_id

    @property
    def form(self):
        """ p-form ("0", "1", "2", "3" or "v") to be accumulated into.
        """
        return self._form

    @property
    def symmetry(self):
        """ Symmetry of the accumulation matrix (diagonal, symmetric, asymmetric, etc.).
        """
        return self._symmetry

    @property
    def operators(self):
        """ List of WeightedMassOperators of the accumulator. Matrices can be accessed e.g. with operators[0].matrix.
        """
        return self._operators

    @property
    def vectors(self):
        """ List of Stencil-/Block-/PolarVectors of the accumulator.
        """
        out = []
        for vec in self._vectors:
            out += [self._derham.boundary_ops[self.form].dot(
                self._derham.extraction_ops[self.form].dot(vec))]

        return out

    @property
    def kernel_name(self):
        """ String that identifies the accumulation kernel.
        """
        return self._kernel_name

    @property
    def kernel(self):
        """ The kernel loaded from the module struphy.pic.accum_kernels.
        """
        return self._kernel

    def accumulate(self, particles, *args_add, **args_control):
        """
        Performs the accumulation into the matrix/vector by calling the chosen accumulation kernel and additional analytical contributions (control variate, optional).

        Parameters
        ----------
        particles : struphy.pic.particles.Particles
            Particles object holding the markers information in format particles.markers.shape == (n_markers, :).

        *args_add
            Additional arguments to be passed to the accumulator kernel, besides the mandatory arguments
            which are prepared automatically (spline bases info, mapping info, data arrays).
            Examples would be parameters for a background kinetic distribution or spline coefficients of a background magnetic field.
            Entries must be pyccel-conform types.

        **args_control
            Keyword arguments for an analytical control variate correction in the accumulation step. Possible keywords are 'control_vec' for a vector correction or 'control_mat' for a matrix correction. Values are a 1d (vector) or 2d (matrix) list with callables or np.ndarrays used for the correction.
        """

        # flags for break
        vec_finished = False
        mat_finished = False

        # reset data
        for dat in self._args_data:
            dat[:] = 0.

        # accumulate into matrix (and vector) with markers
        self.kernel(particles.markers, particles.n_mks,
                    *self._args_fem, *self._domain.args_map,
                    *self._args_data, *args_add)
        # add analytical contribution (control variate) to matrix
        if 'control_mat' in args_control:
            self._operators[0].assemble(weights=args_control['control_mat'])

        # add analytical contribution (control variate) to vector
        if 'control_vec' in args_control and len(self._vectors) > 0:
            WeightedMassOperator.assemble_vec(self._derham.Vh_fem[self._derham.space_to_form[self._space_id]],
                                              self._vectors[0], weight=args_control['control_vec'],
                                              clear=False)

            vec_finished = True

        # add analytical contribution (control variate) to matrix and finish
        if 'control_mat' in args_control:
            self._operators[0].assemble(weights=args_control['control_mat'],
                                        clear=False, verbose=False)
            mat_finished = True

        # finish vector: accumulate ghost regions and update ghost regions
        if not vec_finished:
            for vec in self._vectors:
                vec.exchange_assembly_data()
                vec.update_ghost_regions()

        # finish matrix: accumulate ghost regions, update ghost regions and copy data for symmetric/antisymmetric block matrices
        if not mat_finished:
            for op in self._operators:
                op.matrix.exchange_assembly_data()
                op.matrix.update_ghost_regions()

            if self.symmetry == 'symm':

                self._operators[0].matrix[1, 0]._data[:] = \
                    self._operators[0].matrix[0, 1].T._data
                self._operators[0].matrix[2, 0]._data[:] = \
                    self._operators[0].matrix[0, 2].T._data
                self._operators[0].matrix[2, 1]._data[:] = \
                    self._operators[0].matrix[1, 2].T._data

            elif self.symmetry == 'asym':

                self._operators[0].matrix[1, 0]._data[:] = - \
                    self._operators[0].matrix[0, 1].T._data
                self._operators[0].matrix[2, 0]._data[:] = - \
                    self._operators[0].matrix[0, 2].T._data
                self._operators[0].matrix[2, 1]._data[:] = - \
                    self._operators[0].matrix[1, 2].T._data

            elif self.symmetry == 'pressure':
                for i in range(6):
                    self._operators[i].matrix[1, 0]._data[:] = \
                        self._operators[i].matrix[0, 1].T._data
                    self._operators[i].matrix[2, 0]._data[:] = \
                        self._operators[i].matrix[0, 2].T._data
                    self._operators[i].matrix[2, 1]._data[:] = \
                        self._operators[i].matrix[1, 2].T._data


class AccumulatorVector:
    r"""
    Struphy accumulation for only a vector of the form

    .. math::

        V^\mu_{ijk} = \sum_p \Lambda^\mu_{ijk}(\eta_p) * B^\mu_p

    where :math:`p` runs over the particles, :math:`\Lambda^\mu_{ijk}(\eta_p)` denotes the :math:`ijk`-th basis function
    of the :math:`\mu`-th component of a Derham space (V0, V1, V2, V3) evaluated at the particle position :math:`\eta_p`.

    :math:`B^\mu_p` is a particle-dependent "filling function", to be defined in the module **struphy.pic.accumulation.accum_kernels**.

    Parameters
    ----------
    derham : struphy.feec.psydac_derham.Derham
        Discrete Derham complex.

    domain : struphy.geometry.domains
        Mapping info for evaluating metric coefficients.

    space_id : str
        Space identifier for the matrix/vector (H1, Hcurl, Hdiv, L2 or H1vec) to be accumulated into.

    kernel_name : str
        Name of accumulation kernel.

    Note
    ----
    Struphy accumulation kernels called by ``Accumulator`` objects should be added to ``struphy/pic/accumulation/accum_kernels.py``. 
    Please follow the docstring in `struphy.pic.accumulation.accum_kernels.a_docstring`.
    """

    def __init__(self, derham, domain, space_id, kernel_name):

        self._derham = derham
        self._domain = domain

        self._space_id = space_id

        self._form = derham.space_to_form[space_id]

        # initialize vectors
        self._vectors = []

        # collect all _data attributes needed in accumulation kernel
        self._args_data = ()

        if space_id in ("H1", "L2"):
            self._vectors += [StencilVector(
                derham.Vh_fem[self.form].vector_space)]

        elif space_id in ("Hcurl", "Hdiv", "H1vec"):
            self._vectors += [BlockVector(derham.Vh_fem[self.form].vector_space)]

        for vec in self._vectors:
            if isinstance(vec, StencilVector):
                self._args_data += (vec._data,)
            else:
                for bl in vec.blocks:
                    self._args_data += (bl._data,)

        # fixed FEM arguments for the accumulator kernel
        self._args_fem = (np.array(derham.p),
                          derham.Vh_fem['0'].knots[0],
                          derham.Vh_fem['0'].knots[1],
                          derham.Vh_fem['0'].knots[2],
                          np.array(derham.Vh['0'].starts))

        # load the appropriate accumulation kernel (pyccelized, fast)
        self._kernel_name = kernel_name
        self._kernel = None

        objs = [accums, accums_gc]
        for obj in objs:
            try:
                self._kernel = getattr(obj, self.kernel_name)
            except AttributeError:
                pass
        assert self.kernel is not None

    @property
    def derham(self):
        """ Discrete Derham complex on the logical unit cube.
        """
        return self._derham

    @property
    def domain(self):
        """ Mapping info for evaluating metric coefficients.
        """
        return self._domain

    @property
    def space_id(self):
        """ Space identifier for the matrix/vector (H1, Hcurl, Hdiv, L2 or H1vec) to be accumulated into.
        """
        return self._space_id

    @property
    def form(self):
        """ p-form ("0", "1", "2", "3" or "v") to be accumulated into.
        """
        return self._form

    @property
    def vectors(self):
        """ List of Stencil-/Block-/PolarVectors of the accumulator.
        """
        out = []
        for vec in self._vectors:
            out += [self._derham.boundary_ops[self.form].dot(
                self._derham.extraction_ops[self.form].dot(vec))]

        return out

    @property
    def kernel_name(self):
        """ String that identifies the accumulation kernel.
        """
        return self._kernel_name

    @property
    def kernel(self):
        """ The accumulation kernel.
        """
        return self._kernel

    def accumulate(self, particles, *args_add, **args_control):
        """
        Performs the accumulation into the vector by calling the chosen accumulation kernel and additional analytical contributions (control variate, optional).

        Parameters
        ----------
        particles : struphy.pic.particles.Particles
            Particles object holding the markers information in format particles.markers.shape == (n_markers, :).

        *args_add
            Additional arguments to be passed to the accumulator kernel, besides the mandatory arguments
            which are prepared automatically (spline bases info, mapping info, data arrays).
            Examples would be parameters for a background kinetic distribution or spline coefficients of a background magnetic field.
            Entries must be pyccel-conform types.

        **args_control
            Keyword arguments for an analytical control variate correction in the accumulation step. Possible keywords are 'control_vec' for a vector correction or 'control_mat' for a matrix correction. Values are a 1d (vector) or 2d (matrix) list with callables or np.ndarrays used for the correction.
        """

        # flags for break
        vec_finished = False

        # reset data
        for dat in self._args_data:
            dat[:] = 0.

        # accumulate into matrix (and vector) with markers
        self.kernel(particles.markers, particles.n_mks,
                    *self._args_fem, *self._domain.args_map,
                    *self._args_data, *args_add)

        # add analytical contribution (control variate) to matrix
        if 'control_mat' in args_control:
            self._operators[0].assemble(weights=args_control['control_mat'])

        # add analytical contribution (control variate) to vector
        if 'control_vec' in args_control and len(self._vectors) > 0:
            WeightedMassOperator.assemble_vec(self._derham.Vh_fem[self._derham.space_to_form[self._space_id]],
                                              self._vectors[0], weight=args_control['control_vec'],
                                              clear=False)

            vec_finished = True

        # finish vector: accumulate ghost regions and update ghost regions
        if not vec_finished:
            for vec in self._vectors:
                vec.exchange_assembly_data()
                vec.update_ghost_regions()
