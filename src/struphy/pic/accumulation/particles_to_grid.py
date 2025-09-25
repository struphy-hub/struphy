"Base classes for particle deposition (accumulation) on the grid."

from psydac.ddm.mpi import mpi as MPI
from psydac.linalg.block import BlockVector
from psydac.linalg.stencil import StencilMatrix, StencilVector

import struphy.pic.accumulation.accum_kernels as accums
import struphy.pic.accumulation.accum_kernels_gc as accums_gc
import struphy.pic.accumulation.filter_kernels as filters
from struphy.feec.mass import WeightedMassOperators
from struphy.feec.psydac_derham import Derham
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments
from struphy.pic.base import Particles
from struphy.profiling.profiling import ProfileManager


class Accumulator:
    r"""
    Struphy accumulation (block) matrices and vectors

    .. math::

        M &= (M^{\mu,\nu})_{\mu,\nu}\,,\qquad && M^{\mu,\nu} \in \mathbb R^{\mathbb N^\alpha_\mu \times \mathbb N^\alpha_\nu}\,,
        \\[2mm]
        V &= (V^\mu)_\mu\,,\qquad &&V^\mu \in \mathbb R^{\mathbb N^\alpha_\mu}\,,

    where :math:`N^\alpha_\mu` denotes the dimension of the :math:`\mu`-th component
    of the :class:`~struphy.feec.psydac_derham.Derham` space
    :math:`V_h^\alpha` (:math:`\mu,\nu = 1,2,3` for vector-valued spaces),
    with entries obtained by summing over all particles :math:`p`,

    .. math::

        M^{\mu,\nu}_{ijk,mno} &= \sum_{p=0}^{N-1} \Lambda^\mu_{ijk}(\boldsymbol \eta_p) \, A^{\mu,\nu}_p \, \Lambda^\nu_{mno}(\boldsymbol \eta_p) \,,
        \\[2mm]
        V^\mu_{ijk} &= \sum_{p=0}^{N-1} \Lambda^\mu_{ijk}(\boldsymbol \eta_p) \, B^\mu_p \,.

    Here, :math:`\Lambda^\mu_{ijk}(\boldsymbol \eta_p)` denotes the :math:`ijk`-th basis function
    of the :math:`\mu`-th component of a Derham space evaluated at the particle position :math:`\boldsymbol \eta_p`,
    and :math:`A^{\mu,\nu}_p` and :math:`B^\mu_p` are particle-dependent "filling functions",
    to be defined in the module :mod:`~struphy.pic.accumulation.accum_kernels`.

    Parameters
    ----------
    particles : Particles
        Particles object holding the markers to accumulate.

    space_id : str
        Space identifier for the matrix/vector (H1, Hcurl, Hdiv, L2 or H1vec) to be accumulated into.

    kernel : pyccelized function
        The accumulation kernel.

    derham : Derham
        Discrete FE spaces object.

    args_domain : DomainArguments
        Mapping infos.

    add_vector : bool
        True if, additionally to a matrix, a vector in the same space is to be accumulated. Default=False.

    symmetry : str
        In case of space_id=Hcurl/Hdiv, the symmetry property of the block matrix: diag, asym, symm, pressure or None (=full matrix, default)

    filter_params : dict
        Params for the accumulation filter: use_filter(string, either `three_point or `fourier), repeat(int), alpha(float) and modes(list with int).

    Note
    ----
        Struphy accumulation kernels called by ``Accumulator`` objects must be added to ``struphy/pic/accumulation/accum_kernels.py``
        (6D particles) or ``struphy/pic/accumulation/accum_kernels_gc.py`` (5D particles), see :ref:`accum_kernels`
        and :ref:`accum_kernels_gc` for details.
    """

    def __init__(
        self,
        particles: Particles,
        space_id: str,
        kernel: Pyccelkernel,
        mass_ops: WeightedMassOperators,
        args_domain: DomainArguments,
        *,
        add_vector: bool = False,
        symmetry: str = None,
        filter_params: dict = {
            "use_filter": None,
            "modes": None,
            "repeat": None,
            "alpha": None,
        },
    ):
        self._particles = particles
        self._space_id = space_id
        assert isinstance(kernel, Pyccelkernel), f"{kernel} is not of type Pyccelkernel"
        self._kernel = kernel
        self._derham = mass_ops.derham
        self._args_domain = args_domain

        self._symmetry = symmetry

        self._filter_params = filter_params

        self._form = self.derham.space_to_form[space_id]

        # initialize matrices (instances of WeightedMassOperator)
        self._operators = []

        # special treatment in model LinearMHDVlasovPC (symmetry=pressure, six symmetric BlockMatrices are needed)
        if symmetry == "pressure":
            for _ in range(6):
                operator = mass_ops.create_weighted_mass(
                    space_id,
                    space_id,
                    weights="symm",
                )
                self._operators.append(operator)

        # "normal" treatment (just one matrix)
        else:
            operator = mass_ops.create_weighted_mass(
                space_id,
                space_id,
                weights=symmetry,
            )
            self._operators.append(operator)

        # collect all _data attributes needed in accumulation kernel
        self._args_data = ()

        for op in self._operators:
            if isinstance(op.matrix, StencilMatrix):
                self._args_data += (op.matrix._data,)
            else:
                for a, row in enumerate(op.matrix.blocks):
                    for b, bl in enumerate(row):
                        if symmetry in ["pressure", "symm", "asym", "diag"]:
                            if b >= a and bl is not None:
                                self._args_data += (bl._data,)
                        else:
                            if bl is not None:
                                self._args_data += (bl._data,)

        # initialize vectors
        self._vectors = []
        self._vectors_temp = []
        self._vectors_out = []

        if add_vector:
            # special treatment in model LinearMHDVlasovPC (symmetry=pressure, three BlockVectors are needed)
            if symmetry == "pressure":
                for _ in range(3):
                    self._vectors += [BlockVector(self.derham.Vh[self.form])]
                    self._vectors_temp += [
                        BlockVector(self.derham.Vh[self.form]),
                    ]
                    self._vectors_out += [
                        BlockVector(self.derham.Vh[self.form]),
                    ]

            # normal treatment (just one vector)
            else:
                for op in self._operators:
                    if isinstance(op.matrix, StencilMatrix):
                        self._vectors += [StencilVector(op.matrix.domain)]
                        self._vectors_temp += [StencilVector(op.matrix.domain)]
                        self._vectors_out += [StencilVector(op.matrix.domain)]
                    else:
                        self._vectors += [BlockVector(op.matrix.domain)]
                        self._vectors_temp += [BlockVector(op.matrix.domain)]
                        self._vectors_out += [BlockVector(op.matrix.domain)]

            for vec in self._vectors:
                if isinstance(vec, StencilVector):
                    self._args_data += (vec._data,)
                else:
                    for bl in vec.blocks:
                        self._args_data += (bl._data,)

    def __call__(self, *optional_args, **args_control):
        """
        Performs the accumulation into the matrix/vector by calling the chosen accumulation kernel and additional analytical contributions (control variate, optional).

        Parameters
        ----------
        particles : Particles
            Particles object holding the markers information in format particles.markers.shape == (n_markers, :).

        optional_args : any
            Additional arguments to be passed to the accumulator kernel, besides the mandatory arguments
            which are prepared automatically (spline bases info, mapping info, data arrays).
            Examples would be parameters for a background kinetic distribution or spline coefficients of a background magnetic field.
            Entries must be pyccel-conform types.

        args_control : any
            Keyword arguments for an analytical control variate correction in the accumulation step. Possible keywords are 'control_vec' for a vector correction or 'control_mat' for a matrix correction. Values are a 1d (vector) or 2d (matrix) list with callables or np.ndarrays used for the correction.
        """

        # flags for break
        vec_finished = False
        mat_finished = False

        # reset data
        for dat in self._args_data:
            dat[:] = 0.0

        # accumulate into matrix (and vector) with markers
        with ProfileManager.profile_region("kernel: " + self.kernel.__name__):
            self.kernel(
                self.particles.args_markers,
                self.derham.args_derham,
                self.args_domain,
                *self._args_data,
                *optional_args,
            )

        # apply filter
        if self.filter_params["use_filter"] is not None:
            for vec in self._vectors:
                vec.exchange_assembly_data()
                vec.update_ghost_regions()

                if self.filter_params["use_filter"] == "fourier_in_tor":
                    self.apply_toroidal_fourier_filter(vec, self.filter_params["modes"])

                elif self.filter_params["use_filter"] == "three_point":
                    for _ in range(self.filter_params["repeat"]):
                        for i in range(3):
                            filters.apply_three_point_filter(
                                vec[i]._data,
                                np.array(self.derham.Nel),
                                np.array(self.derham.spl_kind),
                                np.array(self.derham.p),
                                np.array(self.derham.Vh[self.form][i].starts),
                                np.array(self.derham.Vh[self.form][i].ends),
                                alpha=self.filter_params["alpha"],
                            )

                        vec.update_ghost_regions()

                elif self.filter_params["use_filter"] == "hybrid":
                    self.apply_toroidal_fourier_filter(vec, self.filter_params["modes"])

                    for _ in range(self.filter_params["repeat"]):
                        for i in range(2):
                            filters.apply_three_point_filter(
                                vec[i]._data,
                                np.array(self.derham.Nel),
                                np.array(self.derham.spl_kind),
                                np.array(self.derham.p),
                                np.array(self.derham.Vh[self.form][i].starts),
                                np.array(self.derham.Vh[self.form][i].ends),
                                alpha=self.filter_params["alpha"],
                            )

                        vec.update_ghost_regions()

                else:
                    raise NotImplemented(
                        "The type of filter must be fourier or three_point.",
                    )

            vec_finished = True

        if self.particles.clone_config is None:
            num_clones = 1
        else:
            num_clones = self.particles.clone_config.num_clones

        if num_clones > 1:
            for data_array in self._args_data:
                self.particles.clone_config.inter_comm.Allreduce(
                    MPI.IN_PLACE,
                    data_array,
                    op=MPI.SUM,
                )

        # add analytical contribution (control variate) to vector
        if "control_vec" in args_control and len(self._vectors) > 0:
            self._get_L2dofs(
                args_control["control_vec"],
                dofs=self._vectors[0],
                clear=False,
            )
            vec_finished = True

        # add analytical contribution (control variate) to matrix and finish
        if "control_mat" in args_control:
            self._operators[0].assemble(
                weights=args_control["control_mat"],
                clear=False,
                verbose=False,
            )
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

            if self.symmetry == "symm":
                self._operators[0].matrix[0, 1].transpose(
                    out=self._operators[0].matrix[1, 0],
                )
                self._operators[0].matrix[0, 2].transpose(
                    out=self._operators[0].matrix[2, 0],
                )
                self._operators[0].matrix[1, 2].transpose(
                    out=self._operators[0].matrix[2, 1],
                )

            elif self.symmetry == "asym":
                self._operators[0].matrix[0, 1].transpose(
                    out=self._operators[0].matrix[1, 0],
                )
                self._operators[0].matrix[1, 0] *= -1
                self._operators[0].matrix[0, 2].transpose(
                    out=self._operators[0].matrix[2, 0],
                )
                self._operators[0].matrix[2, 0] *= -1
                self._operators[0].matrix[1, 2].transpose(
                    out=self._operators[0].matrix[2, 1],
                )
                self._operators[0].matrix[2, 1] *= -1

            elif self.symmetry == "pressure":
                for i in range(6):
                    self._operators[i].matrix[0, 1].transpose(
                        out=self._operators[i].matrix[1, 0],
                    )
                    self._operators[i].matrix[0, 2].transpose(
                        out=self._operators[i].matrix[2, 0],
                    )
                    self._operators[i].matrix[1, 2].transpose(
                        out=self._operators[i].matrix[2, 1],
                    )

    @property
    def particles(self):
        """Particle object."""
        return self._particles

    @property
    def kernel(self) -> Pyccelkernel:
        """The accumulation kernel."""
        return self._kernel

    @property
    def derham(self):
        """Discrete Derham complex on the logical unit cube."""
        return self._derham

    @property
    def args_domain(self):
        """Mapping info for evaluating metric coefficients."""
        return self._args_domain

    @property
    def space_id(self):
        """Space identifier for the matrix/vector (H1, Hcurl, Hdiv, L2 or H1vec) to be accumulated into."""
        return self._space_id

    @property
    def form(self):
        """p-form ("0", "1", "2", "3" or "v") to be accumulated into."""
        return self._form

    @property
    def symmetry(self):
        """Symmetry of the accumulation matrix (diagonal, symmetric, asymmetric, etc.)."""
        return self._symmetry

    @property
    def operators(self):
        """List of WeightedMassOperators of the accumulator. Matrices can be accessed e.g. with operators[0].matrix."""
        return self._operators

    @property
    def vectors(self):
        """List of Stencil-/Block-/PolarVectors of the accumulator."""
        out = []
        for vec, vec_temp, vec_out in zip(self._vectors, self._vectors_temp, self._vectors_out):
            self._derham.extraction_ops[self.form].dot(vec, out=vec_temp)
            self._derham.boundary_ops[self.form].dot(vec_temp, out=vec_out)
            out += [vec_out]

        return out

    @property
    def filter_params(self):
        """Dict of three components for the accumulation filter parameters: use_filter(string), repeat(int) and alpha(float)."""
        return self._filter_params

    @property
    def filter_params(self):
        """Dict of three components for the accumulation filter parameters: use_filter(string), repeat(int) and alpha(float)."""
        return self._filter_params

    def init_control_variate(self, mass_ops):
        """Set up the use of noise reduction by control variate."""

        from struphy.feec.projectors import L2Projector

        # L2 projector for dofs
        self._get_L2dofs = L2Projector(self.space_id, mass_ops).get_dofs

    def apply_toroidal_fourier_filter(self, vec, modes):
        """
        Applying fourier filter to the spline coefficients of the accumulated vector (toroidal direction).

        Parameters
        ----------
        vec : BlockVector

        modes : list
            Mode numbers which are not filtered out.
        """

        from scipy.fft import irfft, rfft

        tor_Nel = self.derham.Nel[2]

        # Nel along the toroidal direction must be equal or bigger than 2*maximum mode
        assert tor_Nel >= 2 * max(modes)

        pn = self.derham.p
        ir = np.empty(3, dtype=int)

        if (tor_Nel % 2) == 0:
            vec_temp = np.zeros(int(tor_Nel / 2) + 1, dtype=complex)
        else:
            vec_temp = np.zeros(int((tor_Nel - 1) / 2) + 1, dtype=complex)

        # no domain decomposition along the toroidal direction
        assert self.derham.domain_decomposition.nprocs[2] == 1

        for axis in range(3):
            starts = self.derham.Vh[ſelf.form][axis].starts
            ends = self.derham.Vh[self.form][axis].ends

            # index range
            for i in range(3):
                ir[i] = ends[i] + 1 - starts[i]

            # filtering
            for i in range(ir[0]):
                for j in range(ir[1]):
                    vec_temp[:] = 0
                    vec_temp[modes] = rfft(
                        vec[axis]._data[pn[0] + i, pn[1] + j, pn[2] : pn[2] + ir[2]],
                    )[modes]
                    vec[axis]._data[pn[0] + i, pn[1] + j, pn[2] : pn[2] + ir[2]] = irfft(vec_temp, n=tor_Nel)

            vec.update_ghost_regions()

    def show_accumulated_spline_field(self, mass_ops: WeightedMassOperators, eta_direction=0, component=0):
        r"""1D plot of the spline field corresponding to the accumulated vector.
        The latter can be viewed as the rhs of an L2-projection:

        .. math::

            \mathbb M \mathbf a = \sum_p \boldsymbol \Lambda(\boldsymbol \eta_p) * B_p\,.

        The FE coefficients :math:`\mathbf a` determine a FE :class:`~struphy.feec.psydac_derham.SplineFunction`.
        """
        from matplotlib import pyplot as plt

        from struphy.feec.projectors import L2Projector

        # L2 projection
        proj = L2Projector(self.space_id, mass_ops)
        a = proj.solve(self.vectors[0])

        # create field and assign coeffs
        field = self.derham.create_spline_function("accum_field", self.space_id)
        field.vector = a

        # plot field
        eta = np.linspace(0, 1, 100)
        if eta_direction == 0:
            args = (eta, 0.5, 0.5)
        elif eta_direction == 1:
            args = (0.5, eta, 0.5)
        else:
            args = (0.5, 0.5, eta)

        vals = mass_ops.domain.push(field, *args, kind="1", squeeze_out=True)

        plt.plot(eta, vals[component])
        plt.title(
            f'Spline field accumulated with the kernel "{self.kernel}"',
        )
        plt.xlabel(rf"$\eta_{eta_direction + 1}$")
        plt.ylabel("field amplitude")
        plt.show()


class AccumulatorVector:
    r"""
    Same as :class:`~struphy.pic.accumulation.particles_to_grid.Accumulator` but only for vectors :math:`V`.

    Parameters
    ----------
    particles : Particles
        Particles object holding the markers to accumulate.

    space_id : str
        Space identifier for the matrix/vector (H1, Hcurl, Hdiv, L2 or H1vec) to be accumulated into.

    kernel : pyccelized function
        The accumulation kernel.

    derham : Derham
        Discrete FE spaces object.

    args_domain : DomainArguments
        Mapping infos.

    """

    def __init__(
        self,
        particles: Particles,
        space_id: str,
        kernel: Pyccelkernel,
        mass_ops: WeightedMassOperators,
        args_domain: DomainArguments,
    ):
        self._particles = particles
        self._space_id = space_id
        assert isinstance(kernel, Pyccelkernel), f"{kernel} is not of type Pyccelkernel"
        self._kernel = kernel
        self._derham = mass_ops.derham
        self._args_domain = args_domain

        self._form = self.derham.space_to_form[space_id]

        # initialize vectors
        self._vectors = []
        self._vectors_temp = []
        self._vectors_out = []

        # collect all _data attributes needed in accumulation kernel
        self._args_data = ()

        if space_id in ("H1", "L2"):
            self._vectors += [
                StencilVector(self.derham.Vh_fem[self.form].coeff_space),
            ]
            self._vectors_temp += [
                StencilVector(self.derham.Vh_fem[self.form].coeff_space),
            ]
            self._vectors_out += [
                StencilVector(self.derham.Vh_fem[self.form].coeff_space),
            ]

        elif space_id in ("Hcurl", "Hdiv", "H1vec"):
            self._vectors += [
                BlockVector(
                    self.derham.Vh_fem[self.form].coeff_space,
                ),
            ]
            self._vectors_temp += [
                BlockVector(
                    self.derham.Vh_fem[self.form].coeff_space,
                ),
            ]
            self._vectors_out += [
                BlockVector(
                    self.derham.Vh_fem[self.form].coeff_space,
                ),
            ]

        for vec in self._vectors:
            if isinstance(vec, StencilVector):
                self._args_data += (vec._data,)
            else:
                for bl in vec.blocks:
                    self._args_data += (bl._data,)

    def __call__(self, *optional_args, **args_control):
        """
        Performs the accumulation into the vector by calling the chosen accumulation kernel
        and additional analytical contributions (control variate, optional).

        Parameters
        ----------
        optional_args : any
            Additional arguments to be passed to the accumulator kernel, besides the mandatory arguments
            which are prepared automatically (spline bases info, mapping info, data arrays).
            Examples would be parameters for a background kinetic distribution or spline coefficients of a background magnetic field.
            Entries must be pyccel-conform types.

        args_control : any
            Keyword arguments for an analytical control variate correction in the accumulation step.
            Possible keywords are 'control_vec' for a vector correction or 'control_mat' for a matrix correction.
            Values are a 1d (vector) or 2d (matrix) list with callables or np.ndarrays used for the correction.
        """

        # flags for break
        vec_finished = False

        # reset data
        for dat in self._args_data:
            dat[:] = 0.0

        # accumulate into matrix (and vector) with markers
        with ProfileManager.profile_region("kernel: " + self.kernel.__name__):
            self.kernel(
                self.particles.args_markers,
                self.derham._args_derham,
                self.args_domain,
                *self._args_data,
                *optional_args,
            )

        if self.particles.clone_config is None:
            num_clones = 1
        else:
            num_clones = self.particles.clone_config.num_clones

        if num_clones > 1:
            for data_array in self._args_data:
                self.particles.clone_config.inter_comm.Allreduce(
                    MPI.IN_PLACE,
                    data_array,
                    op=MPI.SUM,
                )

        # add analytical contribution (control variate) to vector
        if "control_vec" in args_control and len(self._vectors) > 0:
            self._get_L2dofs(
                args_control["control_vec"],
                dofs=self._vectors[0],
                clear=False,
            )
            vec_finished = True

        # finish vector: accumulate ghost regions and update ghost regions
        if not vec_finished:
            for vec in self._vectors:
                vec.exchange_assembly_data()
                vec.update_ghost_regions()

    @property
    def particles(self):
        """Particle object."""
        return self._particles

    @property
    def kernel(self) -> Pyccelkernel:
        """The accumulation kernel."""
        return self._kernel

    @property
    def derham(self):
        """Discrete Derham complex on the logical unit cube."""
        return self._derham

    @property
    def args_domain(self):
        """Mapping arguments."""
        return self._args_domain

    @property
    def space_id(self):
        """Space identifier for the matrix/vector (H1, Hcurl, Hdiv, L2 or H1vec) to be accumulated into."""
        return self._space_id

    @property
    def form(self):
        """p-form ("0", "1", "2", "3" or "v") to be accumulated into."""
        return self._form

    @property
    def vectors(self):
        """List of Stencil-/Block-/PolarVectors of the accumulator."""
        out = []
        for vec, vec_temp, vec_out in zip(self._vectors, self._vectors_temp, self._vectors_out):
            self._derham.extraction_ops[self.form].dot(vec, out=vec_temp)
            self._derham.boundary_ops[self.form].dot(vec_temp, out=vec_out)
            out += [vec_out]

        return out

    def init_control_variate(self, mass_ops):
        """Set up the use of noise reduction by control variate."""

        from struphy.feec.projectors import L2Projector

        # L2 projector for dofs
        self._get_L2dofs = L2Projector(self.space_id, mass_ops).get_dofs

    def show_accumulated_spline_field(self, mass_ops, eta_direction=0):
        r"""1D plot of the spline field corresponding to the accumulated vector.
        The latter can be viewed as the rhs of an L2-projection:

        .. math::

            \mathbb M \mathbf a = \sum_p \boldsymbol \Lambda(\boldsymbol \eta_p) * B_p\,.

        The FE coefficients :math:`\mathbf a` determine a FE :class:`~struphy.feec.psydac_derham.SplineFunction`.
        """
        from matplotlib import pyplot as plt

        from struphy.feec.projectors import L2Projector

        # L2 projection
        proj = L2Projector(self.space_id, mass_ops)
        a = proj.solve(self.vectors[0])

        # create field and assign coeffs
        field = self.derham.create_spline_function("accum_field", self.space_id)
        field.vector = a

        # plot field
        eta = np.linspace(0, 1, 100)
        if eta_direction == 0:
            args = (eta, 0.5, 0.5)
        elif eta_direction == 1:
            args = (0.5, eta, 0.5)
        else:
            args = (0.5, 0.5, eta)

        plt.plot(eta, field(*args, squeeze_out=True))
        plt.title(
            f'Spline field accumulated with the kernel "{self.kernel}"',
        )
        plt.xlabel(rf"$\eta_{eta_direction + 1}$")
        plt.ylabel("field amplitude")
        plt.show()
