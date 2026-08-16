"Base classes for particle deposition (accumulation) on the grid."

from dataclasses import dataclass

import cunumpy as xp
from cunumpy import PyccelKernel
from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.block import BlockVector
from feectools.linalg.stencil import StencilMatrix, StencilVector
from scope_profiler import ProfileManager

import struphy.pic.accumulation.accum_kernels as accums
import struphy.pic.accumulation.accum_kernels_gc as accums_gc
from struphy.feec.mass import WeightedMassOperators
from struphy.feec.psydac_derham import Derham
from struphy.io.options import LiteralOptions
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments
from struphy.models.variables import PICVariable, SPHVariable
from struphy.pic.accumulation.accum_kernels_cuda import (
    cc_lin_mhd_6d_1_gpu,
    cc_lin_mhd_6d_2_gpu,
    charge_density_0form_gpu,
    linear_vlasov_ampere_gpu,
    pc_lin_mhd_6d_full_gpu,
    pc_lin_mhd_6d_gpu,
    vlasov_maxwell_gpu,
)
from struphy.pic.accumulation.accum_kernels_gc_cuda import (
    cc_lin_mhd_5d_curlb_gpu,
    cc_lin_mhd_5d_D_gpu,
    cc_lin_mhd_5d_gradB_dg_gpu,
    cc_lin_mhd_5d_gradB_gpu,
    gc_mag_density_0form_gpu,
)
from struphy.pic.accumulation.filter import AccumFilter, FilterParameters
from struphy.pic.base import Particles
from struphy.utils.utils import __dataclass_repr_no_defaults__, check_option


class Accumulator:
    r"""
    Approximates integrals of the form

    .. math::

        I_A &= \int_\Omega \int_{\mathbb R^3} \Lambda^\mu_{ijk}(\boldsymbol \eta) \, A^{\mu, \nu}(\boldsymbol \eta, \mathbf v) \, \Lambda^\nu_{mno}(\boldsymbol \eta) \, f^{\textrm{vol}}(\boldsymbol \eta, \mathbf v)\,\mathrm d\mathbf v \textrm d \boldsymbol \eta\,,
        \\[2mm]
        I_B &= \int_\Omega \int_{\mathbb R^3} \Lambda^\mu_{ijk}(\boldsymbol \eta) \, B^\mu(\boldsymbol \eta, \mathbf v) \, f^{\textrm{vol}}(\boldsymbol \eta, \mathbf v)\,\mathrm d\mathbf v \textrm d \boldsymbol \eta\,,

    for given weight functions :math:`A^{\mu,\nu}` and :math:`B^\mu` by Monte-Carlo quadrature through the particle distribution function :math:`f^{\textrm{vol}}`:

    .. math::

        f^{\textrm{vol}}(\boldsymbol \eta, \mathbf v) \approx \sum_{p=0}^{N-1} w_p \, \delta(\boldsymbol \eta - \boldsymbol \eta_p) \, \delta(\mathbf v - \mathbf v_p)\,.

    This results in stencil (block) matrices and vectors

    .. math::

        M &= (M^{\mu,\nu})_{\mu,\nu}\,,\qquad && M^{\mu,\nu} \in \mathbb R^{\mathbb N^\alpha_\mu \times \mathbb N^\alpha_\nu}\,,
        \\[2mm]
        V &= (V^\mu)_\mu\,,\qquad &&V^\mu \in \mathbb R^{\mathbb N^\alpha_\mu}\,,

    where :math:`N^\alpha_\mu` denotes the dimension of the :math:`\mu`-th component
    of the :class:`~struphy.feec.psydac_derham.Derham` space
    :math:`V_h^\alpha` (:math:`\mu,\nu = 1,2,3` for vector-valued spaces),
    with entries obtained by summing over all particles :math:`p`,

    .. math::

        M^{\mu,\nu}_{ijk,mno} &= \sum_{p=0}^{N-1} w_p\, \Lambda^\mu_{ijk}(\boldsymbol \eta_p) \, A^{\mu,\nu}_p \, \Lambda^\nu_{mno}(\boldsymbol \eta_p) \,,
        \\[2mm]
        V^\mu_{ijk} &= \sum_{p=0}^{N-1} w_p\, \Lambda^\mu_{ijk}(\boldsymbol \eta_p) \, B^\mu_p \,.

    Here, :math:`\Lambda^\mu_{ijk}(\boldsymbol \eta_p)` denotes the :math:`ijk`-th basis function
    of the :math:`\mu`-th component of a Derham space.

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
        kernel: PyccelKernel,
        mass_ops: WeightedMassOperators,
        args_domain: DomainArguments,
        *,
        add_vector: bool = False,
        symmetry: str = None,
        filter_params: FilterParameters = None,
    ):
        self._particles = particles
        self._space_id = space_id
        assert isinstance(kernel, PyccelKernel), f"{kernel} is not of type PyccelKernel"
        self._kernel = kernel
        self._derham = mass_ops.derham
        self._args_domain = args_domain

        # profiling region names (precomputed, they are looked up on every call)
        self._region_name = "accum: " + kernel.name
        self._comm_region_name = "accum comm: " + kernel.name

        self._symmetry = symmetry

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
                    self._vectors += [BlockVector(self.derham.coeff_spaces[self.form])]
                    self._vectors_temp += [
                        BlockVector(self.derham.coeff_spaces[self.form]),
                    ]
                    self._vectors_out += [
                        BlockVector(self.derham.coeff_spaces[self.form]),
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

        # initialize filter
        self._accfilter = AccumFilter(filter_params, self._derham, self._space_id)

        # GPU replacement for linear_vlasov_ampere: evaluates DF^-1(eta_p) per
        # marker and atomically scatters into the 6 symmetric V1 -> V1 matrix
        # blocks plus the V1 vector, instead of the CPU's strictly-sequential
        # marker loop. See pusher_kernels_cuda.SUPPORTED_GENERAL_KIND_MAPS
        # for the domains this covers.
        from struphy.pic.pushing.pusher_kernels_cuda import SUPPORTED_GENERAL_KIND_MAPS

        self._gpu_linear_vlasov_ampere = (
            xp.cupy_backend
            and kernel.name == "linear_vlasov_ampere"
            and args_domain.kind_map in SUPPORTED_GENERAL_KIND_MAPS
        )
        if self._gpu_linear_vlasov_ampere:
            import cupy as cp
            import numpy as np

            self._gpu_lva_kind_map = int(args_domain.kind_map)
            self._gpu_lva_params = cp.asarray(np.asarray(args_domain.params, dtype=float), dtype=cp.float64)
            args_derham = self.derham.args_derham
            self._gpu_lva_pn = tuple(int(p) for p in args_derham.pn)
            self._gpu_lva_starts = tuple(int(s) for s in args_derham.starts)
            self._gpu_lva_tn1 = cp.asarray(np.asarray(args_derham.tn1, dtype=float), dtype=cp.float64)
            self._gpu_lva_tn2 = cp.asarray(np.asarray(args_derham.tn2, dtype=float), dtype=cp.float64)
            self._gpu_lva_tn3 = cp.asarray(np.asarray(args_derham.tn3, dtype=float), dtype=cp.float64)

        # GPU replacement for vlasov_maxwell: same 6-block symmetric V1 -> V1
        # matrix-plus-vector fill as linear_vlasov_ampere above, but with a
        # G^-1(eta_p)-based filling (no f0_values/optional_args needed).
        self._gpu_vlasov_maxwell = (
            xp.cupy_backend
            and kernel.name == "vlasov_maxwell"
            and args_domain.kind_map in SUPPORTED_GENERAL_KIND_MAPS
        )
        if self._gpu_vlasov_maxwell:
            import cupy as cp
            import numpy as np

            self._gpu_vm_kind_map = int(args_domain.kind_map)
            self._gpu_vm_params = cp.asarray(np.asarray(args_domain.params, dtype=float), dtype=cp.float64)
            args_derham = self.derham.args_derham
            self._gpu_vm_pn = tuple(int(p) for p in args_derham.pn)
            self._gpu_vm_starts = tuple(int(s) for s in args_derham.starts)
            self._gpu_vm_tn1 = cp.asarray(np.asarray(args_derham.tn1, dtype=float), dtype=cp.float64)
            self._gpu_vm_tn2 = cp.asarray(np.asarray(args_derham.tn2, dtype=float), dtype=cp.float64)
            self._gpu_vm_tn3 = cp.asarray(np.asarray(args_derham.tn3, dtype=float), dtype=cp.float64)

        # GPU replacement for cc_lin_mhd_6d_1: 3-block antisymmetric fill
        # (mat12, mat13, mat23 only, no vector) into whichever of
        # H1vec/Hcurl/Hdiv the propagator's basis_u optional_arg selects.
        # b2_*/basis_u/scale_mat/boundary_cut arrive fresh via optional_args
        # each call (only the spline/domain info below is cached).
        self._gpu_cc_lin_mhd_6d_1 = (
            xp.cupy_backend
            and kernel.name == "cc_lin_mhd_6d_1"
            and args_domain.kind_map in SUPPORTED_GENERAL_KIND_MAPS
        )
        if self._gpu_cc_lin_mhd_6d_1:
            import cupy as cp
            import numpy as np

            self._gpu_cc1_kind_map = int(args_domain.kind_map)
            self._gpu_cc1_params = cp.asarray(np.asarray(args_domain.params, dtype=float), dtype=cp.float64)
            args_derham = self.derham.args_derham
            self._gpu_cc1_pn = tuple(int(p) for p in args_derham.pn)
            self._gpu_cc1_starts = tuple(int(s) for s in args_derham.starts)
            self._gpu_cc1_tn1 = cp.asarray(np.asarray(args_derham.tn1, dtype=float), dtype=cp.float64)
            self._gpu_cc1_tn2 = cp.asarray(np.asarray(args_derham.tn2, dtype=float), dtype=cp.float64)
            self._gpu_cc1_tn3 = cp.asarray(np.asarray(args_derham.tn3, dtype=float), dtype=cp.float64)

        # GPU replacement for cc_lin_mhd_6d_2: same runtime basis_u
        # dispatch as cc_lin_mhd_6d_1, but a full symmetric 6-block
        # matrix-plus-vector fill (like linear_vlasov_ampere/vlasov_maxwell)
        # instead of the 3 antisymmetric off-diagonal blocks only.
        self._gpu_cc_lin_mhd_6d_2 = (
            xp.cupy_backend
            and kernel.name == "cc_lin_mhd_6d_2"
            and args_domain.kind_map in SUPPORTED_GENERAL_KIND_MAPS
        )
        if self._gpu_cc_lin_mhd_6d_2:
            import cupy as cp
            import numpy as np

            self._gpu_cc2_kind_map = int(args_domain.kind_map)
            self._gpu_cc2_params = cp.asarray(np.asarray(args_domain.params, dtype=float), dtype=cp.float64)
            args_derham = self.derham.args_derham
            self._gpu_cc2_pn = tuple(int(p) for p in args_derham.pn)
            self._gpu_cc2_starts = tuple(int(s) for s in args_derham.starts)
            self._gpu_cc2_tn1 = cp.asarray(np.asarray(args_derham.tn1, dtype=float), dtype=cp.float64)
            self._gpu_cc2_tn2 = cp.asarray(np.asarray(args_derham.tn2, dtype=float), dtype=cp.float64)
            self._gpu_cc2_tn3 = cp.asarray(np.asarray(args_derham.tn3, dtype=float), dtype=cp.float64)

        # GPU replacement for cc_lin_mhd_5d_D: 3-block antisymmetric fill like
        # cc_lin_mhd_6d_1, with the guiding-centre density prefactor
        # (1 - b_para/b*_para) / epsilon.
        self._gpu_cc_lin_mhd_5d_D = (
            xp.cupy_backend
            and kernel.name == "cc_lin_mhd_5d_D"
            and args_domain.kind_map in SUPPORTED_GENERAL_KIND_MAPS
        )
        if self._gpu_cc_lin_mhd_5d_D:
            import cupy as cp
            import numpy as np

            self._gpu_cc5d_kind_map = int(args_domain.kind_map)
            self._gpu_cc5d_params = cp.asarray(np.asarray(args_domain.params, dtype=float), dtype=cp.float64)
            args_derham = self.derham.args_derham
            self._gpu_cc5d_pn = tuple(int(p) for p in args_derham.pn)
            self._gpu_cc5d_starts = tuple(int(s) for s in args_derham.starts)
            self._gpu_cc5d_tn1 = cp.asarray(np.asarray(args_derham.tn1, dtype=float), dtype=cp.float64)
            self._gpu_cc5d_tn2 = cp.asarray(np.asarray(args_derham.tn2, dtype=float), dtype=cp.float64)
            self._gpu_cc5d_tn3 = cp.asarray(np.asarray(args_derham.tn3, dtype=float), dtype=cp.float64)

        # GPU replacements for the two remaining 5D current-coupling
        # accumulators. cc_lin_mhd_5d_curlb is a full symmetric 6-block
        # matrix-plus-vector curvature fill; cc_lin_mhd_5d_gradB is
        # vector-only (its 6 matrix args are unused by the CPU body, so the
        # matrices stay zero on both backends). They need the same cached
        # spline/domain data, so one block covers both.
        self._gpu_cc_lin_mhd_5d_curlb = (
            xp.cupy_backend
            and kernel.name == "cc_lin_mhd_5d_curlb"
            and args_domain.kind_map in SUPPORTED_GENERAL_KIND_MAPS
        )
        self._gpu_cc_lin_mhd_5d_gradB = (
            xp.cupy_backend
            and kernel.name == "cc_lin_mhd_5d_gradB"
            and args_domain.kind_map in SUPPORTED_GENERAL_KIND_MAPS
        )
        if self._gpu_cc_lin_mhd_5d_curlb or self._gpu_cc_lin_mhd_5d_gradB:
            import cupy as cp
            import numpy as np

            self._gpu_cg_kind_map = int(args_domain.kind_map)
            self._gpu_cg_params = cp.asarray(np.asarray(args_domain.params, dtype=float), dtype=cp.float64)
            args_derham = self.derham.args_derham
            self._gpu_cg_pn = tuple(int(p) for p in args_derham.pn)
            self._gpu_cg_starts = tuple(int(s) for s in args_derham.starts)
            self._gpu_cg_tn1 = cp.asarray(np.asarray(args_derham.tn1, dtype=float), dtype=cp.float64)
            self._gpu_cg_tn2 = cp.asarray(np.asarray(args_derham.tn2, dtype=float), dtype=cp.float64)
            self._gpu_cg_tn3 = cp.asarray(np.asarray(args_derham.tn3, dtype=float), dtype=cp.float64)
            self._gpu_cg_first_init_idx = int(self.particles.args_markers.first_init_idx)
            self._gpu_cg_mu_idx = int(self.particles.mu_idx)

        # GPU replacement for pc_lin_mhd_6d_full / pc_lin_mhd_6d: the
        # symmetry="pressure" case -- 45-array (36 matrix + 9 vector)
        # velocity-moment "pressure tensor" fill. See accum_kernels_cuda.py
        # for why both variants share one call convention (pc_lin_mhd_6d
        # only ever writes 24 of the 45 arrays, matching the CPU reference).
        self._gpu_pc_lin_mhd_6d_full = (
            xp.cupy_backend
            and kernel.name == "pc_lin_mhd_6d_full"
            and args_domain.kind_map in SUPPORTED_GENERAL_KIND_MAPS
        )
        self._gpu_pc_lin_mhd_6d = (
            xp.cupy_backend
            and kernel.name == "pc_lin_mhd_6d"
            and args_domain.kind_map in SUPPORTED_GENERAL_KIND_MAPS
        )
        if self._gpu_pc_lin_mhd_6d_full or self._gpu_pc_lin_mhd_6d:
            import cupy as cp
            import numpy as np

            self._gpu_pc_kind_map = int(args_domain.kind_map)
            self._gpu_pc_params = cp.asarray(np.asarray(args_domain.params, dtype=float), dtype=cp.float64)
            args_derham = self.derham.args_derham
            self._gpu_pc_pn = tuple(int(p) for p in args_derham.pn)
            self._gpu_pc_starts = tuple(int(s) for s in args_derham.starts)
            self._gpu_pc_tn1 = cp.asarray(np.asarray(args_derham.tn1, dtype=float), dtype=cp.float64)
            self._gpu_pc_tn2 = cp.asarray(np.asarray(args_derham.tn2, dtype=float), dtype=cp.float64)
            self._gpu_pc_tn3 = cp.asarray(np.asarray(args_derham.tn3, dtype=float), dtype=cp.float64)

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
            Keyword arguments for an analytical control variate correction in the accumulation step. Possible keywords are 'control_vec' for a vector correction or 'control_mat' for a matrix correction. Values are a 1d (vector) or 2d (matrix) list with callables or xp.ndarrays used for the correction.
        """
        with ProfileManager.profile_region(self._region_name):
            self._accumulate(*optional_args, **args_control)

    def _accumulate(self, *optional_args, **args_control):
        """Body of :meth:`__call__`, see there."""

        # flags for break
        vec_finished = False
        mat_finished = False

        # reset data
        for dat in self._args_data:
            dat[:] = 0.0

        # accumulate into matrix (and vector) with markers
        if self._gpu_linear_vlasov_ampere and len(optional_args) == 1:
            with ProfileManager.profile_region("kernel: " + self.kernel.name + " [cuda]"):
                (f0_values,) = optional_args
                linear_vlasov_ampere_gpu(
                    self.particles.markers,
                    self._gpu_lva_kind_map,
                    self._gpu_lva_params,
                    f0_values,
                    self._gpu_lva_pn,
                    self._gpu_lva_tn1,
                    self._gpu_lva_tn2,
                    self._gpu_lva_tn3,
                    self._gpu_lva_starts,
                    *self._args_data,
                )
        elif self._gpu_vlasov_maxwell and not optional_args:
            with ProfileManager.profile_region("kernel: " + self.kernel.name + " [cuda]"):
                vlasov_maxwell_gpu(
                    self.particles.markers,
                    self._gpu_vm_kind_map,
                    self._gpu_vm_params,
                    self._gpu_vm_pn,
                    self._gpu_vm_tn1,
                    self._gpu_vm_tn2,
                    self._gpu_vm_tn3,
                    self._gpu_vm_starts,
                    *self._args_data,
                )
        elif self._gpu_cc_lin_mhd_6d_1 and len(optional_args) == 6:
            with ProfileManager.profile_region("kernel: " + self.kernel.name + " [cuda]"):
                b2_1, b2_2, b2_3, basis_u, scale_mat, boundary_cut = optional_args
                cc_lin_mhd_6d_1_gpu(
                    self.particles.markers,
                    self._gpu_cc1_kind_map,
                    self._gpu_cc1_params,
                    self._gpu_cc1_pn,
                    self._gpu_cc1_tn1,
                    self._gpu_cc1_tn2,
                    self._gpu_cc1_tn3,
                    self._gpu_cc1_starts,
                    b2_1,
                    b2_2,
                    b2_3,
                    basis_u,
                    scale_mat,
                    boundary_cut,
                    *self._args_data,
                )
        elif self._gpu_cc_lin_mhd_6d_2 and len(optional_args) == 7:
            with ProfileManager.profile_region("kernel: " + self.kernel.name + " [cuda]"):
                b2_1, b2_2, b2_3, basis_u, scale_mat, scale_vec, boundary_cut = optional_args
                cc_lin_mhd_6d_2_gpu(
                    self.particles.markers,
                    self._gpu_cc2_kind_map,
                    self._gpu_cc2_params,
                    self._gpu_cc2_pn,
                    self._gpu_cc2_tn1,
                    self._gpu_cc2_tn2,
                    self._gpu_cc2_tn3,
                    self._gpu_cc2_starts,
                    b2_1,
                    b2_2,
                    b2_3,
                    basis_u,
                    scale_mat,
                    scale_vec,
                    boundary_cut,
                    *self._args_data,
                )
        elif self._gpu_cc_lin_mhd_5d_D and len(optional_args) == 12:
            with ProfileManager.profile_region("kernel: " + self.kernel.name + " [cuda]"):
                (
                    epsilon, ep_scale,
                    b2_1, b2_2, b2_3,
                    nb1_1, nb1_2, nb1_3,
                    cnb_1, cnb_2, cnb_3,
                    basis_u,
                ) = optional_args
                cc_lin_mhd_5d_D_gpu(
                    self.particles.markers,
                    self._gpu_cc5d_kind_map,
                    self._gpu_cc5d_params,
                    epsilon,
                    ep_scale,
                    self._gpu_cc5d_pn,
                    self._gpu_cc5d_tn1,
                    self._gpu_cc5d_tn2,
                    self._gpu_cc5d_tn3,
                    self._gpu_cc5d_starts,
                    (b2_1, b2_2, b2_3),
                    (nb1_1, nb1_2, nb1_3),
                    (cnb_1, cnb_2, cnb_3),
                    basis_u,
                    *self._args_data,
                )
        elif self._gpu_cc_lin_mhd_5d_curlb and len(optional_args) == 12:
            with ProfileManager.profile_region("kernel: " + self.kernel.name + " [cuda]"):
                (
                    epsilon, ep_scale,
                    b2_1, b2_2, b2_3,
                    nb1_1, nb1_2, nb1_3,
                    cnb_1, cnb_2, cnb_3,
                    basis_u,
                ) = optional_args
                cc_lin_mhd_5d_curlb_gpu(
                    self.particles.markers,
                    self._gpu_cg_kind_map,
                    self._gpu_cg_params,
                    epsilon,
                    ep_scale,
                    self._gpu_cg_pn,
                    self._gpu_cg_tn1,
                    self._gpu_cg_tn2,
                    self._gpu_cg_tn3,
                    self._gpu_cg_starts,
                    (b2_1, b2_2, b2_3),
                    (nb1_1, nb1_2, nb1_3),
                    (cnb_1, cnb_2, cnb_3),
                    basis_u,
                    *self._args_data,
                )
        elif self._gpu_cc_lin_mhd_5d_gradB and len(optional_args) == 17:
            with ProfileManager.profile_region("kernel: " + self.kernel.name + " [cuda]"):
                (
                    epsilon, ep_scale,
                    b2_1, b2_2, b2_3,
                    nb1_1, nb1_2, nb1_3,
                    cnb_1, cnb_2, cnb_3,
                    gpb_1, gpb_2, gpb_3,
                    gpq_1, gpq_2, gpq_3,
                    basis_u,
                ) = optional_args
                # the kernel's own 6 matrix args + vector are already in
                # self._args_data; only the vector blocks are ever written.
                vec_data = self._args_data[6:]
                cc_lin_mhd_5d_gradB_gpu(
                    self.particles.markers,
                    self._gpu_cg_first_init_idx,
                    self._gpu_cg_mu_idx,
                    self._gpu_cg_kind_map,
                    self._gpu_cg_params,
                    epsilon,
                    ep_scale,
                    self._gpu_cg_pn,
                    self._gpu_cg_tn1,
                    self._gpu_cg_tn2,
                    self._gpu_cg_tn3,
                    self._gpu_cg_starts,
                    (b2_1, b2_2, b2_3),
                    (nb1_1, nb1_2, nb1_3),
                    (cnb_1, cnb_2, cnb_3),
                    (gpb_1, gpb_2, gpb_3),
                    (gpq_1, gpq_2, gpq_3),
                    basis_u,
                    *vec_data,
                )
        elif (self._gpu_pc_lin_mhd_6d_full or self._gpu_pc_lin_mhd_6d) and len(optional_args) == 1:
            with ProfileManager.profile_region("kernel: " + self.kernel.name + " [cuda]"):
                (ep_scale,) = optional_args
                pc_fn = pc_lin_mhd_6d_full_gpu if self._gpu_pc_lin_mhd_6d_full else pc_lin_mhd_6d_gpu
                pc_fn(
                    self.particles.markers,
                    self._gpu_pc_kind_map,
                    self._gpu_pc_params,
                    self._gpu_pc_pn,
                    self._gpu_pc_tn1,
                    self._gpu_pc_tn2,
                    self._gpu_pc_tn3,
                    self._gpu_pc_starts,
                    ep_scale,
                    *self._args_data,
                )
        else:
            # no CUDA port for this kernel: fall back to the compiled
            # host-only one. Accumulation kernels only read markers (they
            # write into the grid arrays), so no write-back is needed.
            with (
                ProfileManager.profile_region("kernel: " + self.kernel.name),
                self.particles.host_markers(write=False) as args_markers,
            ):
                self.kernel(
                    args_markers,
                    self.derham.args_derham,
                    self.args_domain,
                    *self._args_data,
                    *optional_args,
                )

        # apply filter
        if self.accfilter.params.use_filter is not None:
            for vec in self._vectors:
                with ProfileManager.profile_region(self._comm_region_name):
                    vec.exchange_assembly_data()
                    vec.update_ghost_regions()

                self.accfilter(vec)
            vec_finished = True

        if self.particles.clone_config is None:
            num_clones = 1
        else:
            num_clones = self.particles.clone_config.num_clones

        if num_clones > 1:
            with ProfileManager.profile_region(self._comm_region_name):
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
            )
            mat_finished = True

        # finish vector: accumulate ghost regions and update ghost regions
        if not vec_finished:
            with ProfileManager.profile_region(self._comm_region_name):
                for vec in self._vectors:
                    vec.exchange_assembly_data()
                    vec.update_ghost_regions()

        # finish matrix: accumulate ghost regions, update ghost regions and copy data for symmetric/antisymmetric block matrices
        if not mat_finished:
            with ProfileManager.profile_region(self._comm_region_name):
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
    def kernel(self) -> PyccelKernel:
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
    def accfilter(self):
        """Callable filters"""
        return self._accfilter

    def init_control_variate(self, mass_ops):
        """Set up the use of noise reduction by control variate."""

        from struphy.feec.mass import L2Projector

        # L2 projector for dofs
        self._get_L2dofs = L2Projector(self.space_id, mass_ops).get_dofs

    def show_accumulated_spline_field(self, mass_ops: WeightedMassOperators, eta_direction=0, component=0):
        r"""1D plot of the spline field corresponding to the accumulated vector.
        The latter can be viewed as the rhs of an L2-projection:

        .. math::

            \mathbb M \mathbf a = \sum_p \boldsymbol \Lambda(\boldsymbol \eta_p) * B_p\,.

        The FE coefficients :math:`\mathbf a` determine a FE :class:`~struphy.feec.psydac_derham.SplineFunction`.
        """
        from matplotlib import pyplot as plt

        from struphy.feec.mass import L2Projector

        # L2 projection
        proj = L2Projector(self.space_id, mass_ops)
        a = proj.solve(self.vectors[0])

        # create field and assign coeffs
        field = self.derham.create_spline_function("accum_field", self.space_id)
        field.vector = a

        # plot field
        eta = xp.linspace(0, 1, 100)
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
    Approximates integrals of the form

    .. math::

        I_B = \int_\Omega \int_{\mathbb R^3} \Lambda^\mu_{ijk}(\boldsymbol \eta) \, B^\mu(\boldsymbol \eta, \mathbf v) \, f^{\textrm{vol}}(\boldsymbol \eta, \mathbf v)\,\mathrm d\mathbf v \textrm d \boldsymbol \eta\,,

    for a given weight function and :math:`B^\mu` by Monte-Carlo quadrature through the particle distribution function :math:`f^{\textrm{vol}}`:

    .. math::

        f^{\textrm{vol}}(\boldsymbol \eta, \mathbf v) \approx \sum_{p=0}^{N-1} w_p \, \delta(\boldsymbol \eta - \boldsymbol \eta_p) \, \delta(\mathbf v - \mathbf v_p)\,.

    This results in a stencil (block) vector

    .. math::

        V = (V^\mu)_\mu\,,\qquad V^\mu \in \mathbb R^{\mathbb N^\alpha_\mu}\,,

    where :math:`N^\alpha_\mu` denotes the dimension of the :math:`\mu`-th component
    of the :class:`~struphy.feec.psydac_derham.Derham` space
    :math:`V_h^\alpha` (:math:`\mu,\nu = 1,2,3` for vector-valued spaces),
    with entries obtained by summing over all particles :math:`p`,

    .. math::

        V^\mu_{ijk} = \sum_{p=0}^{N-1} w_p\, \Lambda^\mu_{ijk}(\boldsymbol \eta_p) \, B^\mu_p \,.

    Here, :math:`\Lambda^\mu_{ijk}(\boldsymbol \eta_p)` denotes the :math:`ijk`-th basis function
    of the :math:`\mu`-th component of a Derham space.

    Similar to :class:`~struphy.pic.accumulation.particles_to_grid.Accumulator` but only for vectors :math:`V`.

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
        kernel: PyccelKernel,
        mass_ops: WeightedMassOperators,
        args_domain: DomainArguments,
        filter_params: FilterParameters = None,
    ):
        self._particles = particles
        self._space_id = space_id
        assert isinstance(kernel, PyccelKernel), f"{kernel} is not of type PyccelKernel"
        self._kernel = kernel
        self._derham = mass_ops.derham
        self._args_domain = args_domain

        # profiling region names (precomputed, they are looked up on every call)
        self._region_name = "accum: " + kernel.name
        self._comm_region_name = "accum comm: " + kernel.name

        self._form = self.derham.space_to_form[space_id]

        # initialize vectors
        self._vectors = []
        self._vectors_temp = []
        self._vectors_out = []

        # collect all _data attributes needed in accumulation kernel
        self._args_data = ()

        if space_id in ("H1", "L2"):
            self._vectors += [
                StencilVector(self.derham.fem_spaces[self.form].coeff_space),
            ]
            self._vectors_temp += [
                StencilVector(self.derham.fem_spaces[self.form].coeff_space),
            ]
            self._vectors_out += [
                StencilVector(self.derham.fem_spaces[self.form].coeff_space),
            ]

        elif space_id in ("Hcurl", "Hdiv", "H1vec"):
            self._vectors += [
                BlockVector(
                    self.derham.fem_spaces[self.form].coeff_space,
                ),
            ]
            self._vectors_temp += [
                BlockVector(
                    self.derham.fem_spaces[self.form].coeff_space,
                ),
            ]
            self._vectors_out += [
                BlockVector(
                    self.derham.fem_spaces[self.form].coeff_space,
                ),
            ]

        for vec in self._vectors:
            if isinstance(vec, StencilVector):
                self._args_data += (vec._data,)
            else:
                for bl in vec.blocks:
                    self._args_data += (bl._data,)

        # initialize filter
        self._accfilter = AccumFilter(filter_params, self._derham, self._space_id)

        # hand-written CUDA replacement for charge_density_0form (the only
        # AccumulatorVector kernel ported so far -- see accum_kernels_cuda.py).
        # No optional_args/domain-mapping support needed for this one.
        self._gpu_charge_density_0form = xp.cupy_backend and kernel.name in (
            "charge_density_0form",
            # gc_density_0form is the same 0-form weight scatter (see
            # accum_kernels_gc_cuda.gc_density_0form_gpu)
            "gc_density_0form",
        )
        if self._gpu_charge_density_0form:
            import cupy as cp

            args_derham = self.derham.args_derham
            self._gpu_cd0_weight_idx = self.particles.index["weights"]
            self._gpu_cd0_pn = tuple(int(p) for p in args_derham.pn)
            self._gpu_cd0_starts = tuple(int(s) for s in args_derham.starts)
            self._gpu_cd0_tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
            self._gpu_cd0_tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
            self._gpu_cd0_tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)

        # hand-written CUDA replacement for gc_mag_density_0form (5D
        # guiding-center analog of charge_density_0form -- see
        # accum_kernels_gc_cuda.py). optional_args = (ep_scale,).
        self._gpu_gc_mag_density_0form = xp.cupy_backend and kernel.name == "gc_mag_density_0form"
        if self._gpu_gc_mag_density_0form:
            import cupy as cp

            args_derham = self.derham.args_derham
            self._gpu_gcmd_mu_idx = int(self.particles.mu_idx)
            self._gpu_gcmd_pn = tuple(int(p) for p in args_derham.pn)
            self._gpu_gcmd_starts = tuple(int(s) for s in args_derham.starts)
            self._gpu_gcmd_tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
            self._gpu_gcmd_tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
            self._gpu_gcmd_tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)

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
            Values are a 1d (vector) or 2d (matrix) list with callables or xp.ndarrays used for the correction.
        """
        with ProfileManager.profile_region(self._region_name):
            self._accumulate(*optional_args, **args_control)

    def _accumulate(self, *optional_args, **args_control):
        """Body of :meth:`__call__`, see there."""

        # flags for break
        vec_finished = False

        # reset data
        for dat in self._args_data:
            dat[:] = 0.0

        # accumulate into matrix (and vector) with markers
        if self._gpu_charge_density_0form and not optional_args:
            with ProfileManager.profile_region("kernel: " + self.kernel.name + " [cuda]"):
                charge_density_0form_gpu(
                    self.particles.markers,
                    self._gpu_cd0_weight_idx,
                    self._gpu_cd0_pn,
                    self._gpu_cd0_tn1,
                    self._gpu_cd0_tn2,
                    self._gpu_cd0_tn3,
                    self._gpu_cd0_starts,
                    self._args_data[0],
                )
        elif self._gpu_gc_mag_density_0form and len(optional_args) == 1:
            with ProfileManager.profile_region("kernel: " + self.kernel.name + " [cuda]"):
                (scale,) = optional_args
                gc_mag_density_0form_gpu(
                    self.particles.markers,
                    self._gpu_gcmd_mu_idx,
                    scale,
                    self._gpu_gcmd_pn,
                    self._gpu_gcmd_tn1,
                    self._gpu_gcmd_tn2,
                    self._gpu_gcmd_tn3,
                    self._gpu_gcmd_starts,
                    self._args_data[0],
                )
        else:
            # no CUDA port for this kernel: fall back to the compiled
            # host-only one. Accumulation kernels only read markers (they
            # write into the grid arrays), so no write-back is needed.
            with (
                ProfileManager.profile_region("kernel: " + self.kernel.name),
                self.particles.host_markers(write=False) as args_markers,
            ):
                self.kernel(
                    args_markers,
                    self.derham.args_derham,
                    self.args_domain,
                    *self._args_data,
                    *optional_args,
                )

        # apply filter
        if self.accfilter.params.use_filter is not None:
            for vec in self._vectors:
                with ProfileManager.profile_region(self._comm_region_name):
                    vec.exchange_assembly_data()
                    vec.update_ghost_regions()

                self.accfilter(vec)
                vec_finished = True

        if self.particles.clone_config is None:
            num_clones = 1
        else:
            num_clones = self.particles.clone_config.num_clones

        if num_clones > 1:
            with ProfileManager.profile_region(self._comm_region_name):
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
            with ProfileManager.profile_region(self._comm_region_name):
                for vec in self._vectors:
                    vec.exchange_assembly_data()
                    vec.update_ghost_regions()

    @property
    def particles(self):
        """Particle object."""
        return self._particles

    @property
    def kernel(self) -> PyccelKernel:
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

    @property
    def accfilter(self):
        """Callable filters"""
        return self._accfilter

    def init_control_variate(self, mass_ops):
        """Set up the use of noise reduction by control variate."""

        from struphy.feec.mass import L2Projector

        # L2 projector for dofs
        self._get_L2dofs = L2Projector(self.space_id, mass_ops).get_dofs

    def show_accumulated_spline_field(self, mass_ops, eta_direction=(True, False, False), save_L2=False):
        r"""1 or 2D plot of the spline field corresponding to the accumulated vector.
        The latter can be viewed as the rhs of an L2-projection:

        .. math::

            \mathbb M \mathbf a = \sum_p \boldsymbol \Lambda(\boldsymbol \eta_p) * B_p\,.

        The FE coefficients :math:`\mathbf a` determine a FE :class:`~struphy.feec.psydac_derham.SplineFunction`.

        :param eta_direction: axes of eta to show accumulation (eta1, eta2, eta3).
        """
        assert sum(eta_direction) < 3, "Current implementation is only possible with 1 and 2D visualization"

        from matplotlib import pyplot as plt

        from struphy.feec.mass import L2Projector

        # L2 projection
        proj = L2Projector(self.space_id, mass_ops)
        a = proj.solve(self.vectors[0])

        if save_L2:
            return a

        # create field and assign coeffs
        field = self.derham.create_spline_function("accum_field", self.space_id)
        field.vector = a

        # plot field

        # initialize axis and slicing
        eta = xp.linspace(0, 1, 100)
        args = [0.5, 0.5, 0.5]

        # fill slices to plot with eta
        plt_axis = xp.flatnonzero(eta_direction)

        for idx in plt_axis:
            args[idx] = eta
        args = tuple(args)

        # field value at specified axes
        field_value = field(*args, squeeze_out=True)

        # One-dimensional case
        if len(plt_axis) == 1:
            plt.plot(eta, field_value)

            plt.xlabel(rf"$\eta_{plt_axis[0] + 1}$")
            plt.ylabel("field amplitude")

        # Two-dimensional case
        elif len(plt_axis) == 2:
            Eta1, Eta2 = xp.meshgrid(eta, eta, indexing="ij")
            pcm = plt.pcolor(Eta1, Eta2, field_value)

            plt.colorbar(pcm, label="field amplitude")
            plt.xlabel(rf"$\eta_{plt_axis[0] + 1}$")
            plt.ylabel(rf"$\eta_{plt_axis[1] + 1}$")

        plt.title(
            f'Spline field accumulated with the kernel "{self.kernel}"',
        )
        plt.show()


@dataclass
class ParticlesToGrid:
    r"""Lightweight, serializable description of a particle-to-grid coupling
    (for example charge- or current deposition) into FEEC degrees of freedom.

    A ``ParticlesToGrid`` does not perform any accumulation itself: it simply bundles
    the pieces needed to build an :class:`~struphy.pic.accumulation.particles_to_grid.AccumulatorVector`.

    Parameters
    ----------
    pic_variable : PICVariable | SPHVariable
        The kinetic variable whose markers (``pic_variable.particles``) are deposited on the grid.

    accum_space : {"H1", "Hcurl", "Hdiv", "L2", "H1vec"}
        FEEC space identifier of the vector to accumulate into.

    accum_kernel : PyccelKernel
        Pyccelized accumulation kernel matching ``accum_space``, for example
        ``PyccelKernel(accum_kernels.charge_density_0form)``.

    Examples
    --------
    >>> from struphy.pic.accumulation import accum_kernels
    >>> from struphy.pic.accumulation.particles_to_grid import ParticlesToGrid
    >>> from struphy.propagators.poisson_solve import PoissonSolve
    >>> from cunumpy import PyccelKernel
    >>> rho = ParticlesToGrid(
    ...     kinetic_ions.var,
    ...     "H1",
    ...     PyccelKernel(accum_kernels.charge_density_0form),
    ... )
    >>> poisson = PoissonSolve(rho=rho, rho_coeffs=alpha**2 / epsilon)
    """

    pic_variable: PICVariable | SPHVariable = None
    accum_space: LiteralOptions.OptsFEECSpace = None
    accum_kernel: PyccelKernel = None

    def __post_init__(self):
        if self.accum_space is not None:
            check_option(self.accum_space, LiteralOptions.OptsFEECSpace)

        assert isinstance(self.accum_kernel, PyccelKernel) or self.accum_kernel is None

    def __repr_no_defaults__(self):
        return __dataclass_repr_no_defaults__(self)
