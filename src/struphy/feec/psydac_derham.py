#!/usr/bin/env python3
import importlib.metadata

import psydac.core.bsplines as bsp
from psydac.ddm.cart import DomainDecomposition
from psydac.ddm.mpi import MockComm, MockMPI
from psydac.ddm.mpi import mpi as MPI
from psydac.feec.derivatives import Curl_3D, Divergence_3D, Gradient_3D
from psydac.feec.global_projectors import Projector_H1, Projector_H1vec, Projector_Hcurl, Projector_Hdiv, Projector_L2
from psydac.fem.grid import FemAssemblyGrid
from psydac.fem.partitioning import create_cart
from psydac.fem.splines import SplineSpace
from psydac.fem.tensor import TensorFemSpace
from psydac.fem.vector import VectorFemSpace
from psydac.linalg.basic import IdentityOperator
from psydac.linalg.block import BlockVector, BlockVectorSpace
from psydac.linalg.stencil import StencilVector, StencilVectorSpace

from struphy.bsplines import evaluation_kernels_3d as eval_3d
from struphy.bsplines.evaluation_kernels_3d import eval_spline_mpi_tensor_product_fixed
from struphy.feec.linear_operators import BoundaryOperator
from struphy.feec.local_projectors_kernels import get_local_problem_size, select_quasi_points
from struphy.feec.projectors import CommutingProjector, CommutingProjectorLocal
from struphy.fields_background.base import FluidEquilibrium, MHDequilibrium
from struphy.fields_background.equils import set_defaults
from struphy.geometry.base import Domain
from struphy.geometry.utilities import TransformedPformComponent
from struphy.initial import perturbations, utilities
from struphy.initial.base import Perturbation
from struphy.initial.perturbations import Noise
from struphy.io.options import FieldsBackground, GivenInBasis, NoiseDirections
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments
from struphy.polar.basic import PolarDerhamSpace, PolarVector
from struphy.polar.extraction_operators import PolarExtractionBlocksC1
from struphy.polar.linear_operators import PolarExtractionOperator, PolarLinearOperator
from struphy.utils.arrays import xp as np


class Derham:
    """
    The discrete Derham sequence on the logical unit cube (3d).

    Check out the corresponding `Struphy API <https://struphy.pages.mpcdf.de/struphy/api/discrete_derham.html>`_ for a hands-on introduction.

    The tensor-product discrete deRham complex is loaded using the `Psydac API <https://github.com/pyccel/psydac>`_
    and then augmented with polar sub-spaces (indicated by a bar) and boundary operators.

    .. image:: ../../pics/polar_derham.png

    Parameters
    ----------
    Nel : list[int]
        Number of elements in each direction.

    p : list[int]
        Spline degree in each direction.

    spl_kind : list[bool]
        Kind of spline in each direction (True=periodic, False=clamped).

    dirichlet_bc : list[list[bool]]
        Whether to apply homogeneous Dirichlet boundary conditions (at left or right boundary in each direction).

    nq_pr : list[int]
        Number of Gauss-Legendre quadrature points in each direction for geometric projectors (default = p+1, leads to exact integration of degree 2p+1 polynomials).

    nquads : list[int]
        Number of Gauss-Legendre quadrature points in each direction (default = p, leads to exact integration of degree 2p-1 polynomials).

    comm : mpi4py.MPI.Intracomm
        MPI communicator (within a clone if domain cloning is used, otherwise MPI.COMM_WORLD)

    mpi_dims_mask: list of bool
        True if the dimension is to be used in the domain decomposition (=default for each dimension).
        If mpi_dims_mask[i]=False, the i-th dimension will not be decomposed.

    with_projectors : bool
        Whether to add global commuting projectors to the diagram.

    polar_ck : int
        Smoothness at a polar singularity at eta_1=0 (default -1 : standard tensor product splines, OR 1 : C1 polar splines)

    local_projectors : bool
        Whether to build the local commuting projectors based on quasi-inter-/histopolation.

    domain : struphy.geometry.base.Domain
        Mapping from logical unit cube to physical domain (only needed in case of polar splines polar_ck=1).
    """

    def __init__(
        self,
        Nel: list | tuple,
        p: list | tuple,
        spl_kind: list | tuple,
        *,
        dirichlet_bc: list | tuple = None,
        nquads: list | tuple = None,
        nq_pr: list | tuple = None,
        comm=None,
        mpi_dims_mask: list = None,
        with_projectors: bool = True,
        polar_ck: int = -1,
        local_projectors: bool = False,
        domain: Domain = None,
    ):
        # number of elements, spline degrees and kind of splines in each direction (periodic vs. clamped)
        assert len(Nel) == 3
        assert len(p) == 3
        assert len(spl_kind) == 3

        self._Nel = Nel
        self._p = p
        self._spl_kind = spl_kind
        self._with_local_projectors = local_projectors

        # boundary conditions at eta=0 and eta=1 in each direction (None for periodic, 'd' for homogeneous Dirichlet)
        if dirichlet_bc is not None:
            assert len(dirichlet_bc) == 3
            # make sure that boundary conditions are compatible with spline space
            assert np.all([bc == (False, False) for i, bc in enumerate(dirichlet_bc) if spl_kind[i]])

        self._dirichlet_bc = dirichlet_bc

        # default p: exact integration of degree 2p+1 polynomials
        if nquads is None:
            self._nquads = [pi + 1 for pi in p]
        else:
            assert len(nquads) == 3
            self._nquads = nquads

        # default p + 1 : exact integration of degree 2p+1 polynomials
        if nq_pr is None:
            self._nq_pr = [pi + 1 for pi in p]
        else:
            assert len(nq_pr) == 3
            self._nq_pr = nq_pr

        # MPI communicators
        self._comm = comm

        # set polar splines (currently standard tensor-product (-1) and C^1 polar splines (+1) are supported)
        assert polar_ck in {-1, 1}
        self._polar_ck = polar_ck

        # derham sequence
        _derham = self.init_derham(
            Nel,
            self.p,
            self.spl_kind,
            comm=self.comm,
            mpi_dims_mask=mpi_dims_mask,
        )
        self._grad, self._curl, self._div = _derham.derivatives_as_matrices

        # expose name-to-form dict
        self._space_to_form = {
            "H1": "0",
            "Hcurl": "1",
            "Hdiv": "2",
            "L2": "3",
            "H1vec": "v",
        }

        _projectors = _derham.projectors(nquads=self.nq_pr)

        # Attributes for vector spaces, FE spline spaces and projectors
        self._Vh = {}
        self._Vh_fem = {}
        # Global projectors
        self._P = {}
        # Local projectors
        if local_projectors:
            self._Ploc = {}

        # info for 1d spline spaces grids
        self._nbasis = {}
        self._spline_types = {}
        self._spline_types_pyccel = {}

        self._proj_grid_pts = {}
        self._proj_grid_wts = {}
        # We only need the subs for the global projector operators, not for the local projectors.
        self._proj_grid_subs = {}

        if local_projectors:
            self._proj_loc_grid_pts = {}
            self._proj_loc_grid_wts = {}

        self._quad_grid_pts = {}
        self._quad_grid_wts = {}
        self._quad_grid_spans = {}
        self._quad_grid_bases = {}

        # i is an int that represents the id of the p-form space. For instance, for V_0, i = 0.
        psydac_ver = importlib.metadata.version("psydac")
        for i, sp_form in enumerate(self.space_to_form.values()):
            # FEM space and projector
            if sp_form == "v":
                _h1vec_space = VectorFemSpace(
                    _derham.V0,
                    _derham.V0,
                    _derham.V0,
                )
                if "dev" in psydac_ver:
                    _h1vec_space.symbolic_space = "H1vec"
                self._Vh_fem[sp_form] = _h1vec_space
                self._P[sp_form] = Projector_H1vec(self.Vh_fem[sp_form])
            else:
                self._Vh_fem[sp_form] = getattr(_derham, "V" + str(i))
                self._P[sp_form] = _projectors[i]

            # Vector space
            self._Vh[sp_form] = self.Vh_fem[sp_form].coeff_space

            # grid attributes
            self._nbasis[sp_form] = []
            self._spline_types[sp_form] = []
            self._spline_types_pyccel[sp_form] = []

            self._proj_grid_pts[sp_form] = []
            self._proj_grid_wts[sp_form] = []
            self._proj_grid_subs[sp_form] = []

            if local_projectors:
                self._proj_loc_grid_pts[sp_form] = []
                self._proj_loc_grid_wts[sp_form] = []

            self._quad_grid_pts[sp_form] = []
            self._quad_grid_wts[sp_form] = []
            self._quad_grid_spans[sp_form] = []
            self._quad_grid_bases[sp_form] = []

            fem_space = self.Vh_fem[sp_form]
            # Here we check if we are working with a vector valued space
            if isinstance(fem_space, VectorFemSpace):
                # We iterate over each component of the vector
                for comp_space in fem_space.spaces:
                    # nquads must be manually set (has been deprecated in psydac)
                    # comp_space.nquads = self.nquads

                    self._nbasis[sp_form] += [[]]
                    self._spline_types[sp_form] += [[]]
                    self._spline_types_pyccel[sp_form] += [[]]
                    self._proj_grid_pts[sp_form] += [[]]
                    self._proj_grid_wts[sp_form] += [[]]
                    if local_projectors:
                        self._proj_loc_grid_pts[sp_form] += [[]]
                        self._proj_loc_grid_wts[sp_form] += [[]]
                    self._proj_grid_subs[sp_form] += [[]]
                    self._quad_grid_pts[sp_form] += [[]]
                    self._quad_grid_wts[sp_form] += [[]]
                    self._quad_grid_spans[sp_form] += [[]]
                    self._quad_grid_bases[sp_form] += [[]]

                    # space iterates over each of the spatial coordinates.
                    for d, (space, s, e, quad_grid, nquad) in enumerate(
                        zip(
                            comp_space.spaces,
                            comp_space.coeff_space.starts,
                            comp_space.coeff_space.ends,
                            self.get_quad_grids(comp_space),
                            self.nquads,
                        ),
                    ):
                        assert isinstance(space, SplineSpace)
                        fag = quad_grid[nquad]
                        assert isinstance(fag, FemAssemblyGrid)

                        self._nbasis[sp_form][-1] += [space.nbasis]
                        self._spline_types[sp_form][-1] += [space.basis]
                        self._spline_types_pyccel[sp_form][-1] += [
                            int(space.basis == "M"),
                        ]

                        if local_projectors:
                            ptsloc, wtsloc = get_pts_and_wts_quasi(
                                space,
                                polar_shift=d == 0 and self.polar_ck == 1,
                            )
                            self._proj_loc_grid_pts[sp_form][-1] += [ptsloc]
                            self._proj_loc_grid_wts[sp_form][-1] += [wtsloc]

                        pts, wts, subs = get_pts_and_wts(
                            space,
                            s,
                            e,
                            n_quad=self.nq_pr[d],
                            polar_shift=d == 0 and self.polar_ck == 1,
                        )
                        self._proj_grid_subs[sp_form][-1] += [subs]

                        self._proj_grid_pts[sp_form][-1] += [pts]
                        self._proj_grid_wts[sp_form][-1] += [wts]
                        self._quad_grid_pts[sp_form][-1] += [fag.points]
                        self._quad_grid_wts[sp_form][-1] += [fag.weights]
                        self._quad_grid_spans[sp_form][-1] += [
                            fag.spans,
                        ]
                        self._quad_grid_bases[sp_form][-1] += [
                            fag.basis,
                        ]

                    self._spline_types_pyccel[sp_form][-1] = np.array(
                        self._spline_types_pyccel[sp_form][-1],
                    )
            # In this case we are working with a scalar valued space
            elif isinstance(fem_space, TensorFemSpace):
                # nquads must be manually set (has been deprecated in psydac)
                # fem_space.nquads = self.nquads

                # space iterates over each of the spatial coordinates.
                for d, (space, s, e, quad_grid, nquad) in enumerate(
                    zip(
                        fem_space.spaces,
                        fem_space.coeff_space.starts,
                        fem_space.coeff_space.ends,
                        self.get_quad_grids(fem_space),
                        self.nquads,
                    ),
                ):
                    assert isinstance(space, SplineSpace)
                    fag = quad_grid[nquad]
                    assert isinstance(fag, FemAssemblyGrid)

                    self._nbasis[sp_form] += [space.nbasis]
                    self._spline_types[sp_form] += [space.basis]
                    self._spline_types_pyccel[sp_form] += [
                        int(space.basis == "M"),
                    ]

                    if local_projectors:
                        ptsloc, wtsloc = get_pts_and_wts_quasi(
                            space,
                            polar_shift=d == 0 and self.polar_ck == 1,
                        )
                        self._proj_loc_grid_pts[sp_form] += [ptsloc]
                        self._proj_loc_grid_wts[sp_form] += [wtsloc]

                    pts, wts, subs = get_pts_and_wts(
                        space,
                        s,
                        e,
                        n_quad=self.nq_pr[d],
                        polar_shift=d == 0 and self.polar_ck == 1,
                    )
                    self._proj_grid_subs[sp_form] += [subs]
                    self._proj_grid_pts[sp_form] += [pts]
                    self._proj_grid_wts[sp_form] += [wts]

                    self._quad_grid_pts[sp_form] += [fag.points]
                    self._quad_grid_wts[sp_form] += [fag.weights]
                    self._quad_grid_spans[sp_form] += [fag.spans]
                    self._quad_grid_bases[sp_form] += [fag.basis]

                self._spline_types_pyccel[sp_form] = np.array(
                    self._spline_types_pyccel[sp_form],
                )
            else:
                raise TypeError(f"{fem_space = } is not a valid type.")

        # break points
        self._breaks = [space.breaks for space in _derham.spaces[0].spaces]

        # index arrays
        self._indN = [
            (
                np.indices((space.ncells, space.degree + 1))[1]
                + np.arange(
                    space.ncells,
                )[:, None]
            )
            % space.nbasis
            for space in self._Vh_fem["0"].spaces
        ]
        self._indD = [
            (
                np.indices((space.ncells, space.degree + 1))[1]
                + np.arange(
                    space.ncells,
                )[:, None]
            )
            % space.nbasis
            for space in self._Vh_fem["3"].spaces
        ]

        # distribute info on domain decomposition
        self._domain_array = self._get_domain_array()
        self._breaks_loc = [
            self.breaks[k][self.domain_decomposition.starts[k] : self.domain_decomposition.ends[k] + 2]
            for k in range(3)
        ]

        self._index_array = self._get_index_array(
            self.domain_decomposition,
        )
        self._index_array_N = self._get_index_array(self._Vh["0"].cart)
        self._index_array_D = self._get_index_array(self._Vh["3"].cart)

        self._neighbours = self._get_neighbours()

        # ------ (Polar) deRham spaces and projectors ------
        if self.polar_ck == -1:
            ck_blocks = None
        else:
            assert domain is not None
            ck_blocks = PolarExtractionBlocksC1(domain, self)

        self._Vh_pol = {}
        self._boundary_ops = {}
        self._extraction_ops = {}
        self._dofs_extraction_ops = {}

        # If we are dealing with local projection operators we must compute the weight w^i_j for interpolation, and from them the weights
        # wh^i_j for histopolation. They can be computed using the quasi-interpolation points for all spatial directions.
        # Fortunately we already have access to them in the form of self._proj_loc_grid_pts[0].
        if local_projectors:
            # Allways call get_weights_local_projector with the grid points and discrete vector space of 0-forms
            self._wij, self._whij = get_weights_local_projector(
                self._proj_loc_grid_pts["0"],
                self.Vh_fem["0"],
            )

        for i, (sp_id, sp_form) in enumerate(self.space_to_form.items()):
            vec_space = self._Vh[sp_form]
            # ------ Extraction operators ------
            # tensor product case
            if self.polar_ck == -1:
                pol_space = self._Vh[sp_form]

                self._extraction_ops[sp_form] = IdentityOperator(pol_space)
                self._dofs_extraction_ops[sp_form] = IdentityOperator(
                    pol_space,
                )

            # C^1 polar spline case
            else:
                pol_space = PolarDerhamSpace(self, sp_id)

                self._extraction_ops[sp_form] = PolarExtractionOperator(
                    vec_space,
                    pol_space,
                    ck_blocks.e_ten_to_pol[sp_form],
                )

                self._dofs_extraction_ops[sp_form] = PolarExtractionOperator(
                    vec_space,
                    pol_space,
                    ck_blocks.p_ten_to_pol[sp_form],
                    ck_blocks.p_ten_to_ten[sp_form],
                )

            self._Vh_pol[sp_form] = pol_space

            # ------ Hom. Dirichlet boundary operators ------
            if self.dirichlet_bc is None:
                self._boundary_ops[sp_form] = IdentityOperator(pol_space)
            else:
                self._boundary_ops[sp_form] = BoundaryOperator(
                    pol_space,
                    sp_id,
                    self.dirichlet_bc,
                )

            # ------ Assemble projectors ------
            if with_projectors:
                if local_projectors:
                    fem_space = self.Vh_fem[sp_form]
                    # We also need the FEM spline space that contains B-splines in all three directions
                    fem_space_B = self.Vh_fem["0"]
                    # As well as the FEM spline space that contains D-splines in all three directions.
                    fem_space_D = self.Vh_fem["3"]
                    self._Ploc[sp_form] = CommutingProjectorLocal(
                        sp_id,
                        sp_form,
                        fem_space,
                        self._proj_loc_grid_pts[sp_form],
                        self._proj_loc_grid_wts[sp_form],
                        self._wij,
                        self._whij,
                        fem_space_B,
                        fem_space_D,
                    )
                self._P[sp_form] = CommutingProjector(
                    self._P[sp_form],
                    self._dofs_extraction_ops[sp_form],
                    self._extraction_ops[sp_form],
                    self._boundary_ops[sp_form],
                )

        # set discrete derivatives with polar linear operators
        if self.polar_ck == 1:
            self._grad = PolarLinearOperator(
                self._Vh_pol["0"],
                self._Vh_pol["1"],
                self._grad,
                ck_blocks.grad_pol_to_ten,
                ck_blocks.grad_pol_to_pol,
                ck_blocks.grad_e3,
            )
            self._curl = PolarLinearOperator(
                self._Vh_pol["1"],
                self._Vh_pol["2"],
                self._curl,
                ck_blocks.curl_pol_to_ten,
                ck_blocks.curl_pol_to_pol,
                ck_blocks.curl_e3,
            )
            self._div = PolarLinearOperator(
                self._Vh_pol["2"],
                self._Vh_pol["3"],
                self._div,
                ck_blocks.div_pol_to_ten,
                ck_blocks.div_pol_to_pol,
                ck_blocks.div_e3,
            )

        # set discrete derivatives with and without boundary operators
        self._grad_bcfree = self._grad
        self._curl_bcfree = self._curl
        self._div_bcfree = self._div

        self._grad = self._boundary_ops["1"] @ self._grad @ self._boundary_ops["0"].T
        self._curl = self._boundary_ops["2"] @ self._curl @ self._boundary_ops["1"].T
        self._div = self._boundary_ops["3"] @ self._div @ self._boundary_ops["2"].T

        # collect arguments for kernels
        self._args_derham = DerhamArguments(
            np.array(self.p),
            self.Vh_fem["0"].knots[0],
            self.Vh_fem["0"].knots[1],
            self.Vh_fem["0"].knots[2],
            np.array(self.Vh["0"].starts),
        )

    @property
    def Nel(self):
        """List of number of elements (=cells) in each direction."""
        return self._Nel

    @property
    def p(self):
        """List of B-spline degrees in each direction."""
        return self._p

    @property
    def spl_kind(self):
        """List of spline type (periodic=True or clamped=False) in each direction."""
        return self._spl_kind

    @property
    def dirichlet_bc(self):
        """None, or list of boundary conditions in each direction.
        Each entry is a list with two entries (left and right boundary), "d" (hom. Dirichlet) or None (periodic).
        """
        return self._dirichlet_bc

    @property
    def nquads(self):
        """List of number of Gauss-Legendre quadrature points in each direction (default = p, leads to exact integration of degree 2p-1 polynomials)."""
        return self._nquads

    @property
    def nq_pr(self):
        """List of number of Gauss-Legendre quadrature points in histopolation (default = p + 1) in each direction."""
        return self._nq_pr

    @property
    def with_local_projectors(self):
        """True if local projectors are to be used instead of the default global ones."""
        return self._with_local_projectors

    @property
    def comm(self):
        """MPI communicator."""
        return self._comm

    @property
    def polar_ck(self):
        """C^k smoothness at eta_1=0."""
        return self._polar_ck

    @property
    def breaks(self):
        """List of break points (=cell interfaces) in each direction."""
        return self._breaks

    @property
    def indN(self):
        """List of 2d arrays holding global spline indices (N) in each element in the three directions."""
        return self._indN

    @property
    def indD(self):
        """List of 2d arrays holding global spline indices (D) in each element in the three directions."""
        return self._indD

    @property
    def domain_decomposition(self):
        """Psydac's domain decomposition object (same for all vector spaces!)."""
        return self._domain_decomposition

    @property
    def domain_array(self):
        """
        A 2d array[float] of shape (comm.Get_size(), 9). The row index denotes the process number and
        for n=0,1,2:

            * domain_array[i, 3*n + 0] holds the LEFT domain boundary of process i in direction eta_(n+1).
            * domain_array[i, 3*n + 1] holds the RIGHT domain boundary of process i in direction eta_(n+1).
            * domain_array[i, 3*n + 2] holds the number of cells of process i in direction eta_(n+1).
        """
        return self._domain_array

    @property
    def breaks_loc(self):
        """
        The domain local to this process.
        """
        return self._breaks_loc

    @property
    def index_array(self):
        """
        A 2d array[int] of shape (comm.Get_size(), 6). The row index denotes the process number and
        for n=0,1,2:

            * arr[i, 2*n + 0] holds the global start index of cells of process i in direction eta_(n+1).
            * arr[i, 2*n + 1] holds the global end index of cells of process i in direction eta_(n+1).
        """
        return self._index_array

    @property
    def index_array_N(self):
        """
        A 2d array[int] of shape (comm.Get_size(), 6). The row index denotes the process number and
        for n=0,1,2:

            * arr[i, 2*n + 0] holds the global start index of B-splines (N) of process i in direction eta_(n+1).
            * arr[i, 2*n + 1] holds the global end index of B-splines (N) of process i in direction eta_(n+1).
        """
        return self._index_array_N

    @property
    def index_array_D(self):
        """
        A 2d array[int] of shape (comm.Get_size(), 6). The row index denotes the process number
        and for n=0,1,2:

            * arr[i, 2*n + 0] holds the global start index of M-splines (D) of process i in direction eta_(n+1).
            * arr[i, 2*n + 1] holds the global end index of M-splines (D) of process i in direction eta_(n+1).
        """
        return self._index_array_D

    @property
    def neighbours(self):
        """
        A 3d array[int] with shape (3,3,3). It contains the 26 neighbouring process ids (rank).
        This is done in terms of N-spline start/end indices. The i-th index indicates direction eta_(i+1).
        0 is a left neighbour, 1 is the same plane as the current process, 2 is a right neighbour.
        For more detail see _get_neighbours().
        """
        return self._neighbours

    @property
    def space_to_form(self):
        """Dictionary containing the names of the continuous spaces and corresponding discrete spaces."""
        return self._space_to_form

    @property
    def Vh(self):
        """Dictionary containing finite-dimensional vector spaces (sub-spaces of continuous spaces, Stencil-/BlockVectorSpace)."""
        return self._Vh

    @property
    def Vh_fem(self):
        """Dictionary containing FEM spline spaces (TensorFem-/VectorFemSpace)."""
        return self._Vh_fem

    @property
    def nbasis(self):
        """Dictionary containing number of 1d basis functions for each component and spatial direction."""
        return self._nbasis

    @property
    def spline_types(self):
        """Dictionary holding 1d spline types for each component and spatial direction, entries either 'B' or 'M'."""
        return self._spline_types

    @property
    def spline_types_pyccel(self):
        """Dictionary holding 1d spline types for each component and spatial direction, entries either 0 (='B') or 1 (='M')."""
        return self._spline_types_pyccel

    @property
    def proj_grid_pts(self):
        """Dictionary of quadrature points for histopolation (or Greville points for interpolation) in format (ii, iq) = (interval, quadrature point)."""
        return self._proj_grid_pts

    @property
    def proj_grid_wts(self):
        """Dictionary of quadrature weights for histopolation (or 1's for interpolation) in format (ii, iq) = (interval, quadrature point)."""
        return self._proj_grid_wts

    @property
    def proj_grid_subs(self):
        """Dictionary of histopolation subintervals (or 0's for interpolation) as 1d arrays.
        A value of 1 indicates that the corresponding cell is the second subinterval of a split Greville cell (for histopolation with even degree)."""
        return self._proj_grid_subs

    @property
    def quad_grid_pts(self):
        """Dictionary of quadrature points for integration over grid cells in format (ni, nq) = (cell, quadrature point)."""
        return self._quad_grid_pts

    @property
    def quad_grid_wts(self):
        """Dictionary of quadrature weights for integration over grid cells in format (ni, nq) = (cell, quadrature point)."""
        return self._quad_grid_wts

    @property
    def quad_grid_spans(self):
        """Dictionary of knot span indices of grid cells."""
        return self._quad_grid_spans

    @property
    def quad_grid_bases(self):
        """Dictionary of basis functions evaluated at quadrature grids in format (ni, bl, 0, nq) = (cell, basis function, derivative=0, quadrature point)."""
        return self._quad_grid_bases

    @property
    def extraction_ops(self):
        """Dictionary holding basis extraction operators, either IdentityOperator or PolarExtractionOperator."""
        return self._extraction_ops

    @property
    def dofs_extraction_ops(self):
        """Dictionary holding dof extraction operators for commuting projectors, either IdentityOperator or PolarExtractionOperator."""
        return self._dofs_extraction_ops

    @property
    def boundary_ops(self):
        """Dictionary holding essential boundary operators (BoundaryOperator) OR IdentityOperators."""
        return self._boundary_ops

    @property
    def P(self):
        """Dictionary holding global commuting projectors."""
        if self.with_local_projectors:
            return self._Ploc
        else:
            return self._P

    @property
    def Vh_pol(self):
        """Polar sub-spaces, either PolarDerhamSpace (with polar splines) or Stencil-/BlockVectorSpace (same as self.Vh)"""
        return self._Vh_pol

    @property
    def grad_bcfree(self):
        """Discrete gradient Vh0_pol (H1) -> Vh1_pol (Hcurl) w/o boundary operator."""
        return self._grad_bcfree

    @property
    def curl_bcfree(self):
        """Discrete curl Vh1_pol (Hcurl) -> Vh2_pol (Hdiv) w/o boundary operator."""
        return self._curl_bcfree

    @property
    def div_bcfree(self):
        """Discrete divergence Vh2_pol (Hdiv) -> Vh3_pol (L2) w/o boundary operator."""
        return self._div_bcfree

    @property
    def grad(self):
        """Discrete gradient Vh0_pol (H1) -> Vh1_pol (Hcurl)."""
        return self._grad

    @property
    def curl(self):
        """Discrete curl Vh1_pol (Hcurl) -> Vh2_pol (Hdiv)."""
        return self._curl

    @property
    def div(self):
        """Discrete divergence Vh2_pol (Hdiv) -> Vh3_pol (L2)."""
        return self._div

    @property
    def args_derham(self):
        """Collection of mandatory arguments for pusher kernels."""
        return self._args_derham

    # --------------------------
    #      methods:
    # --------------------------
    def init_derham(
        self,
        Nel: tuple | list,
        p: tuple | list,
        spl_kind: tuple | list,
        comm=None,
        mpi_dims_mask: tuple | list = None,
    ):
        """Discretize the Derahm complex. Allows for the use of tiny-psydac.

        Parameters
        ----------
        Nel : list[int]
            Number of elements in each direction.

        p : list[int]
            Spline degree in each direction.

        spl_kind : list[bool]
            Kind of spline in each direction (True=periodic, False=clamped).

        comm : mpi4py.MPI.Intracomm
            MPI communicator (within a clone if domain cloning is used, otherwise MPI.COMM_WORLD)

        mpi_dims_mask: list of bool
            True if the dimension is to be used in the domain decomposition (=default for each dimension).
            If mpi_dims_mask[i]=False, the i-th dimension will not be decomposed.
        """

        psydac_ver = importlib.metadata.version("psydac")

        if "dev" in psydac_ver:
            # use tiny-psydac version
            self._domain_decomposition = DomainDecomposition(Nel, spl_kind, comm=comm, mpi_dims_mask=mpi_dims_mask)

            _derham = self._discretize_derham(
                Nel=Nel,
                p=p,
                spl_kind=spl_kind,
                ddm=self.domain_decomposition,
            )
        else:
            from psydac.api.discretization import discretize
            from sympde.topology import Cube
            from sympde.topology import Derham as Derham_psy

            # Psydac symbolic logical domain (unit cube)
            self._domain_log = Cube(
                "C",
                bounds1=(0, 1),
                bounds2=(0, 1),
                bounds3=(0, 1),
            )

            # Psydac symbolic Derham
            self._derham_symb = Derham_psy(self._domain_log)

            # discrete logical domain : the parallelism is initiated here.
            self._domain_log_h = discretize(
                self._domain_log,
                ncells=Nel,
                comm=comm,
                periodic=spl_kind,
                mpi_dims_mask=mpi_dims_mask,
            )

            # Psydac discrete de Rham, projectors and derivatives
            _derham = discretize(
                self._derham_symb,
                self._domain_log_h,
                degree=p,
            )  # , nquads=self.nquads) # nquads can no longer be passed to a call to discretize on a FemSpace #403

        return _derham

    def create_spline_function(
        self,
        name: str,
        space_id: str,
        coeffs: StencilVector | BlockVector = None,
        backgrounds: FieldsBackground | list = None,
        perturbations: Perturbation | list = None,
        domain: Domain = None,
        equil: FluidEquilibrium = None,
        verbose: bool = True,
    ):
        """Creat a callable spline function.

        Parameters
        ----------
        name : str
            Field's key to be used for instance when saving to hdf5 file.

        space_id : str
            Space identifier for the field ("H1", "Hcurl", "Hdiv", "L2" or "H1vec").

        coeffs : StencilVector | BlockVector
            The spline coefficients.

        backgrounds : FieldsBackground | list
            For the initial condition.

        perturbations : Perturbation | list
            For the initial condition.

        domain : Domain
            Mapping for pullback/transform of initial condition.

        equil : FLuidEquilibrium
            Fluid background used for inital condition.
        """
        return SplineFunction(
            name,
            space_id,
            self,
            coeffs,
            backgrounds=backgrounds,
            perturbations=perturbations,
            domain=domain,
            equil=equil,
            verbose=verbose,
        )

    def prepare_eval_tp_fixed(self, grids_1d):
        """Obtain knot span indices and spline basis functions evaluated at tensor product grid.

        Parameters
        ----------
        grids_1d : 3-list of 1d arrays
            Points of the tensor product grid.

        Returns
        -------
        spans : 3-tuple of 2d int arrays
            Knot span indices in each direction in format (n, nq).

        bns : 3-tuple of 3d float arrays
            Values of p + 1 non-zero B-Splines at quadrature points in format (n, nq, basis).

        bds : 3-tuple of 3d float arrays
            Values of p non-zero D-Splines at quadrature points in format (n, nq, basis).
        """

        # spline degree and knot vectors must come from N-spline spaces (V0 space)
        spans, bns, bds = [], [], []

        for etas, space_1d, end in zip(grids_1d, self.Vh_fem["0"].spaces, self.Vh["0"].ends):
            span, bn, bd = self._get_span_and_basis_for_eval_mpi(
                etas,
                space_1d,
                end,
            )
            spans += [span]
            bns += [bn]
            bds += [bd]

        return tuple(spans), tuple(bns), tuple(bds)

    # --------------------------
    #      private methods:
    # --------------------------
    def _discretize_derham(
        self,
        Nel: tuple | list,
        p: tuple | list,
        spl_kind: tuple | list,
        ddm: DomainDecomposition = None,
    ):
        """Call routines copied and simplified from psydac.

        Parameters
        ----------
        Nel : list[int]
            Number of elements in each direction.

        p : list[int]
            Spline degree in each direction.

        spl_kind : list[bool]
            Kind of spline in each direction (True=periodic, False=clamped).

        ddm : DomainDecomposition
            Psaydac domain decomposition object.
        """
        ldim = 3
        bases = ["B"] + ldim * ["M"]
        derham_spaces = ["H1", "Hcurl", "Hdiv", "L2"]

        spaces = [
            self._discretize_space(
                V,
                basis,
                Nel=Nel,
                degree=p,
                spl_kind=spl_kind,
                ddm=ddm,
            )
            for V, basis in zip(derham_spaces, bases)
        ]

        return DiscreteDerham(*spaces)

    def _discretize_space(
        self,
        V: str,
        basis: str,
        *,
        Nel: tuple | list = None,
        degree: tuple | list = None,
        spl_kind: tuple | list = None,
        ddm: DomainDecomposition = None,
    ):
        """
        This function creates discrete Derham spaces over the 3D unit cube (copied partly from psydac).

        Parameters
        ----------
        V : str
            H1, Hcurl, Hdiv or L2 (at the moment).

        basis: str
            Either 'B' (B-splines) or 'M' (D-splines).

        Nel : list[int]
            Number of elements in each direction.

        degree : list[int]
            Spline degree in each direction.

        spl_kind : list[bool]
            Kind of spline in each direction (True=periodic, False=clamped).

        ddm : DomainDecomposition
            Psaydac domain decomposition object.

        For more details see:

        [1] : A. Buffa, J. Rivas, G. Sangalli, and R.G. Vazquez. Isogeometric
        Discrete Differential Forms in Three Dimensions. SIAM J. Numer. Anal.,
        49:818-844, 2011. DOI:10.1137/100786708. (Section 4.1)

        [2] : A. Buffa, C. de Falco, and G. Sangalli. IsoGeometric Analysis:
        Stable elements for the 2D Stokes equation. Int. J. Numer. Meth. Fluids,
        65:1407-1422, 2011. DOI:10.1002/fld.2337. (Section 3)

        [3] : A. Bressan, and G. Sangalli. Isogeometric discretizations of the
        Stokes problem: stability analysis by the macroelement technique. IMA J.
        Numer. Anal., 33(2):629-651, 2013. DOI:10.1093/imanum/drr056.

        Returns
        -------
        Vh : TensorFemSpace | VectorFemSpace
            The discrete FEM space.
        """

        ncells = Nel
        periodic = spl_kind
        degree_i = degree
        multiplicity_i = (1, 1, 1)

        # unit cube
        min_coords = (0.0, 0.0, 0.0)
        max_coords = (1.0, 1.0, 1.0)

        assert (
            len(ncells) == len(periodic) == len(degree_i) == len(multiplicity_i) == len(min_coords) == len(max_coords)
        )

        # Create uniform grid
        grids = [np.linspace(xmin, xmax, num=ne + 1) for xmin, xmax, ne in zip(min_coords, max_coords, ncells)]

        # Create 1D finite element spaces and precompute quadrature data
        spaces_1d = [
            SplineSpace(p, multiplicity=m, grid=grid, periodic=P)
            for p, m, grid, P in zip(degree_i, multiplicity_i, grids, periodic)
        ]

        carts = create_cart([ddm], [spaces_1d])

        Vh = TensorFemSpace(ddm, *spaces_1d, cart=carts[0])

        if V == "H1":
            Wh = Vh
        elif V == "Hcurl":
            spaces = [
                Vh.reduce_degree(axes=[0], multiplicity=Vh.multiplicity[0:1], basis=basis),
                Vh.reduce_degree(axes=[1], multiplicity=Vh.multiplicity[1:2], basis=basis),
                Vh.reduce_degree(axes=[2], multiplicity=Vh.multiplicity[2:], basis=basis),
            ]
            Wh = VectorFemSpace(*spaces)
        elif V == "Hdiv":
            spaces = [
                Vh.reduce_degree(axes=[1, 2], multiplicity=Vh.multiplicity[1:], basis=basis),
                Vh.reduce_degree(axes=[0, 2], multiplicity=[Vh.multiplicity[0], Vh.multiplicity[2]], basis=basis),
                Vh.reduce_degree(axes=[0, 1], multiplicity=Vh.multiplicity[:2], basis=basis),
            ]
            Wh = VectorFemSpace(*spaces)
        elif V == "L2":
            Wh = Vh.reduce_degree(axes=[0, 1, 2], multiplicity=Vh.multiplicity, basis=basis)
        else:
            raise ValueError(f"V must be one of H1, Hcurl, Hdiv or L2, but is {V = }.")

        Wh.symbolic_space = V
        for key in Wh._refined_space:
            Wh.get_refined_space(key).symbolic_space = V

        return Wh

    def _get_domain_array(self):
        """
        Uses mpi.Allgather to distribute information on domain decomposition to all processes.

        Returns
        -------
        dom_arr : np.ndarray
            A 2d array of shape (#MPI processes, 9). The row index denotes the process rank. The columns are for n=0,1,2:
                - arr[i, 3*n + 0] holds the LEFT domain boundary of process i in direction eta_(n+1).
                - arr[i, 3*n + 1] holds the RIGHT domain boundary of process i in direction eta_(n+1).
                - arr[i, 3*n + 2] holds the number of cells of process i in direction eta_(n+1).
        """

        # MPI info
        if self.comm:
            nproc = self.comm.Get_size()
        else:
            nproc = 1

        # send buffer
        dom_arr_loc = np.zeros(9, dtype=float)

        # main array (receive buffers)
        dom_arr = np.zeros(nproc * 9, dtype=float)

        # Get global starts and ends of domain decomposition
        gl_s = self.domain_decomposition.starts
        gl_e = self.domain_decomposition.ends

        # fill local domain array
        for n, (el_sta, el_end, brks) in enumerate(zip(gl_s, gl_e, self.breaks)):
            dom_arr_loc[3 * n + 0] = brks[el_sta + 0]
            dom_arr_loc[3 * n + 1] = brks[el_end + 1]
            dom_arr_loc[3 * n + 2] = el_end - el_sta + 1

        # distribute
        if not isinstance(self.comm, (MockComm, type(None))):
            self.comm.Allgather(dom_arr_loc, dom_arr)
        else:
            dom_arr[:] = dom_arr_loc

        return dom_arr.reshape(nproc, 9)

    def _get_index_array(self, decomposition):
        """
        Uses mpi.Allgather to distribute information on domain/cart decomposition to all processes.

        Parameters
        ----------
        decomposition : DomainDecomposition | CartDecomposition
            Psydac's domain or cart decomposition object. The former is the same for all spaces, the latter different.

        Returns
        -------
        ind_arr : np.ndarray
            A 2d array of shape (#MPI processes, 6). The row index denotes the process rank. The columns are for n=0,1,2:
                - arr[i, 2*n + 0] holds the global start index process i in direction eta_(n+1).
                - arr[i, 2*n + 1] holds the global end index of process i in direction eta_(n+1).
        """

        # MPI info
        if not isinstance(self.comm, (MockComm, type(None))):
            nproc = self.comm.Get_size()
        else:
            nproc = 1

        # send buffer
        ind_arr_loc = np.zeros(6, dtype=int)

        # main array (receive buffers)
        ind_arr = np.zeros(nproc * 6, dtype=int)

        # Get global starts and ends of cart OR domain decomposition
        gl_s = decomposition.starts
        gl_e = decomposition.ends

        # fill local domain array
        for n, (sta, end) in enumerate(zip(gl_s, gl_e)):
            ind_arr_loc[2 * n + 0] = sta
            ind_arr_loc[2 * n + 1] = end

        # distribute
        if not isinstance(self.comm, (MockComm, type(None))):
            self.comm.Allgather(ind_arr_loc, ind_arr)
        else:
            ind_arr[:] = ind_arr_loc

        return ind_arr.reshape(nproc, 6)

    def _get_neighbours(self):
        """
        For each mpi process, compute the 26 neighbouring processes (3x3x3 cube except the most inner element).
        This is done in terms of domain decomposition start/end indices.

        For fixed eta1-index k, eta2 as row index, eta3 as column index, we have:

                    |         |
            (k,0,0) | (k,0,1) | (k,0,2)
                    |         |
            ---------------------------
                    |         |
            (k,1,0) | (k,1,1) | (k,1,2)
                    |         |
            ---------------------------
                    |         |
            (k,2,0) | (k,2,1) | (k,2,2)
                    |         |

        The element is the rank number (can also be itself) and -1 if there is no neighbour.
        The element with index (1,1,1) (center of the cube) is always -1.

        Returns
        -------
        neighbours : np.ndarray
            A 3d array of shape (3,3,3).
            The i-th axis is the direction eta_(i+1). Neighbours along the faces have index with two 1s,
            neighbours along the edges only have one 1, neighbours along the edges have no 1 in the index.
        """

        neighs = np.empty((3, 3, 3), dtype=int)

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    comp = [i, j, k]
                    ind = tuple(comp)
                    neighs[ind] = self._get_neighbour_one_component(comp)

        return neighs

    def _get_neighbour_one_component(self, comp):
        """
        Computes the process id of a neighbour in direction of comp (c.f. _neighbours).

        Parameters
        ----------
        comp : list
            list with 3 entries.

        Returns
        -------
        neigh_id : int
            ID of neighbouring process.
        """
        assert len(comp) == 3

        # get space info
        ncells = self.domain_decomposition.ncells
        kinds = self.domain_decomposition.periods

        # global starts and end cell indices of process
        gl_s = self.domain_decomposition.starts
        gl_e = self.domain_decomposition.ends

        # get communicator info
        rank = self.domain_decomposition.rank
        size = self.domain_decomposition.size

        neigh_id = -1

        # central component is always the process itself
        if comp == [1, 1, 1]:
            return neigh_id

        comp = np.array(comp)
        kinds = np.array(kinds)

        # if only one process: check if comp is neighbour in non-peridic directions, if this is not the case then return the rank as neighbour id
        if size == 1:
            if (comp[kinds == False] == 1).all():
                return rank

        # multiple processes
        else:
            # initialize array which will be compared to the rows of index_array:
            # elements with index 2n are the starts and 2n + 1 are the ends.

            neigh_inds = [None] * 6

            # in each direction find start/end index for neighbour
            for k, co in enumerate(comp):
                if co == 1:
                    neigh_inds[2 * k + 0] = self.index_array[rank, 2 * k + 0]
                    neigh_inds[2 * k + 1] = self.index_array[rank, 2 * k + 1]

                elif co == 0:
                    neigh_inds[2 * k + 1] = gl_s[k] - 1
                    if kinds[k]:
                        neigh_inds[2 * k + 1] %= ncells[k]

                elif co == 2:
                    neigh_inds[2 * k] = gl_e[k] + 1
                    if kinds[k]:
                        neigh_inds[2 * k] %= ncells[k]

                else:
                    raise ValueError(
                        "Wrong value for component; must be 0 or 1 or 2 !",
                    )

            neigh_inds = np.array(neigh_inds)

            # only use indices where information is present to find the neighbours rank
            inds = np.where(neigh_inds != None)

            # find ranks (row index of domain_array) which agree in start/end indices
            index_temp = np.squeeze(self.index_array[:, inds])
            unique_ranks = np.where(
                np.equal(index_temp, neigh_inds[inds]).all(1),
            )[0]

            # if any row satisfies condition, return its index (=rank of neighbour)
            if len(unique_ranks) != 0:
                neigh_id = unique_ranks[0]

        return neigh_id

    def _get_span_and_basis_for_eval_mpi(self, etas, Nspace, end):
        """Compute

        the knot span index,
        pn + 1 values of N-splines,
        pn values of D-splines,

        at each point in etas.

        Parameters
        ----------
        etas : np.array
            1d array of evaluation points (ascending).

        Nspace : SplineSpace
            Psydac object, must be a 1d N-spline space.

        end : int
            End coeff index on current process for N-spline space.

        Returns
        -------
        spans : np.array
            1d array of knot span indices.

        bn : np.array
            2d array of pn + 1 values of N-splines indexed by (eta, spline value).

        bd : np.array
            2d array of pn values of D-splines indexed by (eta, spline value).
        """

        from struphy.bsplines import bsplines_kernels

        # Extract knot vectors, degree and kind of basis
        Tn = Nspace.knots
        pn = Nspace.degree

        spans = np.zeros(etas.size, dtype=int)
        bns = np.zeros((etas.size, pn + 1), dtype=float)
        bds = np.zeros((etas.size, pn), dtype=float)
        bn = np.zeros(pn + 1, dtype=float)
        bd = np.zeros(pn, dtype=float)

        for n in range(etas.size):
            # avoid 1. --> 0. for clamped interpolation
            eta = etas[n] % (1.0 + 1e-14)
            span = bsplines_kernels.find_span(Tn, pn, eta)
            bsplines_kernels.b_d_splines_slim(Tn, pn, eta, span, bn, bd)
            # correct span for mpi spline eval
            if span > end + pn + 1:
                span -= Nspace.nbasis
            spans[n] = span
            bns[n] = bn
            bds[n] = bd

        return spans, bns, bds

    def get_quad_grids(
        self,
        space: TensorFemSpace | VectorFemSpace,
        nquads: tuple | list = None,
    ):
        """Return the 1d quadrature grids in each direction as a tuple."""
        assert self._nquads, "nquads has to be set with self._nquads = nquads"
        if nquads is None:
            nquads = self.nquads
        return tuple({q: gag} for q, gag in zip(nquads, space.get_assembly_grids(*nquads)))


class SplineFunction:
    """
    Initializes a callable spline function with a method for assigning initial conditions.

    Parameters
    ----------
    name : str
        Field's key to be used for instance when saving to hdf5 file.

    space_id : str
        Space identifier for the field ("H1", "Hcurl", "Hdiv", "L2" or "H1vec").

    derham : struphy.feec.psydac_derham.Derham
        Discrete Derham complex.

    coeffs : StencilVector | BlockVector
        The spline coefficients (optional).

    backgrounds : FieldsBackground | list
        For the initial condition.

    perturbations : Perturbation | list
        For the initial condition.

    domain : Domain
        Mapping for pullback/transform of initial condition.
        
    equil : FluidEquilibrium
        Fluid background used for inital condition.
    """

    def __init__(
        self,
        name: str,
        space_id: str,
        derham: Derham,
        coeffs: StencilVector | BlockVector = None,
        backgrounds: FieldsBackground | list = None,
        perturbations: Perturbation | list = None,
        domain: Domain = None,
        equil: FluidEquilibrium = None,
        verbose: bool = True,
    ):
        self._name = name
        self._space_id = space_id
        self._derham = derham
        self._backgrounds = backgrounds
        self._perturbations = perturbations
        self._domain = domain
        self._equil = equil

        # initialize field in memory (FEM space, vector and tensor product (stencil) vector)
        self._space_key = derham.space_to_form[space_id]
        self._space = derham.Vh[self._space_key]
        self._fem_space = derham.Vh_fem[self._space_key]
        assert isinstance(self.space, (StencilVectorSpace, BlockVectorSpace))
        assert isinstance(self.fem_space, (TensorFemSpace, VectorFemSpace))

        if coeffs is not None:
            assert coeffs.space == self.space
            self._vector = coeffs
        else:
            self._vector = derham.Vh_pol[self.space_key].zeros()

        self._vector_stencil = self.space.zeros()

        # transposed basis extraction operator for PolarVector --> Stencil-/BlockVector
        self._ET = derham.extraction_ops[self._space_key].transpose()

        # global indices of each process, and paddings
        if self._space_id in {"H1", "L2"}:
            self._gl_s = self.space.starts
            self._gl_e = self.space.ends
            self._pads = self.space.pads
        else:
            self._gl_s = [comp.starts for comp in self.space.spaces]
            self._gl_e = [comp.ends for comp in self.space.spaces]
            self._pads = [comp.pads for comp in self.space.spaces]

        # dimensions in each direction
        # self._nbasis = derham.nbasis[self._space_key]

        if self._space_id in {"H1", "L2"}:
            self._nbasis = tuple(
                [space.nbasis for space in self.fem_space.spaces],
            )
        else:
            self._nbasis = [tuple([space.nbasis for space in vec_space.spaces]) for vec_space in self.fem_space.spaces]

        if verbose and MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\nAllocated SplineFuntion '{self.name}' in space '{self.space_id}'.")

        if self.backgrounds is not None or self.perturbations is not None:
            self.initialize_coeffs(domain=self.domain, equil=self.equil)

    @property
    def name(self):
        """Name of the field in data container (string)."""
        return self._name

    @property
    def space_id(self):
        """String identifying the continuous space of the field: 'H1', 'Hcurl', 'Hdiv', 'L2' or 'H1vec'."""
        return self._space_id

    @property
    def space_key(self):
        """String identifying the discrete space of the field: '0', '1', '2', '3' or 'v'."""
        return self._space_key

    @property
    def derham(self):
        """3d Derham complex struphy.feec.psydac_derham.Derham."""
        return self._derham

    @property
    def domain(self):
        """Mapping for pullback/transform of initial condition."""
        return self._domain

    @property
    def equil(self):
        """Fluid equilibirum used for initial condition."""
        return self._equil

    @property
    def space(self):
        """Coefficient space (VectorSpace) of the field."""
        return self._space

    @property
    def fem_space(self):
        """FE space (FemSpace) of the field."""
        return self._fem_space

    @property
    def ET(self):
        """Transposed PolarExtractionOperator (or IdentityOperator) for mapping polar coeffs to polar tensor product rings."""
        return self._ET

    @property
    def vector(self):
        """psydac.linalg.stencil.StencilVector or psydac.linalg.block.BlockVector or struphy.polar.basic.PolarVector."""
        return self._vector

    @vector.setter
    def vector(self, value):
        """In-place setter for Stencil-/Block-/PolarVector."""

        if isinstance(self._vector, StencilVector):
            assert isinstance(value, (StencilVector, np.ndarray))

            s1, s2, s3 = self.starts
            e1, e2, e3 = self.ends

            self._vector[s1 : e1 + 1, s2 : e2 + 1, s3 : e3 + 1] = value[s1 : e1 + 1, s2 : e2 + 1, s3 : e3 + 1]

        elif isinstance(self._vector, BlockVector):
            assert isinstance(value, (BlockVector, list, tuple))

            for n in range(3):
                s1, s2, s3 = self.starts[n]
                e1, e2, e3 = self.ends[n]

                self._vector[n][s1 : e1 + 1, s2 : e2 + 1, s3 : e3 + 1] = value[n][s1 : e1 + 1, s2 : e2 + 1, s3 : e3 + 1]

        elif isinstance(self._vector, PolarVector):
            assert isinstance(value, (PolarVector, list, tuple))

            if isinstance(value, PolarVector):
                self._vector.set_vector(value)
            else:
                if isinstance(self._vector.tp, StencilVector):
                    assert isinstance(value[0], np.ndarray)
                    assert isinstance(
                        value[1],
                        (StencilVector, np.ndarray),
                    )

                    self._vector.pol[0][:] = value[0][:]

                    s1, s2, s3 = self.starts
                    e1, e2, e3 = self.ends

                    self._vector.tp[s1 : e1 + 1, s2 : e2 + 1, s3 : e3 + 1] = value[1][
                        s1 : e1 + 1, s2 : e2 + 1, s3 : e3 + 1
                    ]
                else:
                    for n in range(3):
                        assert isinstance(value[n][0], np.ndarray)
                        assert isinstance(
                            value[n][1],
                            (StencilVector, np.ndarray),
                        )

                        self._vector.pol[n][:] = value[n][0][:]

                        s1, s2, s3 = self.starts[n]
                        e1, e2, e3 = self.ends[n]

                        self._vector.tp[n][s1 : e1 + 1, s2 : e2 + 1, s3 : e3 + 1] = value[n][1][
                            s1 : e1 + 1, s2 : e2 + 1, s3 : e3 + 1
                        ]

        self._vector.update_ghost_regions()

    @property
    def starts(self):
        """Global indices of the first FE coefficient on the process, in each direction."""
        return self._gl_s

    @property
    def ends(self):
        """Global indices of the last FE coefficient on the process, in each direction."""
        return self._gl_e

    @property
    def pads(self):
        """Paddings for ghost regions, in each direction."""
        return self._pads

    @property
    def nbasis(self):
        """Tuple(s) of 1d dimensions for each direction."""
        return self._nbasis

    @property
    def vector_stencil(self):
        """Tensor-product Stencil-/BlockVector corresponding to a copy of self.vector in case of Stencil-/Blockvector

        OR

        the extracted coefficients in case of PolarVector. Call self.extract_coeffs() beforehand.
        """
        return self._vector_stencil

    @property
    def backgrounds(self) -> FieldsBackground | list:
        """For the initial condition."""
        return self._backgrounds

    @property
    def perturbations(self) -> Perturbation | list:
        """For the initial condition."""
        return self._perturbations

    ###############
    ### Methods ###
    ###############
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

    def initialize_coeffs(
        self,
        *,
        backgrounds: FieldsBackground | list = None,
        perturbations: Perturbation | list = None,
        domain: Domain = None,
        equil: FluidEquilibrium = None,
    ):
        """
        Set the initial conditions for self.vector.
        """

        # set background paramters
        if backgrounds is not None:
            # if self.backgrounds is not None:
            #     print(f"Attention: overwriting backgrounds for {self.name}")
            self._backgrounds = backgrounds

        # set perturbation paramters
        if perturbations is not None:
            # if self.perturbations is not None:
            #     print(f"Attention: overwriting perturbation parameters for {self.name}")
            self._perturbations = perturbations

        # set domain
        if domain is not None:
            # if self.domain is not None:
            #     print(f"Attention: overwriting domain for {self.name}")
            self._domain = domain

        if isinstance(self.backgrounds, FieldsBackground):
            self._backgrounds = [self.backgrounds]

        if isinstance(self.perturbations, Perturbation):
            self._perturbations = [self.perturbations]

        # start from zero coeffs
        self._vector *= 0.0

        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"Initializing {self.name} ...")

        # add backgrounds to initial vector
        if self.backgrounds is not None:
            for fb in self.backgrounds:
                assert isinstance(fb, FieldsBackground)
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print(f"Adding background {fb} ...")

                # special case of const
                if fb.type == "LogicalConst":
                    vals = fb.values
                    assert isinstance(vals, (list, tuple))

                    if self.space_id in {"H1", "L2"}:

                        def f_tmp(e1, e2, e3):
                            return vals[0] + 0.0 * e1

                        fun = f_tmp
                    else:
                        assert len(vals) == 3
                        fun = []

                        if vals[0] is not None:
                            fun += [lambda e1, e2, e3: vals[0] + 0.0 * e1]
                        else:
                            fun += [lambda e1, e2, e3: 0.0 * e1]

                        if vals[1] is not None:
                            fun += [lambda e1, e2, e3: vals[1] + 0.0 * e1]
                        else:
                            fun += [lambda e1, e2, e3: 0.0 * e1]

                        if vals[2] is not None:
                            fun += [lambda e1, e2, e3: vals[2] + 0.0 * e1]
                        else:
                            fun += [lambda e1, e2, e3: 0.0 * e1]
                else:
                    assert equil is not None
                    var = fb.variable
                    assert var in dir(MHDequilibrium), f"{var = } is not an attribute of any fields background."

                    if self.space_id in {"H1", "L2"}:
                        fun = getattr(equil, var)
                    else:
                        assert (var + "_1") in dir(MHDequilibrium), (
                            f"{(var + '_1') = } is not an attribute of any fields background."
                        )
                        fun = [
                            getattr(equil, var + "_1"),
                            getattr(equil, var + "_2"),
                            getattr(equil, var + "_3"),
                        ]

                # perform projection
                self.vector += self.derham.P[self.space_key](fun)

        # add perturbations to coefficient vector
        if self.perturbations is not None:
            for ptb in self.perturbations:
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print(f"Adding perturbation {ptb} ...")

                # special case of white noise in logical space for different components
                if isinstance(ptb, Noise):
                    # set white noise FE coefficients
                    self._add_noise(
                        direction=ptb.direction,
                        amp=ptb.amp,
                        seed=ptb.seed,
                        n=ptb.comp,
                    )
                # perturbation class
                elif isinstance(ptb, Perturbation):
                    if self.space_id in {"H1", "L2"}:
                        fun = TransformedPformComponent(
                            ptb,
                            ptb.given_in_basis,
                            self.space_key,
                            domain=domain,
                        )
                    elif self.space_id in {"Hcurl", "Hdiv", "H1vec"}:
                        fun_vec = [None] * 3
                        fun_vec[ptb.comp] = ptb

                        # pullback callable for each component
                        fun = []
                        for comp in range(3):
                            fun += [
                                TransformedPformComponent(
                                    fun_vec,
                                    ptb.given_in_basis,
                                    self.space_key,
                                    comp=comp,
                                    domain=domain,
                                ),
                            ]

                    # peform projection
                    self.vector += self.derham.P[self.space_key](fun)

                # TODO: re-add Eigfun and InitFromOutput in new framework

                # loading of MHD eigenfunction (legacy code, might not be up to date)
                # elif "EigFun" in _type:
                #     print("Warning: Eigfun is not regularly tested ...")
                #     from struphy.initial import eigenfunctions

                #     # select class
                #     funs = getattr(eigenfunctions, _type)(
                #         self.derham,
                #         **_params,
                #     )

                #     # select eigenvector and set coefficients
                #     if hasattr(funs, self.name):
                #         eig_vec = getattr(funs, self.name)

                #         self.vector += eig_vec

                # # initialize from existing output file
                # elif "InitFromOutput" in _type:
                #     # select class
                #     o_data = getattr(utilities, _type)(
                #         self.derham,
                #         self.name,
                #         species,
                #         **_params,
                #     )

                #     if isinstance(self.vector, StencilVector):
                #         self.vector._data[:] += o_data.vector

                #     else:
                #         for n in range(3):
                #             self.vector[n]._data[:] += o_data.vector[n]

        # apply boundary operator (in-place)
        self.derham.boundary_ops[self.space_key].dot(
            self._vector.copy(),
            out=self._vector,
        )

        # update ghost regions
        self._vector.update_ghost_regions()

    def initialize_coeffs_from_restart_file(self, file, species=None):
        """
        TODO
        """

        if species is None:
            key = "restart/" + self.name
        else:
            key = "restart/" + species + "_" + self.name

        if isinstance(self.vector, StencilVector):
            self.vector._data[:] = file[key][-1]
        else:
            for n in range(3):
                self.vector[n]._data[:] = file[key + "/" + str(n + 1)][-1]

        self._vector.update_ghost_regions()

    def eval_tp_fixed_loc(self, spans, bases, out=None):
        """Spline evaluation on pre-defined grid.

        Input spans must be on local process, start <= span <= end.

        Parameters
        ----------
        spans : 3-tuple of 1d int arrays
            Knot span indices in each direction (start <= span <= end).

        bases : 3-tuple of 2d float arrays
            Values of non-zero eta basis functions at evaluation points indexed by (eta, basis function).

        Returns
        -------
        out : array[float]
            3d array of spline values S_ijk corresponding to the sizes of spans.
        """

        if isinstance(self.vector, PolarVector):
            vec = self.vector.tp
        else:
            vec = self.vector

        if isinstance(vec, StencilVector):
            assert [span.size for span in spans] == [base.shape[0] for base in bases]

            if out is None:
                out = np.empty([span.size for span in spans], dtype=float)
            else:
                assert out.shape == tuple([span.size for span in spans])

            eval_spline_mpi_tensor_product_fixed(
                *spans,
                *bases,
                vec._data,
                self.derham.spline_types_pyccel[self.space_key],
                np.array(self.derham.p),
                np.array(self.starts),
                out,
            )

        else:
            out_is_none = False
            if out is None:
                out = []
                out_is_none = True

            for i in range(3):
                assert [span.size for span in spans] == [base.shape[0] for base in bases[i]]

                if out_is_none:
                    out += np.empty(
                        [span.size for span in spans],
                        dtype=float,
                    )
                else:
                    assert out[i].shape == tuple(
                        [span.size for span in spans],
                    )

                eval_spline_mpi_tensor_product_fixed(
                    *spans,
                    *bases[i],
                    vec[i]._data,
                    self.derham.spline_types_pyccel[self.space_key][i],
                    np.array(
                        self.derham.p,
                    ),
                    np.array(
                        self.starts[i],
                    ),
                    out[i],
                )

        return out

    def __call__(self, *etas, out=None, tmp=None, squeeze_out=False, local=False):
        """
        Evaluates the spline function on the global domain, unless local=True,
        in which case the spline function is evaluated only on the local domain,
        and the rest is set to zero.

        Parameters
        ----------
        *etas : array-like | tuple
        Logical coordinates at which to evaluate. Two cases are possible:

            1. 2d numpy array, where coordinates are taken from eta1 = etas[:, 0], eta2 = etas[:, 1], etc. (like markers).
            2. list/tuple (eta1, eta2, ...), where eta1, eta2, ... can be float or array-like of various shapes.

        out : array[float] or list
            Array in which to store the values of the spline function at the given point set (list in case of vector-valued spaces).

        tmp : array[float]
            Array that has shape the size of the grid that will be used as a temporary for AllReduce, to avoid creating it a each call.

        flat_eval : bool
            Whether to do a flat evaluation, i.e. f([e11, e12], [e21, e22]) = [f(e11, e21), f(e12, e22)].

        squeeze_out : bool
            Whether to remove singleton dimensions in output "values".

        Returns
        -------
            out : array[float] or list
                The values of the spline function at the given point set (list in case of vector-valued spaces).
        """

        # extract coefficients and update ghost regions
        self.extract_coeffs(update_ghost_regions=True)

        # get knot vectors
        T1, T2, T3 = self.derham.Vh_fem["0"].knots

        # marker evaluation
        if len(etas) == 1:
            marker_evaluation = True
            is_sparse_meshgrid = False
            markers = etas[0]
            assert markers.ndim == 2
            self._flag_pts_not_on_proc(markers)
            tmp_shape = markers.shape[0]
        # 3D meshgrid evaluation
        else:
            marker_evaluation = False
            E1, E2, E3, is_sparse_meshgrid = Domain.prepare_eval_pts(*etas)
            self._flag_pts_not_on_proc(E1, E2, E3)
            tmp_shape = (
                E1.shape[0],
                E2.shape[1],
                E3.shape[2],
            )

        # prepare arrays for AllReduce
        if tmp is None:
            tmp = np.zeros(
                tmp_shape,
                dtype=float,
            )
        else:
            assert isinstance(tmp, np.ndarray)
            assert tmp.shape == tmp_shape
            assert tmp.dtype.type is np.float64
            tmp[:] = 0.0

        # scalar-valued field
        if isinstance(self._vector_stencil, StencilVector):
            kind = self.derham.spline_types_pyccel[self.space_key]

            if is_sparse_meshgrid:
                # eval_mpi needs flagged arrays E1, E2, E3 as input
                eval_3d.eval_spline_mpi_sparse_meshgrid(
                    E1,
                    E2,
                    E3,
                    self._vector_stencil._data,
                    kind,
                    np.array(self.derham.p),
                    T1,
                    T2,
                    T3,
                    np.array(self.starts),
                    tmp,
                )
            elif marker_evaluation:
                # eval_mpi needs flagged arrays E1, E2, E3 as input
                eval_3d.eval_spline_mpi_markers(
                    markers,
                    self._vector_stencil._data,
                    kind,
                    np.array(self.derham.p),
                    T1,
                    T2,
                    T3,
                    np.array(self.starts),
                    tmp,
                )
            else:
                # eval_mpi needs flagged arrays E1, E2, E3 as input
                eval_3d.eval_spline_mpi_matrix(
                    E1,
                    E2,
                    E3,
                    self._vector_stencil._data,
                    kind,
                    np.array(self.derham.p),
                    T1,
                    T2,
                    T3,
                    np.array(self.starts),
                    tmp,
                )

            if self.derham.comm is not None:
                if local == False:
                    self.derham.comm.Allreduce(
                        MPI.IN_PLACE,
                        tmp,
                        op=MPI.SUM,
                    )

            # all processes have all values
            if out is None:
                out = tmp
            else:
                out *= 0.0
                out += tmp

            if squeeze_out:
                out = np.squeeze(out)

            if out.ndim == 0:
                out = out.item()

        # vector-valued field
        else:
            out_is_None = out is None
            if out_is_None:
                out = []
            for n, kind in enumerate(self.derham.spline_types_pyccel[self.space_key]):
                if is_sparse_meshgrid:
                    # eval_mpi needs flagged arrays E1, E2, E3 as input
                    eval_3d.eval_spline_mpi_sparse_meshgrid(
                        E1,
                        E2,
                        E3,
                        self._vector_stencil[n]._data,
                        kind,
                        np.array(self.derham.p),
                        T1,
                        T2,
                        T3,
                        np.array(self.starts[n]),
                        tmp,
                    )
                elif marker_evaluation:
                    # eval_mpi needs flagged arrays E1, E2, E3 as input
                    eval_3d.eval_spline_mpi_markers(
                        markers,
                        self._vector_stencil[n]._data,
                        kind,
                        np.array(self.derham.p),
                        T1,
                        T2,
                        T3,
                        np.array(self.starts[n]),
                        tmp,
                    )
                else:
                    # eval_mpi needs flagged arrays E1, E2, E3 as input
                    eval_3d.eval_spline_mpi_matrix(
                        E1,
                        E2,
                        E3,
                        self._vector_stencil[n]._data,
                        kind,
                        np.array(self.derham.p),
                        T1,
                        T2,
                        T3,
                        np.array(self.starts[n]),
                        tmp,
                    )

                if self.derham.comm is not None:
                    if local == False:
                        self.derham.comm.Allreduce(
                            MPI.IN_PLACE,
                            tmp,
                            op=MPI.SUM,
                        )

                # all processes have all values
                if out_is_None:
                    out += [tmp.copy()]
                else:
                    out[n] *= 0.0
                    out[n] += tmp

                tmp[:] = 0.0

                if squeeze_out:
                    out[-1] = np.squeeze(out[-1])

                if out[-1].ndim == 0:
                    out[-1] = out[-1].item()

        return out

    #######################
    ### Private methods ###
    #######################
    def _flag_pts_not_on_proc(self, *etas):
        """Sets evaluation points outside of process domain to -1 (in place).

        Parameters
        ----------
        *etas : array-like | tuple
        Logical coordinates at which to evaluate. Two cases are possible:

            1. 2d numpy array, where coordinates are taken from eta1 = etas[:, 0], eta2 = etas[:, 1], etc. (like markers).
            2. list/tuple (eta1, eta2, ...), where eta1, eta2, ... can be float or array-like of various shapes."""

        # get domain decompoistion info
        dom_arr = self.derham.domain_array
        if self.derham.comm is not None:
            rank = self.derham.comm.Get_rank()
        else:
            rank = 0

        # marker evaluation
        if len(etas) == 1:
            markers = etas[0]

            # check which particles are on the current process domain
            is_on_proc_domain = np.logical_and(
                markers[:, :3] >= dom_arr[rank, 0::3],
                markers[:, :3] <= dom_arr[rank, 1::3],
            )
            on_proc = np.all(is_on_proc_domain, axis=1)

            markers[~on_proc, :] = -1.0

        # 3D meshgrid evaluation
        else:
            assert len(etas) == 3
            E1, E2, E3 = etas
            # check if eval points are "interior points" in domain_array; if so, add small offset

            if dom_arr[rank, 0] != 0.0:
                E1[E1 == dom_arr[rank, 0]] += 1e-8
            if dom_arr[rank, 1] != 1.0:
                E1[E1 == dom_arr[rank, 1]] += 1e-8

            if dom_arr[rank, 3] != 0.0:
                E2[E2 == dom_arr[rank, 3]] += 1e-8
            if dom_arr[rank, 4] != 1.0:
                E2[E2 == dom_arr[rank, 4]] += 1e-8

            if dom_arr[rank, 6] != 0.0:
                E3[E3 == dom_arr[rank, 6]] += 1e-8
            if dom_arr[rank, 7] != 1.0:
                E3[E3 == dom_arr[rank, 7]] += 1e-8

            # True for eval points on current process
            E1_on_proc = np.logical_and(
                E1 >= dom_arr[rank, 0],
                E1 <= dom_arr[rank, 1],
            )
            E2_on_proc = np.logical_and(
                E2 >= dom_arr[rank, 3],
                E2 <= dom_arr[rank, 4],
            )
            E3_on_proc = np.logical_and(
                E3 >= dom_arr[rank, 6],
                E3 <= dom_arr[rank, 7],
            )

            # flag eval points not on current process
            E1[~E1_on_proc] = -1.0
            E2[~E2_on_proc] = -1.0
            E3[~E3_on_proc] = -1.0

    def _add_noise(
        self,
        direction: NoiseDirections = "e3",
        amp: float = 0.0001,
        seed: int = None,
        n: int = None,
    ):
        """Add noise to a vector component where init_comps==True, otherwise leave at zero.

        Parameters
        ----------
        direction: str
            The direction(s) of variation of the noise: 'e1', 'e2', 'e3', 'e1e2', etc.

        amp: float
            Noise amplitude.

        seed: int
            Seed for the random number generator.

        n : int
            Vector component (0, 1 or 2) to be initialized.
        """

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
            _shape = (
                self._gl_e[0] + 1 - self._gl_s[0],
                self._gl_e[1] + 1 - self._gl_s[1],
                self._gl_e[2] + 1 - self._gl_s[2],
            )
        else:
            _shape = (
                self._gl_e[n][0] + 1 - self._gl_s[n][0],
                self._gl_e[n][1] + 1 - self._gl_s[n][1],
                self._gl_e[n][2] + 1 - self._gl_s[n][2],
            )

        if direction == "e1":
            _amps = self._tmp_noise_for_mpi(
                _shape[0],
                direction=direction,
                amp=amp,
                seed=seed,
            )
            for j in range(_shape[1]):
                for k in range(_shape[2]):
                    vec[sli[0], gl_s[1] + j, gl_s[2] + k] += _amps
            del _amps

        elif direction == "e2":
            _amps = self._tmp_noise_for_mpi(
                _shape[1],
                direction=direction,
                amp=amp,
                seed=seed,
            )
            for j in range(_shape[0]):
                for k in range(_shape[2]):
                    vec[gl_s[0] + j, sli[1], gl_s[2] + k] += _amps

        elif direction == "e3":
            _amps = self._tmp_noise_for_mpi(
                _shape[2],
                direction=direction,
                amp=amp,
                seed=seed,
            )
            for j in range(_shape[0]):
                for k in range(_shape[1]):
                    vec[gl_s[0] + j, gl_s[1] + k, sli[2]] += _amps

        elif direction == "e1e2":
            _amps = self._tmp_noise_for_mpi(
                _shape[0],
                _shape[1],
                direction=direction,
                amp=amp,
                seed=seed,
            )
            for j in range(_shape[2]):
                vec[sli[0], sli[1], gl_s[2] + j] += _amps

        elif direction == "e1e3":
            _amps = self._tmp_noise_for_mpi(
                _shape[0],
                _shape[2],
                direction=direction,
                amp=amp,
                seed=seed,
            )
            for j in range(_shape[1]):
                vec[sli[0], gl_s[1] + j, sli[2]] += _amps

        elif direction == "e2e3":
            _amps = self._tmp_noise_for_mpi(
                _shape[1],
                _shape[2],
                direction=direction,
                amp=amp,
                seed=seed,
            )
            for j in range(_shape[0]):
                vec[gl_s[0] + j, sli[1], sli[2]] += _amps

        elif direction == "e1e2e3":
            _amps = self._tmp_noise_for_mpi(
                _shape[0],
                _shape[1],
                _shape[2],
                direction=direction,
                amp=amp,
                seed=seed,
            )
            vec[sli[0], sli[1], sli[2]] += _amps

        else:
            raise ValueError("Invalid direction for noise.")

    def _tmp_noise_for_mpi(self, *shapes, direction="e3", amp=0.0001, seed=None):
        """Initialize same FEEC noise regardless of number of MPI processes.

        Parameters
        ----------
        shapes : int
            Length of local array size in each direction where noise is to be initialized.

        direction : str
            Noise direction ('e1', 'e2' or 'e3'). Multi-dim. not yet correct.

        amp : float
            Noise amplitude

        seed : int
            Seed for random number generator.

        Returns
        -------
        _amps : np.array
            The noisy FE coefficients in the desired direction (1d, 2d or 3d array)."""

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
        already_drawn = np.zeros(nprocs) == 1.0

        # 1d mid point arrays in each direction
        mid_points = []
        for npr in nprocs:
            delta = 1.0 / npr
            mid_points_i = np.zeros(npr)
            for n in range(npr):
                mid_points_i[n] = delta * (n + 1 / 2)
            mid_points += [mid_points_i]

        if direction == "e1":
            tmp_arrays = np.zeros(nprocs[0]).tolist()
        elif direction == "e2":
            tmp_arrays = np.zeros(nprocs[1]).tolist()
        elif direction == "e3":
            tmp_arrays = np.zeros(nprocs[2]).tolist()
        elif direction == "e1e2":
            tmp_arrays = np.zeros((nprocs[0], nprocs[1])).tolist()
            Warning, f"2d noise in the directions {direction} is not correctly initilaized for MPI !!"
        elif direction == "e1e3":
            tmp_arrays = np.zeros((nprocs[0], nprocs[2])).tolist()
            Warning, f"2d noise in the directions {direction} is not correctly initilaized for MPI !!"
        elif direction == "e2e3":
            tmp_arrays = np.zeros((nprocs[1], nprocs[2])).tolist()
            Warning, f"2d noise in the directions {direction} is not correctly initilaized for MPI !!"
        elif direction == "e1e2e3":
            tmp_arrays = np.zeros((nprocs[0], nprocs[1], nprocs[2])).tolist()
            Warning, f"3d noise in the directions {direction} is not correctly initilaized for MPI !!"
        else:
            raise ValueError("Invalid direction for tmp_arrays.")

        # 3d index of current process from mid points
        inds_current = []
        for n in range(3):
            mid_pt_current = (domain_array[rank, 3 * n] + domain_array[rank, 3 * n + 1]) / 2.0
            inds_current += [np.argmin(np.abs(mid_points[n] - mid_pt_current))]

        # loop over processes
        for i in range(comm_size):
            # 3d index of process i from mid points
            inds = []
            for n in range(3):
                mid_pt = (domain_array[i, 3 * n] + domain_array[i, 3 * n + 1]) / 2.0
                inds += [np.argmin(np.abs(mid_points[n] - mid_pt))]

            if already_drawn[inds[0], inds[1], inds[2]]:
                if direction == "e1":
                    _amps[:] = tmp_arrays[inds[0]]
                elif direction == "e2":
                    _amps[:] = tmp_arrays[inds[1]]
                elif direction == "e3":
                    _amps[:] = tmp_arrays[inds[2]]
                elif direction == "e1e2":
                    _amps[:] = tmp_arrays[inds[0]][inds[1]]
                elif direction == "e1e3":
                    _amps[:] = tmp_arrays[inds[0]][inds[2]]
                elif direction == "e2e3":
                    _amps[:] = tmp_arrays[inds[1]][inds[2]]
                elif direction == "e1e2e3":
                    _amps[:] = tmp_arrays[inds[0]][inds[1]][inds[2]]

            else:
                if direction == "e1":
                    tmp_arrays[inds[0]] = (
                        (
                            np.random.rand(
                                *shapes,
                            )
                            - 0.5
                        )
                        * 2.0
                        * amp
                    )
                    already_drawn[inds[0], :, :] = True
                    _amps[:] = tmp_arrays[inds[0]]
                elif direction == "e2":
                    tmp_arrays[inds[1]] = (
                        (
                            np.random.rand(
                                *shapes,
                            )
                            - 0.5
                        )
                        * 2.0
                        * amp
                    )
                    already_drawn[:, inds[1], :] = True
                    _amps[:] = tmp_arrays[inds[1]]
                elif direction == "e3":
                    tmp_arrays[inds[2]] = (
                        (
                            np.random.rand(
                                *shapes,
                            )
                            - 0.5
                        )
                        * 2.0
                        * amp
                    )
                    already_drawn[:, :, inds[2]] = True
                    _amps[:] = tmp_arrays[inds[2]]
                elif direction == "e1e2":
                    tmp_arrays[inds[0]][inds[1]] = (np.random.rand(*shapes) - 0.5) * 2.0 * amp
                    already_drawn[inds[0], inds[1], :] = True
                    _amps[:] = tmp_arrays[inds[0]][inds[1]]
                elif direction == "e1e3":
                    tmp_arrays[inds[0]][inds[2]] = (np.random.rand(*shapes) - 0.5) * 2.0 * amp
                    already_drawn[inds[0], :, inds[2]] = True
                    _amps[:] = tmp_arrays[inds[0]][inds[2]]
                elif direction == "e2e3":
                    tmp_arrays[inds[1]][inds[2]] = (np.random.rand(*shapes) - 0.5) * 2.0 * amp
                    already_drawn[:, inds[1], inds[2]] = True
                    _amps[:] = tmp_arrays[inds[1]][inds[2]]
                elif direction == "e1e2e3":
                    tmp_arrays[inds[0]][inds[1]][inds[2]] = (np.random.rand(*shapes) - 0.5) * 2.0 * amp
                    already_drawn[inds[0], inds[1], inds[2]] = True
                    _amps[:] = tmp_arrays[inds[0]][inds[1]][inds[2]]

            if np.all(np.array([ind_c == ind for ind_c, ind in zip(inds_current, inds)])):
                return _amps


class DiscreteDerham:
    """A discrete de Rham sequence in 3D.

    Parameters
    ----------
    *spaces : list of TensorFemSpace | VectorFemSpace
        The discrete spaces of the de Rham sequence.
    """

    def __init__(self, *spaces):
        assert len(spaces) == 4
        assert all(isinstance(space, (TensorFemSpace, VectorFemSpace)) for space in spaces)

        self._spaces = spaces
        self._dim = 3

        D0 = Gradient_3D(spaces[0], spaces[1])
        D1 = Curl_3D(spaces[1], spaces[2])
        D2 = Divergence_3D(spaces[2], spaces[3])

        spaces[0].diff = spaces[0].grad = D0
        spaces[1].diff = spaces[1].curl = D1
        spaces[2].diff = spaces[2].div = D2

    # --------------------------------------------------------------------------
    @property
    def dim(self):
        """Dimension of the physical and logical domains, which are assumed to be the same."""
        return self._dim

    @property
    def V0(self):
        """First space of the de Rham sequence : H1 space"""
        return self._spaces[0]

    @property
    def V1(self):
        """Second space of the de Rham sequence :
        - 1d : L2 space
        - 2d : either Hdiv or Hcurl space
        - 3d : Hcurl space"""
        return self._spaces[1]

    @property
    def V2(self):
        """Third space of the de Rham sequence :
        - 2d : L2 space
        - 3d : Hdiv space"""
        return self._spaces[2]

    @property
    def V3(self):
        """Fourth space of the de Rham sequence : L2 space in 3d"""
        return self._spaces[3]

    @property
    def spaces(self):
        """Spaces of the proper de Rham sequence (excluding Hvec)."""
        return self._spaces

    @property
    def derivatives_as_matrices(self):
        """Differential operators of the De Rham sequence as LinearOperator objects."""
        return tuple(V.diff.matrix for V in self.spaces[:-1])

    @property
    def derivatives(self):
        """Differential operators of the De Rham sequence as `DiffOperator` objects.

        Those are objects with `domain` and `codomain` properties that are `FemSpace`,
        they act on `FemField` (they take a `FemField` of their `domain` as input and return
        a `FemField` of their `codomain`.
        """
        return tuple(V.diff for V in self.spaces[:-1])

    # --------------------------------------------------------------------------
    def projectors(self, *, kind="global", nquads=None):
        """Projectors mapping callable functions of the physical coordinates to a
        corresponding `FemField` object in the De Rham sequence.

        Parameters
        ----------
        kind : str
            Type of the projection : at the moment, only global is accepted and
            returns geometric commuting projectors based on interpolation/histopolation
            for the De Rham sequence (GlobalProjector objects).

        nquads : list(int) | tuple(int)
            Number of quadrature points along each direction, to be used in Gauss
            quadrature rule for computing the (approximated) degrees of freedom.

        Returns
        -------
        P0, ..., Pn : callables
            Projectors that can be called on any callable function that maps
            from the physical space to R (scalar case) or R^d (vector case) and
            returns a FemField belonging to the i-th space of the De Rham sequence
        """

        if not (kind == "global"):
            raise NotImplementedError("only global projectors are available")

        if nquads is None:
            nquads = [p + 1 for p in self.V0.degree]
        elif isinstance(nquads, int):
            nquads = [nquads] * self.dim
        else:
            assert hasattr(nquads, "__iter__")
            nquads = list(nquads)

        assert all(isinstance(nq, int) for nq in nquads)
        assert all(nq >= 1 for nq in nquads)

        P0 = Projector_H1(self.V0)
        P1 = Projector_Hcurl(self.V1, nquads)
        P2 = Projector_Hdiv(self.V2, nquads)
        P3 = Projector_L2(self.V3, nquads)

        return P0, P1, P2, P3


def transform_perturbation(
    pert_type: str,
    pert_params: dict,
    space_key: str,
    domain: Domain,
):
    """Creates callabe(s) from perturbation parameters.

    Parameters
    ----------
    pert_type: str
        Class name of the perturbation, see :mod:`~struphy.initial.perturbations`.

    pert_params: dict
        Parameters of the perturbation.

    space_key: str
        The p-form representation of the output: '0', '1', '2' '3' or 'v'.

    domain: Domain
        Domain object (mapping).

    Returns
    -------
    fun: list
        A callable or list of callables in the space defined by ``space_key``.
    """

    if space_key in {"0", "3"}:
        # which transform is to be used: physical, '0' or '3'
        fun_basis = pert_params["given_in_basis"]
        pert_params.pop("given_in_basis")

        # get callable(s) for specified init type
        fun_class = getattr(perturbations, pert_type)
        fun_tmp = [fun_class(**pert_params)]

        # pullback callable
        fun = TransformedPformComponent(
            fun_tmp,
            fun_basis,
            space_key,
            domain=domain,
        )
    elif space_key in {"1", "2", "v"}:
        fun_class = getattr(perturbations, pert_type)
        fun_tmp = []
        fun_basis = []
        bases = pert_params["given_in_basis"]
        pert_params.pop("given_in_basis")
        for component, base in enumerate(bases):
            if base is None:
                # Look ahead to find the next non-None base, assuming len of bases is 3
                next_base = None
                if bases[0] is not None:
                    next_base = bases[0]
                elif bases[1] is not None:
                    next_base = bases[1]
                elif bases[2] is not None:
                    next_base = bases[2]
                # If no non-None base found later, default to "physical"
                fun_basis += [next_base if next_base is not None else "physical"]
                fun_tmp += [None]
            else:
                # which transform is to be used: physical, '1', '2' or 'v'
                fun_basis += [base]
                # function parameters of component
                _params_comp = {}
                for key, val in pert_params.items():
                    if isinstance(val, (list, tuple)):
                        _params_comp[key] = val[component]
                    else:
                        _params_comp[key] = val
                fun_tmp += [fun_class(**_params_comp)]
        # pullback callable
        fun = []
        for n, fform in enumerate(fun_basis):
            fun += [
                TransformedPformComponent(
                    fun_tmp,
                    fform,
                    space_key,
                    comp=n,
                    domain=domain,
                ),
            ]
    return fun


def get_pts_and_wts(space_1d, start, end, n_quad=None, polar_shift=False):
    """Obtain local (to MPI process) projection point sets and weights in one grid direction.

    Parameters
    ----------
    space_1d : SplineSpace
        Psydac object for uni-variate spline space.

    start : int
        Start index on current process.

    end : int
        End index on current process.

    n_quad : int
        Number of quadrature points for Gauss-Legendre histopolation.
        If None, is set to p + 1 where p is the space_1d degree (products of basis functions are integrated exactly).

    polar_shift : bool
        Whether to shift the first interpolation point away from 0.0 by 1e-5 (needed only in eta_1 and for polar domains).

    Returns
    -------
    pts : 2D float array
        Quadrature points (or Greville points for interpolation) in format (ii, iq) = (interval, quadrature point).

    wts : 2D float array
        Quadrature weights (or 1's for interpolation) in format (ii, iq) = (interval, quadrature point).

    subs : 1D int array
        One entry for each interval ii; usually has value 0.
        A value of 1 indicates that the cell ii is the second subinterval of a split Greville cell (for histopolation with even degree)."""

    greville_loc = space_1d.greville[start : end + 1].copy()
    histopol_loc = space_1d.histopolation_grid[start : end + 2].copy()

    # make sure that greville points used for interpolation are in [0, 1]
    assert np.all(np.logical_and(greville_loc >= 0.0, greville_loc <= 1.0))

    # interpolation
    if space_1d.basis == "B":
        x_grid = greville_loc
        pts = greville_loc[:, None]
        wts = np.ones(pts.shape, dtype=float)

        # sub-interval index is always 0 for interpolation.
        subs = np.zeros(pts.shape[0], dtype=int)

        # !! shift away first interpolation point in eta_1 direction for polar domains !!
        if pts[0] == 0.0 and polar_shift:
            pts[0] += 0.00001

    # histopolation
    elif space_1d.basis == "M":
        if space_1d.degree % 2 == 0:
            union_breaks = space_1d.breaks
        else:
            union_breaks = space_1d.breaks[:-1]

        # Make union of Greville and break points
        tmp = set(np.round(space_1d.histopolation_grid, decimals=14)).union(
            np.round(union_breaks, decimals=14),
        )

        tmp = list(tmp)
        tmp.sort()
        tmp_a = np.array(tmp)

        x_grid = tmp_a[
            np.logical_and(
                tmp_a
                >= np.min(
                    histopol_loc,
                )
                - 1e-14,
                tmp_a <= np.max(histopol_loc) + 1e-14,
            )
        ]

        # determine subinterval index (= 0 or 1):
        subs = np.zeros(x_grid[:-1].size, dtype=int)
        for n, x_h in enumerate(x_grid[:-1]):
            add = 1
            for x_g in histopol_loc:
                if abs(x_h - x_g) < 1e-14:
                    add = 0
            subs[n] += add

        # Gauss - Legendre quadrature points and weights
        if n_quad is None:
            # products of basis functions are integrated exactly
            n_quad = space_1d.degree + 1

        pts_loc, wts_loc = np.polynomial.legendre.leggauss(n_quad)

        x, wts = bsp.quadrature_grid(x_grid, pts_loc, wts_loc)

        pts = x % 1.0

    return pts, wts, subs


def get_pts_and_wts_quasi(
    space_1d: SplineSpace,
    *,
    bulk_indices_i: tuple = None,
    mu_nu_values: list = None,
    polar_shift: bool = False,
):
    r"""Obtain local projection point sets and weights in one grid direction for the quasi-interpolation method.
    The quasi-interpolation points are :math:`\nu - \mu +p` equidistant points :math:`\{ x^i_j \}_{0 \leq j < \nu - \mu +p}` in the sub-interval :math:`Q = [\eta_\mu , \eta_\nu]` given by:

    \begin{itemize}
        \item Clamped: 
        .. math:: 
            Q = \left\{\begin{array}{lr}
            [\eta_p = 0, \eta_{p+1}], & i = 0 \,,\\
            {[\eta_p = 0, \eta_{p+i}]}, & 0 < i < p-1\,,\\
            {[\eta_{i+1}, \eta_{i+p}]}, & p-1 \leq i \leq \hat{n}_N - p\,,\\
            {[\eta_{i+1}, \eta_{\hat{n}_N} = 1]}, &  \hat{n}_N - p < i < \hat{n}_N -1\,,\\
            {[\eta_{\hat{n}_N -1}, \eta_{\hat{n}_N} = 1]}, & i = \hat{n}_N -1 \,.
            \end{array} \; \right .
        \item Periodic: 
        .. math::
            Q = [\eta_{i + 1}, \eta_{i + p}] \:\:\:\:\: \forall \:\: i.
    \end{itemize}

    Which are allways a subset of  :math:`\{-(p-1)h,-(p-1)h + \frac{h}{2}, ..., 1-h - \frac{h}{2},1-h \}` for the periodic case.

    Parameters
    ----------
    space_1d : SplineSpace
        Psydac object for uni-variate spline space.

    polar_shift : bool
        Whether to shift the first interpolation point away from 0.0 by 1e-5 (needed only in eta_1 and for polar domains).

    Returns
    -------
    pts : 2D float array
        Quadrature points (or quasi-interpolation points for interpolation) in format (ii, iq) = (interval, quadrature point).

    wts : 2D float array
        Quadrature weights (or 1's for interpolation) in format (ii, iq) = (interval, quadrature point)."""

    # spline space info
    p = space_1d.degree
    h = space_1d.breaks[1]
    N = len(space_1d.breaks) - 1  # number of cells
    knots = space_1d.knots

    # We have two different behaviours depending on whether the spline space is periodic or not
    if space_1d.periodic:
        # interpolation
        if space_1d.basis == "B":
            if p == 1 and h != 1.0:
                x_grid = np.linspace(-(p - 1) * h, 1.0 - h + (h / 2.0), (N + p - 1) * 2)
            else:
                x_grid = np.linspace(-(p - 1) * h, 1.0 - h, (N + p - 1) * 2 - 1)

            pts = x_grid[:, None] % 1.0
            wts = np.ones(pts.shape, dtype=float)

            # !! shift away first interpolation point in eta_1 direction for polar domains !!
            if pts[0] == 0.0 and polar_shift:
                pts[0] += 0.00001

        # histopolation
        elif space_1d.basis == "M":
            # The computation of histopolation points breaks in case we have Nel=1 and periodic boundary conditions since we end up with only one x_grid point.
            # We need to build the histopolation points by hand in this scenario.
            if p == 0 and h == 1.0:
                x_grid = np.array([0.0, 0.5, 1.0])
            elif p == 0 and h != 1.0:
                x_grid = np.linspace(-p * h, 1.0 - h + (h / 2.0), (N + p) * 2)
            else:
                x_grid = np.linspace(-p * h, 1.0 - h, (N + p) * 2 - 1)

            n_quad = p + 1
            # Gauss - Legendre quadrature points and weights
            # products of basis functions are integrated exactly
            pts_loc, wts_loc = np.polynomial.legendre.leggauss(n_quad)

            x, wts = bsp.quadrature_grid(x_grid, pts_loc, wts_loc)
            pts = x % 1.0
    else:
        # interpolation
        if space_1d.basis == "B":
            if p <= 2:
                raise Exception("The local projector with clamped boundary conditions only support p > 2.")

            # Number of B-splines
            N_b = N + p

            # Filling the quasi-interpolation points for i=0 and i=1 (since they are equal)
            x_grid = np.linspace(0.0, knots[p + 1], p + 1)
            x_aux = np.linspace(0.0, knots[p + 1], p + 1)
            x_grid = np.append(x_grid, x_aux)
            # Now we append those for 1<i<p-1
            for i in range(2, p - 1):
                x_aux = np.linspace(knots[p], knots[p + i], p + i)
                x_grid = np.append(x_grid, x_aux)

            # Now we append the points for p-1<= i <= N_b-p
            x_aux = np.linspace(0.0, 1.0, 2 * N + 1)
            x_grid = np.append(x_grid, x_aux)

            # Now the points for N_b-p < i < N_b-1
            for i in range(N_b - p + 1, N_b - 1):
                x_aux = np.linspace(knots[i + 1], knots[N_b], N_b + p - i - 1)
                x_grid = np.append(x_grid, x_aux)
            # Finally we add the pointset for i = N_b-1, which is the same as the one for i = N_b-2
            i = N_b - 2
            x_aux = np.linspace(knots[i + 1], knots[N_b], N_b + p - i - 1)
            x_grid = np.append(x_grid, x_aux)

            if polar_shift:
                for i in range(len(x_grid)):
                    if x_grid[i] == 0.0:
                        x_grid[i] += 0.00001

            pts = x_grid[:, None]
            wts = np.ones(pts.shape, dtype=float)

        # histopolation
        elif space_1d.basis == "M":
            # B-spline degree
            p += 1
            if p <= 2:
                raise Exception("The local projector with clamped boundary conditions only support p > 2.")

            # Number of B-splines
            N_b = N + p

            # IMPORTANT: The way in which the knots are defined is such that for D-splines they have one
            # less padding on the left and one less padding on the right compare to the knots for B-splines.
            # Thus, we must substract 1 to all the indices of the knots here to refere to the same point.

            # Filling the quasi-interpolation points for i=0 and i=1 (since they are equal)
            x_grid = np.linspace(0.0, knots[p], p + 1)
            x_aux = np.linspace(0.0, knots[p], p + 1)
            x_grid = np.append(x_grid, x_aux)
            # Now we append those for 1<i<p-1
            for i in range(2, p - 1):
                x_aux = np.linspace(knots[p - 1], knots[p + i - 1], p + i)
                x_grid = np.append(x_grid, x_aux)

            # Now we append the points for p-1<= i <= N_b-p
            x_aux = np.linspace(0.0, 1.0, 2 * N + 1)
            x_grid = np.append(x_grid, x_aux)

            # Now the points for N_b-p < i < N_b-1
            for i in range(N_b - p + 1, N_b - 1):
                x_aux = np.linspace(knots[i], knots[N_b - 1], N_b + p - i - 1)
                x_grid = np.append(x_grid, x_aux)
            # Finally we add the pointset for i = N_b-1, which is the same as the one for i = N_b-2
            i = N_b - 2
            x_aux = np.linspace(knots[i], knots[N_b - 1], N_b + p - i - 1)
            x_grid = np.append(x_grid, x_aux)

            # Gauss - Legendre quadrature points and weights
            # products of basis functions are integrated exactly
            n_quad = p
            pts_loc, wts_loc = np.polynomial.legendre.leggauss(n_quad)

            x, wts = bsp.quadrature_grid(x_grid, pts_loc, wts_loc)
            pts = x

    return pts, wts


def get_span_and_basis(pts, space):
    """Compute the knot span index and the values of p + 1 basis function at each point in pts.

    Parameters
    ----------
    pts : np.array
        2d array of points (ii, iq) = (interval, quadrature point).

    space : SplineSpace
        Psydac object, the 1d spline space to be projected.

    Returns
    -------
    span : np.array
        2d array indexed by (n, nq), where n is the interval and nq is the quadrature point in the interval.

    basis : np.array
        3d array of values of basis functions indexed by (n, nq, basis function).
    """

    # Extract knot vectors, degree and kind of basis
    T = space.knots
    p = space.degree

    span = np.zeros(pts.shape, dtype=int)
    basis = np.zeros((*pts.shape, p + 1), dtype=float)

    for n in range(pts.shape[0]):
        for nq in range(pts.shape[1]):
            # avoid 1. --> 0. for clamped interpolation
            x = pts[n, nq] % (1.0 + 1e-14)
            span_tmp = bsp.find_span(T, p, x)
            basis[n, nq, :] = bsp.basis_funs_all_ders(
                T,
                p,
                x,
                span_tmp,
                0,
                normalization=space.basis,
            )
            span[n, nq] = span_tmp  # % space.nbasis

    return span, basis


def get_weights_local_projector(pts, fem_space):
    """Compute the geometric weights for interpolation and histopolation.
    Should be called only with the grid points for 0-forms.

    Parameters
    ----------
    pts : np.array
        3d array of points. Contains the quasi-interpolation points in each direction.

    fem_space : SplineSpace
        Psydac object, the 1d spline space to be projected. Should be the 0-form space.

    Returns
    -------
    wij : List of np.array
        List of 2d array indexed by (space_direction, i, j), where i determines for which FEEC coefficient this weights are needed. Used for interpolation.

    whij : List of np.array
        List of 2d array indexed by (space_direction, i, j), where i determines for which FEEC coefficient this weights are needed. Used for histopolation.
    """
    wij = []
    whij = []

    # In the clamped case
    # for a fixed value of i the number of j entries wij can have may change. We need to compute the maximum number of j entries necessary, to make sure that all
    # wij will have the same number of j entries by adding zeros as padding to those that come short of this number. As to why we want all the wij to have the same
    # number of entries? It so we can build a 2D numpy array of them.

    #######
    ##Computing the max number of j entries of wij for each spatial direction
    #######

    # List with the degree of the B-splines in each spatial direction
    plist = np.zeros(3, dtype=int)
    # List with a bool that tell us if the B-splines in each spatial direction are periodic
    periodiclist = []
    # We iterate over each one of the spatial dimension of the 0 fem_space
    for d, space in enumerate(fem_space.spaces):
        plist[d] = space.degree
        periodiclist.append(space.periodic)

    periodiclist = np.array(periodiclist)
    # We get the maximum number of j entries for wij
    lenj1, lenj2, lenj3 = get_local_problem_size(periodiclist, plist, np.array([False, False, False], dtype=bool))

    maxjwij = [lenj1, lenj2, lenj3]

    # Now we must do the same for the whij

    #######
    ##Computing the max number of j entries of whij for each spatial direction
    #######

    # We get the maximum number of j entries for whij
    lenj1, lenj2, lenj3 = get_local_problem_size(periodiclist, plist, np.array([True, True, True], dtype=bool))

    maxjwhij = [lenj1, lenj2, lenj3]

    #######
    ##Building the wij
    #######

    # We iterate over each one of the spatial dimension of the 0 fem_space
    for d, space in enumerate(fem_space.spaces):
        # Extract knot vectors, degree and kind of basis
        T = space.knots
        p = space.degree
        periodic = space.periodic
        x = pts[d].flatten()
        colmatrix = bsp.collocation_matrix(T, p, periodic, "B", x)

        # Number of B-splines
        Nbasis = colmatrix.shape[1]
        wijaux = []
        whijaux = []
        # If we have periodic boundary conditions the minicolocationmatrix will be the same for all i.
        # So we can compute it just once .
        if periodic:
            i = 0
            # We get the indices that tell us which entries of x to get
            xstart, xend = select_quasi_points(int(i), int(p), int(Nbasis), bool(periodic))
            # Now we get the indices that tell us which basis functions to consider
            bstart, bend = select_basis_local(i, p, Nbasis, periodic)
            # We can finally build the minicollocation matrix necessary to obtain the weights wij
            counter = 1
            minicol = colmatrix[xstart:xend, bstart]
            while counter < 2 * p - 1:
                minicol = np.column_stack(
                    (minicol, colmatrix[xstart:xend, (bstart + counter) % Nbasis]),
                )
                counter += 1

            # We need to consider the case in which our minicollocation matrix ends up being just one number
            if np.shape(minicol)[0] == 1:
                # There seems to be a bug with the bsp.collocation_matrix function for the case Nel = 1, p = 1 and periodic, when evaluating the only B-spline at 0 the answer should be 1 not 0.
                if p == 1 and Nbasis == 1:
                    minicol[0] = 1.0
                invmini = 1.0 / minicol[0]
                for i in range(Nbasis):
                    wijaux.append(np.array([invmini]))
            else:
                invmini = np.linalg.inv(minicol)
                for i in range(Nbasis):
                    wijaux.append(invmini[p - 1, :])
        else:
            for i in range(Nbasis):
                # We get the indices that tell us which entries of x to get
                xstart, xend = select_quasi_points(int(i), int(p), int(Nbasis), bool(periodic))
                # Now we get the indices that tell us which basis functions to consider
                bstart, bend = select_basis_local(i, p, Nbasis, periodic)
                # We can finally build the minicollocation matrix necessary to obtain the weights wij
                minicol = colmatrix[xstart:xend, bstart:bend]
                # Now we get its inverse
                invmini = np.linalg.inv(minicol)

                # Now we need to extract the row of invmini that corresponds to the ith histopolation coefficient.
                if i == 0:
                    relevant_row = 0
                elif i < (p - 1):
                    relevant_row = i
                elif i < (Nbasis - 1):
                    relevant_row = p - 1
                elif i == (Nbasis - 1):
                    relevant_row = p

                # At this point auxiliar contains the geometric weights (wi0, wi1, ...)
                auxiliar = invmini[relevant_row, :].tolist()

                # We must now add a padding of zeros at the end of auxiliar to make sure that for all i the wij have the same length.
                # This is necessary to convert these list into arrays later on.
                for j in range(len(auxiliar), maxjwij[d]):
                    auxiliar.append(0.0)

                wijaux.append(np.array(auxiliar))

        wij.append(np.array(wijaux))

        # Now that we know the wij we must use them to compute the whij
        # We begin by adressing the special case p=1 and periodic
        # This is a special case since some of the integrals in the definition of the histopolation operator vanish.
        if periodic:
            # Number of D-splines
            nD = Nbasis
            if p == 1:
                for i in range(nD):
                    whijaux.append(np.array([wijaux[i][0], wijaux[i][0]]))
            else:
                whats = [wijaux[0][0], wijaux[0][0] + wijaux[0][1]]
                for j in range(2, 2 * p - 1):
                    whats.append(wijaux[0][j - 1] + wijaux[0][j])
                whats.append(wijaux[0][2 * p - 2])
                for i in range(nD):
                    whijaux.append(np.array(whats))

        else:
            # Number of D-splines
            nD = Nbasis - 1
            for i in range(nD):
                whats = []

                if i == 0 or i == (Nbasis - 2):
                    for j in range(maxjwhij[d]):
                        if j <= p - 1:
                            sumval = 0.0
                            for q in range(j + 1):
                                sumval += wijaux[i][q] - wijaux[i + 1][q]
                            whats.append(sumval)
                        else:
                            whats.append(0.0)

                elif 0 < i and i < (p - 1):
                    for j in range(maxjwhij[d]):
                        if j <= (p + i - 2):
                            sumval = 0.0
                            for q in range(j + 1, p + i):
                                sumval += wijaux[i][q]
                            whats.append(-1.0 * sumval)
                        elif j == (p + i - 1):
                            whats.append(0.0)
                        elif (p + i) <= j and j <= (2 * p + 2 * i - 1):
                            sumval = 0.0
                            for q in range(j - p - i + 1, p + i + 1):
                                sumval += wijaux[i + 1][q]
                            whats.append(sumval)
                        else:
                            whats.append(0.0)

                elif (p - 1) <= i and i < (Nbasis - p):
                    for j in range(maxjwhij[d]):
                        if j == 0:
                            whats.append(wijaux[i][0])
                        elif j == 1:
                            whats.append(wijaux[i][0] + wijaux[i][1])
                        elif 2 <= j and j <= (2 * p - 2):
                            sumval = 0.0
                            for q in range(j - 1):
                                sumval += wijaux[i][q] - wijaux[i + 1][q]
                            sumval += wijaux[i][j - 1]
                            sumval += wijaux[i][j]
                            whats.append(sumval)

                        elif j == (2 * p - 1):
                            sumval = 0.0
                            for q in range(2 * p - 2):
                                sumval += wijaux[i][q] - wijaux[i + 1][q]
                            sumval += wijaux[i][2 * p - 2]
                            whats.append(sumval)
                        else:
                            whats.append(0.0)

                elif (Nbasis - p) <= i and i < (Nbasis - 2):
                    for j in range(maxjwhij[d]):
                        if j <= (Nbasis + p - i - 3):
                            sumval = 0.0
                            for q in range(j + 1):
                                sumval += wijaux[i][q]
                            whats.append(sumval)

                        elif j == (Nbasis + p - i - 2):
                            whats.append(0.0)

                        elif (Nbasis + p - i - 1) <= j and j <= (2 * Nbasis + 2 * p - 2 * i - 5):
                            sumval = 0.0
                            for q in range(j - Nbasis - p + i + 2):
                                sumval += wijaux[i + 1][q]
                            whats.append(-1.0 * sumval)

                        else:
                            whats.append(0.0)

                whijaux.append(np.array(whats))

        whij.append(np.array(whijaux))

    return wij, whij


# We need a function that tell us which of the basis functions to take for the computation of the wij, for any i
def select_basis_local(i, p, Nbasis, periodic):
    """Determines the start and end indices of the basis functions that must be taken from the collocation matrix to compute the geometric weights wij, for any given i.

    Parameters
    ----------
    i : int
        Index of the wij weights that must be computed.

    p : int
        B-spline degree.

    Nbasis: int
        Number of B-spline.

    periodic: bool
        Whether we have periodic boundary conditions.

    Returns
    -------
    start : int
        Start index of the B-splines that must be consider in the collocation matrix to obtain the wij weights. Inclusive index

    end : int
        End index of the B-splines that must be consider in the collocation matrix to obtain the wij weights. Exclusive index
    """
    if periodic:
        start = (i + 1 - p) % Nbasis
        end = (i + p) % Nbasis
    else:
        if i == 0:
            start = 0
            end = p + 1
        elif i < (p - 1):
            start = 0
            end = p + i
        elif i <= (Nbasis - p):
            start = i + 1 - p
            end = i + p
        elif i < (Nbasis - 1):
            start = i + 1 - p
            end = Nbasis
        elif i == (Nbasis - 1):
            start = Nbasis - 1 - p
            end = Nbasis
    return start, end
