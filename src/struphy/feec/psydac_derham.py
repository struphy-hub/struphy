#!/usr/bin/env python3

from sympde.topology import Cube
from sympde.topology import Derham as Derham_psy

from psydac.api.discretization import discretize
from psydac.fem.vector import VectorFemSpace
from psydac.feec.global_projectors import Projector_H1vec
from psydac.linalg.stencil import StencilVector
from psydac.linalg.block import BlockVector
from psydac.linalg.basic import IdentityOperator

from struphy.feec.linear_operators import BoundaryOperator
from struphy.feec.geom_projectors import PolarCommutingProjector
from struphy.polar.basic import PolarDerhamSpace
from struphy.polar.extraction_operators import PolarExtractionBlocksC1
from struphy.polar.linear_operators import PolarExtractionOperator, PolarLinearOperator
from struphy.polar.basic import PolarVector
from struphy.initial import perturbations
from struphy.initial import eigenfunctions
from struphy.geometry.base import Domain
from struphy.bsplines import evaluation_kernels_3d as eval_3d
from struphy.fields_background.mhd_equil.equils import set_defaults

import numpy as np
from mpi4py import MPI


class Derham:
    """
    API for the discrete Derham sequence on the logical unit cube (3d).

    Check out `Tutorial 09 <https://struphy.pages.mpcdf.de/struphy/tutorials/tutorial_09_discrete_derham.html>`_ for a hands-on introduction.

    The tensor-product discrete deRham complex is loaded from `Psydac <https://github.com/pyccel/psydac>`_ 
    and then augmented with polar sub-spaces (indicated by a bar) and boundary operators.

    .. image:: ../pics/polar_derham.png

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
        MPI communicator.

    mpi_dims_mask: list of bool
        True if the dimension is to be used in the domain decomposition (=default for each dimension). 
        If mpi_dims_mask[i]=False, the i-th dimension will not be decomposed.

    with_projectors : bool
        Whether to add global commuting projectors to the diagram.

    polar_ck : int
        Smoothness at a polar singularity at eta_1=0 (default -1 : standard tensor product splines, OR 1 : C1 polar splines)

    domain : struphy.geometry.domains
        Mapping from logical unit cube to physical domain (only needed in case of polar splines polar_ck=1).
    """

    def __init__(self,
                 Nel,
                 p,
                 spl_kind,
                 dirichlet_bc=None,
                 nquads=None,
                 nq_pr=None,
                 comm=None,
                 mpi_dims_mask=None,
                 with_projectors=True,
                 polar_ck=-1,
                 domain=None):

        # number of elements, spline degrees and kind of splines in each direction (periodic vs. clamped)
        assert len(Nel) == 3
        assert len(p) == 3
        assert len(spl_kind) == 3

        self._Nel = Nel
        self._p = p
        self._spl_kind = spl_kind

        # boundary conditions at eta=0 and eta=1 in each direction (None for periodic, 'd' for homogeneous Dirichlet)
        if dirichlet_bc is not None:
            assert len(dirichlet_bc) == 3
            # make sure that boundary conditions are compatible with spline space
            assert np.all([bc == [False, False] for i, bc in enumerate(dirichlet_bc) if spl_kind[i]])

        self._dirichlet_bc = dirichlet_bc

        # default p: exact integration of degree 2p-1 polynomials
        if nquads is None:
            self._nquads = [pi for pi in p]
        else:
            assert len(nquads) == 3
            self._nquads = nquads

        # default p + 1 : exact integration of degree 2p+1 polynomials
        if nq_pr is None:
            self._nq_pr = [pi + 1 for pi in p]
        else:
            assert len(nq_pr) == 3
            self._nq_pr = nq_pr

        # MPI communicator
        self._comm = comm

        # set polar splines (currently standard tensor-product (-1) and C^1 polar splines (+1) are supported)
        assert polar_ck in {-1, 1}
        self._polar_ck = polar_ck

        # Psydac symbolic logical domain (unit cube)
        self._domain_log = Cube('C', bounds1=(0, 1),
                                bounds2=(0, 1),
                                bounds3=(0, 1))

        # Psydac symbolic Derham
        self._derham_symb = Derham_psy(self._domain_log)

        # discrete logical domain : the parallelism is initiated here.
        self._domain_log_h = discretize(
            self._domain_log,
            ncells=Nel,
            comm=self._comm,
            periodic=self.spl_kind,
            mpi_dims_mask=mpi_dims_mask)

        # Psydac discrete de Rham, projectors and derivatives
        _derham = discretize(self._derham_symb, self._domain_log_h,
                             degree=self.p, nquads=self.nquads)

        self._grad, self._curl, self._div = _derham.derivatives_as_matrices

        _projectors = _derham.projectors(nquads=self._nq_pr)

        # expose name-to-form dict
        self._space_to_form = {'H1': '0',
                               'Hcurl': '1',
                               'Hdiv': '2',
                               'L2': '3',
                               'H1vec': 'v'}

        # Psydac vector space and FEM spline spaces
        _Vnames = {'0': 'H1',
                   '1': 'Hcurl',
                   '2': 'Hdiv',
                   '3': 'L2',
                   'v': 'H1vec'}

        self._Vh = {}
        self._Vh_fem = {}
        self._P = {}

        for i, key in enumerate(_Vnames.keys()):

            if key == 'v':
                self._Vh_fem[key] = VectorFemSpace(
                    _derham.V0, _derham.V0, _derham.V0)
                self._P[key] = Projector_H1vec(self._Vh_fem[key])
            else:
                self._Vh_fem[key] = getattr(_derham, 'V' + str(i))
                self._P[key] = _projectors[i]

            self._Vh[key] = self._Vh_fem[key].vector_space

        # total number of basis functions and spline types of 1d spaces in each direction ('B' or 'M', resp. 0 or 1)
        self._nbasis = {}
        self._spline_types = {}
        self._spline_types_pyccel = {}

        for i, key in enumerate(_Vnames.keys()):
            fem_space = self._Vh_fem[key]

            if key in {'0', '3'}:
                self._nbasis[key] = [
                    space.nbasis for space in fem_space.spaces]
                self._spline_types[key] = [
                    space.basis for space in fem_space.spaces]
                self._spline_types_pyccel[key] = np.array(
                    [int(space.basis == 'M') for space in fem_space.spaces])
            else:
                self._nbasis[key] = [[space.nbasis for space in comp.spaces]
                                     for comp in fem_space.spaces]
                self._spline_types[key] = [
                    [space.basis for space in comp.spaces] for comp in fem_space.spaces]
                self._spline_types_pyccel[key] = [np.array(
                    [int(space.basis == 'M') for space in comp.spaces]) for comp in fem_space.spaces]

        # break points
        self._breaks = [space.breaks for space in _derham.spaces[0].spaces]

        # index arrays
        self._indN = [(np.indices((space.ncells, space.degree + 1))[1] + np.arange(
            space.ncells)[:, None]) % space.nbasis for space in self._Vh_fem['0'].spaces]
        self._indD = [(np.indices((space.ncells, space.degree + 1))[1] + np.arange(
            space.ncells)[:, None]) % space.nbasis for space in self._Vh_fem['3'].spaces]

        # distribute info on domain decomposition
        self._domain_decomposition = self._Vh['0'].cart.domain_decomposition

        self._domain_array = self._get_domain_array()
        self._breaks_loc = [self.breaks[k][self.domain_decomposition.starts[k]:
                                           self.domain_decomposition.ends[k] + 2] for k in range(3)]

        self._index_array = self._get_index_array(
            self._domain_decomposition)
        self._index_array_N = self._get_index_array(self._Vh['0'].cart)
        self._index_array_D = self._get_index_array(self._Vh['3'].cart)

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

        for i, key in enumerate(_Vnames.keys()):

            vec_space = self._Vh[key]

            # ------ Extraction operators ------
            # tensor product case
            if self.polar_ck == -1:

                pol_space = self._Vh[key]

                self._extraction_ops[key] = IdentityOperator(pol_space)
                self._dofs_extraction_ops[key] = IdentityOperator(pol_space)

            # C^1 polar spline case
            else:

                pol_space = PolarDerhamSpace(self, _Vnames[key])

                self._extraction_ops[key] = PolarExtractionOperator(
                    vec_space, pol_space, ck_blocks.e_ten_to_pol[key])

                self._dofs_extraction_ops[key] = PolarExtractionOperator(
                    vec_space, pol_space, ck_blocks.p_ten_to_pol[key], ck_blocks.p_ten_to_ten[key])

            self._Vh_pol[key] = pol_space
            
            # ------ Hom. Dirichlet boundary operators ------
            if self.dirichlet_bc is None:
                self._boundary_ops[key] = IdentityOperator(pol_space)
            else:
                self._boundary_ops[key] = BoundaryOperator(
                    pol_space, _Vnames[key], self.dirichlet_bc)

            # ------ Assemble projectors ------
            if with_projectors:
                self._P[key] = PolarCommutingProjector(
                    self._P[key], self._dofs_extraction_ops[key], self._extraction_ops[key], self._boundary_ops[key])

        # set discrete derivatives with polar linear operators
        if self.polar_ck == 1:
            self._grad = PolarLinearOperator(
                self._Vh_pol['0'], self._Vh_pol['1'], self._grad, ck_blocks.grad_pol_to_ten, ck_blocks.grad_pol_to_pol, ck_blocks.grad_e3)
            self._curl = PolarLinearOperator(
                self._Vh_pol['1'], self._Vh_pol['2'], self._curl, ck_blocks.curl_pol_to_ten, ck_blocks.curl_pol_to_pol, ck_blocks.curl_e3)
            self._div = PolarLinearOperator(
                self._Vh_pol['2'], self._Vh_pol['3'], self._div, ck_blocks.div_pol_to_ten, ck_blocks.div_pol_to_pol, ck_blocks.div_e3)

        # set discrete derivatives with and without boundary operators
        self._grad_bcfree = self._grad
        self._curl_bcfree = self._curl
        self._div_bcfree = self._div

        self._grad = self._boundary_ops['1'] @ self._grad @ self._boundary_ops['0'].T
        self._curl = self._boundary_ops['2'] @ self._curl @ self._boundary_ops['1'].T
        self._div = self._boundary_ops['3'] @ self._div @ self._boundary_ops['2'].T

    @property
    def Nel(self):
        """ List of number of elements (=cells) in each direction.
        """
        return self._Nel

    @property
    def p(self):
        """ List of B-spline degrees in each direction.
        """
        return self._p

    @property
    def spl_kind(self):
        """ List of spline type (periodic=True or clamped=False) in each direction.
        """
        return self._spl_kind

    @property
    def dirichlet_bc(self):
        """ None, or list of boundary conditions in each direction. 
        Each entry is a list with two entries (left and right boundary), "d" (hom. Dirichlet) or None (periodic).
        """
        return self._dirichlet_bc

    @property
    def nquads(self):
        """ List of number of Gauss-Legendre quadrature points in each direction (default = p, = p + 1 points per cell).
        """
        return self._nquads

    @property
    def nq_pr(self):
        """ List of number of Gauss-Legendre quadrature points in histopolation (default = p + 1) in each direction.
        """
        return self._nq_pr

    @property
    def comm(self):
        """ MPI communicator.
        """
        return self._comm

    @property
    def polar_ck(self):
        """ C^k smoothness at eta_1=0.
        """
        return self._polar_ck

    @property
    def breaks(self):
        """ List of break points (=cell interfaces) in each direction.
        """
        return self._breaks

    @property
    def indN(self):
        """ List of 2d arrays holding global spline indices (N) in each element in the three directions.
        """
        return self._indN

    @property
    def indD(self):
        """ List of 2d arrays holding global spline indices (D) in each element in the three directions.
        """
        return self._indD

    @property
    def domain_decomposition(self):
        """ Psydac's domain decomposition object (same for all vector spaces!).
        """
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
        """ Dictionary containing the names of the continuous spaces and corresponding discrete spaces.
        """
        return self._space_to_form

    @property
    def Vh(self):
        """ Dictionary containing finite-dimensional vector spaces (sub-spaces of continuous spaces, Stencil-/BlockVectorSpace).
        """
        return self._Vh

    @property
    def Vh_fem(self):
        """ Dictionary containing FEM spline spaces (TensorFem-/VectorFemSpace).
        """
        return self._Vh_fem

    @property
    def nbasis(self):
        """ Dictionary containing number of 1d basis functions for each component and spatial direction.
        """
        return self._nbasis

    @property
    def spline_types(self):
        """ Dictionary holding 1d spline types for each component and spatial direction, entries either 'B' or 'M'.
        """
        return self._spline_types

    @property
    def spline_types_pyccel(self):
        """ Dictionary holding 1d spline types for each component and spatial direction, entries either 0 (='B') or 1 (='M').
        """
        return self._spline_types_pyccel

    @property
    def extraction_ops(self):
        """ Dictionary holding basis extraction operators, either IdentityOperator or PolarExtractionOperator.
        """
        return self._extraction_ops

    @property
    def dofs_extraction_ops(self):
        """ Dictionary holding dof extraction operators for commuting projectors, either IdentityOperator or PolarExtractionOperator.
        """
        return self._dofs_extraction_ops

    @property
    def boundary_ops(self):
        """ Dictionary holding essential boundary operators (BoundaryOperator) OR IdentityOperators.
        """
        return self._boundary_ops

    @property
    def P(self):
        """ Dictionary holding global commuting projectors.
        """
        return self._P

    @property
    def Vh_pol(self):
        """ Polar sub-spaces, either PolarDerhamSpace (with polar splines) or Stencil-/BlockVectorSpace (same as self.Vh)
        """
        return self._Vh_pol

    @property
    def grad_bcfree(self):
        """ Discrete gradient Vh0_pol (H1) -> Vh1_pol (Hcurl) w/o boundary operator.
        """
        return self._grad_bcfree

    @property
    def curl_bcfree(self):
        """ Discrete curl Vh1_pol (Hcurl) -> Vh2_pol (Hdiv) w/o boundary operator.
        """
        return self._curl_bcfree

    @property
    def div_bcfree(self):
        """ Discrete divergence Vh2_pol (Hdiv) -> Vh3_pol (L2) w/o boundary operator.
        """
        return self._div_bcfree

    @property
    def grad(self):
        """ Discrete gradient Vh0_pol (H1) -> Vh1_pol (Hcurl).
        """
        return self._grad

    @property
    def curl(self):
        """ Discrete curl Vh1_pol (Hcurl) -> Vh2_pol (Hdiv).
        """
        return self._curl

    @property
    def div(self):
        """ Discrete divergence Vh2_pol (Hdiv) -> Vh3_pol (L2).
        """
        return self._div

    # --------------------------
    #      methods:
    # --------------------------

    def create_field(self, name, space_id):
        return self.Field(name, space_id, self)

    # --------------------------
    #      private methods:
    # --------------------------

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
        if self.comm is not None:
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

            dom_arr_loc[3*n + 0] = brks[el_sta + 0]
            dom_arr_loc[3*n + 1] = brks[el_end + 1]
            dom_arr_loc[3*n + 2] = el_end - el_sta + 1

        # distribute
        if self.comm is not None:
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
        if self.comm is not None:
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

            ind_arr_loc[2*n + 0] = sta
            ind_arr_loc[2*n + 1] = end

        # distribute
        if self.comm is not None:
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

            neigh_inds = [None]*6

            # in each direction find start/end index for neighbour
            for k, co in enumerate(comp):
                if co == 1:
                    neigh_inds[2*k + 0] = self.index_array[rank, 2*k + 0]
                    neigh_inds[2*k + 1] = self.index_array[rank, 2*k + 1]

                elif co == 0:
                    neigh_inds[2*k + 1] = gl_s[k] - 1
                    if kinds[k]:
                        neigh_inds[2*k + 1] %= ncells[k]

                elif co == 2:
                    neigh_inds[2*k] = gl_e[k] + 1
                    if kinds[k]:
                        neigh_inds[2*k] %= ncells[k]

                else:
                    raise ValueError(
                        'Wrong value for component; must be 0 or 1 or 2 !')

            neigh_inds = np.array(neigh_inds)

            # only use indices where information is present to find the neighbours rank
            inds = np.where(neigh_inds != None)

            # find ranks (row index of domain_array) which agree in start/end indices
            index_temp = np.squeeze(self.index_array[:, inds])
            unique_ranks = np.where(
                np.equal(index_temp, neigh_inds[inds]).all(1))[0]

            # if any row satisfies condition, return its index (=rank of neighbour)
            if len(unique_ranks) != 0:
                neigh_id = unique_ranks[0]

        return neigh_id

    # --------------------------
    # Inner classes
    # --------------------------
    class Field:
        """
        Initializes a callable field variable (i.e. its FE coefficients) in memory and creates a method for assigning initial conditions.

        Parameters
        ----------
        name : str
            Field's key to be used for instance when saving to hdf5 file.

        space_id : str
            Space identifier for the field ("H1", "Hcurl", "Hdiv", "L2" or "H1vec").

        derham : struphy.feec.psydac_derham.Derham
            Discrete Derham complex.
        """

        def __init__(self, name, space_id, derham):

            self._name = name
            self._space_id = space_id
            self._derham = derham

            # initialize field in memory (FEM space, vector and tensor product (stencil) vector)
            self._space_key = derham.space_to_form[space_id]
            self._space = derham.Vh_fem[self._space_key]

            self._vector = derham.Vh_pol[self._space_key].zeros()

            self._vector_stencil = self._space.vector_space.zeros()

            # transposed basis extraction operator for PolarVector --> Stencil-/BlockVector
            self._ET = derham.extraction_ops[self._space_key].transpose()

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
            """ 3d Derham complex struphy.feec.psydac_derham.Derham.
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
                        assert isinstance(
                            value[1], (StencilVector, np.ndarray))

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

                if self.name in init_params[init_params['type']]['comps']:

                    init_types += [init_params['type']]
                    fun_params += [init_params[init_types[0]].copy()]

            elif type(init_params['type']) == list:

                for n, _type in enumerate(init_params['type']):

                    if self.name in init_params[_type]['comps']:

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
                        AssertionError(
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

                    if 'H1vec' in self.space_id:
                        form_str = 'vector'
                    else:
                        form_str = self.space_key + '_form'

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
            self.derham.boundary_ops[self.space_key].dot(
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

        def __call__(self, eta1, eta2, eta3, out=None, tmp=None, squeeze_output=False, local=False):
            """
            Evaluates the spline function on the global domain, unless local is given to True (in which case the spline function is evaluated only on the local domain).

            Parameters
            ----------
            eta1, eta2, eta3 : array-like
                Logical coordinates at which to evaluate.

            out : array[float] or list
                Array in which to store the values of the spline function at the given point set (list in case of vector-valued spaces).

            tmp : array[float]
                Array that has shape the size of the grid that will be used as a temporary for AllReduce, to avoid creating it a each call.

            flat_eval : bool
                Whether to do a flat evaluation, i.e. f([e11, e12], [e21, e22]) = [f(e11, e21), f(e12, e22)].

            squeeze_output : bool
                Whether to remove singleton dimensions in output "values".

            Returns
            -------
                out : array[float] or list
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
            if tmp is None:
                tmp = np.zeros((E1.shape[0], E2.shape[1],
                                E3.shape[2]), dtype=float)
            else :
                assert isinstance(tmp, np.ndarray)
                assert tmp.shape == (E1.shape[0], E2.shape[1],
                                E3.shape[2])
                assert tmp.dtype.type is np.float64
                tmp[:] = 0.

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
                        self.derham.comm.Allreduce(
                            MPI.IN_PLACE, tmp, op=MPI.SUM)

                # all processes have all values
                if out is None : 
                    out = tmp
                else :
                    out *= 0.
                    out += tmp

                if squeeze_output:
                    out = np.squeeze(out)

                if out.ndim == 0:
                    out = out.item()

            else:

                out_is_None = out is None
                if out_is_None:
                    out = []
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
                    if out_is_None:
                        out += [tmp.copy()]
                    else :
                        out[n] *= 0.
                        out[n] += tmp

                    tmp[:] = 0.

                    if squeeze_output:
                        out[-1] = np.squeeze(out[-1])

                    if out[-1].ndim == 0:
                        out[-1] = out[-1].item()

            return out

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
                    mid_pt = (domain_array[i, 3*n] +
                              domain_array[i, 3*n + 1]) / 2.
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
                        _amps[:] = (np.random.rand(
                            *shapes) - .5) * 2. * amp_size

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

    Depending on the dimension of eta1 either point-wise, tensor-product, slice plane or general (see :ref:`struphy.geometry.base.prepare_arg`).

    Parameters
    ----------
    coords : str
        From which coordinate representation to pull, either 'logical' or 'physical'.

    fun : list
        Callable function components. Has to be length 3 for 1- and 2-forms, length 1 otherwise.

    domain: struphy.geometry.domains
        All things mapping.

    form : str
        Which form to pull: '0_form', '3_form' or 'xxx_1', 'xxx_2', 'xxx_3', where xxx is either '1_form', '2_form' or 'vector'.

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
