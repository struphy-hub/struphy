#!/usr/bin/env python3

from sympde.topology import Cube
from sympde.topology import Derham as Derham_psy

from psydac.api.discretization import discretize
from psydac.fem.vector import VectorFemSpace
from psydac.feec.global_projectors import Projector_H1vec

from struphy.psydac_api.linear_operators import BoundaryOperator, CompositeLinearOperator, IdentityOperator
from struphy.psydac_api.geom_projectors import Projector

from struphy.polar.basic import PolarDerhamSpace
from struphy.polar.extraction_operators import PolarExtractionBlocksC1
from struphy.polar.linear_operators import PolarExtractionOperator, PolarLinearOperator

import numpy as np


class Derham:
    """
    Psydac API for the discrete Derham sequence on the logical unit cube (3d).

    Polar sub-spaces (indicated by a bar) can be added.

    .. image:: ../pics/polar_derham.png

    Parameters
    ----------
    Nel : list[int]
        Number of elements in each direction.

    p : list[int]
        Spline degree in each direction.

    spl_kind : list[bool]
        Kind of spline in each direction (True=periodic, False=clamped).

    bc : list[str]
        Homogeneous Dirichlet boundary condition in each direction.

    nq_pr : list[int]
        Number of Gauss-Legendre quadrature points in histopolation in each direction (default = p + 1).

    quad_order : list[int]
        Degree of Gauss-Legendre quadrature in each direction (default = p, leads to p + 1 quadrature points per cell).

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

    def __init__(self, Nel, p, spl_kind, bc=None, quad_order=None, nq_pr=None, comm=None, mpi_dims_mask=None, with_projectors=True, polar_ck=-1, domain=None):

        # number of elements, spline degrees and kind of splines in each direction (periodic vs. clamped)
        assert len(Nel) == 3
        assert len(p) == 3
        assert len(spl_kind) == 3

        self._Nel = Nel
        self._p = p
        self._spl_kind = spl_kind

        # boundary conditions at eta=0 and eta=1 in each direction (None for periodic, 'd' for homogeneous Dirichlet)
        if bc is None:
            self._bc = [[None, None], [None, None], [None, None]]
        else:
            assert len(bc) == 3
            if spl_kind[0]:
                assert bc[0][0] is None and bc[0][1] is None
            if spl_kind[1]:
                assert bc[1][0] is None and bc[1][1] is None
            if spl_kind[2]:
                assert bc[2][0] is None and bc[2][1] is None

            self._bc = bc

        # default quad_order p: exact integration of products of B-splines
        if quad_order is None:
            self._quad_order = [pi for pi in p]
        else:
            assert len(quad_order) == 3
            self._quad_order = quad_order

        # default number of histopolation quadrature points per interval p + 1 : exact histopolation of products of B-splines
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

        # Psydac symbolic logical domain
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
                             degree=self.p, nquads=self.quad_order)

        self._grad, self._curl, self._div = _derham.derivatives_as_matrices

        _projectors = _derham.projectors(nquads=self._nq_pr)

        # keys for continuous spaces
        self._V = {'0': 'H1',
                   '1': 'Hcurl',
                   '2': 'Hdiv',
                   '3': 'L2',
                   'v': 'H1vec'}

        self._forms_dict = {'H1': '0_form',
                            'Hcurl': '1_form',
                            'Hdiv': '2_form',
                            'L2': '3_form',
                            'H1vec': 'vector'}

        self._spaces_dict = {'H1': '0',
                             'Hcurl': '1',
                             'Hdiv': '2',
                             'L2': '3',
                             'H1vec': 'v'}

        # Psydac vector space and FEM spline spaces
        self._Vh = {}
        self._Vh_fem = {}
        self._P = {}

        for i, key in enumerate(self._V.keys()):

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

        for i, key in enumerate(self._V.keys()):
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
        self._index_array_domain = self._get_index_array(
            self._domain_decomposition)

        self._index_array_N = self._get_index_array(self._Vh['0'].cart)
        self._index_array_D = self._get_index_array(self._Vh['3'].cart)

        self._neighbours = self._get_neighbours()

        # set polar sub-spaces, polar basis extraction operators, polar DOF extraction operators and boundary operators
        if self.polar_ck == -1:
            ck_blocks = None
        else:
            assert domain is not None
            ck_blocks = PolarExtractionBlocksC1(domain, self)

        self._Vh_pol = {}

        self._B = {}
        self._E = {}

        for i, key in enumerate(self._V.keys()):

            vec_space = self._Vh[key]

            # tensor product case
            if self.polar_ck == -1:

                pol_space = self._Vh[key]

                self._E[key] = IdentityOperator(pol_space)
                P_ex = IdentityOperator(pol_space)

            # C^1 polar spline case
            else:

                pol_space = PolarDerhamSpace(self, self._V[key])

                self._E[key] = PolarExtractionOperator(
                    vec_space, pol_space, ck_blocks.e_ten_to_pol[key])
                P_ex = PolarExtractionOperator(
                    vec_space, pol_space, ck_blocks.p_ten_to_pol[key], ck_blocks.p_ten_to_ten[key])

            self._Vh_pol[key] = pol_space
            self._B[key] = BoundaryOperator(pol_space, self._V[key], self._bc)

            if with_projectors:
                self._P[key] = Projector(
                    self._P[key], P_ex, self._E[key], self._B[key])

        # set discrete derivatives with boundary operators
        if self.polar_ck == 1:
            self._grad = PolarLinearOperator(
                self._Vh_pol['0'], self._Vh_pol['1'], self._grad, ck_blocks.grad_pol_to_ten, ck_blocks.grad_pol_to_pol, ck_blocks.grad_e3)
            self._curl = PolarLinearOperator(
                self._Vh_pol['1'], self._Vh_pol['2'], self._curl, ck_blocks.curl_pol_to_ten, ck_blocks.curl_pol_to_pol, ck_blocks.curl_e3)
            self._div = PolarLinearOperator(
                self._Vh_pol['2'], self._Vh_pol['3'], self._div, ck_blocks.div_pol_to_ten, ck_blocks.div_pol_to_pol, ck_blocks.div_e3)

        self._grad = CompositeLinearOperator(
            self._B['1'], self._grad, self._B['0'].T)
        self._curl = CompositeLinearOperator(
            self._B['2'], self._curl, self._B['1'].T)
        self._div = CompositeLinearOperator(
            self._B['3'], self._div, self._B['2'].T)

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
    def bc(self):
        """ List of boundary conditions in each direction.
        """
        return self._bc

    @property
    def quad_order(self):
        """ List of number of Gauss-Legendre quadrature points in each direction (default = p, = p + 1 points per cell).
        """
        return self._quad_order

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
    def index_array_domain(self):
        """
        A 2d array[int] of shape (comm.Get_size(), 6). The row index denotes the process number and
        for n=0,1,2:

            * arr[i, 2*n + 0] holds the global start index of cells of process i in direction eta_(n+1).
            * arr[i, 2*n + 1] holds the global end index of cells of process i in direction eta_(n+1).
        """
        return self._index_array_domain

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
    def forms_dict(self):
        """ Dictionary containing the names of the continuous spaces and corresponding names of differential forms.
        """
        return self._forms_dict

    @property
    def spaces_dict(self):
        """ Dictionary containing the names of the continuous spaces and corresponding discrete spaces.
        """
        return self._spaces_dict

    @property
    def V(self):
        """ Dictionary containing names of continuous functions spaces (H1, Hcurl, Hdiv, L2 and H1vec).
        """
        return self._V

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
    def E(self):
        """ Dictionary holding basis extraction operators, either IdentityOperator or PolarExtractionOperator.
        """
        return self._E

    @property
    def B(self):
        """ Dictionary holding essential boundary operators (BoundaryOperator).
        """
        return self._B

    @property
    def P(self):
        """ Dictionary holding global commuting projectors (BoundaryOperator).
        """
        return self._P

    @property
    def Vh_pol(self):
        """ Polar sub-spaces, either PolarDerhamSpace (with polar splines) or Stencil-/BlockVectorSpace (same as self.Vh)
        """
        return self._Vh_pol

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
    #      public methods:
    # --------------------------

    def create_buffer_types(self, *datas):
        """
        Creates the buffer types for the ghost region sender. Send types are only the slicing information;
        receving has to be saved in a temporary array and then added to the _data object with the correct indices.
        Buffers have the same structure as struphy.psydac_api.psydac_derham.Derham.neighbours, i.e. a 3d array with shape (3,3,3)
        and are initialized with None. If the process has a neighbour, the send/recv information is filled in.

        Parameters
        ----------
        *datas : np.ndarrays
            The 6d (matrices) or 3d (vectors) _data attributes of StencilMatrices/-Vectors whose ghost regions shall be sent.

        Returns
        -------
        send_types : list
            The send types of ghost regions.

        recv_buf : list
            The receive types of ghost regions.
        """

        send_types = []
        recv_buf = []

        neighbours = self.neighbours

        # pads are the same in all spaces (take V0 here)
        pads = self.Vh['0'].pads

        for k, arg in enumerate(datas):
            for comp, neigh in np.ndenumerate(neighbours):

                send_types.append(np.array([[[None]*3]*3]*3))
                recv_buf.append(np.array([[[None]*3]*3]*3))

                if neigh != -1:
                    send_types[k][comp] = self._create_send_buffer_one_component(
                        pads, arg.shape, comp)
                    recv_buf[k][comp] = self._create_recv_buffer_one_component(
                        pads, arg.shape, comp)

        return send_types, recv_buf

    def send_ghost_regions(self, send_types, recv_types, *datas):
        """
        Communicates the ghost regions between all processes using non-blocking communication.
        In order to avoid communication overhead a sending in one direction component is always accompanied
        by a receiving (if neighbour is not -1) in the inverted direction. This guarantees that every send signal
        is received in the same comp iteration.

        Parameters
        ----------
        send_types : list
            The send types of ghost regions (obtained with self.create_buffer_types()).

        recv_buf : list
            The receive types of ghost regions (obtained with self.create_buffer_types()).

        *datas : np.ndarrays
            The 6d (matrices) or 3d (vectors) _data attributes of StencilMatrices/-Vectors whose ghost regions shall be sent.
        """

        comm = self.comm
        neighbours = self.neighbours

        for dat, send_type, recv_type in zip(datas, send_types, recv_types):

            for comp, send_neigh in np.ndenumerate(neighbours):
                inv_comp = self._invert_component(comp)
                recv_neigh = neighbours[inv_comp]

                if send_neigh != -1:
                    send_type_comp = send_type[comp]
                    # sending to component direction.
                    self._send_ghost_regions_one_component(
                        dat, send_neigh, send_type_comp, comp)

                if recv_neigh != -1:
                    recv_type_comp = recv_type[inv_comp]
                    # Receiving from the inverted component direction if there is a neighbour
                    self._recv_ghost_regions_one_component(
                        dat, recv_neigh, recv_type_comp, comp)

                    if len(dat.shape) == 6:
                        recv_type_comp['buf'][:, :, :, :, :, :] == 0.
                        recv_type_comp['buf'][:, :, :, :, :, :] == 0.
                    elif len(dat.shape) == 3:
                        recv_type_comp['buf'][:, :, :] == 0.
                        recv_type_comp['buf'][:, :, :] == 0.
                    else:
                        raise NotImplementedError(
                            'Unknown shape of data object!')

                comm.Barrier()

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
            # initialize array which will be compared to the rows of index_array_domain:
            # elements with index 2n are the starts and 2n + 1 are the ends.

            neigh_inds = [None]*6

            # in each direction find start/end index for neighbour
            for k, co in enumerate(comp):
                if co == 1:
                    neigh_inds[2*k + 0] = self.index_array_domain[rank, 2*k + 0]
                    neigh_inds[2*k + 1] = self.index_array_domain[rank, 2*k + 1]

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
            index_temp = np.squeeze(self.index_array_domain[:, inds])
            unique_ranks = np.where(
                np.equal(index_temp, neigh_inds[inds]).all(1))[0]

            # if any row satisfies condition, return its index (=rank of neighbour)
            if len(unique_ranks) != 0:
                neigh_id = unique_ranks[0]

        return neigh_id

    def _create_send_buffer_one_component(self, pads, arg_shape, comp):
        """
        creates the send buffer in direction for stencil matrices and vectors. Send buffer is the indexing (MPI.Create_subarray)

        Parameters
        ----------
        pads : list
            contains the paddings in each direction.

        arg_shape : tuple
            called by arg.shape.

        comp : tuple
            component for which the send buffer is to be created; entries are in {0,1,2}.
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

    def _create_recv_buffer_one_component(self, pads, arg_shape, comp):
        """
        creates the receive buffer in direction for stencil matrices. The receive buffer is an empty numpy array
        and the indices where the ghost regions will have to be added to. Left and right are swapped compared to
        send-types since _send_ghost_regions() does the sending component-wise. Sending to the left means 

        Parameters
        ----------
        pads : list
            contains the paddings in each direction.

        arg_shape : tuple
            called by arg.shape.

        comp : tuple
            component for which the send buffer is to be created; entries are in {0,1,2}.
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

    def _send_ghost_regions_one_component(self, dat, neighbour, send_type, comp):
        """
        Does the sending for one direction component using non-blocking communication.

        Parameters
        ----------
        dat : array
            Stencil ._data object; numpy array.

        neighbour : int
            tag of the neighbour or -1 if no neighbour.

        send_type : MPI.Create_subarrays object
            MPI.Create_subarrays object; created by _create_buffer_types().

        comp : tuple
            component direction into which the ghost region is to be sent; entries are in {0,1,2}.
        """

        comm = self.comm
        rank = comm.Get_rank()

        send_tag = rank + 1000*comp[0] + 100*comp[1] + 10*comp[2]

        comm.Isend(
            (dat, 1, send_type), dest=neighbour, tag=send_tag)

    def _recv_ghost_regions_one_component(self, dat, neighbour, recv_type, comp):
        """
        Does the receving for one direction component using non-blocking communication.

        Parameters
        ----------
        dat : array
            Stencil ._data object; numpy array.

        neighbour : int
            tag of the neighbour or -1 if no neighbour.

        recv_type : dict
            dictionary with keys 'buf' and 'inds' and values are numpy arrays; created by _create_buffer_types().

        comp : tuple
            component direction from which the ghost region was sent (is only used for computing the tag); entries are in {0,1,2}.
        """
        from mpi4py import MPI

        comm = self.comm

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
        it on the central component (1,1,1).

        Parameters
        ----------
        comp : tuple
            component index in the 3x3x3 cube; entries are in {0,1,2}.

        Returns
        -------
        res : tuple
            inverse component to input.
        """
        res = [-1, -1, -1]

        for k, co in enumerate(comp):
            if co == 1:
                res[k] = 1
            elif co == 0:
                res[k] = 2
            elif co == 2:
                res[k] = 0
            else:
                raise ValueError('Unknown component value!')

        return tuple(res)
