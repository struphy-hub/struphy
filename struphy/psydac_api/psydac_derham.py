#!/usr/bin/env python3

from sympde.topology import Cube
from sympde.topology import Derham as Derham_psy

from psydac.api.discretization import discretize
from psydac.fem.vector import ProductFemSpace
from psydac.feec.global_projectors import Projector_H1vec

from struphy.psydac_api.linear_operators import BoundaryOperator, CompositeLinearOperator, IdentityOperator

from struphy.psydac_api.projectors import Projector

from struphy.polar.basic import PolarDerhamSpace
from struphy.polar.extraction_operators import PolarExtractionBlocksC1
from struphy.polar.linear_operators import PolarExtractionOperator, PolarLinearOperator

import numpy as np


class Derham:
    """
    Psydac API for the discrete Derham sequence on the logical unit cube (3d): Polar sub-spaces can be added.
    
         V0 ----- d0 ----> V1 ----- d1 ----> V2 ----- d1 ----> V3
          |                 |                 |                 |
          |                 |                 |                 |
          | P0              | P1              | P2              | P3
          |                 |                 |                 |
          v                 v                 v                 v
         V0h ---- d0 ----> V1h ---- d1 ----> V2h ---- d2 ----> V3h
          ^                 ^                 ^                 ^
          |                 |                 |                 |
          | (E0)^T          | (E1)^T          | (E2)^T          | (E3)^T
          |                 |                 |                 |
          |                 |                 |                 |
       V0h_pol -- d0 --> V1h_pol -- d1 --> V2h_pol -- d2 --> V3h_pol
       
    In above diagram, d0 <-> grad, d1 <-> curl and d2 <-> div. If polar sub-spaces are added, Pk (k = 0,1,2,3) map to Vkh_pol.
    
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

    with_projectors : bool
        Whether to add global commuting projectors to the diagram.

    polar_ck : int
        Smoothness at a polar singularity at eta_1=0 (default -1 : standard tensor product splines, OR 1 : C1 polar splines)

    domain : struphy.geometry.domains
        Mapping from logical unit cube to physical domain (only needed in case of polar splines polar_ck=1).
    """

    def __init__(self, Nel, p, spl_kind, bc=None, quad_order=None, nq_pr=None, comm=None, with_projectors=True, polar_ck=-1, domain=None):
 
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
            if spl_kind[0]: assert bc[0][0] is None and bc[0][1] is None
            if spl_kind[1]: assert bc[1][0] is None and bc[1][1] is None
            if spl_kind[2]: assert bc[2][0] is None and bc[2][1] is None
            
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
            self._domain_log, ncells=Nel, comm=self._comm)

        # Psydac discrete de Rham, projectors and derivatives (as Stencil-/BlockMatrix)
        _derham = discretize(self._derham_symb, self._domain_log_h,
                             degree=self.p, periodic=self.spl_kind, quad_order=self.quad_order)
        
        self._grad, self._curl, self._div = _derham.derivatives_as_matrices
        
        _projectors = _derham.projectors(nquads=self._nq_pr)

        # keys for continuous spaces
        self._V = {'0' : 'H1', 
                   '1' : 'Hcurl', 
                   '2' : 'Hdiv', 
                   '3' : 'L2', 
                   'v' : 'H1vec'}
        
        self._forms_dict = {'H1'    : '0_form', 
                            'Hcurl' : '1_form', 
                            'Hdiv'  : '2_form', 
                            'L2'    : '3_form', 
                            'H1vec' : 'vector'}
        
        self._spaces_dict = {'H1'    : '0', 
                             'Hcurl' : '1', 
                             'Hdiv'  : '2', 
                             'L2'    : '3', 
                             'H1vec' : 'v'}
        
        # Psydac vector space and FEM spline spaces
        self._Vh = {}
        self._Vh_fem = {}
        self._P = {}
        
        for i, key in enumerate(self._V.keys()):
            
            if key == 'v':
                self._Vh_fem[key] = ProductFemSpace(_derham.V0, _derham.V0, _derham.V0)
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
                self._nbasis[key] = [space.nbasis for space in fem_space.spaces]
                self._spline_types[key] = [space.basis for space in fem_space.spaces]
                self._spline_types_pyccel[key] = np.array([int(space.basis == 'M') for space in fem_space.spaces])
            else:
                self._nbasis[key] = [[space.nbasis for space in comp.spaces] for comp in fem_space.spaces]
                self._spline_types[key] = [[space.basis for space in comp.spaces] for comp in fem_space.spaces]
                self._spline_types_pyccel[key] = [np.array([int(space.basis == 'M') for space in comp.spaces]) for comp in fem_space.spaces]
            
        # break points
        self._breaks = [space.breaks for space in _derham.spaces[0].spaces]
        
        # index arrays
        self._indN = [(np.indices((space.ncells, space.degree + 1))[1] + np.arange(space.ncells)[:, None])%space.nbasis for space in self._Vh_fem['0'].spaces]
        self._indD = [(np.indices((space.ncells, space.degree + 1))[1] + np.arange(space.ncells)[:, None])%space.nbasis for space in self._Vh_fem['3'].spaces]
        
        # distribute info on domain decomposition
        self._domain_array, self._index_array_N, self._index_array_D = self._get_decomp_arrays()
        if comm is not None:
            self._neighbours = self._get_neighbours()
        
        # set polar sub-spaces, polar basis extraction operators, polar DOF extraction operators and boundary operators
        if self.polar_ck == -1:
            ck_blocks = None
        else:
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
                
                self._E[key] = PolarExtractionOperator(vec_space, pol_space, ck_blocks.e_ten_to_pol[key])
                P_ex = PolarExtractionOperator(vec_space, pol_space, ck_blocks.p_ten_to_pol[key], ck_blocks.p_ten_to_ten[key])
                
            self._Vh_pol[key] = pol_space
            self._B[key] = BoundaryOperator(pol_space, self._V[key], self._bc)
            
            if with_projectors:
                self._P[key] = Projector(self._P[key], P_ex, self._E[key], self._B[key])
        
        # set discrete derivatives with boundary operators
        if self.polar_ck == 1:
            self._grad = PolarLinearOperator(self._Vh_pol['0'], self._Vh_pol['1'], self._grad, ck_blocks.grad_pol_to_ten, ck_blocks.grad_pol_to_pol, ck_blocks.grad_e3)
            self._curl = PolarLinearOperator(self._Vh_pol['1'], self._Vh_pol['2'], self._curl, ck_blocks.curl_pol_to_ten, ck_blocks.curl_pol_to_pol, ck_blocks.curl_e3)
            self._div  = PolarLinearOperator(self._Vh_pol['2'], self._Vh_pol['3'], self._div , ck_blocks.div_pol_to_ten , ck_blocks.div_pol_to_pol , ck_blocks.div_e3 )
            
        self._grad = CompositeLinearOperator(self._B['1'], self._grad, self._B['0'].transpose())
        self._curl = CompositeLinearOperator(self._B['2'], self._curl, self._B['1'].transpose())
        self._div  = CompositeLinearOperator(self._B['3'], self._div , self._B['2'].transpose())      
   
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
        """ Dictionary containing FEM spline spaces (TensorFem-/ProductFemSpace).
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

        
    def _get_decomp_arrays(self):
        """
        Uses mpi.Allgather to distribute information on domain decomposition to all processes.

        Returns
        -------
            dom_arr : array[float]
                A 2d array[float] of shape (comm.Get_size(), 9). 
                    - The row index denotes the process number. 
                    - Let n=0,1,2 : 
                        arr[i, 3*n + 0] holds the LEFT domain boundary of process i in direction eta_(n+1).
                        arr[i, 3*n + 1] holds the RIGHT domain boundary of process i in direction eta_(n+1).
                        arr[i, 3*n + 2] holds the number of cells of process i in direction eta_(n+1).
                    
            ind_arr_0 : array[int]
                A 2d array[int] of shape (comm.Get_size(), 6). 
                    - The row index denotes the process number.
                    - Let n=0,1,2 :
                        arr[i, 2*n + 0] holds the global start index of B-splines (N) of process i in direction eta_(n+1).
                        arr[i, 2*n + 1] holds the global end index of B-splines (N) of process i in direction eta_(n+1).

            ind_arr_3 : array[int]
                A 2d array[int] of shape (comm.Get_size(), 6). 
                    - The row index denotes the process number.
                    - Let n=0,1,2 :
                        arr[i, 2*n + 0] holds the global start index of M-splines (D) of process i in direction eta_(n+1).
                        arr[i, 2*n + 1] holds the global end index of M-splines (D) of process i in direction eta_(n+1).
        """

        # mpi info
        if self.comm is not None:
            nproc = self.comm.Get_size()
        else:
            nproc = 1
        #rank = self.comm.Get_rank()

        # Send buffer
        dom_arr_loc = np.zeros(9, dtype=float)
        ind_arr_0_loc = np.zeros(6, dtype=int)
        ind_arr_3_loc = np.zeros(6, dtype=int)

        # Main arrays (receive buffers)
        dom_arr = np.zeros(nproc * 9, dtype=float)
        ind_arr_0 = np.zeros(nproc * 6, dtype=int)
        ind_arr_3 = np.zeros(nproc * 6, dtype=int)
        
        # get V0 and V3 FEM spaces
        V0 = self.Vh_fem['0']
        V3 = self.Vh_fem['3']

        # Get process info
        starts_0 = V0.vector_space.starts
        ends_0 = V0.vector_space.ends

        starts_3 = V3.vector_space.starts
        ends_3 = V3.vector_space.ends

        # Fill local domain array
        for n, (el_sta, el_end, brks) in enumerate(zip(V0.local_domain[0], V0.local_domain[1], self.breaks)):

            dom_arr_loc[3*n] = brks[el_sta]
            dom_arr_loc[3*n + 1] = brks[el_end + 1]
            dom_arr_loc[3*n + 2] = el_end - el_sta + 1

        # Fill local index arrays 
        for n, (gl_sta, gl_end, brks) in enumerate(zip(starts_0, ends_0, self.breaks)):

            ind_arr_0_loc[2*n] = gl_sta
            ind_arr_0_loc[2*n + 1] = gl_end

        for n, (gl_sta, gl_end, brks) in enumerate(zip(starts_3, ends_3, self.breaks)):

            ind_arr_3_loc[2*n] = gl_sta
            ind_arr_3_loc[2*n + 1] = gl_end

        # Distribute
        if self.comm is not None:
            self.comm.Allgather(dom_arr_loc, dom_arr)
            self.comm.Allgather(ind_arr_0_loc, ind_arr_0)
            self.comm.Allgather(ind_arr_3_loc, ind_arr_3)
        else:
            dom_arr = dom_arr_loc
            ind_arr_0 = ind_arr_0_loc
            ind_arr_3 = ind_arr_3_loc

        return dom_arr.reshape(nproc, 9), ind_arr_0.reshape(nproc, 6), ind_arr_3.reshape(nproc, 6)

    def _get_neighbours(self):
        """
        For each mpi process, compute the 26 neighbouring processes (3x3x3 cube except the most inner element).
        This is done in terms of N-spline start/end indices.

        Returns
        -------
            neighbours : array[int]
                A 3d array[int] of shape (3,3,3).
                The i-th axis is the direction eta_(i+1). Neighbours along the faces have index with two 1s,
                neighbours along the edges only have one 1, neighbours along the edges have no 1 in the index.

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
        """

        neighs = np.empty( (3,3,3), dtype=int )

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    comp = [i,j,k]
                    ind = tuple(comp)
                    neighs[ind] = self._get_neigh_1_comp(comp)
        
        return neighs

    def _get_neigh_1_comp(self, comp):
        """
        Computes the process id of a neighbour in direction of comp (c.f. _neighbours).

        Parameters
        ----------
            comp : list
                list with 3 entries

        Returns
        -------
            res : int
                id of neighbouring process
        """
        assert len(comp) == 3
        
        # get V0 and V3 FEM spaces
        V0 = self.Vh_fem['0']
        V3 = self.Vh_fem['3']

        # Get space info
        dims = [space.nbasis for space in V0.spaces]
        index_arr = self.index_array_N
        kinds = self.spl_kind
        
        # Get process info
        starts = V0.vector_space.starts
        ends = V0.vector_space.ends

        # Get communicator info
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()

        res = -1

        # central component is always the process itself
        if comp == [1,1,1]:
            return res

        comp = np.array(comp)
        kinds = np.array(kinds)

        # if only one process: check if comp is neighbour in non-peridic directions, if this is not the case then return the rank as neighbour id
        if size == 1:
            if (comp[kinds == False] == 1).all():
                return rank

        # multiple processes
        else:
            # initialize array which will be compared to the rows of index_arr; elements with index 2n are the starts and 2n+1 are the ends.
            neigh_inds = [None]*6
            
            # in each direction find start/end index for neighbour
            for k, co in enumerate(comp):
                if co == 1:
                    neigh_inds[2*k] = index_arr[rank, 2*k]
                    neigh_inds[2*k+1] = index_arr[rank, 2*k+1]
                
                elif co == 0:
                    neigh_inds[2*k+1] = starts[k] - 1
                    if kinds[k]:
                        neigh_inds[2*k+1] %= dims[k]

                elif co == 2:
                    neigh_inds[2*k] = ends[k] + 1
                    if kinds[k]:
                        neigh_inds[2*k] %= dims[k]

                else:
                    raise ValueError('Wrong value for component; must be 0 or 1 or 2 !')
            
            neigh_inds = np.array(neigh_inds)

            # only use indices where information is present to find the neighbours rank
            inds = np.where(neigh_inds != None)

            # find ranks (row index of index_arr) which agree in start/end indices
            index_temp = np.squeeze(index_arr[:, inds])
            unique_ranks = np.where( np.equal( index_temp, neigh_inds[inds] ).all(1) )[0]

            # if any row satisfies condition, return its index (=rank of neighbour)
            if len(unique_ranks) != 0:
                res = unique_ranks[0]
        
        return res
    
    


def index_to_domain(gl_start, gl_end, pad, ind_mat, breaks):
    """
    Transform the psydac decomposition of spline indices into a domain decomposition (1d).

    Parameters
    ----------
        gl_start : int
            Global start index on mpi process.

        gl_end : int
            Global end index on mpi process.

        pad : int
            Padding on mpi process (size of ghost region in spline coeffs).

        ind_mat : array[int]
            2d array of shape (Nel, p + 1) of indices of non-vanishing splines in each element (or cell).
            From Derham.indN_psy or Derham.indD_psy.

        breaks : list
            Break points (=cell interfaces) in [0, 1].

    Returns
    -------
        Left and right boundary [le, ri] of local 1d domain.
    """

    # Is it a B- or a D-spline?
    is_D_spline = False
    if ind_mat.shape[1] == pad:
        is_D_spline = True

    ind_le = gl_start
    ind_ri = gl_end + pad - is_D_spline
    if ind_ri > np.amax(ind_mat):
        ind_ri -= pad - is_D_spline

    assert ind_le < ind_ri

    le = None
    ri = None
    for n in range(ind_mat.shape[0]):

        if ind_le == ind_mat[n, 0]:
            le = breaks[n]
            n1 = n

        if ind_ri == ind_mat[n, -1]:
            ri = breaks[n + 1]
            n_cells_loc = n - n1 + 1

    return le, ri, n_cells_loc

