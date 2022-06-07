#!/usr/bin/env python3

from psydac.api.discretization import discretize
from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL
from psydac.fem.vector import ProductFemSpace
import psydac.core.bsplines as bsp

from sympde.topology import elements_of
from sympde.expr import BilinearForm, integral
from sympde.calculus import dot
from sympde.topology import Cube, Derham
from sympde.topology.mapping import Mapping

from sympy import sqrt

from struphy.psydac_api.H1vec_psydac import Projector_H1vec

import numpy as np
from mpi4py import MPI


class DerhamBuild:
    '''Psydac API for discrete Derham sequence on the logical domain, and mass matrices.'''

    def __init__(self, Nel, p, spl_kind, nq_pr=None, quad_order=None, der_as_mat=True, F=None, comm=None):
        '''
        Parameters
        ----------
            Nel: list[int]
                Number of elements in each direction.

            p: list[int]
                Spline degree in each direction.

            spl_kind: list[boolean]
                Kind of spline in each direction (True=periodic, False=clamped).

            nq_pr: list[int]
                Number of Gauss-Legendre quadrature points in histopolation in each direction (default = p + 1).
                
            quad_order: list[int]
                Degree of Gauss-Legendre quadrature in each direction (default = p, leads to p + 1 quadrature points per cell).

            der_as_mat: boolean
                Whether derivatives are returned as matrices (True) or operators (False).

            F: Psydac symbolic mapping
                The mapping from logical to physical space.

            comm: mpi_comm'''

        # Input parameters:
        assert len(Nel) == 3
        self._Nel = Nel

        assert len(p) == 3
        self._p = p

        assert len(spl_kind) == 3
        self._spl_kind = spl_kind

        assert isinstance(der_as_mat, bool)
        self._der_as_mat= der_as_mat

        self._comm = comm

        if nq_pr == None:
            # exact histopolation of products of B-splines
            self._nq_pr = [pi + 1 for pi in p]
        else:
            assert len(nq_pr) == 3
            self._nq_pr = nq_pr
            
        if quad_order == None:
            # exact integration of products of B-splines
            self._quad_order = [pi for pi in p]
        else:
            assert len(quad_order) == 3
            self._quad_order = quad_order

        if F == None:
            self._F = Cube('C', bounds1=(0, 1), bounds2=(
                0, 1), bounds3=(0, 1))  # no mapping
        else:
            assert isinstance(F, Mapping)
            self._F = F

        self._DF = self._F.jacobian
        self._sqrt_g = sqrt((self._DF.T*self._DF).det())
        self._DFinv = self._DF.inv()

        # Psydac symbolic logical domain
        self._domain_log = Cube('C', bounds1=(
            0, 1), bounds2=(0, 1), bounds3=(0, 1))

        # Psydac symbolic Derham
        self._derham_symb = Derham(self._domain_log)
        
        # Discrete logical domain
        # logical domain, the parallelism is initiated here.
        self._domain_log_h = discretize(
            self._domain_log, ncells=Nel, comm=self._comm)

        # Discrete De Rham
        _derham = discretize(self._derham_symb, self._domain_log_h,
                             degree=self.p, periodic=self.spl_kind, quad_order=self.quad_order)

        # Psydac spline spaces
        self._V0 = _derham.V0
        self._V1 = _derham.V1
        self._V2 = _derham.V2
        self._V3 = _derham.V3
        # H1xH1xH1 (needed in pressure coupling for instance)
        self._V0vec = ProductFemSpace(self._V0, self._V0, self._V0)

        # Psydac projectors
        self._P0, self._P1, self._P2, self._P3 = _derham.projectors(
            nquads=self.nq_pr)
        # interpolation in all components
        self._P0vec = Projector_H1vec(self._V0vec)

        # Psydac derivative operators
        if der_as_mat:
            self._grad, self._curl, self._div = _derham.derivatives_as_matrices
        else:
            self._grad, self._curl, self._div = _derham.derivatives_as_operators

        # Break points
        self._breaks = [space.breaks for space in _derham.spaces[0].spaces]

        # only for M1 Mac users
        PSYDAC_BACKEND_GPYCCEL['flags'] = '-O3 -march=native -mtune=native -ffast-math -ffree-line-length-none'

        # Distribute info on domain decomposition
        if comm is not None:
            self._domain_array, self._index_array_N, self._index_array_D = self._get_decomp_arrays()
            self._neighbours = self._get_neighbours()

    @property
    def Nel(self):
        '''List of number of elements (=cells) in each direction.'''
        return self._Nel

    @property
    def p(self):
        '''List of spline degrees in each direction.'''
        return self._p

    @property
    def spl_kind(self):
        '''List of spline type (periodic=True or clamped=False) in each direction.'''
        return self._spl_kind

    @property
    def nq_pr(self):
        '''List of number of Gauss-Legendre quadrature points in histopolation (default = p + 1) in each direction.'''
        return self._nq_pr
    
    @property
    def quad_order(self):
        '''List of number of Gauss-Legendre quadrature points in each direction (default = p, leads to p + 1 points per cell).'''
        return self._quad_order

    @property
    def der_as_mat(self):
        '''Whether derivatives are returned as matrices (True) or operators (False).'''
        return self._der_as_mat

    @property
    def F(self):
        '''Psydac mapping used in mass matrices.'''
        return self._F

    @property
    def comm(self):
        '''MPI communicator.'''
        return self._comm

    @property
    def breaks(self):
        """List of break points (=cell interfaces) in the three directions."""
        return self._breaks

    @property
    def domain_array(self):
        '''A 2d np.array of shape (comm.Get_size, 6). 
            - The row index denotes the process number. 
            - Let n=0,1,2: 
                arr[i, 2*n] holds the LEFT domain boundary of process i in direction eta_(n+1).
                arr[i, 2*n + 1] holds the RIGHT domain boundary of process i in direction eta_(n+1).'''
        return self._domain_array

    @property
    def index_array_N(self):
        '''A 2d np.array of shape (comm.Get_size, 6). 
            - The row index denotes the process number. 
            - arr[i, 2*n] holds the global start index of N-splines of process i in direction eta_(n+1).
            - arr[i, 2*n + 1] holds the global end index of N-splines of process i in direction eta_(n+1).'''
        return self._index_array_N

    @property
    def index_array_D(self):
        '''A 2d np.array of shape (comm.Get_size, 6). 
            - The row index denotes the process number. 
            - arr[i, 2*n] holds the global start index of D-splines of process i in direction eta_(n+1).
            - arr[i, 2*n + 1] holds the global end index of D-splines of process i in direction eta_(n+1).'''
        return self._index_array_D

    @property
    def neighbours(self):
        '''A 1d np.array of shape (6). 
                - arr[2*n] holds the left neighbouring process of process i in direction eta_(n+1).
                - arr[2*n + 1] holds the right neighbouring of process i in direction eta_(n+1).
                Values are -1 if process is at a boundary.'''
        return self._neighbours

    def assemble_M0(self):
        '''Assemble mass matrix for L2-scalar product in V0.'''

        _u0, _v0 = elements_of(self._derham_symb.V0, names='u0, v0')

        _a0 = BilinearForm((_u0, _v0), integral(
            self._domain_log, _u0 * _v0 * self._sqrt_g))

        self._a0_h = discretize(
            _a0, self._domain_log_h, (self._V0, self._V0), backend=PSYDAC_BACKEND_GPYCCEL)

        self._M0 = self._a0_h.assemble()

    def assemble_M1(self):

        _u1, _v1 = elements_of(self._derham_symb.V1, names='u1, v1')

        _a1 = BilinearForm((_u1, _v1), integral(
            self._domain_log, dot(self._DFinv.T*_u1, self._DFinv.T*_v1) * self._sqrt_g))

        self._a1_h = discretize(
            _a1, self._domain_log_h, (self._V1, self._V1), backend=PSYDAC_BACKEND_GPYCCEL)

        self._M1 = self._a1_h.assemble()

    def assemble_M2(self):

        _u2, _v2 = elements_of(self._derham_symb.V2, names='u2, v2')

        _a2 = BilinearForm((_u2, _v2), integral(
            self._domain_log, dot(self._DF*_u2, self._DF*_v2) / self._sqrt_g))

        self._a2_h = discretize(
            _a2, self._domain_log_h, (self._V2, self._V2), backend=PSYDAC_BACKEND_GPYCCEL)

        self._M2 = self._a2_h.assemble()

    def assemble_M3(self):

        _u3, _v3 = elements_of(self._derham_symb.V3, names='u3, v3')

        _a3 = BilinearForm((_u3, _v3), integral(
            self._domain_log, _u3 * _v3 / self._sqrt_g))

        self._a3_h = discretize(
            _a3, self._domain_log_h, (self._V3, self._V3), backend=PSYDAC_BACKEND_GPYCCEL)

        self._M3 = self._a3_h.assemble()

    def _get_decomp_arrays(self):
        '''Uses mpi.Allgather to distribute information on domain decomposition to all processes.

        Returns
        -------
            dom_arr : np.array
                A 2d np.array of shape (comm.Get_size, 6). 
                - The row index denotes the process number. 
                - arr[i, 2*n] holds the LEFT domain boundary of process i in direction eta_(n+1).
                - arr[i, 2*n + 1] holds the RIGHT domain boundary of process i in direction eta_(n+1).
                    
            ind_arr_0 : np.array
                A 2d np.array of shape (comm.Get_size, 6). 
                - The row index denotes the process number. 
                - arr[i, 2*n] holds the global start index of N-splines of process i in direction eta_(n+1).
                - arr[i, 2*n + 1] holds the global end index of N-splines of process i in direction eta_(n+1).

            ind_arr_3 : np.array
                Same as ind_arr_0 but for D-splines.
        '''

        # mpi info
        nproc = self.comm.Get_size()
        #rank = self.comm.Get_rank()

        # Send buffer
        dom_arr_loc = np.zeros(9, dtype=float)
        ind_arr_0_loc = np.zeros(6, dtype=int)
        ind_arr_3_loc = np.zeros(6, dtype=int)

        # Main arrays (receive buffers)
        dom_arr = np.zeros(nproc * 9, dtype=float)
        ind_arr_0 = np.zeros(nproc * 6, dtype=int)
        ind_arr_3 = np.zeros(nproc * 6, dtype=int)

        # Get process info
        starts_0 = self.V0.vector_space.starts
        ends_0 = self.V0.vector_space.ends

        starts_3 = self.V3.vector_space.starts
        ends_3 = self.V3.vector_space.ends

        # Fill local domain array
        for n, (el_sta, el_end, brks) in enumerate(zip(self.V0.local_domain[0], self.V0.local_domain[1], self.breaks)):

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
        self.comm.Allgather(dom_arr_loc, dom_arr)
        self.comm.Allgather(ind_arr_0_loc, ind_arr_0)
        self.comm.Allgather(ind_arr_3_loc, ind_arr_3)

        return dom_arr.reshape(nproc, 9), ind_arr_0.reshape(nproc, 6), ind_arr_3.reshape(nproc, 6)

    def _get_neighbours(self):
        '''For each mpi process, compute the 6 neighbouring processes (two in each direction eta_n).
        This is done in terms of N-spline start/end indices.

        Returns
        -------
            neighbours : np.array
                A 1d np.array of shape (6). 
                - arr[2*n] holds the left neighbouring process of current process in direction eta_(n+1).
                - arr[2*n + 1] holds the right neighbouring of current process in direction eta_(n+1).
                Value is -1 if no neighbour in that direction.
        '''

        # Get space info
        dims = [space.nbasis for space in self.V0.spaces]

        # Get process info
        starts = self.V0.vector_space.starts
        ends = self.V0.vector_space.ends

        neighbours = -1 * np.ones(6, dtype=int)

        for n, (start, end, dim, kind) in enumerate(zip(starts, ends, dims, self.spl_kind)):

            # start/end indices of the right/left neighbours
            neigh_start = end + 1
            neigh_end = start - 1
            if kind: 
                neigh_start %= dim
                neigh_end %= dim

            # get process starts/ends in the other two directions
            n_p = (n + 1)%3
            n_m = (n - 1)%3 
            start_p = starts[n_p]
            start_m = starts[n_m]
            end_p = ends[n_p]
            end_m = ends[n_m]

            #if self.comm.Get_rank() == 0: print(f'n={n}, n_p={n_p}, n_m={n_m}, dim={dim}, neigh_start={neigh_start}, neigh_end={neigh_end}')

            for i, inds in enumerate(self.index_array_N):

                #if self.comm.Get_rank() == 0: print(f'process={i}, inds2n={inds[2*n]}, inds2n1={inds[2*n + 1]}')

                # right neighbour
                if inds[2*n] == neigh_start and inds[2*n_p] == start_p and inds[2*n_m] == start_m:

                    neighbours[2*n + 1] = i
                    # process cannot be a neighbour to itself
                    if i == self.comm.Get_rank():
                        neighbours[2*n + 1] = -1
                    
                # left neighbour
                if inds[2*n + 1] == neigh_end and inds[2*n_p + 1] == end_p and inds[2*n_m + 1] == end_m:

                    neighbours[2*n] = i
                    # process cannot be a neighbour to itself
                    if i == self.comm.Get_rank():
                        neighbours[2*n] = -1

        return neighbours

    @property
    def V0(self):
        '''Discrete H1 space.'''
        return self._V0

    @property
    def V1(self):
        '''Discrete H(curl) space.'''
        return self._V1

    @property
    def V2(self):
        '''Discrete H(div) space.'''
        return self._V2

    @property
    def V3(self):
        '''Discrete L2 space.'''
        return self._V3

    @property
    def V0vec(self):
        '''Discrete H1xH1xH1 space.'''
        return self._V0vec

    @property
    def P0(self):
        '''Interpolation into discrete H1 space.'''
        return self._P0

    @property
    def P1(self):
        '''Inter-/histopolation into discrete H(curl) space.'''
        return self._P1

    @property
    def P2(self):
        '''Inter-/histopolation into discrete H(div) space.'''
        return self._P2

    @property
    def P3(self):
        '''Histopolation into discrete L2 space.'''
        return self._P3

    @property
    def P0vec(self):
        '''Interpolation into discrete H1xH1xH1 space.'''
        return self._P0vec

    @property
    def grad(self):
        '''Gradient H1 -> H(curl).'''
        return self._grad

    @property
    def curl(self):
        '''Curl H(curl) -> H(div).'''
        return self._curl

    @property
    def div(self):
        '''Divergence H(div) -> L2.'''
        return self._div

    @property
    def M0(self):
        '''Mass matrix for L2-scalar product in V0.'''
        if hasattr(self, '_M0'):
            return self._M0
        else:
            raise AttributeError('M0 not assembled.')

    @property
    def M1(self):
        '''Mass matrix for L2-scalar product in V1.'''
        if hasattr(self, '_M1'):
            return self._M1
        else:
            raise AttributeError('M1 not assembled.')

    @property
    def M2(self):
        '''Mass matrix for L2-scalar product in V2.'''
        if hasattr(self, '_M2'):
            return self._M2
        else:
            raise AttributeError('M2 not assembled.')

    @property
    def M3(self):
        '''Mass matrix for L2-scalar product in V3.'''
        if hasattr(self, '_M3'):
            return self._M3
        else:
            raise AttributeError('M3 not assembled.')


def index_to_domain(gl_start, gl_end, pad, ind_mat, breaks, spl_kind):
    '''Transform the psydac decomposition of spline indices into a domain decomposition (1d).

    Parameters
    ----------
        gl_start : int
            Global start index on mpi process.

        gl_end : int
            Global end index on mpi process.

        pad : int
            Padding on mpi process (size of ghost region in spline coeffs).

        ind_mat : np.array
            2d array of shape (Nel, p + 1) of indices of non-vanishing splines in each element (or cell).
            From DerhamBuild.indN_psy or DerhamBuild.indD_psy.

        breaks : list
            Break points (=cell interfaces) in [0, 1].

    Returns
    -------
        Left and right boundary [le, ri] of local 1d domain.'''

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

