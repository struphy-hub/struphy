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

    def __init__(self, Nel, p, spl_kind, nq_pr=None, der_as_mat=True, F=None, comm=None):
        '''
        Parameters
        ----------
            Nel: 3-list
                Number of elements in each direction.

            p: 3-list
                Spline degree in each direction.

            spl_kind: 3-list
                Kind of spline in each direction (True=periodic, False=clamped).

            nq_pr: 3-list
                Number of Gauss-Legendre quadrature points in hitopolation (default = p + 1).

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

        if nq_pr is not None: assert len(nq_pr) == 3
        self._nq_pr = nq_pr

        assert isinstance(der_as_mat, bool)
        self._der_as_mat= der_as_mat

        assert isinstance(F, Mapping)
        self._F = F

        self._comm = comm

        # Set defaults
        if nq_pr == None:
            # exact histopolation of products of B-splines
            _nq_pr = [pi + 1 for pi in p]
        else:
            _nq_pr = nq_pr

        if F == None:
            _F = Cube('C', bounds1=(0, 1), bounds2=(
                0, 1), bounds3=(0, 1))  # no mapping
        else:
            _F = F

        self._DF = _F.jacobian
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
            self._domain_log, ncells=Nel, comm=comm)

        # Discrete De Rham
        _derham = discretize(self._derham_symb, self._domain_log_h,
                             degree=p, periodic=self._spl_kind)

        # Psydac spline spaces
        # --------------------
        self._V0 = _derham.V0
        self._V1 = _derham.V1
        self._V2 = _derham.V2
        self._V3 = _derham.V3
        # H1xH1xH1 (needed in pressure coupling for instance)
        self._V0vec = ProductFemSpace(self._V0, self._V0, self._V0)

        # Psydac projectors
        # -----------------
        self._P0, self._P1, self._P2, self._P3 = _derham.projectors(
            nquads=_nq_pr)
        # interpolation in all components
        self._P0vec = Projector_H1vec(self._V0vec)

        # Psydac derivative operators
        # ---------------------------
        if der_as_mat:
            self._grad, self._curl, self._div = _derham.derivatives_as_matrices
        else:
            self._grad, self._curl, self._div = _derham.derivatives_as_operators

        # global indices of non-vanishing splines in each element in format (Nel, p + 1)
        self._breaks = [space.breaks for space in _derham.spaces[0].spaces]

        self._indN_psy = []
        #self._indN3 = []
        for space in _derham.spaces[0].spaces:

            #breaks = space.breaks
            p = space.degree
            knots = space.knots

            tmp = bsp.elements_spans(knots, p)
            tmp_arr = np.empty((tmp.size, p + 1), dtype=int)
            for i in range(p + 1):
                tmp_arr[:, -(i + 1)] = tmp[:] - i
            self._indN_psy += [tmp_arr]

            # tmp3 = []
            # for pt in breaks[:-1]:
            #     tmp3 += [bsp.find_span(knots, p, pt + 1e-8)]

            # assert len(tmp3) == tmp.size

            # for i in range(p + 1):
            #     tmp_arr[:, -(i + 1)] = np.array(tmp3) - i
            # self._indN3 += [tmp_arr]

        self._indD_psy = []
        for space in _derham.spaces[3].spaces:

            p = space.degree
            tmp = bsp.elements_spans(space.knots, p)
            tmp_arr = np.empty((tmp.size, p + 1), dtype=int)

            for i in range(p + 1):
                tmp_arr[:, -(i + 1)] = tmp[:] - i

            self._indD_psy += [tmp_arr]
        
        self._NbaseN = np.array(
            [_derham.spaces[0].spaces[k].nbasis for k in range(3)])
        self._NbaseD = np.array([0, 0, 0])
        for k in range(3):
            if spl_kind[k]:
                self._NbaseD[k] = self._NbaseN[k] 
            else:
                self._NbaseD[k] = self._NbaseN[k] - 1

        self._indN = np.array([(np.indices((_derham.spaces[0].ncells[k], _derham.spaces[0].degree[k] + 1 - 0))[1] + np.arange(_derham.spaces[0].ncells[k])[
                              :, None]) % self._NbaseN[k] for k in range(3)], dtype=object)      # global indices of non-vanishing B-splines on each cell
        self._indD = np.array([(np.indices((_derham.spaces[0].ncells[k], _derham.spaces[0].degree[k] + 1 - 1))[1] + np.arange(_derham.spaces[0].ncells[k])[
                              :, None]) % self._NbaseD[k] for k in range(3)], dtype=object)      # global indices of non-vanishing D-splines on each cell

        # only for M1 Mac users
        PSYDAC_BACKEND_GPYCCEL['flags'] = '-O3 -march=native -mtune=native -ffast-math -ffree-line-length-none'

        # Distribute info on domain decomposition
        if comm is not None:
            self._domain_array, self._index_array = self._get_decomp_arrays()

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
        '''List of number of Gauss-Legendre quadrature points in hitopolation (default = p + 1) in each direction.'''
        return self._nq_pr

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
    def domain_array(self):
        '''A 2d np.array of shape (comm.Get_size, 6). 
            - The row index denotes the process number. 
            - Let n=0,1,2: 
                arr[i, 2*n] holds the LEFT domain boundary of process i in direction eta_(n+1).
                arr[i, 2*n + 1] holds the RIGHT domain boundary of process i in direction eta_(n+1).'''
        return self._domain_array

    @property
    def index_array(self):
        '''A 2d np.array of shape (comm.Get_size, 9). 
            - The row index denotes the process number. 
            - Let n=0,1,2: 
                arr[i, 3*n] holds the global start index of process i in direction eta_(n+1).
                arr[i, 3*n + 1] holds the global end index of process i in direction eta_(n+1).
                arr[i, 3*n + 2] holds the number of cells in the domain of process i in direction eta_(n+1).'''
        return self._index_array

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
            dom_arr_0 : np.array
                A 2d np.array of shape (comm.Get_size, 6). 
                - The row index denotes the process number. 
                - Let n=0,1,2: 
                    arr[i, 2*n] holds the LEFT domain boundary of process i in direction eta_(n+1).
                    arr[i, 2*n + 1] holds the RIGHT domain boundary of process i in direction eta_(n+1).
                    
            ind_arr_0 : np.array
                A 2d np.array of shape (comm.Get_size, 9). 
                - The row index denotes the process number. 
                - Let n=0,1,2: 
                    arr[i, 3*n] holds the global start index of process i in direction eta_(n+1).
                    arr[i, 3*n + 1] holds the global end index of process i in direction eta_(n+1).
                    arr[i, 3*n + 2] holds the number of cells in the domain of process i in direction eta_(n+1).'''

        # mpi info
        nproc = self.comm.Get_size()
        #rank = self.comm.Get_rank()

        # Send buffer
        dom_arr_loc = np.zeros(6, dtype=float)
        # loc_arr_0new = np.zeros(6, dtype=float)
        # loc_arr_3 = np.zeros(6, dtype=float)
        # loc_arr_3new = np.zeros(6, dtype=float)

        ind_arr_loc = np.zeros(9, dtype=int)

        # Main arrays (receive buffers)
        dom_arr_0 = np.zeros(nproc * 6, dtype=float)
        # dom_arr_0new = np.zeros(nproc * 6, dtype=float)
        # dom_arr_3 = np.zeros(nproc * 6, dtype=float)
        # dom_arr_3new = np.zeros(nproc * 6, dtype=float)

        ind_arr_0 = np.zeros(nproc * 9, dtype=int)

        # Get global indices on process (only for B-splines in each direction, D-splines yield same domain, see example_psydac_parallel)
        gl_stas = self.V0.vector_space.starts
        gl_ends = self.V0.vector_space.ends
        gl_pads = self.V0.vector_space.pads

        # Fill local arrays with local domain
        for n, (gl_sta, gl_end, pad, ind_mat, brks, splk) in enumerate(zip(gl_stas, gl_ends, gl_pads, self.indN_psy, self.breaks, self.spl_kind)):

            le, ri, n_cells_loc = index_to_domain(gl_sta, gl_end, pad, ind_mat, brks, splk)
            dom_arr_loc[2*n] = le
            dom_arr_loc[2*n + 1] = ri

            ind_arr_loc[3*n] = gl_sta
            ind_arr_loc[3*n + 1] = gl_end
            ind_arr_loc[3*n + 2] = n_cells_loc

        # for n, (el_sta, el_end, brks) in enumerate(zip(self.V0.local_domain[0], self.V0.local_domain[1], self.breaks)):
        #     loc_arr_0new[2*n] = brks[el_sta]
        #     loc_arr_0new[2*n + 1] = brks[el_end + 1]

        # gl_stas = self.V3.vector_space.starts
        # gl_ends = self.V3.vector_space.ends
        # gl_pads = self.V3.vector_space.pads

        # for n, (gl_sta, gl_end, pad, ind_mat, brks, splk) in enumerate(zip(gl_stas, gl_ends, gl_pads, self.indD_psy, self.breaks, self.spl_kind)):

        #     le, ri = index_to_domain(gl_sta, gl_end, pad, ind_mat, brks, splk)
        #     loc_arr_3[2*n] = le
        #     loc_arr_3[2*n + 1] = ri

        # for n, (el_sta, el_end, brks) in enumerate(zip(self.V3.local_domain[0], self.V3.local_domain[1], self.breaks)):
        #     loc_arr_3new[2*n] = brks[el_sta]
        #     loc_arr_3new[2*n + 1] = brks[el_end + 1]

        # For testing (to be commented out):

        self.comm.Allgather(dom_arr_loc, dom_arr_0)
        # self.comm.Allgather(loc_arr_0new, dom_arr_0new)
        # self.comm.Allgather(loc_arr_3, dom_arr_3)
        # self.comm.Allgather(loc_arr_3new, dom_arr_3new)

        self.comm.Allgather(ind_arr_loc, ind_arr_0)

        # if rank == 0:
        #     print(f'rank {rank} |\n dom_arr_0:\n {dom_arr_0.reshape((nproc, 6))}')#,\n dom_arr_0new:\n {dom_arr_0new.reshape((nproc, 6))},\n dom_arr_3:\n {dom_arr_3.reshape((nproc, 6))},\n dom_arr_3new:\n {dom_arr_3new.reshape((nproc, 6))}')

        return dom_arr_0.reshape(nproc, 6), ind_arr_0.reshape(nproc, 9)

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

    @property
    def NbaseN(self):
        """np.array of B-splines in each direction"""
        return self._NbaseN

    @property
    def NbaseD(self):
        """np.array of D-splines in each direction"""
        return self._NbaseD

    @property
    def indN(self):
        """global indices of non-vanishing B-splines in each direction in each cell"""
        return self._indN

    @property
    def indD(self):
        """global indices of non-vanishing B-splines in each direction in each cell"""
        return self._indD

    @property
    def indN_psy(self):
        """List of psydac global indices in each direction of non-vanishing B-splines in each cell, as 2d np.array of shape (Nel, p + 1)."""
        return self._indN_psy

    @property
    def indD_psy(self):
        """List of psydac global indices in each direction of non-vanishing B-splines in each cell, as 2d np.array of shape (Nel, p + 1)"""
        return self._indD_psy

    @property
    def breaks(self):
        """List of break points (=cell interfaces) in the three directions."""
        return self._breaks


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
            n_cells_loc = n - n1

    return le, ri, n_cells_loc

