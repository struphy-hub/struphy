#!/usr/bin/env python3

from psydac.api.discretization import discretize
from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL
from psydac.fem.vector import ProductFemSpace

from psydac.core.bsplines import elevate_knots
from psydac.utilities.utils import unroll_edges

from sympde.topology import elements_of
from sympde.expr import BilinearForm, integral
from sympde.calculus import dot
from sympde.topology import Cube
from sympde.topology import Derham as Derham_psy
from sympde.topology.mapping import Mapping

from sympy import sqrt

#from struphy.psydac_api.global_projectors import Projector_H1, Projector_Hcurl, Projector_Hdiv, Projector_L2, Projector_H1vec
from struphy.psydac_api.H1vec_psydac import Projector_H1vec

from struphy.psydac_api.mass_psydac import get_mass

import struphy.feec.bsplines as bsp

import numpy as np
from mpi4py import MPI


class Derham:
    """
    Psydac API for 
    
    1. the discrete Derham sequence on the logical unit cube (3d)
    2. corresponding mass matrices from mapping F
    
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

        comm : Intracomm
            MPI communicator from mpi4py.MPI.Intracomm.
    """

    def __init__(self, Nel, p, spl_kind, nq_pr=None, quad_order=None, der_as_mat=True, F=None, comm=None):
 
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
        self._derham_symb = Derham_psy(self._domain_log)
        
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

        # index arrays
        self._indN = [(np.indices((space.ncells, space.degree + 1))[1] + np.arange(space.ncells)[:, None])%space.nbasis for space in self._V0.spaces]
        self._indD = [(np.indices((space.ncells, space.degree + 1))[1] + np.arange(space.ncells)[:, None])%space.nbasis for space in self._V3.spaces]

        # only for M1 Mac users
        PSYDAC_BACKEND_GPYCCEL['flags'] = '-O3 -march=native -mtune=native -ffast-math -ffree-line-length-none'

        # Distribute info on domain decomposition
        if comm is not None:
            self._domain_array, self._index_array_N, self._index_array_D = self._get_decomp_arrays()
            self._neighbours = self._get_neighbours()

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
    def nq_pr(self):
        """ List of number of Gauss-Legendre quadrature points in histopolation (default = p + 1) in each direction.
        """
        return self._nq_pr
    
    @property
    def quad_order(self):
        """ List of number of Gauss-Legendre quadrature points in each direction (default = p, = p + 1 points per cell).
        """
        return self._quad_order

    @property
    def der_as_mat(self):
        """ Whether derivatives are returned as matrices (True) or operators (False).
        """
        return self._der_as_mat

    @property
    def F(self):
        """ Psydac mapping used in mass matrices.
        """
        return self._F

    @property
    def comm(self):
        """ MPI communicator.
        """
        return self._comm

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
        A 1d array[int] of shape (6,). For n=0,1,2:
        
            * arr[2*n + 0] holds the LEFT neighbouring process of process in direction eta_(n+1).
            * arr[2*n + 1] holds the RIGHT neighbouring of process in direction eta_(n+1).

        Values are -1 if process is at a domain boundary (non-periodic case).
        """
        return self._neighbours

    def assemble_M0(self):
        """ Assemble mass matrix for L2-scalar product in V0.
        """

        print('Assembling M0 ...')
        _u0, _v0 = elements_of(self._derham_symb.V0, names='u0, v0')

        _a0 = BilinearForm((_u0, _v0), integral(
            self._domain_log, _u0 * _v0 * self._sqrt_g))

        self._a0_h = discretize(
            _a0, self._domain_log_h, (self._V0, self._V0), backend=PSYDAC_BACKEND_GPYCCEL)

        self._M0 = self._a0_h.assemble()
        print('Done.')

    def assemble_M1(self):
        """ Assemble mass matrix for L2-scalar product in V1.
        """

        print('Assembling M1 ...')
        _u1, _v1 = elements_of(self._derham_symb.V1, names='u1, v1')

        _a1 = BilinearForm((_u1, _v1), integral(
            self._domain_log, dot(self._DFinv.T*_u1, self._DFinv.T*_v1) * self._sqrt_g))

        self._a1_h = discretize(
            _a1, self._domain_log_h, (self._V1, self._V1), backend=PSYDAC_BACKEND_GPYCCEL)

        self._M1 = self._a1_h.assemble()
        print('Done.')

    def assemble_M2(self):
        """ Assemble mass matrix for L2-scalar product in V2.
        """

        print('Assembling M2 ...')
        _u2, _v2 = elements_of(self._derham_symb.V2, names='u2, v2')

        _a2 = BilinearForm((_u2, _v2), integral(
            self._domain_log, dot(self._DF*_u2, self._DF*_v2) / self._sqrt_g))

        self._a2_h = discretize(
            _a2, self._domain_log_h, (self._V2, self._V2), backend=PSYDAC_BACKEND_GPYCCEL)

        self._M2 = self._a2_h.assemble()
        print('Done.')

    def assemble_M3(self):
        """ Assemble mass matrix for L2-scalar product in V3.
        """

        print('Assembling M3 ...')
        _u3, _v3 = elements_of(self._derham_symb.V3, names='u3, v3')

        _a3 = BilinearForm((_u3, _v3), integral(
            self._domain_log, _u3 * _v3 / self._sqrt_g))

        self._a3_h = discretize(
            _a3, self._domain_log_h, (self._V3, self._V3), backend=PSYDAC_BACKEND_GPYCCEL)

        self._M3 = self._a3_h.assemble()
        print('Done.')
            
    def assemble_M0_nonsymb(self, domain):
        """ 
        Assemble mass matrix for L2-scalar product in V0 without psydac's symbolic mapping.
        
        Parameters
        ----------
            domain : Domain
                Mapped domain object from struphy.geometry.domain_3d
        """

        metric = [[lambda e1, e2, e3 : abs(domain.evaluate(e1, e2, e3, 'det_df'))]]
        
        self._M0 = get_mass(self.V0, self.V0, metric)
        
    def assemble_M1_nonsymb(self, domain):
        """ 
        Assemble mass matrix for L2-scalar product in V1 without psydac's symbolic mapping.
        
        Parameters
        ----------
            domain : Domain
                Mapped domain object from struphy.geometry.domain_3d
        """
        
        metric = []
        
        metric += [[]]

        metric[-1] += [lambda e1, e2, e3 : domain.evaluate(e1, e2, e3, 'g_inv_11')*abs(domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : domain.evaluate(e1, e2, e3, 'g_inv_12')*abs(domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : domain.evaluate(e1, e2, e3, 'g_inv_13')*abs(domain.evaluate(e1, e2, e3, 'det_df'))]
        
        metric += [[]]

        metric[-1] += [lambda e1, e2, e3 : domain.evaluate(e1, e2, e3, 'g_inv_21')*abs(domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : domain.evaluate(e1, e2, e3, 'g_inv_22')*abs(domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : domain.evaluate(e1, e2, e3, 'g_inv_23')*abs(domain.evaluate(e1, e2, e3, 'det_df'))]
        
        metric += [[]]

        metric[-1] += [lambda e1, e2, e3 : domain.evaluate(e1, e2, e3, 'g_inv_31')*abs(domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : domain.evaluate(e1, e2, e3, 'g_inv_32')*abs(domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : domain.evaluate(e1, e2, e3, 'g_inv_33')*abs(domain.evaluate(e1, e2, e3, 'det_df'))]
        
        self._M1 = get_mass(self.V1, self.V1, metric)
        
    def assemble_M2_nonsymb(self, domain):
        """ 
        Assemble mass matrix for L2-scalar product in V2 without psydac's symbolic mapping.
        
        Parameters
        ----------
            domain : Domain
                Mapped domain object from struphy.geometry.domain_3d
        """
        
        metric = []
        
        metric += [[]]

        metric[-1] += [lambda e1, e2, e3 : domain.evaluate(e1, e2, e3, 'g_11')/abs(domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : domain.evaluate(e1, e2, e3, 'g_12')/abs(domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : domain.evaluate(e1, e2, e3, 'g_13')/abs(domain.evaluate(e1, e2, e3, 'det_df'))]
        
        metric += [[]]

        metric[-1] += [lambda e1, e2, e3 : domain.evaluate(e1, e2, e3, 'g_21')/abs(domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : domain.evaluate(e1, e2, e3, 'g_22')/abs(domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : domain.evaluate(e1, e2, e3, 'g_23')/abs(domain.evaluate(e1, e2, e3, 'det_df'))]
        
        metric += [[]]

        metric[-1] += [lambda e1, e2, e3 : domain.evaluate(e1, e2, e3, 'g_31')/abs(domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : domain.evaluate(e1, e2, e3, 'g_32')/abs(domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : domain.evaluate(e1, e2, e3, 'g_33')/abs(domain.evaluate(e1, e2, e3, 'det_df'))]
        
        self._M2 = get_mass(self.V2, self.V2, metric)
        
    def assemble_M3_nonsymb(self, domain):
        """ 
        Assemble mass matrix for L2-scalar product in V3 without psydac's symbolic mapping.
        
        Parameters
        ----------
            domain : Domain
                Mapped domain object from struphy.geometry.domain_3d
        """

        metric = [[lambda e1, e2, e3 : 1/abs(domain.evaluate(e1, e2, e3, 'det_df'))]]
        
        self._M3 = get_mass(self.V3, self.V3, metric)
        
    def assemble_M0vec_nonsymb(self, domain):
        """ 
        Assemble mass matrix for L2-scalar product in V0vec without psydac's symbolic mapping.
        
        Parameters
        ----------
            domain : Domain
                Mapped domain object from struphy.geometry.domain_3d
        """
        
        metric = []
        
        metric += [[]]

        metric[-1] += [lambda e1, e2, e3 : domain.evaluate(e1, e2, e3, 'g_11')*abs(domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : domain.evaluate(e1, e2, e3, 'g_12')*abs(domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : domain.evaluate(e1, e2, e3, 'g_13')*abs(domain.evaluate(e1, e2, e3, 'det_df'))]
        
        metric += [[]]

        metric[-1] += [lambda e1, e2, e3 : domain.evaluate(e1, e2, e3, 'g_21')*abs(domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : domain.evaluate(e1, e2, e3, 'g_22')*abs(domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : domain.evaluate(e1, e2, e3, 'g_23')*abs(domain.evaluate(e1, e2, e3, 'det_df'))]
        
        metric += [[]]

        metric[-1] += [lambda e1, e2, e3 : domain.evaluate(e1, e2, e3, 'g_31')*abs(domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : domain.evaluate(e1, e2, e3, 'g_32')*abs(domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : domain.evaluate(e1, e2, e3, 'g_33')*abs(domain.evaluate(e1, e2, e3, 'det_df'))]
        
        self._M0vec = get_mass(self.V0vec, self.V0vec, metric)
        
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

        # Get space info
        dims = [space.nbasis for space in self.V0.spaces]
        index_arr = self.index_array_N
        kinds = self.spl_kind

        # Get process info
        starts = self.V0.vector_space.starts
        ends = self.V0.vector_space.ends

        # Get communicator info
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()

        res = -1

        # central component is always the process itself
        if comp == [1,1,1]:
            return rank

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
    
    @property
    def V0(self):
        """ Discrete H1 space.
        """
        return self._V0

    @property
    def V1(self):
        """ Discrete H(curl) space.
        """
        return self._V1

    @property
    def V2(self):
        """ Discrete H(div) space.
        """
        return self._V2

    @property
    def V3(self):
        """ Discrete L2 space.
        """
        return self._V3

    @property
    def V0vec(self):
        """ Discrete H1 x H1 x H1 space.
        """
        return self._V0vec

    @property
    def P0(self):
        """ Interpolation into discrete H1 space.
        """
        return self._P0

    @property
    def P1(self):
        """ Inter-/histopolation into discrete H(curl) space.
        """
        return self._P1

    @property
    def P2(self):
        """ Inter-/histopolation into discrete H(div) space.
        """
        return self._P2

    @property
    def P3(self):
        """ Histopolation into discrete L2 space.
        """
        return self._P3

    @property
    def P0vec(self):
        """ Interpolation into discrete H1 x H1 x H1 space.
        """
        return self._P0vec

    @property
    def grad(self):
        """ Discrete gradient H1 -> H(curl).
        """
        return self._grad

    @property
    def curl(self):
        """ Discrete curl H(curl) -> H(div).
        """
        return self._curl

    @property
    def div(self):
        """ Discrete divergence H(div) -> L2.
        """
        return self._div

    @property
    def M0(self):
        """ Mass matrix for L2-scalar product in V0.
        """
        if hasattr(self, '_M0'):
            return self._M0
        else:
            raise AttributeError('M0 not assembled.')

    @property
    def M1(self):
        """ Mass matrix for L2-scalar product in V1.
        """
        if hasattr(self, '_M1'):
            return self._M1
        else:
            raise AttributeError('M1 not assembled.')

    @property
    def M2(self):
        """ Mass matrix for L2-scalar product in V2.
        """
        if hasattr(self, '_M2'):
            return self._M2
        else:
            raise AttributeError('M2 not assembled.')

    @property
    def M3(self):
        """ Mass matrix for L2-scalar product in V3.
        """
        if hasattr(self, '_M3'):
            return self._M3
        else:
            raise AttributeError('M3 not assembled.')
            
    @property
    def M0vec(self):
        """ Mass matrix for L2-scalar product in V0vec.
        """
        if hasattr(self, '_M0vec'):
            return self._M0vec
        else:
            raise AttributeError('M0vec not assembled.')


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

