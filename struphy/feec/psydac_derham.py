#!/usr/bin/env python3

from psydac.api.discretization import discretize
from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL
from psydac.fem.vector import ProductFemSpace

from sympde.topology import elements_of
from sympde.expr import BilinearForm, integral
from sympde.calculus import dot
from sympde.topology import Cube, Derham

from sympy import sqrt

from struphy.psydac_linear_operators.H1vec_psydac import Projector_H1vec


class Derham_build:
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

        # Boundary conditions
        self._spl_kind = spl_kind

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

    # Psydac mass matrices
    # --------------------
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
