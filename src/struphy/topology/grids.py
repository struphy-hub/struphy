import numpy as np


class TensorProductGrid:
    """Grid as a tensor product of 1d grids.
    
    Parameters
    ----------
    Nel : tuple[int]
        Number of elements in each direction.

    p : tuple[int]
        Spline degree in each direction.

    spl_kind : tuple[bool]
        Kind of spline in each direction (True=periodic, False=clamped).

    dirichlet_bc : tuple[tuple[bool]]
        Whether to apply homogeneous Dirichlet boundary conditions (at left or right boundary in each direction).

    nquads : tuple[int]
        Number of Gauss-Legendre quadrature points in each direction (default = p, leads to exact integration of degree 2p-1 polynomials).

    nq_pr : tuple[int]
        Number of Gauss-Legendre quadrature points in each direction for geometric projectors (default = p+1, leads to exact integration of degree 2p+1 polynomials).
        
    mpi_dims_mask: Tuple of bool
        True if the dimension is to be used in the domain decomposition (=default for each dimension).
        If mpi_dims_mask[i]=False, the i-th dimension will not be decomposed.
    """
    
    def __init__(self, 
                Nel: tuple,
                p: tuple,
                spl_kind: tuple,
                *,
                dirichlet_bc: tuple = None,
                nquads: tuple = None,
                nq_pr: tuple = None,
                mpi_dims_mask: tuple = None,):
        
        assert len(Nel) == len(p) == len(spl_kind)
        
        self._Nel = Nel
        self._p = p
        self._spl_kind = spl_kind
        
        # boundary conditions at eta=0 and eta=1 in each direction (None for periodic, 'd' for homogeneous Dirichlet)
        if dirichlet_bc is not None:
            assert len(dirichlet_bc) == len(Nel)
            # make sure that boundary conditions are compatible with spline space
            assert np.all([bc == [False, False] for i, bc in enumerate(dirichlet_bc) if spl_kind[i]])

        self._dirichlet_bc = dirichlet_bc

        # default p: exact integration of degree 2p+1 polynomials
        if nquads is None:
            self._nquads = tuple([pi + 1 for pi in p])
        else:
            assert len(nquads) == len(Nel)
            self._nquads = nquads

        # default p + 1 : exact integration of degree 2p+1 polynomials
        if nq_pr is None:
            self._nq_pr = tuple([pi + 1 for pi in p])
        else:
            assert len(nq_pr) == len(Nel)
            self._nq_pr = nq_pr

        # mpi domain decomposition directions
        if mpi_dims_mask is None:
            self._mpi_dims_mask = (True,)*len(Nel)
        else:
            assert len(mpi_dims_mask) == len(Nel)
            self._mpi_dims_mask = mpi_dims_mask

    @property
    def Nel(self):
        """Tuple of number of elements (=cells) in each direction."""
        return self._Nel

    @property
    def p(self):
        """Tuple of B-spline degrees in each direction."""
        return self._p

    @property
    def spl_kind(self):
        """Tuple of spline type (periodic=True or clamped=False) in each direction."""
        return self._spl_kind

    @property
    def dirichlet_bc(self):
        """None, or Tuple of boundary conditions in each direction.
        Each entry is a list with two entries (left and right boundary), "d" (hom. Dirichlet) or None (periodic).
        """
        return self._dirichlet_bc

    @property
    def nquads(self):
        """Tuple of number of Gauss-Legendre quadrature points in each direction (default = p, leads to exact integration of degree 2p-1 polynomials)."""
        return self._nquads

    @property
    def nq_pr(self):
        """Tuple of number of Gauss-Legendre quadrature points in histopolation (default = p + 1) in each direction."""
        return self._nq_pr
    
    @property
    def mpi_dims_mask(self):
        """Tuple of bool; whether to use direction in domain decomposition."""
        return self._mpi_dims_mask