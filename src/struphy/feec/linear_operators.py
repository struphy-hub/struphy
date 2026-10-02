from feectools.linalg.basic import LinearOperator, Vector, VectorSpace
from feectools.linalg.stencil import StencilVectorSpace

from struphy.feec.utilities import apply_essential_bc_to_array
from struphy.polar.basic import PolarDerhamSpace


class BoundaryOperator(LinearOperator):
    r"""
    Applies homogeneous Dirichlet boundary conditions to a vector.

    Parameters
    ----------
    vector_space : feectools.linalg.basic.VectorSpace
        The vector space associated to the operator.

    space_id : str
        Symbolic space ID of vector_space (H1, Hcurl, Hdiv, L2 or H1vec).

    dirichlet_bc : tuple[tuple[bool]]
        Whether to apply homogeneous Dirichlet boundary conditions (at left or right boundary in each direction).
    """

    def __init__(self, vector_space, space_id, dirichlet_bc):
        assert isinstance(vector_space, VectorSpace)
        assert isinstance(space_id, str)

        self._domain = vector_space
        self._codomain = vector_space
        self._dtype = vector_space.dtype

        self._space_id = space_id
        self._bc = dirichlet_bc

        assert isinstance(dirichlet_bc, tuple)
        assert len(dirichlet_bc) == 3

        # number of non-zero elements in poloidal/toroidal direction
        if isinstance(vector_space, PolarDerhamSpace):
            vec_space_ten = vector_space.parent_space
        else:
            vec_space_ten = vector_space

        if isinstance(vec_space_ten, StencilVectorSpace):
            n_pts = vec_space_ten.npts
        else:
            n_pts = [comp.npts for comp in vec_space_ten.spaces]

        dim_nz1_pol = 1
        dim_nz2_pol = 1
        dim_nz3_pol = 1

        dim_nz1_tor = 1
        dim_nz2_tor = 1
        dim_nz3_tor = 1

        if space_id == "H1":
            if isinstance(vector_space, PolarDerhamSpace):
                dim_nz1_pol *= (n_pts[0] - vector_space.n_rings[0] - self.bc[0][1]) * n_pts[1]
                dim_nz1_pol += vector_space.n_polar[0]
            else:
                dim_nz1_pol *= n_pts[0] - self.bc[0][0] - self.bc[0][1]
                dim_nz1_pol *= n_pts[1] - self.bc[1][0] - self.bc[1][1]

            dim_nz1_tor *= n_pts[2] - self.bc[2][0] - self.bc[2][1]

            self._dim_nz_pol = (dim_nz1_pol,)
            self._dim_nz_tor = (dim_nz1_tor,)

            self._dim_nz = (dim_nz1_pol * dim_nz1_tor,)

        elif space_id == "Hcurl":
            if isinstance(vector_space, PolarDerhamSpace):
                dim_nz1_pol *= (n_pts[0][0] - vector_space.n_rings[0]) * n_pts[0][1]
                dim_nz1_pol += vector_space.n_polar[0]

                dim_nz2_pol *= (n_pts[1][0] - vector_space.n_rings[1] - self.bc[0][1]) * n_pts[1][1]
                dim_nz2_pol += vector_space.n_polar[1]

                dim_nz3_pol *= (n_pts[2][0] - vector_space.n_rings[2] - self.bc[0][1]) * n_pts[2][1]
                dim_nz3_pol += vector_space.n_polar[2]
            else:
                dim_nz1_pol *= n_pts[0][0]
                dim_nz1_pol *= n_pts[0][1] - self.bc[1][0] - self.bc[1][1]

                dim_nz2_pol *= n_pts[1][0] - self.bc[0][0] - self.bc[0][1]
                dim_nz2_pol *= n_pts[1][1]

                dim_nz3_pol *= n_pts[2][0] - self.bc[0][0] - self.bc[0][1]
                dim_nz3_pol *= n_pts[2][1] - self.bc[1][0] - self.bc[1][1]

            dim_nz1_tor *= n_pts[0][2] - self.bc[2][0] - self.bc[2][1]
            dim_nz2_tor *= n_pts[1][2] - self.bc[2][0] - self.bc[2][1]
            dim_nz3_tor *= n_pts[2][2]

            self._dim_nz_pol = (dim_nz1_pol, dim_nz2_pol, dim_nz3_pol)
            self._dim_nz_tor = (dim_nz1_tor, dim_nz2_tor, dim_nz3_tor)

            self._dim_nz = (dim_nz1_pol * dim_nz1_tor, dim_nz2_pol * dim_nz2_tor, dim_nz3_pol * dim_nz3_tor)

        elif space_id == "Hdiv" or space_id == "H1vec":
            if isinstance(vector_space, PolarDerhamSpace):
                dim_nz1_pol *= (n_pts[0][0] - vector_space.n_rings[0] - self.bc[0][1]) * n_pts[0][1]
                dim_nz1_pol += vector_space.n_polar[0]

                dim_nz2_pol *= (n_pts[1][0] - vector_space.n_rings[1]) * n_pts[1][1]
                dim_nz2_pol += vector_space.n_polar[1]

                dim_nz3_pol *= (n_pts[2][0] - vector_space.n_rings[2]) * n_pts[2][1]
                dim_nz3_pol += vector_space.n_polar[2]
            else:
                dim_nz1_pol *= n_pts[0][0] - self.bc[0][0] - self.bc[0][1]
                dim_nz1_pol *= n_pts[0][1]

                dim_nz2_pol *= n_pts[1][0]
                dim_nz2_pol *= n_pts[1][1] - self.bc[1][0] - self.bc[1][1]

                dim_nz3_pol *= n_pts[2][0]
                dim_nz3_pol *= n_pts[2][1]

            dim_nz1_tor *= n_pts[0][2]
            dim_nz2_tor *= n_pts[1][2]
            dim_nz3_tor *= n_pts[2][2] - self.bc[2][0] - self.bc[2][1]

            self._dim_nz_pol = (dim_nz1_pol, dim_nz2_pol, dim_nz3_pol)
            self._dim_nz_tor = (dim_nz1_tor, dim_nz2_tor, dim_nz3_tor)

            self._dim_nz = (dim_nz1_pol * dim_nz1_tor, dim_nz2_pol * dim_nz2_tor, dim_nz3_pol * dim_nz3_tor)

        else:
            if isinstance(vector_space, PolarDerhamSpace):
                dim_nz1_pol *= (n_pts[0] - vector_space.n_rings[0]) * n_pts[1]
                dim_nz1_pol += vector_space.n_polar[0]
            else:
                dim_nz1_pol *= n_pts[0]
                dim_nz1_pol *= n_pts[1]

            dim_nz1_tor *= n_pts[2]

            self._dim_nz_pol = (dim_nz1_pol,)
            self._dim_nz_tor = (dim_nz1_tor,)

            self._dim_nz = (dim_nz1_pol * dim_nz1_tor,)

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def dtype(self):
        return self._dtype

    def tosparse(self):
        return self.toarray(is_sparse=True)

    @property
    def bc(self):
        return self._bc

    @property
    def dim_nz_pol(self):
        return self._dim_nz_pol

    @property
    def dim_nz_tor(self):
        return self._dim_nz_tor

    @property
    def dim_nz(self):
        return self._dim_nz

    def dot(self, v, out=None):
        """
        Dot product of the operator with a vector.

        Parameters
        ----------
        v : feectools.linalg.basic.Vector
            The input (domain) vector.

        out : feectools.linalg.basic.Vector, optional
            If given, the output will be written in-place into this vector.

        Returns
        -------
        out : feectools.linalg.basic.Vector
            The output (codomain) vector.
        """

        assert isinstance(v, Vector)
        assert v.space == self._domain

        if out is None:
            out = v.copy()
        else:
            assert isinstance(out, Vector)
            assert out.space == self._codomain
            v.copy(out=out)

        # apply boundary conditions to output vector
        apply_essential_bc_to_array(self._space_id, out, self.bc)

        return out

    def transpose(self, conjugate=False):
        """
        Returns the transposed operator.
        """
        return BoundaryOperator(self._domain, self._space_id, self.bc)
