import cunumpy as xp
from psydac.ddm.mpi import mpi as MPI
from psydac.linalg.basic import Vector, VectorSpace
from psydac.linalg.block import BlockVector
from psydac.linalg.stencil import StencilVector


class PolarDerhamSpace(VectorSpace):
    """
    Derham space with polar basis in eta1-eta2.

    Parameters
    ----------
    derham : struphy.feec.psydac_derham.Derham
        Discrete Derham complex.

    space_id : str
        Space identifier for the field (H1, Hcurl, Hdiv, L2 or H1vec).
    """

    def __init__(self, derham, space_id):
        assert derham.spl_kind[0] == False, "Spline basis in eta1 must be clamped"
        assert derham.spl_kind[1], "Spline basis in eta2 must be periodic"
        assert (derham.Nel[1] / 3) % 1 == 0.0, "Number of elements in eta2 must be a multiple of 3"

        assert derham.p[0] > 1 and derham.p[1] > 1, "Spline degrees in (eta1, eta2) must be at least two"

        # other properties
        self._dtype = float
        self._comm = derham.comm
        self._space_id = space_id

        # dimensions of 1d spaces
        self._n = [space.nbasis for space in derham.Vh_fem["0"].spaces]
        self._d = [space.nbasis for space in derham.Vh_fem["3"].spaces]

        self._parent_space = derham.Vh[derham.space_to_form[space_id]]
        self._parallel = self._parent_space.parallel

        self._starts = self.parent_space.starts
        self._ends = self._parent_space.ends

        # polar properties
        if space_id == "H1":
            self._n_polar = (3,)
            self._n_rings = (2,)
            self._dimension = ((self.n[0] - self.n_rings[0]) * self.n[1] + self.n_polar[0]) * self.n[2]
            self._n2 = (self.n[1],)
            self._n3 = (self.n[2],)
            self._type_of_basis_3 = (derham.spline_types["0"][2],)
        elif space_id == "Hcurl":
            self._n_polar = (0, 2, 3)
            self._n_rings = (1, 2, 2)
            dim1 = ((self.d[0] - self.n_rings[0]) * self.n[1] + self.n_polar[0]) * self.n[2]
            dim2 = ((self.n[0] - self.n_rings[1]) * self.d[1] + self.n_polar[1]) * self.n[2]
            dim3 = ((self.n[0] - self.n_rings[2]) * self.n[1] + self.n_polar[2]) * self.d[2]
            self._dimension = dim1 + dim2 + dim3
            self._n2 = (self.n[1], self.d[1], self.n[1])
            self._n3 = (self.n[2], self.n[2], self.d[2])
            self._type_of_basis_3 = (
                derham.spline_types["1"][0][2],
                derham.spline_types["1"][1][2],
                derham.spline_types["1"][2][2],
            )
        elif space_id == "Hdiv":
            self._n_polar = (2, 0, 0)
            self._n_rings = (2, 1, 1)
            dim1 = ((self.n[0] - self.n_rings[0]) * self.d[1] + self.n_polar[0]) * self.d[2]
            dim2 = ((self.d[0] - self.n_rings[1]) * self.n[1] + self.n_polar[1]) * self.d[2]
            dim3 = ((self.d[0] - self.n_rings[2]) * self.d[1] + self.n_polar[2]) * self.n[2]
            self._dimension = dim1 + dim2 + dim3
            self._n2 = (self.d[1], self.n[1], self.d[1])
            self._n3 = (self.d[2], self.d[2], self.n[2])
            self._type_of_basis_3 = (
                derham.spline_types["2"][0][2],
                derham.spline_types["2"][1][2],
                derham.spline_types["2"][2][2],
            )
        elif space_id == "L2":
            self._n_polar = (0,)
            self._n_rings = (1,)
            self._dimension = ((self.d[0] - self.n_rings[0]) * self.d[1] + self.n_polar[0]) * self.d[2]
            self._n2 = (self.d[1],)
            self._n3 = (self.d[2],)
            self._type_of_basis_3 = (derham.spline_types["3"][2],)
        elif space_id == "H1vec":
            self._n_polar = (3, 3, 3)
            self._n_rings = (2, 2, 2)
            self._dimension = (((self.n[0] - self.n_rings[0]) * self.n[1] + self.n_polar[0]) * self.n[2]) * 3
            self._n2 = (self.n[1], self.n[1], self.n[1])
            self._n3 = (self.n[2], self.n[2], self.n[2])
            self._type_of_basis_3 = (
                derham.spline_types["v"][0][2],
                derham.spline_types["v"][1][2],
                derham.spline_types["v"][2][2],
            )
        else:
            raise ValueError("Space not supported.")

        self._n_comps = len(self.n_polar)

        if self.n_comps == 1:
            if self.starts[0] == 0:
                assert self.ends[0] > self.n_rings[0], "MPI coeff decomposition in eta_1 too small for polar splines!"
        else:
            for n in range(3):
                if self.starts[n][0] == 0:
                    assert self.ends[n][0] > self.n_rings[n], (
                        "MPI coeff decomposition in eta_1 too small for polar splines!"
                    )

    @property
    def dtype(self):
        """TODO"""
        return self._dtype

    @property
    def comm(self):
        """TODO"""
        return self._comm

    @property
    def space_id(self):
        """TODO"""
        return self._space_id

    @property
    def n_comps(self):
        """TODO"""
        return self._n_comps

    @property
    def n(self):
        """TODO"""
        return self._n

    @property
    def d(self):
        """TODO"""
        return self._d

    @property
    def parent_space(self):
        """The parent space (StencilVectorSpace or BlockVectorSpace) of which the PolarDerhamSpace is a sub-space."""
        return self._parent_space

    @property
    def starts(self):
        """TODO"""
        return self._starts

    @property
    def ends(self):
        """TODO"""
        return self._ends

    @property
    def dimension(self):
        """TODO"""
        return self._dimension

    @property
    def n_polar(self):
        """Number of polar basis functions in each component (tuple for vector-valued)."""
        return self._n_polar

    @property
    def n_rings(self):
        """Number of rings to be set to zero in tensor-product basis (tuple for vector-valued)."""
        return self._n_rings

    @property
    def n2(self):
        """Tuple holding total (global) number of basis function in eta2, for each component."""
        return self._n2

    @property
    def n3(self):
        """Tuple holding total (global) number of basis function in eta3, for each component."""
        return self._n3

    @property
    def type_of_basis_3(self):
        """Tuple holding type of spline basis (B-splines or M-splines), for each component."""
        return self._type_of_basis_3

    @property
    def parallel(self):
        return self._parallel

    def zeros(self):
        """
        Creates an element of the vector space filled with zeros.
        """
        return PolarVector(self)

    def axpy(self, a, x, y):
        y += a * x
        pass

    def inner(self, x, y):
        assert isinstance(x, PolarVector)
        return x.dot(y)


class PolarVector(Vector):
    """
    Element of a PolarDerhamSpace.

    An instance of a PolarVector consists of two parts:
        1. a list of xp.arrays of the polar coeffs (not distributed)
        2. a tensor product StencilVector/BlockVector of the parent space with inner rings set to zero (distributed).

    Parameters
    ----------
    V : PolarDerhamSpace
        Vector space which the polar vector to be created belongs to.
    """

    def __init__(self, V):
        assert isinstance(V, PolarDerhamSpace)
        self._space = V
        self._dtype = V.dtype

        # initialize polar coeffs
        self._pol = [xp.zeros((m, n)) for m, n in zip(V.n_polar, V.n3)]

        # full tensor product vector
        self._tp = V.parent_space.zeros()

    @property
    def space(self):
        """TODO"""
        return self._space

    @property
    def dtype(self):
        """TODO"""
        return self._dtype

    @property
    def pol(self):
        """Polar coefficients as xp.array."""
        return self._pol

    @pol.setter
    def pol(self, v):
        """In-place setter for polar coefficients."""
        assert isinstance(v, list)
        assert len(v) == self.space.n_comps
        for n in range(self.space.n_comps):
            self._pol[n][:] = v[n]

    @property
    def tp(self):
        """Tensor product Stencil-/BlockVector with inner rings set to zero."""
        return self._tp

    @tp.setter
    def tp(self, v):
        """In-place setter for tensor product Stencil-/BlockVector with constraint that inner rings must be zero."""
        assert v.space == self.space.parent_space

        if isinstance(v, StencilVector):
            self._tp[:] = v[:]
        elif isinstance(v, BlockVector):
            for n, starts in enumerate(v.space.starts):
                self._tp[n][:] = v[n][:]
        else:
            raise ValueError(
                "Attribute can only be set with instances of either StencilVector or BlockVector!",
            )

        self.set_tp_coeffs_to_zero()

    @property
    def ghost_regions_in_sync(self):
        """Whether ghost regions of tensor product part are up-to-date."""
        return self.tp.ghost_regions_in_sync

    def dot(self, v):
        """
        Scalar product with another instance of PolarVector.
        """
        assert isinstance(v, PolarVector)
        assert v.space == self.space

        # tensor-product part
        out = self.tp.inner(v.tp)

        # polar part
        out += sum([a1.flatten().dot(a2.flatten()) for a1, a2 in zip(self.pol, v.pol)])

        return out

    def set_vector(self, v):
        """
        In-place setter for polar + tensor product coeffiecients.
        """
        assert isinstance(v, PolarVector)

        # tensor-product part
        self.tp = v.tp

        # polar part
        self.pol = v.pol

    def set_tp_coeffs_to_zero(self):
        """
        Sets inner tensor-product rings that make up the polar splines to zero.
        """
        set_tp_rings_to_zero(self.tp, self.space.n_rings)

    def toarray(self, allreduce=False):
        """
        Converts the polar vector to a 1d numpy array.
        """

        if isinstance(self.tp, StencilVector):
            s1, s2, s3 = self.space.starts
            e1, e2, e3 = self.space.ends

            out = self.tp.toarray()[self.space.n_rings[0] * self.space.n[1] * self.space.n3[0] :]

            # allreduce tensor-product part
            if self.space.comm is not None and allreduce:
                self.space.comm.Allreduce(MPI.IN_PLACE, out, op=MPI.SUM)

            out = xp.concatenate((self.pol[0].flatten(), out))

        else:
            out1 = self.tp[0].toarray()[self.space.n_rings[0] * self.space.n[1] * self.space.n3[0] :]
            out2 = self.tp[1].toarray()[self.space.n_rings[1] * self.space.n[1] * self.space.n3[1] :]
            out3 = self.tp[2].toarray()[self.space.n_rings[2] * self.space.n[1] * self.space.n3[2] :]

            # allreduce tensor-product part
            if self.space.comm is not None and allreduce:
                self.space.comm.Allreduce(MPI.IN_PLACE, out1, op=MPI.SUM)
                self.space.comm.Allreduce(MPI.IN_PLACE, out2, op=MPI.SUM)
                self.space.comm.Allreduce(MPI.IN_PLACE, out3, op=MPI.SUM)

            out = xp.concatenate(
                (
                    self.pol[0].flatten(),
                    out1,
                    self.pol[1].flatten(),
                    out2,
                    self.pol[2].flatten(),
                    out3,
                ),
            )

        return out

    def toarray_tp(self):
        """
        Converts the Stencil-/BlockVector to a 1d numpy array but NOT the polar part.
        """
        return self.pol, self.tp.toarray()

    def copy(self, out=None):
        """TODO"""
        w = out or PolarVector(self.space)
        # copy stencil part
        self._tp.copy(out=w.tp)
        # copy polar part
        for n, pl in enumerate(self._pol):
            xp.copyto(w._pol[n], pl, casting="no")
        return w

    def __neg__(self):
        """TODO"""
        w = PolarVector(self.space)
        if isinstance(w.tp, StencilVector):
            w._pol[0][:] = -self.pol[0]
            w._tp[:] = -self.tp[:]
        else:
            for n in range(3):
                w._pol[n][:] = -self.pol[n]
                w._tp[n][:] = -self.tp[n][:]
        return w

    def __mul__(self, a):
        """TODO"""
        w = PolarVector(self.space)
        if isinstance(w.tp, StencilVector):
            w._pol[0][:] = self.pol[0] * a
            w._tp[:] = self.tp[:] * a
        else:
            for n in range(3):
                w._pol[n][:] = self.pol[n] * a
                w._tp[n][:] = self.tp[n][:] * a
        return w

    def __rmul__(self, a):
        """TODO"""
        w = PolarVector(self.space)
        if isinstance(w.tp, StencilVector):
            w._pol[0][:] = a * self.pol[0]
            w._tp[:] = a * self.tp[:]
        else:
            for n in range(3):
                w._pol[n][:] = a * self.pol[n]
                w._tp[n][:] = a * self.tp[n][:]
        return w

    def __add__(self, v):
        """TODO"""
        assert isinstance(v, PolarVector)
        assert v.space == self.space

        w = PolarVector(self.space)
        if isinstance(w.tp, StencilVector):
            w._pol[0][:] = self.pol[0] + v.pol[0]
            w._tp[:] = self.tp[:] + v.tp[:]
        else:
            for n in range(3):
                w._pol[n][:] = self.pol[n] + v.pol[n]
                w._tp[n][:] = self.tp[n][:] + v.tp[n][:]
        return w

    def __sub__(self, v):
        """TODO"""
        assert isinstance(v, PolarVector)
        assert v.space == self.space

        w = PolarVector(self.space)
        if isinstance(w.tp, StencilVector):
            w._pol[0][:] = self.pol[0] - v.pol[0]
            w._tp[:] = self.tp[:] - v.tp[:]
        else:
            for n in range(3):
                w._pol[n][:] = self.pol[n] - v.pol[n]
                w._tp[n][:] = self.tp[n][:] - v.tp[n][:]
        return w

    def __imul__(self, a):
        """TODO"""
        if isinstance(self.tp, StencilVector):
            self._pol[0] *= a
            self._tp *= a
        else:
            for n in range(3):
                self._pol[n] *= a
                self._tp[n] *= a
        return self

    def __iadd__(self, v):
        """TODO"""
        assert isinstance(v, PolarVector)
        assert v.space == self.space

        if isinstance(self.tp, StencilVector):
            self._pol[0] += v.pol[0]
            self._tp += v.tp
        else:
            for n in range(3):
                self._pol[n] += v.pol[n]
                self._tp[n] += v.tp[n]
        return self

    def __isub__(self, v):
        """TODO"""
        assert isinstance(v, PolarVector)
        assert v.space == self.space

        if isinstance(self.tp, StencilVector):
            self._pol[0] -= v.pol[0]
            self._tp -= v.tp
        else:
            for n in range(3):
                self._pol[n] -= v.pol[n]
                self._tp[n] -= v.tp[n]
        return self

    # def update_ghost_regions(self, *, direction=None):
    def update_ghost_regions(self):
        """
        Update ghost regions before performing non-local access to vector
        elements (e.g. in matrix-vector product).

        Parameters
        ----------
        direction : int
            Single direction along which to operate (if not specified, all of them).

        """
        # self._tp.update_ghost_regions(direction=direction)
        self._tp.update_ghost_regions()

        # def update_ghost_regions(self, *, direction=None):

    def exchange_assembly_data(self):
        """
        Exchange assembly data before performing non-local access to vector
        elements (e.g. in matrix-vector product).

        Parameters
        ----------
        direction : int
            Single direction along which to operate (if not specified, all of them).

        """
        # self._tp.update_ghost_regions(direction=direction)
        self._tp.exchange_assembly_data()

    def conjugate(self):
        """No need for complex conjugate"""
        pass


def set_tp_rings_to_zero(v, n_rings):
    """
    Sets a certain number of rings of a Stencil-/BlockVector in eta_1 direction to zero.

    Parameters
    ----------
    v : StencilVector | BlockVector
        The vector whose inner rings shall be set to zero.

    n_rings : tuple
        The number of rings that shall be set to zero (has length 1 for StencilVector and 3 for BlockVector).
    """
    assert isinstance(n_rings, tuple)

    if isinstance(v, StencilVector):
        if v.starts[0] == 0:
            v[: n_rings[0], :, :] = 0.0
    elif isinstance(v, BlockVector):
        for n, starts in enumerate(v.space.starts):
            if starts[0] == 0:
                v[n][: n_rings[n], :, :] = 0.0
    else:
        raise ValueError(
            "Input vector must be an instance of StencilVector of BlockVector!",
        )
