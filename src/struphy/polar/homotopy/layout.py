"""Ordering of the coefficients of a 2D tensor-product component of shape ``(nx, ny)`` (``x`` radial, ``y``
angular).

* :data:`STRUPHY`: the ordering of Struphy's ``StencilVector``s, ``i * ny + j`` (the radial index ``i`` varies
  slowest, a ring ``i`` is a contiguous block), Kronecker products ``kron(A_x, A_y)``;
* :data:`CORE`: the ordering of PolarFEEC.jl, ``i + j * nx`` (the radial index varies fastest),
  Kronecker products ``kron(A_y, A_x)``.

The polar construction is written for any layout; the Derham complex uses :data:`STRUPHY`.
"""

import numpy as np
import scipy.sparse as sp


class Layout:
    """Linear indices, reshapes and Kronecker products of one coefficient ordering."""

    def __init__(self, name, order):
        self.name = name
        #: the NumPy order (``"C"`` or ``"F"``) of ``reshape(nx, ny)`` / ``ravel`` of a component
        self.order = order

    def __repr__(self):
        return f"Layout({self.name!r})"

    def index(self, i, j, nx, ny):
        """Linear index of the coefficient ``(i, j)`` (broadcasting)."""
        i, j = np.asarray(i), np.asarray(j)
        return i * ny + j if self.order == "C" else i + j * nx

    def ring(self, linear, nx, ny):
        """Radial index ``i`` of linear indices."""
        linear = np.asarray(linear)
        return linear // ny if self.order == "C" else linear % nx

    def angle(self, linear, nx, ny):
        """Angular index ``j`` of linear indices."""
        linear = np.asarray(linear)
        return linear % ny if self.order == "C" else linear // nx

    def kron(self, radial, angular):
        """Sparse matrix of ``radial (x) angular`` acting on coefficients in this layout."""
        if self.order == "C":
            return sp.kron(radial, angular, format="csr")
        return sp.kron(angular, radial, format="csr")

    def kron_vector(self, radial, angular):
        """Coefficients ``radial[i] * angular[j]`` in this layout."""
        if self.order == "C":
            return np.kron(radial, angular)
        return np.kron(angular, radial)

    def radial_indices(self, nx, ny, internal_rings, offset=0):
        """Sorted linear indices of the internal (radial index ``< internal_rings``) and external coefficients."""
        if not 0 <= internal_rings <= nx:
            raise ValueError(f"the number of internal rings must lie between 0 and {nx}")
        linear = np.arange(nx * ny)
        ring = self.ring(linear, nx, ny)
        return offset + linear[ring < internal_rings], offset + linear[ring >= internal_rings]


STRUPHY = Layout("struphy", "C")
CORE = Layout("core", "F")
