"""Memory layout of the marker buffer.

:attr:`Particles.markers <struphy.pic.base.Particles.markers>` has shape ``(n_cols, n_rows)``, one column per marker,
and is stored in C order (struct-of-arrays): every quantity (row ``j``) is one contiguous block of ``n_rows`` values.
``markers[j, ip]`` is the ``j``-th entry of marker ``ip``; ``markers[j]`` is quantity ``j`` of all markers, and
``markers[:, ip]`` all quantities of marker ``ip``. Operations on quantities (``markers[j]``, ``markers[a:b] = ...``,
a mask on one row) then touch only the rows involved, and on the GPU neighbouring threads (markers) read neighbouring
addresses.

The same convention, one column per point, holds for every array of markers or points passed to an evaluation
(:class:`~struphy.geometry.base.Domain`, :class:`~struphy.feec.psydac_derham.SplineFunction`, the equilibria):
``(k, n)`` with the logical coordinates in rows 0 to 2.

The kernels take C-ordered arrays of this shape. Selecting markers (``markers[:, valid_mks]``) gives a Fortran-ordered
copy (NumPy keeps the selected axis innermost): fine to read or to write back with ``markers[:, idx] = ...``, but convert
it with ``xp.ascontiguousarray`` before passing it to a kernel or to MPI.

See struphy-hub/struphy#707.
"""

import cunumpy as xp

ROW_ALIGNMENT = 32
"""The number of markers ``n_rows`` is a multiple of this, so that every row of the buffer starts on a 256-byte
boundary (32 doubles)."""


def padded_n_rows(n_rows: int) -> int:
    """Smallest multiple of :data:`ROW_ALIGNMENT` that is at least ``n_rows``."""
    return -(-int(n_rows) // ROW_ALIGNMENT) * ROW_ALIGNMENT


def marker_array(n_cols: int, n_rows: int):
    """Zero marker array of shape ``(n_cols, n_rows)`` on the active backend."""
    return xp.zeros((n_cols, n_rows), dtype=float)
