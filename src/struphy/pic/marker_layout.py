"""Memory layout of the marker buffer.

:attr:`Particles.markers <struphy.pic.base.Particles.markers>` has shape ``(n_rows, n_cols)``, one row per marker,
and is stored column-major (Fortran order, struct-of-arrays): every column is one contiguous block of ``n_rows``
values. Column operations (``markers[:, j]``, ``markers[:, a:b] = ...``, a mask on one column) then touch only
the columns involved, and on the GPU neighbouring threads (markers) read neighbouring addresses. Indexing is
unchanged: ``markers[ip, j]`` is still the ``j``-th entry of marker ``ip``.

Kernels see the buffer as follows:

* pyccel kernels declare it as ``float[:, :](order=F)`` (in :class:`~struphy.kernel_arguments.pusher_args_kernels.MarkerArguments`
  and in the kernels that take ``markers`` directly); pyccel checks the order at every call, so a C-ordered array
  raises a ``TypeError`` instead of being read wrongly.
* CUDA kernels read it through ``Array2D`` views, which carry the strides of the array.

Copies made by fancy or boolean row indexing (``markers[valid_mks]``), ``.copy()`` and ``xp.concatenate`` are
C-ordered. They are fine to read, or to write back into the buffer with ``markers[rows] = ...``; to pass such an
array to a kernel, convert it with :func:`as_marker_array` first.

See struphy-hub/struphy#707.
"""

import cunumpy as xp

MARKER_ORDER = "F"
"""Memory order of the marker buffer and of every marker array passed to a kernel."""

ROW_ALIGNMENT = 32
"""The number of rows is a multiple of this, so that every column starts on a 256-byte boundary (32 doubles)."""


def padded_n_rows(n_rows: int) -> int:
    """Smallest multiple of :data:`ROW_ALIGNMENT` that is at least ``n_rows``."""
    return -(-int(n_rows) // ROW_ALIGNMENT) * ROW_ALIGNMENT


def marker_array(n_rows: int, n_cols: int):
    """Zero marker array of shape ``(n_rows, n_cols)`` in :data:`MARKER_ORDER` on the active backend."""
    return xp.zeros((n_rows, n_cols), dtype=float, order=MARKER_ORDER)


def as_marker_array(markers):
    """``markers`` in :data:`MARKER_ORDER`, for passing to a kernel; no copy if it already has that order."""
    return xp.asfortranarray(markers)


def has_marker_layout(markers) -> bool:
    """Whether ``markers`` is a 2D array in :data:`MARKER_ORDER` (as the kernels expect)."""
    return markers.ndim == 2 and bool(markers.flags.f_contiguous)


def check_marker_layout(markers, name: str = "markers"):
    """Raise a ``TypeError`` if ``markers`` is not a 2D array in :data:`MARKER_ORDER`."""
    if not has_marker_layout(markers):
        raise TypeError(
            f"{name} must be a 2D Fortran-ordered (column-major) array, got shape {markers.shape} with "
            f"C_CONTIGUOUS={markers.flags.c_contiguous}, F_CONTIGUOUS={markers.flags.f_contiguous}; "
            "see struphy.pic.marker_layout"
        )
