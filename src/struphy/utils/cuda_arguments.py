"""CUDA counterparts of the pyccelized kernel argument classes.

The classes in :mod:`struphy.kernel_arguments.pusher_args_kernels` hold references to the arrays
of their owner (e.g. ``Particles.markers``), but, once compiled with pyccel, they only accept
NumPy arrays. The classes here take the same constructor arguments, hold references to the
owner's **CuPy** arrays (no copies are made, non-CuPy arrays raise) and flatten them into the
arguments of a ``cupy.RawKernel``, see :attr:`CudaArguments.values`.

The order of :attr:`CudaArguments.values` defines the corresponding part of the CUDA kernel
signature (``double*`` for float arrays, ``long long*`` for int arrays, ``bool*`` for bool
arrays, ``int`` for int scalars):

* :class:`CudaMarkerArguments` -> ``double* markers, bool* valid_mks, int n_markers, int n_cols,
  int Np, int vdim, int weight_idx, int first_diagnostics_idx, int first_init_idx,
  int first_shift_idx, int residual_idx, int first_free_idx, int mu_idx, long long* bc_type``
* :class:`CudaDomainArguments` -> ``int kind_map, double* params, long long* degree,
  double* t1, double* t2, double* t3, long long* ind1, long long* ind2, long long* ind3,
  double* cx, double* cy, double* cz``
* :class:`CudaDerhamArguments` -> ``long long* pn, double* tn1, double* tn2, double* tn3,
  long long* starts``
"""

import numpy as np


def _cupy_array(name: str, arr, dtype):
    """Return ``arr`` itself after checking it is a C-contiguous CuPy array of ``dtype`` (never copies)."""
    if not hasattr(arr, "__cuda_array_interface__"):
        raise TypeError(
            f"{name} must be a CuPy array (device memory), got {type(arr)}; no host-device copies are made."
        )
    if arr.dtype != dtype:
        raise TypeError(f"{name} must have dtype {np.dtype(dtype)}, got {arr.dtype}.")
    if not arr.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous.")
    return arr


class CudaArguments:
    """Base class; :attr:`values` is the flat tuple passed to the CUDA kernel.

    Instances are read-only after construction, such that :attr:`values` always matches the attributes.
    """

    _names: tuple[str, ...] = ()

    def _freeze(self):
        object.__setattr__(self, "_values", tuple(getattr(self, name) for name in self._names))

    def __setattr__(self, name, value):
        if hasattr(self, "_values"):
            raise AttributeError(f"{type(self).__name__} is read-only.")
        object.__setattr__(self, name, value)

    @property
    def values(self) -> tuple:
        """Flat CUDA kernel arguments."""
        return self._values

    @property
    def n_threads(self) -> int | None:
        """Number of CUDA threads needed for this argument, None if no preference."""
        return None


class CudaMarkerArguments(CudaArguments):
    """CUDA version of :class:`~struphy.kernel_arguments.pusher_args_kernels.MarkerArguments` (same parameters)."""

    _names = (
        "markers",
        "valid_mks",
        "n_markers",
        "n_cols",
        "Np",
        "vdim",
        "weight_idx",
        "first_diagnostics_idx",
        "first_init_idx",
        "first_shift_idx",
        "residual_idx",
        "first_free_idx",
        "mu_idx",
        "bc_type",
    )

    def __init__(
        self,
        markers,
        valid_mks,
        Np: int,
        vdim: int,
        weight_idx: int,
        first_diagnostics_idx: int,
        first_pusher_idx: int,
        first_shift_idx: int,
        residual_idx: int,
        first_free_idx: int,
        mu_idx: int,
        bc_type,
    ):
        self.markers = _cupy_array("markers", markers, np.float64)
        self.valid_mks = _cupy_array("valid_mks", valid_mks, np.bool_)
        self.n_markers = np.int32(markers.shape[0])
        self.n_cols = np.int32(markers.shape[1])
        self.Np = np.int32(Np)
        self.vdim = np.int32(vdim)
        self.weight_idx = np.int32(weight_idx)
        self.first_diagnostics_idx = np.int32(first_diagnostics_idx)
        self.first_init_idx = np.int32(first_pusher_idx)
        self.first_shift_idx = np.int32(first_shift_idx)
        self.residual_idx = np.int32(residual_idx)
        self.first_free_idx = np.int32(first_free_idx)
        self.mu_idx = np.int32(mu_idx)
        self.bc_type = _cupy_array("bc_type", bc_type, np.int64)
        self._freeze()

    @property
    def n_threads(self) -> int:
        return int(self.n_markers)


class CudaDomainArguments(CudaArguments):
    """CUDA version of :class:`~struphy.kernel_arguments.pusher_args_kernels.DomainArguments` (same parameters)."""

    _names = ("kind_map", "params", "degree", "t1", "t2", "t3", "ind1", "ind2", "ind3", "cx", "cy", "cz")

    def __init__(self, kind_map: int, params, degree, t1, t2, t3, ind1, ind2, ind3, cx, cy, cz):
        self.kind_map = np.int32(kind_map)
        self.params = _cupy_array("params", params, np.float64)
        self.degree = _cupy_array("degree", degree, np.int64)
        self.t1 = _cupy_array("t1", t1, np.float64)
        self.t2 = _cupy_array("t2", t2, np.float64)
        self.t3 = _cupy_array("t3", t3, np.float64)
        self.ind1 = _cupy_array("ind1", ind1, np.int64)
        self.ind2 = _cupy_array("ind2", ind2, np.int64)
        self.ind3 = _cupy_array("ind3", ind3, np.int64)
        self.cx = _cupy_array("cx", cx, np.float64)
        self.cy = _cupy_array("cy", cy, np.float64)
        self.cz = _cupy_array("cz", cz, np.float64)
        self._freeze()


class CudaDerhamArguments(CudaArguments):
    """CUDA version of :class:`~struphy.kernel_arguments.pusher_args_kernels.DerhamArguments` (same parameters).

    The scratch arrays ``bn1, ..., bd3`` of the pyccel version are not needed; in CUDA they are per-thread local arrays.
    """

    _names = ("pn", "tn1", "tn2", "tn3", "starts")

    def __init__(self, pn, tn1, tn2, tn3, starts):
        self.pn = _cupy_array("pn", pn, np.int64)
        self.tn1 = _cupy_array("tn1", tn1, np.float64)
        self.tn2 = _cupy_array("tn2", tn2, np.float64)
        self.tn3 = _cupy_array("tn3", tn3, np.float64)
        self.starts = _cupy_array("starts", starts, np.int64)
        self._freeze()
