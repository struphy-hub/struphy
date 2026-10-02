"""CUDA counterparts of the pyccelized kernel argument classes.

The compiled classes in :mod:`struphy.kernel_arguments.pusher_args_kernels` only accept NumPy
arrays. The classes here take the same constructor arguments, keep references to **CuPy** arrays
(no copies, other arrays raise) and flatten them into the arguments of a ``cupy.RawKernel``.
The order of :attr:`values` is the corresponding part of the CUDA kernel signature.
"""

from abc import ABC, abstractmethod

import numpy as np


class Argument(ABC):
    """Base class for objects that provide arguments to a CUDA kernel."""

    @abstractmethod
    def get_cuda_args(self) -> tuple:
        """Return this object's arguments in CUDA kernel signature order."""
        raise NotImplementedError


def _cupy_array(name: str, arr, dtype):
    """Check that ``arr`` is a C-contiguous CuPy array of ``dtype`` (never converts or copies).

    Parameters
    ----------
    name : str
        Name of the argument, for error messages.

    arr : cupy.ndarray
        The array to check.

    dtype : type
        Expected dtype.

    Returns
    -------
    cupy.ndarray
        ``arr`` itself.
    """
    if not hasattr(arr, "__cuda_array_interface__"):
        raise TypeError(f"{name} must be a CuPy array, got {type(arr)}.")
    if arr.dtype != dtype or not arr.flags.c_contiguous:
        raise TypeError(f"{name} must be a C-contiguous array of dtype {np.dtype(dtype)}.")
    return arr


class CudaMarkerArguments(Argument):
    """CUDA version of :class:`~struphy.kernel_arguments.pusher_args_kernels.MarkerArguments`.

    CUDA signature of :attr:`values`: ``double* markers, bool* valid_mks, int n_markers, int n_cols, int Np,
    int vdim, int weight_idx, int first_diagnostics_idx, int first_init_idx, int first_shift_idx,
    int residual_idx, int first_free_idx, int mu_idx, long long* bc_type``

    Parameters
    ----------
    markers : cupy.ndarray[float]
        Markers array (C-contiguous, float64).

    valid_mks : cupy.ndarray[bool]
        True for valid markers (not holes or ghosts).

    Np : int
        Total number of particles.

    vdim : int
        Dimension of velocity space.

    weight_idx : int
        Column index of particle weight.

    first_diagnostics_idx : int
        Starting index for diagnostics columns.

    first_pusher_idx : int
        Starting buffer marker index number for pusher.

    first_shift_idx : int
        First index for storing shifts due to boundary conditions in eta-space.

    residual_idx : int
        Column for storing the residual in iterative pushers.

    first_free_idx : int
        First index for storing auxiliary quantities for each particle.

    mu_idx : int
        Column index of particle magnetic moment.

    bc_type : cupy.ndarray[int]
        Kinetic boundary condition in each logical direction (int64).

    Attributes
    ----------
    markers : cupy.ndarray[float]
        The markers array passed in (no copy).

    valid_mks : cupy.ndarray[bool]
        The array of valid markers passed in (no copy).

    n_markers : int
        Number of rows of ``markers``, e.g. for the number of CUDA threads.

    values : tuple
        Flat CUDA kernel arguments, see the signature above.
    """

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
        self.n_markers = markers.shape[0]
        self.n_cols = np.int32(markers.shape[1])
        self.Np = np.int32(Np)
        self.vdim = np.int32(vdim)
        self.weight_idx = np.int32(weight_idx)
        self.first_diagnostics_idx = np.int32(first_diagnostics_idx)
        self.first_pusher_idx = np.int32(first_pusher_idx)
        self.first_shift_idx = np.int32(first_shift_idx)
        self.residual_idx = np.int32(residual_idx)
        self.first_free_idx = np.int32(first_free_idx)
        self.mu_idx = np.int32(mu_idx)
        self.bc_type = _cupy_array("bc_type", bc_type, np.int64)

    def get_cuda_args(self) -> tuple:
        return (
            self.markers,
            self.valid_mks,
            np.int32(self.n_markers),
            self.n_cols,
            self.Np,
            self.vdim,
            self.weight_idx,
            self.first_diagnostics_idx,
            self.first_pusher_idx,
            self.first_shift_idx,
            self.residual_idx,
            self.first_free_idx,
            self.mu_idx,
            self.bc_type,
        )


class CudaDerhamArguments(Argument):
    """CUDA version of :class:`~struphy.kernel_arguments.pusher_args_kernels.DerhamArguments`.

    CUDA signature of :meth:`get_cuda_args`: ``long long* pn, double* tn1, double* tn2, double* tn3, long long* starts``

    The scratch arrays of the pyccel class (``bn1``, ..., ``bd3``) are not part of it; CUDA kernels use
    per-thread local arrays instead.

    Parameters
    ----------
    pn : cupy.ndarray[int]
        Spline degrees of :class:`~struphy.feec.psydac_derham.Derham` (int64).

    tn1, tn2, tn3 : cupy.ndarray[float]
        Knot sequences of :class:`~struphy.feec.psydac_derham.Derham`.

    starts : cupy.ndarray[int]
        Start indices (current MPI process) of :class:`~struphy.feec.psydac_derham.Derham` (int64).
    """

    def __init__(self, pn, tn1, tn2, tn3, starts):
        self.pn = _cupy_array("pn", pn, np.int64)
        self.tn1, self.tn2, self.tn3 = (_cupy_array("tn", t, np.float64) for t in (tn1, tn2, tn3))
        self.starts = _cupy_array("starts", starts, np.int64)

    def get_cuda_args(self) -> tuple:
        return (self.pn, self.tn1, self.tn2, self.tn3, self.starts)


class CudaDomainArguments(Argument):
    """CUDA version of :class:`~struphy.kernel_arguments.pusher_args_kernels.DomainArguments`.

    CUDA signature of :attr:`values`: ``int kind_map, double* params, long long* degree, double* t1,
    double* t2, double* t3, long long* ind1, long long* ind2, long long* ind3, double* cx, double* cy, double* cz``

    Parameters
    ----------
    kind_map : int
        Mapping identifier of :class:`~struphy.geometry.base.Domain`.

    params : cupy.ndarray[float]
        Mapping parameters.

    degree : cupy.ndarray[int]
        Spline degrees of the mapping.

    t1, t2, t3 : cupy.ndarray[float]
        Knot sequences of the mapping.

    ind1, ind2, ind3 : cupy.ndarray[int]
        Indices of non-vanishing splines in format (number of mapping grid cells, degree + 1).

    cx, cy, cz : cupy.ndarray[float]
        Spline coefficients (control points) of the mapping.

    Attributes
    ----------
    values : tuple
        Flat CUDA kernel arguments, see the signature above.
    """

    def __init__(self, kind_map: int, params, degree, t1, t2, t3, ind1, ind2, ind3, cx, cy, cz):
        self.kind_map = np.int32(kind_map)
        self.params = _cupy_array("params", params, np.float64)
        self.degree = _cupy_array("degree", degree, np.int64)
        self.t1, self.t2, self.t3 = (_cupy_array("t", t, np.float64) for t in (t1, t2, t3))
        self.ind1, self.ind2, self.ind3 = (_cupy_array("ind", ind, np.int64) for ind in (ind1, ind2, ind3))
        self.cx, self.cy, self.cz = (_cupy_array("c", c, np.float64) for c in (cx, cy, cz))

    def get_cuda_args(self) -> tuple:
        return (
            self.kind_map,
            self.params,
            self.degree,
            self.t1,
            self.t2,
            self.t3,
            self.ind1,
            self.ind2,
            self.ind3,
            self.cx,
            self.cy,
            self.cz,
        )
