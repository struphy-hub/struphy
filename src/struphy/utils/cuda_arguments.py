"""CUDA counterparts of the pyccelized kernel argument classes.

The compiled classes in :mod:`struphy.kernel_arguments.pusher_args_kernels` only accept NumPy
arrays. The classes here take the same constructor arguments, keep references to **CuPy** arrays
(no copies, other arrays raise) and flatten them into the arguments of a ``cupy.RawKernel``.
The order of :attr:`values` is the corresponding part of the CUDA kernel signature.
"""

import numpy as np


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


class CudaMarkerArguments:
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
        self.values = (
            self.markers,
            self.valid_mks,
            *(
                np.int32(i)
                for i in (
                    markers.shape[0],
                    markers.shape[1],
                    Np,
                    vdim,
                    weight_idx,
                    first_diagnostics_idx,
                    first_pusher_idx,
                    first_shift_idx,
                    residual_idx,
                    first_free_idx,
                    mu_idx,
                )
            ),
            _cupy_array("bc_type", bc_type, np.int64),
        )


class CudaDomainArguments:
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
        self.values = (
            np.int32(kind_map),
            _cupy_array("params", params, np.float64),
            _cupy_array("degree", degree, np.int64),
            *(_cupy_array("t", t, np.float64) for t in (t1, t2, t3)),
            *(_cupy_array("ind", ind, np.int64) for ind in (ind1, ind2, ind3)),
            *(_cupy_array("c", c, np.float64) for c in (cx, cy, cz)),
        )
