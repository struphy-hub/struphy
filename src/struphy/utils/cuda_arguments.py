"""CUDA counterparts of the pyccelized kernel argument classes.

The compiled classes in :mod:`struphy.kernel_arguments.pusher_args_kernels` only accept NumPy
arrays. The classes here take the same constructor arguments, keep references to **CuPy** arrays
(no copies, other arrays raise) and flatten them into the arguments of a ``cupy.RawKernel``.
The order of :attr:`values` is the corresponding part of the CUDA kernel signature.
"""

import numpy as np


def _cupy_array(name: str, arr, dtype):
    """Return ``arr`` itself after checking it is a C-contiguous CuPy array of ``dtype``."""
    if not hasattr(arr, "__cuda_array_interface__"):
        raise TypeError(f"{name} must be a CuPy array, got {type(arr)}.")
    if arr.dtype != dtype or not arr.flags.c_contiguous:
        raise TypeError(f"{name} must be a C-contiguous array of dtype {np.dtype(dtype)}.")
    return arr


class CudaMarkerArguments:
    """CUDA version of :class:`~struphy.kernel_arguments.pusher_args_kernels.MarkerArguments`.

    CUDA signature: ``double* markers, bool* valid_mks, int n_markers, int n_cols, int Np, int vdim,
    int weight_idx, int first_diagnostics_idx, int first_init_idx, int first_shift_idx,
    int residual_idx, int first_free_idx, int mu_idx, long long* bc_type``
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

    CUDA signature: ``int kind_map, double* params, long long* degree, double* t1, double* t2, double* t3,
    long long* ind1, long long* ind2, long long* ind3, double* cx, double* cy, double* cz``
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
