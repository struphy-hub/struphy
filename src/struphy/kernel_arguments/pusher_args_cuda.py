"""CUDA versions of the argument classes in :mod:`struphy.kernel_arguments.pusher_args_kernels`.

Each class here corresponds to the pyccel class of the same name without the ``Cuda`` prefix: it takes the
same constructor arguments and has the same attributes, but holds CuPy arrays and is passed to CUDA kernels
as one C struct (``kernel_arguments/pusher_args.cuh``, generated from :attr:`fields`). The owners
(:class:`~struphy.pic.base.Particles`, :class:`~struphy.feec.psydac_derham.Derham`,
:class:`~struphy.geometry.base.Domain`) create the pyccel class on the NumPy backend and the CUDA class on the
CuPy backend.
"""

import numpy as np
from cunumpy.arguments import CudaStructArguments


def _device_array(name, arr, dtype, ndim=None):
    """Check that ``arr`` is a C-contiguous CuPy array of ``dtype`` (and ``ndim``) on the current device (never copies)."""
    import cupy as cp

    if not isinstance(arr, cp.ndarray):
        raise TypeError(f"{name} must be a CuPy array, got {type(arr)}")
    if arr.device.id != cp.cuda.runtime.getDevice():
        raise ValueError(f"{name} must be on the current CUDA device")
    if arr.dtype != dtype or not arr.flags.c_contiguous:
        raise TypeError(f"{name} must be a C-contiguous array of dtype {np.dtype(dtype)}")
    if ndim is not None and arr.ndim != ndim:
        raise TypeError(f"{name} must be a {ndim}-dimensional array, got {arr.ndim} dimensions")
    return arr


class CudaMarkerArguments(CudaStructArguments):
    """CUDA version of :class:`~struphy.kernel_arguments.pusher_args_kernels.MarkerArguments` (``MarkerArgs``)."""

    struct_name = "MarkerArgs"
    fields = (
        ("markers", "Array2D<double>"),
        ("valid_mks", "bool*"),
        ("n_markers", "int"),
        ("Np", "int"),
        ("vdim", "int"),
        ("weight_idx", "int"),
        ("first_diagnostics_idx", "int"),
        ("first_pusher_idx", "int"),
        ("first_shift_idx", "int"),
        ("residual_idx", "int"),
        ("first_free_idx", "int"),
        ("mu_idx", "int"),
        ("bc_type", "long long*"),
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
        self.markers = _device_array("markers", markers, np.float64)
        if markers.ndim != 2:
            raise TypeError("markers must be a two-dimensional array.")
        self.valid_mks = _device_array("valid_mks", valid_mks, np.bool_)
        self.Np = Np
        self.vdim = vdim
        self.weight_idx = weight_idx
        self.n_markers = markers.shape[0]
        self.first_diagnostics_idx = first_diagnostics_idx
        self.first_pusher_idx = first_pusher_idx
        self.first_shift_idx = first_shift_idx
        self.residual_idx = residual_idx
        self.first_free_idx = first_free_idx
        self.mu_idx = mu_idx
        self.bc_type = _device_array("bc_type", bc_type, np.int64)
        self.pack()


class CudaDerhamArguments(CudaStructArguments):
    """CUDA version of :class:`~struphy.kernel_arguments.pusher_args_kernels.DerhamArguments` (``DerhamArgs``).

    The pyccel scratch arrays ``bn1``, ..., ``bd3`` are per-thread local arrays in the CUDA kernels, so they are
    not part of the struct; CUDA pointers do not carry a length, so the knot lengths ``nt1``, ``nt2``, ``nt3`` are.
    """

    struct_name = "DerhamArgs"
    fields = (
        ("pn", "long long*"),
        ("tn1", "double*"),
        ("tn2", "double*"),
        ("tn3", "double*"),
        ("starts", "long long*"),
        ("nt1", "int"),
        ("nt2", "int"),
        ("nt3", "int"),
    )

    def __init__(self, pn, tn1, tn2, tn3, starts):
        self.pn = _device_array("pn", pn, np.int64)
        if bool(((pn < 1) | (pn > 8)).any()):
            raise ValueError("CUDA spline degrees must be between 1 and 8.")
        self.tn1, self.tn2, self.tn3 = (_device_array("tn", t, np.float64) for t in (tn1, tn2, tn3))
        self.starts = _device_array("starts", starts, np.int64)
        self.nt1, self.nt2, self.nt3 = len(tn1), len(tn2), len(tn3)
        self.pack()


class CudaDomainArguments(CudaStructArguments):
    """CUDA version of :class:`~struphy.kernel_arguments.pusher_args_kernels.DomainArguments` (``DomainArgs``).

    The knot vectors ``t1``, ``t2``, ``t3``, the spline indices ``ind1``, ``ind2``, ``ind3`` and the control points
    ``cx``, ``cy``, ``cz`` of the spline mappings are array views (pointer, shape, strides), because the device
    needs their shapes: ``find_span`` needs the number of knots, which a CUDA pointer does not carry.
    """

    struct_name = "DomainArgs"
    fields = (
        ("kind_map", "int"),
        ("params", "double*"),
        ("degree", "long long*"),
        ("t1", "Array1D<double>"),
        ("t2", "Array1D<double>"),
        ("t3", "Array1D<double>"),
        ("ind1", "Array2D<long long>"),
        ("ind2", "Array2D<long long>"),
        ("ind3", "Array2D<long long>"),
        ("cx", "Array3D<double>"),
        ("cy", "Array3D<double>"),
        ("cz", "Array3D<double>"),
    )

    def __init__(self, kind_map: int, params, degree, t1, t2, t3, ind1, ind2, ind3, cx, cy, cz):
        self.kind_map = kind_map
        self.params = _device_array("params", params, np.float64)
        self.degree = _device_array("degree", degree, np.int64)
        if kind_map < 10 and bool((degree > 8).any()):
            raise ValueError("CUDA spline mapping degrees must be at most 8.")
        self.t1, self.t2, self.t3 = (_device_array("t", t, np.float64, ndim=1) for t in (t1, t2, t3))
        self.ind1, self.ind2, self.ind3 = (_device_array("ind", ind, np.int64, ndim=2) for ind in (ind1, ind2, ind3))
        self.cx, self.cy, self.cz = (_device_array("c", c, np.float64, ndim=3) for c in (cx, cy, cz))
        self.pack()
