"""Kernel argument bundles shared by the NumPy and CuPy backends.

cunumpy owns struct packing, scalar validation, host argument resolution and
copy/pickle handling. The compiled host classes remain plain pyccel classes.
"""

from pathlib import Path

import cunumpy as xp
import numpy as np
from cunumpy.cuda import write_cuda_header
from cunumpy.kernels import Kernel, PyccelStructArguments

from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


def _kernel_array(name, arr, dtype):
    """Validate owner arrays without copying or converting them."""
    if not isinstance(arr, np.ndarray):
        import cupy as cp

        if not isinstance(arr, cp.ndarray):
            raise TypeError(f"{name} must be a NumPy or CuPy array")
        if arr.device.id != cp.cuda.runtime.getDevice():
            raise ValueError(f"{name} must be on the current CUDA device")
    if arr.dtype != dtype or not arr.flags.c_contiguous:
        raise TypeError(f"{name} must be a C-contiguous array of dtype {np.dtype(dtype)}")
    return arr


class CudaMarkerArguments(PyccelStructArguments):
    """MarkerArguments on the host; MarkerArgs by value on CUDA."""

    struct_name = "MarkerArgs"
    fields = (
        ("markers", "double*"),
        ("valid_mks", "bool*"),
        ("n_markers", "int"),
        ("n_cols", "int"),
        ("Np", "int"),
        ("vdim", "int"),
        ("weight_idx", "int"),
        ("first_diagnostics_idx", "int"),
        ("first_init_idx", "int"),
        ("first_shift_idx", "int"),
        ("residual_idx", "int"),
        ("first_free_idx", "int"),
        ("mu_idx", "int"),
        ("bc_type", "long long*"),
    )
    host_class = MarkerArguments
    host_fields = (
        "markers",
        "valid_mks",
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
        self.markers = _kernel_array("markers", markers, np.float64)
        self.valid_mks = _kernel_array("valid_mks", valid_mks, np.bool_)
        self.n_markers = markers.shape[0]
        self.n_cols = markers.shape[1]
        self.Np = Np
        self.vdim = vdim
        self.weight_idx = weight_idx
        self.first_diagnostics_idx = first_diagnostics_idx
        self.first_init_idx = first_pusher_idx
        self.first_shift_idx = first_shift_idx
        self.residual_idx = residual_idx
        self.first_free_idx = first_free_idx
        self.mu_idx = mu_idx
        self.bc_type = _kernel_array("bc_type", bc_type, np.int64)
        if self.has_device_arrays():
            self.pack()


class CudaDerhamArguments(PyccelStructArguments):
    """DerhamArguments on the host; DerhamArgs by value on CUDA."""

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
    host_class = DerhamArguments
    host_fields = ("pn", "tn1", "tn2", "tn3", "starts")

    def __init__(self, pn, tn1, tn2, tn3, starts):
        self.pn = _kernel_array("pn", pn, np.int64)
        if self.has_device_arrays() and bool(((pn < 1) | (pn > 8)).any()):
            raise ValueError("CUDA spline degrees must be between 1 and 8.")
        self.tn1, self.tn2, self.tn3 = (_kernel_array("tn", t, np.float64) for t in (tn1, tn2, tn3))
        self.starts = _kernel_array("starts", starts, np.int64)
        self.nt1, self.nt2, self.nt3 = len(tn1), len(tn2), len(tn3)
        if self.has_device_arrays():
            self.pack()


class CudaDomainArguments(PyccelStructArguments):
    """DomainArguments on the host; DomainArgs by value on CUDA."""

    struct_name = "DomainArgs"
    fields = (
        ("kind_map", "int"),
        ("params", "double*"),
        ("degree", "long long*"),
        ("t1", "double*"),
        ("t2", "double*"),
        ("t3", "double*"),
        ("ind1", "long long*"),
        ("ind2", "long long*"),
        ("ind3", "long long*"),
        ("cx", "double*"),
        ("cy", "double*"),
        ("cz", "double*"),
    )
    host_class = DomainArguments
    host_fields = ("kind_map", "params", "degree", "t1", "t2", "t3", "ind1", "ind2", "ind3", "cx", "cy", "cz")
    host_copies = True  # Existing read-only geometry evaluations still use pyccel.

    def __init__(self, kind_map: int, params, degree, t1, t2, t3, ind1, ind2, ind3, cx, cy, cz):
        self.kind_map = kind_map
        self.params = _kernel_array("params", params, np.float64)
        self.degree = _kernel_array("degree", degree, np.int64)
        self.t1, self.t2, self.t3 = (_kernel_array("t", t, np.float64) for t in (t1, t2, t3))
        self.ind1, self.ind2, self.ind3 = (_kernel_array("ind", ind, np.int64) for ind in (ind1, ind2, ind3))
        self.cx, self.cy, self.cz = (_kernel_array("c", c, np.float64) for c in (cx, cy, cz))
        if self.has_device_arrays():
            self.pack()


CUDA_STRUCTS = tuple(cls.struct for cls in (CudaMarkerArguments, CudaDerhamArguments, CudaDomainArguments))
CUDA_INCLUDE_DIR = Path(__file__).resolve().parents[2]
CUDA_OPTIONS = {"structs": CUDA_STRUCTS, "include_dirs": (CUDA_INCLUDE_DIR,)}


def prepare_kernel(kernel) -> Kernel:
    """The kernel as a :class:`cunumpy.kernels.Kernel`, checked for the active backend at setup.

    A plain function or ``PyccelKernel`` becomes a ``Kernel`` without CUDA version. On the CuPy backend, a kernel
    without CUDA version raises here, at setup, not in the time loop, and the CUDA kernel is compiled now.
    """
    if not isinstance(kernel, Kernel):
        kernel = Kernel(kernel)
    if xp.cupy_backend:
        if not kernel.has_cuda:
            raise NotImplementedError(f"No CUDA version of kernel {kernel.name!r} (expected {kernel.cuda_path}).")
        kernel.compile()
    return kernel


def write_pusher_header(path):
    """Generate the committed ABI header from the cunumpy field definitions."""
    source = write_cuda_header(path, CUDA_STRUCTS, guard="STRUPHY_PUSHER_ARGS_CUH")
    source += "\n#define MARKER(args, ip, j) ((args).markers[(long long)(ip) * (args).n_cols + (j)])\n"
    Path(path).write_text(source)
    return source
