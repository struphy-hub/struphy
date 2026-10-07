"""CUDA build options and headers for the kernel argument classes.

The argument classes come in pairs in :mod:`struphy.kernel_arguments`: a pyccel class for the NumPy backend
(e.g. ``pusher_args_kernels.MarkerArguments``) and a CUDA class for the CuPy backend
(e.g. ``pusher_args_cuda.CudaMarkerArguments``). This module collects the C structs of the CUDA classes for
building CUDA kernels and for generating the committed headers.
"""

from pathlib import Path

from cunumpy.arguments import write_cuda_header

from struphy.kernel_arguments.local_projectors_args_cuda import CudaLocalProjectorsArguments
from struphy.kernel_arguments.pusher_args_cuda import CudaDerhamArguments, CudaDomainArguments, CudaMarkerArguments

PUSHER_STRUCTS = tuple(cls.struct for cls in (CudaMarkerArguments, CudaDerhamArguments, CudaDomainArguments))
LOCAL_PROJECTORS_STRUCTS = (CudaLocalProjectorsArguments.struct,)
CUDA_STRUCTS = PUSHER_STRUCTS + LOCAL_PROJECTORS_STRUCTS
CUDA_INCLUDE_DIR = Path(__file__).resolve().parents[2]
CUDA_OPTIONS = {"structs": CUDA_STRUCTS, "include_dirs": (CUDA_INCLUDE_DIR,)}


def write_pusher_header(path):
    """Generate the committed ABI header ``pusher_args.cuh`` from the CUDA argument classes."""
    return write_cuda_header(path, PUSHER_STRUCTS, guard="STRUPHY_PUSHER_ARGS_CUH")


def write_local_projectors_header(path):
    """Generate the committed ABI header ``local_projectors_args.cuh`` from the CUDA argument class."""
    return write_cuda_header(path, LOCAL_PROJECTORS_STRUCTS, guard="STRUPHY_LOCAL_PROJECTORS_ARGS_CUH")
