"""Shared calls for particle helpers on the NumPy and CuPy backends."""

from pathlib import Path

import cunumpy as xp
from cunumpy.cuda import CudaKernel
from cunumpy.kernels import Kernel, PyccelKernel

from struphy.pic.pushing import pusher_utilities_kernels
from struphy.utils.cuda_arguments import CUDA_OPTIONS


_reflect = Kernel(
    PyccelKernel(pusher_utilities_kernels.reflect),
    CudaKernel.from_file(Path(__file__).with_name("reflect_cuda.cu"), **CUDA_OPTIONS),
)


def reflect(markers, args_domain, outside_inds, axis):
    """Reflect selected velocities in place, with the same arguments on both backends.

    Positions must already be inside the logical cube. CUDA supports Cuboid
    mappings; the NumPy backend retains all mappings supported by Pyccel.
    """
    if xp.get_backend() == "cupy" and args_domain.kind_map != 10:
        raise NotImplementedError("CUDA reflection currently supports only Cuboid mappings.")
    _reflect(markers, args_domain, outside_inds, axis, n_threads=outside_inds.size)
