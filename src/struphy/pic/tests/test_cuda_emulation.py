"""CPU emulation of every CUDA kernel, compared with its pyccel kernel (runs without a GPU).

Uses the same ``<name>_test_args.py`` modules as the GPU parity test.
"""

import cunumpy as xp
import numpy as np
import pytest
from cunumpy.kernel_testing import emulation_compiler

from struphy.kernel_arguments.pusher_args_cuda import CudaDerhamArguments, CudaDomainArguments, CudaMarkerArguments
from struphy.pic.tests.cuda_emulation import emulate_struct_kernel
from struphy.pic.tests.test_cuda_parity import CUDA_CASES

# pyccel argument class name -> its CUDA version
CUDA_CLASSES = {cls.__name__.removeprefix("Cuda"): cls for cls in (CudaMarkerArguments, CudaDerhamArguments, CudaDomainArguments)}

requires_compiler = pytest.mark.skipif(emulation_compiler() is None, reason="no C++ compiler")


def arrays(args):
    """The arrays among the arguments and inside the argument objects, as host copies."""
    out = []
    for a in args:
        if isinstance(a, np.ndarray):
            out.append(a.copy())
        elif type(a).__name__ in CUDA_CLASSES:  # a pyccel argument object: the fields of its CUDA struct
            fields = (getattr(a, f.name, None) for f in CUDA_CLASSES[type(a).__name__].struct.fields)
            out += [f.copy() for f in fields if isinstance(f, np.ndarray)]
    return out


@requires_compiler
@pytest.mark.parametrize("kernel", CUDA_CASES)
def test_emulated_parity(kernel):
    module = kernel.test_args
    rtol, atol = getattr(module, "RTOL", 1e-12), getattr(module, "ATOL", 0.0)
    with xp.use_backend("numpy"):
        for seed in range(len(module.CASES)):
            host_args = module.make_args("numpy", seed)
            emulated_args = module.make_args("numpy", seed)
            before = arrays(host_args)
            kernel(*host_args)
            emulate_struct_kernel(kernel.cuda_kernel, *emulated_args)
            host, emulated = arrays(host_args), arrays(emulated_args)
            assert any(not np.array_equal(a, b) for a, b in zip(before, host)), "the kernel changed nothing"
            for h, e in zip(host, emulated):
                np.testing.assert_allclose(e, h, rtol=rtol, atol=atol)
