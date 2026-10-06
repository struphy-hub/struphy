"""CPU emulation of every CUDA kernel, compared with its pyccel kernel (runs without a GPU).

Uses the same ``<name>_test_args.py`` modules as the GPU parity test.
"""

import cunumpy as xp
import numpy as np
import pytest
from cunumpy.kernel_testing import emulation_compiler

from struphy.geometry.tests import spline_mapping_cases
from struphy.kernel_arguments.pusher_args_cuda import CudaDerhamArguments, CudaDomainArguments, CudaMarkerArguments
from struphy.pic.tests.cuda_emulation import emulate_struct_kernel
from struphy.pic.tests.test_cuda_parity import CUDA_CASES

# pyccel argument class name -> its CUDA version
CUDA_CLASSES = {
    cls.__name__.removeprefix("Cuda"): cls for cls in (CudaMarkerArguments, CudaDerhamArguments, CudaDomainArguments)
}

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
            n_threads = getattr(module, "N_THREADS", None)  # as in cunumpy's check_parity
            if callable(n_threads):
                n_threads = n_threads(emulated_args)
            emulate_struct_kernel(kernel.cuda_kernel, *emulated_args, n_threads=n_threads)
            host, emulated = arrays(host_args), arrays(emulated_args)
            assert any(not np.array_equal(a, b) for a, b in zip(before, host)), "the kernel changed nothing"
            for h, e in zip(host, emulated):
                np.testing.assert_allclose(e, h, rtol=rtol, atol=atol)


@requires_compiler
@pytest.mark.parametrize("degree", range(1, 9))
def test_emulated_der_splines(degree):
    """The device b_splines_slim and b_der_splines_slim, emulated on the CPU, agree with pyccel."""
    from cunumpy.cuda import CudaKernel

    from struphy.geometry.tests.spline_mapping_cases import DER_SPLINES_SOURCE, der_splines_case
    from struphy.utils.cuda_arguments import CUDA_OPTIONS

    knots, pts, expected = der_splines_case(degree)
    out = np.zeros(expected.shape)
    kernel = CudaKernel(DER_SPLINES_SOURCE, "evaluate_der_splines", **CUDA_OPTIONS)
    emulate_struct_kernel(kernel, knots, len(knots), degree, pts, out, len(pts), n_threads=len(pts))
    np.testing.assert_allclose(out, expected, rtol=1e-13, atol=1e-13)


@requires_compiler
@pytest.mark.parametrize("name", list(spline_mapping_cases.DOMAINS))
def test_emulated_spline_mappings(name):
    """The device spline mappings (kind_map 0-2) and their Jacobians, emulated on the CPU, agree with pyccel."""
    args_domain = spline_mapping_cases.host_domain(name).args_domain
    etas = spline_mapping_cases.points()
    expected = spline_mapping_cases.expected(args_domain, etas)
    out = np.zeros(expected.size)
    emulate_struct_kernel(
        spline_mapping_cases.make_kernel(),
        *spline_mapping_cases.flat_inputs(etas),
        args_domain,
        out,
        out.size,
        n_threads=out.size,
    )
    np.testing.assert_allclose(out.reshape(expected.shape), expected, rtol=1e-13, atol=1e-13)
