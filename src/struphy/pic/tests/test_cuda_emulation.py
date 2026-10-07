"""CPU emulation of every CUDA pusher and accumulation kernel, compared with its pyccel kernel (runs without a GPU)."""

import cunumpy as xp
import numpy as np
import pytest
from cunumpy.kernel_testing import emulation_compiler

from struphy.ode.utils import ButcherTableau
from struphy.pic.accumulation.kernels import catalog as accum_catalog
from struphy.pic.pushing.kernels import catalog
from struphy.pic.tests.cuda_emulation import emulate_struct_kernel
from struphy.pic.tests.test_cuda_parity import ACCUM_CUDA_NAMES, ACCUM_FACTORIES, CUDA_NAMES, FACTORIES

requires_compiler = pytest.mark.skipif(emulation_compiler() is None, reason="no C++ compiler")


def arrays(args):
    """The arrays among the arguments and inside the argument objects, as host copies."""
    out = []
    for a in args:
        if isinstance(a, np.ndarray):
            out.append(a.copy())
        elif hasattr(a, "host_fields"):
            out += [getattr(a, f).copy() for f in a.host_fields if isinstance(getattr(a, f), np.ndarray)]
    return out


@requires_compiler
@pytest.mark.parametrize("name", CUDA_NAMES)
@pytest.mark.parametrize("bc", [(0, 0, 0), (1, 1, 1), (2, 0, 1)])
@pytest.mark.parametrize("method", ["forward_euler", "rk4"])
def test_emulated_parity(name, bc, method):
    if name != "push_eta_stage" and method != "forward_euler":
        pytest.skip("only push_eta_stage depends on the Butcher tableau")
    kernel = catalog[name]
    stages = ButcherTableau(method).n_stages if name == "push_eta_stage" else 1
    with xp.use_backend("numpy"):
        host_args = FACTORIES[name](bc, method)
        emulated_args = FACTORIES[name](bc, method)
        n = host_args[0].n_markers
        for stage in range(stages):
            kernel.host_kernel(0.2, stage, *host_args)
            emulate_struct_kernel(kernel.cuda_kernel, 0.2, stage, *emulated_args, n_threads=n)
            for host, emulated in zip(arrays(host_args), arrays(emulated_args)):
                np.testing.assert_allclose(emulated, host, rtol=1e-13, atol=1e-14)


@requires_compiler
@pytest.mark.parametrize("name", ACCUM_CUDA_NAMES)
@pytest.mark.parametrize("bc", [(0, 0, 0), (2, 0, 1)])
def test_emulated_accum_parity(name, bc):
    kernel = accum_catalog[name]
    with xp.use_backend("numpy"):
        host_args = ACCUM_FACTORIES[name](bc)
        emulated_args = ACCUM_FACTORIES[name](bc)
        kernel.host_kernel(*host_args)
        emulate_struct_kernel(kernel.cuda_kernel, *emulated_args, n_threads=host_args[0].n_markers)
        assert any(np.any(a != 0) for a in arrays(host_args[3:])), "nothing was accumulated"
        for host, emulated in zip(arrays(host_args), arrays(emulated_args)):
            np.testing.assert_allclose(emulated, host, rtol=1e-13, atol=1e-14)
