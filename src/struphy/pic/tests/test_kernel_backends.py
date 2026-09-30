"""Proof of concept: 1:1 pyccel and CUDA kernels, dispatched by the cunumpy backend."""

import cunumpy
import numpy as np
import pytest

from struphy.pic.pushing.demo_cuda import make_demo_arguments, push_eta_linear

requires_cupy = pytest.mark.skipif(not cunumpy.cupy_available(), reason="CuPy/GPU not available")


@pytest.mark.parametrize("backend", ["numpy", pytest.param("cupy", marks=requires_cupy)])
def test_push_eta_linear(backend):
    """The kernel of the active backend updates the marker array in place."""
    dt = 0.1
    with cunumpy.use_backend(backend):
        args_markers, args_domain = make_demo_arguments(1000)
        markers = args_markers.markers
        valid = cunumpy.to_numpy(args_markers.valid_mks)
        expected = cunumpy.to_numpy(markers).copy()
        expected[valid, 0:3] += dt * expected[valid, 3:6]

        push_eta_linear(dt, 0, args_markers, args_domain)

        assert np.allclose(cunumpy.to_numpy(markers), expected, rtol=1e-14, atol=0.0)


@requires_cupy
def test_cuda_arguments_reject_host_arrays():
    """Host arrays are never copied to the device."""
    with cunumpy.use_backend("numpy"):
        host_markers, _ = make_demo_arguments(10)
    with cunumpy.use_backend("cupy"), pytest.raises(TypeError, match="CuPy array"):
        from struphy.utils.cuda_arguments import CudaMarkerArguments

        CudaMarkerArguments(host_markers.markers, host_markers.valid_mks, 10, 3, 6, 7, 8, 14, 17, 18, 4, np.zeros(3))
