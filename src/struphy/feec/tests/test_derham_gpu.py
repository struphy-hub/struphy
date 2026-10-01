"""Derham on the CuPy backend: creation and the CUDA Derham arguments (CUDA_STRATEGY.md, PR 7).

The tests marked ``requires_cupy`` need a GPU; they compare the CuPy backend with the NumPy backend.
"""

import cunumpy
import numpy as np
import pytest

requires_cupy = pytest.mark.skipif(not cunumpy.cupy_available(), reason="CuPy/GPU not available")

DECOMPOSITION = ("domain_array", "index_array", "index_array_N", "index_array_D", "neighbours")


def make_derham(bcs=(None, None, None), local_projectors=False):
    """Derham on 8 x 6 x 4 elements with degrees (2, 3, 1), for the active cunumpy backend."""
    from feectools.ddm.mpi import mpi as MPI

    from struphy.feec.psydac_derham import Derham
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    options = DerhamOptions(degree=(2, 3, 1), bcs=bcs, local_projectors=local_projectors)
    return Derham(TensorProductGrid(num_elements=(8, 6, 4)), options, comm=MPI.COMM_WORLD)


def test_cuda_args_derham_needs_device_arrays():
    """On the NumPy backend the Derham arrays are host arrays, which are never copied to the device."""
    with cunumpy.use_backend("numpy"):
        derham = make_derham()
        with pytest.raises(TypeError, match="CuPy array"):
            derham.cuda_args_derham


@requires_cupy
@pytest.mark.parametrize("bcs", [(None, None, None), (("dirichlet", "free"), None, ("free", "dirichlet"))])
def test_derham_on_cupy(bcs):
    """Same decomposition and kernel arguments on both backends, and CUDA arguments holding device copies."""
    derhams = {}
    for backend in ("numpy", "cupy"):
        with cunumpy.use_backend(backend):
            derhams[backend] = make_derham(bcs)
    host, device = derhams["numpy"], derhams["cupy"]

    for name in DECOMPOSITION:
        assert cunumpy.is_gpu(getattr(device, name)), name
        assert np.array_equal(cunumpy.to_numpy(getattr(device, name)), getattr(host, name)), name

    # the pyccel arguments are host arrays on both backends (feectools knots are host arrays)
    for name in ("pn", "tn1", "tn2", "tn3", "starts"):
        assert np.array_equal(getattr(device.args_derham, name), getattr(host.args_derham, name)), name

    with cunumpy.use_backend("cupy"):
        args = device.cuda_args_derham
        assert device.cuda_args_derham is args  # built once
        expected = (host.args_derham.pn, *host.V0fem.knots, host.args_derham.starts)
        for name, value in zip(("pn", "tn1", "tn2", "tn3", "starts"), expected):
            assert cunumpy.is_gpu(getattr(args, name)), name
            assert np.array_equal(cunumpy.to_numpy(getattr(args, name)), value), name


@requires_cupy
def test_local_projectors_not_supported_on_cupy():
    """Local projectors have no device implementation yet; the Derham fails when it is created."""
    with cunumpy.use_backend("cupy"), pytest.raises(NotImplementedError, match="Local projectors"):
        make_derham(local_projectors=True)
