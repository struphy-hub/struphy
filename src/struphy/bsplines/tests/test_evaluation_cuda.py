"""Shared spline-evaluation signatures and CUDA view indexing."""

from itertools import product

import cunumpy as xp
import numpy as np
import pytest

from struphy.bsplines import evaluation

requires_cupy = pytest.mark.skipif(not xp.cupy_available(), reason="CuPy/GPU not available")


@requires_cupy
@pytest.mark.parametrize("mode", ["markers", "matrix", "sparse_meshgrid"])
@pytest.mark.parametrize("kind", list(product((0, 1), repeat=3)))
@pytest.mark.parametrize("empty", [False, True])
def test_evaluation_parity(mode, kind, empty):
    """Identical arguments, strided data/output, flagged points and nonzero starts."""
    kernel = getattr(evaluation, "eval_spline_mpi_" + mode)
    rng = np.random.default_rng(34)
    coeff = rng.normal(size=(32, 36, 40))
    degree = np.array([2, 3, 1], dtype=np.int64)
    knots = [np.r_[np.zeros(p), np.linspace(0, 1, 9), np.ones(p)] for p in degree]
    results = []
    for backend in ("numpy", "cupy"):
        with xp.use_backend(backend):
            data = xp.asarray(coeff)[::2, ::2, ::2]
            metadata = (
                xp.asarray(kind, dtype=xp.int64),
                xp.asarray(degree),
                *(xp.asarray(np.repeat(t, 2))[::2] for t in knots),
                xp.asarray([1, 1, 1], dtype=xp.int64),
            )
            if mode == "markers":
                points = xp.full((0 if empty else 129, 6), 0.5)[:, ::2]
                if not empty:
                    points[0, 0] = -1.0
                coords = (points,)
                storage = xp.full((points.shape[0] * 2,), 17.0)
                out = storage[::2]
            else:
                axes = [xp.linspace(0.3, 0.7, n) for n in (0 if empty else 7, 5, 4)]
                coords = (axes[0][:, None, None], axes[1][None, :, None], axes[2][None, None, :])
                if mode == "matrix":
                    coords = tuple(xp.broadcast_to(c, (axes[0].size, 5, 4)).copy() for c in coords)
                if not empty:
                    coords[1][0, 0, 0] = -1.0
                storage = xp.full((axes[0].size, 5, 8), 17.0)
                out = storage[:, :, ::2]
            kernel(*coords, data, *metadata, out, n_threads=out.size)
            results.append(xp.to_numpy(out).copy())
            assert bool(xp.all(storage[..., 1::2] == 17.0))
    np.testing.assert_allclose(results[1], results[0], rtol=1e-12, atol=1e-12)


@requires_cupy
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_view_validation(ndim):
    from struphy.utils.kernel_backends import CudaKernel

    source = f"""#include "struphy/kernel_arguments/array_view.cuh"
    extern "C" __global__ void check(Array{ndim}D<double> a) {{ }}"""
    kernel = CudaKernel(source, "check")
    with xp.use_backend("cupy"):
        for bad in (np.zeros((2,) * ndim), xp.zeros((2,) * ndim, dtype=xp.float32), xp.zeros((2,) * (ndim + 1))):
            with pytest.raises(TypeError, match=f"Array{ndim}D"):
                kernel(bad, n_threads=1)


@pytest.mark.parametrize("space", ["H1", "Hcurl", "Hdiv", "L2"])
@pytest.mark.parametrize("mode", ["markers", "matrix", "sparse_meshgrid"])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("target_backend", ["numpy", "cupy"])
def test_spline_function_backends(space, mode, local, target_backend):
    from struphy.feec.tests.test_derham_gpu import make_derham

    if target_backend == "cupy" and not xp.cupy_available():
        pytest.skip("CuPy/GPU not available")
    results = []
    for backend in ("numpy", target_backend):
        with xp.use_backend(backend):
            try:
                derham = make_derham()
            except TypeError as exc:
                if backend == "cupy" and "Implicit conversion to a NumPy array" in str(exc):
                    pytest.xfail("feectools Derham construction mixes NumPy breaks and CuPy indices")
                raise
            spline = derham.create_spline_function("test", space)
            vector = spline.vector
            components = [vector] if space in ("H1", "L2") else [vector[i] for i in range(3)]
            for i, component in enumerate(components):
                component._data[:] = i + 1.0
            axes = [np.array([0.2, 0.6]), np.array([0.3, 0.7]), np.array([0.4])]
            if mode == "markers":
                coords = (xp.asarray([[0.2, 0.3, 0.4], [0.6, 0.7, 0.4], [-0.1, 0.3, 0.4]]),)
                shape = (3,)
            elif mode == "matrix":
                coords = tuple(xp.asarray(a) for a in np.meshgrid(*axes, indexing="ij"))
                shape = (2, 2, 1)
            else:
                coords = tuple(xp.asarray(a) for a in axes)
                shape = (2, 2, 1)
            expected = spline(*coords, local=local)
            out = xp.empty(shape) if len(components) == 1 else [xp.empty(shape) for _ in components]
            tmp = xp.full(shape, 99.0)
            actual = spline(*coords, out=out, tmp=tmp, local=local)
            assert actual is out
            values = [actual] if len(components) == 1 else actual
            expected_values = [expected] if len(components) == 1 else expected
            for a, e in zip(values, expected_values):
                np.testing.assert_allclose(xp.to_numpy(a), xp.to_numpy(e))
            squeezed = spline(*coords, squeeze_out=True, local=local)
            for a, e in zip([squeezed] if len(components) == 1 else squeezed, values):
                np.testing.assert_allclose(xp.to_numpy(a), np.squeeze(xp.to_numpy(e)))
            results.append([xp.to_numpy(v).copy() for v in values])
    for a, b in zip(*results):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
