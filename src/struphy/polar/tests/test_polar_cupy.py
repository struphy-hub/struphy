"""Polar splines on the CuPy backend (struphy-hub/struphy#695, CUDA_STRATEGY.md).

The polar blocks (:class:`~struphy.polar.extraction_operators.PolarExtractionBlocksC1`) are built on the host on
every backend; on CuPy the polar operators apply device copies of them (``cupyx.scipy.sparse``) to the device
coefficients. :func:`check_polar_backends` builds the same polar Derham on the NumPy and the CuPy backend and
compares extraction operators, derivatives, ``PolarVector`` arithmetic, mass matrices, projectors and polar
``SplineFunction`` coefficients.

Without a GPU it runs on cunumpy's fake CuPy (which rejects host/device mixing like CuPy), in a subprocess, with
two substitutes for what the fake cannot do: the device sparse module (``cupyx.scipy.sparse``) is replaced by
:class:`DenseDeviceSparse` (dense products of device arrays), and kernels are called in their host version (the
fake cannot launch CUDA kernels; the feectools stencil kernels then copy their arrays to the host and back).
"""

import subprocess
import sys

import cunumpy
import numpy as np
import pytest

from struphy.geometry.tests.test_domain import _cupy_installed, serial_child_env

requires_cupy = pytest.mark.skipif(not cunumpy.cupy_available(), reason="CuPy/GPU not available")

MAPPINGS = ("IGAPolarCylinder", "IGAPolarTorus")
FORMS = ("0", "1", "2", "3", "v")
SPACE_IDS = {"0": "H1", "1": "Hcurl", "2": "Hdiv", "3": "L2", "v": "H1vec"}


class DenseDeviceSparse:
    """Stand-in for ``cupyx.scipy.sparse`` on the fake CuPy: the block as a dense array of the active backend."""

    class csr_matrix:
        def __init__(self, host):
            self._array = cunumpy.asarray(host.toarray())
            self.shape = host.shape

        def dot(self, x):
            return self._array @ x


def make_domain(mapping):
    from struphy import domains

    if mapping == "IGAPolarCylinder":
        return domains.IGAPolarCylinder(num_elements=(6, 6), degree=(2, 3), a=1.2, Lz=3.0)
    if mapping == "IGAPolarTorus":
        return domains.IGAPolarTorus(num_elements=(6, 6), degree=(3, 2), a=0.8, R0=3.0, sfl=True)
    raise ValueError(mapping)


def make_polar_derham(mapping, with_mass=True):
    """Polar Derham (with projectors), its domain and mass matrices (or None), on the active backend."""
    from maybempi import MPI

    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.psydac_derham import Derham
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    domain = make_domain(mapping)
    options = DerhamOptions(
        degree=(domain.degree[0], domain.degree[1], 1),
        bcs=(("dirichlet", "free"), None, None),
        nquads_proj=(4, 4, 2),
        polar_splines=True,
    )
    derham = Derham(TensorProductGrid(num_elements=(6, 6, 4)), options, comm=MPI.COMM_WORLD, domain=domain)
    return domain, derham, WeightedMassOperators(derham, domain) if with_mass else None


def _stencils(vec):
    """The Stencil- (or Block-) vector part of vec as a list of StencilVectors."""
    from feectools.linalg.stencil import StencilVector

    tp = vec.tp if hasattr(vec, "tp") else vec
    return [tp] if isinstance(tp, StencilVector) else list(tp.blocks)


def to_host(vec):
    """Data of a Stencil-/Block-/PolarVector as a list of NumPy arrays (polar coeffs first)."""
    pol = list(vec.pol) if hasattr(vec, "pol") else []
    return [cunumpy.to_numpy(a) for a in pol] + [cunumpy.to_numpy(s._data) for s in _stencils(vec)]


def fill(vec, seed):
    """Fill vec (on the active backend) with reproducible random values (same on every backend)."""
    rng = np.random.default_rng(seed)
    if hasattr(vec, "pol"):
        vec.pol = [cunumpy.asarray(rng.random(p.shape)) for p in vec.pol]
    for s in _stencils(vec):
        s._data[...] = cunumpy.asarray(rng.random(s._data.shape))
    if hasattr(vec, "pol"):
        vec.set_tp_coeffs_to_zero()
    vec.update_ghost_regions()
    return vec


def assert_on_backend(vec, backend):
    """All data of vec lives on the given backend."""
    pol = list(vec.pol) if hasattr(vec, "pol") else []
    for a in pol + [s._data for s in _stencils(vec)]:
        assert cunumpy.is_gpu(a) == (backend == "cupy"), type(a)


def polar_results(mapping, backend, with_mass=True):
    """Results of the polar operations on `backend`, as host arrays (dict name -> list of arrays)."""
    from struphy.polar.basic import PolarVector

    res = {}
    with cunumpy.use_backend(backend):
        domain, derham, mass = make_polar_derham(mapping, with_mass)

        def record(name, vec):
            assert_on_backend(vec, backend)
            res[name] = to_host(vec)

        for n, form in enumerate(FORMS):
            E = derham.extraction_ops[form]
            x = fill(E.domain.zeros(), seed=10 + n)
            y = fill(PolarVector(E.codomain), seed=20 + n)
            record(f"E{form}", E.dot(x))
            record(f"E{form}T", E.transpose().dot(y))
            P = derham.dofs_extraction_ops[form]
            record(f"P{form}", P.dot(x))
            record(f"P{form}T", P.transpose().dot(y))

            # PolarVector arithmetic
            z = fill(PolarVector(E.codomain), seed=30 + n)
            record(f"add{form}", y + z)
            record(f"sub{form}", y - z)
            record(f"mul{form}", y * 1.5)
            record(f"rmul{form}", 0.5 * y)
            record(f"neg{form}", -y)
            w = y.copy()
            w += z
            w -= 2.0 * z
            w *= 3.0
            record(f"inplace{form}", w)
            res[f"dot{form}"] = [np.asarray(float(y.dot(z)))]
            res[f"toarray{form}"] = [cunumpy.to_numpy(y.toarray())]

            # mass matrices (E M E^T with the polar extraction operators)
            if with_mass:
                record(f"M{form}", getattr(mass, "M" + form).dot(y))

        for name, op in (("grad", derham.grad), ("curl", derham.curl), ("div", derham.div)):
            x = fill(PolarVector(op.domain), seed=40)
            y = fill(PolarVector(op.codomain), seed=41)
            record(name, op.dot(x))
            record(name + "T", op.transpose().dot(y))

        # commuting projectors (polar case: iterative solve with the polar extraction operators)
        def f(e1, e2, e3):
            return cunumpy.sin(np.pi * e1) * cunumpy.cos(2 * np.pi * e2) * cunumpy.cos(2 * np.pi * e3) + e1**2

        record("Pi0", derham.P0(f))
        record("Pi1", derham.P1([f, f, f]))
        record("Pi2", derham.P2([f, f, f]))
        record("Pi3", derham.P3(f))

        # polar SplineFunction: polar coefficients, extracted tensor-product coefficients
        for form in ("0", "2"):
            field = derham.create_spline_function("f" + form, SPACE_IDS[form])
            assert isinstance(field.vector, PolarVector)
            fill(field.vector, seed=50)
            field.extract_coeffs()
            record("field" + form, field.vector)
            record("field_stencil" + form, field.vector_stencil)
            # restart: polar coefficients recovered from the extracted tensor-product coefficients
            restored = field._restart_extraction_op().dot(field.vector_stencil)
            record("restart" + form, restored)
            np.testing.assert_allclose(to_host(restored)[0], to_host(field.vector)[0], rtol=1e-10, atol=1e-12)

    return res


def check_polar_backends(mapping, with_mass=True):
    """The polar operations agree on the NumPy and the CuPy backend."""
    host = polar_results(mapping, "numpy", with_mass)
    device = polar_results(mapping, "cupy", with_mass)
    assert host.keys() == device.keys()
    for name in host:
        assert len(host[name]) == len(device[name]), name
        for a, b in zip(host[name], device[name]):
            np.testing.assert_allclose(b, a, rtol=1e-9, atol=1e-11, err_msg=name)


def check_polar_backends_fake_cupy(mapping):
    """:func:`check_polar_backends` on the fake CuPy (see the module docstring), without the mass matrices.

    The mass matrix weights are evaluated by struphy's CUDA geometry kernels, which take CUDA argument objects
    and cannot fall back to their pyccel versions; the polar mass matrices are compared on the GPU only.
    """
    from cunumpy._dispatch import Kernel

    from struphy.polar import linear_operators

    Kernel.get_kernel = lambda self: self._host_kernel
    linear_operators.set_device_sparse_module(DenseDeviceSparse)
    check_polar_backends(mapping, with_mass=False)


def test_device_blocks_numpy_backend():
    """On the NumPy backend the polar operators apply their host (SciPy) blocks, as before."""
    import scipy.sparse as sp

    from struphy.polar.linear_operators import DeviceSparseMatrix, set_device_sparse_module

    with cunumpy.use_backend("numpy"):
        domain, derham, _ = make_polar_derham("IGAPolarCylinder", with_mass=False)
        E = derham.extraction_ops["1"]
        assert E._backend_blocks("blocks_ten_to_pol") is E.blocks_ten_to_pol
        assert E._backend_blocks("blocks_e3") is E.blocks_e3
        assert all(sp.issparse(b) for row in E.blocks_ten_to_pol for b in row if b is not None)

        # the device copy (here with the dense stand-in on the NumPy backend) computes the same products
        previous = set_device_sparse_module(DenseDeviceSparse)
        try:
            rng = np.random.default_rng(0)
            for row in derham.dofs_extraction_ops["1"].blocks_ten_to_ten + E.blocks_ten_to_pol:
                for blk in row:
                    if blk is None:
                        continue
                    x = rng.random((blk.shape[1], 3))
                    np.testing.assert_allclose(DeviceSparseMatrix(blk).dot(x), blk.dot(x), rtol=1e-14)
            empty = DeviceSparseMatrix(sp.csr_matrix((0, 4)))
            assert empty.dot(np.ones((4, 2))).shape == (0, 2)
        finally:
            set_device_sparse_module(previous)


@pytest.mark.skipif(_cupy_installed(), reason="the fake CuPy cannot replace an installed CuPy")
@pytest.mark.parametrize("mapping", MAPPINGS)
def test_polar_fake_cupy(mapping):
    """Without a GPU: polar Derham on cunumpy's fake CuPy agrees with the NumPy backend.

    Runs in a subprocess because the fake CuPy must be installed before cunumpy is imported.
    """
    code = (
        "from struphy.polar.tests.test_polar_cupy import check_polar_backends_fake_cupy; "
        f"check_polar_backends_fake_cupy({mapping!r})"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=serial_child_env(CUNUMPY_FAKE_CUPY="1"), capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-4000:]


@requires_cupy
@pytest.mark.parametrize("mapping", MAPPINGS)
def test_polar_on_cupy(mapping):
    """Polar Derham on the GPU (cupyx.scipy.sparse for the polar blocks) agrees with the NumPy backend."""
    check_polar_backends(mapping)
