"""Polar splines on the CuPy backend (struphy-hub/struphy#695, CUDA_STRATEGY.md).

The polar blocks (:class:`~struphy.polar.homotopy.derham_blocks.PolarExtractionBlocksHomotopy`) are built on the host on
every backend; on CuPy the polar operators apply device copies of them (``cupyx.scipy.sparse``) to the device
coefficients. :func:`check_polar_backends` builds the same polar Derham on the NumPy and the CuPy backend and
compares extraction operators, derivatives, ``PolarVector`` arithmetic, mass matrices, projectors and polar
``SplineFunction`` coefficients.

Without a GPU it runs on cunumpy's fake CuPy with CPU-emulated CUDA launches and a dense device stand-in
for cupyx sparse products and triangular solves. Host/device transfers remain visible to cunumpy's counters.
"""

import os
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

    class linalg:
        class SuperLU:
            """Device-array triangular solves with the same permutations as CuPy's SuperLU."""

            def __init__(self, factor):
                self.L = cunumpy.asarray(factor.L.toarray())
                self.U = cunumpy.asarray(factor.U.toarray())
                self.rows = cunumpy.asarray(np.argsort(factor.perm_r))
                self.cols = cunumpy.asarray(factor.perm_c)

            def solve(self, rhs):
                y = cunumpy.linalg.solve(self.L, rhs[self.rows])
                return cunumpy.linalg.solve(self.U, y)[self.cols]


def make_domain(mapping, num_elements=(6, 6)):
    from struphy import domains

    if mapping == "IGAPolarCylinder":
        return domains.IGAPolarCylinder(num_elements=num_elements, degree=(2, 3), a=1.2, Lz=3.0)
    if mapping == "IGAPolarTorus":
        return domains.IGAPolarTorus(num_elements=num_elements, degree=(3, 2), a=0.8, R0=3.0, sfl=True)
    raise ValueError(mapping)


def make_polar_derham(mapping, with_mass=True, poloidal_solver="schur", mpi_dims_mask=None):
    """Polar Derham (with projectors), its domain and mass matrices (or None), on the active backend."""
    from maybempi import MPI

    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.psydac_derham import Derham
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    size = MPI.COMM_WORLD.Get_size()
    # the process at the pole holds the polar rings and two tensor-product rings in eta1
    num_elements = (max(6, 6 * size), max(6, 3 * size), 4)
    domain = make_domain(mapping, num_elements[:2])
    options = DerhamOptions(
        degree=(domain.degree[0], domain.degree[1], 1),
        bcs=(("dirichlet", "free"), None, None),
        nquads_proj=(4, 4, 2),
        polar_splines=True,
    )
    derham = Derham(
        TensorProductGrid(
            num_elements=num_elements,
            mpi_dims_mask=mpi_dims_mask,
        ),
        options,
        comm=MPI.COMM_WORLD,
        domain=domain,
    )
    return domain, derham, WeightedMassOperators(derham, domain) if with_mass else None


def _stencils(vec):
    """The Stencil- (or Block-) vector part of vec as a list of StencilVectors."""
    from feectools.linalg.stencil import StencilVector

    tp = vec.tp if hasattr(vec, "tp") else vec
    return [tp] if isinstance(tp, StencilVector) else list(tp.blocks)


def to_host(vec, interior=False):
    """Data of a Stencil-/Block-/PolarVector as a list of NumPy arrays (polar coeffs first)."""
    pol = list(vec.pol) if hasattr(vec, "pol") else []
    from struphy.polar.kron_2dx1d import _interior

    return [cunumpy.to_numpy(a) for a in pol] + [
        cunumpy.to_numpy(_interior(s) if interior else s._data) for s in _stencils(vec)
    ]


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


def polar_results(mapping, backend, with_mass=True, poloidal_solver="schur", mpi_dims_mask=None):
    """Results of the polar operations on `backend`, as host arrays (dict name -> list of arrays)."""
    from struphy.polar.basic import PolarVector

    res = {}
    with cunumpy.use_backend(backend):
        domain, derham, mass = make_polar_derham(mapping, with_mass, poloidal_solver, mpi_dims_mask)

        def record(name, vec):
            assert_on_backend(vec, backend)
            res[name] = to_host(vec)

        # Host assembly accepts both backend callables and device-resident sampled weights.
        from struphy.polar.kron_2dx1d import reduced_weight_2d

        pts = [cunumpy.asarray(a).ravel() for a in derham.spline_attributes["H1"].quad_grid_pts[0]]
        mesh = cunumpy.meshgrid(*pts, indexing="ij")
        for reduction in ("midpoint", "average"):

            def weight(e1, e2, e3):
                return 1.0 + e1 + e2 + (e3 if reduction == "average" else 0.0)

            sampled = weight(*mesh)
            from_array = reduced_weight_2d(sampled, derham, reduction)
            from_function = reduced_weight_2d(weight, derham, reduction)
            np.testing.assert_allclose(from_array, from_function, rtol=1e-12, atol=1e-12)
            res["weight-" + reduction] = [from_array]
            if with_mass:
                from struphy.feec.preconditioner import PolarMassMatrixPreconditioner

                W = mass.create_weighted_mass("H1", "H1", weights=[[sampled]], assemble=True)
                pc = PolarMassMatrixPreconditioner(W, poloidal_solver=poloidal_solver, weight_reduction=reduction)
                record("array-pc-" + reduction, pc.dot(fill(W.domain.zeros(), seed=70)))

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
            res[f"toarray{form}"] = [cunumpy.to_numpy(y.toarray(allreduce=True))]

            # mass matrices (E M E^T with the polar extraction operators)
            if with_mass:
                from struphy.feec.preconditioner import PolarMassMatrixPreconditioner

                M = getattr(mass, "M" + form)
                record(f"M{form}", M.dot(y))
                pc = PolarMassMatrixPreconditioner(M, poloidal_solver=poloidal_solver)
                record(f"pcM{form}", pc.dot(M.dot(y)))
                # Factor/device caches are populated by the first application.
                if backend == "cupy" and (derham.comm.Get_size() == 1 or cunumpy.mpi.get_mpi_cuda_aware()):
                    with cunumpy.profiling.count_transfers() as transfers:
                        pc.dot(y)
                    assert transfers.total == 0, transfers.report()

        for name, op in (("grad", derham.grad), ("curl", derham.curl), ("div", derham.div)):
            x = fill(PolarVector(op.domain), seed=40)
            y = fill(PolarVector(op.codomain), seed=41)
            record(name, op.dot(x))
            record(name + "T", op.transpose().dot(y))

        # Direct Kronecker projectors, or ILU-preconditioned iterations.
        for form in ("0", "1", "2", "3"):
            P = getattr(derham, "P" + form)
            x = fill(P.I.domain.zeros(), seed=60)
            for transposed in (False, True):
                for apply_bc in (False, True):
                    target = P.boundary_op.dot(x) if apply_bc else x
                    op = (P.I0T if apply_bc else P.IT) if transposed else (P.I0 if apply_bc else P.I)
                    rhs = op.dot(target)
                    solved = P.solve(rhs, transposed=transposed, apply_bc=apply_bc)
                    solved.update_ghost_regions()
                    target.update_ghost_regions()
                    record(f"solve{form}-{transposed}-{apply_bc}", solved)
                    for actual, expected in zip(to_host(solved, interior=True), to_host(target, interior=True)):
                        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
            # Test operator and preconditioner applications without counting Krylov scalar reductions.
            P.pc.dot(x)
            if backend == "cupy" and (derham.comm.Get_size() == 1 or cunumpy.mpi.get_mpi_cuda_aware()):
                with cunumpy.profiling.count_transfers() as transfers:
                    P.I.dot(x)
                    P.pc.dot(x)
                assert transfers.total == 0, transfers.report()

        # Commuting projection of analytic functions.
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


def check_polar_backends(mapping, with_mass=True, poloidal_solver="schur", mpi_dims_mask=None):
    """The polar operations agree on the NumPy and the CuPy backend."""
    host = polar_results(mapping, "numpy", with_mass, poloidal_solver, mpi_dims_mask)
    device = polar_results(mapping, "cupy", with_mass, poloidal_solver, mpi_dims_mask)
    assert host.keys() == device.keys()
    for name in host:
        assert len(host[name]) == len(device[name]), name
        for a, b in zip(host[name], device[name]):
            np.testing.assert_allclose(b, a, rtol=1e-9, atol=1e-11, err_msg=name)


def check_polar_backends_fake_cupy(mapping, poloidal_solver="schur"):
    """Full polar parity, including mass matrices, with CPU-emulated CUDA launches."""
    from struphy.pic.tests.cuda_emulation import emulated_launches
    from struphy.polar import linear_operators

    previous = linear_operators.set_device_sparse_module(DenseDeviceSparse)
    try:
        with emulated_launches():
            check_polar_backends(mapping, poloidal_solver=poloidal_solver)
    finally:
        linear_operators.set_device_sparse_module(previous)


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
            # the polar dofs of the homotopy construction are F~^T I^{-1} (a composition without own blocks)
            dofs = derham.dofs_extraction_ops["1"]
            for row in getattr(dofs, "blocks_ten_to_ten", []) + E.blocks_ten_to_pol:
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
@pytest.mark.parametrize("poloidal_solver", ("schur", "lu", "ilu"))
def test_polar_fake_cupy(mapping, poloidal_solver):
    """Without a GPU: polar Derham on cunumpy's fake CuPy agrees with the NumPy backend.

    Runs in a subprocess because the fake CuPy must be installed before cunumpy is imported.
    """
    code = (
        "from struphy.polar.tests.test_polar_cupy import check_polar_backends_fake_cupy; "
        f"check_polar_backends_fake_cupy({mapping!r}, {poloidal_solver!r})"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=serial_child_env(CUNUMPY_FAKE_CUPY="1"), capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-4000:]


@requires_cupy
@pytest.mark.parametrize("mapping", MAPPINGS)
@pytest.mark.parametrize("poloidal_solver", ("schur", "lu", "ilu"))
def test_polar_on_cupy(mapping, poloidal_solver):
    """Polar Derham on the GPU (cupyx.scipy.sparse for the polar blocks) agrees with the NumPy backend."""
    check_polar_backends(mapping, poloidal_solver=poloidal_solver)


def check_polar_mpi(cuda_aware, mpi_dims_mask, fake=True):
    """Backend parity with both Kronecker redistributions and ranks away from the pole."""
    from contextlib import nullcontext

    from struphy.pic.tests.cuda_emulation import emulated_launches
    from struphy.polar.linear_operators import set_device_sparse_module

    previous_mpi = cunumpy.mpi.get_mpi_cuda_aware()
    previous_sparse = set_device_sparse_module(DenseDeviceSparse) if fake else None
    try:
        if cuda_aware is None:
            # feectools halo exchanges require CUDA-aware MPI. Never probe
            # an unknown MPI build with real device addresses (it may crash).
            if os.environ.get("STRUPHY_TEST_CUDA_AWARE_MPI") != "1":
                pytest.skip("full GPU MPI parity needs STRUPHY_TEST_CUDA_AWARE_MPI=1 on a CUDA-aware MPI build")
            cuda_aware = True
        cunumpy.mpi.set_mpi_cuda_aware(cuda_aware)
        with emulated_launches() if fake else nullcontext():
            check_polar_backends("IGAPolarCylinder", with_mass=False, mpi_dims_mask=mpi_dims_mask)
    finally:
        cunumpy.mpi.set_mpi_cuda_aware(previous_mpi)
        if fake:
            set_device_sparse_module(previous_sparse)


@pytest.mark.skipif(
    not getattr(sys.modules.get("cupy"), "__cunumpy_fake__", False), reason="requires CUNUMPY_FAKE_CUPY=1"
)
@pytest.mark.mpi
@pytest.mark.parametrize("cuda_aware", (True, False))
@pytest.mark.parametrize("mpi_dims_mask", ((True, False, False), (False, True, False), (False, False, True)))
def test_polar_mpi_fake_cupy(cuda_aware, mpi_dims_mask):
    check_polar_mpi(cuda_aware, mpi_dims_mask)


@requires_cupy
@pytest.mark.mpi
@pytest.mark.parametrize("mpi_dims_mask", ((True, False, False), (False, True, False), (False, False, True)))
def test_polar_mpi_on_cupy(mpi_dims_mask):
    check_polar_mpi(None, mpi_dims_mask, fake=False)


def check_polar_collectives(cuda_aware):
    """Polar collectives alone, including zero-column chunks, independent of feectools halos."""
    from maybempi import MPI

    from struphy.polar.kron_2dx1d import _BoxTranspose
    from struphy.polar.linear_operators import _allreduce_sum

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    previous = cunumpy.mpi.get_mpi_cuda_aware()
    try:
        cunumpy.mpi.set_mpi_cuda_aware(cuda_aware)
        with cunumpy.use_backend("cupy"):
            local = cunumpy.asarray(np.arange(6.0).reshape(2, 3) + 100 * rank)
            transpose = _BoxTranspose(comm, (2 * rank,), (2 * rank + 1,), (2 * size,), 3)
            with cunumpy.profiling.count_transfers() as transfers:
                full = transpose.forward(local)
                restored = transpose.backward(full)
            expected = np.concatenate([np.arange(6.0).reshape(2, 3) + 100 * r for r in range(size)])
            a, b = transpose.my_chunk
            np.testing.assert_array_equal(cunumpy.to_numpy(full), expected[:, a:b])
            np.testing.assert_array_equal(cunumpy.to_numpy(restored), cunumpy.to_numpy(local))
            if cuda_aware or size == 1:
                assert transfers.total == 0, transfers.report()
            else:
                assert transfers.to_host == transfers.to_device == 2, transfers.report()
            values = cunumpy.asarray(np.arange(5.0) + rank)
            _allreduce_sum(comm, values)
            np.testing.assert_array_equal(cunumpy.to_numpy(values), size * np.arange(5.0) + size * (size - 1) / 2)
    finally:
        cunumpy.mpi.set_mpi_cuda_aware(previous)


@pytest.mark.skipif(
    not getattr(sys.modules.get("cupy"), "__cunumpy_fake__", False), reason="requires CUNUMPY_FAKE_CUPY=1"
)
@pytest.mark.mpi
@pytest.mark.parametrize("cuda_aware", (True, False))
def test_polar_collectives_mpi_fake(cuda_aware):
    check_polar_collectives(cuda_aware)


@requires_cupy
@pytest.mark.mpi
def test_polar_collectives_mpi_on_cupy():
    """Real device buffers staged through a regular MPI build; no CUDA-aware MPI needed."""
    check_polar_collectives(False)
