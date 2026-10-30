"""The matrix path of :class:`~struphy.pic.accumulation.particles_to_grid.Accumulator` on the CuPy backend.

``linear_vlasov_ampere`` accumulates the symmetric V1 -> V1 block matrix and the V1 vector of ``EfieldWeightsCoupling``;
the matrix data (``StencilMatrix._data``) must be device arrays that are zeroed, passed to the CUDA kernel and turned
into the Schur operator without host copies. Checked against the NumPy backend on a GPU and, without one, on
cunumpy's fake CuPy with CPU-emulated kernel launches (in a subprocess, see :func:`check_accumulator_matrix_fake_cupy`).
"""

from dataclasses import dataclass
from typing import Any

import cunumpy as xp
import numpy as np
import pytest
from cunumpy.kernel_testing import emulation_compiler, requires_cupy

from struphy.geometry.tests.test_domain import _cupy_installed, run_fake_cupy_child

ALPHA, KAPPA, VTH, DT = 1.3, 0.7, 0.9, 0.05
N_SCHUR_ITERATIONS = 3


@dataclass
class MarkerOwner:
    """What the accumulators read from ``Particles``: the marker arguments, no clones.

    The same markers on both backends (``Particles`` draws them with the backend's random generator).
    """

    args_markers: Any
    clone_config: Any = None


def accumulate_linear_vlasov_ampere(guard=None):
    """Accumulate ``linear_vlasov_ampere`` like ``EfieldWeightsCoupling`` on the active backend, then solve with it.

    Colella mapping (non-diagonal Jacobian), clamped splines in eta1 and periodic ones in eta2, eta3, so the three
    V1 components have different sizes. `guard` (a context manager factory, e.g. one that fails on host transfers)
    encloses the accumulation, the scaled operator and its application, as in a time step of the propagator; the
    Schur solve is not guarded (see below).

    Returns a dict of host arrays: the nine kernel data arrays (``mat11`` ... ``vec3``) after the accumulation (ghost
    regions exchanged), the assembled block matrix as dense blocks, the accumulated vector after the boundary
    operators, ``BC.dot(x)`` with the operator scaled as in ``EfieldWeightsCoupling`` and the Schur solution after
    ``N_SCHUR_ITERATIONS`` iterations of the conjugate gradient method.
    """
    import contextlib

    from feectools.linalg.stencil import StencilMatrix

    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.psydac_derham import Derham
    from struphy.geometry.domains import Colella
    from struphy.io.options import DerhamOptions
    from struphy.linear_algebra.schur_solver import SchurSolver
    from struphy.linear_algebra.solver import SolverParameters
    from struphy.pic.accumulation.kernels.linear_vlasov_ampere import linear_vlasov_ampere
    from struphy.pic.accumulation.particles_to_grid import Accumulator
    from struphy.pic.tests.kernel_test_args import N_MARKERS, marker_arguments
    from struphy.topology.grids import TensorProductGrid

    guard = guard or contextlib.nullcontext
    derham = Derham(
        TensorProductGrid(num_elements=(6, 5, 4)),
        DerhamOptions(degree=(2, 3, 1), bcs=(("free", "free"), None, None)),
        comm=None,
    )
    domain = Colella(Lx=2.0, Ly=3.0, alpha=0.05, Lz=4.0)
    mass_ops = WeightedMassOperators(derham, domain)
    args_markers, _ = marker_arguments((0, 0, 0))
    accum = Accumulator(
        MarkerOwner(args_markers),
        "Hcurl",
        linear_vlasov_ampere,
        mass_ops,
        domain.args_domain,
        add_vector=True,
        symmetry="symm",
    )
    matrix = accum.operators[0].matrix

    # the kernel writes into the data of the operator's blocks (and vector), not into copies
    blocks = [matrix[a, b] for a in range(3) for b in range(a, 3)]
    assert all(isinstance(block, StencilMatrix) for block in blocks)
    assert all(data is block._data for data, block in zip(accum._args_data[:6], blocks))
    assert all(data is vec._data for data, vec in zip(accum._args_data[6:], accum._vectors[0].blocks))
    assert all(xp.is_gpu(data) == (xp.get_backend() == "cupy") for data in accum._args_data)

    rng = np.random.default_rng(16)
    f0_values = xp.asarray(rng.random(N_MARKERS))
    for data in accum._args_data:
        data[...] = 1.0  # left over from a previous step: the accumulator zeroes it

    x = accum._vectors[0].space.zeros()
    for block in x.blocks:
        block._data[...] = xp.asarray(rng.normal(size=block._data.shape))
    x.update_ghost_regions()

    schur_solver = SchurSolver(
        mass_ops.M1,
        ALPHA**2 * KAPPA**2 * accum.operators[0] / (4 * VTH**2),
        "pcg",
        # a few iterations (each one launches nine stencil dot kernels, which the emulation compiles one by one)
        solver_params=SolverParameters(tol=1e-30, maxiter=N_SCHUR_ITERATIONS),
    )

    with guard():
        accum(f0_values)
        # as in EfieldWeightsCoupling.__call__
        schur_solver.BC = accum.operators[0]
        schur_solver.BC *= (-1) * ALPHA**2 * KAPPA**2 / (4 * VTH**2)
        bc_x = schur_solver.BC.dot(x)
        byn = accum.vectors[0].copy()
        byn *= ALPHA**2 * KAPPA / 2.0
    # not guarded: feectools' StencilVector.inner returns its result on the host (xp.to_numpy), so every inner
    # product of the conjugate gradient method copies one scalar from the device
    solution, info = schur_solver(xn=x, Byn=byn, dt=DT)

    out = {name: xp.to_numpy(data) for name, data in zip(MATRIX_DATA_NAMES, accum._args_data)}
    for a in range(3):
        for b in range(3):
            out[f"block{a}{b}"] = xp.to_numpy(matrix[a, b].toarray())
    out["vector"] = xp.to_numpy(accum.vectors[0].toarray())
    out["bc_x"] = xp.to_numpy(bc_x.toarray())
    out["solution"] = xp.to_numpy(solution.toarray())
    out["niter"] = np.array(info["niter"])
    return out


MATRIX_DATA_NAMES = ("mat11", "mat12", "mat13", "mat22", "mat23", "mat33", "vec1", "vec2", "vec3")


def compare_with_numpy(result):
    """Compare a CuPy-backend `result` of :func:`accumulate_linear_vlasov_ampere` with the NumPy backend's."""
    with xp.use_backend("numpy"):
        expected = accumulate_linear_vlasov_ampere()
    assert expected["niter"] == result["niter"] == N_SCHUR_ITERATIONS
    assert np.any(expected["mat12"] != 0.0) and np.any(expected["vec3"] != 0.0)
    # the transposed blocks of the symmetric matrix are filled from the accumulated ones
    np.testing.assert_allclose(expected["block10"], expected["block01"].T, rtol=1e-14, atol=1e-14)
    for name, value in expected.items():
        scale = max(np.max(np.abs(value)), 1.0)
        # atomics add in another order than the serial loop; the solve repeats it
        np.testing.assert_allclose(result[name], value, rtol=1e-10, atol=1e-12 * scale, err_msg=name)


@requires_cupy
def test_accumulator_matrix_on_cupy():
    """On a GPU: the accumulated matrix and vector, the Schur operator and solve agree with NumPy; the accumulation and
    the operator run without host/device transfers."""
    from cunumpy.profiling import assert_no_transfers

    with xp.use_backend("cupy"):
        accumulate_linear_vlasov_ampere()  # warm up (NVRTC compilation)
        result = accumulate_linear_vlasov_ampere(guard=assert_no_transfers)
    compare_with_numpy(result)


def fake_cupy_transfer_guard():
    """On cunumpy's fake CuPy: fail on every explicit host copy of a device array (``.get()``, ``cupy.asnumpy``).

    The fake CuPy already rejects implicit conversions (``numpy.asarray`` of a device array, mixing host and device
    arrays); this closes the explicit ones, which are legal but are transfers inside a time step.
    """
    import contextlib

    import cupy

    @contextlib.contextmanager
    def guard():
        def fail(*args, **kwargs):
            raise AssertionError("host copy of a device array during the accumulation")

        saved = cupy.ndarray.get, cupy.asnumpy
        cupy.ndarray.get, cupy.asnumpy = fail, fail
        try:
            yield
        finally:
            cupy.ndarray.get, cupy.asnumpy = saved

    return guard


def check_accumulator_matrix_fake_cupy():
    """Run in a child process with ``CUNUMPY_FAKE_CUPY=1``: the CuPy backend with emulated kernel launches."""
    from struphy.pic.tests.cuda_emulation import emulated_launches

    with emulated_launches(), xp.use_backend("cupy"):
        result = accumulate_linear_vlasov_ampere(guard=fake_cupy_transfer_guard())
    compare_with_numpy(result)


@pytest.mark.skipif(_cupy_installed(), reason="the fake CuPy cannot replace an installed CuPy")
@pytest.mark.skipif(emulation_compiler() is None, reason="no C++ compiler")
def test_accumulator_matrix_fake_cupy():
    """Without a GPU: the matrix path of the Accumulator on the CuPy backend (fake CuPy, emulated launches).

    Device data are zeroed and passed to the CUDA kernel, the ghost regions, the transposed blocks, the scaled
    operator and the Schur solve run on device arrays, and no array is copied to the host from the accumulation to
    the operator's application. Runs in a
    subprocess because the fake CuPy must be installed before cunumpy is imported.
    """
    code = "from struphy.pic.tests.test_accum_matrix_cupy import check_accumulator_matrix_fake_cupy as c; c()"
    run_fake_cupy_child(code)
