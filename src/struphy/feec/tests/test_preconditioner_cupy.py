"""Mass-matrix preconditioners on the CuPy backend (struphy-hub/struphy#689, part of #650).

The 1d mass matrices and solvers of the Kronecker preconditioners are host (NumPy) setup data on every
backend; the Kronecker stencil matrices, the diagonal scaling and the vectors they are applied to live on
the device, and feectools' KroneckerLinearSolver solves device data with dense inverses built from the host
1d solvers. These tests build the preconditioners on the CuPy backend, check that applying them makes no
host/device transfers, and compare their action with the NumPy backend:

* without a GPU, with cunumpy's fake CuPy (host memory, CuPy semantics) in a subprocess, the CUDA kernels
  launched on the way (feectools' stencil kernels, struphy's geometry kernels) run by CPU emulation;
* on a GPU (skipped otherwise), directly.
"""

import cunumpy
import numpy as np
import pytest

from struphy.geometry.tests.test_domain import _cupy_installed, run_fake_cupy_child

requires_cupy = pytest.mark.skipif(not cunumpy.cupy_available(), reason="CuPy/GPU not available")

SPACES = ("M0", "M1", "M2")
# (class name, keyword arguments)
PRECONDITIONERS = (
    ("MassMatrixPreconditioner", {"diagonal_scaling": False}),
    ("MassMatrixPreconditioner", {}),
    ("MassMatrixPreconditioner", {"weight_reduction": "average"}),
    ("MassMatrixPreconditioner", {"dim_reduce": None}),
)
BCS = {
    "periodic": (None, None, None),
    "clamped": (("dirichlet", "dirichlet"), ("free", "free"), ("free", "dirichlet")),
}


def _build(backend, bcs):
    """Derham (6 x 5 x 4 elements, degrees (2, 3, 1)) and mass operators on a Colella mapping, on `backend`."""
    from maybempi import MPI

    from struphy import domains
    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.psydac_derham import Derham
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    with cunumpy.use_backend(backend):
        options = DerhamOptions(degree=(2, 3, 1), bcs=bcs)
        derham = Derham(TensorProductGrid(num_elements=(6, 5, 4)), options, comm=MPI.COMM_WORLD)
        return WeightedMassOperators(derham, domains.Colella())


def _blocks(v):
    """The stencil vectors of a stencil or block vector."""
    return list(v.blocks) if hasattr(v, "blocks") else [v]


def _random_vector(space, rng):
    v = space.zeros()
    for block in _blocks(v):
        block._data[...] = cunumpy.asarray(rng.random(block._data.shape))
    v.update_ghost_regions()
    return v


def _to_device(v_host, space):
    """A vector of `space` (active backend) with the data of the host vector `v_host`."""
    v = space.zeros()
    for block, block_host in zip(_blocks(v), _blocks(v_host)):
        block._data[...] = cunumpy.asarray(block_host._data)
    v.update_ghost_regions()
    return v


def _to_host(v):
    return np.concatenate([cunumpy.to_numpy(block._data).ravel() for block in _blocks(v)])


def check_preconditioners_on_cupy(bcs_name):
    """Every preconditioner of :data:`PRECONDITIONERS` for M0, M1 and M2 gives the NumPy result on the CuPy backend.

    Applying the preconditioners (``dot``, also in place) makes no host/device transfers.
    """
    from struphy.feec import preconditioner

    bcs = BCS[bcs_name]
    host, device = _build("numpy", bcs), _build("cupy", bcs)
    rng = np.random.default_rng(1)

    for space in SPACES:
        with cunumpy.use_backend("numpy"):
            M_host = getattr(host, space)
            v_host = _random_vector(M_host.domain, rng)
        with cunumpy.use_backend("cupy"):
            M_device = getattr(device, space)
            v_device = _to_device(v_host, M_device.domain)

        for name, kwargs in PRECONDITIONERS:
            cls = getattr(preconditioner, name)
            case = (bcs_name, space, name, kwargs)
            with cunumpy.use_backend("numpy"):
                expected = _to_host(cls(M_host, **kwargs).dot(v_host))
            with cunumpy.use_backend("cupy"):
                pc = cls(M_device, **kwargs)
                out_inplace = pc.codomain.zeros()
                with cunumpy.profiling.assert_no_transfers():
                    out = pc.dot(v_device)
                    # in-place application
                    pc.dot(v_device, out=out_inplace)
                for block in _blocks(out):
                    assert cunumpy.is_gpu(block._data), case
            assert np.linalg.norm(expected) > 0, case
            np.testing.assert_allclose(_to_host(out), expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max())
            np.testing.assert_allclose(_to_host(out_inplace), expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max())


def check_preconditioners_on_fake_cupy(bcs_name):
    """:func:`check_preconditioners_on_cupy` with the CUDA kernel launches emulated on the CPU."""
    from cunumpy.kernel_testing import emulated_launches, fake_cupy_active

    from struphy.pic.tests.cuda_emulation import EMULATION_OPTIONS

    assert fake_cupy_active()
    with emulated_launches(options=EMULATION_OPTIONS):
        check_preconditioners_on_cupy(bcs_name)


@pytest.mark.skipif(_cupy_installed(), reason="the fake CuPy cannot replace an installed CuPy")
@pytest.mark.parametrize("bcs_name", list(BCS))
def test_mass_preconditioners_fake_cupy(bcs_name):
    """Without a GPU: the preconditioners build and apply on cunumpy's fake CuPy, which rejects host/device mixing.

    Runs in a serial subprocess (on MPI rank 0 only) because the fake CuPy must be installed before cunumpy is imported.
    """
    run_fake_cupy_child(
        "from struphy.feec.tests.test_preconditioner_cupy import check_preconditioners_on_fake_cupy; "
        f"check_preconditioners_on_fake_cupy({bcs_name!r})"
    )


@requires_cupy
@pytest.mark.parametrize("bcs_name", list(BCS))
def test_mass_preconditioners_on_cupy(bcs_name):
    """On a GPU, or in this process on the fake CuPy (``CUNUMPY_FAKE_CUPY=1``, launches emulated on the CPU): the
    preconditioners give the NumPy result on the CuPy backend."""
    from cunumpy.kernel_testing import fake_cupy_active

    if fake_cupy_active():
        check_preconditioners_on_fake_cupy(bcs_name)
    else:
        check_preconditioners_on_cupy(bcs_name)
