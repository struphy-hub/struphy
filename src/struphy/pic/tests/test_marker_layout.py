"""The marker buffer has one column per marker and is C-ordered (struct-of-arrays), see
:mod:`struphy.pic.marker_layout`.

The CuPy cases run on a GPU, or without one on cunumpy's fake CuPy with every CUDA launch emulated on the CPU::

    mpirun -n 1 env CUNUMPY_FAKE_CUPY=1 pytest src/struphy/pic/tests/test_marker_layout.py
    mpirun -n 2 env CUNUMPY_FAKE_CUPY=1 pytest --with-mpi src/struphy/pic/tests/test_marker_layout.py
"""

import contextlib

import cunumpy as xp
import numpy as np
import pytest

from struphy.models.tests.gpu_e2e import device_context, requires_cupy_backend
from struphy.pic.marker_layout import MARKER_ALIGNMENT, marker_array, padded_n_markers

# CuPy on a GPU, or cunumpy's fake CuPy
BACKENDS = ["numpy", pytest.param("cupy", marks=requires_cupy_backend)]

N_COLS = 25
MARKER_INDICES = (3, 6, 7, 8, 14, 17, 18, 4)  # vdim, weight_idx, ..., mu_idx


def test_padded_n_rows():
    assert [padded_n_markers(n) for n in (0, 1, 31, 32, 33, 64, 65)] == [0, 32, 32, 32, 64, 64, 96]
    assert MARKER_ALIGNMENT == 32


@pytest.mark.parametrize("backend", BACKENDS)
def test_marker_array(backend):
    with xp.use_backend(backend):
        markers = marker_array(7, 64)
        assert markers.shape == (7, 64) and markers.flags.c_contiguous
        assert xp.all(markers == 0.0)
        # one quantity of all markers is contiguous
        assert markers[2].flags.c_contiguous and markers[2:5].flags.c_contiguous


def _marker_arguments(markers, valid):
    from struphy.kernel_arguments.pusher_args_cuda import CudaMarkerArguments
    from struphy.kernel_arguments.pusher_args_kernels import MarkerArguments

    args_class = CudaMarkerArguments if xp.get_backend() == "cupy" else MarkerArguments
    n = markers.shape[1]
    return args_class(markers, valid, n, *MARKER_INDICES, xp.zeros(3, dtype=np.int64), False)


def _device_context(backend):
    """Emulated CUDA launches on the fake CuPy (nothing on a GPU or on NumPy)."""
    return device_context() if backend == "cupy" else contextlib.nullcontext()


@pytest.mark.parametrize("backend", BACKENDS)
def test_kernel_accepts_marker_copies(backend):
    """A C-ordered copy of the valid markers (``xp.ascontiguousarray(markers[:, valid])``) goes to MarkerArguments and
    a kernel like the buffer, and the kernel moves the copied markers like the markers of the buffer."""
    from struphy.geometry.domains import Cuboid
    from struphy.ode.utils import ButcherTableau
    from struphy.pic.pushing.kernels.push_eta_stage import push_eta_stage

    rng = np.random.default_rng(7)
    host = rng.random((N_COLS, 64))
    host[3:6] = rng.uniform(-2, 2, (3, 64))
    host[8:14] = host[:6]
    host[18:21] = 0.0
    host[:, 0] = -1.0  # a hole
    host_valid = np.ones(64, dtype=bool)
    host_valid[0] = False

    with xp.use_backend(backend), _device_context(backend):
        markers, valid = xp.array(host), xp.asarray(host_valid)
        # selecting markers gives a Fortran-ordered copy on NumPy; the kernels take C-ordered arrays
        if backend == "numpy":
            assert markers[:, valid].flags.f_contiguous and not markers[:, valid].flags.c_contiguous
        copy = xp.ascontiguousarray(markers[:, valid])
        single = xp.ascontiguousarray(markers[:, 5:6])
        assert copy.flags.c_contiguous and single.flags.c_contiguous

        args_buffer = _marker_arguments(markers, valid)
        args_copy = _marker_arguments(copy, xp.ones(copy.shape[1], dtype=bool))
        args_single = _marker_arguments(single, xp.ones(1, dtype=bool))
        assert args_buffer.n_markers == 64 and args_copy.n_markers == 63 and args_single.n_markers == 1

        butcher = ButcherTableau("forward_euler")
        butcher_args = (xp.asarray(butcher.a_stage), xp.asarray(butcher.b), xp.asarray(butcher.c), butcher.n_stages)
        args_domain = Cuboid(r1=2.0, r2=3.0, r3=4.0).args_domain
        for args in (args_buffer, args_copy, args_single):
            push_eta_stage(0.1, 0, args, args_domain, *butcher_args, n_threads=args.n_markers)

        np.testing.assert_array_equal(xp.to_numpy(copy), xp.to_numpy(markers[:, valid]))
        np.testing.assert_array_equal(xp.to_numpy(single), xp.to_numpy(markers[:, 5:6]))
        assert not np.array_equal(xp.to_numpy(copy[:3]), host[:3, 1:])


@pytest.mark.parametrize("backend", BACKENDS)
def test_marker_buffer_layout(backend):
    """Loading, pushing, boundary conditions, sorting between ranks, evaluation at markers and binning keep the
    buffer of shape (n_cols, n_markers_max) and C-ordered, and evaluation at copies of the valid markers gives the values
    at the buffer."""
    from maybempi import MPI

    from struphy import BoundaryParameters, LoadingParameters, WeightsParameters, domains, perturbations
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.io.options import DerhamOptions
    from struphy.ode.utils import ButcherTableau
    from struphy.pic.particles import Particles6D
    from struphy.pic.pushing.kernels.push_eta_stage import push_eta_stage
    from struphy.pic.pushing.kernels.push_vxb_analytic import push_vxb_analytic
    from struphy.pic.pushing.pusher import Pusher
    from struphy.topology.grids import TensorProductGrid

    comm = MPI.COMM_WORLD
    with xp.use_backend(backend), _device_context(backend):
        domain = domains.Colella(Lx=2.0, Ly=3.0, alpha=0.1, Lz=4.0)
        derham = Derham(TensorProductGrid(num_elements=(8, 6, 4)), DerhamOptions(degree=(2, 3, 1)), comm=comm)
        particles = Particles6D(
            comm_world=comm,
            domain_decomp=(derham.domain_array, derham.domain_decomposition.nprocs),
            loading_params=LoadingParameters(Np=1000, seed=1234, spatial="uniform"),
            weights_params=WeightsParameters(),
            boundary_params=BoundaryParameters(bc=("periodic", "reflect", "periodic")),
            domain=domain,
        )
        particles.draw_markers()
        particles.mpi_sort_markers()
        particles.initialize_weights()

        buffer = particles.markers
        assert buffer.shape == (particles.n_cols, particles.n_markers_max) and buffer.flags.c_contiguous
        assert particles.n_markers_max % MARKER_ALIGNMENT == 0

        butcher = ButcherTableau("rk4")
        push_eta = Pusher(
            particles,
            push_eta_stage,
            (xp.asarray(butcher.a_stage), xp.asarray(butcher.b), xp.asarray(butcher.c), butcher.n_stages),
            domain.args_domain,
            alpha_in_kernel=1.0,
            n_stages=butcher.n_stages,
            pushes_eta=True,
        )
        _, b2 = create_equal_random_arrays(derham.V2fem, seed=3456, flattened=True)
        push_vxb = Pusher(
            particles,
            push_vxb_analytic,
            (derham.args_derham, b2[0]._data, b2[1]._data, b2[2]._data),
            domain.args_domain,
            alpha_in_kernel=1.0,
            pushes_eta=False,
        )
        for _ in range(3):
            push_eta(0.1)
            push_vxb(0.1)
            particles.mpi_sort_markers()

        # the buffer is never replaced, only written to
        assert particles.markers is buffer
        assert buffer.shape == (particles.n_cols, particles.n_markers_max) and buffer.flags.c_contiguous

        # evaluation at copies of the valid markers (Fortran-ordered on NumPy, converted by Domain and SplineFunction)
        # equals evaluation at the buffer
        valid = particles.valid_mks
        columns = particles.markers_wo_holes_and_ghost
        if backend == "numpy":
            assert columns.flags.f_contiguous and not columns.flags.c_contiguous
        n_valid = columns.shape[1]
        assert n_valid == int(xp.count_nonzero(valid))
        det_buffer = domain.jacobian_det(buffer, remove_outside=False)[valid]
        np.testing.assert_array_equal(xp.to_numpy(domain.jacobian_det(columns)), xp.to_numpy(det_buffer))
        pulled = domain.pull(lambda x, y, z: x * y + z, columns, kind="3")
        assert pulled.shape == (n_valid,)
        np.testing.assert_array_equal(
            xp.to_numpy(pulled),
            xp.to_numpy(domain.pull(lambda x, y, z: x * y + z, buffer, kind="3", remove_outside=False)[valid]),
        )

        # spline evaluation at the markers (local: every rank has its own markers)
        field = derham.create_spline_function("f", "H1")
        field.initialize_coeffs(perturbations=perturbations.ModesCos(ls=(1,), ms=(1,), ns=(1,), amps=(1.0,)))
        values = field(columns, local=True)
        assert values.shape == (n_valid,)
        np.testing.assert_array_equal(xp.to_numpy(values), xp.to_numpy(field(buffer, local=True)[valid]))

        # binning reads single rows of the valid markers
        f_slice, _ = particles.binning(
            (True, False, False, True, False, False), (xp.linspace(0.0, 1.0, 9), xp.linspace(-4.0, 4.0, 9))
        )
        assert f_slice.shape == (8, 8)
        assert particles.markers is buffer and buffer.flags.c_contiguous
