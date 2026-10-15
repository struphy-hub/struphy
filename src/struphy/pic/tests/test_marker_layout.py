"""The marker buffer is column-major (struct-of-arrays), see :mod:`struphy.pic.marker_layout`.

The CuPy cases run on a GPU, or without one on cunumpy's fake CuPy with every CUDA launch emulated on the CPU::

    mpirun -n 1 env CUNUMPY_FAKE_CUPY=1 pytest src/struphy/pic/tests/test_marker_layout.py
    mpirun -n 2 env CUNUMPY_FAKE_CUPY=1 pytest --with-mpi src/struphy/pic/tests/test_marker_layout.py
"""

import contextlib

import cunumpy as xp
import numpy as np
import pytest

from struphy.models.tests.gpu_e2e import device_context, requires_cupy_backend
from struphy.pic.marker_layout import (
    ROW_ALIGNMENT,
    as_marker_array,
    check_marker_layout,
    has_marker_layout,
    marker_array,
    padded_n_rows,
)

# CuPy on a GPU, or cunumpy's fake CuPy
BACKENDS = ["numpy", pytest.param("cupy", marks=requires_cupy_backend)]


def test_padded_n_rows():
    assert [padded_n_rows(n) for n in (0, 1, 31, 32, 33, 64, 65)] == [0, 32, 32, 32, 64, 64, 96]
    assert ROW_ALIGNMENT == 32


@pytest.mark.parametrize("backend", BACKENDS)
def test_layout_helpers(backend):
    with xp.use_backend(backend):
        markers = marker_array(64, 7)
        assert markers.shape == (64, 7) and has_marker_layout(markers)
        check_marker_layout(markers)
        # no copy for an array in the marker layout, nor for a block of its columns
        assert as_marker_array(markers) is markers
        assert has_marker_layout(markers[:, 2:5])

        # row gathers and .copy() are C-ordered: converted (with a copy) before they go to a kernel
        rows = markers[xp.arange(0, 64, 2)]
        assert not has_marker_layout(rows)
        with pytest.raises(TypeError, match="Fortran-ordered"):
            check_marker_layout(rows)
        converted = as_marker_array(rows)
        assert has_marker_layout(converted)
        np.testing.assert_array_equal(xp.to_numpy(converted), xp.to_numpy(rows))

        # a single row in C order is F-contiguous for NumPy, but its strides decrease, which pyccel rejects
        row = xp.asarray([[0.2, 0.4, 0.6]])
        assert row.flags.f_contiguous and not has_marker_layout(row)
        assert has_marker_layout(as_marker_array(row))


def test_pyccel_rejects_row_major_markers():
    """pyccel checks the memory order of the marker arrays at every call, so a missed conversion fails loudly."""
    from struphy.kernel_arguments.pusher_args_kernels import MarkerArguments

    args = (np.ones(4, dtype=bool), 4, 3, 6, 7, 8, 14, 17, 18, 4, np.zeros(3, dtype=np.int64), False)
    MarkerArguments(np.zeros((4, 25), order="F"), *args)
    with pytest.raises(TypeError, match="ordering"):
        MarkerArguments(np.zeros((4, 25)), *args)

    # a single row: as_marker_array gives it the strides pyccel accepts
    one = (np.ones(1, dtype=bool), 1, *args[2:])
    with pytest.raises(TypeError, match="ordering"):
        MarkerArguments(np.zeros((1, 25)), *one)
    MarkerArguments(as_marker_array(np.zeros((1, 25))), *one)


def _device_context(backend):
    """Emulated CUDA launches on the fake CuPy (nothing on a GPU or on NumPy)."""
    return device_context() if backend == "cupy" else contextlib.nullcontext()


@pytest.mark.parametrize("backend", BACKENDS)
def test_markers_stay_column_major(backend):
    """Loading, pushing, boundary conditions, sorting between ranks, evaluation at markers and binning keep the
    buffer column-major, and evaluation at row-major copies of the markers gives the same values."""
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
        check_marker_layout(buffer)
        assert buffer.shape[0] % ROW_ALIGNMENT == 0

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
        check_marker_layout(particles.markers)

        # evaluation at row-major copies of the markers equals evaluation at the buffer (converted at the entry)
        rows = particles.markers_wo_holes_and_ghost
        assert rows.flags.c_contiguous
        n_valid = rows.shape[0]
        det_buffer = domain.jacobian_det(particles.markers, remove_outside=False)[particles.valid_mks]
        np.testing.assert_array_equal(xp.to_numpy(domain.jacobian_det(rows)), xp.to_numpy(det_buffer))
        np.testing.assert_array_equal(
            xp.to_numpy(domain.pull(lambda x, y, z: x * y + z, rows, kind="3")),
            xp.to_numpy(domain.pull(lambda x, y, z: x * y + z, as_marker_array(rows), kind="3")),
        )

        # spline evaluation at row-major markers (local: every rank has its own markers)
        field = derham.create_spline_function("f", "H1")
        field.initialize_coeffs(perturbations=perturbations.ModesCos(ls=(1,), ms=(1,), ns=(1,), amps=(1.0,)))
        values = field(rows, local=True)
        assert values.shape == (n_valid,)
        np.testing.assert_array_equal(xp.to_numpy(values), xp.to_numpy(field(as_marker_array(rows), local=True)))

        # binning reads single columns of the valid markers
        f_slice, _ = particles.binning(
            (True, False, False, True, False, False), (xp.linspace(0.0, 1.0, 9), xp.linspace(-4.0, 4.0, 9))
        )
        assert f_slice.shape == (8, 8)
        check_marker_layout(particles.markers)


def _run_with_removal(backend, track_rows: bool):
    """Particles with removing boundaries in eta1, pushed and sorted between ranks; the valid markers by ID at the
    end, and the bound n_rows_used checked after every step. Without `track_rows`, every row counts as used."""
    from unittest import mock

    from maybempi import MPI

    from struphy import BoundaryParameters, LoadingParameters, WeightsParameters, domains
    from struphy.feec.psydac_derham import Derham
    from struphy.io.options import DerhamOptions
    from struphy.ode.utils import ButcherTableau
    from struphy.pic.base import Particles
    from struphy.pic.particles import Particles6D
    from struphy.pic.pushing.kernels.push_eta_stage import push_eta_stage
    from struphy.pic.pushing.pusher import Pusher
    from struphy.topology.grids import TensorProductGrid

    def all_rows(self, n_emptied=0, n_filled=0):
        self._set_n_rows_used(self.n_rows)

    def check_bound(particles):
        n = particles.n_rows_used
        assert particles.args_markers.n_markers == n
        assert bool(xp.all(particles.markers[n:, 0] == -1.0)), "a row after n_rows_used is not a hole"
        assert bool(xp.all(particles.holes[n:]))

    patches = (
        contextlib.nullcontext()
        if track_rows
        else mock.patch.multiple(Particles, _track_rows=all_rows, count_rows_used=lambda self: all_rows(self))
    )
    comm = MPI.COMM_WORLD
    with xp.use_backend(backend), _device_context(backend), patches:
        domain = domains.Cuboid(r1=2.0, r2=3.0, r3=4.0)
        derham = Derham(TensorProductGrid(num_elements=(8, 4, 2)), DerhamOptions(degree=(2, 2, 1)), comm=comm)
        particles = Particles6D(
            comm_world=comm,
            domain_decomp=(derham.domain_array, derham.domain_decomposition.nprocs),
            loading_params=LoadingParameters(Np=2000, seed=4321, spatial="uniform"),
            weights_params=WeightsParameters(),
            boundary_params=BoundaryParameters(bc=("remove", "periodic", "periodic")),
            domain=domain,
            bufsize=0.5,
        )
        particles.draw_markers()
        particles.mpi_sort_markers()
        particles.initialize_weights()
        if track_rows:
            check_bound(particles)
            assert particles.n_rows_used < particles.n_rows

        butcher = ButcherTableau("forward_euler")
        push_eta = Pusher(
            particles,
            push_eta_stage,
            (xp.asarray(butcher.a_stage), xp.asarray(butcher.b), xp.asarray(butcher.c), butcher.n_stages),
            domain.args_domain,
            alpha_in_kernel=1.0,
            n_stages=butcher.n_stages,
            pushes_eta=True,
        )
        for _ in range(4):
            push_eta(0.3)
            particles.mpi_sort_markers()
            if track_rows:
                check_bound(particles)

        assert particles.n_lost_markers > 0, "the test should remove markers"
        valid = xp.to_numpy(particles.markers_wo_holes_and_ghost)
        return valid[np.argsort(valid[:, -1])], particles.n_lost_markers


@pytest.mark.parametrize("backend", BACKENDS)
def test_rows_used(backend):
    """Kernels and column operations over the rows that can hold markers (n_rows_used) give the same markers as
    over all rows, and every row after n_rows_used stays a hole through loading, pushes that remove markers at the
    boundary and the exchange between ranks."""
    tracked, lost_tracked = _run_with_removal(backend, track_rows=True)
    full, lost_full = _run_with_removal(backend, track_rows=False)
    assert lost_tracked == lost_full
    np.testing.assert_array_equal(tracked, full)
