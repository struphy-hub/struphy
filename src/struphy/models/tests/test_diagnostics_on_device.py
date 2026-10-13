"""Diagnostics and output on the device (struphy-hub/struphy#697).

During a CuPy run the per-step diagnostics (scalar quantities, binned distribution functions, saved markers) are
computed on the device, and only the reduced results reach the host: the scalars in one copy per update
(``Scalars.to_host``), every other saved dataset in one copy per output step (``DataContainer.save_data``).

Each ``check_*`` function runs on the active backend and compares with the NumPy backend. Without a GPU they run in a
subprocess on cunumpy's fake CuPy (``CUNUMPY_FAKE_CUPY=1``), which rejects host/device mixing; there, implicit
device-to-host conversions (``float()``, ``int()``, ``bool()`` and ``__index__`` of a device array, each of which
waits for the device) raise as well, and the copies made through cunumpy are counted with
``cunumpy.profiling.count_transfers``. With a GPU the same checks run on the real CuPy (without the implicit-sync
guard, which needs the fake). The checks use mock particles and need no CUDA kernel.
"""

import contextlib
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import cunumpy as xp
import numpy as np
import pytest
from cunumpy.kernel_testing import requires_cupy

from struphy.geometry.tests.test_domain import _cupy_installed, serial_child_env

N_ROWS = 12  # marker rows: 8 valid markers, 2 ghosts, 2 holes
N_COLS = 13  # Particles6D columns: eta (3), v (3), w, s0, w0, 2 auxiliary, box, ID


def _markers():
    """Particles6D-like markers on the host: 8 valid markers, 2 ghosts (ID -2) and 2 holes (all -1)."""
    rng = np.random.default_rng(697)
    markers = rng.random((N_ROWS, N_COLS))
    markers[:, 3:6] = rng.normal(size=(N_ROWS, 3))
    markers[:, -1] = np.arange(N_ROWS)
    markers[8:10, -1] = -2.0  # ghosts
    markers[10:] = -1.0  # holes
    return markers


class _MockParticles:
    """The parts of :class:`~struphy.pic.base.Particles` used by the diagnostics, on the active backend."""

    vdim = 3
    Np = 8
    n_lost_markers = 3
    clone_config = None
    index = {"pos": slice(0, 3), "vel": slice(3, 6), "coords": slice(0, 6), "weights": 6, "s0": 7, "w0": 8, "ids": -1}

    def __init__(self):
        self._markers = xp.asarray(_markers())
        self._valid_mks = ~xp.logical_or(self._markers[:, 0] == -1.0, self._markers[:, -1] == -2.0)

    @property
    def markers(self):
        return self._markers

    @property
    def valid_mks(self):
        return self._valid_mks

    @property
    def markers_wo_holes_and_ghost(self):
        return self.markers[self.valid_mks]

    @property
    def velocities(self):
        return self.markers[self.valid_mks, self.index["vel"]]

    @property
    def weights(self):
        return self.markers[self.valid_mks, self.index["weights"]]


def _fake_cupy_active():
    return bool(getattr(sys.modules.get("cupy"), "__cunumpy_fake__", False))


@contextlib.contextmanager
def no_implicit_syncs():
    """On the fake CuPy, raise if a device array is converted to a host scalar implicitly (a device sync)."""
    if not _fake_cupy_active():
        yield
        return
    import cupy

    def forbidden(name):
        def raise_(self, *args, **kwargs):
            raise AssertionError(f"implicit device-to-host conversion: ndarray.{name}")

        return raise_

    names = ("__float__", "__int__", "__bool__", "__index__")
    saved = {name: getattr(cupy.ndarray, name) for name in names}
    for name in names:
        setattr(cupy.ndarray, name, forbidden(name))
    try:
        yield
    finally:
        for name, method in saved.items():
            setattr(cupy.ndarray, name, method)


@contextlib.contextmanager
def _serial_propagator_base():
    """``Propagator.derham`` without a communicator, as the PIC scalars need it for their MPI sum."""
    from struphy.propagators.base import Propagator

    saved = Propagator.__dict__["derham"]
    Propagator.derham = SimpleNamespace(comm=None)
    try:
        yield
    finally:
        Propagator.derham = saved


def _scalars(particles):
    from struphy.models.scalars import FunctionScalarPIC, KineticEnergyPIC, LostMarkersPIC, Scalars
    from struphy.models.variables import PICVariable

    class _Variable(PICVariable):
        def __init__(self):
            self._particles = particles

    var = _Variable()
    kinetic = KineticEnergyPIC(var, normalization=2.0)
    # a device reduction, as LinearVlasovAmpereOneSpecies._compute_en_w returns
    weights = FunctionScalarPIC(lambda: xp.sum(particles.weights**2), var)
    lost = LostMarkersPIC(var)
    return Scalars(en_kin=kinetic, en_w=weights, lost=lost, en_tot=kinetic + weights)


def _scalar_values(backend):
    with xp.use_backend(backend), _serial_propagator_base():
        particles = _MockParticles()
        scalars = _scalars(particles)
        with xp.profiling.count_transfers() as counter, no_implicit_syncs():
            scalars.update()
        values = {key: scalars.host_value(key).copy() for key in scalars.dct}

        # the host values follow the next update, and are views of one host buffer
        particles.markers[:, 6] *= 2.0
        scalars.update()
        doubled = scalars.host_value("en_kin")[0]
    return values, counter, doubled


def check_scalars():
    """The scalars stay on the device; one copy of all values per update, no implicit syncs."""
    values, counter, doubled = _scalar_values(xp.get_backend())
    reference, reference_counter, reference_doubled = _scalar_values("numpy")

    assert reference_counter.events == []
    assert values.keys() == reference.keys()
    for key in values:
        np.testing.assert_allclose(values[key], reference[key], rtol=1e-14, err_msg=key)
    np.testing.assert_allclose(doubled, reference_doubled, rtol=1e-14)
    np.testing.assert_allclose(doubled, 2.0 * values["en_kin"][0], rtol=1e-14)
    np.testing.assert_allclose(values["en_tot"], values["en_kin"] + values["en_w"], rtol=1e-14)
    assert values["lost"][0] == 3.0

    if xp.get_backend() == "cupy":
        kinds = [event.kind for event in counter.events]
        assert kinds == ["to_host"], counter.report()
        assert counter.events[0].nbytes == 8 * len(values), counter.report()


def _binned(backend, output_quantity):
    from struphy.particles.parameters import BinningPlot
    from struphy.pic.base import Particles

    with xp.use_backend(backend):
        particles = _MockParticles()
        bin_plot = BinningPlot(slice="e1_v1", n_bins=(4, 3), ranges=((0.0, 1.0), (-2.0, 2.0)))
        components = [True, False, False, True, False, False]
        with xp.profiling.count_transfers() as counter, no_implicit_syncs():
            f_slice, df_slice = Particles.binning(
                particles, components, bin_plot.bin_edges, output_quantity=output_quantity, divide_by_jac=False
            )
            bin_plot.f[:] = f_slice
            bin_plot.df[:] = df_slice
        assert type(bin_plot.f) is type(particles.markers)
        return xp.to_numpy(bin_plot.f), xp.to_numpy(bin_plot.df), counter


def check_binning():
    """Marker binning (``xp.histogramdd``) runs on the device of the markers, without transfers."""
    for output_quantity in ("density", "current_1", "energy_tensor_12", "heat_flux_2"):
        f_slice, df_slice, counter = _binned(xp.get_backend(), output_quantity)
        f_reference, df_reference, _ = _binned("numpy", output_quantity)
        assert counter.events == [], counter.report()
        assert np.any(f_reference != 0.0) and np.any(df_reference != 0.0)
        np.testing.assert_allclose(f_slice, f_reference, rtol=1e-13, atol=1e-15, err_msg=output_quantity)
        np.testing.assert_allclose(df_slice, df_reference, rtol=1e-13, atol=1e-15, err_msg=output_quantity)


def _saved_markers(backend):
    from struphy.models.base import StruphyModel
    from struphy.models.species import ParticleSpecies
    from struphy.models.variables import PICVariable
    from struphy.pic.base import Particles

    # update_markers_to_be_saved checks for a Particles instance
    mock_class = type("_MockParticlesInstance", (_MockParticles, Particles), {})
    mock_class.__abstractmethods__ = frozenset()

    with xp.use_backend(backend):
        particles = mock_class()

        class _Variable(PICVariable):
            def __init__(self):
                self._particles = particles
                self._n_to_save = 5
                self._saved_markers = xp.zeros((5, N_COLS))

        class _Species(ParticleSpecies):
            def __init__(self):
                self._variables = {"var": _Variable()}

        species = _Species()
        model = SimpleNamespace(particle_species={"ions": species})
        with xp.profiling.count_transfers() as counter, no_implicit_syncs():
            StruphyModel.update_markers_to_be_saved(model)
        return xp.to_numpy(species.variables["var"].saved_markers), counter


def check_saved_markers():
    """The markers to be saved are gathered on the device, without transfers."""
    saved, counter = _saved_markers(xp.get_backend())
    reference, _ = _saved_markers("numpy")
    assert counter.events == [], counter.report()
    # IDs 0-4 are valid markers here; the ghosts and holes have no ID below n_to_save
    np.testing.assert_array_equal(reference[:, -1], [0.0, 1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(saved, reference)


def check_output():
    """``DataContainer.save_data`` copies each device dataset once, through cunumpy, and host datasets not at all."""
    import h5py

    from struphy.io.output_handling import DataContainer

    with tempfile.TemporaryDirectory() as path:
        os.makedirs(os.path.join(path, "data"))
        data = DataContainer(path)
        host = {"time/value": np.full(1, 0.5), "scalar/en": np.full(1, 2.0)}
        device = {"feec/e": xp.arange(24.0).reshape(2, 3, 4), "kinetic/f": xp.ones((4, 3))}
        data.add_data(host | device)
        with xp.profiling.count_transfers() as counter, no_implicit_syncs():
            data.save_data()
        with h5py.File(data.file_path) as file:
            np.testing.assert_array_equal(file["feec/e"][-1], np.arange(24.0).reshape(2, 3, 4))
            np.testing.assert_array_equal(file["scalar/en"][:], [2.0, 2.0])

    if xp.get_backend() == "cupy":
        assert [event.kind for event in counter.events] == ["to_host"] * len(device), counter.report()
        assert sum(event.nbytes for event in counter.events) == sum(a.nbytes for a in device.values())
    else:
        assert counter.events == []


def _flagged(backend):
    from struphy.feec.psydac_derham import SplineFunction

    # two processes split along eta1 at 0.5; this is rank 0
    domain_array = np.array([[0.0, 0.5, 4, 0.0, 1.0, 6, 0.0, 1.0, 4], [0.5, 1.0, 4, 0.0, 1.0, 6, 0.0, 1.0, 4]])
    with xp.use_backend(backend):
        spline = SplineFunction.__new__(SplineFunction)
        spline._derham = SimpleNamespace(domain_array=xp.asarray(domain_array), comm=None)
        E1, E2, E3 = (xp.asarray(a) for a in np.meshgrid(*[np.linspace(0.0, 1.0, 5)] * 3, indexing="ij"))
        markers = xp.asarray(np.random.default_rng(1).random((10, 3)))
        spline._flag_pts_not_on_proc(E1, E2, E3)  # first call: caches the host bounds
        with xp.profiling.count_transfers() as counter, no_implicit_syncs():
            spline._flag_pts_not_on_proc(E1, E2, E3)
            spline._flag_pts_not_on_proc(markers)
        return [xp.to_numpy(a) for a in (E1, E2, E3, markers)], counter


def check_flag_points():
    """SplineFunction flags the points of other processes on the device; the bounds are cached on the host once."""
    flagged, counter = _flagged(xp.get_backend())
    reference, _ = _flagged("numpy")
    assert counter.events == [], counter.report()
    assert np.any(reference[0] == -1.0) and np.any(reference[3] == -1.0)
    for a, b in zip(flagged, reference):
        np.testing.assert_array_equal(a, b)


CHECKS = ("check_scalars", "check_binning", "check_saved_markers", "check_output", "check_flag_points")


@pytest.mark.mpi_skip
@pytest.mark.parametrize("check", CHECKS)
def test_on_numpy(check):
    """The checks on the NumPy backend: no transfers, the same results as before."""
    with xp.use_backend("numpy"):
        globals()[check]()


@pytest.mark.mpi_skip
@pytest.mark.skipif(_cupy_installed(), reason="the fake CuPy cannot replace an installed CuPy")
@pytest.mark.parametrize("check", CHECKS)
def test_on_fake_cupy(check):
    """Without a GPU: the checks on cunumpy's fake CuPy, in a subprocess (the fake must be installed first)."""
    code = f"import cunumpy as xp; xp.set_backend('cupy'); from {__name__} import {check}; {check}()"
    result = subprocess.run(
        [sys.executable, "-c", code], env=serial_child_env(CUNUMPY_FAKE_CUPY="1"), capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-4000:]


@pytest.mark.mpi_skip
@requires_cupy
@pytest.mark.parametrize("check", CHECKS)
def test_on_cupy(check):
    """With a GPU: the same checks on the real CuPy."""
    with xp.use_backend("cupy"):
        globals()[check]()


@pytest.mark.mpi_skip
@requires_cupy
def test_eval_tp_fixed_loc_on_cupy():
    """``SplineFunction.eval_tp_fixed_loc`` (the CUDA ``eval_spline_mpi_tensor_product_fixed``) agrees with NumPy."""
    from struphy.feec.tests.test_derham_gpu import make_derham

    pts = [np.linspace(0.05, 0.95, n) for n in (5, 4, 3)]
    results = {}
    for backend in ("numpy", "cupy"):
        with xp.use_backend(backend):
            derham = make_derham()
            spans, bns, bds = derham.prepare_eval_tp_fixed([xp.asarray(p) for p in pts])
            for space, bases in (("H1", bns), ("L2", bds)):
                field = derham.create_spline_function(space, space)
                rng = np.random.default_rng(3)
                field.vector._data[:] = xp.asarray(rng.random(field.vector._data.shape))
                with xp.profiling.count_transfers() as counter:
                    out = field.eval_tp_fixed_loc(spans, bases)
                assert counter.events == [], counter.report()
                results[backend, space] = xp.to_numpy(out)
    for space in ("H1", "L2"):
        np.testing.assert_allclose(results["cupy", space], results["numpy", space], rtol=1e-13, atol=1e-14)
