"""Shared helpers of the end-to-end GPU tests of the models (issue #689).

A test runs a small simulation once on the NumPy backend and once on the CuPy backend, compares the two, and checks
that the time loop of the CuPy run makes no host/device transfers through cunumpy, apart from an explicit budget for
the diagnostics and the output (:class:`TransferBudget`).

The CuPy run needs a GPU, or cunumpy's fake CuPy (``CUNUMPY_FAKE_CUPY=1``), on which every CUDA launch is emulated
on the CPU (:func:`device_context`). Without either, a test can still run the comparison in a serial child process
on the fake CuPy (:func:`run_fake_cupy_child`).

Only transfers made through cunumpy are counted (``to_numpy``, ``to_cupy``, host fallbacks of kernels, ...).
Implicit synchronizations such as ``float(device_scalar)`` are not seen; use ``nsys`` for those.
"""

import contextlib
import os
from collections.abc import Callable
from dataclasses import dataclass, field

import cunumpy as xp
import h5py
import numpy as np
import pytest
from cunumpy.kernel_testing import fake_cupy_active
from cunumpy.profiling import TransferCounter, count_transfers
from maybempi import MPI

from struphy.io.output_handling import DataContainer

# transfer kinds that move data between host and device (device_copy is device-only and allowed)
HOST_DEVICE_KINDS = ("to_host", "to_device", "kernel_conversion", "fallback")


def cupy_backend_available() -> bool:
    """Whether the CuPy backend can run here: a GPU, or the fake CuPy (with emulated launches)."""
    return fake_cupy_active() or xp.cupy_available()


requires_cupy_backend = pytest.mark.skipif(
    not cupy_backend_available(), reason="needs a GPU or the fake CuPy (CUNUMPY_FAKE_CUPY=1)"
)


@contextlib.contextmanager
def device_context():
    """Context for the CuPy run: on the fake CuPy every CUDA launch is emulated on the CPU, on a GPU nothing.

    On the fake CuPy, :meth:`CudaKernel.compile` (called by ``Simulation.compile_cuda_kernels`` before the time loop)
    does nothing: the fake CuPy cannot compile CUDA, and the emulated launches compile their own C++ version.
    ``compile_cuda_kernels`` still checks that every kernel of the time loop has a CUDA version.
    """
    if not fake_cupy_active():
        yield
        return
    from unittest import mock

    from cunumpy.kernel_testing import emulated_launches
    from cunumpy.kernels import CudaKernel

    from struphy.pic.tests.cuda_emulation import EMULATION_OPTIONS

    with (
        emulated_launches(options=EMULATION_OPTIONS),
        mock.patch.object(CudaKernel, "compile", lambda self, **kw: None),
    ):
        yield


@dataclass
class TransferBudget:
    """Transfers recorded per phase of the time loop (the setup before it is not counted).

    ``integrate`` (the propagators) may only copy scalars to the host: the convergence test of the Krylov solvers
    reads the 8-byte residual norm once per iteration (feectools keeps everything else of the solve on the device).
    No arrays, no host-to-device copies, no host kernels. ``diagnostics`` (scalars, saved markers, binned
    distribution functions) may copy small results to the host: at most one scalar per tracked scalar and call,
    never a host-to-device copy or a host kernel. ``output`` (``DataContainer.save_data``) may copy each saved
    dataset to the host once, nothing more.
    """

    n_steps: int
    integrate: TransferCounter = field(default_factory=TransferCounter)
    diagnostics: TransferCounter = field(default_factory=TransferCounter)
    output: TransferCounter = field(default_factory=TransferCounter)
    # allowed device-to-host copies by `output`: one per saved dataset, and their bytes
    output_allowed_copies: int = 0
    output_allowed_bytes: int = 0
    n_integrate_calls: int = 0
    n_diagnostics_calls: int = 0

    def report(self) -> str:
        return (
            f"integrate:\n{self.integrate.report()}\n"
            f"diagnostics:\n{self.diagnostics.report()}\n"
            f"output:\n{self.output.report()}"
        )

    def check(self, n_scalars: int):
        report = self.report()
        assert self.n_integrate_calls == self.n_steps, report

        # 1. the propagators: only scalars to the host (solver convergence tests), nothing else
        for e in self.integrate.events:
            if e.kind == "to_host":
                assert e.nbytes is not None and e.nbytes <= 8, report
            else:
                assert e.kind not in HOST_DEVICE_KINDS, report

        # 2. diagnostics: no host-to-device copies, no host kernels, only scalar-sized downloads
        for kind in ("to_device", "kernel_conversion", "fallback"):
            assert self.diagnostics.count(kind) == 0, report
        assert self.diagnostics.to_host <= n_scalars * self.n_diagnostics_calls, report
        assert self.diagnostics.bytes_to_host <= 8 * n_scalars * self.n_diagnostics_calls, report

        # 3. output: each saved dataset once to the host
        for kind in ("to_device", "kernel_conversion", "fallback"):
            assert self.output.count(kind) == 0, report
        assert self.output.to_host <= self.output_allowed_copies, report
        assert self.output.bytes_to_host <= self.output_allowed_bytes, report


class _Patcher:
    """Minimal ``monkeypatch``: set attributes, and restore them in reverse order."""

    def __init__(self):
        self._saved = []

    def setattr(self, obj, name, value):
        self._saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        while self._saved:
            obj, name, value = self._saved.pop()
            setattr(obj, name, value)


def instrument(sim, patcher, n_steps: int) -> TransferBudget:
    """Count the transfers of the time loop of ``sim.run()`` per phase.

    Counting starts once the HDF5 datasets are created, the last step of the setup in :meth:`Simulation.run`. The
    model methods are patched on the instance, ``save_data`` on the class (the ``DataContainer`` is created inside
    ``run``).
    """
    budget = TransferBudget(n_steps=n_steps)
    in_loop = [False]
    model = sim.model

    def counted(fn, phase: str, calls: str | None = None):
        def wrapper(*args, **kwargs):
            if not in_loop[0]:
                return fn(*args, **kwargs)
            with count_transfers() as counter:
                out = fn(*args, **kwargs)
            getattr(budget, phase).events.extend(counter.events)
            if calls is not None:
                setattr(budget, calls, getattr(budget, calls) + 1)
            return out

        return wrapper

    original_init_datasets = sim._initialize_hdf5_datasets

    def init_datasets(*args, **kwargs):
        out = original_init_datasets(*args, **kwargs)
        in_loop[0] = True
        return out

    patcher.setattr(sim, "_initialize_hdf5_datasets", init_datasets)
    patcher.setattr(model, "integrate", counted(model.integrate, "integrate", "n_integrate_calls"))
    patcher.setattr(
        model,
        "update_scalar_quantities",
        counted(model.update_scalar_quantities, "diagnostics", "n_diagnostics_calls"),
    )
    for name in ("update_markers_to_be_saved", "update_distr_functions"):
        patcher.setattr(model, name, counted(getattr(model, name), "diagnostics"))

    original_save_data = DataContainer.save_data

    def save_data(self, keys=None):
        if not in_loop[0]:
            return original_save_data(self, keys=keys)
        for key in self._dset_dict if keys is None else keys:
            val = self._dset_dict[key]
            budget.output_allowed_copies += 1
            budget.output_allowed_bytes += int(getattr(val, "nbytes", 8))
        with count_transfers() as counter:
            original_save_data(self, keys=keys)
        budget.output.events.extend(counter.events)

    patcher.setattr(DataContainer, "save_data", save_data)
    return budget


@dataclass
class RunResult:
    """Host copies of what is compared between the backends."""

    scalars: dict  # name -> time series (rank 0 only, else empty)
    fields: dict  # FEEC variable name -> local coefficients at the end
    markers: np.ndarray  # all valid markers (gathered from all ranks), sorted by marker ID
    n_scalars: int
    budget: TransferBudget


def gather_markers(particles) -> np.ndarray:
    """The valid markers of all ranks on the host, sorted by marker ID (the last column)."""
    comm = MPI.COMM_WORLD
    local = np.asarray(xp.to_numpy(particles.markers_wo_holes_and_ghost))
    parts = comm.allgather(local) if comm.Get_size() > 1 else [local]
    markers = np.concatenate(parts, axis=0)
    return markers[np.argsort(markers[:, -1])]


def run(
    backend: str,
    make_simulation: Callable[[str, str], object],
    out_folders: str,
    n_steps: int,
    fields: Callable[[object], dict],
    particles: Callable[[object], object],
) -> RunResult:
    """Run the simulation made by ``make_simulation(out_folders, sim_folder)`` on ``backend``; copy results to host.

    ``fields(model)`` returns the FEEC variables to compare (name -> ``FEECVariable``), ``particles(model)`` the
    ``Particles`` object whose markers are compared.
    """
    with xp.use_backend(backend), device_context() if backend == "cupy" else contextlib.nullcontext():
        sim = make_simulation(out_folders, backend)
        patcher = _Patcher()
        budget = instrument(sim, patcher, n_steps)
        try:
            sim.run()
        finally:
            patcher.undo()

        model = sim.model
        coeffs = {name: np.asarray(xp.to_numpy(var.spline.vector.toarray())) for name, var in fields(model).items()}
        markers = gather_markers(particles(model))

        scalars = {}
        if MPI.COMM_WORLD.Get_rank() == 0:
            with h5py.File(os.path.join(sim.env.path_out, "data", "data_proc0.hdf5"), "r") as f:
                for key in model.scalars.dct:
                    scalars[key] = f["scalar"][key][()].ravel()

    return RunResult(scalars=scalars, fields=coeffs, markers=markers, n_scalars=len(model.scalars.dct), budget=budget)


def shared_out_folder(tmp_path_factory, name: str) -> str:
    """One output folder for all ranks (the one of rank 0; ``tmp_path`` differs per rank)."""
    comm = MPI.COMM_WORLD
    path = str(tmp_path_factory.mktemp(name)) if comm.Get_rank() == 0 else None
    return comm.bcast(path, root=0)
