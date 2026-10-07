"""End-to-end GPU test of :class:`~struphy.models.LinearVlasovAmpereOneSpecies` (issue #689).

A small run of the model (the energy-conservation setup of
``verification/test_verif_LinearVlasovAmpereOneSpecies.py`` with fewer markers) is done once on
the NumPy backend and once on the CuPy backend. The test checks that

1. both runs give the same energies, electric-field coefficients and marker weights/positions,
   within tolerances explained in :func:`compare_runs`, and
2. the time loop of the CuPy run does no host/device transfers through cunumpy, apart from an
   explicit budget for the diagnostics and the output (see :class:`TransferBudget`).

The CuPy part needs a GPU (``requires_cupy``) and is skipped otherwise. It is MPI-compatible: under
``mpirun -n 2 pytest --with-mpi`` both runs use the same domain decomposition, and the markers are
gathered and compared by marker ID.

Only transfers made through cunumpy are counted (``to_numpy``, ``to_cupy``, host fallbacks of
kernels, ...). Implicit synchronizations such as ``float(device_scalar)`` in the convergence check
of the iterative solver are not seen; use ``nsys`` for those.
"""

import logging
import os
from dataclasses import dataclass, field

import cunumpy as xp
import h5py
import numpy as np
import pytest
from cunumpy.kernel_testing import requires_cupy
from cunumpy.profiling import TransferCounter, count_transfers
from maybempi import MPI

from struphy import (
    BoundaryParameters,
    DerhamOptions,
    EnvironmentOptions,
    LoadingParameters,
    SavingParameters,
    Simulation,
    SortingParameters,
    Time,
    WeightsParameters,
    domains,
    grids,
    maxwellians,
    perturbations,
)
from struphy.io.output_handling import DataContainer
from struphy.linear_algebra.solver import SolverParameters
from struphy.models import LinearVlasovAmpereOneSpecies

logger = logging.getLogger("struphy")

AMPLITUDE = 0.1
R1 = 12.56  # box of length 4*pi, k = 0.5
NUM_ELEMENTS = (8, 1, 1)
PPC = 40  # 320 markers
DT = 0.1
N_STEPS = 3
SOLVER_TOL = 1e-12

# transfer kinds that move data between host and device (device_copy is device-only and allowed)
HOST_DEVICE_KINDS = ("to_host", "to_device", "kernel_conversion", "fallback")


def make_simulation(out_folders: str, sim_folder: str) -> Simulation:
    """The model on a periodic 1D Cuboid, with the active cunumpy backend.

    Deterministic marker loading (``sobol_standard``) so that both backends start from the same
    markers: ``pseudo_random`` draws from ``xp.random``, whose streams differ between NumPy and CuPy.
    No B0 and no E0, so the time step is ``PushEta`` followed by ``EfieldWeightsCoupling``. No sorting
    boxes (the verification test uses them): the model does not need them, and the box sorting
    kernels have no CUDA version yet.
    """
    model = LinearVlasovAmpereOneSpecies(alpha=1.0, epsilon=-1.0, with_B0=False, with_E0=False)

    env = EnvironmentOptions(out_folders=out_folders, sim_folder=sim_folder)
    time_opts = Time(dt=DT, Tend=round(N_STEPS * DT, 14))  # 3 * 0.1 > 0.3 would add a step
    domain = domains.Cuboid(r1=R1)
    grid = grids.TensorProductGrid(num_elements=NUM_ELEMENTS)
    derham_opts = DerhamOptions(degree=(3, 1, 1))

    model.kinetic_ions.set_markers(
        loading_params=LoadingParameters(ppc=PPC, loading="sobol_standard"),
        weights_params=WeightsParameters(),
        boundary_params=BoundaryParameters(),
        # no sorting boxes: optional for this model, and box sorting has no device version yet
        sorting_params=SortingParameters(),
        saving_params=SavingParameters(),
        bufsize=0.4,
    )

    model.propagators.push_eta.options = model.propagators.push_eta.Options()
    # a tight solver tolerance, so that the backends differ by round-off and not by where pcg stops
    model.propagators.coupling_Eweights.options = model.propagators.coupling_Eweights.Options(
        solver_params=SolverParameters(tol=SOLVER_TOL),
    )
    model.initial_poisson.options = model.initial_poisson.Options(
        stab_mat="M0",
        stab_eps=1e-6,
        solver_params=SolverParameters(tol=SOLVER_TOL),
    )

    model.kinetic_ions.var.add_background(maxwellians.Maxwellian3D(n=(1.0, None)))
    perturbation = perturbations.ModesCos(ls=(1,), amps=(AMPLITUDE,))
    model.kinetic_ions.var.add_initial_condition(maxwellians.Maxwellian3D(n=(1.0, perturbation)))

    return Simulation(
        model=model,
        env=env,
        time_opts=time_opts,
        domain=domain,
        grid=grid,
        derham_opts=derham_opts,
    )


@dataclass
class TransferBudget:
    """Transfers recorded per phase of the time loop (setup before it is not counted).

    ``integrate`` (the propagators) must not transfer at all. ``diagnostics`` (scalars, saved
    markers, binned distribution functions, coefficient extraction) may copy small results to the
    host: at most a few scalars per call, never a host-to-device copy or a host kernel. ``output``
    (``DataContainer.save_data``) may copy each saved dataset to the host once, nothing more.
    """

    integrate: TransferCounter = field(default_factory=TransferCounter)
    diagnostics: TransferCounter = field(default_factory=TransferCounter)
    output: TransferCounter = field(default_factory=TransferCounter)
    # allowed device-to-host copies by `output`: one per saved dataset, and their bytes
    output_allowed_copies: int = 0
    output_allowed_bytes: int = 0
    n_integrate_calls: int = 0
    n_diagnostics_calls: int = 0

    def check(self, n_scalars: int):
        report = (
            f"integrate:\n{self.integrate.report()}\n"
            f"diagnostics:\n{self.diagnostics.report()}\n"
            f"output:\n{self.output.report()}"
        )
        assert self.n_integrate_calls == N_STEPS, report

        # 1. the propagators: nothing between host and device
        assert all(e.kind not in HOST_DEVICE_KINDS for e in self.integrate.events), report

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


def instrument(sim: Simulation, monkeypatch) -> TransferBudget:
    """Count the transfers of the time loop of ``sim.run()`` per phase.

    Counting starts once the HDF5 datasets are created, the last step of the setup in
    :meth:`Simulation.run`. The model methods are patched on the instance, ``save_data`` on the class
    (the ``DataContainer`` is created inside ``run``).
    """
    budget = TransferBudget()
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

    monkeypatch.setattr(sim, "_initialize_hdf5_datasets", init_datasets)
    monkeypatch.setattr(model, "integrate", counted(model.integrate, "integrate", "n_integrate_calls"))
    monkeypatch.setattr(
        model,
        "update_scalar_quantities",
        counted(model.update_scalar_quantities, "diagnostics", "n_diagnostics_calls"),
    )
    for name in ("update_markers_to_be_saved", "update_distr_functions"):
        monkeypatch.setattr(model, name, counted(getattr(model, name), "diagnostics"))

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

    monkeypatch.setattr(DataContainer, "save_data", save_data)
    return budget


@dataclass
class RunResult:
    """Host copies of what is compared between the backends."""

    scalars: dict  # name -> time series (rank 0 only, else empty)
    e_coeffs: np.ndarray  # local electric-field coefficients at the end
    markers: np.ndarray  # all valid markers (gathered), sorted by marker ID
    n_scalars: int
    budget: TransferBudget


def run(backend: str, out_folders: str, monkeypatch) -> RunResult:
    """Run the simulation on ``backend`` and copy the results to the host."""
    with xp.use_backend(backend):
        sim = make_simulation(out_folders, sim_folder=backend)
        budget = instrument(sim, monkeypatch)
        sim.run()
        monkeypatch.undo()

        model = sim.model
        comm = MPI.COMM_WORLD

        e_coeffs = np.asarray(xp.to_numpy(model.em_fields.e_field.spline.vector.toarray()))

        particles = model.kinetic_ions.var.particles
        local = np.asarray(xp.to_numpy(particles.markers_wo_holes_and_ghost))
        parts = comm.allgather(local) if comm.Get_size() > 1 else [local]
        markers = np.concatenate(parts, axis=0)
        markers = markers[np.argsort(markers[:, -1])]

        scalars = {}
        if comm.Get_rank() == 0:
            with h5py.File(os.path.join(sim.env.path_out, "data", "data_proc0.hdf5"), "r") as f:
                for key in model.scalars.dct:
                    scalars[key] = f["scalar"][key][()].ravel()

    return RunResult(
        scalars=scalars, e_coeffs=e_coeffs, markers=markers, n_scalars=len(model.scalars.dct), budget=budget
    )


def check_reference(ref: RunResult):
    """Physics sanity of the NumPy run: the comparison is only meaningful if the reference is right."""
    assert ref.markers.shape[0] == PPC * int(np.prod(NUM_ELEMENTS))
    if MPI.COMM_WORLD.Get_rank() == 0:
        en_E, en_tot = ref.scalars["en_E"], ref.scalars["en_tot"]
        assert en_tot.size == N_STEPS + 1
        # energy moves between field and particles ...
        assert abs(en_E[-1] - en_E[0]) > 1e-3 * en_tot[0], en_E
        # ... and the Crank-Nicolson step conserves the total
        drift = np.max(np.abs(en_tot - en_tot[0])) / en_tot[0]
        assert drift < 1e-6, f"{en_tot = }, {drift = }"


def compare_runs(ref: RunResult, gpu: RunResult):
    """Compare the CuPy run against the NumPy run.

    Tolerances:

    * Velocities do not change (no E0, no B0) and positions are pushed explicitly
      (``eta += dt * v / r1``, mod 1): the same floating-point operations on both backends, up to
      fused multiply-adds on the GPU, so ``rtol = atol = 1e-13``.
    * Weights, field coefficients and energies come out of ``EfieldWeightsCoupling``: a
      Crank-Nicolson step whose Schur complement is solved by pcg to a relative residual of
      ``SOLVER_TOL = 1e-12``, and whose matrix and vector are accumulated over markers (on the GPU
      with atomic adds, i.e. in a different order). On this grid of 8 elements the preconditioned
      system has a condition number of order 10-100, so the two solutions differ by at most
      ~1e-10 relative to their norm after 3 steps; the tolerance ``1e-8`` (relative to the largest
      entry) leaves a factor 100 against that and is still far below the change per step (~1e-2).
    """
    rtol = 1e-8
    np.testing.assert_array_equal(gpu.markers[:, -1], ref.markers[:, -1])  # same markers
    np.testing.assert_allclose(gpu.markers[:, 3:6], ref.markers[:, 3:6], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(gpu.markers[:, 0:3], ref.markers[:, 0:3], rtol=1e-13, atol=1e-13)

    w_ref, w_gpu = ref.markers[:, 6], gpu.markers[:, 6]
    np.testing.assert_allclose(w_gpu, w_ref, rtol=0, atol=rtol * np.max(np.abs(w_ref)))

    assert gpu.e_coeffs.shape == ref.e_coeffs.shape
    np.testing.assert_allclose(gpu.e_coeffs, ref.e_coeffs, rtol=0, atol=rtol * np.max(np.abs(ref.e_coeffs)))

    if MPI.COMM_WORLD.Get_rank() == 0:
        for key, series in ref.scalars.items():
            np.testing.assert_allclose(gpu.scalars[key], series, rtol=rtol, err_msg=key)


@pytest.fixture
def out_folders(tmp_path_factory):
    """One output folder for all ranks (the one of rank 0; ``tmp_path`` differs per rank)."""
    comm = MPI.COMM_WORLD
    path = str(tmp_path_factory.mktemp("gpu_e2e_LinearVlasovAmpereOneSpecies")) if comm.Get_rank() == 0 else None
    return comm.bcast(path, root=0)


def test_numpy_reference(out_folders, monkeypatch):
    """The NumPy run that the GPU run is compared against; on NumPy nothing can be transferred."""
    ref = run("numpy", out_folders, monkeypatch)
    check_reference(ref)
    ref.budget.check(n_scalars=ref.n_scalars)
    assert ref.budget.integrate.total == ref.budget.diagnostics.total == ref.budget.output.total == 0


@requires_cupy
def test_cupy_matches_numpy_without_transfers(out_folders, monkeypatch):
    """The CuPy run agrees with the NumPy run, and its time loop stays on the device."""
    ref = run("numpy", out_folders, monkeypatch)
    check_reference(ref)
    gpu = run("cupy", out_folders, monkeypatch)
    compare_runs(ref, gpu)
    gpu.budget.check(n_scalars=gpu.n_scalars)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-ra"])
