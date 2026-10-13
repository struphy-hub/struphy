"""End-to-end GPU test of :class:`~struphy.models.LinearVlasovAmpereOneSpecies` (issue #689).

A small run of the model (the energy-conservation setup of
``verification/test_verif_LinearVlasovAmpereOneSpecies.py`` with fewer markers) is done once on
the NumPy backend and once on the CuPy backend. The test checks that

1. both runs give the same energies, electric-field coefficients and marker weights/positions,
   within tolerances explained in :func:`compare_runs`, and
2. the time loop of the CuPy run does no host/device transfers through cunumpy, apart from an
   explicit budget for the diagnostics and the output (see :class:`~struphy.models.tests.gpu_e2e.TransferBudget`).

The CuPy part needs a GPU, or cunumpy's fake CuPy (``CUNUMPY_FAKE_CUPY=1``), on which the CUDA launches are emulated
on the CPU, and is skipped otherwise. It is MPI-compatible: under
``mpirun -n 2 pytest --with-mpi`` both runs use the same domain decomposition, and the markers are
gathered and compared by marker ID.

The shared parts (transfer budget, instrumentation, gathering the markers) are in
:mod:`struphy.models.tests.gpu_e2e`.
"""

import logging

import numpy as np
import pytest
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
from struphy.linear_algebra.solver import SolverParameters
from struphy.models import LinearVlasovAmpereOneSpecies
from struphy.models.tests.gpu_e2e import RunResult, requires_cupy_backend, run, shared_out_folder

logger = logging.getLogger("struphy")

AMPLITUDE = 0.1
R1 = 12.56  # box of length 4*pi, k = 0.5
NUM_ELEMENTS = (8, 1, 1)
PPC = 40  # 320 markers
DT = 0.1
N_STEPS = 3
SOLVER_TOL = 1e-12


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


def run_model(backend: str, out_folders: str) -> RunResult:
    """Run the simulation on ``backend`` and copy the results to the host."""
    return run(
        backend,
        make_simulation,
        out_folders,
        n_steps=N_STEPS,
        fields=lambda model: {"e_field": model.em_fields.e_field},
        particles=lambda model: model.kinetic_ions.var.particles,
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

    e_ref, e_gpu = ref.fields["e_field"], gpu.fields["e_field"]
    assert e_gpu.shape == e_ref.shape
    np.testing.assert_allclose(e_gpu, e_ref, rtol=0, atol=rtol * np.max(np.abs(e_ref)))

    if MPI.COMM_WORLD.Get_rank() == 0:
        for key, series in ref.scalars.items():
            np.testing.assert_allclose(gpu.scalars[key], series, rtol=rtol, err_msg=key)


@pytest.fixture
def out_folders(tmp_path_factory):
    return shared_out_folder(tmp_path_factory, "gpu_e2e_LinearVlasovAmpereOneSpecies")


def test_numpy_reference(out_folders):
    """The NumPy run that the GPU run is compared against; on NumPy nothing can be transferred."""
    ref = run_model("numpy", out_folders)
    check_reference(ref)
    ref.budget.check(n_scalars=ref.n_scalars)
    assert ref.budget.integrate.total == ref.budget.diagnostics.total == ref.budget.output.total == 0


@requires_cupy_backend
def test_cupy_matches_numpy_without_transfers(out_folders):
    """The CuPy run agrees with the NumPy run, and its time loop stays on the device.

    On a GPU, or on the fake CuPy (``CUNUMPY_FAKE_CUPY=1``) with every CUDA launch emulated on the CPU.
    """
    ref = run_model("numpy", out_folders)
    check_reference(ref)
    gpu = run_model("cupy", out_folders)
    compare_runs(ref, gpu)
    gpu.budget.check(n_scalars=gpu.n_scalars)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-ra"])
