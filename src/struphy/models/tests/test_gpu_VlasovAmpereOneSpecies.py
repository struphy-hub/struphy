"""End-to-end GPU test of :class:`~struphy.models.VlasovAmpereOneSpecies` (issue #689).

A small Landau-damping run (the setup of ``verification/test_verif_VlasovAmpereOneSpecies.py`` with fewer cells and
markers, full-f weights, and a larger perturbation so that energy moves within three steps) is done once on the NumPy
backend and once on the CuPy backend. The test checks that

1. both runs give the same energies, electric-field coefficients and markers, within the tolerances explained in
   :func:`compare_runs`, and
2. the time loop of the CuPy run does no host/device transfers through cunumpy, apart from an explicit budget for
   the diagnostics and the output (:class:`~struphy.models.tests.gpu_e2e.TransferBudget`).

A time step is ``PushEta`` followed by ``VlasovAmpereCoupling`` (no B0). The setup also runs the initial Poisson
solve (``charge_density_0form`` and the ``L2Projector`` with its ``MassMatrixPreconditioner``) on the device.

The CuPy part needs a GPU, or cunumpy's fake CuPy (``CUNUMPY_FAKE_CUPY=1``), on which every CUDA launch is emulated on
the CPU. Without either, :func:`test_fake_cupy_matches_numpy` runs the comparison in a serial child process on the
fake CuPy. The tests are MPI-compatible (``mpirun -n 2 pytest --with-mpi``): both runs use the same domain
decomposition, and the markers are gathered and compared by marker ID.
"""

import logging

import numpy as np
import pytest
from maybempi import MPI

from struphy import (
    BinningPlot,
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
from struphy.geometry.tests.test_domain import run_fake_cupy_child
from struphy.linear_algebra.solver import SolverParameters
from struphy.models import VlasovAmpereOneSpecies
from struphy.models.tests.gpu_e2e import (
    RunResult,
    cupy_backend_available,
    requires_cupy_backend,
    run,
    shared_out_folder,
)

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

    Deterministic marker loading (``sobol_standard``), so that both backends start from the same markers:
    ``pseudo_random`` draws from ``xp.random``, whose streams differ between NumPy and CuPy. No sorting boxes, since
    ``SortingBoxes`` does not work on the CuPy backend yet (#726); the model does not need them. Full-f weights: with
    the control variate, ``kinetic_energy`` at t=0 is computed with the full-f weights and afterwards with the delta-f
    weights, so ``total_energy`` cannot be checked.
    """
    model = VlasovAmpereOneSpecies(alpha=1.0, epsilon=-1.0, with_B0=False)

    env = EnvironmentOptions(out_folders=out_folders, sim_folder=sim_folder, save_restart=False)
    time_opts = Time(dt=DT, Tend=round(N_STEPS * DT, 14))

    model.kinetic_ions.set_markers(
        loading_params=LoadingParameters(ppc=PPC, loading="sobol_standard"),
        weights_params=WeightsParameters(),
        boundary_params=BoundaryParameters(),
        sorting_params=SortingParameters(),
        # the binned distribution function of the verification test, so that the diagnostics run on the device too
        saving_params=SavingParameters(
            binning_plots=(BinningPlot(slice="e1_v1", n_bins=(16, 16), ranges=((0.0, 1.0), (-5.0, 5.0))),),
        ),
        bufsize=0.4,
    )

    model.propagators.push_eta.options = model.propagators.push_eta.Options()
    # tight solver tolerances, so that the backends differ by round-off and not by where pcg stops
    model.propagators.coupling_va.options = model.propagators.coupling_va.Options(
        solver_params=SolverParameters(tol=SOLVER_TOL),
    )
    # stab_eps=1e-6, not the default 1e-14: the Monte-Carlo net charge puts (net charge) / stab_eps into the constant
    # mode of phi, so with 1e-14 phi is ~1e14 and E = -grad(phi) loses ~14 digits to cancellation (5% of E here, on
    # either backend)
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
        domain=domains.Cuboid(r1=R1),
        grid=grids.TensorProductGrid(num_elements=NUM_ELEMENTS),
        derham_opts=DerhamOptions(degree=(3, 1, 1)),
    )


def run_model(backend: str, out_folders: str) -> RunResult:
    """Run the simulation on ``backend`` and copy the results to the host."""
    return run(
        backend,
        make_simulation,
        out_folders,
        n_steps=N_STEPS,
        fields=lambda model: {"e_field": model.em_fields.e_field, "phi": model.em_fields.phi},
        particles=lambda model: model.kinetic_ions.var.particles,
    )


def check_reference(ref: RunResult):
    """Physics sanity of the NumPy run: the comparison is only meaningful if the reference is right."""
    assert ref.markers.shape[0] == PPC * int(np.prod(NUM_ELEMENTS))
    if MPI.COMM_WORLD.Get_rank() == 0:
        en_E = ref.scalars["electric_energy"]
        en_tot = ref.scalars["total_energy"]
        assert en_tot.size == N_STEPS + 1
        # energy moves between field and particles ...
        assert abs(en_E[-1] - en_E[0]) > 1e-2 * abs(en_E[0]), en_E
        # ... and the Crank-Nicolson coupling conserves the total from step to step, to the solver tolerance. The value
        # at t=0 differs by ~2e-4 relative (the initial diagnostic, not the time stepping), so it is checked loosely.
        drift = np.max(np.abs(en_tot[1:] - en_tot[1])) / abs(en_tot[1])
        assert drift < 1e-9, f"{en_tot = }, {drift = }"
        assert abs(en_tot[1] - en_tot[0]) < 1e-3 * abs(en_tot[0]), en_tot


def compare_runs(ref: RunResult, gpu: RunResult):
    """Compare the CuPy run against the NumPy run.

    Velocities, the electric field and the energies come out of ``VlasovAmpereCoupling``, a Crank-Nicolson step whose
    Schur complement is solved by pcg to a relative residual of ``SOLVER_TOL = 1e-12``, with the matrix and vector
    accumulated over markers (on the GPU with atomic adds, i.e. in another order). On this grid of 8 elements the
    preconditioned system has a condition number of order 10-100, so the two solutions differ by ~1e-10 relative to
    their norm after three steps. Positions are pushed explicitly with these velocities, so they inherit
    ``DT * 1e-10``. The tolerance ``1e-8`` (relative to the largest entry) leaves a factor 100 and is still far below
    the change per step (~5e-2 in the electric energy). Full-f weights do not change, so they must agree to round-off.
    """
    rtol = 1e-8
    np.testing.assert_array_equal(gpu.markers[:, -1], ref.markers[:, -1])  # same markers

    def close(a, b, name):
        np.testing.assert_allclose(a, b, rtol=0, atol=rtol * max(np.max(np.abs(b)), 1e-300), err_msg=name)

    close(gpu.markers[:, 0:3], ref.markers[:, 0:3], "positions")
    close(gpu.markers[:, 3:6], ref.markers[:, 3:6], "velocities")
    np.testing.assert_allclose(gpu.markers[:, 6], ref.markers[:, 6], rtol=1e-13, atol=0, err_msg="weights")
    for name, coeffs in ref.fields.items():
        assert gpu.fields[name].shape == coeffs.shape, name
        close(gpu.fields[name], coeffs, name)

    if MPI.COMM_WORLD.Get_rank() == 0:
        for key, series in ref.scalars.items():
            np.testing.assert_allclose(gpu.scalars[key], series, rtol=rtol, err_msg=key)


def check_on_cupy(out_folders: str):
    """The whole comparison, for the fake-CuPy child process of :func:`test_fake_cupy_matches_numpy`."""
    ref = run_model("numpy", out_folders)
    check_reference(ref)
    gpu = run_model("cupy", out_folders)
    compare_runs(ref, gpu)
    gpu.budget.check(n_scalars=gpu.n_scalars)


@pytest.fixture
def out_folders(tmp_path_factory):
    return shared_out_folder(tmp_path_factory, "gpu_e2e_VlasovAmpereOneSpecies")


def test_numpy_reference(out_folders):
    """The NumPy run that the GPU run is compared against; on NumPy nothing can be transferred."""
    ref = run_model("numpy", out_folders)
    check_reference(ref)
    ref.budget.check(n_scalars=ref.n_scalars)
    assert ref.budget.integrate.total == ref.budget.diagnostics.total == ref.budget.output.total == 0


@requires_cupy_backend
def test_cupy_matches_numpy_without_transfers(out_folders):
    """The CuPy run agrees with the NumPy run, and its time loop stays on the device.

    On a GPU, or on the fake CuPy (``CUNUMPY_FAKE_CUPY=1``) with every CUDA launch emulated on the CPU; under
    ``mpirun -n 2`` with two ranks.
    """
    check_on_cupy(out_folders)


@pytest.mark.skipif(cupy_backend_available(), reason="runs in this process (test_cupy_matches_numpy_without_transfers)")
def test_fake_cupy_matches_numpy(out_folders):
    """Without a GPU and without ``CUNUMPY_FAKE_CUPY``: the comparison in a serial child process on the fake CuPy."""
    run_fake_cupy_child(
        f"from struphy.models.tests.test_gpu_VlasovAmpereOneSpecies import check_on_cupy; check_on_cupy({out_folders!r})"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-ra"])
