"""A restarted run gives the same result as an uninterrupted one: run to T/2, restart from the checkpoint with a new
Simulation (as in a new job) and continue to T, then compare the final state with a run straight to T."""

import os

import numpy as np
from maybempi import MPI

from struphy import (
    DerhamOptions,
    EnvironmentOptions,
    LoadingParameters,
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
from struphy.models import Maxwell, VlasovAmpereOneSpecies

DT = 0.05


def _maxwell(out_folder: str, sim_folder: str, Tend: float, restart: bool = False) -> Simulation:
    model = Maxwell()
    model.em_fields.e_field.add_perturbation(perturbations.Noise(amp=0.1, comp=0, seed=123))
    model.em_fields.e_field.add_perturbation(perturbations.Noise(amp=0.1, comp=1, seed=124))
    return Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=out_folder, sim_folder=sim_folder, restart=restart),
        time_opts=Time(dt=DT, Tend=Tend),
        domain=domains.Cuboid(r3=20.0),
        grid=grids.TensorProductGrid(num_elements=(1, 1, 32)),
        derham_opts=DerhamOptions(degree=(1, 1, 3)),
    )


def _vlasov_ampere(out_folder: str, sim_folder: str, Tend: float, restart: bool = False) -> Simulation:
    model = VlasovAmpereOneSpecies(alpha=1.0, epsilon=-1.0)
    model.kinetic_ions.set_markers(
        loading_params=LoadingParameters(ppc=50, seed=1234),
        weights_params=WeightsParameters(control_variate=True),
        sorting_params=SortingParameters(boxes_per_dim=(8, 1, 1), do_sort=True),
        bufsize=0.4,
    )
    # the solver of the coupling step must not start from the previous solution (recycle): this state is not in the
    # checkpoint, the restarted run would differ within the solver tolerance (about 1e-7 relative)
    model.propagators.coupling_va.options = model.propagators.coupling_va.Options(
        solver_params=SolverParameters(recycle=False)
    )
    model.kinetic_ions.var.add_background(maxwellians.Maxwellian3D(n=(1.0, None)))
    perturbation = perturbations.ModesCos(ls=(1,), amps=(1e-2,))
    model.kinetic_ions.var.add_initial_condition(maxwellians.Maxwellian3D(n=(1.0, perturbation)))
    return Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=out_folder, sim_folder=sim_folder, restart=restart),
        time_opts=Time(dt=DT, Tend=Tend),
        domain=domains.Cuboid(r1=12.56),
        grid=grids.TensorProductGrid(num_elements=(16, 1, 1)),
        derham_opts=DerhamOptions(degree=(3, 1, 1)),
    )


def _run_straight_and_restarted(make_sim, out_folder: str, n_steps: int = 4):
    """The model of a run straight to n_steps dt, and the model of a run to n_steps/2 dt that is continued to
    n_steps dt by a new Simulation restarting from its checkpoint."""
    T = n_steps * DT
    straight = make_sim(out_folder, "straight", T)
    straight.run()

    make_sim(out_folder, "restarted", T / 2).run()
    restarted = make_sim(out_folder, "restarted", T, restart=True)
    restarted.run()
    return straight, restarted


def _feec(sim: Simulation, species: str, variable: str) -> np.ndarray:
    return getattr(getattr(sim.model, species), variable).spline.vector.toarray()


def _all_ranks(ok: bool) -> bool:
    """The same outcome on all ranks (the arrays compared are rank-local)."""
    comm = MPI.COMM_WORLD
    return bool(comm.allreduce(int(ok), op=MPI.MIN)) if comm.Get_size() > 1 else ok


def test_restart_maxwell(mpi_out_folder: str):
    straight, restarted = _run_straight_and_restarted(_maxwell, mpi_out_folder)

    assert restarted.time_state["value"][0] == straight.time_state["value"][0]
    for variable in ("e_field", "b_field"):
        a = _feec(straight, "em_fields", variable)
        b = _feec(restarted, "em_fields", variable)
        assert _all_ranks(np.any(a != 0.0) or a.size == 0), f"{variable} is zero: nothing to compare"
        assert _all_ranks(np.allclose(b, a, rtol=1e-12, atol=1e-14)), f"{variable}: restarted run differs"


def test_restart_vlasov_ampere(mpi_out_folder: str):
    straight, restarted = _run_straight_and_restarted(_vlasov_ampere, mpi_out_folder)

    assert restarted.time_state["value"][0] == straight.time_state["value"][0]
    a = _feec(straight, "em_fields", "e_field")
    b = _feec(restarted, "em_fields", "e_field")
    assert _all_ranks(np.allclose(b, a, rtol=1e-12, atol=1e-14)), "e_field: restarted run differs"

    # markers: the same ones on each rank, compared by their IDs (the order in the arrays may differ)
    def markers_by_id(sim):
        p = sim.model.kinetic_ions.var.particles
        m = p.markers[:, p.valid_mks]
        ids = m[p.index["ids"]]
        order = np.argsort(ids)
        rows = np.r_[p.index["pos"], p.index["vel"], p.index["weights"]]
        return ids[order], m[rows][:, order]

    ids_a, m_a = markers_by_id(straight)
    ids_b, m_b = markers_by_id(restarted)
    assert _all_ranks(np.array_equal(ids_a, ids_b)), "the restarted run has other markers on this rank"
    assert _all_ranks(np.allclose(m_b, m_a, rtol=1e-12, atol=1e-14)), "markers of the restarted run differ"


if __name__ == "__main__":
    out = os.path.join(os.getcwd(), "struphy_restart_tests")
    test_restart_maxwell(out)
    test_restart_vlasov_ampere(out)
