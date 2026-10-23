import logging
import math
import os
import shutil

import h5py
import numpy as np
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
from struphy.models import VlasovMaxwellQuasiNeutral

logger = logging.getLogger("struphy")

# growth rate of the amplitude of B3 for k = 1.25, vth2^2 / vth1^2 = 12 (Kraus et al. 2017); the same for kappa = 0,
# since the displacement-current term gamma^2 ~ 8e-4 of the kappa = 1 dispersion relation is << k^2 + 1
GAMMA_EXACT = 0.02784


def _weibel_growth_rate(
    kappa: float, Np: int, Tend: float, t_fit: tuple, dt: float = 1.0, loading: str = "sobol_antithetic"
) -> float:
    """Runs the Weibel instability of the example weibel_quasi_neutral (1D-3V, k = 1.25, full-f markers) and returns
    the growth rate fitted to the magnetic energy, en_B ~ exp(2 gamma t), for t_fit[0] < t < t_fit[1].

    The antithetic Sobol loading (each Sobol point reflected u -> 1 - u in all phase-space directions, 64 markers
    per point, so that the mean velocities of each group vanish) removes most of the noise of the current, which
    limits the fit: with pseudo-random markers the fitted rate scatters by about 7 % between seeds
    (1e5 markers)."""
    model = VlasovMaxwellQuasiNeutral(alpha=1.0, epsilon=-1.0)

    test_folder = os.path.join(os.getcwd(), "struphy_verification_tests")
    out_folders = os.path.join(test_folder, "VlasovMaxwellQuasiNeutral")
    env = EnvironmentOptions(out_folders=out_folders, sim_folder=f"weibel_kappa{kappa:g}_Np{Np}_{loading}")

    k = 1.25
    vth1 = 0.02 / math.sqrt(2)
    vth2 = vth1 * math.sqrt(12)

    model.kinetic_ions.set_markers(
        loading_params=LoadingParameters(
            Np=Np,
            loading=loading,
            set_zero_velocity=(False, False, True),
            moments=(0.0, 0.0, 0.0, vth1, vth2, 1.0),
        ),
        weights_params=WeightsParameters(control_variate=False),
        boundary_params=BoundaryParameters(),
        sorting_params=SortingParameters(boxes_per_dim=(16, 1, 1), do_sort=True),
        saving_params=SavingParameters(n_markers=0),
        bufsize=2.0,
    )
    model.propagators.coupling.options = model.propagators.coupling.Options(kappa=kappa)
    model.propagators.push_eta.options = model.propagators.push_eta.Options()
    model.propagators.push_vxb.options = model.propagators.push_vxb.Options()
    model.initial_poisson.options = model.initial_poisson.Options(stab_mat="M0")

    model.kinetic_ions.var.add_background(maxwellians.Maxwellian3D(vth1=(vth1, None), vth2=(vth2, None)))
    model.em_fields.b_field.add_perturbation(perturbation=perturbations.ModesCos(amps=(-1e-4,), ls=(1,), comp=2))

    sim = Simulation(
        model=model,
        env=env,
        time_opts=Time(dt=dt, Tend=Tend, split_algo="LieTrotter"),
        domain=domains.Cuboid(r1=2 * math.pi / k),
        grid=grids.TensorProductGrid(num_elements=(32, 1, 1)),
        derham_opts=DerhamOptions(degree=(3, 1, 1)),
    )
    sim.run()

    # the scalars are global (reduced over the ranks), so the file of rank 0 holds them for all ranks
    comm = MPI.COMM_WORLD
    comm.Barrier()
    with h5py.File(os.path.join(env.path_out, "data", "data_proc0.hdf5"), "r") as file:
        t = np.asarray(file["time/value"])
        en_B = np.asarray(file["scalar/en_B"])
    comm.Barrier()

    mask = (t > t_fit[0]) & (t < t_fit[1])
    gamma = np.polyfit(t[mask], np.log(en_B[mask]), 1)[0] / 2
    return float(gamma)


def test_weibel_growth_rate_quasi_neutral():
    """Weibel instability in the quasi-neutral limit (kappa = 0, no displacement current): the magnetic energy grows
    with the analytical growth rate of the Vlasov-Maxwell system, gamma = 0.02784, which the displacement current does
    not change. With 1e5 full-f markers (antithetic Sobol loading) and dt = 1 the fitted rate is close to the
    analytical one (about 0.5 %), on 1, 2 and 4 MPI ranks."""
    gamma = _weibel_growth_rate(kappa=0.0, Np=100_000, Tend=160.0, t_fit=(80.0, 150.0))
    logger.info(f"Weibel growth rate (kappa = 0): {gamma = }, analytical {GAMMA_EXACT}")
    assert abs(gamma / GAMMA_EXACT - 1) < 0.07, f"{gamma = } vs. {GAMMA_EXACT = }"

    comm = MPI.COMM_WORLD
    comm.Barrier()
    if comm.Get_rank() == 0:
        shutil.rmtree(os.path.join(os.getcwd(), "struphy_verification_tests"), ignore_errors=True)
    comm.Barrier()


if __name__ == "__main__":
    test_weibel_growth_rate_quasi_neutral()
