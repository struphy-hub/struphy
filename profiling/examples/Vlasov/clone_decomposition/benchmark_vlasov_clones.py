"""Benchmark clone versus spatial decomposition for a small Vlasov case.

The benchmark keeps the physical grid fixed and compares every valid pair of
``(num_clones, mpi_dims_mask)`` choices.  The deliberately small 4-by-4 grid
keeps one-clone spatial decompositions valid while making clone-heavy choices
available for comparison.

Run from the repository root with, for example::

    mpirun -n 4 python profiling/examples/Vlasov/clone_decomposition/benchmark_vlasov_clones.py
"""

from __future__ import annotations

import argparse
import logging

from feectools.ddm.mpi import mpi as MPI

from struphy import (
    BaseUnits,
    BoundaryParameters,
    DerhamOptions,
    EnvironmentOptions,
    LoadingParameters,
    Simulation,
    SortingParameters,
    Time,
    WeightsParameters,
    domains,
    equils,
    grids,
    maxwellians,
)
from struphy.models import LinearMHDVlasovCC
from struphy.topology import optimize_parallel_configuration


NUM_ELEMENTS = (8, 4, 1)
DEGREE = (2, 2, 1)
PPC = 2000
WARMUPS = 0
REPETITIONS = 1
TIME_STEPS = 5


def run_one_step(num_clones: int, mask: tuple[bool, bool, bool], output_dir: str) -> None:
    model = LinearMHDVlasovCC(hot_epsilon=1.0)
    for propagator in (
        model.propagators.couple_dens,
        model.propagators.shear_alf,
        model.propagators.couple_curr,
        model.propagators.push_eta,
        model.propagators.push_vxb,
        model.propagators.mag_sonic,
    ):
        propagator.options = propagator.Options()
    model.energetic_ions.var.add_background(maxwellians.Maxwellian3D(n=(1.0, None)))
    model.energetic_ions.set_markers(
        loading_params=LoadingParameters(ppc=PPC, seed=1234),
        weights_params=WeightsParameters(control_variate=False),
        boundary_params=BoundaryParameters(),
        sorting_params=SortingParameters(boxes_per_dim=NUM_ELEMENTS, do_sort=True),
        bufsize=0.4,
    )

    sim = Simulation(
        model=model,
        env=EnvironmentOptions(
            out_folders=output_dir,
            sim_folder=f"clones_{num_clones}_mask_{''.join('1' if value else '0' for value in mask)}",
            num_clones=num_clones,
            restart=False,
            save_restart=False,
        ),
        time_opts=Time(dt=0.05, Tend=TIME_STEPS * 0.05),
        domain=domains.Cuboid(r1=12.56),
        equil=equils.HomogenSlab(),
        grid=grids.TensorProductGrid(num_elements=NUM_ELEMENTS, mpi_dims_mask=mask),
        derham_opts=DerhamOptions(degree=DEGREE),
    )
    sim.run(profiling_activated=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", type=int, default=0, help="Unique profiling-run identifier.")
    args, _ = parser.parse_known_args()

    output = f"sim_{args.id:02d}"
    logging.getLogger("struphy").setLevel(logging.ERROR)
    result = optimize_parallel_configuration(
        NUM_ELEMENTS,
        lambda num_clones, mask: run_one_step(num_clones, mask, output),
        comm=MPI.COMM_WORLD,
        mask_pattern=(True, 1, 1),
        min_local_elements=DEGREE,
        warmups=WARMUPS,
        repetitions=REPETITIONS,
    )

    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"best configuration: clones={result.best_num_clones}, mask={result.best_mask}")
        for timing in result.timings:
            print(
                f"clones={timing.num_clones}, mask={timing.mask}: "
                f"{timing.seconds:.6f} s"
            )

        one_clone = [timing for timing in result.timings if timing.num_clones == 1]
        if one_clone:
            baseline = min(timing.seconds for timing in one_clone)
            speedup = baseline / min(timing.seconds for timing in result.timings)
            print(
                "speedup over best one-clone configuration under mask pattern: "
                f"{speedup:.2f}x ({(speedup - 1.0) * 100:.1f}%)"
            )
        else:
            print("one-clone configuration: unavailable under mask pattern")


if __name__ == "__main__":
    main()
