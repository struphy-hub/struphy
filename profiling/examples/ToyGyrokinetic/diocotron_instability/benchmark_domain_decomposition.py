"""Benchmark domain-decomposition masks for a cylindrical ToyDrift PIC case.

This is an intentionally anisotropic case: the logical grid has many radial
cells (512) and fewer azimuthal cells (16), while particles are loaded and
sorted in the cylindrical domain.  On the reference 8-rank run, automatic
selection prefers radial-only decomposition and was about 30% faster than the
default ``(True, True, True)`` mask.  The exact speedup is machine-dependent.

Run from the repository root, for example::

    mpirun -n 8 python profiling/examples/ToyGyrokinetic/diocotron_instability/benchmark_domain_decomposition.py

The benchmark times one ``Simulation.run(one_time_step=True)`` call, including
allocation, output setup, and profiling.  The decomposition comparison itself
is performed by the public optimizer.
"""

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
from struphy.models import ToyDrift
from struphy.topology import optimize_domain_decomposition


NUM_ELEMENTS = (512, 16, 1)
DEFAULT_MASK = (True, True, True)
WARMUPS = 1
REPETITIONS = 2


def run_one_step(
    mask: tuple[bool, bool, bool],
    output_dir: str,
    num_elements: tuple[int, int, int],
) -> None:
    """Build one candidate and perform one allocation/integration step."""
    model = ToyDrift(epsilon=1.0, alpha=1.0, base_units=BaseUnits(kBT=1.0))
    domain = domains.HollowCylinder(a1=1.0, a2=10.0, Lz=10.0)
    equil = equils.HomogenSlab()

    model.kinetic_ions.set_markers(
        loading_params=LoadingParameters(ppc=50, loading="sobol_standard", spatial="disc"),
        weights_params=WeightsParameters(control_variate=True, reject_weights=True, threshold=0.0001),
        boundary_params=BoundaryParameters(),
        sorting_params=SortingParameters(
            boxes_per_dim=num_elements,
            do_sort=True,
            sorting_frequency=1,
        ),
        bufsize=2.0,
    )
    model.kinetic_ions.var.add_background(
        maxwellians.GyroMaxwellian2D(n=(0.0, None), B0=2.0),
    )

    sim = Simulation(
        model=model,
        env=EnvironmentOptions(
            out_folders=output_dir,
            sim_folder="mask_" + "".join("1" if value else "0" for value in mask),
            restart=False,
        ),
        time_opts=Time(dt=0.01, Tend=0.01, split_algo="LieTrotter"),
        domain=domain,
        equil=equil,
        grid=grids.TensorProductGrid(num_elements=num_elements, mpi_dims_mask=mask),
        derham_opts=DerhamOptions(
            degree=(2, 2, 1),
            bcs=(("dirichlet", "dirichlet"), None, None),
        ),
    )
    sim.run(one_time_step=True, profiling_activated=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", type=int, default=0, help="Unique profiling-run identifier.")
    args, _ = parser.parse_known_args()

    output = f"sim_{args.id:02d}"

    logging.getLogger("struphy").setLevel(logging.ERROR)
    result = optimize_domain_decomposition(
        NUM_ELEMENTS,
        lambda mask: run_one_step(mask, output, NUM_ELEMENTS),
        comm=MPI.COMM_WORLD,
        min_local_elements=(2, 2, 1),
        warmups=WARMUPS,
        repetitions=REPETITIONS,
    )

    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"best mask: {result.best_mask}")
        for timing in result.timings:
            print(f"{timing.mask}: {timing.seconds:.6f} s")

        default = next(
            timing.seconds for timing in result.timings if timing.mask == DEFAULT_MASK
        )
        speedup = default / min(timing.seconds for timing in result.timings)
        print(f"speedup over default: {speedup:.2f}x ({(speedup - 1.0) * 100:.1f}%)")


if __name__ == "__main__":
    main()
