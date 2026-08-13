"""
Worker script for :mod:`test_pusher_openmp_speedup`.

Builds a marker set and times repeated calls to the pyccelized kernel
``push_vxb_analytic``. Prints the average wall time per call (in seconds) as
the last line on stdout.

Usage::

    OMP_NUM_THREADS=<n> python _bench_push_vxb_analytic.py <num_elements> <degree> <ppc>
"""

import sys
import time

from cunumpy import PyccelKernel
from feectools.ddm.mpi import mpi as MPI

from struphy import LoadingParameters, domains
from struphy.feec.psydac_derham import Derham
from struphy.feec.utilities import create_equal_random_arrays
from struphy.io.options import DerhamOptions
from struphy.pic.particles import Particles6D
from struphy.pic.pushing import pusher_kernels
from struphy.pic.pushing.pusher import Pusher as Pusher_psy
from struphy.topology.grids import TensorProductGrid

N_REPS = 5


def main(num_elements: int, degree: int, ppc: int) -> float:
    comm = MPI.COMM_WORLD

    domain = domains.Colella(Lx=2.0, Ly=3.0, alpha=0.1, Lz=4.0)

    grid = TensorProductGrid(num_elements=[num_elements] * 3)
    derham_opts = DerhamOptions(degree=[degree] * 3, bcs=(None, None, None))
    derham = Derham(grid, derham_opts, comm=comm)

    domain_array = derham.domain_array
    nprocs = derham.domain_decomposition.nprocs
    domain_decomp = (domain_array, nprocs)

    loading_params = LoadingParameters(ppc=ppc, seed=1234, moments=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0), spatial="uniform")

    particles = Particles6D(
        comm_world=comm,
        domain_decomp=domain_decomp,
        loading_params=loading_params,
    )
    particles.draw_markers()
    comm.Barrier()
    particles.mpi_sort_markers()
    comm.Barrier()

    _, b2_eq_psy = create_equal_random_arrays(derham.V2fem, seed=2345, flattened=True)
    _, b2_psy = create_equal_random_arrays(derham.V2fem, seed=3456, flattened=True)

    pusher_psy = Pusher_psy(
        particles,
        PyccelKernel(pusher_kernels.push_vxb_analytic),
        (
            derham.args_derham,
            b2_eq_psy[0]._data + b2_psy[0]._data,
            b2_eq_psy[1]._data + b2_psy[1]._data,
            b2_eq_psy[2]._data + b2_psy[2]._data,
        ),
        domain.args_domain,
        alpha_in_kernel=1.0,
    )

    dt = 0.1

    # warm-up call (first call may involve additional one-off setup cost)
    pusher_psy(dt)

    t0 = time.perf_counter()
    for _ in range(N_REPS):
        pusher_psy(dt)
    t1 = time.perf_counter()

    return (t1 - t0) / N_REPS


if __name__ == "__main__":
    num_elements = int(sys.argv[1])
    degree = int(sys.argv[2])
    ppc = int(sys.argv[3])

    avg_time = main(num_elements, degree, ppc)
    print(avg_time)
