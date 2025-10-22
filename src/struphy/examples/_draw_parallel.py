from psydac.ddm.mpi import mpi as MPI

from struphy.feec.psydac_derham import Derham
from struphy.geometry import domains
from struphy.pic.particles import Particles6D
from struphy.utils.arrays import xp


def main():
    """
    TODO
    """
    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()
    rank = comm.Get_rank()

    # parameters
    Nel = [8, 16, 4]
    p = [2, 2, 2]
    spl_kind = [False, True, True]

    loading_type = "pseudo_random"
    loading_params = {
        "type": loading_type,
        "seed": 1234,
        "moments": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    }

    # create domain
    dom_type = "HollowTorus"
    domain_class = getattr(domains, dom_type)
    domain = domain_class(sfl=True)

    # create de rham object
    derham = Derham(Nel, p, spl_kind, comm=comm)

    if rank == 0:
        print()
        print("Domain decomposition according to : ")
        print(derham.domain_array)

    # create particles
    particles = Particles6D(
        ppc=10,
        bc=["periodic", "periodic", "periodic"],
        loading_params=loading_params,
    )

    comm.Barrier()
    print("Number of particles w/wo holes on each process before sorting : ")
    print("Rank", rank, ":", particles.n_mks_loc, particles.markers.shape[0])

    domain.show(
        grid_info=derham.domain_array,
        markers=particles.markers_wo_holes,
    )

    # sort particles according to domain decomposition
    comm.Barrier()
    particles.mpi_sort_markers()

    comm.Barrier()
    print("Number of particles w/wo holes on each process after sorting : ")
    print("Rank", rank, ":", particles.n_mks_loc, particles.markers.shape[0])

    domain.show(
        grid_info=derham.domain_array,
        markers=particles.markers_wo_holes,
    )

    # are all markers in the correct domain?
    conds = xp.logical_and(
        particles.markers[:, :3] > derham.domain_array[rank, 0::3],
        particles.markers[:, :3] < derham.domain_array[rank, 1::3],
    )

    holes = particles.markers[:, 0] == -1.0
    stay = xp.all(conds, axis=1)

    error_mks = particles.markers[xp.logical_and(~stay, ~holes)]

    print(
        f"rank {rank} | markers not on correct process: {xp.nonzero(xp.logical_and(~stay, ~holes))} \
            \n corresponding positions:\n {error_mks[:, :3]}"
    )

    assert error_mks.size == 0


if __name__ == "__main__":
    main()
