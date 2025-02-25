import pytest
from mpi4py import MPI

@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("Nel", [[8, 9, 5], [7, 8, 9]])
@pytest.mark.parametrize("Np", [1000, 999])
@pytest.mark.parametrize("num_clones", [1,2])
def test_pconf(Nel, Np,num_clones):
    from struphy.utils.parallel_config import ParallelConfig

    comm = MPI.COMM_WORLD
    species = "ions"
    params = {
        "grid": {
            "Nel": Nel,
        },
        "kinetic":{
            species:{
                "markers":{
                    "Np":Np,
                }
            }
        }
    }

    pconf = ParallelConfig(params=params, comm=comm,num_clones=num_clones)

    assert pconf.get_global_Np(species) == Np
    if Np % num_clones == 0:
        assert pconf.get_clone_Np(species) == Np / num_clones
    
    # Print outputs
    pconf.print_clone_config()
    pconf.print_particle_config()
    print(pconf.get_clone_Np(species))
    print(pconf.get_clone_ppc(species))
    print(pconf.get_global_Np(species))
    print(pconf.get_global_ppc(species))


if __name__ == "__main__":
    test_pconf([8,8,8], 999, 1)