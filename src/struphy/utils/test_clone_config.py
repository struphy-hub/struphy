import pytest
from mpi4py import MPI


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("Nel", [[8, 9, 5], [7, 8, 9]])
@pytest.mark.parametrize("Np", [1000, 999])
@pytest.mark.parametrize("num_clones", [1, 2])
def test_clone_config(Nel, Np, num_clones):
    from struphy.utils.clone_config import CloneConfig

    comm = MPI.COMM_WORLD
    species = "ions"
    params = {
        "grid": {
            "Nel": Nel,
        },
        "kinetic": {
            species: {
                "markers": {
                    "Np": Np,
                }
            }
        },
    }

    pconf = CloneConfig(params=params, comm=comm, num_clones=num_clones)
    assert pconf.get_Np_global(species_name=species) == Np
    if Np % num_clones == 0:
        assert pconf.get_Np_clone(Np) == Np / num_clones

    # Print outputs
    pconf.print_clone_config()
    pconf.print_particle_config()
    print(f"{pconf.get_Np_clone(Np) = }")


if __name__ == "__main__":
    test_clone_config([8, 8, 8], 999, 2)
