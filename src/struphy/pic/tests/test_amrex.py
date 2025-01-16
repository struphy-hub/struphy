import pytest


@pytest.mark.mpi
def test_amrex():
    from struphy.geometry.domains import Cuboid

    l1 = -5
    r1 = 5.
    l2 = -7
    r2 = 7.
    l3 = -1.
    r3 = 1.
    domain = Cuboid(l1=l1, r1=r1, l2=l2, r2=r2, l3=l3, r3=r3)

    from struphy.pic.amrex import Amrex

    # initialize amrex
    amrex = Amrex()

    from struphy.pic.particles import Particles6D

    # mandatory parameters
    name = 'test'
    Np = 15
    bc = ['periodic', 'periodic', 'periodic']
    loading = 'pseudo_random'

    # optional
    loading_params = {'seed': None}

    # instantiate Particle object, pass the amrex object
    particles = Particles6D(name=name,
                            Np=Np,
                            bc=bc,
                            loading=loading,
                            domain=domain,
                            loading_params=loading_params,
                            amrex=amrex)

    particles.draw_markers()

    # finalize amrex
    amrex.finalize()


if __name__ == "__main__":
    test_amrex()
