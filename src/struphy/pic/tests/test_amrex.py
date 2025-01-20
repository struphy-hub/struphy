import pytest


@pytest.mark.mpi
def test_amrex(plot=False, verbose=False):
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
    amrex_particles = Particles6D(name=name,
                                  Np=Np,
                                  bc=bc,
                                  loading=loading,
                                  domain=domain,
                                  loading_params=loading_params,
                                  amrex=amrex)
    struphy_particles = Particles6D(name=name,
                                    Np=Np,
                                    bc=bc,
                                    loading=loading,
                                    domain=domain,
                                    loading_params=loading_params)

    amrex_particles.draw_markers()
    struphy_particles.draw_markers()

    amrex_positions = amrex_particles.positions
    struphy_positions = struphy_particles.positions

    assert type(amrex_positions) == type(struphy_positions)
    assert amrex_positions.dtype == struphy_positions.dtype
    assert amrex_positions.shape == struphy_positions.shape
    assert amrex_positions.size == struphy_positions.size

    # positions on the physical domain Omega
    struphy_pushed_pos = domain(struphy_positions).T
    amrex_pushed_pos = domain(amrex_positions).T

    if plot:
        from matplotlib import pyplot as plt

        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

        fig = plt.figure()
        ax = fig.gca()

        for i, pos in enumerate(struphy_pushed_pos):
            ax.scatter(pos[0], pos[1], c=colors[i % 4])
            ax.arrow(pos[0], pos[1], struphy_particles.velocities[i, 0],
                     struphy_particles.velocities[i, 1], color=colors[i % 4], head_width=.2)

        ax.plot([l1, l1], [l2, r2], 'k')
        ax.plot([r1, r1], [l2, r2], 'k')
        ax.plot([l1, r1], [l2, l2], 'k')
        ax.plot([l1, r1], [r2, r2], 'k')
        ax.set_xlim(-6.5, 6.5)
        ax.set_ylim(-9, 9)
        ax.set_title('Initial conditions (Struphy)')
        plt.show()

        fig = plt.figure()
        ax = fig.gca()

        for i, pos in enumerate(amrex_pushed_pos):
            ax.scatter(pos[0], pos[1], c=colors[i % 4])
            ax.arrow(pos[0], pos[1], amrex_particles.velocities[i, 0],
                     amrex_particles.velocities[i, 1], color=colors[i % 4], head_width=.2)

        ax.plot([l1, l1], [l2, r2], 'k')
        ax.plot([r1, r1], [l2, r2], 'k')
        ax.plot([l1, r1], [l2, l2], 'k')
        ax.plot([l1, r1], [r2, r2], 'k')
        ax.set_xlim(-6.5, 6.5)
        ax.set_ylim(-9, 9)
        ax.set_title('Initial conditions (Amrex)')
        plt.show()

    if verbose:
        print("Struphy positions\n", struphy_particles.positions)
        print("Amrex positions\n", amrex_particles.positions)

        print("Struphy velocities\n", struphy_particles.velocities)
        print("Amrex velocities\n", amrex_particles.velocities)

    # finalize amrex
    amrex.finalize()


if __name__ == "__main__":
    test_amrex(True, True)
