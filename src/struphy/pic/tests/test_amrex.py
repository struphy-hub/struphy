import pytest
from matplotlib import pyplot as plt
from struphy.geometry.domains import Cuboid, HollowCylinder
from struphy.pic.amrex import Amrex
from struphy.pic.particles import Particles6D
from struphy.propagators.propagators_markers import PushEta
import math
import numpy as np


@pytest.mark.mpi
def test_amrex_box(plot=False, verbose=False):

    l1 = -5
    r1 = 5.
    l2 = -7
    r2 = 7.
    l3 = -1.
    r3 = 1.
    domain = Cuboid(l1=l1, r1=r1, l2=l2, r2=r2, l3=l3, r3=r3)

    # initialize amrex
    amrex = Amrex()

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
        plt.savefig("./initial_struphy.jpg")

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
        plt.savefig("./initial_amrex.jpg")

    if verbose:
        print("Struphy positions\n", struphy_particles.positions)
        print("Amrex positions\n", amrex_particles.positions)

        print("Struphy velocities\n", struphy_particles.velocities)
        print("Amrex velocities\n", amrex_particles.velocities)

    # default parameters of Propagator
    opts_eta = PushEta.options(default=True)
    if verbose:
        print(opts_eta)

    # pass simulation parameters to Propagator class
    PushEta.domain = domain

    # instantiate Propagator object
    struphy_prop_eta = PushEta(struphy_particles)
    amrex_prop_eta = PushEta(amrex_particles)

    # time stepping
    Tend = 10.0
    dt = 0.2
    Nt = int(Tend / dt)

    struphy_pos = np.zeros((Nt + 1, Np, 3), dtype=float)
    amrex_pos = np.zeros((Nt + 1, Np, 3), dtype=float)
    alpha = np.ones(Nt + 1, dtype=float)

    struphy_pos[0] = struphy_pushed_pos
    amrex_pos[0] = amrex_pushed_pos

    time = 0.0
    n = 0
    while time < (Tend - dt):
        time += dt
        n += 1

        # advance in time
        struphy_prop_eta(dt)
        amrex_prop_eta(dt)

        # positions on the physical domain Omega
        struphy_pos[n] = domain(struphy_particles.positions).T
        amrex_pos[n] = domain(amrex_particles.positions).T

        # scaling for plotting
        alpha[n] = (Tend - time) / Tend

        if verbose:
            print("Time:", time)

    if plot:
        fig = plt.figure()
        ax = fig.gca()

        for i in range(Np):
            ax.scatter(struphy_pos[:, i, 0], struphy_pos[:, i, 1], c=colors[i % 4], alpha=alpha)

        ax.plot([l1, l1], [l2, r2], 'k')
        ax.plot([r1, r1], [l2, r2], 'k')
        ax.plot([l1, r1], [l2, l2], 'k')
        ax.plot([l1, r1], [r2, r2], 'k')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-6.5, 6.5)
        ax.set_ylim(-9, 9)
        ax.set_title(f'{math.ceil(Tend/dt)} time steps (full color at t=0) (Struphy)')

        plt.savefig("./timesteps_struphy.jpg")

        fig = plt.figure()
        ax = fig.gca()

        for i in range(Np):
            ax.scatter(amrex_pos[:, i, 0], amrex_pos[:, i, 1], c=colors[i % 4], alpha=alpha)

        ax.plot([l1, l1], [l2, r2], 'k')
        ax.plot([r1, r1], [l2, r2], 'k')
        ax.plot([l1, r1], [l2, l2], 'k')
        ax.plot([l1, r1], [r2, r2], 'k')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-6.5, 6.5)
        ax.set_ylim(-9, 9)
        ax.set_title(f'{math.ceil(Tend/dt)} time steps (full color at t=0) (Amrex)')

        plt.savefig("./timesteps_amrex.jpg")

    # finalize amrex
    amrex.finalize()


@pytest.mark.mpi
def test_amrex_cylinder(plot=False, verbose=False):

    a1 = 0.
    a2 = 5.
    Lz = 1.
    domain = HollowCylinder(a1=a1, a2=a2, Lz=Lz)

    # initialize amrex
    amrex = Amrex()
    
    # instantiate Particle object
    name = 'test'
    Np = 15
    bc = ['periodic', 'periodic', 'periodic']
    loading = 'pseudo_random'
    loading_params = {'seed': None}

    particles = Particles6D(name=name,
                            Np=Np,
                            bc=bc,
                            loading=loading,
                            loading_params=loading_params,
                            amrex=amrex)

    # instantiate another Particle object
    name = 'test_uni'
    loading_params = {'seed': None, 'spatial': 'disc'}
    particles_uni = Particles6D(name=name,
                                Np=Np,
                                bc=bc,
                                loading=loading,
                                loading_params=loading_params,
                                amrex=amrex)

    particles.draw_markers()
    particles_uni.draw_markers()

    # positions on the physical domain Omega
    pushed_pos = domain(particles.positions).T
    pushed_pos_uni = domain(particles_uni.positions).T

    if plot:
        fig = plt.figure(figsize=(10, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(pushed_pos[:, 0], pushed_pos[:, 1], s=2.)
        circle1 = plt.Circle((0, 0), a2, color='k', fill=False)
        ax = plt.gca()
        ax.add_patch(circle1)
        ax.set_aspect('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Draw uniform in logical space')

        plt.subplot(1, 2, 2)
        plt.scatter(pushed_pos_uni[:, 0], pushed_pos_uni[:, 1], s=2.)
        circle2 = plt.Circle((0, 0), a2, color='k', fill=False)
        ax = plt.gca()
        ax.add_patch(circle2)
        ax.set_aspect('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Draw uniform on disc')

        plt.savefig("./disc_uniform_amrex.jpg")

    # instantiate Particle object
    name = 'test'
    Np = 15
    bc = ['periodic', 'periodic', 'periodic']
    loading = 'pseudo_random'
    loading_params = {'seed': None}

    particles = Particles6D(name=name,
                            Np=Np,
                            bc=bc,
                            loading=loading,
                            domain=domain,
                            loading_params=loading_params,
                            amrex=amrex)

    particles.draw_markers()

    # positions on the physical domain Omega
    pushed_pos = domain(particles.positions).T

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    if plot:
        fig = plt.figure()
        ax = fig.gca()

        for n, pos in enumerate(pushed_pos):
            ax.scatter(pos[0], pos[1], c=colors[n % 4])
            ax.arrow(pos[0], pos[1], particles.velocities[n, 0],
                     particles.velocities[n, 1], color=colors[n % 4], head_width=.2)

        circle1 = plt.Circle((0, 0), a2, color='k', fill=False)

        ax.add_patch(circle1)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Initial conditions')
        plt.savefig("./initial_cylinder_amrex.jpg")

    # pass simulation parameters to Propagator class
    PushEta.options(default=True)
    PushEta.domain = domain

    # instantiate Propagator object
    prop_eta = PushEta(particles)

    # time stepping
    Tend = 10.
    dt = .2
    Nt = int(Tend / dt)

    pos = np.zeros((Nt + 1, Np, 3), dtype=float)
    alpha = np.ones(Nt + 1, dtype=float)

    pos[0] = pushed_pos

    time = 0.
    n = 0
    while time < (Tend - dt):
        if verbose:
            print("Time:", time)

        time += dt
        n += 1

        # advance in time
        prop_eta(dt)

        # positions on the physical domain Omega
        pos[n] = domain(particles.positions).T

        # scaling for plotting
        alpha[n] = (Tend - time)/Tend

    if plot:
        # make scatter plot for each particle in xy-plane
        for i in range(Np):
            ax.scatter(pos[:, i, 0], pos[:, i, 1], c=colors[i % 4], alpha=alpha)

        circle1 = plt.Circle((0, 0), a2, color='k', fill=False)

        ax.add_patch(circle1)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{math.ceil(Tend/dt)} time steps (full color at t=0)')

        plt.savefig("./final_cylinder_amrex.jpg")

    # finalize amrex
    amrex.finalize()


if __name__ == "__main__":
    test_amrex_cylinder(plot=True, verbose=True)
