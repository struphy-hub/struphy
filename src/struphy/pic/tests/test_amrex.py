import math

import numpy as np
import pytest
from matplotlib import pyplot as plt

from struphy.geometry.domains import Cuboid, HollowCylinder
from struphy.pic.amrex import Amrex
from struphy.pic.particles import Particles6D
from struphy.propagators.propagators_markers import PushEta

Np = 4

def test_amrex_box(plot=False, verbose=False):
    l1 = -5
    r1 = 5.0
    l2 = -7
    r2 = 7.0
    l3 = -1.0
    r3 = 1.0
    domain = Cuboid(l1=l1, r1=r1, l2=l2, r2=r2, l3=l3, r3=r3)

    # initialize amrex
    amrex = Amrex()

    # mandatory parameters
    name = "test"
    
    bc = ["periodic", "periodic", "periodic"]
    loading = "pseudo_random"

    # optional
    loading_params = {"seed": None}

    # instantiate Particle object, pass the amrex object
    amrex_particles = Particles6D(
        name=name, Np=Np, bc=bc, loading=loading, domain=domain, loading_params=loading_params, amrex=amrex
    )
    struphy_particles = Particles6D(
        name=name, Np=Np, bc=bc, loading=loading, domain=domain, loading_params=loading_params
    )

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
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

        plot_box(
            struphy_pushed_pos,
            struphy_particles.velocities,
            colors,
            l1,
            l2,
            r1,
            r2,
            "Initial conditions (Struphy)",
            "./box_initial_struphy.jpg",
        )

        plot_box(
            amrex_pushed_pos,
            amrex_particles.velocities,
            colors,
            l1,
            l2,
            r1,
            r2,
            "Initial conditions (Amrex)",
            "./box_initial_amrex.jpg",
        )

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
    dt = 0.5
    Nt = int(Tend / dt)

    struphy_pos = np.zeros((Nt, Np, 3), dtype=float)
    amrex_pos = np.zeros((Nt, Np, 3), dtype=float)
    alpha = np.ones(Nt, dtype=float)

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
        plot_box_over_time(
            struphy_pos,
            Np,
            colors,
            alpha,
            l1,
            l2,
            r1,
            r2,
            f"{math.ceil(Tend / dt)} time steps (full color at t=0) (Struphy)",
            "./box_final_struphy.jpg",
        )

        plot_box_over_time(
            amrex_pos,
            Np,
            colors,
            alpha,
            l1,
            l2,
            r1,
            r2,
            f"{math.ceil(Tend / dt)} time steps (full color at t=0) (Amrex)",
            "./box_final_amrex.jpg",
        )

    # finalize amrex
    amrex.finalize()


def test_amrex_cylinder(plot=False, verbose=False):
    a1 = 0.0
    a2 = 5.0
    Lz = 1.0
    domain = HollowCylinder(a1=a1, a2=a2, Lz=Lz)

    # initialize amrex
    amrex = Amrex()

    # instantiate Particle object
    name = "test"
    
    bc = ["periodic", "periodic", "periodic"]
    loading = "pseudo_random"
    loading_params = {"seed": None}

    amrex_particles = Particles6D(
        name=name, Np=Np, bc=bc, loading=loading, domain=domain, loading_params=loading_params, amrex=amrex
    )

    struphy_particles = Particles6D(
        name=name, Np=Np, bc=bc, loading=loading, domain=domain, loading_params=loading_params
    )

    amrex_particles.draw_markers()
    struphy_particles.draw_markers()

    # positions on the physical domain Omega
    amrex_pushed_pos = domain(amrex_particles.positions).T
    struphy_pushed_pos = domain(struphy_particles.positions).T

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    if plot:
        plot_cylinder(
            amrex_pushed_pos,
            amrex_particles.velocities,
            colors,
            a2,
            "Initial conditions (amrex)",
            "./cylinder_initial_amrex.jpg",
        )

        plot_cylinder(
            struphy_pushed_pos,
            struphy_particles.velocities,
            colors,
            a2,
            "Initial conditions (struphy)",
            "./cylinder_initial_struphy.jpg",
        )

    # pass simulation parameters to Propagator class
    PushEta.options(default=True)
    PushEta.domain = domain

    # instantiate Propagator object
    amrex_prop_eta = PushEta(amrex_particles)
    struphy_prop_eta = PushEta(struphy_particles)

    # time stepping
    Tend = 10.0
    dt = 0.5
    Nt = int(Tend / dt)

    amrex_pos = np.zeros((Nt, Np, 3), dtype=float)
    struphy_pos = np.zeros((Nt, Np, 3), dtype=float)
    alpha = np.ones(Nt, dtype=float)

    amrex_pos[0] = amrex_pushed_pos
    struphy_pos[0] = struphy_pushed_pos

    time = 0.0
    n = 0
    while time < (Tend - dt):
        if verbose:
            print("Time:", time)

        time += dt
        n += 1

        # advance in time
        struphy_prop_eta(dt)
        amrex_prop_eta(dt)

        # positions on the physical domain Omega
        amrex_pos[n] = domain(amrex_particles.positions).T
        struphy_pos[n] = domain(struphy_particles.positions).T

        # scaling for plotting
        alpha[n] = (Tend - time) / Tend

    if plot:
        plot_cylinder_over_time(
            amrex_pos,
            Np,
            colors,
            alpha,
            a2,
            f"{math.ceil(Tend / dt)} time steps (full color at t=0) (amrex)",
            "./cylinder_final_amrex.jpg",
        )

        plot_cylinder_over_time(
            struphy_pos,
            Np,
            colors,
            alpha,
            a2,
            f"{math.ceil(Tend / dt)} time steps (full color at t=0) (struphy)",
            "./cylinder_final_struphy.jpg",
        )

    # finalize amrex
    amrex.finalize()


def test_amrex_draw_uniform_cylinder(plot=False, verbose=False):
    a1 = 0.0
    a2 = 5.0
    Lz = 1.0
    domain = HollowCylinder(a1=a1, a2=a2, Lz=Lz)

    # instantiate Particle object
    name = "test"
    Np = 1000
    bc = ["periodic", "periodic", "periodic"]
    loading = "pseudo_random"
    loading_params = {"seed": None}

    struphy_particles = Particles6D(name=name, Np=Np, bc=bc, loading=loading, loading_params=loading_params)

    # instantiate another Particle object
    name = "test_uni"
    loading_params = {"seed": None, "spatial": "disc"}
    struphy_particles_uni = Particles6D(name=name, Np=Np, bc=bc, loading=loading, loading_params=loading_params)

    struphy_particles.draw_markers()
    struphy_particles_uni.draw_markers()

    # positions on the physical domain Omega
    struphy_pushed_pos = domain(struphy_particles.positions).T
    struphy_pushed_pos_uni = domain(struphy_particles_uni.positions).T

    if plot:
        fig = plt.figure(figsize=(10, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(struphy_pushed_pos[:, 0], struphy_pushed_pos[:, 1], s=2.0)
        circle1 = plt.Circle((0, 0), a2, color="k", fill=False)
        ax = plt.gca()
        ax.add_patch(circle1)
        ax.set_aspect("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Draw uniform in logical space")

        plt.subplot(1, 2, 2)
        plt.scatter(struphy_pushed_pos_uni[:, 0], struphy_pushed_pos_uni[:, 1], s=2.0)
        circle2 = plt.Circle((0, 0), a2, color="k", fill=False)
        ax = plt.gca()
        ax.add_patch(circle2)
        ax.set_aspect("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Draw uniform on disc")
        plt.suptitle("Struphy")
        plt.savefig("./uniform_draw_cylinder_struphy.jpg")

    # instantiate Amrex object
    amrex = Amrex()

    # instantiate Particle object
    name = "test"
    Np = 1000
    bc = ["periodic", "periodic", "periodic"]
    loading = "pseudo_random"
    loading_params = {"seed": None}

    amrex_particles = Particles6D(name=name, Np=Np, bc=bc, loading=loading, loading_params=loading_params, amrex=amrex)

    # instantiate another Particle object
    name = "test_uni"
    loading_params = {"seed": None, "spatial": "disc"}
    amrex_particles_uni = Particles6D(
        name=name, Np=Np, bc=bc, loading=loading, loading_params=loading_params, amrex=amrex
    )

    amrex_particles.draw_markers()
    amrex_particles_uni.draw_markers()

    # positions on the physical domain Omega
    amrex_pushed_pos = domain(amrex_particles.positions).T
    amrex_pushed_pos_uni = domain(amrex_particles_uni.positions).T

    if plot:
        fig = plt.figure(figsize=(10, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(amrex_pushed_pos[:, 0], amrex_pushed_pos[:, 1], s=2.0)
        circle1 = plt.Circle((0, 0), a2, color="k", fill=False)
        ax = plt.gca()
        ax.add_patch(circle1)
        ax.set_aspect("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Draw uniform in logical space")

        plt.subplot(1, 2, 2)
        plt.scatter(amrex_pushed_pos_uni[:, 0], amrex_pushed_pos_uni[:, 1], s=2.0)
        circle2 = plt.Circle((0, 0), a2, color="k", fill=False)
        ax = plt.gca()
        ax.add_patch(circle2)
        ax.set_aspect("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Draw uniform on disc")
        plt.suptitle("Amrex")
        plt.savefig("./uniform_draw_cylinder_amrex.jpg")

    # finalize Amrex
    amrex.finalize()


def test_amrex_boundary_conditions_box(plot=False, verbose=False):
    l1 = -5
    r1 = 5.0
    l2 = -7
    r2 = 7.0
    l3 = -1.0
    r3 = 1.0
    domain = Cuboid(l1=l1, r1=r1, l2=l2, r2=r2, l3=l3, r3=r3)

    # simulation parameters
    Tend = 20.0
    dt = 0.5
    
    # initialize amrex
    amrex = Amrex()

    bc = ["reflect", "periodic", "periodic"]

    struphy_particles, amrex_particles = initialize_and_draw_struphy_amrex(domain, Np, bc, amrex)

    struphy_pos, amrex_pos, alpha = push_eta(struphy_particles, amrex_particles, domain, Np, Tend, dt, plot, verbose)

    if plot:
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

        plot_box_over_time(
            struphy_pos, Np, colors, alpha, l1, l2, r1, r2, "Struphy boundary behaviour", "./bc_box_struphy.jpg"
        )

        plot_box_over_time(
            amrex_pos, Np, colors, alpha, l1, l2, r1, r2, "Amrex boundary behaviour", "./bc_box_amrex.jpg"
        )

    amrex.finalize()


def test_amrex_boundary_conditions_cylinder(plot=False, verbose=False):
    a1 = 0.0
    a2 = 5.0
    Lz = 1.0
    domain = HollowCylinder(a1=a1, a2=a2, Lz=Lz)

    # simulation parameters
    Tend = 20.0
    dt = 0.5
    000

    # initialize amrex
    amrex = Amrex()

    bc = ["reflect", "periodic", "periodic"]

    struphy_particles, amrex_particles = initialize_and_draw_struphy_amrex(domain, Np, bc, amrex)

    struphy_pos, amrex_pos, alpha = push_eta(struphy_particles, amrex_particles, domain, Np, Tend, dt, plot, verbose)

    if plot:
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

        plot_cylinder_over_time(
            struphy_pos, Np, colors, alpha, a2, "Struphy boundary behaviour", "./bc_cylinder_struphy.jpg"
        )

        plot_cylinder_over_time(amrex_pos, Np, colors, alpha, a2, "Amrex boundary behaviour", "./bc_cylinder_amrex.jpg")

    amrex.finalize()


def initialize_and_draw_struphy_amrex(domain, Np, bc, amrex):
    # mandatory parameters
    name = "test"
    loading = "pseudo_random"

    # optional
    loading_params = {"seed": None}

    # instantiate Particle object, pass the amrex object
    amrex_particles = Particles6D(
        name=name, Np=Np, bc=bc, loading=loading, domain=domain, loading_params=loading_params, amrex=amrex
    )
    struphy_particles = Particles6D(
        name=name, Np=Np, bc=bc, loading=loading, domain=domain, loading_params=loading_params
    )

    amrex_particles.draw_markers()
    struphy_particles.draw_markers()

    return struphy_particles, amrex_particles


def plot_box(positions, velocities, colors, l1, l2, r1, r2, title, path):
    fig = plt.figure()
    ax = fig.gca()

    for i, pos in enumerate(positions):
        ax.scatter(pos[0], pos[1], c=colors[i % 4])
        ax.arrow(
            pos[0],
            pos[1],
            velocities[i, 0],
            velocities[i, 1],
            color=colors[i % 4],
            head_width=0.2,
        )

    ax.plot([l1, l1], [l2, r2], "k")
    ax.plot([r1, r1], [l2, r2], "k")
    ax.plot([l1, r1], [l2, l2], "k")
    ax.plot([l1, r1], [r2, r2], "k")
    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-9, 9)
    ax.set_title(title)
    plt.grid()
    plt.savefig(path)


def plot_box_over_time(pos, Np, colors, alpha, l1, l2, r1, r2, title, path):
    fig = plt.figure()
    ax = fig.gca()

    for i in range(Np):
        ax.scatter(pos[:, i, 0], pos[:, i, 1], c=colors[i % 4], alpha=alpha)

    ax.plot([l1, l1], [l2, r2], "k")
    ax.plot([r1, r1], [l2, r2], "k")
    ax.plot([l1, r1], [l2, l2], "k")
    ax.plot([l1, r1], [r2, r2], "k")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-9, 9)
    ax.set_title(title)

    plt.grid()
    plt.savefig(path)


def push_eta(struphy_particles, amrex_particles, domain, Np, Tend, dt, plot, verbose):
    # pass simulation parameters to Propagator class
    PushEta.domain = domain

    # instantiate Propagator object
    struphy_prop_eta = PushEta(struphy_particles)
    amrex_prop_eta = PushEta(amrex_particles)

    if plot:
        # time stepping
        Nt = int(Tend / dt)

        struphy_pos = np.zeros((Nt, Np, 3), dtype=float)
        amrex_pos = np.zeros((Nt, Np, 3), dtype=float)
        alpha = np.ones(Nt, dtype=float)

        amrex_positions = amrex_particles.positions
        struphy_positions = struphy_particles.positions

        # positions on the physical domain Omega
        struphy_pushed_pos = domain(struphy_positions).T
        amrex_pushed_pos = domain(amrex_positions).T

        struphy_pos[0] = struphy_pushed_pos
        amrex_pos[0] = amrex_pushed_pos
    else: 
        struphy_pos = None
        amrex_pos = None
        alpha = None
        
    n = 0
    time = 0.0
    while time < (Tend - dt):
        time += dt
        n += 1

        # advance in time
        struphy_prop_eta(dt)
        amrex_prop_eta(dt)

        if plot:
            # positions on the physical domain Omega
            struphy_pos[n] = domain(struphy_particles.positions).T
            amrex_pos[n] = domain(amrex_particles.positions).T

            # scaling for plotting
            alpha[n] = (Tend - time) / Tend
        if verbose:
            print("Time: ", time)

    return struphy_pos, amrex_pos, alpha


def plot_cylinder_over_time(pos, Np, colors, alpha, a2, title, path):
    fig = plt.figure()
    ax = fig.gca()

    # make scatter plot for each particle in xy-plane
    for i in range(Np):
        ax.scatter(pos[:, i, 0], pos[:, i, 1], c=colors[i % 4], alpha=alpha)

    circle1 = plt.Circle((0, 0), a2, color="k", fill=False)

    ax.add_patch(circle1)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    plt.grid()
    plt.savefig(path)


def plot_cylinder(positions, velocities, colors, a2, title, path):
    fig = plt.figure()
    ax = fig.gca()

    for n, pos in enumerate(positions):
        ax.scatter(pos[0], pos[1], c=colors[n % 4])
        ax.arrow(
            pos[0],
            pos[1],
            velocities[n, 0],
            velocities[n, 1],
            color=colors[n % 4],
            head_width=0.2,
        )

    circle1 = plt.Circle((0, 0), a2, color="k", fill=False)

    ax.add_patch(circle1)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    plt.grid()
    plt.savefig(path)


if __name__ == "__main__":
    test_amrex_boundary_conditions_box(plot=True, verbose=True)
    test_amrex_boundary_conditions_cylinder(plot=True, verbose=True)
    test_amrex_box(plot=True, verbose=True)
    test_amrex_cylinder(plot=True, verbose=True)
    test_amrex_draw_uniform_cylinder(plot=True, verbose=True)
