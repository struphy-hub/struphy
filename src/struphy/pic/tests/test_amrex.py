import math

import numpy as np
import pytest
from matplotlib import pyplot as plt

from struphy.feec.psydac_derham import Derham
from struphy.fields_background.generic import GenericCartesianFluidEquilibrium
from struphy.geometry.domains import Cuboid, HollowCylinder
from struphy.pic.amrex import Amrex
from struphy.pic.particles import Particles6D, ParticlesSPH
from struphy.propagators.propagators_markers import PushEta, PushVinEfield

try:
    import amrex.space3d as amr
except ImportError:
    amr = None

Np = 4
seed = 123456


@pytest.mark.skipif(amr == None, reason="pyAMReX is not installed")
def test_amrex_push_v_in_e_field(plot=False, verbose=False):
    # initialize Amrex
    amrex = Amrex()

    # define domain
    l1 = -0.5
    r1 = 0.5
    l2 = -0.5
    r2 = 0.5
    l3 = 0.0
    r3 = 1.0
    domain = Cuboid(l1=l1, r1=r1, l2=l2, r2=r2, l3=l3, r3=r3)

    # define the initial flow
    # Cartesian velocity in physical space. Must return the components as a tuple.
    def u_fun(x, y, z):
        ux = -np.cos(np.pi * x) * np.sin(np.pi * y)
        uy = np.sin(np.pi * x) * np.cos(np.pi * y)
        uz = 0 * x
        return ux, uy, uz

    # Equilibrium pressure in physical space.
    p_fun = lambda x, y, z: 0.5 * (np.sin(np.pi * x) ** 2 + np.sin(np.pi * y) ** 2)
    # Equilibrium number density in physical space.
    n_fun = lambda x, y, z: 1.0 + 0 * x

    bel_flow = GenericCartesianFluidEquilibrium(u_xyz=u_fun, p_xyz=p_fun, n_xyz=n_fun)
    bel_flow.domain = domain

    p_xyz = bel_flow.p_xyz
    # 0-form pressure on logical cube [0, 1]^3.
    p0 = bel_flow.p0

    # particle boundary conditions
    bc = ["reflect", "reflect", "periodic"]

    # instantiate Particle object (for random drawing of markers)
    Np = 1000

    particles_1_amrex = ParticlesSPH(
        bc=bc,
        domain=domain,
        bckgr_params=bel_flow,
        Np=Np,
        amrex=amrex,
    )
    
    particles_1_struphy = ParticlesSPH(
        bc=bc,
        domain=domain,
        bckgr_params=bel_flow,
        Np=Np,
    )

    particles_1_amrex.draw_markers(sort=False)
    particles_1_struphy.draw_markers(sort=False)

    particles_1_amrex.initialize_weights()
    particles_1_struphy.initialize_weights()

    # pass simulation parameters to Propagator class
    PushEta.domain = domain

    # instantiate Propagator object
    prop_eta_1_amrex = PushEta(particles_1_amrex, algo="forward_euler")
    prop_eta_1_struphy = PushEta(particles_1_struphy, algo="forward_euler")

    Nel = [64, 64, 1]  # Number of grid cells
    p = [3, 3, 1]  # spline degrees
    spl_kind = [False, False, True]  # spline types (clamped vs. periodic)

    derham = Derham(Nel, p, spl_kind)

    p_coeffs = derham.P["0"](p0)

    # instantiate Propagator object
    PushVinEfield.domain = domain
    PushVinEfield.derham = derham

    p_h = derham.create_field("pressure", "H1")
    p_h.vector = p_coeffs

    grad_p = derham.grad.dot(p_coeffs)
    grad_p.update_ghost_regions()  # very important, we will move it inside grad
    grad_p *= -1.0
    prop_v_1_amrex = PushVinEfield(particles_1_amrex, e_field=grad_p)
    prop_v_1_struphy = PushVinEfield(particles_1_struphy, e_field=grad_p)

    if plot:
        fig = plt.figure(figsize=(15, 8))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        pos_1 = domain(particles_1_amrex.positions).T
        ax1.scatter(pos_1[:, 0], pos_1[:, 1], pos_1[:, 2])
        ax1.set_title("starting positions Amrex")

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        pos_2 = domain(particles_1_struphy.positions).T
        ax2.scatter(pos_2[:, 0],pos_2[:, 1],pos_2[:, 2])
        ax2.set_title("starting positions Struphy")

        plt.savefig("./push_v_efield_start.jpg")

    # time stepping
    dt = 0.02
    Nt = 200

    # random particles
    pos_1_amrex = np.zeros((Nt + 1, particles_1_amrex.Np, 3), dtype=float)
    velo_1_amrex = np.zeros((Nt + 1, particles_1_amrex.Np, 3), dtype=float)
    energy_1_amrex = np.zeros((Nt + 1, particles_1_amrex.Np), dtype=float)
    
    pos_1_struphy = np.zeros((Nt + 1, particles_1_struphy.Np, 3), dtype=float)
    velo_1_struphy = np.zeros((Nt + 1, particles_1_struphy.Np, 3), dtype=float)
    energy_1_struphy = np.zeros((Nt + 1, particles_1_struphy.Np), dtype=float)

    pos_1_amrex[0] = domain(particles_1_amrex.positions).T
    velo_1_amrex[0] = particles_1_amrex.velocities
    energy_1_amrex[0] = 0.5 * (velo_1_amrex[0, :, 0] ** 2 + velo_1_amrex[0, :, 1] ** 2) + p_h(particles_1_amrex.positions)

    pos_1_struphy[0] = domain(particles_1_struphy.positions).T
    velo_1_struphy[0] = particles_1_struphy.velocities
    energy_1_struphy[0] = 0.5 * (velo_1_struphy[0, :, 0] ** 2 + velo_1_struphy[0, :, 1] ** 2) + p_h(particles_1_struphy.positions)

    time = 0.0
    time_vec = np.zeros(Nt + 1, dtype=float)
    n = 0
    while n < Nt:
        time += dt
        n += 1
        time_vec[n] = time

        # advance in time
        prop_eta_1_amrex(dt / 2)
        prop_v_1_amrex(dt)
        prop_eta_1_amrex(dt / 2)
        
        prop_eta_1_struphy(dt / 2)
        prop_v_1_struphy(dt)
        prop_eta_1_struphy(dt / 2)

        # positions on the physical domain Omega
        pos_1_amrex[n] = domain(particles_1_amrex.positions).T
        velo_1_amrex[n] = particles_1_amrex.velocities
        energy_1_amrex[n] = 0.5 * (velo_1_amrex[n, :, 0] ** 2 + velo_1_amrex[n, :, 1] ** 2) + p_h(particles_1_amrex.positions)

        pos_1_struphy[n] = domain(particles_1_struphy.positions).T
        velo_1_struphy[n] = particles_1_struphy.velocities
        energy_1_struphy[n] = 0.5 * (velo_1_struphy[n, :, 0] ** 2 + velo_1_struphy[n, :, 1] ** 2) + p_h(particles_1_struphy.positions)


    if plot:
        # energy plots (amrex)
        fig = plt.figure(figsize=(13, 6))

        plt.subplot(2, 2, 1)
        plt.plot(time_vec, energy_1_amrex[:, 0])
        plt.title("particle 1")
        plt.xlabel("time")
        plt.ylabel("energy")

        plt.subplot(2, 2, 2)
        plt.plot(time_vec, energy_1_amrex[:, 1])
        plt.title("particle 2")
        plt.xlabel("time")
        plt.ylabel("energy")

        plt.subplot(2, 2, 3)
        plt.plot(time_vec, energy_1_amrex[:, 2])
        plt.title("particle 3")
        plt.xlabel("time")
        plt.ylabel("energy")

        plt.subplot(2, 2, 4)
        plt.plot(time_vec, energy_1_amrex[:, 3])
        plt.title("particle 4")
        plt.xlabel("time")
        plt.ylabel("energy")

        plt.suptitle("Amrex")

        plt.savefig("./energy_amrex.jpg")

        plt.figure(figsize=(12, 28))

        coloring = np.select(
            [pos_1_amrex[0, :, 0] <= -0.2, np.abs(pos_1_amrex[0, :, 0]) < +0.2, pos_1_amrex[0, :, 0] >= 0.2], [-1.0, 0.0, +1.0]
        )

        interval = Nt / 20
        plot_ct = 0
        for i in range(Nt):
            if i % interval == 0:
                print(f"{i = }")
                plot_ct += 1
                plt.subplot(5, 2, plot_ct)
                ax = plt.gca()
                plt.scatter(pos_1_amrex[i, :, 0], pos_1_amrex[i, :, 1], c=coloring)
                plt.axis("square")
                plt.title("n0_scatter")
                plt.xlim(l1, r1)
                plt.ylim(l2, r2)
                plt.colorbar()
                plt.title(f"Gas at t={i * dt}")
            if plot_ct == 10:
                break

        plt.suptitle("Amrex")

        plt.savefig("./position_amrex.jpg")
        
        # energy plots (struphy)
        fig = plt.figure(figsize=(13, 6))

        plt.subplot(2, 2, 1)
        plt.plot(time_vec, energy_1_struphy[:, 0])
        plt.title("particle 1")
        plt.xlabel("time")
        plt.ylabel("energy")

        plt.subplot(2, 2, 2)
        plt.plot(time_vec, energy_1_struphy[:, 1])
        plt.title("particle 2")
        plt.xlabel("time")
        plt.ylabel("energy")

        plt.subplot(2, 2, 3)
        plt.plot(time_vec, energy_1_struphy[:, 2])
        plt.title("particle 3")
        plt.xlabel("time")
        plt.ylabel("energy")

        plt.subplot(2, 2, 4)
        plt.plot(time_vec, energy_1_struphy[:, 3])
        plt.title("particle 4")
        plt.xlabel("time")
        plt.ylabel("energy")
        
        plt.suptitle("Struphy")

        plt.savefig("./energy_struphy.jpg")

        plt.figure(figsize=(12, 28))

        coloring = np.select(
            [pos_1_struphy[0, :, 0] <= -0.2, np.abs(pos_1_struphy[0, :, 0]) < +0.2, pos_1_struphy[0, :, 0] >= 0.2], [-1.0, 0.0, +1.0]
        )

        interval = Nt / 20
        plot_ct = 0
        for i in range(Nt):
            if i % interval == 0:
                print(f"{i = }")
                plot_ct += 1
                plt.subplot(5, 2, plot_ct)
                ax = plt.gca()
                plt.scatter(pos_1_struphy[i, :, 0], pos_1_struphy[i, :, 1], c=coloring)
                plt.axis("square")
                plt.title("n0_scatter")
                plt.xlim(l1, r1)
                plt.ylim(l2, r2)
                plt.colorbar()
                plt.title(f"Gas at t={i * dt}")
            if plot_ct == 10:
                break

        plt.suptitle("Struphy")

        plt.savefig("./position_struphy.jpg")

    amrex.finalize()


@pytest.mark.skipif(amr == None, reason="pyAMReX is not installed")
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


@pytest.mark.skipif(amr == None, reason="pyAMReX is not installed")
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


@pytest.mark.skipif(amr == None, reason="pyAMReX is not installed")
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


@pytest.mark.skipif(amr == None, reason="pyAMReX is not installed")
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


@pytest.mark.skipif(amr == None, reason="pyAMReX is not installed")
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
    loading_params = {"seed": seed}

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
    test_amrex_push_v_in_e_field(plot=True, verbose=True)


# add flat_eval option for jacobians (evaluate metric coef) DONE
# fix reflect bug DONE
# (merge) DONE
# profiling with more cores
# work on GPU with cupy
# transform push_v_with_efield
