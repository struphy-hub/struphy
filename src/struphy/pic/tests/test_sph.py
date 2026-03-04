import cunumpy as xp
import pytest
from feectools.ddm.mpi import MockComm
from feectools.ddm.mpi import mpi as MPI
from matplotlib import pyplot as plt

from struphy import (
    BinningPlot,
    BoundaryParameters,
    LoadingParameters,
    WeightsParameters,
    domains,
    equils,
    perturbations,
)
from struphy.fields_background.equils import ConstantVelocity
from struphy.geometry.base import Domain
from struphy.pic.particles import ParticlesSPH


@pytest.mark.parametrize("boxes_per_dim", [(24, 1, 1)])
@pytest.mark.parametrize("kernel", ["trigonometric_1d", "gaussian_1d", "linear_1d"])
@pytest.mark.parametrize("derivative", [0, 1])
@pytest.mark.parametrize("bc_x", ["periodic", "mirror", "fixed"])
@pytest.mark.parametrize("eval_pts", [11, 16])
@pytest.mark.parametrize("tesselation", [False, True])
def test_sph_evaluation_1d(
    boxes_per_dim,
    kernel,
    derivative,
    bc_x,
    eval_pts,
    tesselation,
    show_plot=False,
):
    if isinstance(MPI.COMM_WORLD, MockComm):
        comm = None
        rank = 0
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

    # DOMAIN object
    dom_type = "Cuboid"
    dom_params = {"l1": 1.0, "r1": 2.0, "l2": 10.0, "r2": 20.0, "l3": 100.0, "r3": 200.0}
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    if tesselation:
        if kernel == "trigonometric_1d" and derivative == 1:
            ppb = 100
        else:
            ppb = 4
        loading_params = LoadingParameters(ppb=ppb, seed=1607, loading="tesselation")
    else:
        if derivative == 0:
            ppb = 1000
        else:
            ppb = 20000
        loading_params = LoadingParameters(ppb=ppb, seed=223)

    # background
    background = ConstantVelocity(n=1.5, density_profile="constant")
    background.domain = domain

    pert = {"n": perturbations.ModesCos(ls=(1,), amps=(1e-0,))}

    if derivative == 0:
        fun_exact = lambda e1, e2, e3: 1.5 + xp.cos(2 * xp.pi * e1)
    else:
        fun_exact = lambda e1, e2, e3: -2 * xp.pi * xp.sin(2 * xp.pi * e1)

    boundary_params = BoundaryParameters(bc_sph=(bc_x, "periodic", "periodic"))

    particles = ParticlesSPH(
        comm_world=comm,
        loading_params=loading_params,
        boundary_params=boundary_params,
        boxes_per_dim=boxes_per_dim,
        bufsize=1.0,
        domain=domain,
        background=background,
        perturbations=pert,
        n_as_volume_form=True,
    )

    # eval points
    eta1 = xp.linspace(0, 1.0, eval_pts)
    eta2 = xp.array([0.0])
    eta3 = xp.array([0.0])

    particles.draw_markers(sort=False, verbose=False)
    if comm is not None:
        particles.mpi_sort_markers()
    particles.initialize_weights()
    h1 = 1 / boxes_per_dim[0]
    h2 = 1 / boxes_per_dim[1]
    h3 = 1 / boxes_per_dim[2]
    ee1, ee2, ee3 = xp.meshgrid(eta1, eta2, eta3, indexing="ij")
    test_eval = particles.eval_density(
        ee1,
        ee2,
        ee3,
        h1=h1,
        h2=h2,
        h3=h3,
        kernel_type=kernel,
        derivative=derivative,
    )

    if comm is None:
        all_eval = test_eval
    else:
        all_eval = xp.zeros_like(test_eval)
        comm.Allreduce(test_eval, all_eval, op=MPI.SUM)

    exact_eval = fun_exact(ee1, ee2, ee3)
    err_max_norm = xp.max(xp.abs(all_eval - exact_eval)) / xp.max(xp.abs(exact_eval))

    if rank == 0:
        print(f"\n{boxes_per_dim =}")
        print(f"{kernel =}, {derivative =}")
        print(f"{bc_x =}, {eval_pts =}, {tesselation =}, {err_max_norm =}")
        if show_plot:
            plt.figure(figsize=(12, 8))
            plt.plot(ee1.squeeze(), fun_exact(ee1, ee2, ee3).squeeze(), label="exact")
            plt.plot(ee1.squeeze(), all_eval.squeeze(), "--.", label="eval_sph")
            plt.xlabel("e1")
            plt.legend()
            plt.show()

    if tesselation:
        if derivative == 0:
            assert err_max_norm < 0.0081
        else:
            assert err_max_norm < 0.027
    else:
        if derivative == 0:
            assert err_max_norm < 0.05
        else:
            assert err_max_norm < 0.37


@pytest.mark.parametrize("boxes_per_dim", [(12, 12, 1)])
@pytest.mark.parametrize("kernel", ["trigonometric_2d", "gaussian_2d", "linear_2d"])
@pytest.mark.parametrize("derivative", [0, 1, 2])
@pytest.mark.parametrize("bc_x", ["periodic", "mirror", "fixed"])
@pytest.mark.parametrize("bc_y", ["periodic", "mirror", "fixed"])
@pytest.mark.parametrize("eval_pts", [11, 16])
def test_sph_evaluation_2d(
    boxes_per_dim,
    kernel,
    derivative,
    bc_x,
    bc_y,
    eval_pts,
    show_plot=False,
):
    if isinstance(MPI.COMM_WORLD, MockComm):
        comm = None
        rank = 0
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

    tesselation = True

    # DOMAIN object
    dom_type = "Cuboid"
    dom_params = {"l1": 1.0, "r1": 2.0, "l2": 0.0, "r2": 2.0, "l3": 100.0, "r3": 200.0}
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    if kernel == "trigonometric_2d" and derivative != 0:
        ppb = 100
    else:
        ppb = 16

    loading_params = LoadingParameters(ppb=ppb, loading="tesselation")

    # background
    background = ConstantVelocity(n=1.5, density_profile="constant")
    background.domain = domain

    pert = {"n": perturbations.ModesCosCos(ls=(1,), ms=(1,), amps=(1e-0,))}

    if derivative == 0:
        fun_exact = lambda e1, e2, e3: 1.5 + xp.cos(2 * xp.pi * e1) * xp.cos(2 * xp.pi * e2)
    elif derivative == 1:
        fun_exact = lambda e1, e2, e3: -2 * xp.pi * xp.sin(2 * xp.pi * e1) * xp.cos(2 * xp.pi * e2)
    else:
        fun_exact = lambda e1, e2, e3: -2 * xp.pi * xp.cos(2 * xp.pi * e1) * xp.sin(2 * xp.pi * e2)

    # boundary conditions
    boundary_params = BoundaryParameters(bc_sph=(bc_x, bc_y, "periodic"))

    # eval points
    eta1 = xp.linspace(0, 1.0, eval_pts)
    eta2 = xp.linspace(0, 1.0, eval_pts)
    eta3 = xp.array([0.0])

    # particles object
    particles = ParticlesSPH(
        comm_world=comm,
        loading_params=loading_params,
        boundary_params=boundary_params,
        boxes_per_dim=boxes_per_dim,
        bufsize=1.0,
        domain=domain,
        background=background,
        perturbations=pert,
        n_as_volume_form=True,
        verbose=False,
    )

    particles.draw_markers(sort=False, verbose=False)
    if comm is not None:
        particles.mpi_sort_markers()
    particles.initialize_weights()
    h1 = 1 / boxes_per_dim[0]
    h2 = 1 / boxes_per_dim[1]
    h3 = 1 / boxes_per_dim[2]
    ee1, ee2, ee3 = xp.meshgrid(eta1, eta2, eta3, indexing="ij")
    test_eval = particles.eval_density(
        ee1,
        ee2,
        ee3,
        h1=h1,
        h2=h2,
        h3=h3,
        kernel_type=kernel,
        derivative=derivative,
    )

    if comm is None:
        all_eval = test_eval
    else:
        all_eval = xp.zeros_like(test_eval)
        comm.Allreduce(test_eval, all_eval, op=MPI.SUM)

    exact_eval = fun_exact(ee1, ee2, ee3)
    err_max_norm = xp.max(xp.abs(all_eval - exact_eval)) / xp.max(xp.abs(exact_eval))

    if rank == 0:
        print(f"\n{boxes_per_dim =}")
        print(f"{kernel =}, {derivative =}")
        print(f"{bc_x =}, {bc_y =}, {eval_pts =}, {tesselation =}, {err_max_norm =}")
        if show_plot:
            plt.figure(figsize=(12, 24))
            plt.subplot(2, 1, 1)
            plt.pcolor(ee1.squeeze(), ee2.squeeze(), fun_exact(ee1, ee2, ee3).squeeze())
            plt.title("exact")
            plt.subplot(2, 1, 2)
            plt.pcolor(ee1.squeeze(), ee2.squeeze(), all_eval.squeeze())
            plt.title("sph eval")
            plt.xlabel("e1")
            plt.xlabel("e2")
            plt.show()

    if derivative == 0:
        assert err_max_norm < 0.031
    else:
        assert err_max_norm < 0.069


@pytest.mark.parametrize("boxes_per_dim", [(12, 8, 8)])
@pytest.mark.parametrize("kernel", ["trigonometric_3d", "gaussian_3d", "linear_3d", "linear_isotropic_3d"])
@pytest.mark.parametrize("derivative", [0, 3])
@pytest.mark.parametrize("bc_x", ["periodic"])
@pytest.mark.parametrize("bc_y", ["periodic"])
@pytest.mark.parametrize("bc_z", ["periodic", "mirror", "fixed"])
@pytest.mark.parametrize("eval_pts", [11])
def test_sph_evaluation_3d(
    boxes_per_dim,
    kernel,
    derivative,
    bc_x,
    bc_y,
    bc_z,
    eval_pts,
    show_plot=False,
):
    if isinstance(MPI.COMM_WORLD, MockComm):
        comm = None
        rank = 0
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

    tesselation = True

    # DOMAIN object
    dom_type = "Cuboid"
    dom_params = {"l1": 1.0, "r1": 2.0, "l2": 0.0, "r2": 2.0, "l3": -1.0, "r3": 2.0}
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    if kernel in ("trigonometric_3d", "linear_isotropic_3d") and derivative != 0:
        ppb = 100
    else:
        ppb = 64

    loading_params = LoadingParameters(ppb=ppb, loading="tesselation")

    # background
    background = ConstantVelocity(n=1.5, density_profile="constant")
    background.domain = domain

    if derivative == 0:
        fun_exact = lambda e1, e2, e3: 1.5 + 0.0 * e1
    else:
        fun_exact = lambda e1, e2, e3: 0.0 * e1

    # boundary conditions
    boundary_params = BoundaryParameters(bc_sph=(bc_x, bc_y, bc_z))

    # eval points
    eta1 = xp.linspace(0, 1.0, eval_pts)
    eta2 = xp.linspace(0, 1.0, eval_pts)
    eta3 = xp.linspace(0, 1.0, eval_pts)

    # particles object
    particles = ParticlesSPH(
        comm_world=comm,
        loading_params=loading_params,
        boundary_params=boundary_params,
        boxes_per_dim=boxes_per_dim,
        bufsize=2.0,
        domain=domain,
        background=background,
        n_as_volume_form=True,
        verbose=False,
    )

    particles.draw_markers(sort=False, verbose=False)
    if comm is not None:
        particles.mpi_sort_markers()
    particles.initialize_weights()
    h1 = 1 / boxes_per_dim[0]
    h2 = 1 / boxes_per_dim[1]
    h3 = 1 / boxes_per_dim[2]
    ee1, ee2, ee3 = xp.meshgrid(eta1, eta2, eta3, indexing="ij")
    test_eval = particles.eval_density(
        ee1,
        ee2,
        ee3,
        h1=h1,
        h2=h2,
        h3=h3,
        kernel_type=kernel,
        derivative=derivative,
    )

    if comm is None:
        all_eval = test_eval
    else:
        all_eval = xp.zeros_like(test_eval)
        comm.Allreduce(test_eval, all_eval, op=MPI.SUM)

    exact_eval = fun_exact(ee1, ee2, ee3)
    err_max_norm = xp.max(xp.abs(all_eval - exact_eval))

    if rank == 0:
        print(f"\n{boxes_per_dim =}")
        print(f"{kernel =}, {derivative =}")
        print(f"{bc_x =}, {bc_y =}, {bc_z =}, {eval_pts =}, {tesselation =}, {err_max_norm =}")
        if show_plot:
            print(f"\n{fun_exact(ee1, ee2, ee3)[5, 5, 5] =}")
            print(f"{ee1[5, 5, 5] =}, {ee2[5, 5, 5] =}, {ee3[5, 5, 5] =}")
            print(f"{all_eval[5, 5, 5] =}")

            print(f"\n{ee1[4, 4, 4] =}, {ee2[4, 4, 4] =}, {ee3[4, 4, 4] =}")
            print(f"{all_eval[4, 4, 4] =}")

            print(f"\n{ee1[3, 3, 3] =}, {ee2[3, 3, 3] =}, {ee3[3, 3, 3] =}")
            print(f"{all_eval[3, 3, 3] =}")

            print(f"\n{ee1[2, 2, 2] =}, {ee2[2, 2, 2] =}, {ee3[2, 2, 2] =}")
            print(f"{all_eval[2, 2, 2] =}")

            print(f"\n{ee1[1, 1, 1] =}, {ee2[1, 1, 1] =}, {ee3[1, 1, 1] =}")
            print(f"{all_eval[1, 1, 1] =}")

            print(f"\n{ee1[0, 0, 0] =}, {ee2[0, 0, 0] =}, {ee3[0, 0, 0] =}")
            print(f"{all_eval[0, 0, 0] =}")
            # plt.figure(figsize=(12, 24))
            # plt.subplot(2, 1, 1)
            # plt.pcolor(ee1[0, :, :], ee2[0, :, :], fun_exact(ee1, ee2, ee3)[0, :, :])
            # plt.title("exact")
            # plt.subplot(2, 1, 2)
            # plt.pcolor(ee1[0, :, :], ee2[0, :, :], all_eval[0, :, :])
            # plt.title("sph eval")
            # plt.xlabel("e1")
            # plt.xlabel("e2")
            # plt.show()

    assert err_max_norm < 0.03


@pytest.mark.parametrize("boxes_per_dim", [(12, 1, 1)])
@pytest.mark.parametrize("bc_x", ["periodic", "mirror", "fixed"])
@pytest.mark.parametrize("eval_pts", [11, 16])
@pytest.mark.parametrize("tesselation", [False, True])
def test_evaluation_SPH_Np_convergence_1d(boxes_per_dim, bc_x, eval_pts, tesselation, show_plot=False):
    if isinstance(MPI.COMM_WORLD, MockComm):
        comm = None
        rank = 0
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

    # DOMAIN object
    dom_type = "Cuboid"
    dom_params = {"l1": 0.0, "r1": 3.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    if tesselation:
        ppbs = [4, 8, 16, 32, 64]
        Nps = [None] * len(ppbs)
    else:
        Nps = [(2**k) * 10**3 for k in range(-2, 9)]
        ppbs = [None] * len(Nps)

    # background
    background = ConstantVelocity(n=1.5, density_profile="constant")
    background.domain = domain

    # perturbation]}
    if bc_x in ("periodic", "fixed"):
        fun_exact = lambda e1, e2, e3: 1.5 - xp.sin(2 * xp.pi * e1)
        pert = {"n": perturbations.ModesSin(ls=(1,), amps=(-1e-0,))}
    elif bc_x == "mirror":
        fun_exact = lambda e1, e2, e3: 1.5 - xp.cos(2 * xp.pi * e1)
        pert = {"n": perturbations.ModesCos(ls=(1,), amps=(-1e-0,))}

    # exact solution
    eta1 = xp.linspace(0, 1.0, eval_pts)  # add offset for non-periodic boundary conditions, TODO: implement Neumann
    eta2 = xp.array([0.0])
    eta3 = xp.array([0.0])
    ee1, ee2, ee3 = xp.meshgrid(eta1, eta2, eta3, indexing="ij")
    exact_eval = fun_exact(ee1, ee2, ee3)

    # boundary conditions
    boundary_params = BoundaryParameters(bc_sph=(bc_x, "periodic", "periodic"))

    # loop
    err_vec = []
    for Np, ppb in zip(Nps, ppbs):
        if tesselation:
            loading_params = LoadingParameters(ppb=ppb, loading="tesselation")
        else:
            loading_params = LoadingParameters(Np=Np, seed=1607)

        particles = ParticlesSPH(
            comm_world=comm,
            loading_params=loading_params,
            boundary_params=boundary_params,
            boxes_per_dim=boxes_per_dim,
            bufsize=1.0,
            domain=domain,
            background=background,
            perturbations=pert,
            n_as_volume_form=True,
            verbose=False,
        )

        particles.draw_markers(sort=False, verbose=False)
        if comm is not None:
            particles.mpi_sort_markers()
        particles.initialize_weights()
        h1 = 1 / boxes_per_dim[0]
        h2 = 1 / boxes_per_dim[1]
        h3 = 1 / boxes_per_dim[2]

        test_eval = particles.eval_density(ee1, ee2, ee3, h1=h1, h2=h2, h3=h3)

        if comm is None:
            all_eval = test_eval
        else:
            all_eval = xp.zeros_like(test_eval)
            comm.Allreduce(test_eval, all_eval, op=MPI.SUM)

        if show_plot and rank == 0:
            plt.figure()
            plt.plot(ee1.squeeze(), exact_eval.squeeze(), label="exact")
            plt.plot(ee1.squeeze(), all_eval.squeeze(), "--.", label="eval_sph")
            plt.title(f"{Np =}, {ppb =}")
            # plt.savefig(f"fun_{Np}_{ppb}.png")

        diff = xp.max(xp.abs(all_eval - exact_eval)) / xp.max(xp.abs(exact_eval))
        err_vec += [diff]
        print(f"{Np =}, {ppb =}, {diff =}")

    if tesselation:
        fit = xp.polyfit(xp.log(ppbs), xp.log(err_vec), 1)
        xvec = ppbs
    else:
        fit = xp.polyfit(xp.log(Nps), xp.log(err_vec), 1)
        xvec = Nps

    if show_plot and rank == 0:
        plt.figure(figsize=(12, 8))
        plt.loglog(xvec, err_vec, label="Convergence")
        plt.loglog(xvec, xp.exp(fit[1]) * xp.array(xvec) ** (fit[0]), "--", label=f"fit with slope {fit[0]}")
        plt.legend()
        plt.show()
        # plt.savefig(f"Convergence_SPH_{tesselation=}")

    if rank == 0:
        print(f"\n{bc_x =}, {eval_pts =}, {tesselation =}, {fit[0] =}")

    if tesselation:
        assert fit[0] < 2e-3
    else:
        assert xp.abs(fit[0] + 0.5) < 0.1  # Monte Carlo rate


@pytest.mark.parametrize("boxes_per_dim", [(12, 1, 1)])
@pytest.mark.parametrize("bc_x", ["periodic", "fixed", "mirror"])
@pytest.mark.parametrize("eval_pts", [11, 16])
@pytest.mark.parametrize("tesselation", [False, True])
def test_evaluation_SPH_h_convergence_1d(boxes_per_dim, bc_x, eval_pts, tesselation, show_plot=False):
    if isinstance(MPI.COMM_WORLD, MockComm):
        comm = None
        rank = 0
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

    # DOMAIN object
    dom_type = "Cuboid"
    dom_params = {"l1": 0.0, "r1": 3.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    if tesselation:
        Np = None
        ppb = 160
        loading_params = LoadingParameters(ppb=ppb, loading="tesselation")
    else:
        Np = 160000
        ppb = None
        loading_params = LoadingParameters(Np=Np, ppb=ppb, seed=1607)

    # background
    background = ConstantVelocity(n=1.5, density_profile="constant")
    background.domain = domain

    # perturbation
    if bc_x in ("periodic", "fixed"):
        fun_exact = lambda e1, e2, e3: 1.5 - xp.sin(2 * xp.pi * e1)
        pert = {"n": perturbations.ModesSin(ls=(1,), amps=(-1e-0,))}
    elif bc_x == "mirror":
        fun_exact = lambda e1, e2, e3: 1.5 - xp.cos(2 * xp.pi * e1)
        pert = {"n": perturbations.ModesCos(ls=(1,), amps=(-1e-0,))}

    # exact solution
    eta1 = xp.linspace(0, 1.0, eval_pts)  # add offset for non-periodic boundary conditions, TODO: implement Neumann
    eta2 = xp.array([0.0])
    eta3 = xp.array([0.0])
    ee1, ee2, ee3 = xp.meshgrid(eta1, eta2, eta3, indexing="ij")
    exact_eval = fun_exact(ee1, ee2, ee3)

    # boundary conditions
    boundary_params = BoundaryParameters(bc_sph=(bc_x, "periodic", "periodic"))

    # loop
    h_vec = [((2**k) * 10**-3 * 0.25) for k in range(2, 12)]
    err_vec = []
    for h1 in h_vec:
        particles = ParticlesSPH(
            comm_world=comm,
            loading_params=loading_params,
            boundary_params=boundary_params,
            boxes_per_dim=boxes_per_dim,
            bufsize=1.0,
            domain=domain,
            background=background,
            perturbations=pert,
            n_as_volume_form=True,
            verbose=False,
        )

        particles.draw_markers(sort=False, verbose=False)
        if comm is not None:
            particles.mpi_sort_markers()
        particles.initialize_weights()
        h2 = 1 / boxes_per_dim[1]
        h3 = 1 / boxes_per_dim[2]

        test_eval = particles.eval_density(ee1, ee2, ee3, h1=h1, h2=h2, h3=h3)

        if comm is None:
            all_eval = test_eval
        else:
            all_eval = xp.zeros_like(test_eval)
            comm.Allreduce(test_eval, all_eval, op=MPI.SUM)

        if show_plot and rank == 0:
            plt.figure()
            plt.plot(ee1.squeeze(), exact_eval.squeeze(), label="exact")
            plt.plot(ee1.squeeze(), all_eval.squeeze(), "--.", label="eval_sph")
            plt.title(f"{h1 =}")
            # plt.savefig(f"fun_{h1}.png")

        # error in max-norm
        diff = xp.max(xp.abs(all_eval - exact_eval)) / xp.max(xp.abs(exact_eval))

        print(f"{h1 =}, {diff =}")

        if tesselation and h1 < 0.256:
            assert diff < 0.036

        err_vec += [diff]

    if tesselation:
        fit = xp.polyfit(xp.log(h_vec[1:5]), xp.log(err_vec[1:5]), 1)
    else:
        fit = xp.polyfit(xp.log(h_vec[:-2]), xp.log(err_vec[:-2]), 1)

    if show_plot and rank == 0:
        plt.figure(figsize=(12, 8))
        plt.loglog(h_vec, err_vec, label="Convergence")
        plt.loglog(h_vec, xp.exp(fit[1]) * xp.array(h_vec) ** (fit[0]), "--", label=f"fit with slope {fit[0]}")
        plt.legend()
        plt.show()
        # plt.savefig("Convergence_SPH")

    if rank == 0:
        print(f"\n{bc_x =}, {eval_pts =}, {tesselation =}, {fit[0] =}")

    if not tesselation:
        assert xp.abs(fit[0] + 0.5) < 0.1  # Monte Carlo rate


@pytest.mark.parametrize("boxes_per_dim", [(12, 1, 1)])
@pytest.mark.parametrize("bc_x", ["periodic", "fixed", "mirror"])
@pytest.mark.parametrize("eval_pts", [11, 16])
@pytest.mark.parametrize("tesselation", [False, True])
def test_evaluation_mc_Np_and_h_convergence_1d(boxes_per_dim, bc_x, eval_pts, tesselation, show_plot=False):
    if isinstance(MPI.COMM_WORLD, MockComm):
        comm = None
        rank = 0
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

    # DOMAIN object
    dom_type = "Cuboid"
    dom_params = {"l1": 0.0, "r1": 3.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    if tesselation:
        ppbs = [4, 8, 16, 32, 64]
        Nps = [None] * len(ppbs)
    else:
        Nps = [(2**k) * 10**3 for k in range(-2, 9)]
        ppbs = [None] * len(Nps)

    # background
    background = ConstantVelocity(n=1.5, density_profile="constant")
    background.domain = domain

    # perturbation
    if bc_x in ("periodic", "fixed"):
        fun_exact = lambda e1, e2, e3: 1.5 - xp.sin(2 * xp.pi * e1)
        pert = {"n": perturbations.ModesSin(ls=(1,), amps=(-1e-0,))}
    elif bc_x == "mirror":
        fun_exact = lambda e1, e2, e3: 1.5 - xp.cos(2 * xp.pi * e1)
        pert = {"n": perturbations.ModesCos(ls=(1,), amps=(-1e-0,))}

    # exact solution
    eta1 = xp.linspace(0, 1.0, eval_pts)
    eta2 = xp.array([0.0])
    eta3 = xp.array([0.0])
    ee1, ee2, ee3 = xp.meshgrid(eta1, eta2, eta3, indexing="ij")
    exact_eval = fun_exact(ee1, ee2, ee3)

    # boundary conditions
    boundary_params = BoundaryParameters(bc_sph=(bc_x, "periodic", "periodic"))

    h_arr = [((2**k) * 10**-3 * 0.25) for k in range(2, 12)]
    err_vec = []
    for h in h_arr:
        err_vec += [[]]
        for Np, ppb in zip(Nps, ppbs):
            if tesselation:
                loading_params = LoadingParameters(ppb=ppb, loading="tesselation")
            else:
                loading_params = LoadingParameters(Np=Np, seed=1607)

            particles = ParticlesSPH(
                comm_world=comm,
                loading_params=loading_params,
                boundary_params=boundary_params,
                boxes_per_dim=boxes_per_dim,
                bufsize=1.0,
                domain=domain,
                background=background,
                perturbations=pert,
                n_as_volume_form=True,
                verbose=False,
            )

            particles.draw_markers(sort=False, verbose=False)
            if comm is not None:
                particles.mpi_sort_markers()
            particles.initialize_weights()

            h2 = 1 / boxes_per_dim[1]
            h3 = 1 / boxes_per_dim[2]

            test_eval = particles.eval_density(ee1, ee2, ee3, h1=h, h2=h2, h3=h3)

            if comm is None:
                all_eval = test_eval
            else:
                all_eval = xp.zeros_like(test_eval)
                comm.Allreduce(test_eval, all_eval, op=MPI.SUM)

            # error in max-norm
            diff = xp.max(xp.abs(all_eval - exact_eval)) / xp.max(xp.abs(exact_eval))
            err_vec[-1] += [diff]

            if rank == 0:
                print(f"{Np =}, {ppb =}, {diff =}")
                # if show_plot:
                #     plt.figure()
                #     plt.plot(ee1.squeeze(), fun_exact(ee1, ee2, ee3).squeeze(), label="exact")
                #     plt.plot(ee1.squeeze(), all_eval.squeeze(), "--.", label="eval_sph")
                #     plt.title(f"{h = }, {Np = }")
                #     # plt.savefig(f"fun_h{h}_N{Np}_ppb{ppb}.png")

    err_vec = xp.array(err_vec)
    err_min = xp.min(err_vec)

    if show_plot and rank == 0:
        if tesselation:
            h_mesh, n_mesh = xp.meshgrid(xp.log10(h_arr), xp.log10(ppbs), indexing="ij")
        if not tesselation:
            h_mesh, n_mesh = xp.meshgrid(xp.log10(h_arr), xp.log10(Nps), indexing="ij")
        plt.figure(figsize=(6, 6))
        plt.pcolor(h_mesh, n_mesh, xp.log10(err_vec), shading="auto")
        plt.title("Error")
        plt.colorbar(label="log10(error)")
        plt.xlabel("log10(h)")
        plt.ylabel("log10(particles)")

        min_indices = xp.argmin(err_vec, axis=0)
        min_h_values = []
        for mi in min_indices:
            min_h_values += [xp.log10(h_arr[mi])]
        if tesselation:
            log_particles = xp.log10(ppbs)
        else:
            log_particles = xp.log10(Nps)
        plt.plot(min_h_values, log_particles, "r-", label="Min error h for each Np", linewidth=2)
        plt.legend()
        # plt.savefig("SPH_conv_in_h_and_N.png")

        plt.show()

    if rank == 0:
        print(f"\n{tesselation =}, {bc_x =}, {err_min =}")

    if tesselation:
        if bc_x == "periodic":
            assert xp.min(err_vec) < 7.7e-5
        elif bc_x == "fixed":
            assert err_min < 7.7e-5
        else:
            assert err_min < 7.7e-5
    else:
        if bc_x in ("periodic", "fixed"):
            assert err_min < 0.0089
        else:
            assert err_min < 0.021


@pytest.mark.parametrize("boxes_per_dim", [(24, 24, 1)])
@pytest.mark.parametrize("bc_x", ["periodic", "fixed", "mirror"])
@pytest.mark.parametrize("bc_y", ["periodic", "fixed", "mirror"])
@pytest.mark.parametrize("tesselation", [False, True])
def test_evaluation_SPH_Np_convergence_2d(boxes_per_dim, bc_x, bc_y, tesselation, show_plot=False):
    if isinstance(MPI.COMM_WORLD, MockComm):
        comm = None
        rank = 0
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

    # DOMAIN object
    dom_type = "Cuboid"

    Lx = 1.0
    Ly = 1.0
    dom_params = {"l1": 0.0, "r1": Lx, "l2": 0.0, "r2": Ly, "l3": 0.0, "r3": 1.0}
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    if tesselation:
        ppbs = [4, 8, 16, 32, 64, 200]
        Nps = [None] * len(ppbs)
    else:
        Nps = [(2**k) * 10**3 for k in range(-2, 9)]
        ppbs = [None] * len(Nps)

    # background
    background = ConstantVelocity(n=1.5, density_profile="constant")
    background.domain = domain

    # perturbation
    if bc_x in ("periodic", "fixed"):
        if bc_y in ("periodic", "fixed"):
            fun_exact = lambda x, y, z: 1.5 - xp.sin(2 * xp.pi / Lx * x) * xp.sin(2 * xp.pi / Ly * y)
            pert = {"n": perturbations.ModesSinSin(ls=(1,), ms=(1,), amps=(-1e-0,))}
        elif bc_y == "mirror":
            fun_exact = lambda x, y, z: 1.5 - xp.sin(2 * xp.pi / Lx * x) * xp.cos(2 * xp.pi / Ly * y)
            pert = {"n": perturbations.ModesSinCos(ls=(1,), ms=(1,), amps=(-1e-0,))}

    elif bc_x == "mirror":
        if bc_y in ("periodic", "fixed"):
            fun_exact = lambda x, y, z: 1.5 - xp.cos(2 * xp.pi / Lx * x) * xp.sin(2 * xp.pi / Ly * y)
            pert = {"n": perturbations.ModesCosSin(ls=(1,), ms=(1,), amps=(-1e-0,))}
        elif bc_y == "mirror":
            fun_exact = lambda x, y, z: 1.5 - xp.cos(2 * xp.pi / Lx * x) * xp.cos(2 * xp.pi / Ly * y)
            pert = {"n": perturbations.ModesCosCos(ls=(1,), ms=(1,), amps=(-1e-0,))}

    # exact solution
    eta1 = xp.linspace(0, 1.0, 41)
    eta2 = xp.linspace(0, 1.0, 86)
    eta3 = xp.array([0.0])
    ee1, ee2, ee3 = xp.meshgrid(eta1, eta2, eta3, indexing="ij")
    x, y, z = domain(eta1, eta2, eta3)
    exact_eval = fun_exact(x, y, z)

    # boundary conditions
    boundary_params = BoundaryParameters(bc_sph=(bc_x, bc_y, "periodic"))

    err_vec = []
    for Np, ppb in zip(Nps, ppbs):
        if tesselation:
            loading_params = LoadingParameters(ppb=ppb, loading="tesselation")
        else:
            loading_params = LoadingParameters(Np=Np, seed=1607)

        particles = ParticlesSPH(
            comm_world=comm,
            loading_params=loading_params,
            boundary_params=boundary_params,
            boxes_per_dim=boxes_per_dim,
            bufsize=1.0,
            box_bufsize=4.0,
            domain=domain,
            background=background,
            perturbations=pert,
            n_as_volume_form=True,
            verbose=False,
        )
        if rank == 0:
            print(f"{particles.domain_array}")

        particles.draw_markers(sort=False, verbose=False)
        if comm is not None:
            particles.mpi_sort_markers()
        particles.initialize_weights()
        h1 = 1 / boxes_per_dim[0]
        h2 = 1 / boxes_per_dim[1]
        h3 = 1 / boxes_per_dim[2]

        test_eval = particles.eval_density(ee1, ee2, ee3, h1=h1, h2=h2, h3=h3, kernel_type="gaussian_2d")

        if comm is None:
            all_eval = test_eval
        else:
            all_eval = xp.zeros_like(test_eval)
            comm.Allreduce(test_eval, all_eval, op=MPI.SUM)

        # error in max-norm
        diff = xp.max(xp.abs(all_eval - exact_eval)) / xp.max(xp.abs(exact_eval))
        err_vec += [diff]

        if tesselation:
            assert diff < 0.06

        if rank == 0:
            print(f"{Np =}, {ppb =}, {diff =}")
            if show_plot:
                fig, ax = plt.subplots()
                d = ax.pcolor(ee1.squeeze(), ee2.squeeze(), all_eval.squeeze(), label="eval_sph", vmin=1.0, vmax=2.0)
                fig.colorbar(d, ax=ax, label="2d_SPH")
                ax.set_xlabel("ee1")
                ax.set_ylabel("ee2")
                ax.set_title(f"{Np}_{ppb =}")
                # fig.savefig(f"2d_sph_{Np}_{ppb}.png")

    if tesselation:
        fit = xp.polyfit(xp.log(ppbs), xp.log(err_vec), 1)
        xvec = ppbs
    else:
        fit = xp.polyfit(xp.log(Nps), xp.log(err_vec), 1)
        xvec = Nps

    if show_plot and rank == 0:
        plt.figure(figsize=(12, 8))
        plt.loglog(xvec, err_vec, label="Convergence")
        plt.loglog(xvec, xp.exp(fit[1]) * xp.array(xvec) ** (fit[0]), "--", label=f"fit with slope {fit[0]}")
        plt.legend()
        plt.show()
        # plt.savefig(f"Convergence_SPH_{tesselation=}")

    if rank == 0:
        print(f"\n{bc_x =}, {tesselation =}, {fit[0] =}")

    if not tesselation:
        assert xp.abs(fit[0] + 0.5) < 0.1  # Monte Carlo rate


@pytest.mark.parametrize("boxes_per_dim", [(12, 1, 1)])
@pytest.mark.parametrize("kernel", ["trigonometric_1d", "gaussian_1d", "linear_1d"])
@pytest.mark.parametrize("derivative", [0, 1])
@pytest.mark.parametrize("bc_x", ["periodic", "mirror", "fixed"])
@pytest.mark.parametrize("eval_pts", [11, 16])
@pytest.mark.parametrize("tesselation", [False, True])
def test_sph_velocity_evaluation(
    boxes_per_dim,
    kernel,
    derivative,
    bc_x,
    eval_pts,
    tesselation,
    show_plot=False,
):
    if isinstance(MPI.COMM_WORLD, MockComm):
        comm = None
        rank = 0
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

    # DOMAIN object
    dom_type = "Cuboid"
    dom_params = {"l1": 0.0, "r1": 1.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    if tesselation:
        ppb = 10
        loading_params = LoadingParameters(ppb=ppb, seed=1607, loading="tesselation")
    else:
        ppb = 400
        loading_params = LoadingParameters(ppb=ppb, seed=223)

    # test velocity profile
    Lx = dom_params["r1"] - dom_params["l1"]

    def u_xyz(x, y, z):
        ux = xp.cos(2 * xp.pi / Lx * x)
        uy = 0.0 * x
        uz = 0.0 * x
        return (ux, uy, uz)

    def du_xyz(x, y, z):
        ux = -2 * xp.pi / Lx * xp.sin(2 * xp.pi / Lx * x)
        uy = 0.0 * x
        uz = 0.0 * x
        return (ux, uy, uz)

    background = equils.GenericCartesianFluidEquilibrium(u_xyz=u_xyz)
    background.domain = domain

    boundary_params = BoundaryParameters(bc_sph=(bc_x, "periodic", "periodic"))

    particles = ParticlesSPH(
        comm_world=comm,
        loading_params=loading_params,
        boundary_params=boundary_params,
        boxes_per_dim=boxes_per_dim,
        bufsize=2.0,
        box_bufsize=4.0,
        domain=domain,
        background=background,
        n_as_volume_form=True,
        verbose=False,
    )

    eta1 = xp.linspace(0, 1.0, eval_pts)
    eta2 = xp.array([0.0])
    eta3 = xp.array([0.0])
    ee1, ee2, ee3 = xp.meshgrid(eta1, eta2, eta3, indexing="ij")

    particles.draw_markers(sort=False, verbose=False)
    if comm is not None:
        particles.mpi_sort_markers()
    particles.initialize_weights()

    e1_bins = xp.linspace(0, 1.0, 200, endpoint=True)
    dv = e1_bins[1] - e1_bins[0]

    binned_res, r2 = particles.binning([True, False, False, False, False, False], [e1_bins], "current_1")

    v1_plot = e1_bins[:-1] + dv / 2

    if show_plot:
        plt.plot(v1_plot, binned_res, "r*", label="From binning")
        plt.title(r"Full-$f$: Maxwellian in $v_1$-direction")
        plt.xlabel(r"$v_1$")
        plt.ylabel(r"$f(v_1)$")
        plt.legend()
        # plt.savefig("Binning_v1.png")

    h1 = 1 / boxes_per_dim[0]
    h2 = 1 / boxes_per_dim[1]
    h3 = 1 / boxes_per_dim[2]

    v1, v2, v3 = particles.eval_velocity(
        ee1,
        ee2,
        ee3,
        h1=h1,
        h2=h2,
        h3=h3,
        kernel_type=kernel,
        derivative=derivative,
    )

    if derivative == 0:
        v1_e, v2_e, v3_e = background.u_xyz(ee1, ee2, ee3)
    else:
        v1_e, v2_e, v3_e = du_xyz(ee1, ee2, ee3)

    if comm is not None:
        all_velo1 = xp.zeros_like(v1)
        all_velo2 = xp.zeros_like(v2)
        all_velo3 = xp.zeros_like(v3)
        comm.Allreduce(v1, all_velo1, op=MPI.SUM)
        comm.Allreduce(v2, all_velo2, op=MPI.SUM)
        comm.Allreduce(v3, all_velo3, op=MPI.SUM)
    else:
        all_velo1, all_velo2, all_velo3 = v1, v2, v3

    err_ux = xp.max(xp.abs(all_velo1 - v1_e)) / xp.max(xp.abs(v1_e))
    err_uy = xp.max(xp.abs(all_velo2 - v2_e)) / xp.max(xp.abs(v2_e))
    err_uz = xp.max(xp.abs(all_velo3 - v3_e)) / xp.max(xp.abs(v3_e))

    if rank == 0:
        print(f"\n{boxes_per_dim = }")
        print(f"{kernel = }, {derivative = }")
        print(f"{bc_x = }, {eval_pts = }, {tesselation = }")
        print(f"Velocity errors: ux={err_ux:.3e}, uy={err_uy:.3e}, uz={err_uz:.3e}")

        if show_plot:
            plt.figure(figsize=(12, 6))
            plt.plot(ee1.squeeze(), v1_e.squeeze(), label="exact vx")
            plt.plot(ee1.squeeze(), all_velo1.squeeze(), "--.", label="SPH vx")
            plt.xlabel("e1")
            plt.ylabel("Velocity (vx)")
            plt.legend()
            plt.grid(True)
            # plt.savefig("image_test.png")
            plt.show()

    if tesselation:
        assert err_ux < 2.5e-2
    else:
        assert err_ux < 1.84e-1


@pytest.mark.parametrize("boxes_per_dim", [(12, 12, 1)])
@pytest.mark.parametrize("kernel", ["gaussian_2d"])  # "trigonometric_2d", "linear_2d"])
@pytest.mark.parametrize("derivative", [0, 1, 2])
@pytest.mark.parametrize("bc_x", ["periodic", "mirror", "fixed"])
@pytest.mark.parametrize("bc_y", ["periodic", "mirror", "fixed"])
@pytest.mark.parametrize("eval_pts", [11])
@pytest.mark.parametrize("tesselation", [False, True])
def test_sph_velocity_evaluation_2d(
    boxes_per_dim,
    kernel,
    derivative,
    bc_x,
    bc_y,
    eval_pts,
    tesselation,
    show_plot=False,
):
    if isinstance(MPI.COMM_WORLD, MockComm):
        comm = None
        rank = 0
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

    dom_type = "Cuboid"
    l1 = 0.0
    r1 = 2.0
    l2 = 0.0
    r2 = 3.0
    dom_params = {"l1": l1, "r1": r1, "l2": l2, "r2": r2, "l3": 0.0, "r3": 1.0}
    domain_class = getattr(domains, dom_type)
    domain: Domain = domain_class(**dom_params)

    if tesselation:
        ppb = 50
        loading_params = LoadingParameters(ppb=ppb, seed=1607, loading="tesselation")
    else:
        ppb = 400
        loading_params = LoadingParameters(ppb=ppb, seed=223)

    Lx = r1 - l1
    Ly = r2 - l2

    # analytic 2D velocity field:
    # u_x = cos(2π e1) * cos(2π e2)
    # u_y = sin(2π e1) * sin(2π e2)
    # u_z = 0
    def u_xyz(x, y, z):
        ux = xp.cos(2 * xp.pi / Lx * x) * xp.cos(2 * xp.pi / Ly * y)
        uy = xp.cos(2 * xp.pi / Lx * x) * xp.cos(2 * xp.pi / Ly * y)
        uz = 0.0 * z
        return (ux, uy, uz)

    def du_deta1(eta1, eta2, eta3):
        du1 = (
            -2
            * xp.pi
            / Lx
            * xp.sin(2 * xp.pi * eta1 + 2 * xp.pi * l1 / Lx)
            * xp.cos(2 * xp.pi * eta2 + 2 * xp.pi * l2 / Ly)
        )
        du2 = (
            -2
            * xp.pi
            / Ly
            * xp.sin(2 * xp.pi * eta1 + 2 * xp.pi * l1 / Lx)
            * xp.cos(2 * xp.pi * eta2 + 2 * xp.pi * l2 / Ly)
        )
        du3 = 0.0 * z
        return (du1, du2, du3)

    def du_deta2(eta1, eta2, eta3):
        du1 = (
            -2
            * xp.pi
            / Lx
            * xp.cos(2 * xp.pi * l1 / Lx + 2 * xp.pi * eta1)
            * xp.sin(2 * xp.pi * l2 / Ly + 2 * xp.pi * eta2)
        )
        du2 = (
            -2
            * xp.pi
            / Ly
            * xp.cos(2 * xp.pi * eta1 + 2 * xp.pi * l1 / Lx)
            * xp.sin(2 * xp.pi * l2 / Ly + 2 * xp.pi * eta2)
        )
        du3 = 0.0 * z
        return (du1, du2, du3)

    background = equils.GenericCartesianFluidEquilibrium(u_xyz=u_xyz)
    background.domain = domain

    boundary_params = BoundaryParameters(bc_sph=(bc_x, bc_y, "periodic"))

    verbose = False

    particles = ParticlesSPH(
        comm_world=comm,
        loading_params=loading_params,
        boundary_params=boundary_params,
        boxes_per_dim=boxes_per_dim,
        bufsize=2.0,
        box_bufsize=4.0,
        domain=domain,
        background=background,
        n_as_volume_form=True,
        verbose=verbose,
    )

    # evaluation grids
    eta1 = xp.linspace(0, 1.0, eval_pts)
    eta2 = xp.linspace(0, 1.0, eval_pts)
    eta3 = xp.array([0.0])
    ee1, ee2, ee3 = xp.meshgrid(eta1, eta2, eta3, indexing="ij")

    x = xp.linspace(l1, r1, eval_pts)
    y = xp.linspace(l2, r2, eval_pts)
    z = xp.array([0.0])
    xx, yy, zz = xp.meshgrid(x, y, z, indexing="ij")

    # initialize particles
    particles.draw_markers(sort=False, verbose=verbose)
    if comm is not None:
        particles.mpi_sort_markers()
    particles.initialize_weights()

    # evaluate velocity (and derivatives) via SPH
    h1 = 1 / boxes_per_dim[0]
    h2 = 1 / boxes_per_dim[1]
    h3 = 1 / boxes_per_dim[2]

    v_log = particles.eval_velocity(
        ee1,
        ee2,
        ee3,
        h1=h1,
        h2=h2,
        h3=h3,
        kernel_type=kernel,
        derivative=derivative,
    )
    v1, v2, v3 = v_log

    if derivative == 0:
        # push-forward of vector field velocity
        v1, v2, v3 = domain.push(v_log, ee1, ee2, ee3, kind="v")
        v1_e, v2_e, v3_e = background.u_xyz(xx, yy, zz)
    elif derivative == 1:
        v1_e, v2_e, v3_e = du_deta1(ee1, ee2, ee3)  # du_dx(xx, yy, zz)
    else:  # derivative == 2
        v1_e, v2_e, v3_e = du_deta2(ee1, ee2, ee3)  # du_dy(xx, yy, zz)

    if comm is not None:
        all_velo1 = xp.zeros_like(v1)
        all_velo2 = xp.zeros_like(v2)
        all_velo3 = xp.zeros_like(v3)
        comm.Allreduce(v1, all_velo1, op=MPI.SUM)
        comm.Allreduce(v2, all_velo2, op=MPI.SUM)
        comm.Allreduce(v3, all_velo3, op=MPI.SUM)
    else:
        all_velo1, all_velo2, all_velo3 = v1, v2, v3

    def abs_err(num, exact):
        max_exact = xp.max(xp.abs(exact))

        return xp.max(xp.abs(num - exact)) / max_exact

    if derivative == 0:
        err_ux = abs_err(all_velo1, v1_e)
        err_uy = abs_err(all_velo2, v2_e)
    elif derivative == 1:
        err_ux = abs_err(all_velo1, v1_e)
        err_uy = abs_err(all_velo2, v2_e)
    elif derivative == 2:
        err_ux = abs_err(all_velo1, v1_e)
        err_uy = abs_err(all_velo2, v2_e)

    if rank == 0:
        print(f"\n{boxes_per_dim = }")
        print(f"{kernel = }, {derivative = }")
        print(f"{bc_x = }, {bc_y = }, {eval_pts = }")
        print(f"Velocity errors: ux={err_ux:.3e}, uy={err_uy:.3e}")

        if show_plot:
            plt.figure(figsize=(12, 24))
            # --- vx plots ---
            plt.subplot(3, 2, 1)
            plt.pcolor(ee1.squeeze(), ee2.squeeze(), v1_e.squeeze())
            plt.title("Exact v₁ (uₓ)")
            plt.colorbar()

            plt.subplot(3, 2, 3)
            plt.pcolor(ee1.squeeze(), ee2.squeeze(), all_velo1.squeeze())
            plt.title("SPH v₁ (uₓ)")
            plt.colorbar()

            plt.subplot(3, 2, 5)
            plt.pcolor(ee1.squeeze(), ee2.squeeze(), (all_velo1 - v1_e).squeeze())
            plt.title("Error v₁ (uₓ)")
            plt.colorbar()

            # --- vy plots ---
            plt.subplot(3, 2, 2)
            plt.pcolor(ee1.squeeze(), ee2.squeeze(), v2_e.squeeze())
            plt.title("Exact v₂ (u_y)")
            plt.colorbar()

            plt.subplot(3, 2, 4)
            plt.pcolor(ee1.squeeze(), ee2.squeeze(), all_velo2.squeeze())
            plt.title("SPH v₂ (u_y)")
            plt.colorbar()

            plt.subplot(3, 2, 6)
            plt.pcolor(ee1.squeeze(), ee2.squeeze(), (all_velo2 - v2_e).squeeze())
            plt.title("Error v₂ (u_y)")
            plt.colorbar()

            plt.tight_layout()
            plt.savefig("image_test_2d.png")
            plt.show()

            plt.figure(figsize=(8, 8))
            plt.quiver(
                ee1.squeeze(),
                ee2.squeeze(),
                all_velo1.squeeze(),
                all_velo2.squeeze(),
                scale=30,
                pivot="mid",
                color="blue",
            )
            plt.title("SPH Velocity Field (v₁, v₂)")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.axis("equal")
            plt.tight_layout()
            plt.savefig("image_test_2d_quiver.png")
            plt.show()

    # tolerances: conservative values aligned with your 2D density thresholds
    if tesselation:
        assert err_ux < 1.55e-2
        assert err_uy < 1.55e-2
    elif derivative == 0:
        assert err_ux < 0.09
        assert err_uy < 0.09
    elif derivative == 1:
        assert err_ux < 0.632
        assert err_uy < 0.632
    else:
        assert err_ux < 0.52
        assert err_uy < 0.52

    # ensure z-component is negligible (absolute)
    # assert err_uz_abs < 1e-6


@pytest.mark.parametrize("boxes_per_dim", [(12, 12, 1)])
@pytest.mark.parametrize("kernel", ["gaussian_2d"])  # "trigonometric_2d", "linear_2d"]))
@pytest.mark.parametrize("bc_x", ["periodic"])
@pytest.mark.parametrize("bc_y", ["periodic"])
@pytest.mark.parametrize("eval_pts", [11])
@pytest.mark.parametrize("tesselation", [True])
def test_sph_viscosity_evaluation_2d(
    boxes_per_dim,
    kernel,
    bc_x,
    bc_y,
    eval_pts,
    tesselation,
    show_plot=False,
):
    if isinstance(MPI.COMM_WORLD, MockComm):
        comm = None
        rank = 0
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

    dom_type = "Cuboid"
    l1 = 0.0
    r1 = 1.0
    l2 = 0.0
    r2 = 1.0
    dom_params = {"l1": l1, "r1": r1, "l2": l2, "r2": r2, "l3": 0.0, "r3": 1.0}
    domain_class = getattr(domains, dom_type)
    domain: Domain = domain_class(**dom_params)

    if tesselation:
        ppb = 50
        loading_params = LoadingParameters(ppb=ppb, seed=1607, loading="tesselation")
    else:
        ppb = 2000
        loading_params = LoadingParameters(ppb=ppb, seed=223)

    Lx = r1 - l1
    Ly = r2 - l2

    # -----------------------------
    # 1. velocity field
    # -----------------------------
    def u_xyz(x, y, z):
        """
        Compute the 3D Taylor-Green vortex velocity at points x, y, z.
        u = [sin(x)cos(y)cos(z), -cos(x)sin(y)cos(z), 0]
        """
        u_x = xp.sin(2 * xp.pi * x) * xp.cos(2 * xp.pi * y)
        u_y = xp.cos(2 * xp.pi * x) * xp.cos(2 * xp.pi * y)
        u_z = xp.zeros_like(x)

        return u_x, u_y, u_z

    # -----------------------------
    # 2. Divergence of pi
    # -----------------------------
    mu = 1e-3

    def div_pi_analytic(x, y, z, mu=None):
        """
        Returns:
            div_Pi_x, div_Pi_y, div_Pi_z
        """

        div_x = mu * (
            (28 / 3) * xp.pi**2 * xp.sin(2 * xp.pi * x) * xp.cos(2 * xp.pi * y)
            - (4 * xp.pi**2 / 3) * xp.sin(2 * xp.pi * x) * xp.sin(2 * xp.pi * y)
        )
        div_y = mu * (
            (28 / 3) * xp.pi**2 * xp.cos(2 * xp.pi * x) * xp.cos(2 * xp.pi * y)
            + (4 * xp.pi**2 / 3) * xp.cos(2 * xp.pi * x) * xp.sin(2 * xp.pi * y)
        )
        div_z = mu * xp.zeros_like(x)

        return div_x, div_y, div_z

    background = equils.GenericCartesianFluidEquilibrium(u_xyz=u_xyz)
    background.domain = domain

    boundary_params = BoundaryParameters(bc_sph=(bc_x, bc_y, "periodic"))

    verbose = False

    particles = ParticlesSPH(
        comm_world=comm,
        loading_params=loading_params,
        boundary_params=boundary_params,
        boxes_per_dim=boxes_per_dim,
        bufsize=2.0,
        box_bufsize=4.0,
        domain=domain,
        background=background,
        n_as_volume_form=True,
        verbose=verbose,
    )

    # initialize particles
    particles.draw_markers(sort=False, verbose=verbose)
    if comm is not None:
        particles.mpi_sort_markers()
    particles.initialize_weights()

    # evaluation grids
    eta1 = xp.linspace(0, 1.0, eval_pts)
    eta2 = xp.linspace(0, 1.0, eval_pts)
    eta3 = xp.array([0.0])
    ee1, ee2, ee3 = xp.meshgrid(eta1, eta2, eta3, indexing="ij")

    x = xp.linspace(l1, r1, eval_pts)
    y = xp.linspace(l2, r2, eval_pts)
    z = xp.array([0.0])
    xx, yy, zz = xp.meshgrid(x, y, z, indexing="ij")

    # exact values
    density_exact = background.n_xyz(xx, yy, zz)
    vx_exact, vy_exact, _ = background.u_xyz(xx, yy, zz)

    # evaluate density
    h1 = 1 / boxes_per_dim[0]
    h2 = 1 / boxes_per_dim[1]
    h3 = 1 / boxes_per_dim[2]

    density = particles.eval_density(
        ee1,
        ee2,
        ee3,
        h1=h1,
        h2=h2,
        h3=h3,
        kernel_type=kernel,
        derivative=0,
    )
    if rank == 0:
        print(f"{density.shape = }")
        print(f"{xp.min(density) = }, {xp.max(density) = }")

    if show_plot:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.pcolor(xx.squeeze(), yy.squeeze(), density.squeeze(), vmin=xp.min(density), vmax=xp.max(density))
        plt.title("density sph")
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.pcolor(xx.squeeze(), yy.squeeze(), density_exact.squeeze(), vmin=xp.min(density), vmax=xp.max(density))
        plt.title("density exact")
        plt.colorbar()
        # plt.savefig("density")
        plt.show()

    # evaluate velocity
    vx, vy, vz = particles.eval_velocity(
        ee1,
        ee2,
        ee3,
        h1=h1,
        h2=h2,
        h3=h3,
        kernel_type=kernel,
        derivative=0,
    )
    if rank == 0:
        print(f"{vx.shape = }, {vy.shape = }")
        print(f"{xp.min(vx) = }, {xp.max(vx) = }")
        print(f"{xp.min(vy) = }, {xp.max(vy) = }")

    if show_plot:
        plt.figure(figsize=(10, 8))

        plt.subplot(2, 2, 1)
        plt.pcolor(xx.squeeze(), yy.squeeze(), vx.squeeze(), vmin=xp.min(vx), vmax=xp.max(vx))
        plt.title("vx sph")
        plt.colorbar()

        plt.subplot(2, 2, 3)
        plt.pcolor(xx.squeeze(), yy.squeeze(), vx_exact.squeeze(), vmin=xp.min(vx), vmax=xp.max(vx))
        plt.title("vx exact")
        plt.colorbar()

        plt.subplot(2, 2, 2)
        plt.pcolor(xx.squeeze(), yy.squeeze(), vy.squeeze())
        plt.title("vy sph")
        plt.colorbar()

        plt.subplot(2, 2, 4)
        plt.pcolor(xx.squeeze(), yy.squeeze(), vy_exact.squeeze(), vmin=xp.min(vy), vmax=xp.max(vy))
        plt.title("vy exact")
        plt.colorbar()

        plt.show()
        # plt.savefig("velocity")

    # evaluate div of viscosity tensor (numerically)
    div_viscosity = particles.eval_div_viscosity(
        ee1,
        ee2,
        ee3,
        h1=h1,
        h2=h2,
        h3=h3,
        mu=mu,
        kernel_type=kernel,
    )
    gamma_x = div_viscosity[0]
    gamma_y = div_viscosity[1]
    gamma_z = div_viscosity[2]

    div_pi_exact = div_pi_analytic(xx, yy, zz, mu=mu)
    div_pi_x = div_pi_exact[0]
    div_pi_y = div_pi_exact[1]
    div_pi_z = div_pi_exact[2]

    if comm is not None:
        all_div_x = xp.zeros_like(gamma_x)
        all_div_y = xp.zeros_like(gamma_y)
        all_div_z = xp.zeros_like(gamma_z)

        comm.Allreduce(gamma_x, all_div_x, op=MPI.SUM)
        comm.Allreduce(gamma_y, all_div_y, op=MPI.SUM)
        comm.Allreduce(gamma_z, all_div_z, op=MPI.SUM)
    else:
        all_div_x, all_div_y, all_div_z = gamma_x, gamma_y, gamma_z

    def abs_err(num, exact):
        max_exact = xp.max(xp.abs(exact))
        return xp.max(xp.abs(num - exact)) / max_exact

    # compute errors
    err_div_x = abs_err(all_div_x, div_pi_x)
    err_div_y = abs_err(all_div_y, div_pi_y)
    # err_div_z = abs_err(all_div_z, div_pi_z)

    if rank == 0:
        print(f"\n{boxes_per_dim = }")
        print(f"{kernel = }")
        print(f"{bc_x = }, {bc_y = }, {eval_pts = }")
        print(f"Divergence of viscosity errors: gx={err_div_x:.3e}, gy={err_div_y:.3e}")
        # , gz={err_div_z:.3e}

    if show_plot:
        # --- gamma_x and gamma_y plots ---
        plt.figure(figsize=(18, 18))

        # gamma_x plots
        plt.subplot(3, 3, 1)
        plt.pcolor(ee1.squeeze(), ee2.squeeze(), div_pi_x.squeeze())
        plt.title("Exact div_viscosity_x")
        plt.colorbar()
        # plt.savefig("Exact div_viscosity_x")

        plt.subplot(3, 3, 4)
        plt.pcolor(ee1.squeeze(), ee2.squeeze(), all_div_x.squeeze())
        plt.title("SPH div_viscosity_x")
        plt.colorbar()
        # plt.savefig("SPH div_viscosity_x")
        plt.subplot(3, 3, 7)
        plt.pcolor(ee1.squeeze(), ee2.squeeze(), (all_div_x - div_pi_x).squeeze())
        plt.title("Error div_viscosity_x")
        plt.colorbar()
        # plt.savefig("Error div_viscosity_x")

        # gamma_y plots
        plt.subplot(3, 3, 2)
        plt.pcolor(ee1.squeeze(), ee2.squeeze(), div_pi_y.squeeze())
        plt.title("Exact div_viscosity_y")
        plt.colorbar()
        # plt.savefig("Exact div_viscosity_y")

        plt.subplot(3, 3, 5)
        plt.pcolor(ee1.squeeze(), ee2.squeeze(), all_div_y.squeeze())
        plt.title("SPH div_viscosity_y")
        plt.colorbar()
        # plt.savefig("SPH div_viscosity_y")

        plt.subplot(3, 3, 8)
        plt.pcolor(ee1.squeeze(), ee2.squeeze(), (all_div_y - div_pi_y).squeeze())
        plt.title("Error div_viscosity_y")
        plt.colorbar()

        # gamma_z plots
        plt.subplot(3, 3, 3)
        plt.pcolor(ee1.squeeze(), ee2.squeeze(), div_pi_z.squeeze())
        plt.title("Exact div_viscosity_z")
        plt.colorbar()

        plt.subplot(3, 3, 6)
        plt.pcolor(ee1.squeeze(), ee2.squeeze(), all_div_z.squeeze())
        plt.title("SPH div_viscosity_z")
        plt.colorbar()

        plt.subplot(3, 3, 9)
        plt.pcolor(ee1.squeeze(), ee2.squeeze(), (all_div_z - div_pi_z).squeeze())
        plt.title("Error div_viscosity_z")
        plt.colorbar()

        plt.tight_layout()
        # plt.savefig("div_viscosity_2d_all.png")
        plt.show()
        # plt.savefig("viscosity_2d_all")
        # else:
        #    assert err_div_x <  5.8e-01
        #    assert err_div_y < 5.9e-01

        # ------------------------------
        # Convergence study
        # # ------------------------------
        # if rank == 0:
        #     ppb_list = [1, 5, 10, 15, 25]
        #     errors = []

        #     for ppb_val in ppb_list:
        #         loading_params = LoadingParameters(
        #             ppb=ppb_val, seed=223, loading="tesselation" if tesselation else None
        #         )

        #         particles = ParticlesSPH(
        #             comm_world=comm,
        #             loading_params=loading_params,
        #             boundary_params=boundary_params,
        #             boxes_per_dim=boxes_per_dim,
        #             bufsize=2.0,
        #             box_bufsize=4.0,
        #             domain=domain,
        #             background=background,
        #             n_as_volume_form=True,
        #             verbose=False,
        #         )

        #         particles.draw_markers(sort=False, verbose=False)
        #         if comm is not None:
        #             particles.mpi_sort_markers()
        #         particles.initialize_weights()

        #         div_viscosity = particles.eval_div_viscosity(
        #             ee1,
        #             ee2,
        #             ee3,
        #             h1=h1,
        #             h2=h2,
        #             h3=h3,
        #             kernel_type=kernel,
        #         )

        #         gamma_x = div_viscosity[0]
        #         err = abs_err(gamma_x, div_pi_x)
        #         errors.append(float(err))

        #     ppb_arr = xp.array(ppb_list)
        #     errors = xp.array(errors)

        #     coeffs = xp.polyfit(xp.log(ppb_arr), xp.log(errors), 1)
        #     order = coeffs[0]

        #     print("\nSampling convergence study")
        #     print("ppb:", ppb_arr)
        #     print("errors:", errors)
        #     print(f"Observed scaling ≈ ppb^{order:.3f}")

        #     plt.figure()
        #     plt.loglog(ppb_arr, errors, "o-")
        #     plt.xlabel("ppb")
        #     plt.ylabel("Relative error")
        #     plt.savefig("ppb_convergence")

    if tesselation:
        assert err_div_x < 3.5e-2
        assert err_div_y < 3.5e-2



    # test_sph_velocity_evaluation_2d(
    #     (12, 12, 1), "gaussian_2d", 1, "periodic", "periodic", 101, tesselation=False, show_plot=True
    # )

    # test_sph_velocity_evaluation(
    #    (12, 1, 1),
    #    "gaussian_1d",
    #    1,
    #    "periodic",
    #   11,
    #    tesselation=False,
    #    show_plot=True,
    # )

    # test_sph_evaluation_1d(
    #     (24, 1, 1),
    #     "trigonometric_1d",
    #     # "gaussian_1d",
    #     1,
    #     # "periodic",
    #     "mirror",
    #     16,
    #     tesselation=False,
    #     show_plot=True,
    # )

    # test_sph_evaluation_2d(
    #     (12, 12, 1),
    #     # "trigonometric_2d",
    #     "gaussian_2d",
    #     1,
    #     "periodic",
    #     "periodic",
    #     16,
    #     show_plot=True
    # )

    # test_sph_evaluation_3d(
    #     (12, 8, 8),
    #     # "trigonometric_2d",
    #     "gaussian_3d",
    #     2,
    #     "periodic",
    #     "periodic",
    #     "periodic",
    #     11,
    #     show_plot=True
    # )

    # for nb in range(4, 25):
    #     print(f"\n{nb = }")
    # test_evaluation_SPH_Np_convergence_1d((12,1,1), "fixed", eval_pts=16, tesselation=False, show_plot=True)
    # test_evaluation_SPH_h_convergence_1d((12,1,1), "periodic", eval_pts=16, tesselation=True, show_plot=True)
    # test_evaluation_mc_Np_and_h_convergence_1d((12,1,1),"mirror", eval_pts=16, tesselation = False,  show_plot=True)
    # test_evaluation_SPH_Np_convergence_2d((24, 24, 1), "periodic", "periodic",  tesselation=True, show_plot=True)
    # test_evaluation_SPH_Np_convergence_2d((24, 24, 1), "periodic", "fixed", tesselation=True, show_plot=True)
    # test_evaluation_SPH_Np_convergence_2d((32, 32, 1), "fixed", "periodic", tesselation=True, show_plot=True)
    # test_evaluation_SPH_Np_convergence_2d((32, 32, 1), "fixed", "fixed",   tesselation=True, show_plot=True)
    # test_evaluation_SPH_Np_convergence_2d((32, 32, 1), "mirror", "mirror",  tesselation=True, show_plot=True)


@pytest.mark.parametrize("boxes_per_dim", [(12, 1, 1)])
@pytest.mark.parametrize("kernel", ["gaussian_1d", "linear_1d"])
@pytest.mark.parametrize("tesselation", [False, True])
@pytest.mark.parametrize("direction", ["x", "y", "z"])  
def test_sph_no_slip_boundary_1d(
    boxes_per_dim,
    kernel,
    tesselation,
    direction,
    show_plot=False,
):
    
    import sys
    import numpy
    numpy.set_printoptions(threshold=sys.maxsize, linewidth=200, precision=3, suppress=True)
    
    if isinstance(MPI.COMM_WORLD, MockComm):
        comm = None
        rank = 0
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

 
    dom_type = "Cuboid"
    dom_params = {"l1": 0.0, "r1": 1.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    if tesselation:
        ppb = 4
        loading_params = LoadingParameters(ppb=ppb, loading="tesselation")
    else:
        ppb = 200
        loading_params = LoadingParameters(ppb=ppb, seed=223)

    if direction == "x":
        def u_xyz(x, y, z):
            return (3.0 * xp.ones_like(x), xp.zeros_like(x), xp.zeros_like(x))
    elif direction == "y":
        def u_xyz(x, y, z):
            return (xp.zeros_like(x), xp.ones_like(x), xp.zeros_like(x))
    else:  
        def u_xyz(x, y, z):
            return (xp.zeros_like(x), xp.zeros_like(x), xp.ones_like(x))

    background = equils.GenericCartesianFluidEquilibrium(u_xyz=u_xyz)
    background.domain = domain
    if direction == "x":
        boundary_params = BoundaryParameters(bc_sph=("noslip", "periodic", "periodic"))
    elif direction == "y":
        boundary_params = BoundaryParameters(bc_sph=("periodic", "noslip", "periodic"))
    else:
        boundary_params = BoundaryParameters(bc_sph=("periodic", "periodic", "noslip"))
        

    particles = ParticlesSPH(
        comm_world=comm,
        loading_params=loading_params,
        boundary_params=boundary_params,
        boxes_per_dim=boxes_per_dim,
        bufsize=1.0,
        box_bufsize=2.0,
        domain=domain,
        background=background,
        n_as_volume_form=True,
        verbose=False,
    )

    particles.draw_markers(sort=True, verbose=False)
    if rank == 0:
        ghost_inds = xp.where(particles.ghost_particles)[0]
        print(f"After do_sort: {len(ghost_inds)} ghosts")
        if len(ghost_inds) > 0:
            print("First 10 ghost eta1:", particles.markers[ghost_inds[:10], 0])
    particles.initialize_weights()

    if direction == "x":
        eta1 = xp.linspace(0.0, 1.0, 100)
        eta2 = xp.array([0.5])
        eta3 = xp.array([0.5])
    elif direction == "y":
        eta1 = xp.array([0.5])
        eta2 = xp.linspace(0.0, 1.0, 100)
        eta3 = xp.array([0.5])
    else:
        eta1 = xp.array([0.5])
        eta2 = xp.array([0.5])
        eta3 = xp.linspace(0.0, 1.0, 100)

    ee1, ee2, ee3 = xp.meshgrid(eta1, eta2, eta3, indexing="ij")
    
    h1 = 1 / boxes_per_dim[0]
    h2 = 1 / boxes_per_dim[1]
    h3 = 1 / boxes_per_dim[2]

    v1, v2, v3 = particles.eval_velocity(
        ee1, ee2, ee3,
        h1=h1, h2=h2, h3=h3,
        kernel_type=kernel,
        derivative=0,
    )
    
    # if rank == 0 and len(ghost_inds) > 0:
    #     print("Ghost coefficients after eval:", particles.markers[ghost_inds[:10], particles.first_free_idx])
    #     print("Ghost positions after eval:", particles.markers[ghost_inds[:10], 0])
  
    if comm is not None:
        all_v1 = xp.zeros_like(v1)
        all_v2 = xp.zeros_like(v2)
        all_v3 = xp.zeros_like(v3)
        comm.Allreduce(v1, all_v1, op=MPI.SUM)
        comm.Allreduce(v2, all_v2, op=MPI.SUM)
        comm.Allreduce(v3, all_v3, op=MPI.SUM)
    else:
        all_v1, all_v2, all_v3 = v1, v2, v3

    v1_squeezed = all_v1.squeeze()
    v2_squeezed = all_v2.squeeze()
    v3_squeezed = all_v3.squeeze()

    v_wall_left = (v1_squeezed[0], v2_squeezed[0], v3_squeezed[0])
    v_wall_right = (v1_squeezed[-1], v2_squeezed[-1], v3_squeezed[-1])

    v_interior = (v1_squeezed[1:-1], v2_squeezed[1:-1], v3_squeezed[1:-1])

    if rank == 0:
        print("\nVelocity at interior points:")
        for idx, eta in enumerate(eta1[2:]):
            print(f"eta1 = {eta:.8f}, v_x = {v1_squeezed[2+idx]:.6f}, v_y = {v2_squeezed[2+idx]:.6f}, v_z = {v3_squeezed[2+idx]:.6f}")
    
            print(f"\nLeft wall (eta1={eta1[0]}): v_x={v1_squeezed[0]:.6f}, v_y={v2_squeezed[0]:.6f}, v_z={v3_squeezed[0]:.6f}")
            print(f"Right wall (eta1={eta1[-1]}): v_x={v1_squeezed[-1]:.6f}, v_y={v2_squeezed[-1]:.6f}, v_z={v3_squeezed[-1]:.6f}")
        
    if rank == 0 and show_plot:
        if direction == "x":
            x_plot = eta1.squeeze()
            xlabel = r'$\eta_1$'
        elif direction == "y":
            x_plot = eta2.squeeze()
            xlabel = r'$\eta_2$'
        else:  # direction == "z"
            x_plot = eta3.squeeze()
            xlabel = r'$\eta_3$'

        plt.figure(figsize=(8, 5))
        plt.plot(x_plot, v1_squeezed, 'o-', label='$v_x$')
        plt.plot(x_plot, v2_squeezed, 's-', label='$v_y$')
        plt.plot(x_plot, v3_squeezed, 'd-', label='$v_z$')
        plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
        plt.axhline(1, color='gray', linestyle='--', linewidth=0.5)
        plt.xlabel(xlabel)
        plt.ylabel('velocity')
        plt.title(f'No-slip test ({direction}-direction, {kernel}, tesselation={tesselation})')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig("bc_sph")

    if tesselation:
        tol_wall = 3e-3   
        tol_interior = 5e-2
    else:
        tol_wall = 3e-3   
        tol_interior = 1.5e-1
    
    for comp, name in zip([0, 1, 2], ["x", "y", "z"]):
        val_left = [v_wall_left[0], v_wall_left[1], v_wall_left[2]][comp]
        val_right = [v_wall_right[0], v_wall_right[1], v_wall_right[2]][comp]
        #assert xp.abs(val_left) < tol_wall, f"Left wall {name}-velocity not zero: {val_left}"
        #assert xp.abs(val_right) < tol_wall, f"Right wall {name}-velocity not zero: {val_right}"

    # The component in the chosen direction should be 1,the other two should be near zero.
    if direction == "x":
        interior_vals = v_interior[0]
        other1 = v_interior[1]
        other2 = v_interior[2]
    elif direction == "y":
        interior_vals = v_interior[1]
        other1 = v_interior[0]
        other2 = v_interior[2]
    else:  
        interior_vals = v_interior[2]
        other1 = v_interior[0]
        other2 = v_interior[1]

    rel_error = xp.max(xp.abs(interior_vals[7:-7] - 1.0)) / 1.0
    print(f"{rel_error=}")
    #assert rel_error < tol_interior, f"Interior {direction}-velocity error too large: {rel_error}"
        
    #assert xp.max(xp.abs(other1)) < tol_interior, f"Interior non‑dominant component too large: {xp.max(xp.abs(other1))}"
    #assert xp.max(xp.abs(other2)) < tol_interior, f"Interior non‑dominant component too large: {xp.max(xp.abs(other2))}"
        
if __name__ == "__main__":
    test_sph_no_slip_boundary_1d(
        (12, 1, 1),
        "gaussian_1d",
        tesselation= False,
        direction = "y",
        show_plot=True,
    )