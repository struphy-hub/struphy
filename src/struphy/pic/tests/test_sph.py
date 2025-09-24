from time import time

import numpy as np
import pytest
from matplotlib import pyplot as plt
from mpi4py import MPI

from struphy.feec.psydac_derham import Derham
from struphy.geometry import domains
from struphy.pic.particles import ParticlesSPH


@pytest.mark.parametrize("boxes_per_dim", [(24, 1, 1)])
@pytest.mark.parametrize("kernel", ["trigonometric_1d", "gaussian_1d", "linear_1d"])
@pytest.mark.parametrize("derivative", [0, 1])
@pytest.mark.parametrize("bc_x", ["periodic", "mirror", "fixed"])
@pytest.mark.parametrize("eval_pts", [11, 16])
@pytest.mark.parametrize("tesselation", [False, True])
def test_sph_evaluation_1d(boxes_per_dim, kernel, derivative, bc_x, eval_pts, tesselation, show_plot=False,):
    comm = MPI.COMM_WORLD

    # DOMAIN object
    dom_type = "Cuboid"
    dom_params = {"l1": 1.0, "r1": 2.0, "l2": 10.0, "r2": 20.0, "l3": 100.0, "r3": 200.0}
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    if tesselation:
        loading = "tesselation"
        loading_params = {"n_quad": 1}
        if kernel == "trigonometric_1d" and derivative == 1:
            ppb = 100
        else:
            ppb = 4
    else:
        loading = "pseudo_random"
        loading_params = {"seed": 223}
        if derivative == 0:
            ppb = 1000
        else:
            ppb = 20000

    # background
    cst_vel = {"density_profile": "constant", "n": 1.5}
    bckgr_params = {"ConstantVelocity": cst_vel, "pforms": ["vol", None]}

    mode_params = {"given_in_basis": "0", "ls": [1], "amps": [1e-0]}
    modes = {"ModesCos": mode_params}
    pert_params = {"n": modes}

    if derivative == 0:
        fun_exact = lambda e1, e2, e3: 1.5 + np.cos(2 * np.pi * e1)
    else:
        fun_exact = lambda e1, e2, e3: -2 * np.pi * np.sin(2 * np.pi * e1)

    # boundary conditions
    bc_sph = [bc_x, "periodic", "periodic"]
    
    # eval points
    eta1 = np.linspace(0, 1.0, eval_pts)
    eta2 = np.array([0.0])
    eta3 = np.array([0.0])

    # particles object
    particles = ParticlesSPH(
        comm_world=comm,
        ppb=ppb,
        boxes_per_dim=boxes_per_dim,
        bc_sph=bc_sph,
        bufsize=1.0,
        loading=loading,
        loading_params=loading_params,
        domain=domain,
        bckgr_params=bckgr_params,
        pert_params=pert_params,
        verbose=False,
    )

    particles.draw_markers(sort=False, verbose=False)
    particles.mpi_sort_markers()
    particles.initialize_weights()
    h1 = 1 / boxes_per_dim[0]
    h2 = 1 / boxes_per_dim[1]
    h3 = 1 / boxes_per_dim[2]
    ee1, ee2, ee3 = np.meshgrid(eta1, eta2, eta3, indexing="ij")
    test_eval = particles.eval_density(ee1, ee2, ee3, h1=h1, h2=h2, h3=h3, kernel_type=kernel, derivative=derivative,)
    all_eval = np.zeros_like(test_eval)

    comm.Allreduce(test_eval, all_eval, op=MPI.SUM)
    
    exact_eval = fun_exact(ee1, ee2, ee3)
    err_max_norm = np.max(np.abs(all_eval - exact_eval)) / np.max(np.abs(exact_eval))
    
    if comm.Get_rank() == 0:
        print(f"\n{boxes_per_dim = }")
        print(f"{kernel = }, {derivative =}")
        print(f"{bc_x = }, {eval_pts = }, {tesselation = }, {err_max_norm = }")
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
def test_sph_evaluation_2d(boxes_per_dim, kernel, derivative, bc_x, bc_y, eval_pts, show_plot=False,):
    comm = MPI.COMM_WORLD

    tesselation = True

    # DOMAIN object
    dom_type = "Cuboid"
    dom_params = {"l1": 1.0, "r1": 2.0, "l2": 0.0, "r2": 2.0, "l3": 100.0, "r3": 200.0}
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    loading = "tesselation"
    loading_params = {"n_quad": 1}
    if kernel == "trigonometric_2d" and derivative != 0:
        ppb = 100
    else:
        ppb = 16

    # background
    cst_vel = {"density_profile": "constant", "n": 1.5}
    bckgr_params = {"ConstantVelocity": cst_vel, "pforms": ["vol", None]}

    mode_params = {"given_in_basis": "0", "ls": [1], "ms": [1], "amps": [1.]}
    modes = {"ModesCosCos": mode_params}
    pert_params = {"n": modes}
    
    if derivative == 0:
        fun_exact = lambda e1, e2, e3: 1.5 + np.cos(2 * np.pi * e1) * np.cos(2 * np.pi * e2)
    elif derivative == 1:
        fun_exact = lambda e1, e2, e3: -2 * np.pi * np.sin(2 * np.pi * e1) * np.cos(2 * np.pi * e2)
    else: 
        fun_exact = lambda e1, e2, e3: -2 * np.pi * np.cos(2 * np.pi * e1) * np.sin(2 * np.pi * e2)
        
    # boundary conditions
    bc_sph = [bc_x, bc_y, "periodic"]
    
    # eval points
    eta1 = np.linspace(0, 1.0, eval_pts)
    eta2 = np.linspace(0, 1.0, eval_pts)
    eta3 = np.array([0.0])

    # particles object
    particles = ParticlesSPH(
        comm_world=comm,
        ppb=ppb,
        boxes_per_dim=boxes_per_dim,
        bc_sph=bc_sph,
        bufsize=1.0,
        loading=loading,
        loading_params=loading_params,
        domain=domain,
        bckgr_params=bckgr_params,
        pert_params=pert_params,
        verbose=False,
    )

    particles.draw_markers(sort=False, verbose=False)
    particles.mpi_sort_markers()
    particles.initialize_weights()
    h1 = 1 / boxes_per_dim[0]
    h2 = 1 / boxes_per_dim[1]
    h3 = 1 / boxes_per_dim[2]
    ee1, ee2, ee3 = np.meshgrid(eta1, eta2, eta3, indexing="ij")
    test_eval = particles.eval_density(ee1, ee2, ee3, h1=h1, h2=h2, h3=h3, kernel_type=kernel, derivative=derivative,)
    all_eval = np.zeros_like(test_eval)

    comm.Allreduce(test_eval, all_eval, op=MPI.SUM)
    
    exact_eval = fun_exact(ee1, ee2, ee3)
    err_max_norm = np.max(np.abs(all_eval - exact_eval)) / np.max(np.abs(exact_eval))
    
    if comm.Get_rank() == 0:
        print(f"\n{boxes_per_dim = }")
        print(f"{kernel = }, {derivative =}")
        print(f"{bc_x = }, {bc_y = }, {eval_pts = }, {tesselation = }, {err_max_norm = }")
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
def test_sph_evaluation_3d(boxes_per_dim, kernel, derivative, bc_x, bc_y, bc_z, eval_pts, show_plot=False,):
    comm = MPI.COMM_WORLD

    tesselation = True

    # DOMAIN object
    dom_type = "Cuboid"
    dom_params = {"l1": 1.0, "r1": 2.0, "l2": 0.0, "r2": 2.0, "l3": -1.0, "r3": 2.0}
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    loading = "tesselation"
    loading_params = {"n_quad": 1}
    if kernel in ("trigonometric_3d", "linear_isotropic_3d") and derivative != 0:
        ppb = 100
    else:
        ppb = 64

    # background
    cst_vel = {"density_profile": "constant", "n": 1.5}
    bckgr_params = {"ConstantVelocity": cst_vel, "pforms": ["vol", None]}
    
    if derivative == 0:
        fun_exact = lambda e1, e2, e3: 1.5 + 0.0*e1
    else: 
        fun_exact = lambda e1, e2, e3: 0.0*e1
        
    # boundary conditions
    bc_sph = [bc_x, bc_y, bc_z]
    
    # eval points
    eta1 = np.linspace(0, 1.0, eval_pts)
    eta2 = np.linspace(0, 1.0, eval_pts)
    eta3 = np.linspace(0, 1.0, eval_pts)

    # particles object
    particles = ParticlesSPH(
        comm_world=comm,
        ppb=ppb,
        boxes_per_dim=boxes_per_dim,
        bc_sph=bc_sph,
        bufsize=2.0,
        loading=loading,
        loading_params=loading_params,
        domain=domain,
        bckgr_params=bckgr_params,
        # pert_params=pert_params,
        verbose=False,
    )

    particles.draw_markers(sort=False, verbose=False)
    particles.mpi_sort_markers()
    particles.initialize_weights()
    h1 = 1 / boxes_per_dim[0]
    h2 = 1 / boxes_per_dim[1]
    h3 = 1 / boxes_per_dim[2]
    ee1, ee2, ee3 = np.meshgrid(eta1, eta2, eta3, indexing="ij")
    test_eval = particles.eval_density(ee1, ee2, ee3, h1=h1, h2=h2, h3=h3, kernel_type=kernel, derivative=derivative,)
    all_eval = np.zeros_like(test_eval)

    comm.Allreduce(test_eval, all_eval, op=MPI.SUM)
    
    exact_eval = fun_exact(ee1, ee2, ee3)
    err_max_norm = np.max(np.abs(all_eval - exact_eval))
    
    if comm.Get_rank() == 0:
        print(f"\n{boxes_per_dim = }")
        print(f"{kernel = }, {derivative =}")
        print(f"{bc_x = }, {bc_y = }, {bc_z = }, {eval_pts = }, {tesselation = }, {err_max_norm = }")
        if show_plot:
            print(f"\n{fun_exact(ee1, ee2, ee3)[5, 5, 5] = }")
            print(f"{ee1[5, 5, 5] = }, {ee2[5, 5, 5] = }, {ee3[5, 5, 5] = }")
            print(f"{all_eval[5, 5, 5] = }")
            
            print(f"\n{ee1[4, 4, 4] = }, {ee2[4, 4, 4] = }, {ee3[4, 4, 4] = }")
            print(f"{all_eval[4, 4, 4] = }")
            
            print(f"\n{ee1[3, 3, 3] = }, {ee2[3, 3, 3] = }, {ee3[3, 3, 3] = }")
            print(f"{all_eval[3, 3, 3] = }")
            
            print(f"\n{ee1[2, 2, 2] = }, {ee2[2, 2, 2] = }, {ee3[2, 2, 2] = }")
            print(f"{all_eval[2, 2, 2] = }")
            
            print(f"\n{ee1[1, 1, 1] = }, {ee2[1, 1, 1] = }, {ee3[1, 1, 1] = }")
            print(f"{all_eval[1, 1, 1] = }")
            
            print(f"\n{ee1[0, 0, 0] = }, {ee2[0, 0, 0] = }, {ee3[0, 0, 0] = }")
            print(f"{all_eval[0, 0, 0] = }")
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


# @pytest.mark.mpi(min_size=2)
# @pytest.mark.parametrize("boxes_per_dim", [(8, 1, 1), (10, 1, 1), (24, 1, 1)])
# @pytest.mark.parametrize("bc_x", ["periodic", "mirror", "fixed"])
# @pytest.mark.parametrize("tesselation", [False, True])
# def test_evaluation_SPH_Np_convergence_1d(boxes_per_dim, bc_x, tesselation, show_plot=False):
#     comm = MPI.COMM_WORLD

#     # DOMAIN object
#     dom_type = "Cuboid"
#     dom_params = {"l1": 0.0, "r1": 3.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}
#     domain_class = getattr(domains, dom_type)
#     domain = domain_class(**dom_params)

#     if tesselation:
#         loading = "tesselation"
#         loading_params = {"n_quad": 1}
#         # ppbs = [5000, 10000, 15000, 20000, 25000]
#         ppbs = [4, 8, 16, 32, 64]
#         Nps = [None] * len(ppbs)
#     else:
#         loading = "pseudo_random"
#         loading_params = {"seed": 1607}
#         Nps = [(2**k) * 10**3 for k in range(-2, 9)]
#         ppbs = [None] * len(Nps)

#     # background
#     cst_vel = {"density_profile": "constant", "n": 1.5}
#     bckgr_params = {"ConstantVelocity": cst_vel, "pforms": ["vol", None]}

#     # perturbation
#     mode_params = {"given_in_basis": "0", "ls": [1], "amps": [-1e-0]}
#     if bc_x in ("periodic", "fixed"):
#         fun_exact = lambda e1, e2, e3: 1.5 - np.sin(2 * np.pi * e1)
#         modes = {"ModesSin": mode_params}
#     elif bc_x == "mirror":
#         fun_exact = lambda e1, e2, e3: 1.5 - np.cos(2 * np.pi * e1)
#         modes = {"ModesCos": mode_params}
#     pert_params = {"n": modes}
    
#     # exact solution
#     eta1 = np.linspace(0, 1.0, 100)  # add offset for non-periodic boundary conditions, TODO: implement Neumann
#     eta2 = np.array([0.0])
#     eta3 = np.array([0.0])
#     ee1, ee2, ee3 = np.meshgrid(eta1, eta2, eta3, indexing="ij")
#     exact_eval = fun_exact(ee1, ee2, ee3)

#     # loop
#     err_vec = []
#     for Np, ppb in zip(Nps, ppbs):
#         particles = ParticlesSPH(
#             comm_world=comm,
#             Np=Np,
#             ppb=ppb,
#             boxes_per_dim=boxes_per_dim,
#             bc_sph=[bc_x, "periodic", "periodic"],
#             bufsize=1.0,
#             loading=loading,
#             loading_params=loading_params,
#             domain=domain,
#             bckgr_params=bckgr_params,
#             pert_params=pert_params,
#             verbose=False,
#         )

#         particles.draw_markers(sort=False, verbose=False)
#         particles.mpi_sort_markers()
#         particles.initialize_weights()
#         h1 = 1 / boxes_per_dim[0]
#         h2 = 1 / boxes_per_dim[1]
#         h3 = 1 / boxes_per_dim[2]
        
#         test_eval = particles.eval_density(ee1, ee2, ee3, h1=h1, h2=h2, h3=h3)
#         all_eval = np.zeros_like(test_eval)

#         comm.Allreduce(test_eval, all_eval, op=MPI.SUM)

#         if show_plot and comm.Get_rank() == 0:
#             plt.figure()
#             plt.plot(ee1.squeeze(), exact_eval.squeeze(), label="exact")
#             plt.plot(ee1.squeeze(), all_eval.squeeze(), "--.", label="eval_sph")
#             plt.savefig(f"fun_{Np}_{ppb}.png")

#         diff = np.max(np.abs(all_eval - exact_eval)) / np.max(np.abs(exact_eval))
#         err_vec += [diff]
#         print(f"{Np = }, {ppb = }, {diff = }")

#         if tesselation:
#             assert diff < 0.018

#     if tesselation:
#         fit = np.polyfit(np.log(ppbs), np.log(err_vec), 1)
#         xvec = ppbs
#     else:
#         fit = np.polyfit(np.log(Nps), np.log(err_vec), 1)
#         xvec = Nps
#     print(fit)

#     if show_plot and comm.Get_rank() == 0:
#         plt.figure(figsize=(12, 8))
#         plt.loglog(xvec, err_vec, label="Convergence")
#         plt.loglog(xvec, np.exp(fit[1]) * np.array(xvec) ** (fit[0]), "--", label=f"fit with slope {fit[0]}")
#         plt.legend()
#         plt.show()
#         plt.savefig(f"Convergence_SPH_{tesselation=}")

#     if not tesselation:
#         assert np.abs(fit[0] + 0.5) < 0.1

#     err_min = np.min(err_vec)

#     if comm.Get_rank() == 0:
#         print(f"{tesselation = }, {bc_x = }, {err_min = }")

#     if tesselation:
#         if bc_x == "periodic":
#             assert err_min < 0.0069
#         elif bc_x == "fixed":
#             assert err_min < 0.0069
#         else:
#             assert err_min < 0.0069
#     else:
#         if bc_x in ("periodic", "fixed"):
#             assert err_min < 0.012
#         else:
#             assert err_min < 0.024


# @pytest.mark.mpi(min_size=2)
# @pytest.mark.parametrize("boxes_per_dim", [(8, 1, 1), (16, 1, 1), (24, 1, 1)])
# @pytest.mark.parametrize("bc_x", ["periodic", "fixed", "mirror"])
# @pytest.mark.parametrize("tesselation", [False, True])
# def test_evaluation_SPH_h_convergence_1d(boxes_per_dim, bc_x, tesselation, show_plot=False):
#     comm = MPI.COMM_WORLD

#     # DOMAIN object
#     dom_type = "Cuboid"
#     dom_params = {"l1": 0.0, "r1": 3.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}
#     domain_class = getattr(domains, dom_type)
#     domain = domain_class(**dom_params)

#     if tesselation:
#         loading = "tesselation"
#         loading_params = {"seed": 1607}
#         Np = None
#         ppb = 8
#     else:
#         loading = "pseudo_random"
#         loading_params = {"seed": 1607}
#         Np = 50000
#         ppb = None

#     cst_vel = {"density_profile": "constant", "n": 1.5}
#     bckgr_params = {"ConstantVelocity": cst_vel, "pforms": ["vol", None]}

#     # perturbation 
#     mode_params = {"given_in_basis": "0", "ls": [1], "amps": [-1e-0]}
#     if bc_x in ("periodic", "fixed"):
#         fun_exact = lambda e1, e2, e3: 1.5 - np.sin(2 * np.pi * e1)
#         modes = {"ModesSin": mode_params}
#     elif bc_x == "mirror":
#         fun_exact = lambda e1, e2, e3: 1.5 - np.cos(2 * np.pi * e1)
#         modes = {"ModesCos": mode_params}
#     pert_params = {"n": modes}
    
#     # exact solution
#     eta1 = np.linspace(0, 1.0, 100)  # add offset for non-periodic boundary conditions, TODO: implement Neumann
#     eta2 = np.array([0.0])
#     eta3 = np.array([0.0])
#     ee1, ee2, ee3 = np.meshgrid(eta1, eta2, eta3, indexing="ij")
#     exact_eval = fun_exact(ee1, ee2, ee3)

#     # parameters
#     h_vec = [((2**k) * 10**-3 * 0.25) for k in range(2, 12)]
#     err_vec = []
#     for h1 in h_vec:
#         particles = ParticlesSPH(
#             comm_world=comm,
#             Np=Np,
#             ppb=ppb,
#             boxes_per_dim=boxes_per_dim,
#             bc_sph=[bc_x, "periodic", "periodic"],
#             bufsize=1.0,
#             loading=loading,
#             loading_params=loading_params,
#             domain=domain,
#             bckgr_params=bckgr_params,
#             pert_params=pert_params,
#             verbose=False,
#         )

#         particles.draw_markers(sort=False, verbose=False)
#         particles.mpi_sort_markers()
#         particles.initialize_weights()
#         h2 = 1 / boxes_per_dim[1]
#         h3 = 1 / boxes_per_dim[2]
        
#         test_eval = particles.eval_density(ee1, ee2, ee3, h1=h1, h2=h2, h3=h3)
#         all_eval = np.zeros_like(test_eval)

#         comm.Allreduce(test_eval, all_eval, op=MPI.SUM)

#         if show_plot and comm.Get_rank() == 0:
#             plt.figure()
#             plt.plot(ee1.squeeze(), exact_eval.squeeze(), label="exact")
#             plt.plot(ee1.squeeze(), all_eval.squeeze(), "--.", label="eval_sph")
#             plt.savefig(f"fun_{h1}.png")

#         # error in max-norm
#         diff = np.max(np.abs(all_eval - exact_eval)) / np.max(np.abs(exact_eval))
#         print(f"{h1 = }, {diff = }")
#         err_vec += [diff]

#     fit = np.polyfit(np.log(h_vec), np.log(err_vec), 1)
#     print(f"{fit = }")
#     err_min = np.min(err_vec)
#     print(f"{err_min = }")

#     if show_plot and comm.Get_rank() == 0:
#         plt.figure(figsize=(12, 8))
#         plt.loglog(h_vec, err_vec, label="Convergence")
#         plt.loglog(h_vec, np.exp(fit[1]) * np.array(h_vec) ** (fit[0]), "--", label=f"fit with slope {fit[0]}")
#         plt.legend()
#         plt.show()
#         plt.savefig("Convergence_SPH")

#     if comm.Get_rank() == 0:
#         print(f"{tesselation = }, {bc_x = }, {err_min = }")

#     if tesselation:
#         if bc_x == "periodic":
#             assert err_min < 0.0019
#         elif bc_x == "fixed":
#             assert err_min < 0.0019
#         else:
#             assert err_min < 0.0019
#     else:
#         if bc_x in ("periodic", "fixed"):
#             assert err_min < 0.042
#         else:
#             assert err_min < 0.051


# @pytest.mark.mpi(min_size=2)
# @pytest.mark.parametrize("boxes_per_dim", [(8, 1, 1), (24, 1, 1)])
# @pytest.mark.parametrize("bc_x", ["periodic", "fixed", "mirror"])
# @pytest.mark.parametrize("tesselation", [False, True])
# def test_evaluation_mc_Np_and_h_convergence_1d(boxes_per_dim, bc_x, tesselation, show_plot=False):
#     comm = MPI.COMM_WORLD

#     # DOMAIN object
#     dom_type = "Cuboid"
#     dom_params = {"l1": 0.0, "r1": 3.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}
#     domain_class = getattr(domains, dom_type)
#     domain = domain_class(**dom_params)

#     if tesselation:
#         loading = "tesselation"
#         loading_params = {"n_quad": 1}
#         # ppbs = [5000, 10000, 15000, 20000, 25000]
#         ppbs = [4, 8, 16, 32, 64]
#         Nps = [None] * len(ppbs)

#     else:
#         loading = "pseudo_random"
#         loading_params = {"seed": 1607}
#         Nps = [(2**k) * 10**3 for k in range(-2, 9)]
#         ppbs = [None] * len(Nps)

#     cst_vel = {"density_profile": "constant", "n": 1.5}
#     bckgr_params = {"ConstantVelocity": cst_vel, "pforms": ["vol", None]}

#     # perturbation 
#     mode_params = {"given_in_basis": "0", "ls": [1], "amps": [-1e-0]}
#     if bc_x in ("periodic", "fixed"):
#         fun_exact = lambda e1, e2, e3: 1.5 - np.sin(2 * np.pi * e1)
#         modes = {"ModesSin": mode_params}
#     elif bc_x == "mirror":
#         fun_exact = lambda e1, e2, e3: 1.5 - np.cos(2 * np.pi * e1)
#         modes = {"ModesCos": mode_params}
#     pert_params = {"n": modes}

#     # exact solution
#     eta1 = np.linspace(0, 1.0, 100)
#     eta2 = np.array([0.0])
#     eta3 = np.array([0.0])
#     ee1, ee2, ee3 = np.meshgrid(eta1, eta2, eta3, indexing="ij")
#     exact_eval = fun_exact(ee1, ee2, ee3)

#     h_arr = [((2**k) * 10**-3 * 0.25) for k in range(2, 12)]
#     err_vec = []
#     for h in h_arr:
#         err_vec += [[]]
#         for Np, ppb in zip(Nps, ppbs):
#             particles = ParticlesSPH(
#                 comm_world=comm,
#                 Np=Np,
#                 ppb=ppb,
#                 boxes_per_dim=boxes_per_dim,
#                 bc_sph=[bc_x, "periodic", "periodic"],
#                 bufsize=1.0,
#                 loading=loading,
#                 loading_params=loading_params,
#                 domain=domain,
#                 bckgr_params=bckgr_params,
#                 pert_params=pert_params,
#                 verbose=False,
#             )

#             particles.draw_markers(sort=False, verbose=False)
#             particles.mpi_sort_markers()
#             particles.initialize_weights()

#             h2 = 1 / boxes_per_dim[1]
#             h3 = 1 / boxes_per_dim[2]

#             test_eval = particles.eval_density(ee1, ee2, ee3, h1=h, h2=h2, h3=h3)
#             all_eval = np.zeros_like(test_eval)
#             comm.Allreduce(test_eval, all_eval, op=MPI.SUM)

#             if show_plot and comm.Get_rank() == 0:
#                 plt.figure()
#                 plt.plot(ee1.squeeze(), fun_exact(ee1, ee2, ee3).squeeze(), label="exact")
#                 plt.plot(ee1.squeeze(), all_eval.squeeze(), "--.", label="eval_sph")
#                 plt.title(f"{h = }, {Np = }")
#                 plt.savefig(f"fun_h{h}_N{Np}_ppb{ppb}.png")

#             # error in max-norm
#             diff = np.max(np.abs(all_eval - exact_eval)) / np.max(np.abs(exact_eval))
#             err_vec[-1] += [diff]
#             print(f"{Np = }, {ppb = }, {diff = }")

#     err_vec = np.array(err_vec)
#     err_min = np.min(err_vec)
#     print(f"\n{err_min = }")

#     if show_plot and comm.Get_rank() == 0:
#         if tesselation:
#             h_mesh, n_mesh = np.meshgrid(np.log10(h_arr), np.log10(ppbs), indexing="ij")
#         if not tesselation:
#             h_mesh, n_mesh = np.meshgrid(np.log10(h_arr), np.log10(Nps), indexing="ij")
#         plt.figure(figsize=(6, 6))
#         plt.pcolor(h_mesh, n_mesh, np.log10(err_vec), shading="auto")
#         plt.title("Error")
#         plt.colorbar(label="log10(error)")
#         plt.xlabel("log10(h)")
#         plt.ylabel("log10(particles)")

#         min_indices = np.argmin(err_vec, axis=0)
#         min_h_values = []
#         for mi in min_indices:
#             min_h_values += [np.log10(h_arr[mi])]
#         if tesselation:
#             log_particles = np.log10(ppbs)
#         else:
#             log_particles = np.log10(Nps)
#         plt.plot(min_h_values, log_particles, "r-", label="Min error h for each Np", linewidth=2)
#         plt.legend()
#         plt.savefig("SPH_conv_in_h_and_N.png")

#         plt.show()

#     if comm.Get_rank() == 0:
#         print(f"{tesselation = }, {bc_x = }, {err_min = }")

#     if tesselation:
#         if bc_x == "periodic":
#             assert np.min(err_vec) < 7.7e-5
#         elif bc_x == "fixed":
#             assert err_min < 7.7e-5
#         else:
#             assert err_min < 7.7e-5
#     else:
#         if bc_x in ("periodic", "fixed"):
#             assert err_min < 0.0089
#         else:
#             assert err_min < 0.021


# # @pytest.mark.mpi(min_size=2)
# @pytest.mark.parametrize("boxes_per_dim", [(32, 32, 1)])
# @pytest.mark.parametrize("bc_x", ["periodic", "fixed", "mirror"])
# @pytest.mark.parametrize("bc_y", ["periodic", "fixed", "mirror"])
# @pytest.mark.parametrize("tesselation", [False, True])
# def test_evaluation_SPH_Np_convergence_2d(boxes_per_dim, bc_x, bc_y, tesselation, show_plot=False):

#     comm = MPI.COMM_WORLD

#     # DOMAIN object
#     dom_type = "Cuboid"
    
#     Lx = 1.0
#     Ly = 1.0
#     dom_params = {"l1": 0.0, "r1": Lx, "l2": 0.0, "r2": Ly, "l3": 0.0, "r3": 1.0}
#     domain_class = getattr(domains, dom_type)
#     domain = domain_class(**dom_params)

#     if tesselation:
#         loading = "tesselation"
#         loading_params = {"n_quad": 1}
#         ppbs = [100]#4, 8, 16, 32, 64, 200]
#         Nps = [None] * len(ppbs)
#     else:
#         loading = "pseudo_random"
#         loading_params = {"seed": 1607}
#         Nps = [(2**k) * 10**3 for k in range(-2, 9)]
#         ppbs = [None] * len(Nps)

#     cst_vel = {"density_profile": "constant", "n": 1.5}
#     bckgr_params = {"ConstantVelocity": cst_vel, "pforms": ["vol", None]}

#     # perturbation
#     mode_params = {"given_in_basis": "0", "ls": [1], "ms": [1], "amps": [-1e-0]}

#     if bc_x in ("periodic", "fixed"):
#         if bc_y in ("periodic", "fixed"):
#             fun_exact = lambda x, y, z: 1.5 - np.sin(2 * np.pi/Lx * x) * np.sin(2 * np.pi/Ly * y)
#             modes = {"ModesSinSin": mode_params}
#         elif bc_y == "mirror":
#             fun_exact = lambda x, y, z: 1.5 - np.sin(2 * np.pi/Lx * x) * np.cos(2 * np.pi/Ly * y)
#             modes = {"ModesSinCos": mode_params}

#     elif bc_x == "mirror":
#         if bc_y in ("periodic", "fixed"):
#             fun_exact = lambda x, y, z: 1.5 - np.cos(2 * np.pi/Lx * x) * np.sin(2 * np.pi/Ly * y)
#             modes = {"ModesCosSin": mode_params}
#         elif bc_y == "mirror":
#             fun_exact = lambda x, y, z: 1.5 - np.cos(2 * np.pi/Lx * x) * np.cos(2 * np.pi/Ly * y)
#             modes = {"ModesCosCos": mode_params}

#     pert_params = {"n": modes}
    
#     # exact solution
#     eta1 = np.linspace(0, 1.0, 11)
#     eta2 = np.linspace(0, 1.0, 11)
#     eta3 = np.array([0.0])
#     ee1, ee2, ee3 = np.meshgrid(eta1, eta2, eta3, indexing="ij")
#     x, y, z = domain(eta1, eta2, eta3)
#     exact_eval = fun_exact(x, y, z)

#     # bckgr = GenericCartesianFluidEquilibrium(n_xyz=n_fun)
#     # bckgr.domain = domain

#     err_vec = []
#     for Np, ppb in zip(Nps, ppbs):
#         particles = ParticlesSPH(
#             comm_world=comm,
#             Np=Np,
#             ppb=ppb,
#             boxes_per_dim=boxes_per_dim,
#             bc_sph=[bc_x, bc_y, "periodic"],
#             bufsize=1.0,
#             box_bufsize=4.0,
#             loading=loading,
#             loading_params=loading_params,
#             domain=domain,
#             bckgr_params=bckgr_params,
#             # pert_params=pert_params,
#             verbose=False,
#             mpi_dims_mask=[True, False, False],
#         )
#         if comm.Get_rank() == 0:
#             print(f"{particles.domain_array}")
        
#         particles.draw_markers(sort=False, verbose=False)
#         particles.mpi_sort_markers()
#         particles.initialize_weights()
#         h1 = 1 / boxes_per_dim[0]
#         h2 = 1 / boxes_per_dim[1]
#         h3 = 1 / boxes_per_dim[2]
        
#         test_eval = particles.eval_density(ee1, ee2, ee3, h1=h1, h2=h2, h3=h3, kernel_type="gaussian_2d")
#         all_eval = np.zeros_like(test_eval)

#         comm.Allreduce(test_eval, all_eval, op=MPI.SUM)

#         if comm.Get_rank() == 0:
#             print(f"{all_eval.squeeze().shape}")
#             print(f"{all_eval.squeeze()[0]}")
#             print(f"{all_eval.squeeze().T[0]}")

#         if show_plot and comm.Get_rank() == 0:
#             fig, ax = plt.subplots()
#             d = ax.pcolor(ee1.squeeze(), ee2.squeeze(), all_eval.squeeze(), label="eval_sph", vmin=1.0, vmax=2.0)
#             fig.colorbar(d, ax=ax, label="2d_SPH")
#             ax.set_xlabel("ee1")
#             ax.set_ylabel("ee2")
#             ax.set_title(f"{Np}_{ppb = }")

#             fig.savefig(f"2d_sph_{Np}_{ppb}.png")

#         # error in max-norm
#         diff = np.max(np.abs(all_eval - exact_eval)) / np.max(np.abs(exact_eval))
#         err_vec += [diff]
#         print(f"{Np = }, {ppb = }, {diff = }")

#     err_min = np.min(err_vec)

#     if show_plot and comm.Get_rank() == 0:
#         plt.figure(figsize=(12, 8))
#         print(f"{Nps = }")
#         print(f"{ppbs = }")
#         print(f"{err_vec = }")
#         if tesselation:
#             plt.xlabel("ppb")
#             plt.loglog(ppbs, err_vec, label="Convergence")
#         else:
#             plt.xlabel("Np")
#             plt.loglog(Nps, err_vec, label="Convergence")
#         plt.ylabel("max-norm error")
#         # plt.loglog(ppbs, np.exp(fit[1])*np.array(ppbs)**(fit[0]), "--", label = f"fit with slope {fit[0]}")
#         plt.legend()
#         plt.savefig("Convergence_SPH_2d_tesselation")

#         plt.show()

#     if comm.Get_rank() == 0:
#         print(f"{tesselation = }, {bc_x = }, {err_min = }")

#     # if tesselation:
#     #     if bc_x in ("periodic", "fixed") and bc_y in ("periodic", "fixed"):
#     #         assert err_min < 0.002
#     #     elif bc_x in ("periodic", "fixed") and bc_y == "mirror":
#     #         assert err_min < 0.002
#     #     elif bc_x == "mirror" and bc_y in ("periodic", "fixed"):
#     #         assert err_min < 0.002
#     #     elif bc_x == "mirror" and bc_y == "mirror":
#     #         assert err_min < 0.002
#     # else:
#     #     if bc_x in ("periodic", "fixed") and bc_y in ("periodic", "fixed"):
#     #         assert err_min < 0.23
#     #     elif bc_x in ("periodic", "fixed") and bc_y == "mirror":
#     #         assert err_min < 0.25
#     #     elif bc_x == "mirror" and bc_y in ("periodic", "fixed"):
#     #         assert err_min < 0.25
#     #     elif bc_x == "mirror" and bc_y == "mirror":
#     #         assert err_min < 0.3


if __name__ == "__main__":
    # test_sph_evaluation_1d(
    #     (24, 1, 1),
    #     "trigonometric_1d",
    #     # "gaussian_1d",
    #     1,
    #     "periodic",
    #     # "mirror",
    #     10,
    #     tesselation=True,
    #     show_plot=True
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
    
    test_sph_evaluation_3d(
        (12, 8, 8),
        # "trigonometric_2d",
        "gaussian_3d",
        2,
        "periodic",
        "periodic",
        "periodic",
        11,
        show_plot=True
    )

    # for nb in range(4, 25):
    #     print(f"\n{nb = }")
    # test_evaluation_SPH_Np_convergence_1d((24,1,1), "fixed", tesselation=True, show_plot=True)
    # test_evaluation_SPH_h_convergence_1d((8,1,1), "periodic", tesselation = True, show_plot=False)
    # test_evaluation_mc_Np_and_h_convergence_1d((16,1,1),"mirror",tesselation = False,  show_plot=False)
    # test_evaluation_SPH_Np_convergence_2d((24, 24, 1), "periodic", "periodic",  tesselation=True, show_plot=True)
    # test_evaluation_SPH_Np_convergence_2d((16, 16, 1), "periodic", "fixed", tesselation=True, show_plot=True)
    # test_evaluation_SPH_Np_convergence_2d((32, 32, 1), "fixed", "periodic", tesselation=True, show_plot=True)
    # test_evaluation_SPH_Np_convergence_2d((32, 32, 1), "fixed", "fixed",   tesselation=True, show_plot=True)
    # test_evaluation_SPH_Np_convergence_2d((32, 32, 1), "mirror", "mirror",  tesselation=True, show_plot=True)
