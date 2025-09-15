from time import time

import numpy as np
import pytest
from mpi4py import MPI
from matplotlib import pyplot as plt

from struphy.feec.psydac_derham import Derham
from struphy.geometry import domains
from struphy.pic.particles import ParticlesSPH


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("Np", [40000])
@pytest.mark.parametrize("boxes_per_dim", [(8, 1, 1), (10, 1, 1)])
@pytest.mark.parametrize("ppb", [4, 10])
@pytest.mark.parametrize("bc_x", ["periodic", "mirror", "fixed"])
@pytest.mark.parametrize("tesselation", [False, True])
def test_sph_evaluation(Np, boxes_per_dim, ppb, bc_x, tesselation, show_plot=False):
    comm = MPI.COMM_WORLD

    # DOMAIN object
    dom_type = "Cuboid"
    dom_params = {"l1": 1.0, "r1": 2.0, "l2": 10.0, "r2": 20.0, "l3": 100.0, "r3": 200.0}
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    if tesselation:
        loading = "tesselation"
        loading_params = {"n_quad": 1}
        Np = None
    else:
        loading = "pseudo_random"
        loading_params = {"seed": 1607}
        ppb = None

    # background
    cst_vel = {"density_profile": "constant", "n": 1.5}
    bckgr_params = {"ConstantVelocity": cst_vel, "pforms": ["vol", None]}
    
    # perturbation and exact solution
    mode_params = {"given_in_basis": "0", "ls": [1], "amps": [-1e-0]}
    if bc_x in ("periodic", "fixed"):
        fun_exact = lambda e1, e2, e3: 1.5 - np.sin(2 * np.pi * e1)
        modes = {"ModesSin": mode_params}
    elif bc_x == "mirror":
        fun_exact = lambda e1, e2, e3: 1.5 - np.cos(2 * np.pi * e1)
        modes = {"ModesCos": mode_params}
    pert_params = {"n": modes}

    # particles object
    particles = ParticlesSPH(
        comm_world=comm,
        Np=Np,
        ppb=ppb,
        boxes_per_dim=boxes_per_dim,
        bc_sph=[bc_x, "periodic", "periodic"],
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
    eta1 = np.linspace(0, 1.0, 100)  # add offset for non-periodic boundary conditions, TODO: implement Neumann
    eta2 = np.array([0.0])
    eta3 = np.array([0.0])
    ee1, ee2, ee3 = np.meshgrid(eta1, eta2, eta3, indexing="ij")
    test_eval = particles.eval_density(ee1, ee2, ee3, h1=h1, h2=h2, h3=h3)
    all_eval = np.zeros_like(test_eval)

    comm.Allreduce(test_eval, all_eval, op=MPI.SUM)

    if show_plot and comm.Get_rank() == 0:
        plt.figure(figsize=(12, 8))
        plt.plot(ee1.squeeze(), fun_exact(ee1, ee2, ee3).squeeze(), label="exact")
        plt.plot(ee1.squeeze(), all_eval.squeeze(), "--.", label="eval_sph")
        plt.legend()
        plt.show()

    exact_eval = fun_exact(ee1, ee2, ee3)
    err_max_norm = np.max(np.abs(all_eval - exact_eval)) / np.max(np.abs(exact_eval))
    
    if comm.Get_rank() == 0:
        print(f'{tesselation = }, {bc_x = }, {err_max_norm = }')
        
    if tesselation:
        if bc_x == "periodic":
            assert err_max_norm < 0.0031
        elif bc_x == "fixed":
            assert err_max_norm < 0.0031
        else:
            assert err_max_norm < 0.0031
    else:
        if bc_x in ("periodic", "fixed"):
            assert err_max_norm < 0.0366
        else: 
            assert err_max_norm < 0.046


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("boxes_per_dim", [(8, 1, 1), (10, 1, 1)])
@pytest.mark.parametrize("bc_x", ["periodic", "mirror", "fixed"])
@pytest.mark.parametrize("tesselation", [False, True])
def test_evaluation_SPH_Np_convergence_1d(boxes_per_dim, bc_x, tesselation, show_plot=False):
    comm = MPI.COMM_WORLD

    # DOMAIN object
    dom_type = "Cuboid"
    dom_params = {"l1": 0.0, "r1": 3.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)
    
    if tesselation: 
        loading = "tesselation"
        loading_params = {"n_quad": 1}
        # ppbs = [5000, 10000, 15000, 20000, 25000]
        ppbs = [4, 8, 16, 32, 64]
        Nps = [None]*len(ppbs)
    else: 
        loading = "pseudo_random"
        loading_params = {"seed": 1607}
        Nps = [(2**k)*10**3 for k in range(-2, 9)]
        ppbs = [None]*len(Nps)

    # background
    cst_vel = {"density_profile": "constant", "n": 1.5}
    bckgr_params = {"ConstantVelocity": cst_vel, "pforms": ["vol", None]}

    # perturbation and exact solution
    mode_params = {"given_in_basis": "0", "ls": [1], "amps": [-1e-0]}
    if bc_x in ("periodic", "fixed"):
        fun_exact = lambda e1, e2, e3: 1.5 - np.sin(2 * np.pi * e1)
        modes = {"ModesSin": mode_params}
    elif bc_x == "mirror":
        fun_exact = lambda e1, e2, e3: 1.5 - np.cos(2 * np.pi * e1)
        modes = {"ModesCos": mode_params}
    pert_params = {"n": modes}
    
    # loop
    err_vec = []
    for Np, ppb in zip(Nps, ppbs): 
        particles = ParticlesSPH(
            comm_world=comm,
            Np=Np,
            ppb=ppb, 
            boxes_per_dim=boxes_per_dim,
            bc_sph=[bc_x, "periodic", "periodic"],
            bufsize=1.0,
            loading=loading,
            loading_params=loading_params,
            domain=domain,
            bckgr_params=bckgr_params,
            pert_params=pert_params,
            verbose = False, 
            )
        
        particles.draw_markers(sort=False, verbose=False)
        particles.mpi_sort_markers()
        particles.initialize_weights()
        h1 = 1 / boxes_per_dim[0]
        h2 = 1 / boxes_per_dim[1]
        h3 = 1 / boxes_per_dim[2]
        eta1 = np.linspace(0, 1.0, 100)  # add offset for non-periodic boundary conditions, TODO: implement Neumann
        eta2 = np.array([0.0])
        eta3 = np.array([0.0])
        ee1, ee2, ee3 = np.meshgrid(eta1, eta2, eta3, indexing="ij")
        test_eval = particles.eval_density(ee1, ee2, ee3, h1=h1, h2=h2, h3=h3)
        all_eval = np.zeros_like(test_eval)
        
        comm.Allreduce(test_eval, all_eval, op=MPI.SUM)
        
        if show_plot and comm.Get_rank() == 0:
            plt.figure()
            plt.plot(ee1.squeeze(), fun_exact(ee1, ee2, ee3).squeeze(), label="exact")
            plt.plot(ee1.squeeze(), all_eval.squeeze(), "--.", label="eval_sph")
            plt.savefig(f"fun_{Np}_{ppb}.png")
            
        diff = np.max(np.abs(all_eval - fun_exact(ee1,ee2,ee3)))
        err_vec += [diff]
        print(f'{Np = }, {ppb = }, {diff = }')
        
        if tesselation:
            assert diff < 0.018
        
    
    if tesselation:    
        fit = np.polyfit(np.log(ppbs), np.log(err_vec), 1)
        xvec = ppbs
    else:
        fit = np.polyfit(np.log(Nps), np.log(err_vec), 1)
        xvec = Nps
    print(fit)

    if show_plot and comm.Get_rank() == 0:
        plt.figure(figsize=(12, 8))
        plt.loglog(xvec, err_vec, label = "Convergence")
        plt.loglog(xvec, np.exp(fit[1])*np.array(xvec)**(fit[0]), "--", label = f"fit with slope {fit[0]}")
        plt.legend() 
        plt.show()
        plt.savefig(f"Convergence_SPH_{tesselation=}")

    if not tesselation:
        assert np.abs(fit[0] + 0.5) < 0.1
    
    exact_eval = fun_exact(ee1, ee2, ee3)
    err_max_norm = np.max(np.abs(all_eval - exact_eval)) / np.max(np.abs(exact_eval))
    
    if comm.Get_rank() == 0:
        print(f'{tesselation = }, {bc_x = }, {err_max_norm = }')
        
    if tesselation:
        if bc_x == "periodic":
            assert err_max_norm < 0.004
        elif bc_x == "fixed":
            assert err_max_norm < 0.004
        else:
            assert err_max_norm < 0.004
    else:
        if bc_x in ("periodic", "fixed"):
            assert err_max_norm < 0.008
        else: 
            assert err_max_norm < 0.017
    

@pytest.mark.mpi(min_size=2)
#@pytest.mark.parametrize("Np", [40000, 46200])
@pytest.mark.parametrize("boxes_per_dim", [(8, 1, 1), (16, 1, 1)])
#@pytest.mark.parametrize("ppb", [4, 10])
@pytest.mark.parametrize("bc_x", ["periodic", "fixed", "mirror"])
@pytest.mark.parametrize("tesselation", [False, True])
def test_evaluation_SPH_h_convergence_1d(boxes_per_dim, bc_x, tesselation, show_plot=False):
    comm = MPI.COMM_WORLD

    # DOMAIN object
    dom_type = "Cuboid"
    dom_params = {"l1": 0.0, "r1": 1.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)
    
    if tesselation: 
        loading = "tesselation"
        loading_params = {"seed": 1607}
        Np = None
        ppb = 8
    else: 
        loading = "pseudo_random"
        loading_params = {"seed": 1607}
        Np = 50000
        ppb = None
        
    cst_vel = {"density_profile": "constant", "n": 1.5}
    bckgr_params = {"ConstantVelocity": cst_vel, "pforms": ["vol", None]}

    # perturbation and exact solution
    mode_params = {"given_in_basis": "0", "ls": [1], "amps": [-1e-0]}
    if bc_x in ("periodic", "fixed"):
        fun_exact = lambda e1, e2, e3: 1.5 - np.sin(2 * np.pi * e1)
        modes = {"ModesSin": mode_params}
    elif bc_x == "mirror":
        fun_exact = lambda e1, e2, e3: 1.5 - np.cos(2 * np.pi * e1)
        modes = {"ModesCos": mode_params}
    pert_params = {"n": modes}

    
    #parameters
    h_vec = [((2**k)*10**-3*0.25) for k in range(2, 12)]
    err_vec = []
    for h1 in h_vec:
        particles = ParticlesSPH(
        comm_world=comm,
        Np=Np,
        ppb = ppb, 
        boxes_per_dim=boxes_per_dim,
        bc_sph=[bc_x, "periodic", "periodic"],
        bufsize=1.0,
        loading=loading,
        loading_params=loading_params,
        domain=domain,
        bckgr_params=bckgr_params,
        pert_params=pert_params,
        verbose = False, 
        )

        particles.draw_markers(sort=False, verbose = False)
        particles.mpi_sort_markers()
        particles.initialize_weights()
        h2 = 1 / boxes_per_dim[1]
        h3 = 1 / boxes_per_dim[2]
        eta1 = np.linspace(0, 1.0, 100)  # add offset for non-periodic boundary conditions, TODO: implement Neumann
        eta2 = np.array([0.0])
        eta3 = np.array([0.0])
        ee1, ee2, ee3 = np.meshgrid(eta1, eta2, eta3, indexing="ij")
        test_eval = particles.eval_density(ee1, ee2, ee3, h1=h1, h2=h2, h3=h3)
        all_eval = np.zeros_like(test_eval)
        
        comm.Allreduce(test_eval, all_eval, op=MPI.SUM)
        
        if show_plot and comm.Get_rank() == 0:
            plt.figure()
            plt.plot(ee1.squeeze(), fun_exact(ee1, ee2, ee3).squeeze(), label="exact")
            plt.plot(ee1.squeeze(), all_eval.squeeze(), "--.", label="eval_sph")
            plt.savefig(f"fun_{h1}.png")
            
        diff = np.max(np.abs(all_eval - fun_exact(ee1,ee2,ee3)))
        print(f'{h1 = }, {diff = }')
        err_vec += [diff]
        
        
    fit = np.polyfit(np.log(h_vec), np.log(err_vec), 1)
    print(fit)
    
    if show_plot and comm.Get_rank() == 0:
        plt.figure(figsize=(12, 8))
        plt.loglog(h_vec, err_vec, label = "Convergence")
        plt.loglog(h_vec, np.exp(fit[1])*np.array(h_vec)**(fit[0]), "--", label = f"fit with slope {fit[0]}")
        plt.legend() 
        plt.show()
        plt.savefig("Convergence_SPH")
    
    if tesselation: 
        assert np.min(err_vec) < 0.03
    else: 
        assert np.min(err_vec) < 0.094
    
    exact_eval = fun_exact(ee1, ee2, ee3)
    err_max_norm = np.max(np.abs(all_eval - exact_eval)) / np.max(np.abs(exact_eval))
    
    if comm.Get_rank() == 0:
        print(f'{tesselation = }, {bc_x = }, {err_max_norm = }')
        
    if tesselation:
        if bc_x == "periodic":
            assert err_max_norm < 0.22
        elif bc_x == "fixed":
            assert err_max_norm < 0.24
        else:
            assert err_max_norm < 0.22
    else:
        if bc_x in ("periodic", "fixed"):
            assert err_max_norm < 0.24
        else: 
            assert err_max_norm < 0.23


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("boxes_per_dim", [(8, 1, 1), (16, 1, 1)])
@pytest.mark.parametrize("bc_x", ["periodic", "fixed", "mirror"])
def test_evaluation_mc_Np_and_h_convergence_1d(boxes_per_dim, bc_x, tesselation, show_plot=False):
    comm = MPI.COMM_WORLD

    # DOMAIN object
    dom_type = "Cuboid"
    dom_params = {"l1": 0.0, "r1": 3.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)
    
    if tesselation: 
        loading = "tesselation"
        loading_params = {"n_quad": 1}
        # ppbs = [5000, 10000, 15000, 20000, 25000]
        ppbs = [4, 8, 16, 32, 64]
        Nps = [None]*len(ppbs)
    
    else: 
        loading = "pseudo_random"
        loading_params = {"seed": 1607}
        Nps = [(2**k)*10**3 for k in range(-2, 9)]
        ppbs = [None]*len(Nps)
    
    cst_vel = {"density_profile": "constant", "n": 1.5}
    bckgr_params = {"ConstantVelocity": cst_vel, "pforms": ["vol", None]}

    # perturbation and exact solution
    mode_params = {"given_in_basis": "0", "ls": [1], "amps": [-1e-0]}
    if bc_x in ("periodic", "fixed"):
        fun_exact = lambda e1, e2, e3: 1.5 - np.sin(2 * np.pi * e1)
        modes = {"ModesSin": mode_params}
    elif bc_x == "mirror":
        fun_exact = lambda e1, e2, e3: 1.5 - np.cos(2 * np.pi * e1)
        modes = {"ModesCos": mode_params}
    pert_params = {"n": modes}

  
    h_arr = [((2**k)*10**-3*0.25) for k in range(2, 12)]
    err_vec = []
    for h in h_arr:
        err_vec += [[]]
        for Np, ppb in zip(Nps, ppbs):
            particles = ParticlesSPH(
                comm_world=comm,
                Np=Np,
                ppb= ppb,
                boxes_per_dim=boxes_per_dim,
                bc_sph=[bc_x, "periodic", "periodic"],
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

            h2 = 1 / boxes_per_dim[1]
            h3 = 1 / boxes_per_dim[2]

            eta1 = np.linspace(0, 1.0, 100)
            eta2 = np.array([0.0])
            eta3 = np.array([0.0])
            ee1, ee2, ee3 = np.meshgrid(eta1, eta2, eta3, indexing="ij")

            test_eval = particles.eval_density(ee1, ee2, ee3, h1=h, h2=h2, h3=h3)
            all_eval = np.zeros_like(test_eval)
            comm.Allreduce(test_eval, all_eval, op=MPI.SUM)
            
            if show_plot and comm.Get_rank() == 0:
                plt.figure()
                plt.plot(ee1.squeeze(), fun_exact(ee1, ee2, ee3).squeeze(), label="exact")
                plt.plot(ee1.squeeze(), all_eval.squeeze(), "--.", label="eval_sph")
                plt.title(f'{h = }, {Np = }')
                plt.savefig(f"fun_h{h}_N{Np}_ppb{ppb}.png")

            diff = np.max(np.abs(all_eval - fun_exact(ee1, ee2, ee3)))
            err_vec[-1] += [diff]
            print(f'{Np = }, {ppb = }, {diff = }')

    err_vec = np.array(err_vec)
    print(np.min(err_vec))

    if show_plot and comm.Get_rank() == 0:
        if tesselation: 
            h_mesh, n_mesh = np.meshgrid(np.log10(h_arr), np.log10(ppbs), indexing='ij')
        if not tesselation: 
            h_mesh, n_mesh = np.meshgrid(np.log10(h_arr), np.log10(Nps), indexing='ij')
        plt.figure(figsize=(6, 6))
        plt.pcolor(h_mesh, n_mesh, np.log10(err_vec), shading='auto')
        plt.title('Error')
        plt.colorbar(label='log10(error)')
        plt.xlabel('log10(h)')
        plt.ylabel('log10(particles)')

        min_indices = np.argmin(err_vec, axis=0)
        min_h_values = []
        for mi in min_indices:
            min_h_values += [np.log10(h_arr[mi])]
        if tesselation: 
             log_particles = np.log10(ppbs)
        else: 
            log_particles = np.log10(Nps)
        plt.plot(min_h_values, log_particles, 'r-', label='Min error h for each Np', linewidth=2)
        plt.legend()
        plt.savefig("SPH_conv_in_h_and_N.png")
        
        plt.show()
        
        assert np.min(err_vec) < 0.1
    
    
    exact_eval = fun_exact(ee1, ee2, ee3)
    err_max_norm = np.max(np.abs(all_eval - exact_eval)) / np.max(np.abs(exact_eval))
    
    if comm.Get_rank() == 0:
        print(f'{tesselation = }, {bc_x = }, {err_max_norm = }')
        
    if tesselation:
        if bc_x == "periodic":
            assert err_max_norm < 0.47
        elif bc_x == "fixed":
            assert err_max_norm < 0.47
        else:
            assert err_max_norm < 0.47
    else:
        if bc_x in ("periodic", "fixed"):
            assert err_max_norm < 0.47
        else: 
            assert err_max_norm < 0.47



@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("boxes_per_dim", [(16, 1, 1)])
@pytest.mark.parametrize("bc_x", ["periodic", "fixed", "mirror"])
@pytest.mark.parametrize("bc_y", ["periodic", "fixed", "mirror"])
@pytest.mark.parametrize("tesselation", [False, True])
def test_evaluation_SPH_Np_convergence_2d(boxes_per_dim, bc_x, bc_y, tesselation, show_plot=False):
    from struphy.fields_background.generic import GenericCartesianFluidEquilibrium
    
    comm = MPI.COMM_WORLD

    # DOMAIN object
    dom_type = "Cuboid"
    dom_params = {"l1": 0.0, "r1": 1.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)
    
    
    if tesselation:
        loading = "tesselation"
        loading_params = {"n_quad": 1}
        ppbs = [4, 8, 16, 32, 64, 200]
        Nps = [None]*len(ppbs)
    else: 
        loading = "pseudo_random"
        loading_params = {"seed": 1607}
        Nps = [(2**k)*10**3 for k in range(-2, 9)]
        ppbs = [None] * len(Nps) 

    cst_vel = {"density_profile": "constant", "n": 1.0}
    bckgr_params = {"ConstantVelocity": cst_vel, "pforms": ["vol", None]}
    
    # perturbation and exact solution
    mode_params = {"given_in_basis": "0", "ls": [1], "ms": [1], "amps": [-1e-0]}

    if bc_x in ("periodic", "fixed"):
        if bc_y in ("periodic", "fixed"):
            fun_exact = lambda e1, e2, e3: 1.0 - np.sin(2 * np.pi * e1) * np.sin(2 * np.pi * e2)
            modes = {"ModesSinSin": mode_params}
        elif bc_y == "mirror":
            fun_exact = lambda e1, e2, e3: 1.0 - np.sin(2 * np.pi * e1) * np.cos(2 * np.pi * e2)
            modes = {"ModesSinCos": mode_params}

    elif bc_x == "mirror":
        if bc_y in ("periodic", "fixed"):
            fun_exact = lambda e1, e2, e3: 1.0 - np.cos(2 * np.pi * e1) * np.sin(2 * np.pi * e2)
            modes = {"ModesCosSin": mode_params}
        elif bc_y == "mirror":
            fun_exact = lambda e1, e2, e3: 1.0 - np.cos(2 * np.pi * e1) * np.cos(2 * np.pi * e2)
            modes = {"ModesCosCos": mode_params}

    pert_params = {"n": modes}

    
    #bckgr = GenericCartesianFluidEquilibrium(n_xyz=n_fun)
    #bckgr.domain = domain
    
    
    err_vec = [] 
    for Np, ppb in zip(Nps, ppbs): 
        particles = ParticlesSPH(
        comm_world=comm,
        Np=Np,
        ppb = ppb, 
        boxes_per_dim=boxes_per_dim,
        bc_sph=[bc_x,bc_y, "periodic"],
        bufsize=1.0,
        box_bufsize=4.0,
        loading=loading,  
        loading_params=loading_params,
        domain=domain,
        bckgr_params=bckgr_params,
        pert_params = pert_params,  
        verbose = False,
        )
        particles.draw_markers(sort = False, verbose = False)
        particles.mpi_sort_markers()
        particles.initialize_weights()
        h1 = 1 / boxes_per_dim[0]
        h2 = 1 / boxes_per_dim[1]
        h3 = 1 / boxes_per_dim[2]
        eta1 = np.linspace(0, 1.0, 100) 
        eta2 = np.linspace(0, 1.0, 40)
        eta3 = np.array([0.0])
        ee1, ee2, ee3 = np.meshgrid(eta1, eta2, eta3, indexing="ij")
        x,y,z = domain(eta1,eta2,eta3)
        test_eval = particles.eval_density(ee1, ee2, ee3, h1=h1, h2=h2, h3=h3, kernel_type = "gaussian_2d")
        all_eval = np.zeros_like(test_eval)
            
        comm.Allreduce(test_eval, all_eval, op=MPI.SUM)
        
        if show_plot and comm.Get_rank() == 0:
            fig, ax = plt.subplots()
            d = ax.pcolor(ee1.squeeze(), ee2.squeeze(), all_eval.squeeze(), label = "eval_sph") 
            fig.colorbar(d, ax=ax, label='2d_SPH')
            ax.set_xlabel('ee1')
            ax.set_ylabel('ee2')
            ax.set_title(f'{Np}_{ppb = }')
                
            fig.savefig(f"2d_sph_{Np}_{ppb}.png")  
                
        diff = np.max(np.abs(all_eval - fun_exact(x,y,z)))
        err_vec += [diff]
        print(f'{Np = }, {ppb = }, {diff = }')
        
        
        if tesselation: 
            assert diff < 10.0
         

    if show_plot and comm.Get_rank() == 0:
        plt.figure(figsize=(12, 8))
        print(ppbs)
        print(err_vec)
        plt.loglog(ppbs, err_vec, label = "Convergence")
        #plt.loglog(ppbs, np.exp(fit[1])*np.array(ppbs)**(fit[0]), "--", label = f"fit with slope {fit[0]}")
        plt.legend() 
        plt.savefig("Convergence_SPH_2d_tesselation")
            
        plt.show()
    
    if not tesselation: 
        assert np.min(err_vec) < 2.0
    
    exact_eval = fun_exact(ee1, ee2, ee3)
    err_max_norm = np.max(np.abs(all_eval - exact_eval)) / np.max(np.abs(exact_eval))
    
    if comm.Get_rank() == 0:
        print(f'{tesselation = }, {bc_x = }, {err_max_norm = }')
        
    if tesselation:
        if bc_x in ("periodic", "fixed") and bc_y in ("periodic", "fixed"):
            assert err_max_norm < 2.0
        elif bc_x in ("periodic", "fixed") and bc_y == "mirror":
            assert err_max_norm < 0.031
        elif bc_x == "mirror" and bc_y in ("periodic", "fixed"):
            assert err_max_norm < 0.031
        elif bc_x == "mirror" and bc_y == "mirror":
            assert err_max_norm < 0.436
    else:
        if bc_x in ("periodic", "fixed") and bc_y in ("periodic", "fixed"):
            assert err_max_norm < 0.099
        elif bc_x in ("periodic", "fixed") and bc_y == "mirror":
            assert err_max_norm < 0.101
        elif bc_x == "mirror" and bc_y in ("periodic", "fixed"):
            assert err_max_norm < 0.101
        elif bc_x == "mirror" and bc_y == "mirror":
            assert err_max_norm < 0.444




if __name__ == "__main__":
     #test_sph_evaluation(
     #     70000,
     #     (12, 1, 1),
     #     None,
     #     "mirror",
     #     tesselation=False,
     #     show_plot=True
     # ) #for Tesselation false, test with at least Np = 30000  
    
    #test_evaluation_SPH_Np_convergence_1d((12,1,1), "mirror", tesselation=True, show_plot=True)
    #test_evaluation_SPH_h_convergence_1d((8,1,1), "mirror", tesselation = False, show_plot=True)
    #test_evaluation_mc_Np_and_h_convergence_1d((16,1,1),"mirror",tesselation = True,  show_plot=True)
    test_evaluation_SPH_Np_convergence_2d((32,32,1), "periodic", "fixed", tesselation = False , show_plot=True)

    
