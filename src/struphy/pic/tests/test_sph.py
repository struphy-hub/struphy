from time import time

import numpy as np
import pytest
from mpi4py import MPI

from struphy.feec.psydac_derham import Derham
from struphy.geometry import domains
from struphy.pic.particles import ParticlesSPH


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("Np", [40000, 46200])
@pytest.mark.parametrize("bc_x", ["periodic", "reflect", "remove"])
def test_evaluation_mc(Np, bc_x, show_plot=False):
    comm = MPI.COMM_WORLD

    # DOMAIN object
    dom_type = "Cuboid"
    dom_params = {"l1": 1.0, "r1": 2.0, "l2": 10.0, "r2": 20.0, "l3": 100.0, "r3": 200.0}
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    boxes_per_dim = (16, 1, 1)
    loading_params = {"seed": 1607}

    cst_vel = {"density_profile": "constant", "n": 1.0}
    bckgr_params = {"ConstantVelocity": cst_vel, "pforms": ["vol", None]}

    mode_params = {"given_in_basis": "0", "ls": [1], "amps": [1e-0]}
    modes = {"ModesSin": mode_params}
    pert_params = {"n": modes}

    fun_exact = lambda e1, e2, e3: 1.0 + np.sin(2 * np.pi * e1)

    particles = ParticlesSPH(
        comm_world=comm,
        Np=Np,
        boxes_per_dim=boxes_per_dim,
        bc=[bc_x, "periodic", "periodic"],
        bufsize=1.0,
        loading_params=loading_params,
        domain=domain,
        bckgr_params=bckgr_params,
        pert_params=pert_params,
    )

    particles.draw_markers(sort=False)
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
        from matplotlib import pyplot as plt

        plt.figure(figsize=(12, 8))
        plt.plot(ee1.squeeze(), fun_exact(ee1, ee2, ee3).squeeze(), label="exact")
        plt.plot(ee1.squeeze(), all_eval.squeeze(), "--.", label="eval_sph")

        plt.show()

    # print(f'{fun_exact(ee1, ee2, ee3) = }')
    # print(f'{comm.Get_rank() = }, {all_eval = }')
    # print(f'{np.max(np.abs(all_eval - fun_exact(ee1, ee2, ee3))) = }')
    assert np.all(np.abs(all_eval - fun_exact(ee1, ee2, ee3)) < 0.065)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("boxes_per_dim", [(8, 1, 1), (10, 1, 1)])
@pytest.mark.parametrize("ppb", [4, 9])
@pytest.mark.parametrize("bc_x", ["periodic", "reflect", "remove"])
def test_evaluation_tesselation(boxes_per_dim, ppb, bc_x, show_plot=False):
    comm = MPI.COMM_WORLD

    # DOMAIN object
    dom_type = "Cuboid"
    dom_params = {"l1": 1.0, "r1": 2.0, "l2": 10.0, "r2": 20.0, "l3": 100.0, "r3": 200.0}
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    loading = "tesselation"
    loading_params = {"n_quad": 1}

    cst_vel = {"density_profile": "constant", "n": 1.0}
    bckgr_params = {"ConstantVelocity": cst_vel, "pforms": ["vol", None]}

    mode_params = {"given_in_basis": "0", "ls": [1], "amps": [1e-0]}
    modes = {"ModesSin": mode_params}
    pert_params = {"n": modes}

    fun_exact = lambda e1, e2, e3: 1.0 + np.sin(2 * np.pi * e1)

    particles = ParticlesSPH(
        comm_world=comm,
        ppb=ppb,
        boxes_per_dim=boxes_per_dim,
        bc=[bc_x, "periodic", "periodic"],
        bufsize=1.0,
        loading=loading,
        loading_params=loading_params,
        domain=domain,
        bckgr_params=bckgr_params,
        pert_params=pert_params,
        verbose=True,
    )

    particles.draw_markers(sort=False)
    particles.mpi_sort_markers()
    particles.initialize_weights()
    h1 = 1 / boxes_per_dim[0]
    h2 = 1 / boxes_per_dim[1]
    h3 = 1 / boxes_per_dim[2]
    eta1 = np.linspace(0, 1.0, 10)  # add offset for non-periodic boundary conditions, TODO: implement Neumann
    eta2 = np.array([0.0])
    eta3 = np.array([0.0])
    ee1, ee2, ee3 = np.meshgrid(eta1, eta2, eta3, indexing="ij")
    test_eval = particles.eval_density(ee1, ee2, ee3, h1=h1, h2=h2, h3=h3)
    all_eval = np.zeros_like(test_eval)

    # rank = comm.Get_rank()
    # print(f'{rank = }, {test_eval.squeeze() = }')

    comm.Allreduce(test_eval, all_eval, op=MPI.SUM)

    if show_plot and comm.Get_rank() == 0:
        from matplotlib import pyplot as plt

        plt.figure(figsize=(12, 8))
        plt.plot(ee1.squeeze(), fun_exact(ee1, ee2, ee3).squeeze(), label="exact")
        plt.plot(ee1.squeeze(), all_eval.squeeze(), "--.", label="eval_sph")
        plt.legend()

        plt.show()

    # print(f'{fun_exact(ee1, ee2, ee3) = }')
    # print(f'{comm.Get_rank() = }, {all_eval = }')
    # print(f'{np.max(np.abs(all_eval - fun_exact(ee1, ee2, ee3))) = }')
    assert np.all(np.abs(all_eval - fun_exact(ee1, ee2, ee3)) < 0.017)


if __name__ == "__main__":
    test_evaluation_mc(40000, "periodic", show_plot=True)
    # test_evaluation_tesselation(
    #     (8, 1, 1),
    #     4,
    #     "periodic",
    #     show_plot=True
    # )
