from time import time

import numpy as np
import pytest
from mpi4py import MPI

from struphy.feec.psydac_derham import Derham
from struphy.fields_background.equils import ConstantVelocity
from struphy.geometry import domains
from struphy.initial import perturbations
from struphy.pic.particles import ParticlesSPH
from struphy.pic.utilities import BoundaryParameters, LoadingParameters, WeightsParameters


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
    loading_params = LoadingParameters(Np=Np, seed=1607)
    boundary_params = BoundaryParameters(bc=(bc_x, "periodic", "periodic"))

    background = ConstantVelocity(n=1.0, density_profile="constant")
    background.domain = domain

    pert = {"n": perturbations.ModesSin(ls=(1,), amps=(1e-0,))}

    fun_exact = lambda e1, e2, e3: 1.0 + np.sin(2 * np.pi * e1)

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
        plt.legend()

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

    loading_params = LoadingParameters(ppb=ppb, loading="tesselation")
    boundary_params = BoundaryParameters(bc=(bc_x, "periodic", "periodic"))

    background = ConstantVelocity(n=1.0, density_profile="constant")
    background.domain = domain

    pert = {"n": perturbations.ModesSin(ls=(1,), amps=(1e-0,))}

    fun_exact = lambda e1, e2, e3: 1.0 + np.sin(2 * np.pi * e1)

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
    # test_evaluation_mc(40000, "periodic", show_plot=True)
    test_evaluation_tesselation((8, 1, 1), 4, "periodic", show_plot=True)
