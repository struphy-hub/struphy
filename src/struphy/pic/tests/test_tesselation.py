from time import time

import pytest
from matplotlib import pyplot as plt
from psydac.ddm.mpi import mpi as MPI

from struphy.feec.psydac_derham import Derham
from struphy.geometry import domains
from struphy.pic.particles import ParticlesSPH
from struphy.utils.arrays import xp as np


@pytest.mark.parametrize("ppb", [8, 12])
@pytest.mark.parametrize("nx", [16, 10, 24])
@pytest.mark.parametrize("ny", [1, 16, 10])
@pytest.mark.parametrize("nz", [1, 14, 12])
def test_draw(ppb, nx, ny, nz):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    dom_type = "Cuboid"
    dom_params = {"l1": 1.0, "r1": 2.0, "l2": 10.0, "r2": 20.0, "l3": 100.0, "r3": 200.0}
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    boxes_per_dim = (nx, ny, nz)
    bc = ["periodic"] * 3
    loading = "tesselation"
    bufsize = 0.5

    # instantiate Particle object
    particles = ParticlesSPH(
        comm_world=comm,
        ppb=ppb,
        boxes_per_dim=boxes_per_dim,
        bc=bc,
        loading=loading,
        domain=domain,
        verbose=False,
        bufsize=bufsize,
    )
    particles.draw_markers(sort=False)

    # print(f'{particles.markers[:, :3] = }')
    # print(f'{rank = }, {particles.positions = }')

    # test
    tiles_x = int(nx / particles.nprocs[0] * particles.tesselation.nt_per_dim[0])
    tiles_y = int(ny / particles.nprocs[1] * particles.tesselation.nt_per_dim[1])
    tiles_z = int(nz / particles.nprocs[2] * particles.tesselation.nt_per_dim[2])

    xl = particles.domain_array[rank, 0]
    xr = particles.domain_array[rank, 1]
    yl = particles.domain_array[rank, 3]
    yr = particles.domain_array[rank, 4]
    zl = particles.domain_array[rank, 6]
    zr = particles.domain_array[rank, 7]

    eta1 = np.linspace(xl, xr, tiles_x + 1)[:-1] + (xr - xl) / (2 * tiles_x)
    eta2 = np.linspace(yl, yr, tiles_y + 1)[:-1] + (yr - yl) / (2 * tiles_y)
    eta3 = np.linspace(zl, zr, tiles_z + 1)[:-1] + (zr - zl) / (2 * tiles_z)

    ee1, ee2, ee3 = np.meshgrid(eta1, eta2, eta3, indexing="ij")
    e1 = ee1.flatten()
    e2 = ee2.flatten()
    e3 = ee3.flatten()

    # print(f'\n{rank = }, {e1 = }')

    assert np.allclose(particles.positions[:, 0], e1)
    assert np.allclose(particles.positions[:, 1], e2)
    assert np.allclose(particles.positions[:, 2], e3)


@pytest.mark.parametrize("ppb", [8, 12])
@pytest.mark.parametrize("nx", [10, 8, 6])
@pytest.mark.parametrize("ny", [1, 16, 10])
@pytest.mark.parametrize("nz", [1, 14, 11])
@pytest.mark.parametrize("n_quad", [1, 2, 3])
def test_cell_average(ppb, nx, ny, nz, n_quad, show_plot=False):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    dom_type = "Cuboid"
    dom_params = {"l1": 0.0, "r1": 1.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    boxes_per_dim = (nx, ny, nz)
    bc = ["periodic"] * 3
    loading = "tesselation"
    loading_params = {"n_quad": n_quad}
    bufsize = 0.5

    cst_vel = {"ux": 0.0, "uy": 0.0, "uz": 0.0, "density_profile": "constant"}
    bckgr_params = {"ConstantVelocity": cst_vel}

    mode_params = {"given_in_basis": "0", "ls": [1], "amps": [1e-0]}
    modes = {"ModesSin": mode_params}
    pert_params = {"n": modes}

    # instantiate Particle object
    particles = ParticlesSPH(
        comm_world=comm,
        ppb=ppb,
        boxes_per_dim=boxes_per_dim,
        bc=bc,
        loading=loading,
        loading_params=loading_params,
        domain=domain,
        verbose=False,
        bufsize=bufsize,
        bckgr_params=bckgr_params,
        pert_params=pert_params,
    )

    particles.draw_markers(sort=False)
    particles.initialize_weights()

    if show_plot:
        tiles_x = nx * particles.tesselation.nt_per_dim[0]
        tiles_y = ny * particles.tesselation.nt_per_dim[1]

        xl = particles.domain_array[rank, 0]
        xr = particles.domain_array[rank, 1]
        yl = particles.domain_array[rank, 3]
        yr = particles.domain_array[rank, 4]

        eta1 = np.linspace(xl, xr, tiles_x + 1)
        eta2 = np.linspace(yl, yr, tiles_y + 1)

        if ny == nz == 1:
            plt.figure(figsize=(15, 10))
            plt.plot(particles.positions[:, 0], np.zeros_like(particles.weights), "o", label="markers")
            plt.plot(particles.positions[:, 0], particles.weights, "-o", label="weights")
            plt.plot(
                np.linspace(xl, xr, 100),
                particles.f_init(np.linspace(xl, xr, 100), 0.5, 0.5).squeeze(),
                "--",
                label="f_init",
            )
            plt.vlines(np.linspace(xl, xr, nx + 1), 0, 2, label="sorting boxes", color="k")
            ax = plt.gca()
            ax.set_xticks(eta1)
            ax.set_yticks(eta2)
            plt.tick_params(labelbottom=False)
            plt.grid()
            plt.legend()
            plt.title("Initial weights and markers from tesselation")

        if nz == 1:
            plt.figure(figsize=(25, 10))

            plt.subplot(1, 2, 1)
            ax = plt.gca()
            ax.set_xticks(np.linspace(0, 1, nx + 1))
            ax.set_yticks(np.linspace(0, 1, ny + 1))
            coloring = particles.weights
            plt.scatter(particles.positions[:, 0], particles.positions[:, 1], c=coloring, s=40)
            plt.grid(c="k")
            plt.axis("square")
            plt.title("initial markers")
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.colorbar()

            plt.subplot(1, 2, 2)
            ax = plt.gca()
            ax.set_xticks(np.linspace(0, 1, nx + 1))
            ax.set_yticks(np.linspace(0, 1, ny + 1))
            coloring = particles.weights
            pos1 = np.linspace(xl, xr, 100)
            pos2 = np.linspace(yl, yr, 100)
            pp1, pp2 = np.meshgrid(pos1, pos2, indexing="ij")
            plt.pcolor(pp1, pp2, particles.f_init(pp1, pp2, 0.5).squeeze())
            plt.grid(c="k")
            plt.axis("square")
            plt.title("initial condition")
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.colorbar()

        plt.show()

    # test
    print(f"\n{rank = }, {np.max(np.abs(particles.weights - particles.f_init(particles.positions))) = }")
    assert np.max(np.abs(particles.weights - particles.f_init(particles.positions))) < 0.012


if __name__ == "__main__":
    # test_draw(8, 16, 1, 1)
    test_cell_average(8, 6, 16, 14, n_quad=2, show_plot=True)
