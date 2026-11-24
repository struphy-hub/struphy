"""Tests of the SPH evaluation with the physical smoothing kernel (:mod:`struphy.pic.sph_setup`,
:meth:`~struphy.pic.particles.ParticlesSPH.eval_density` and related methods).

The kernel is radial in physical space, so ``eval_density`` returns the physical density
:math:`\\rho(F(\\boldsymbol\\eta))`. The backgrounds below are created with ``n_as_volume_form=True``: the density
``n`` (with perturbation) is a logical 3-form and the physical density is :math:`n / \\sqrt g`.
"""

import logging

import cunumpy as xp
import pytest
from matplotlib import pyplot as plt
from maybempi import MPI, SerialComm

from struphy import (
    BoundaryParameters,
    LoadingParameters,
    SortingParameters,
    SPHParameters,
    domains,
    equils,
    perturbations,
)
from struphy.fields_background.equils import ConstantVelocity
from struphy.pic.particles import ParticlesSPH
from struphy.pic.sph_setup import setup_sph_kernel

logger = logging.getLogger("struphy")

KERNELS = ["gaussian", "wendland_c2", "cubic_spline"]


def _comm():
    """The world communicator, or None in serial runs."""
    if isinstance(MPI.COMM_WORLD, SerialComm):
        return None, 0
    return MPI.COMM_WORLD, MPI.COMM_WORLD.Get_rank()


def _allreduce(arr, comm):
    """Sum of the evaluations of all processes (points outside the local domain are zero)."""
    if comm is None:
        return arr
    out = xp.zeros_like(arr)
    comm.Allreduce(arr, out, op=MPI.SUM)
    return out


def _particles(
    domain,
    boxes_per_dim,
    bc_sph,
    loading_params,
    background,
    comm,
    pert=None,
    kernel="gaussian",
    kernel_width=None,
    bufsize=1.0,
    box_bufsize=2.0,
    dims_mask=(True, True, True),
):
    """ParticlesSPH with drawn and MPI-sorted markers and initialized weights."""
    background.domain = domain
    particles = ParticlesSPH(
        comm_world=comm,
        loading_params=loading_params,
        boundary_params=BoundaryParameters(bc_sph=bc_sph),
        sorting_params=SortingParameters(boxes_per_dim=boxes_per_dim, box_bufsize=box_bufsize, dims_mask=dims_mask),
        sph_params=SPHParameters(kernel=kernel, kernel_width=kernel_width),
        bufsize=bufsize,
        domain=domain,
        background=background,
        perturbations=pert,
        n_as_volume_form=True,
    )
    particles.draw_markers(sort=False)
    if comm is not None:
        particles.mpi_sort_markers()
    particles.initialize_weights()
    return particles


def _rel_err(num, exact):
    return float(xp.max(xp.abs(num - exact)) / xp.max(xp.abs(exact)))


# -------------------------
# density, 1D, 2D, 3D kernels
# -------------------------


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("derivative", [0, 1])
@pytest.mark.parametrize("bc_x", ["periodic", "mirror", "fixed"])
@pytest.mark.parametrize("tesselation", [False, True])
def test_sph_evaluation_1d(kernel, derivative, bc_x, tesselation, show_plot=False):
    """1D kernel (boxes only in eta1) on a Cuboid with transverse area A = 6: the kernel is divided by A, so the
    physical density (n / sqrt g) is recovered."""
    comm, rank = _comm()
    boxes_per_dim = (24, 1, 1)
    domain = domains.Cuboid(l1=0.0, r1=1.0, l2=0.0, r2=2.0, l3=0.0, r3=3.0)
    sqrt_g = 6.0

    # h = h_max = 0.99 / 24 is about 8 (tesselation) marker spacings
    if tesselation:
        loading_params = LoadingParameters(ppb=8, seed=1607, loading="tesselation")
    else:
        loading_params = LoadingParameters(ppb=1000 if derivative == 0 else 20000, seed=223)

    background = ConstantVelocity(n=1.5, density_profile="constant")
    pert = {"n": perturbations.ModesCos(ls=(1,), amps=(1.0,))}
    if derivative == 0:
        fun_exact = lambda e1: (1.5 + xp.cos(2 * xp.pi * e1)) / sqrt_g
    else:
        fun_exact = lambda e1: -2 * xp.pi * xp.sin(2 * xp.pi * e1) / sqrt_g

    particles = _particles(
        domain, boxes_per_dim, (bc_x, "periodic", "periodic"), loading_params, background, comm, pert, kernel
    )
    assert particles.sph_kernel.dim == 1
    assert particles.sph_kernel.scale == pytest.approx(1 / sqrt_g)

    ee1, ee2, ee3 = xp.meshgrid(xp.linspace(0, 1.0, 16), xp.array([0.0]), xp.array([0.0]), indexing="ij")
    all_eval = _allreduce(particles.eval_density(ee1, ee2, ee3, derivative=derivative), comm)
    err = _rel_err(all_eval, fun_exact(ee1))
    logger.info(f"{kernel =}, {derivative =}, {bc_x =}, {tesselation =}, {err =}")

    if show_plot and rank == 0:
        plt.plot(ee1.squeeze(), fun_exact(ee1).squeeze(), label="exact")
        plt.plot(ee1.squeeze(), all_eval.squeeze(), "--.", label="eval_sph")
        plt.legend()
        plt.show()

    if tesselation:
        tol = 0.01 if derivative == 0 else 0.03
    else:
        # Monte-Carlo noise, amplified by the kernel derivative
        tol = 0.05 if derivative == 0 else 0.4
    assert err < tol, err


BC_PAIRS_2D = [("periodic", "periodic"), ("mirror", "periodic"), ("periodic", "fixed"), ("mirror", "mirror")]


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("derivative", [0, 1, 2])
@pytest.mark.parametrize("bc_xy", BC_PAIRS_2D)
def test_sph_evaluation_2d(kernel, derivative, bc_xy, show_plot=False):
    """2D kernel on a Cuboid of size 1 x 2 x 3 (non-square boxes); physical derivatives d/dx, d/dy."""
    comm, rank = _comm()
    bc_x, bc_y = bc_xy
    Lx, Ly, Lz = 1.0, 2.0, 3.0
    domain = domains.Cuboid(l1=0.0, r1=Lx, l2=0.0, r2=Ly, l3=0.0, r3=Lz)
    sqrt_g = Lx * Ly * Lz

    # square boxes of width 1/12 with 6 x 6 markers: h = h_max is about 4 marker spacings
    loading_params = LoadingParameters(ppb=36, loading="tesselation")
    background = ConstantVelocity(n=1.5, density_profile="constant")
    pert = {"n": perturbations.ModesCosCos(ls=(1,), ms=(1,), amps=(1.0,))}

    c1 = lambda e: xp.cos(2 * xp.pi * e)
    s1 = lambda e: xp.sin(2 * xp.pi * e)
    if derivative == 0:
        fun_exact = lambda e1, e2: (1.5 + c1(e1) * c1(e2)) / sqrt_g
    elif derivative == 1:
        fun_exact = lambda e1, e2: -2 * xp.pi / Lx * s1(e1) * c1(e2) / sqrt_g
    else:
        fun_exact = lambda e1, e2: -2 * xp.pi / Ly * c1(e1) * s1(e2) / sqrt_g

    particles = _particles(
        domain, (12, 24, 1), (bc_x, bc_y, "periodic"), loading_params, background, comm, pert, kernel
    )
    assert particles.sph_kernel.dim == 2

    eta = xp.linspace(0, 1.0, 11)
    ee1, ee2, ee3 = xp.meshgrid(eta, eta, xp.array([0.0]), indexing="ij")
    all_eval = _allreduce(particles.eval_density(ee1, ee2, ee3, derivative=derivative), comm)
    err = _rel_err(all_eval, fun_exact(ee1, ee2))
    logger.info(f"{kernel =}, {derivative =}, {bc_x =}, {bc_y =}, {err =}")

    if show_plot and rank == 0:
        plt.subplot(2, 1, 1)
        plt.pcolor(ee1.squeeze(), ee2.squeeze(), fun_exact(ee1, ee2).squeeze())
        plt.subplot(2, 1, 2)
        plt.pcolor(ee1.squeeze(), ee2.squeeze(), all_eval.squeeze())
        plt.show()

    assert err < (0.03 if derivative == 0 else 0.07), err


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("derivative", [0, 3])
@pytest.mark.parametrize("bc_z", ["periodic", "mirror", "fixed"])
def test_sph_evaluation_3d(kernel, derivative, bc_z):
    """3D kernel, uniform density on a Cuboid of size 1 x 2 x 3."""
    comm, rank = _comm()
    domain = domains.Cuboid(l1=1.0, r1=2.0, l2=0.0, r2=2.0, l3=-1.0, r3=2.0)
    sqrt_g = 6.0

    # cubic boxes of width 1/4 with 5 x 5 x 5 markers: h = h_max is about 5 marker spacings;
    # bufsize for the ghost layer of the small sub-domains on 4 processes
    loading_params = LoadingParameters(ppb=125, loading="tesselation")
    background = ConstantVelocity(n=1.5, density_profile="constant")
    particles = _particles(
        domain, (4, 8, 12), ("periodic", "periodic", bc_z), loading_params, background, comm, kernel=kernel, bufsize=4.0
    )
    assert particles.sph_kernel.dim == 3

    eta = xp.linspace(0, 1.0, 7)
    ee1, ee2, ee3 = xp.meshgrid(eta, eta, eta, indexing="ij")
    all_eval = _allreduce(particles.eval_density(ee1, ee2, ee3, derivative=derivative), comm)
    exact = 1.5 / sqrt_g
    err = float(xp.max(xp.abs(all_eval - (exact if derivative == 0 else 0.0)))) / exact
    logger.info(f"{kernel =}, {derivative =}, {bc_z =}, {err =}")
    # relative to the density; the derivative is compared in units of rho / (length 1)
    assert err < 0.03, err


# -------------------------
# convergence in N and h
# -------------------------


def _pert_1d(bc_x):
    """Perturbation compatible with the boundary condition, and the exact logical density (3-form)."""
    if bc_x in ("periodic", "fixed"):
        return {"n": perturbations.ModesSin(ls=(1,), amps=(-1.0,))}, lambda e1: 1.5 - xp.sin(2 * xp.pi * e1)
    return {"n": perturbations.ModesCos(ls=(1,), amps=(-1.0,))}, lambda e1: 1.5 - xp.cos(2 * xp.pi * e1)


@pytest.mark.parametrize("bc_x", ["periodic", "mirror", "fixed"])
@pytest.mark.parametrize("tesselation", [False, True])
def test_evaluation_SPH_Np_convergence_1d(bc_x, tesselation, show_plot=False):
    """Error versus the number of markers at fixed h = h_max: Monte-Carlo rate N^-1/2 for random loading, no
    increase for tesselation (the smoothing error dominates)."""
    comm, rank = _comm()
    Lx = 3.0
    domain = domains.Cuboid(l1=0.0, r1=Lx, l2=0.0, r2=1.0, l3=0.0, r3=1.0)
    pert, fun_exact = _pert_1d(bc_x)

    ee1, ee2, ee3 = xp.meshgrid(xp.linspace(0, 1.0, 16), xp.array([0.0]), xp.array([0.0]), indexing="ij")
    exact_eval = fun_exact(ee1) / Lx

    if tesselation:
        loadings = [LoadingParameters(ppb=ppb, loading="tesselation") for ppb in (4, 8, 16, 32, 64)]
        xvec = [4, 8, 16, 32, 64]
    else:
        xvec = [(2**k) * 10**3 for k in range(-2, 7)]
        loadings = [LoadingParameters(Np=Np, seed=1607) for Np in xvec]

    err_vec = []
    for loading_params in loadings:
        background = ConstantVelocity(n=1.5, density_profile="constant")
        particles = _particles(
            domain, (12, 1, 1), (bc_x, "periodic", "periodic"), loading_params, background, comm, pert
        )
        all_eval = _allreduce(particles.eval_density(ee1, ee2, ee3), comm)
        err_vec += [_rel_err(all_eval, exact_eval)]

    fit = xp.polyfit(xp.log(xvec), xp.log(err_vec), 1)
    logger.info(f"{bc_x =}, {tesselation =}, {err_vec =}, {fit[0] =}")

    if show_plot and rank == 0:
        plt.loglog(xvec, err_vec)
        plt.show()

    if tesselation:
        assert fit[0] < 2e-3
    else:
        assert abs(fit[0] + 0.5) < 0.1, fit[0]  # Monte Carlo rate


@pytest.mark.parametrize("bc_x", ["periodic", "mirror", "fixed"])
@pytest.mark.parametrize("tesselation", [False, True])
def test_evaluation_SPH_h_convergence_1d(bc_x, tesselation, show_plot=False):
    """Error versus the physical kernel width h in [h_min, h_max] at fixed N: Monte-Carlo noise ~ (N h)^-1/2 for
    random loading at small h; small errors for tesselation."""
    comm, rank = _comm()
    Lx = 3.0
    domain = domains.Cuboid(l1=0.0, r1=Lx, l2=0.0, r2=1.0, l3=0.0, r3=1.0)
    pert, fun_exact = _pert_1d(bc_x)

    ee1, ee2, ee3 = xp.meshgrid(xp.linspace(0, 1.0, 16), xp.array([0.0]), xp.array([0.0]), indexing="ij")
    exact_eval = fun_exact(ee1) / Lx

    if tesselation:
        loading_params = LoadingParameters(ppb=160, loading="tesselation")
    else:
        loading_params = LoadingParameters(Np=160000, seed=1607)

    # h_max = 0.99 * 3 / 12 = 0.2475; h_min = 2 * 3 / 1920 = 0.0031 (tesselation), 2 * 3 / 160000 (random)
    if tesselation:
        h_vec = [0.004 * 2**k for k in range(6)] + [0.24]
    else:
        h_vec = [0.0005 * 2**k for k in range(7)]
    err_vec = []
    for h in h_vec:
        background = ConstantVelocity(n=1.5, density_profile="constant")
        particles = _particles(
            domain, (12, 1, 1), (bc_x, "periodic", "periodic"), loading_params, background, comm, pert, kernel_width=h
        )
        assert particles.sph_kernel.h == h
        all_eval = _allreduce(particles.eval_density(ee1, ee2, ee3), comm)
        err_vec += [_rel_err(all_eval, exact_eval)]

    logger.info(f"{bc_x =}, {tesselation =}, {err_vec =}")
    if show_plot and rank == 0:
        plt.loglog(h_vec, err_vec)
        plt.show()

    if tesselation:
        assert max(err_vec) < 0.036, err_vec
    else:
        # noise-dominated range (h <= 0.032); the max norm over 16 points fluctuates, hence the wide range
        fit = xp.polyfit(xp.log(h_vec), xp.log(err_vec), 1)
        assert abs(fit[0] + 0.5) < 0.15, fit[0]  # Monte Carlo rate


@pytest.mark.parametrize("bc_x", ["periodic", "mirror", "fixed"])
@pytest.mark.parametrize("tesselation", [False, True])
def test_evaluation_mc_Np_and_h_convergence_1d(bc_x, tesselation):
    """Smallest error over a grid of kernel widths and marker numbers."""
    comm, rank = _comm()
    Lx = 3.0
    domain = domains.Cuboid(l1=0.0, r1=Lx, l2=0.0, r2=1.0, l3=0.0, r3=1.0)
    pert, fun_exact = _pert_1d(bc_x)

    ee1, ee2, ee3 = xp.meshgrid(xp.linspace(0, 1.0, 16), xp.array([0.0]), xp.array([0.0]), indexing="ij")
    exact_eval = fun_exact(ee1) / Lx

    if tesselation:
        loadings = [LoadingParameters(ppb=ppb, loading="tesselation") for ppb in (16, 32, 64)]
    else:
        loadings = [LoadingParameters(Np=Np, seed=1607) for Np in (16000, 64000, 256000)]

    err_vec = []
    # h_min = 2 * 3 / 192 = 0.031 for the smallest tesselation
    for h in (0.04, 0.08, 0.16, 0.24):
        for loading_params in loadings:
            background = ConstantVelocity(n=1.5, density_profile="constant")
            particles = _particles(
                domain,
                (12, 1, 1),
                (bc_x, "periodic", "periodic"),
                loading_params,
                background,
                comm,
                pert,
                kernel_width=h,
            )
            all_eval = _allreduce(particles.eval_density(ee1, ee2, ee3), comm)
            err_vec += [_rel_err(all_eval, exact_eval)]

    err_min = min(err_vec)
    logger.info(f"{tesselation =}, {bc_x =}, {err_min =}")
    if tesselation:
        assert err_min < 1e-3, err_min
    else:
        # mirror walls double the noise near the wall (correlated ghost markers)
        assert err_min < (0.021 if bc_x == "mirror" else 0.01), err_min


@pytest.mark.parametrize("bc_xy", [("periodic", "periodic"), ("fixed", "mirror"), ("mirror", "fixed")])
@pytest.mark.parametrize("tesselation", [False, True])
def test_evaluation_SPH_Np_convergence_2d(bc_xy, tesselation):
    """Error versus the number of markers with a 2D kernel."""
    comm, rank = _comm()
    bc_x, bc_y = bc_xy
    domain = domains.Cuboid(l1=0.0, r1=1.0, l2=0.0, r2=1.0, l3=0.0, r3=1.0)

    sx = lambda e: xp.sin(2 * xp.pi * e) if bc_x != "mirror" else xp.cos(2 * xp.pi * e)
    sy = lambda e: xp.sin(2 * xp.pi * e) if bc_y != "mirror" else xp.cos(2 * xp.pi * e)
    mode = "Modes" + ("Cos" if bc_x == "mirror" else "Sin") + ("Cos" if bc_y == "mirror" else "Sin")
    pert = {"n": getattr(perturbations, mode)(ls=(1,), ms=(1,), amps=(-1.0,))}

    ee1, ee2, ee3 = xp.meshgrid(xp.linspace(0, 1.0, 21), xp.linspace(0, 1.0, 21), xp.array([0.0]), indexing="ij")
    exact_eval = 1.5 - sx(ee1) * sy(ee2)

    if tesselation:
        # ppb >= 8: with 4 markers per box h_min = 2 / 48 > h_max = 0.99 / 24
        xvec = [8, 16, 32, 64]
        loadings = [LoadingParameters(ppb=ppb, loading="tesselation") for ppb in xvec]
    else:
        xvec = [(2**k) * 10**3 for k in range(2, 8)]
        loadings = [LoadingParameters(Np=Np, seed=1607) for Np in xvec]

    err_vec = []
    for loading_params in loadings:
        background = ConstantVelocity(n=1.5, density_profile="constant")
        particles = _particles(
            domain, (24, 24, 1), (bc_x, bc_y, "periodic"), loading_params, background, comm, pert, box_bufsize=4.0
        )
        all_eval = _allreduce(particles.eval_density(ee1, ee2, ee3), comm)
        err_vec += [_rel_err(all_eval, exact_eval)]
        if tesselation:
            assert err_vec[-1] < 0.06, err_vec

    logger.info(f"{bc_x =}, {bc_y =}, {tesselation =}, {err_vec =}")
    if not tesselation:
        fit = xp.polyfit(xp.log(xvec), xp.log(err_vec), 1)
        assert abs(fit[0] + 0.5) < 0.1, fit[0]  # Monte Carlo rate


# -------------------------
# curved mapping
# -------------------------


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("bc_x", ["periodic", "mirror"])
def test_sph_density_colella(kernel, bc_x):
    """Physical density on a curved mapping (Colella): uniform logical 3-form, so the exact physical density
    1.5 / sqrt(g) varies in space; box-based and naive evaluation agree."""
    comm, rank = _comm()
    domain = domains.Colella(Lx=1.0, Ly=1.0, alpha=0.05, Lz=1.0)
    background = ConstantVelocity(n=1.5, density_profile="constant")
    particles = _particles(
        domain,
        (12, 12, 1),
        (bc_x, "periodic", "periodic"),
        LoadingParameters(ppb=16, loading="tesselation"),
        background,
        comm,
        kernel=kernel,
    )
    eta = xp.linspace(0.0, 1.0, 13)
    ee1, ee2, ee3 = xp.meshgrid(eta, eta, xp.array([0.0]), indexing="ij")
    box = _allreduce(particles.eval_density(ee1, ee2, ee3), comm)
    naive = _allreduce(particles.eval_density(ee1, ee2, ee3, fast=False), comm)
    exact = 1.5 / domain.jacobian_det(ee1, ee2, ee3)
    err = _rel_err(box, exact)
    logger.info(f"Colella {kernel =}, {bc_x =}, {err =}")
    # ~16 markers per box and h ~ 2.7 marker spacings: ~1% smoothing/lattice error; the mirror wall is not flat in
    # Colella (first-order mirror images), hence the larger tolerance there
    assert err < (0.02 if bc_x == "periodic" else 0.04), err
    if bc_x == "periodic":
        # the naive sum skips the mirror ghosts, so it agrees with the box-based sum only without walls
        assert xp.allclose(naive, box, rtol=1e-10, atol=1e-12)


# -------------------------
# velocity and viscosity
# -------------------------


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("derivative", [0, 1])
@pytest.mark.parametrize("bc_x", ["periodic", "mirror", "fixed"])
@pytest.mark.parametrize("tesselation", [False, True])
def test_sph_velocity_evaluation(kernel, derivative, bc_x, tesselation, show_plot=False):
    """Mean velocity u_x = cos(2 pi x / Lx) and its derivative, 1D kernel."""
    comm, rank = _comm()
    Lx = 1.0
    domain = domains.Cuboid(l1=0.0, r1=Lx, l2=0.0, r2=1.0, l3=0.0, r3=1.0)

    if tesselation:
        loading_params = LoadingParameters(ppb=10, seed=1607, loading="tesselation")
    else:
        loading_params = LoadingParameters(ppb=400, seed=223)

    def u_xyz(x, y, z):
        return (xp.cos(2 * xp.pi / Lx * x), 0.0 * x, 0.0 * x)

    def du_xyz(x, y, z):
        return (-2 * xp.pi / Lx * xp.sin(2 * xp.pi / Lx * x), 0.0 * x, 0.0 * x)

    background = equils.GenericCartesianFluidEquilibrium(u_xyz=u_xyz)
    particles = _particles(
        domain,
        (12, 1, 1),
        (bc_x, "periodic", "periodic"),
        loading_params,
        background,
        comm,
        kernel=kernel,
        bufsize=2.0,
        box_bufsize=4.0,
    )

    ee1, ee2, ee3 = xp.meshgrid(xp.linspace(0, 1.0, 16), xp.array([0.0]), xp.array([0.0]), indexing="ij")
    v1, v2, v3 = particles.eval_velocity(ee1, ee2, ee3, derivative=derivative)
    v1 = _allreduce(v1, comm)
    v1_e = (u_xyz if derivative == 0 else du_xyz)(ee1, ee2, ee3)[0]
    err = _rel_err(v1, v1_e)
    logger.info(f"{kernel =}, {derivative =}, {bc_x =}, {tesselation =}, {err =}")

    if show_plot and rank == 0:
        plt.plot(ee1.squeeze(), v1_e.squeeze(), label="exact")
        plt.plot(ee1.squeeze(), v1.squeeze(), "--.", label="SPH")
        plt.legend()
        plt.show()

    if tesselation:
        assert err < 0.025, err
    else:
        assert err < 0.2, err


@pytest.mark.parametrize(
    "derivative,tesselation",
    # the derivatives with random loading are dominated by Monte-Carlo noise (errors ~ 1 at 400 markers per box)
    [(0, False), (0, True), (1, True), (2, True)],
)
@pytest.mark.parametrize("bc_xy", BC_PAIRS_2D)
def test_sph_velocity_evaluation_2d(derivative, tesselation, bc_xy):
    """Cartesian mean velocity u_x = u_y = cos(2 pi x / Lx) cos(2 pi y / Ly) and its physical derivatives on a
    2 x 3 Cuboid, 2D Gaussian kernel. The markers carry the Cartesian velocity u(x_p)."""
    comm, rank = _comm()
    bc_x, bc_y = bc_xy
    Lx, Ly = 2.0, 3.0
    domain = domains.Cuboid(l1=0.0, r1=Lx, l2=0.0, r2=Ly, l3=0.0, r3=1.0)

    if tesselation:
        loading_params = LoadingParameters(ppb=50, seed=1607, loading="tesselation")
    else:
        loading_params = LoadingParameters(ppb=400, seed=223)

    kx, ky = 2 * xp.pi / Lx, 2 * xp.pi / Ly

    def u_xyz(x, y, z):
        u = xp.cos(kx * x) * xp.cos(ky * y)
        return (u, u, 0.0 * z)

    background = equils.GenericCartesianFluidEquilibrium(u_xyz=u_xyz)
    particles = _particles(
        domain, (12, 12, 1), (bc_x, bc_y, "periodic"), loading_params, background, comm, bufsize=2.0, box_bufsize=4.0
    )

    eta = xp.linspace(0, 1.0, 11)
    ee1, ee2, ee3 = xp.meshgrid(eta, eta, xp.array([0.0]), indexing="ij")
    xx, yy, zz = Lx * ee1, Ly * ee2, 0.0 * ee3
    v1, v2, v3 = (_allreduce(v, comm) for v in particles.eval_velocity(ee1, ee2, ee3, derivative=derivative))

    if derivative == 0:
        exact = u_xyz(xx, yy, zz)[0]
    elif derivative == 1:
        exact = -kx * xp.sin(kx * xx) * xp.cos(ky * yy)
    else:
        exact = -ky * xp.cos(kx * xx) * xp.sin(ky * yy)

    err_ux = _rel_err(v1, exact)
    err_uy = _rel_err(v2, exact)
    logger.info(f"{derivative =}, {bc_x =}, {bc_y =}, {tesselation =}, {err_ux =}, {err_uy =}")

    # Monte-Carlo noise for random loading (max norm over 121 points)
    tol = 0.02 if tesselation else 0.15
    assert err_ux < tol, err_ux
    assert err_uy < tol, err_uy


@pytest.mark.parametrize("kernel", KERNELS)
def test_sph_viscosity_evaluation_2d(kernel):
    """Divergence of the viscous stress sigma = -mu (grad u + grad u^T - 2/3 div u I) for a Taylor-Green-like flow
    on the unit square."""
    comm, rank = _comm()
    domain = domains.Cuboid(l1=0.0, r1=1.0, l2=0.0, r2=1.0, l3=0.0, r3=1.0)
    mu = 1e-3

    def u_xyz(x, y, z):
        return (
            xp.sin(2 * xp.pi * x) * xp.cos(2 * xp.pi * y),
            xp.cos(2 * xp.pi * x) * xp.cos(2 * xp.pi * y),
            xp.zeros_like(x),
        )

    def div_sigma(x, y):
        s, c = xp.sin, xp.cos
        p = 2 * xp.pi
        div_x = mu * ((28 / 3) * xp.pi**2 * s(p * x) * c(p * y) - (4 * xp.pi**2 / 3) * s(p * x) * s(p * y))
        div_y = mu * ((28 / 3) * xp.pi**2 * c(p * x) * c(p * y) + (4 * xp.pi**2 / 3) * c(p * x) * s(p * y))
        return div_x, div_y

    background = equils.GenericCartesianFluidEquilibrium(u_xyz=u_xyz)
    particles = _particles(
        domain,
        (12, 12, 1),
        ("periodic", "periodic", "periodic"),
        LoadingParameters(ppb=50, seed=1607, loading="tesselation"),
        background,
        comm,
        kernel=kernel,
        bufsize=2.0,
        box_bufsize=4.0,
    )

    eta = xp.linspace(0, 1.0, 11)
    ee1, ee2, ee3 = xp.meshgrid(eta, eta, xp.array([0.0]), indexing="ij")
    gx, gy, gz = (_allreduce(g, comm) for g in particles.eval_div_viscosity(ee1, ee2, ee3, mu=mu))
    ex, ey = div_sigma(ee1, ee2)
    err_x = _rel_err(gx, ex)
    err_y = _rel_err(gy, ey)
    logger.info(f"{kernel =}, {err_x =}, {err_y =}")
    # second derivatives of the SPH interpolant (two kernel gradients)
    assert err_x < 0.05, err_x
    assert err_y < 0.05, err_y
    assert float(xp.max(xp.abs(gz))) < 1e-12


# -------------------------
# no-slip walls
# -------------------------


@pytest.mark.parametrize("tesselation", [False, True])
@pytest.mark.parametrize("direction", ["x", "y", "z"])
def test_sph_no_slip_boundary_1d(tesselation, direction, show_plot=False):
    """Uniform flow along the resolved direction with no-slip walls: zero velocity at the walls."""
    comm, rank = _comm()
    domain = domains.Cuboid(l1=0.0, r1=1.0, l2=0.0, r2=1.0, l3=0.0, r3=1.0)
    k = "xyz".index(direction)

    if tesselation:
        loading_params = LoadingParameters(ppb=4, loading="tesselation")
    else:
        loading_params = LoadingParameters(ppb=200, seed=223)

    def u_xyz(x, y, z):
        u = [xp.zeros_like(x)] * 3
        u[k] = xp.ones_like(x)
        return tuple(u)

    boxes_per_dim = tuple(12 if i == k else 1 for i in range(3))
    bc_sph = tuple("noslip" if i == k else "periodic" for i in range(3))
    background = equils.GenericCartesianFluidEquilibrium(u_xyz=u_xyz)
    # bufsize: room for the ghost particles of the wall and of the neighbouring processes (4 ranks)
    particles = _particles(domain, boxes_per_dim, bc_sph, loading_params, background, comm, bufsize=2.0)

    pts = [xp.array([0.5])] * 3
    pts[k] = xp.linspace(0.0, 1.0, 100)
    ee1, ee2, ee3 = xp.meshgrid(*pts, indexing="ij")
    v = [_allreduce(vi, comm).squeeze() for vi in particles.eval_velocity(ee1, ee2, ee3)]

    if show_plot and rank == 0:
        for vi in v:
            plt.plot(pts[k], vi)
        plt.show()

    tol_wall = 3e-3
    tol_interior = 5e-2 if tesselation else 1.6e-1
    for c in range(3):
        assert abs(v[c][0]) < tol_wall, f"left wall velocity {c} not zero: {v[c][0]}"
        assert abs(v[c][-1]) < tol_wall, f"right wall velocity {c} not zero: {v[c][-1]}"

    interior = v[k][1:-1]
    assert abs(interior[0]) < 0.5
    assert abs(interior[-1]) < 0.5
    rel_error = float(xp.max(xp.abs(interior[7:-7] - 1.0)))
    assert rel_error < tol_interior, rel_error


def test_sph_no_slip_boundary_2d():
    """2D no-slip test: periodic x, noslip y, MPI decomposition in x, so that each process handles both walls."""
    comm, rank = _comm()
    domain = domains.Cuboid(l1=0.0, r1=1.0, l2=0.0, r2=1.0, l3=0.0, r3=1.0)

    def u_xyz(x, y, z):
        return (xp.ones_like(x), xp.zeros_like(x), xp.zeros_like(x))

    background = equils.GenericCartesianFluidEquilibrium(u_xyz=u_xyz)
    particles = _particles(
        domain,
        (8, 8, 1),
        ("periodic", "noslip", "periodic"),
        LoadingParameters(ppb=16, loading="tesselation"),
        background,
        comm,
        bufsize=3.0,
        dims_mask=(True, True, False),
    )

    ee1, ee2, ee3 = xp.meshgrid(xp.linspace(0.0, 1.0, 8), xp.linspace(0.0, 1.0, 50), xp.array([0.5]), indexing="ij")
    v1 = _allreduce(particles.eval_velocity(ee1, ee2, ee3)[0], comm)
    v1_avg = xp.mean(v1[:, :, 0], axis=0)

    assert abs(float(v1_avg[0])) < 5e-2, v1_avg[0]
    assert abs(float(v1_avg[-1])) < 5e-2, v1_avg[-1]
    assert float(xp.max(xp.abs(v1_avg[5:-5] - 1.0))) < 1e-1


# -------------------------
# box-based vs naive, box capacity
# -------------------------


def _make_sph_particles_1d(bc_x, comm, kernel="cubic_spline"):
    """Small 1D ParticlesSPH setup (8 boxes, 8 markers per box) with drawn markers and weights."""
    background = ConstantVelocity(n=1.5, density_profile="constant")
    pert = {"n": perturbations.ModesSin(ls=(1,), amps=(0.5,))}
    return _particles(
        domains.Cuboid(),
        (8, 1, 1),
        (bc_x, "periodic", "periodic"),
        LoadingParameters(ppb=8, seed=1607, loading="tesselation"),
        background,
        comm,
        pert,
        kernel,
    )


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("derivative", [0, 1])
def test_sph_naive_vs_box_evaluation(kernel, derivative):
    """The naive evaluation (fast=False) must agree with the box-based one (periodic: no ghosts are skipped)."""
    comm, _ = _comm()
    particles = _make_sph_particles_1d("periodic", comm, kernel)

    ee1, ee2, ee3 = xp.meshgrid(xp.linspace(0.0, 1.0, 17), xp.array([0.0]), xp.array([0.0]), indexing="ij")
    box = _allreduce(particles.eval_density(ee1, ee2, ee3, derivative=derivative, fast=True), comm)
    naive = _allreduce(particles.eval_density(ee1, ee2, ee3, derivative=derivative, fast=False), comm)
    assert xp.allclose(naive, box, rtol=1e-12, atol=1e-12)


def test_sph_full_box_row_detected():
    """A box holding as many markers as the box array has columns has no -1 terminator and must be rejected."""
    if not isinstance(MPI.COMM_WORLD, SerialComm):
        return  # error path calls Abort() under MPI

    particles = _make_sph_particles_1d("periodic", None)
    particles.put_particles_in_boxes()
    max_in_box = int(xp.max(xp.bincount(xp.int64(particles.markers_wo_holes[-2]))))

    # one spare column: fits, the -1 terminator is kept
    n_boxes = particles.sorting_boxes.boxes.shape[0]
    particles.sorting_boxes._boxes = xp.full((n_boxes, max_in_box + 1), -1, dtype=int)
    particles.put_particles_in_boxes()

    # no spare column: the fullest box row would have no terminator
    particles.sorting_boxes._boxes = xp.full((n_boxes, max_in_box), -1, dtype=int)
    with pytest.raises(RuntimeError, match="box_bufsize"):
        particles.put_particles_in_boxes()


# -------------------------
# kernel setup
# -------------------------


@pytest.mark.mpi_skip
@pytest.mark.parametrize("boxes_per_dim", [(12, 1, 1), (6, 8, 1), (4, 6, 8)])
def test_sph_kernel_setup_cuboid(boxes_per_dim):
    """h_max = 0.99 min_i L_i / n_i over the resolved directions, scale = 1 / A_perp, period vectors, h_min."""
    L = (1.5, 2.0, 3.0)
    domain = domains.Cuboid(l1=0.0, r1=L[0], l2=0.0, r2=L[1], l3=0.0, r3=L[2])
    Np = 10_000
    k = setup_sph_kernel(domain, boxes_per_dim, ("periodic",) * 3, Np, "wendland_c2", None, 2.0, 0)
    res = [i for i in range(3) if boxes_per_dim[i] > 1]
    assert k.dim == len(res)
    assert k.h_max == pytest.approx(0.99 * min(L[i] / boxes_per_dim[i] for i in res))
    assert k.h == k.h_max
    area = xp.prod([L[i] for i in range(3) if i not in res])
    assert k.scale == pytest.approx(1 / area)
    vol = xp.prod([L[i] for i in res])
    assert k.h_min == pytest.approx(2.0 * (vol / Np) ** (1 / len(res)))
    for i in range(3):
        expected = xp.zeros(3)
        if i in res:
            expected[i] = L[i]
        assert xp.allclose(k.period[i], expected)
        assert k.proj[i, i] == (1.0 if i in res else 0.0)


@pytest.mark.mpi_skip
def test_sph_kernel_setup_errors():
    """Clear errors for kernel widths outside [h_min, h_max], too few markers, non-affine unresolved directions and
    polar mappings without an annulus."""
    cuboid = domains.Cuboid()
    # kernel_width larger than h_max = 0.99 / 12
    with pytest.raises(ValueError, match="larger than h_max"):
        setup_sph_kernel(cuboid, (12, 1, 1), ("periodic",) * 3, 1000, "gaussian", 0.1, 2.0, 0)
    # kernel_width smaller than h_min = 2 / 1000
    with pytest.raises(ValueError, match="smaller than h_min"):
        setup_sph_kernel(cuboid, (12, 1, 1), ("periodic",) * 3, 1000, "gaussian", 1e-3, 2.0, 0)
    # too few markers: h_min = 2 / 20 > h_max
    with pytest.raises(ValueError, match="too small"):
        setup_sph_kernel(cuboid, (12, 1, 1), ("periodic",) * 3, 20, "gaussian", None, 2.0, 0)
    # Colella is not affine in eta2, which has no sorting boxes
    colella = domains.Colella(Lx=1.0, Ly=1.0, alpha=0.05, Lz=1.0)
    with pytest.raises(ValueError, match="affine"):
        setup_sph_kernel(colella, (12, 1, 1), ("periodic",) * 3, 10_000, "gaussian", None, 2.0, 0)
    # two boxes in a periodic direction: both neighbour boxes hold the same markers (double counting)
    with pytest.raises(ValueError, match="at least 3 sorting boxes"):
        setup_sph_kernel(cuboid, (2, 1, 1), ("periodic",) * 3, 10_000, "gaussian", None, 2.0, 0)
    # ... but two boxes with walls are fine
    setup_sph_kernel(cuboid, (2, 1, 1), ("mirror", "periodic", "periodic"), 10_000, "gaussian", None, 2.0, 0)
    # no resolved direction
    with pytest.raises(ValueError, match="more than one sorting box"):
        setup_sph_kernel(cuboid, (1, 1, 1), ("periodic",) * 3, 10_000, "gaussian", None, 2.0, 0)
    # polar mapping with the axis: the angular box width vanishes at r = 0
    disc = domains.HollowCylinder(a1=0.0, a2=1.0, Lz=1.0)
    with pytest.raises(ValueError, match="annulus"):
        setup_sph_kernel(disc, (8, 8, 1), ("mirror", "periodic", "periodic"), 100_000, "gaussian", None, 2.0, 0)
    # an annulus works
    annulus = domains.HollowCylinder(a1=0.5, a2=1.0, Lz=1.0)
    k = setup_sph_kernel(annulus, (4, 16, 1), ("mirror", "periodic", "periodic"), 100_000, "gaussian", None, 2.0, 0)
    # limited by the angular box width at the inner radius, 2 pi a1 / 16
    # (spline approximation of the circle, hence the loose tolerance)
    assert k.h_max == pytest.approx(0.99 * min(0.5 / 4, 2 * xp.pi * 0.5 / 16), rel=0.05)


def test_sph_kernel_width_too_large():
    """ParticlesSPH sets up the kernel at its first use (species without SPH sums need none) and raises there for a
    kernel width larger than h_max (on all processes)."""
    comm, _ = _comm()
    background = ConstantVelocity(n=1.5, density_profile="constant")
    background.domain = domains.Cuboid()
    particles = ParticlesSPH(
        comm_world=comm,
        loading_params=LoadingParameters(ppb=8, loading="tesselation"),
        boundary_params=BoundaryParameters(bc_sph=("periodic",) * 3),
        sorting_params=SortingParameters(boxes_per_dim=(8, 1, 1)),
        sph_params=SPHParameters(kernel_width=0.2),
        domain=domains.Cuboid(),
        background=background,
        n_as_volume_form=True,
    )
    with pytest.raises(ValueError, match="larger than h_max"):
        particles.sph_kernel
    with pytest.raises(ValueError, match="larger than h_max"):
        particles.sph_kernel_args


if __name__ == "__main__":
    test_sph_evaluation_1d("gaussian", 0, "periodic", True, show_plot=True)
