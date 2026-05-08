import logging

import cunumpy as xp
import matplotlib.pyplot as plt
import pytest
from feectools.ddm.mpi import mpi as MPI

from struphy import domains, equils, set_logging_level
from struphy.feec.mass import L2Projector, WeightedMassOperators
from struphy.feec.psydac_derham import Derham
from struphy.geometry.base import Domain
from struphy.io.options import DerhamOptions
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.propagators.implicit_diffusion import ImplicitDiffusion
from struphy.topology.grids import TensorProductGrid

logger = logging.getLogger("struphy")
set_logging_level(logging.INFO)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
# plt.rcParams.update({'font.size': 22})


@pytest.mark.convergence
@pytest.mark.parametrize("direction", [0, 1])
@pytest.mark.parametrize("bc_type", ["periodic", "dirichlet", "neumann"])
@pytest.mark.parametrize(
    "mapping",
    [
        ["Cuboid", {"l1": 0.0, "r1": 4.0, "l2": 0.0, "r2": 2.0, "l3": 0.0, "r3": 3.0}],
        ["Orthogonal", {"Lx": 4.0, "Ly": 2.0, "alpha": 0.1, "Lz": 3.0}],
    ],
)
@pytest.mark.parametrize("projected_rhs", [False, True])
def test_poisson_M1perp_1d(direction, bc_type, mapping, projected_rhs, show_plot=False):
    """
    Test the convergence of Poisson solver with M1perp diffusion matrix
    in 1D by means of manufactured solutions.
    """

    # create domain object
    dom_type = mapping[0]
    dom_params = mapping[1]

    domain_class = getattr(domains, dom_type)
    domain: Domain = domain_class(**dom_params)

    if dom_type == "Cuboid":
        Lx = dom_params["r1"] - dom_params["l1"]
        Ly = dom_params["r2"] - dom_params["l2"]
        Lz = dom_params["r3"] - dom_params["l3"]
    else:
        Lx = dom_params["Lx"]
        Ly = dom_params["Ly"]
        Lz = dom_params["Lz"]

    Nels = [2**n for n in range(3, 9)]
    p_values = [1, 2]
    for pi in p_values:
        errors = []
        h_vec = []
        if show_plot:
            plt.figure(f"degree {pi =}, {direction + 1 =}, {bc_type =}, {mapping[0] =}", figsize=(24, 16))
            plt.figure(f"degree {pi =}, {direction + 1 =}, {bc_type =}, {mapping[0] =}", figsize=(24, 16))
            plt.figure(f"degree {pi =}, {direction + 1 =}, {bc_type =}, {mapping[0] =}", figsize=(24, 16))

        for n, Neli in enumerate(Nels):
            # boundary conditions (overwritten below)
            bcs = (None, None, None)

            # manufactured solution
            e1 = 0.0
            e2 = 0.0
            e3 = 0.0
            if direction == 0:
                num_elements = [Neli, 1, 1]
                degree = [pi, 1, 1]
                e1 = xp.linspace(0.0, 1.0, 50)

                if bc_type == "neumann":
                    bcs = (("free", "free"), None, None)

                    def sol1_xyz(x, y, z):
                        return xp.cos(xp.pi / Lx * x)

                    def rho1_xyz(x, y, z):
                        return xp.cos(xp.pi / Lx * x) * (xp.pi / Lx) ** 2
                else:
                    if bc_type == "dirichlet":
                        bcs = (("dirichlet", "dirichlet"), None, None)

                    def sol1_xyz(x, y, z):
                        return xp.sin(2 * xp.pi / Lx * x)

                    def rho1_xyz(x, y, z):
                        return xp.sin(2 * xp.pi / Lx * x) * (2 * xp.pi / Lx) ** 2

            elif direction == 1:
                num_elements = [1, Neli, 1]
                degree = [1, pi, 1]
                e2 = xp.linspace(0.0, 1.0, 50)

                if bc_type == "neumann":
                    bcs = (None, ("free", "free"), None)

                    def sol1_xyz(x, y, z):
                        return xp.cos(xp.pi / Ly * y)

                    def rho1_xyz(x, y, z):
                        return xp.cos(xp.pi / Ly * y) * (xp.pi / Ly) ** 2
                else:
                    if bc_type == "dirichlet":
                        bcs = (None, ("dirichlet", "dirichlet"), None)

                    def sol1_xyz(x, y, z):
                        return xp.sin(2 * xp.pi / Ly * y)

                    def rho1_xyz(x, y, z):
                        return xp.sin(2 * xp.pi / Ly * y) * (2 * xp.pi / Ly) ** 2
            else:
                logger.info("Direction should be either 0 or 1")

            # create derham object
            logger.info(f"{bcs =}")
            grid = TensorProductGrid(num_elements=num_elements)
            derham_opts = DerhamOptions(degree=degree, bcs=bcs)
            derham = Derham(grid, derham_opts, comm=comm)

            # mass matrices
            mass_ops = WeightedMassOperators(derham, domain)

            Propagator.derham = derham
            Propagator.domain = domain
            Propagator.mass_ops = mass_ops

            # pullbacks of right-hand side
            def rho_pulled(e1, e2, e3):
                return domain.pull(rho1_xyz, e1, e2, e3, kind="0", squeeze_out=False)

            # define how to pass rho
            if projected_rhs:
                rho = FEECVariable(space="H1")
                rho.allocate(derham=derham, domain=domain)
                rho.spline.vector = derham.P0(rho_pulled)
            else:
                rho = rho_pulled

            # create Poisson solver
            solver_params = SolverParameters(
                tol=1.0e-13,
                maxiter=3000,
                info=True,
                verbose=False,
                recycle=False,
            )

            _phi = FEECVariable(space="H1")
            _phi.allocate(derham=derham, domain=domain)

            poisson_solver = ImplicitDiffusion()
            poisson_solver.variables.phi = _phi

            poisson_solver.options = poisson_solver.Options(
                sigma_1=1e-12,
                sigma_2=0.0,
                sigma_3=1.0,
                divide_by_dt=True,
                diffusion_mat="M1perp",
                rho=rho,
                solver="pcg",
                precond="MassMatrixPreconditioner",
                solver_params=solver_params,
            )

            poisson_solver.allocate()

            # Solve Poisson (call propagator with dt=1.)
            dt = 1.0
            poisson_solver(dt)

            # push numerical solution and compare
            sol_val1 = domain.push(_phi.spline, e1, e2, e3, kind="0")
            x, y, z = domain(e1, e2, e3)
            analytic_value1 = sol1_xyz(x, y, z)

            if show_plot:
                plt.figure(f"degree {pi =}, {direction + 1 =}, {bc_type =}, {mapping[0] =}")
                plt.subplot(2, 3, n + 1)
                if direction == 0:
                    plt.plot(x[:, 0, 0], sol_val1[:, 0, 0], "ob", label="numerical")
                    plt.plot(x[:, 0, 0], analytic_value1[:, 0, 0], "r--", label="exact")
                    plt.xlabel("x")
                elif direction == 1:
                    plt.plot(y[0, :, 0], sol_val1[0, :, 0], "ob", label="numerical")
                    plt.plot(y[0, :, 0], analytic_value1[0, :, 0], "r--", label="exact")
                    plt.xlabel("y")
                plt.title(f"{num_elements =}")
                plt.legend()

            error = xp.max(xp.abs(analytic_value1 - sol_val1))
            logger.info(f"{direction =}, {pi =}, {Neli =}, {error=}")

            errors.append(error)
            h = 1 / (Neli)
            h_vec.append(h)

        m, _ = xp.polyfit(xp.log(Nels), xp.log(errors), deg=1)
        logger.info(f"For {pi =}, solution converges in {direction=} with rate {-m =} ")
        assert -m > (pi + 1 - 0.07)

        # Plot convergence in 1D
        if show_plot:
            plt.figure(
                f"Convergence for degree {pi =}, {direction + 1 =}, {bc_type =}, {mapping[0] =}",
                figsize=(12, 8),
            )
            plt.plot(h_vec, errors, "o", label=f"degree={degree[direction]}")
            plt.plot(
                h_vec,
                [
                    h ** (degree[direction] + 1) / h_vec[direction] ** (degree[direction] + 1) * errors[direction]
                    for h in h_vec
                ],
                "k--",
                label="correct rate degree+1",
            )
            plt.yscale("log")
            plt.xscale("log")
            plt.xlabel("Grid Spacing h")
            plt.ylabel("Error")
            plt.title("Poisson solver")
            plt.legend()

    if show_plot and rank == 0:
        plt.show()


@pytest.mark.parametrize("num_elements", [[64, 64, 1]])
@pytest.mark.parametrize("degree", [[1, 1, 1], [2, 2, 1]])
@pytest.mark.parametrize("bc_type", ["periodic", "dirichlet", "neumann"])
@pytest.mark.parametrize(
    "mapping",
    [
        ["Cuboid", {"l1": 0.0, "r1": 4.0, "l2": 0.0, "r2": 2.0, "l3": 0.0, "r3": 1.0}],
        ["Orthogonal", {"Lx": 4.0, "Ly": 2.0, "alpha": 0.1, "Lz": 1.0}],
    ],
)
@pytest.mark.parametrize("projected_rhs", [False, True])
def test_poisson_M1perp_2d(num_elements, degree, bc_type, mapping, projected_rhs, show_plot=False):
    """
    Test the Poisson solver with M1perp diffusion matrix
    by means of manufactured solutions in 2D .
    """

    # create domain object
    dom_type = mapping[0]
    dom_params = mapping[1]

    domain_class = getattr(domains, dom_type)
    domain: Domain = domain_class(**dom_params)

    if dom_type == "Cuboid":
        Lx = dom_params["r1"] - dom_params["l1"]
        Ly = dom_params["r2"] - dom_params["l2"]
    else:
        Lx = dom_params["Lx"]
        Ly = dom_params["Ly"]

    # manufactured solution in 1D (overwritten for "neumann")
    def sol1_xyz(x, y, z):
        return xp.sin(2 * xp.pi / Lx * x)

    def rho1_xyz(x, y, z):
        return xp.sin(2 * xp.pi / Lx * x) * (2 * xp.pi / Lx) ** 2

    # boundary conditions
    if bc_type == "periodic":
        bcs = (None, None, None)

        # manufactured solution in 2D
        def sol2_xyz(x, y, z):
            return xp.sin(2 * xp.pi * x / Lx + 4 * xp.pi / Ly * y)

        def rho2_xyz(x, y, z):
            ddx = xp.sin(2 * xp.pi / Lx * x + 4 * xp.pi / Ly * y) * (2 * xp.pi / Lx) ** 2
            ddy = xp.sin(2 * xp.pi / Lx * x + 4 * xp.pi / Ly * y) * (4 * xp.pi / Ly) ** 2
            return ddx + ddy

    elif bc_type == "dirichlet":
        bcs = (("dirichlet", "dirichlet"), None, None)

        # manufactured solution in 2D
        def sol2_xyz(x, y, z):
            return xp.sin(xp.pi * x / Lx) * xp.sin(4 * xp.pi / Ly * y)

        def rho2_xyz(x, y, z):
            ddx = xp.sin(xp.pi * x / Lx) * xp.sin(4 * xp.pi / Ly * y) * (xp.pi / Lx) ** 2
            ddy = xp.sin(xp.pi * x / Lx) * xp.sin(4 * xp.pi / Ly * y) * (4 * xp.pi / Ly) ** 2
            return ddx + ddy

    elif bc_type == "neumann":
        bcs = (("free", "free"), None, None)

        # manufactured solution in 2D
        def sol2_xyz(x, y, z):
            return xp.cos(xp.pi * x / Lx) * xp.sin(4 * xp.pi / Ly * y)

        def rho2_xyz(x, y, z):
            ddx = xp.cos(xp.pi * x / Lx) * xp.sin(4 * xp.pi / Ly * y) * (xp.pi / Lx) ** 2
            ddy = xp.cos(xp.pi * x / Lx) * xp.sin(4 * xp.pi / Ly * y) * (4 * xp.pi / Ly) ** 2
            return ddx + ddy

        # manufactured solution in 1D
        def sol1_xyz(x, y, z):
            return xp.cos(xp.pi / Lx * x)

        def rho1_xyz(x, y, z):
            return xp.cos(xp.pi / Lx * x) * (xp.pi / Lx) ** 2

    # create derham object
    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs)
    derham = Derham(grid, derham_opts, comm=comm)

    # create weighted mass operators
    mass_ops = WeightedMassOperators(derham, domain)

    Propagator.derham = derham
    Propagator.domain = domain
    Propagator.mass_ops = mass_ops

    # evaluation grid
    e1 = xp.linspace(0.0, 1.0, 50)
    e2 = xp.linspace(0.0, 1.0, 50)
    e3 = xp.linspace(0.0, 1.0, 1)

    # pullbacks of right-hand side
    def rho1_pulled(e1, e2, e3):
        return domain.pull(rho1_xyz, e1, e2, e3, kind="0", squeeze_out=False)

    def rho2_pulled(e1, e2, e3):
        return domain.pull(rho2_xyz, e1, e2, e3, kind="0", squeeze_out=False)

    # how to pass right-hand sides
    if projected_rhs:
        rho1 = FEECVariable(space="H1")
        rho1.allocate(derham=derham, domain=domain)
        rho1.spline.vector = derham.P0(rho1_pulled)

        rho2 = FEECVariable(space="H1")
        rho2.allocate(derham=derham, domain=domain)
        rho2.spline.vector = derham.P0(rho2_pulled)
    else:
        rho1 = rho1_pulled
        rho2 = rho2_pulled

    # Create Poisson solvers
    solver_params = SolverParameters(
        tol=1.0e-13,
        maxiter=3000,
        info=True,
        verbose=False,
        recycle=False,
    )

    _phi1 = FEECVariable(space="H1")
    _phi1.allocate(derham=derham, domain=domain)

    poisson_solver1 = ImplicitDiffusion()
    poisson_solver1.variables.phi = _phi1

    poisson_solver1.options = poisson_solver1.Options(
        sigma_1=1e-8,
        sigma_2=0.0,
        sigma_3=1.0,
        divide_by_dt=True,
        diffusion_mat="M1perp",
        rho=rho1,
        solver="pcg",
        precond="MassMatrixPreconditioner",
        solver_params=solver_params,
    )

    poisson_solver1.allocate()

    _phi2 = FEECVariable(space="H1")
    _phi2.allocate(derham=derham, domain=domain)

    poisson_solver2 = ImplicitDiffusion()
    poisson_solver2.variables.phi = _phi2

    poisson_solver2.options = poisson_solver2.Options(
        sigma_1=1e-8,
        sigma_2=0.0,
        sigma_3=1.0,
        divide_by_dt=True,
        diffusion_mat="M1perp",
        rho=rho2,
        solver="pcg",
        precond="MassMatrixPreconditioner",
        solver_params=solver_params,
    )

    poisson_solver2.allocate()

    # Solve Poisson equation (call propagator with dt=1.)
    dt = 1.0
    poisson_solver1(dt)
    poisson_solver2(dt)

    # push numerical solutions
    sol_val1 = domain.push(_phi1.spline, e1, e2, e3, kind="0")
    sol_val2 = domain.push(_phi2.spline, e1, e2, e3, kind="0")

    x, y, z = domain(e1, e2, e3)
    analytic_value1 = sol1_xyz(x, y, z)
    analytic_value2 = sol2_xyz(x, y, z)

    # compute error
    error1 = xp.max(xp.abs(analytic_value1 - sol_val1))
    error2 = xp.max(xp.abs(analytic_value2 - sol_val2))

    logger.info(f"{degree =}, {bc_type =}, {mapping =}")
    logger.info(f"{error1 =}")
    logger.info(f"{error2 =}")
    logger.info("")

    if show_plot and rank == 0:
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.title("1D solution")
        plt.plot(x[:, 0, 0], sol_val1[:, 0, 0], "ob", label="numerical")
        plt.plot(x[:, 0, 0], analytic_value1[:, 0, 0], "r--", label="exact")
        plt.legend()
        plt.subplot(2, 2, 2)
        plt.title("2D numerical solution")
        plt.pcolor(x[:, :, 0], y[:, :, 0], sol_val2[:, :, 0], vmin=-1.0, vmax=1.0)
        plt.colorbar()
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")
        plt.subplot(2, 2, 4)
        plt.title("2D true solution")
        plt.pcolor(x[:, :, 0], y[:, :, 0], analytic_value2[:, :, 0], vmin=-1.0, vmax=1.0)
        plt.colorbar()
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")

        plt.show()

    assert error1 < 0.0044
    assert error2 < 0.023


@pytest.mark.parametrize("num_elements", [[32, 32, 16]])
@pytest.mark.parametrize("degree", [[1, 1, 1], [2, 2, 1]])
@pytest.mark.parametrize(
    "mapping",
    [
        ["Cuboid", {"l1": 0.0, "r1": 1.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}],
        ["Colella", {"Lx": 1.0, "Ly": 1.0, "alpha": 0.1, "Lz": 1.0}],
    ],
)
def test_poisson_M1perp_3d_compare_M1(num_elements, degree, mapping, show_plot=False):
    """
    Test the Poisson solver with M1perp diffusion matrix
    by comparing 3d simulation using M1perp and M1 diffusion matrices, and integrating over the third direction.
    The two analytical solutions must be exactly the same in both cases.
    """

    from time import time

    # create domain object
    dom_type = mapping[0]
    dom_params = mapping[1]

    domain_class = getattr(domains, dom_type)
    domain: Domain = domain_class(**dom_params)

    equil = equils.HomogenSlab()
    equil.domain = domain

    # evaluation grid
    e1 = xp.linspace(0.0, 1.0, num_elements[0])
    e2 = xp.linspace(0.0, 1.0, num_elements[1])
    e3 = xp.linspace(0.0, 1.0, num_elements[2])

    # solution and right-hand side on unit cube
    def rho(e1, e2, e3):
        dd1 = xp.sin(xp.pi * e1) * xp.sin(2 * xp.pi * e2) * (1 + xp.cos(2 * xp.pi * e3)) * (xp.pi) ** 2
        return dd1

    # create 3d derham object
    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=(None, None, None))
    derham = Derham(grid, derham_opts, comm=comm)

    mass_ops = WeightedMassOperators(derham, domain, eq_mhd=equil)

    Propagator.derham = derham
    Propagator.domain = domain
    Propagator.mass_ops = mass_ops

    # discrete right-hand sides
    # l2_proj = L2Projector("H1", mass_ops)

    # Create 3d Poisson solver
    solver_params = SolverParameters(
        tol=1.0e-13,
        maxiter=3000,
        info=True,
        verbose=False,
        recycle=False,
    )

    _phi_M1 = FEECVariable(space="H1")
    _phi_M1.allocate(derham=derham, domain=domain)

    poisson_solver_M1 = ImplicitDiffusion()
    poisson_solver_M1.variables.phi = _phi_M1

    poisson_solver_M1.options = poisson_solver_M1.Options(
        sigma_1=1e-8,
        sigma_2=0.0,
        sigma_3=1.0,
        divide_by_dt=False,
        diffusion_mat="M1",
        rho=rho,
        solver="pcg",
        precond="MassMatrixPreconditioner",
        solver_params=solver_params,
    )

    poisson_solver_M1.allocate()

    s = _phi_M1.spline.starts
    e = _phi_M1.spline.ends

    _phi_M1perp = FEECVariable(space="H1")
    _phi_M1perp.allocate(derham=derham, domain=domain)

    poisson_solver_M1perp = ImplicitDiffusion()
    poisson_solver_M1perp.variables.phi = _phi_M1perp

    poisson_solver_M1perp.options = poisson_solver_M1perp.Options(
        sigma_1=1e-8,
        sigma_2=0.0,
        sigma_3=1.0,
        divide_by_dt=False,
        diffusion_mat="M1perp",
        rho=rho,
        solver="pcg",
        precond="MassMatrixPreconditioner",
        solver_params=solver_params,
    )

    poisson_solver_M1perp.allocate()

    # Solve M1 Poisson equation (call propagator with dt=1.)
    dt = 1.0
    t0 = time()
    poisson_solver_M1(dt)
    t1 = time()
    logger.info(f"rank {rank}, M1 3d solve time = {t1 - t0}")

    # Solve M1perp Poisson equation (call propagator with dt=1.)
    t0 = time()
    poisson_solver_M1perp(dt)
    t1 = time()
    logger.info(f"rank {rank}, M1perp 3d solve time = {t1 - t0}")

    # push numerical solutions
    sol_val_M1 = domain.push(_phi_M1.spline, e1, e2, e3, kind="0")
    sol_val_M1perp = domain.push(_phi_M1perp.spline, e1, e2, e3, kind="0")
    x, y, z = domain(e1, e2, e3)

    logger.info(f"max diff: {xp.max(xp.abs(sol_val_M1 - sol_val_M1perp))}")
    logger.info(
        f"max diff of the averaged solutions (over e3): {xp.max(xp.abs(xp.trapezoid(sol_val_M1 - sol_val_M1perp, e3, axis=2) / (e3[-1] - e3[0])))}"
    )
    assert xp.max(xp.abs(xp.trapezoid(sol_val_M1 - sol_val_M1perp, e3, axis=2)/(e3[-1]-e3[0]))) < 0.001
    if show_plot and rank == 0:
        plt.figure("e1-e2 plane", figsize=(24, 16))
        plt.subplot(2, 3, 1)
        plt.title("charge density averaged over e3")
        plt.pcolor(
            x[:, :, 0],
            y[:, :, 0],
            xp.transpose(xp.sum(rho(*xp.meshgrid(e1, e2, e3)), axis=2)) / len(e3),
            shading="nearest",
        )
        plt.colorbar()
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")
        plt.subplot(2, 3, 4)
        plt.title(f"charge density at e3={e3[len(e3) // 2]:.2f}")
        plt.pcolor(
            x[:, :, 0], y[:, :, 0], xp.transpose(rho(*xp.meshgrid(e1, e2, e3))[:, :, len(e3) // 2]), shading="nearest"
        )
        plt.colorbar()
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")
        plt.subplot(2, 3, 2)
        plt.title(f"phi at e3={e3[len(e3) // 2]:.2f} with M1 solve")
        plt.pcolor(x[:, :, 0], y[:, :, 0], sol_val_M1[:, :, len(e3) // 2])
        plt.colorbar()
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")
        plt.subplot(2, 3, 5)
        plt.title(f"phi at e3={e3[len(e3) // 2]:.2f} with M1perp solve")
        plt.pcolor(x[:, :, 0], y[:, :, 0], sol_val_M1perp[:, :, len(e3) // 2])
        plt.colorbar()
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")
        plt.subplot(2, 3, 3)
        plt.title("average over e3 of M1 solve")
        plt.pcolor(x[:, :, 0], y[:, :, 0], xp.trapezoid(sol_val_M1, e3, axis=2) / (e3[-1] - e3[0]))
        plt.colorbar()
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")
        plt.subplot(2, 3, 6)
        plt.title("average over e3 of M1perp solve")
        plt.pcolor(x[:, :, 0], y[:, :, 0], xp.trapezoid(sol_val_M1perp, e3, axis=2) / (e3[-1] - e3[0]))
        plt.colorbar()
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")
        plt.show()

#@pytest.mark.skip(reason="Not clear if the 2.5d strategy is sound.")
@pytest.mark.parametrize("num_elements", [[32, 32, 16]])
@pytest.mark.parametrize("degree", [[1, 1, 1], [2, 2, 1]])
@pytest.mark.parametrize(
    "mapping",
    [
        ["Cuboid", {"l1": 0.0, "r1": 1.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}],
    ],
)
def test_poisson_M1perp_3d_compare_2p5d(num_elements, degree, mapping, show_plot=False):
    """
    Test the Poisson solver with M1perp diffusion matrix
    by comparing 3d simulation to a loop over 2d simulations.
    """

    from time import time

    # create domain object
    dom_type = mapping[0]
    dom_params = mapping[1]

    domain_class = getattr(domains, dom_type)
    domain_3D: Domain = domain_class(**dom_params)

    # boundary conditions
    bcs = (None, None, ("free", "free"))

    # evaluation grid
    e1 = xp.linspace(0.0, 1.0, num_elements[0])
    e2 = xp.linspace(0.0, 1.0, num_elements[1])
    e3 = xp.linspace(0.0, 1.0, num_elements[2])

    # solution and right-hand side on unit cube
    def rho_physical(x, y, z):
        dd1 = xp.sin(2 * xp.pi * x) * xp.sin(xp.pi * y) * xp.cos(xp.pi * z) * (xp.pi) ** 2
        return dd1

    # create 3d derham object
    grid_3D = TensorProductGrid(num_elements=num_elements)
    derham_opts_3D = DerhamOptions(degree=degree, bcs=bcs)
    derham_3D = Derham(grid_3D, derham_opts_3D, comm=comm)

    mass_ops_3D = WeightedMassOperators(derham_3D, domain_3D, eq_mhd=equils.HomogenSlab(B0x=0.0, B0y=0.0, B0z=1.0))

    Propagator.derham = derham_3D
    Propagator.domain = domain_3D
    Propagator.mass_ops = mass_ops_3D

    rho_logical_3D = lambda e1, e2, e3: rho_physical(*domain_3D(e1, e2, e3))

    # discrete right-hand sides
    #l2_proj = L2Projector("H1", mass_ops_3D)
    # rho_vec = l2_proj.get_dofs(rho, apply_bc=True)

    # logger.info(f"{rho_vec[:].shape =}")

    # Create 3d Poisson solver
    solver_params = SolverParameters(
        tol=1.0e-13,
        maxiter=3000,
        info=True,
        verbose=False,
        recycle=False,
    )

    _phi = FEECVariable(space="H1")
    _phi.allocate(derham=derham_3D, domain=domain_3D)

    _phi_2p5d = FEECVariable(space="H1")
    _phi_2p5d.allocate(derham=derham_3D, domain=domain_3D)

    poisson_solver_3d = ImplicitDiffusion()
    poisson_solver_3d.variables.phi = _phi

    poisson_solver_3d.options = poisson_solver_3d.Options(
        sigma_1=1e-8,
        sigma_2=0.0,
        sigma_3=1.0,
        divide_by_dt=True,
        diffusion_mat="M1perp",
        rho=rho_logical_3D,
        solver="pcg",
        precond="MassMatrixPreconditioner",
        solver_params=solver_params,
    )

    poisson_solver_3d.allocate()

    s = _phi.spline.starts
    e = _phi.spline.ends

    # Solve 3d Poisson equation (call propagator with dt=1.)
    dt = 1.0
    t0 = time()
    poisson_solver_3d(dt)
    t1 = time()

    logger.info(f"rank {rank}, 3d solve time = {t1 - t0}")

    # create 2.5d deRham object for each slice in e3, and solve 2d Poisson equations
    num_elements_new = [num_elements[0], num_elements[1], 1]
    degree_sliced = degree.copy()
    degree_sliced[2] = 1
    dom_params_sliced = dom_params.copy()
    t0 = time()
    t_inner = 0.0
    for n in range(s[2], e[2] + 1):
        dom_params_sliced["l3"] = dom_params["l3"] + (dom_params["r3"] - dom_params["l3"]) / (len(e3) + degree[2]) * n
        dom_params_sliced["r3"] = dom_params["l3"] + (dom_params["r3"] - dom_params["l3"]) / (len(e3) + degree[2]) * (
            n + 1
        )
        domain_sliced = domain_class(**dom_params_sliced)

        grid_sliced = TensorProductGrid(num_elements=num_elements_new)
        derham_opts_sliced = DerhamOptions(degree=degree_sliced, bcs=bcs)
        derham_sliced = Derham(grid_sliced, derham_opts_sliced, comm=comm)

        mass_ops_sliced = WeightedMassOperators(derham_sliced, domain_sliced)

        Propagator.derham = derham_sliced
        Propagator.mass_ops = mass_ops_sliced
        Propagator.domain = domain_sliced

        rho_logical_sliced = lambda e1, e2, e3: rho_physical(*domain_sliced(e1, e2, e3))

        _phi_small = FEECVariable(space="H1")
        _phi_small.allocate(derham=derham_sliced, domain=domain_sliced)

        poisson_solver_2p5d = ImplicitDiffusion()
        poisson_solver_2p5d.variables.phi = _phi_small

        poisson_solver_2p5d.options = poisson_solver_2p5d.Options(
            sigma_1=1e-8,
            sigma_2=0.0,
            sigma_3=1.0,
            divide_by_dt=True,
            diffusion_mat="M1",
            rho=rho_logical_sliced,
            solver="pcg",
            precond="MassMatrixPreconditioner",
            solver_params=solver_params,
        )
        poisson_solver_2p5d.allocate()

        t0i = time()
        poisson_solver_2p5d(dt)
        t1i = time()
        t_inner += t1i - t0i
        _tmp = _phi_small.spline.vector.copy()
        _phi_2p5d.spline.vector[s[0] : e[0] + 1, s[1] : e[1] + 1, n] = _tmp[s[0] : e[0] + 1, s[1] : e[1] + 1, 0]
    t1 = time()

    logger.info(f"rank {rank}, 2.5d pure solve time (without copy) = {t_inner}")
    logger.info(f"rank {rank}, 2.5d solve time = {t1 - t0}")

    # push numerical solutions
    sol_val = domain_3D.push(_phi.spline, e1, e2, e3, kind="0")
    sol_val_2p5d = domain_3D.push(_phi_2p5d.spline, e1, e2, e3, kind="0")
    x, y, z = domain_3D(e1, e2, e3)

    logger.info(f"mean diff: {xp.mean(xp.abs(sol_val - sol_val_2p5d))}")
    logger.info(f"max diff: {xp.max(xp.abs(sol_val - sol_val_2p5d))}")
    assert xp.max(xp.abs(sol_val - sol_val_2p5d)) < 0.01

    if show_plot and rank == 0:
        plt.figure("e1-e2 plane", figsize=(24, 16))
        plot_id_e3 = [0, len(e3) // 2, len(e3) - 1]
        plot_id_e2 = [0, len(e2) // 2, len(e2) - 1]
        for n in range(3):
            plt.subplot(2, 3, n + 1)
            plt.title(f"e3 = {e3[plot_id_e3[n]]} from 3d solve")
            plt.pcolor(x[:, :, plot_id_e3[n]], y[:, :, plot_id_e3[n]], sol_val[:, :, plot_id_e3[n]])
            plt.colorbar()
            ax = plt.gca()
            ax.set_aspect("equal", adjustable="box")
            plt.subplot(2, 3, 4 + n)
            plt.title(f"e3 = {e3[plot_id_e3[n]]} from 2.5d solve")
            plt.pcolor(x[:, :, plot_id_e3[n]], y[:, :, plot_id_e3[n]], sol_val_2p5d[:, :, plot_id_e3[n]])
            plt.colorbar()
            ax = plt.gca()
            ax.set_aspect("equal", adjustable="box")
        plt.figure("e1-e3 plane", figsize=(24, 16))
        for n in range(3):
            plt.subplot(2, 3, n + 1)
            plt.title(f"e2 = {e2[plot_id_e2[n]]} from 3d solve")
            plt.pcolor(x[:, plot_id_e2[n], :], z[:, plot_id_e2[n], :], sol_val[:, plot_id_e2[n], :])
            plt.colorbar()
            ax = plt.gca()
            ax.set_aspect("equal", adjustable="box")
            plt.subplot(2, 3, 4 + n)
            plt.title(f"e2 = {e2[plot_id_e2[n]]} from 2.5d solve")
            plt.pcolor(x[:, plot_id_e2[n], :], z[:, plot_id_e2[n], :], sol_val_2p5d[:, plot_id_e2[n], :])
            plt.colorbar()
            ax = plt.gca()
            ax.set_aspect("equal", adjustable="box")

        plt.show()


if __name__ == "__main__":
    direction = 0
    bc_type = "dirichlet"
    mapping = ["Cuboid", {"l1": 0.0, "r1": 4.0, "l2": 0.0, "r2": 2.0, "l3": 0.0, "r3": 3.0}]
    # mapping = ["Orthogonal", {"Lx": 4.0, "Ly": 2.0, "alpha": 0.1, "Lz": 3.0}]
    # test_poisson_M1perp_1d(direction, bc_type, mapping, projected_rhs=True, show_plot=True)

    num_elements = [64, 64, 1]
    degree = [2, 2, 1]
    bc_type = "neumann"
    mapping = ["Cuboid", {"l1": 0.0, "r1": 4.0, "l2": 0.0, "r2": 2.0, "l3": 0.0, "r3": 3.0}]
    mapping = ["Orthogonal", {"Lx": 4.0, "Ly": 2.0, "alpha": 0.1, "Lz": 1.0}]
    # test_poisson_M1perp_2d(num_elements, degree, bc_type, mapping, projected_rhs=True, show_plot=True)

    num_elements = [50, 50, 50]
    degree = [2, 2, 1]
    mapping = ["Cuboid", {"l1": 0.0, "r1": 1.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}]
    # test_poisson_M1perp_3d_compare_M1(num_elements, degree, mapping, show_plot=True)
    num_elements = [50, 50, 50]
    # test_poisson_M1perp_3d_compare_2p5d(num_elements, degree, mapping, show_plot=True)
