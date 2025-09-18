import matplotlib.pyplot as plt
import pytest
from psydac.ddm.mpi import mpi as MPI

from struphy.feec.mass import WeightedMassOperators
from struphy.feec.projectors import L2Projector
from struphy.feec.psydac_derham import Derham
from struphy.geometry import domains
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.utils.arrays import xp as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
# plt.rcParams.update({'font.size': 22})


@pytest.mark.parametrize("direction", [0, 1])
@pytest.mark.parametrize("bc_type", ["periodic", "dirichlet", "neumann"])
@pytest.mark.parametrize(
    "mapping",
    [
        ["Cuboid", {"l1": 0.0, "r1": 4.0, "l2": 0.0, "r2": 2.0, "l3": 0.0, "r3": 3.0}],
        ["Orthogonal", {"Lx": 4.0, "Ly": 2.0, "alpha": 0.1, "Lz": 3.0}],
    ],
)
def test_poisson_M1perp_1d(direction, bc_type, mapping, show_plot=False):
    """
    Test the convergence of Poisson solver with M1perp diffusion matrix
    in 1D by means of manufactured solutions.
    """

    # create domain object
    dom_type = mapping[0]
    dom_params = mapping[1]

    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

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
            plt.figure(f"degree {pi = }, {direction + 1 = }, {bc_type = }, {mapping[0] = }", figsize=(24, 16))
            plt.figure(f"degree {pi = }, {direction + 1 = }, {bc_type = }, {mapping[0] = }", figsize=(24, 16))
            plt.figure(f"degree {pi = }, {direction + 1 = }, {bc_type = }, {mapping[0] = }", figsize=(24, 16))

        for n, Neli in enumerate(Nels):
            # boundary conditions (overwritten below)
            spl_kind = [True, True, True]
            dirichlet_bc = None

            # manufactured solution
            e1 = 0.0
            e2 = 0.0
            e3 = 0.0
            if direction == 0:
                Nel = [Neli, 1, 1]
                p = [pi, 1, 1]
                e1 = np.linspace(0.0, 1.0, 50)

                if bc_type == "neumann":
                    spl_kind = [False, True, True]

                    def sol1_xyz(x, y, z):
                        return np.cos(np.pi / Lx * x)

                    def rho1_xyz(x, y, z):
                        return np.cos(np.pi / Lx * x) * (np.pi / Lx) ** 2
                else:
                    if bc_type == "dirichlet":
                        spl_kind = [False, True, True]
                        dirichlet_bc = [(not kd,) * 2 for kd in spl_kind]
                        dirichlet_bc = tuple(dirichlet_bc)

                    def sol1_xyz(x, y, z):
                        return np.sin(2 * np.pi / Lx * x)

                    def rho1_xyz(x, y, z):
                        return np.sin(2 * np.pi / Lx * x) * (2 * np.pi / Lx) ** 2

            elif direction == 1:
                Nel = [1, Neli, 1]
                p = [1, pi, 1]
                e2 = np.linspace(0.0, 1.0, 50)

                if bc_type == "neumann":
                    spl_kind = [True, False, True]

                    def sol1_xyz(x, y, z):
                        return np.cos(np.pi / Ly * y)

                    def rho1_xyz(x, y, z):
                        return np.cos(np.pi / Ly * y) * (np.pi / Ly) ** 2
                else:
                    if bc_type == "dirichlet":
                        spl_kind = [True, False, True]
                        dirichlet_bc = [(not kd,) * 2 for kd in spl_kind]
                        dirichlet_bc = tuple(dirichlet_bc)

                    def sol1_xyz(x, y, z):
                        return np.sin(2 * np.pi / Ly * y)

                    def rho1_xyz(x, y, z):
                        return np.sin(2 * np.pi / Ly * y) * (2 * np.pi / Ly) ** 2
            else:
                print("Direction should be either 0 or 1")

            # create derham object
            print(f"{dirichlet_bc = }")
            derham = Derham(Nel, p, spl_kind, dirichlet_bc=dirichlet_bc, comm=comm)

            # mass matrices
            mass_ops = WeightedMassOperators(derham, domain)

            Propagator.derham = derham
            Propagator.domain = domain
            Propagator.mass_ops = mass_ops

            # pullbacks of right-hand side
            def rho1(e1, e2, e3):
                return domain.pull(rho1_xyz, e1, e2, e3, kind="0", squeeze_out=True)

            rho_vec = L2Projector("H1", mass_ops).get_dofs(rho1, apply_bc=True)

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
                rho=rho_vec,
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
                plt.figure(f"degree {pi = }, {direction + 1 = }, {bc_type = }, {mapping[0] = }")
                plt.subplot(2, 3, n + 1)
                if direction == 0:
                    plt.plot(x[:, 0, 0], sol_val1[:, 0, 0], "ob", label="numerical")
                    plt.plot(x[:, 0, 0], analytic_value1[:, 0, 0], "r--", label="exact")
                    plt.xlabel("x")
                elif direction == 1:
                    plt.plot(y[0, :, 0], sol_val1[0, :, 0], "ob", label="numerical")
                    plt.plot(y[0, :, 0], analytic_value1[0, :, 0], "r--", label="exact")
                    plt.xlabel("y")
                plt.title(f"{Nel = }")
                plt.legend()

            error = np.max(np.abs(analytic_value1 - sol_val1))
            print(f"{direction = }, {pi = }, {Neli = }, {error=}")

            errors.append(error)
            h = 1 / (Neli)
            h_vec.append(h)

        m, _ = np.polyfit(np.log(Nels), np.log(errors), deg=1)
        print(f"For {pi = }, solution converges in {direction=} with rate {-m = } ")
        assert -m > (pi + 1 - 0.06)

        # Plot convergence in 1D
        if show_plot:
            plt.figure(
                f"Convergence for degree {pi = }, {direction + 1 = }, {bc_type = }, {mapping[0] = }", figsize=(12, 8)
            )
            plt.plot(h_vec, errors, "o", label=f"p={p[direction]}")
            plt.plot(
                h_vec,
                [h ** (p[direction] + 1) / h_vec[direction] ** (p[direction] + 1) * errors[direction] for h in h_vec],
                "k--",
                label="correct rate p+1",
            )
            plt.yscale("log")
            plt.xscale("log")
            plt.xlabel("Grid Spacing h")
            plt.ylabel("Error")
            plt.title(f"Poisson solver")
            plt.legend()

    if show_plot and rank == 0:
        plt.show()


@pytest.mark.parametrize("Nel", [[64, 64, 1]])
@pytest.mark.parametrize("p", [[1, 1, 1], [2, 2, 1]])
@pytest.mark.parametrize("bc_type", ["periodic", "dirichlet", "neumann"])
@pytest.mark.parametrize(
    "mapping",
    [
        ["Cuboid", {"l1": 0.0, "r1": 4.0, "l2": 0.0, "r2": 2.0, "l3": 0.0, "r3": 1.0}],
        ["Orthogonal", {"Lx": 4.0, "Ly": 2.0, "alpha": 0.1, "Lz": 1.0}],
    ],
)
def test_poisson_M1perp_2d(Nel, p, bc_type, mapping, show_plot=False):
    """
    Test the Poisson solver with M1perp diffusion matrix
    by means of manufactured solutions in 2D .
    """

    # create domain object
    dom_type = mapping[0]
    dom_params = mapping[1]

    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    if dom_type == "Cuboid":
        Lx = dom_params["r1"] - dom_params["l1"]
        Ly = dom_params["r2"] - dom_params["l2"]
    else:
        Lx = dom_params["Lx"]
        Ly = dom_params["Ly"]

    # manufactured solution in 1D (overwritten for "neumann")
    def sol1_xyz(x, y, z):
        return np.sin(2 * np.pi / Lx * x)

    def rho1_xyz(x, y, z):
        return np.sin(2 * np.pi / Lx * x) * (2 * np.pi / Lx) ** 2

    # boundary conditions
    dirichlet_bc = None

    if bc_type == "periodic":
        spl_kind = [True] * 3

        # manufactured solution in 2D
        def sol2_xyz(x, y, z):
            return np.sin(2 * np.pi * x / Lx + 4 * np.pi / Ly * y)

        def rho2_xyz(x, y, z):
            ddx = np.sin(2 * np.pi / Lx * x + 4 * np.pi / Ly * y) * (2 * np.pi / Lx) ** 2
            ddy = np.sin(2 * np.pi / Lx * x + 4 * np.pi / Ly * y) * (4 * np.pi / Ly) ** 2
            return ddx + ddy

    elif bc_type == "dirichlet":
        spl_kind = [False, True, True]
        dirichlet_bc = [(not kd,) * 2 for kd in spl_kind]
        dirichlet_bc = tuple(dirichlet_bc)
        print(f"{dirichlet_bc = }")

        # manufactured solution in 2D
        def sol2_xyz(x, y, z):
            return np.sin(np.pi * x / Lx) * np.sin(4 * np.pi / Ly * y)

        def rho2_xyz(x, y, z):
            ddx = np.sin(np.pi * x / Lx) * np.sin(4 * np.pi / Ly * y) * (np.pi / Lx) ** 2
            ddy = np.sin(np.pi * x / Lx) * np.sin(4 * np.pi / Ly * y) * (4 * np.pi / Ly) ** 2
            return ddx + ddy

    elif bc_type == "neumann":
        spl_kind = [False, True, True]

        # manufactured solution in 2D
        def sol2_xyz(x, y, z):
            return np.cos(np.pi * x / Lx) * np.sin(4 * np.pi / Ly * y)

        def rho2_xyz(x, y, z):
            ddx = np.cos(np.pi * x / Lx) * np.sin(4 * np.pi / Ly * y) * (np.pi / Lx) ** 2
            ddy = np.cos(np.pi * x / Lx) * np.sin(4 * np.pi / Ly * y) * (4 * np.pi / Ly) ** 2
            return ddx + ddy

        # manufactured solution in 1D
        def sol1_xyz(x, y, z):
            return np.cos(np.pi / Lx * x)

        def rho1_xyz(x, y, z):
            return np.cos(np.pi / Lx * x) * (np.pi / Lx) ** 2

    # create derham object
    derham = Derham(Nel, p, spl_kind, dirichlet_bc=dirichlet_bc, comm=comm)

    # create weighted mass operators
    mass_ops = WeightedMassOperators(derham, domain)

    Propagator.derham = derham
    Propagator.domain = domain
    Propagator.mass_ops = mass_ops

    # evaluation grid
    e1 = np.linspace(0.0, 1.0, 50)
    e2 = np.linspace(0.0, 1.0, 50)
    e3 = np.linspace(0.0, 1.0, 1)

    # pullbacks of right-hand side
    def rho1(e1, e2, e3):
        return domain.pull(rho1_xyz, e1, e2, e3, kind="0", squeeze_out=True)

    def rho2(e1, e2, e3):
        return domain.pull(rho2_xyz, e1, e2, e3, kind="0", squeeze_out=True)

    # discrete right-hand sides
    l2_proj = L2Projector("H1", mass_ops)
    rho_vec1 = l2_proj.get_dofs(rho1, apply_bc=True)
    rho_vec2 = l2_proj.get_dofs(rho2, apply_bc=True)

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
        rho=rho_vec1,
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
        rho=rho_vec2,
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
    error1 = np.max(np.abs(analytic_value1 - sol_val1))
    error2 = np.max(np.abs(analytic_value2 - sol_val2))

    print(f"{p = }, {bc_type = }, {mapping = }")
    print(f"{error1 = }")
    print(f"{error2 = }")
    print("")

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
    assert error2 < 0.021


@pytest.mark.skip(reason="Not clear if the 2.5d strategy is sound.")
@pytest.mark.parametrize("Nel", [[32, 32, 16]])
@pytest.mark.parametrize("p", [[1, 1, 1], [2, 2, 1]])
@pytest.mark.parametrize(
    "mapping",
    [
        ["Cuboid", {"l1": 0.0, "r1": 1.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}],
        ["Colella", {"Lx": 1.0, "Ly": 1.0, "alpha": 0.1, "Lz": 1.0}],
    ],
)
def test_poisson_M1perp_3d_compare_2p5d(Nel, p, mapping, show_plot=False):
    """
    Test the Poisson solver with M1perp diffusion matrix
    by comparing 3d simulation to a loop over 2d simulations.
    Dirichlet boundary conditions in eta1.
    """

    from time import time

    # create domain object
    dom_type = mapping[0]
    dom_params = mapping[1]

    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    # boundary conditions
    spl_kind = [False, True, True]
    dirichlet_bc = ((True, True), (False, False), (False, False))

    # evaluation grid
    e1 = np.linspace(0.0, 1.0, 50)
    e2 = np.linspace(0.0, 1.0, 60)
    e3 = np.linspace(0.0, 1.0, 30)

    # solution and right-hand side on unit cube
    def rho(e1, e2, e3):
        dd1 = np.sin(np.pi * e1) * np.sin(4 * np.pi * e2) * np.cos(2 * np.pi * e3) * (np.pi) ** 2
        dd2 = np.sin(np.pi * e1) * np.sin(4 * np.pi * e2) * np.cos(2 * np.pi * e3) * (4 * np.pi) ** 2
        return dd1 + dd2

    # create 3d derham object
    derham = Derham(Nel, p, spl_kind, dirichlet_bc=dirichlet_bc, comm=comm)

    mass_ops = WeightedMassOperators(derham, domain)

    Propagator.derham = derham
    Propagator.domain = domain
    Propagator.mass_ops = mass_ops

    # discrete right-hand sides
    l2_proj = L2Projector("H1", mass_ops)
    rho_vec = l2_proj.get_dofs(rho, apply_bc=True)

    print(f"{rho_vec[:].shape = }")

    # Create 3d Poisson solver
    solver_params = SolverParameters(
        tol=1.0e-13,
        maxiter=3000,
        info=True,
        verbose=False,
        recycle=False,
    )

    _phi = FEECVariable(space="H1")
    _phi.allocate(derham=derham, domain=domain)

    _phi_2p5d = FEECVariable(space="H1")
    _phi_2p5d.allocate(derham=derham, domain=domain)

    poisson_solver_3d = ImplicitDiffusion()
    poisson_solver_3d.variables.phi = _phi

    poisson_solver_3d.options = poisson_solver_3d.Options(
        sigma_1=1e-8,
        sigma_2=0.0,
        sigma_3=1.0,
        divide_by_dt=True,
        diffusion_mat="M1perp",
        rho=rho_vec,
        solver="pcg",
        precond="MassMatrixPreconditioner",
        solver_params=solver_params,
    )

    poisson_solver_3d.allocate()

    s = _phi.spline.starts
    e = _phi.spline.ends

    # create 2.5d deRham object
    Nel_new = [Nel[0], Nel[1], 1]
    p[2] = 1
    spl_kind[2] = True
    derham = Derham(Nel_new, p, spl_kind, dirichlet_bc=dirichlet_bc, comm=comm)

    mass_ops = WeightedMassOperators(derham, domain)

    Propagator.derham = derham
    Propagator.mass_ops = mass_ops

    _phi_small = FEECVariable(space="H1")
    _phi_small.allocate(derham=derham, domain=domain)
    rhs = derham.create_spline_function("rhs", "H1")

    poisson_solver_2p5d = ImplicitDiffusion()
    poisson_solver_2p5d.variables.phi = _phi_small

    poisson_solver_2p5d.options = poisson_solver_2p5d.Options(
        sigma_1=1e-8,
        sigma_2=0.0,
        sigma_3=1.0,
        divide_by_dt=True,
        diffusion_mat="M1perp",
        rho=rhs.vector,
        solver="pcg",
        precond="MassMatrixPreconditioner",
        solver_params=solver_params,
    )

    poisson_solver_2p5d.allocate()

    # Solve Poisson equation (call propagator with dt=1.)
    dt = 1.0
    t0 = time()
    poisson_solver_3d(dt)
    t1 = time()

    print(f"rank {rank}, 3d solve time = {t1 - t0}")

    t0 = time()
    t_inner = 0.0
    for n in range(s[2], e[2] + 1):
        # scale the rhs with Nel[2] !!
        rhs.vector[s[0] : e[0] + 1, s[1] : e[1] + 1, 0] = rho_vec[s[0] : e[0] + 1, s[1] : e[1] + 1, n] * Nel[2]
        t0i = time()
        poisson_solver_2p5d(dt)
        t1i = time()
        t_inner += t1i - t0i
        _tmp = _phi_small.spline.vector.copy()
        _phi_2p5d.spline.vector[s[0] : e[0] + 1, s[1] : e[1] + 1, n] = _tmp[s[0] : e[0] + 1, s[1] : e[1] + 1, 0]
    t1 = time()

    print(f"rank {rank}, 2.5d pure solve time (without copy) = {t_inner}")
    print(f"rank {rank}, 2.5d solve time = {t1 - t0}")

    # push numerical solutions
    sol_val = domain.push(_phi.spline, e1, e2, e3, kind="0")
    sol_val_2p5d = domain.push(_phi_2p5d.spline, e1, e2, e3, kind="0")
    x, y, z = domain(e1, e2, e3)

    print("max diff:", np.max(np.abs(sol_val - sol_val_2p5d)))
    assert np.max(np.abs(sol_val - sol_val_2p5d)) < 0.026

    if show_plot and rank == 0:
        plt.figure("e1-e2 plane", figsize=(24, 16))
        for n in range(3):
            plt.subplot(2, 3, n + 1)
            plt.title(f"e3 = {e3[n * 6]} from 3d solve")
            plt.pcolor(x[:, :, n * 6], y[:, :, n * 6], sol_val[:, :, n * 6], vmin=-1.0, vmax=1.0)
            plt.colorbar()
            ax = plt.gca()
            ax.set_aspect("equal", adjustable="box")
            plt.subplot(2, 3, 4 + n)
            plt.title(f"e3 = {e3[n * 6]} from 2.5d solve")
            plt.pcolor(x[:, :, n * 6], y[:, :, n * 6], sol_val_2p5d[:, :, n * 6], vmin=-1.0, vmax=1.0)
            plt.colorbar()
            ax = plt.gca()
            ax.set_aspect("equal", adjustable="box")
        plt.figure("e1-e3 plane", figsize=(24, 16))
        for n in range(3):
            plt.subplot(2, 3, n + 1)
            plt.title(f"e2 = {e2[n * 12]} from 3d solve")
            plt.pcolor(x[:, n * 12, :], z[:, n * 12, :], sol_val[:, n * 12, :], vmin=-1.0, vmax=1.0)
            plt.colorbar()
            ax = plt.gca()
            ax.set_aspect("equal", adjustable="box")
            plt.subplot(2, 3, 4 + n)
            plt.title(f"e2 = {e2[n * 12]} from 2.5d solve")
            plt.pcolor(x[:, n * 12, :], z[:, n * 12, :], sol_val_2p5d[:, n * 12, :], vmin=-1.0, vmax=1.0)
            plt.colorbar()
            ax = plt.gca()
            ax.set_aspect("equal", adjustable="box")

        plt.show()


if __name__ == "__main__":
    direction = 0
    bc_type = "dirichlet"
    mapping = ["Cuboid", {"l1": 0.0, "r1": 4.0, "l2": 0.0, "r2": 2.0, "l3": 0.0, "r3": 3.0}]
    mapping = ["Orthogonal", {"Lx": 4.0, "Ly": 2.0, "alpha": 0.1, "Lz": 3.0}]
    test_poisson_M1perp_1d(direction, bc_type, mapping, show_plot=True)

    # Nel = [64, 64, 1]
    # p = [2, 2, 1]
    # bc_type = 'neumann'
    # #mapping = ['Cuboid', {'l1': 0., 'r1': 4., 'l2': 0., 'r2': 2., 'l3': 0., 'r3': 3.}]
    # mapping = ['Orthogonal', {'Lx': 4., 'Ly': 2., 'alpha': .1, 'Lz': 1.}]
    # test_poisson_M1perp_2d(Nel, p, bc_type, mapping, show_plot=True)

    Nel = [64, 64, 16]
    p = [2, 2, 1]
    mapping = ["Cuboid", {"l1": 0.0, "r1": 1.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}]
    # test_poisson_M1perp_3d_compare_2p5d(Nel, p, mapping, show_plot=True)
