import cunumpy as xp
import matplotlib.pyplot as plt
import pytest
from psydac.ddm.mpi import mpi as MPI

from struphy.feec.mass import WeightedMassOperators
from struphy.feec.projectors import L2Projector
from struphy.feec.psydac_derham import Derham
from struphy.geometry import domains
from struphy.geometry.base import Domain
from struphy.initial import perturbations
from struphy.kinetic_background.maxwellians import Maxwellian3D
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.pic.accumulation.accum_kernels import charge_density_0form
from struphy.pic.accumulation.particles_to_grid import AccumulatorVector
from struphy.pic.particles import Particles6D
from struphy.pic.utilities import (
    BinningPlot,
    BoundaryParameters,
    LoadingParameters,
    WeightsParameters,
)
from struphy.propagators.base import Propagator
from struphy.propagators.propagators_fields import ImplicitDiffusion, Poisson
from struphy.utils.pyccel import Pyccelkernel

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
plt.rcParams.update({"font.size": 22})


@pytest.mark.parametrize("direction", [0, 1, 2])
@pytest.mark.parametrize("bc_type", ["periodic", "dirichlet", "neumann"])
@pytest.mark.parametrize(
    "mapping",
    [
        ["Cuboid", {"l1": 0.0, "r1": 4.0, "l2": 0.0, "r2": 2.0, "l3": 0.0, "r3": 3.0}],
        ["Orthogonal", {"Lx": 4.0, "Ly": 2.0, "alpha": 0.1, "Lz": 3.0}],
    ],
)
@pytest.mark.parametrize("projected_rhs", [False, True])
def test_poisson_1d(
    direction: int,
    bc_type: str,
    mapping: list[str, dict],
    projected_rhs: bool,
    show_plot: bool = False,
):
    """
    Test the convergence of Poisson solver in 1D by means of manufactured solutions.
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
            spl_kind = [True, True, True]
            dirichlet_bc = None

            # manufactured solution
            e1 = 0.0
            e2 = 0.0
            e3 = 0.0
            if direction == 0:
                Nel = [Neli, 1, 1]
                p = [pi, 1, 1]
                e1 = xp.linspace(0.0, 1.0, 50)

                if bc_type == "neumann":
                    spl_kind = [False, True, True]

                    def sol1_xyz(x, y, z):
                        return xp.cos(xp.pi / Lx * x)

                    def rho1_xyz(x, y, z):
                        return xp.cos(xp.pi / Lx * x) * (xp.pi / Lx) ** 2
                else:
                    if bc_type == "dirichlet":
                        spl_kind = [False, True, True]
                        dirichlet_bc = [(not kd,) * 2 for kd in spl_kind]
                        dirichlet_bc = tuple(dirichlet_bc)

                    def sol1_xyz(x, y, z):
                        return xp.sin(2 * xp.pi / Lx * x)

                    def rho1_xyz(x, y, z):
                        return xp.sin(2 * xp.pi / Lx * x) * (2 * xp.pi / Lx) ** 2

            elif direction == 1:
                Nel = [1, Neli, 1]
                p = [1, pi, 1]
                e2 = xp.linspace(0.0, 1.0, 50)

                if bc_type == "neumann":
                    spl_kind = [True, False, True]

                    def sol1_xyz(x, y, z):
                        return xp.cos(xp.pi / Ly * y)

                    def rho1_xyz(x, y, z):
                        return xp.cos(xp.pi / Ly * y) * (xp.pi / Ly) ** 2
                else:
                    if bc_type == "dirichlet":
                        spl_kind = [True, False, True]
                        dirichlet_bc = [(not kd,) * 2 for kd in spl_kind]
                        dirichlet_bc = tuple(dirichlet_bc)

                    def sol1_xyz(x, y, z):
                        return xp.sin(2 * xp.pi / Ly * y)

                    def rho1_xyz(x, y, z):
                        return xp.sin(2 * xp.pi / Ly * y) * (2 * xp.pi / Ly) ** 2

            elif direction == 2:
                Nel = [1, 1, Neli]
                p = [1, 1, pi]
                e3 = xp.linspace(0.0, 1.0, 50)

                if bc_type == "neumann":
                    spl_kind = [True, True, False]

                    def sol1_xyz(x, y, z):
                        return xp.cos(xp.pi / Lz * z)

                    def rho1_xyz(x, y, z):
                        return xp.cos(xp.pi / Lz * z) * (xp.pi / Lz) ** 2
                else:
                    if bc_type == "dirichlet":
                        spl_kind = [True, True, False]
                        dirichlet_bc = [(not kd,) * 2 for kd in spl_kind]
                        dirichlet_bc = tuple(dirichlet_bc)

                    def sol1_xyz(x, y, z):
                        return xp.sin(2 * xp.pi / Lz * z)

                    def rho1_xyz(x, y, z):
                        return xp.sin(2 * xp.pi / Lz * z) * (2 * xp.pi / Lz) ** 2
            else:
                print("Direction should be either 0, 1 or 2")

            # create derham object
            derham = Derham(Nel, p, spl_kind, dirichlet_bc=dirichlet_bc, comm=comm)

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
                rho.spline.vector = derham.P["0"](rho_pulled)
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

            poisson_solver = Poisson()
            poisson_solver.variables.phi = _phi

            poisson_solver.options = poisson_solver.Options(
                stab_eps=1e-12,
                # sigma_2=0.0,
                # sigma_3=1.0,
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
                elif direction == 2:
                    plt.plot(z[0, 0, :], sol_val1[0, 0, :], "ob", label="numerical")
                    plt.plot(z[0, 0, :], analytic_value1[0, 0, :], "r--", label="exact")
                    plt.xlabel("z")
                plt.title(f"{Nel =}")
                plt.legend()

            error = xp.max(xp.abs(analytic_value1 - sol_val1))
            print(f"{direction =}, {pi =}, {Neli =}, {error=}")

            errors.append(error)
            h = 1 / (Neli)
            h_vec.append(h)

        m, _ = xp.polyfit(xp.log(Nels), xp.log(errors), deg=1)
        print(f"For {pi =}, solution converges in {direction=} with rate {-m =} ")
        assert -m > (pi + 1 - 0.07)

        # Plot convergence in 1D
        if show_plot:
            plt.figure(
                f"Convergence for degree {pi =}, {direction + 1 =}, {bc_type =}, {mapping[0] =}",
                figsize=(12, 8),
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


@pytest.mark.parametrize(
    "mapping",
    [
        ["Cuboid", {"l1": 0.0, "r1": 4.0, "l2": 0.0, "r2": 2.0, "l3": 0.0, "r3": 3.0}],
        # ["Orthogonal", {"Lx": 4.0, "Ly": 2.0, "alpha": 0.1, "Lz": 3.0}],
    ],
)
def test_poisson_accum_1d(mapping, do_plot=False):
    """Pass accumulators as rhs."""
    # create domain object
    dom_type = mapping[0]
    dom_params = mapping[1]

    domain_class = getattr(domains, dom_type)
    domain: Domain = domain_class(**dom_params)

    if dom_type == "Cuboid":
        Lx = dom_params["r1"] - dom_params["l1"]
    else:
        Lx = dom_params["Lx"]

    # create derham object
    Nel = (16, 1, 1)
    p = (2, 1, 1)
    spl_kind = (True, True, True)
    derham = Derham(Nel, p, spl_kind, comm=comm)

    # mass matrices
    mass_ops = WeightedMassOperators(derham, domain)

    Propagator.derham = derham
    Propagator.domain = domain
    Propagator.mass_ops = mass_ops

    # 6D particle object
    domain_array = derham.domain_array
    nprocs = derham.domain_decomposition.nprocs
    domain_decomp = (domain_array, nprocs)

    lp = LoadingParameters(ppc=4000, seed=765)
    wp = WeightsParameters(control_variate=True)
    bp = BoundaryParameters()

    backgr = Maxwellian3D(n=(1.0, None))
    l = 1
    amp = 1e-1
    pert = perturbations.ModesCos(ls=(l,), amps=(amp,))
    maxw = Maxwellian3D(n=(1.0, pert))

    pert_exact = lambda x, y, z: amp * xp.cos(l * 2 * xp.pi / Lx * x)
    phi_exact = lambda x, y, z: amp / (l * 2 * xp.pi / Lx) ** 2 * xp.cos(l * 2 * xp.pi / Lx * x)
    e_exact = lambda x, y, z: amp / (l * 2 * xp.pi / Lx) * xp.sin(l * 2 * xp.pi / Lx * x)

    particles = Particles6D(
        comm_world=comm,
        domain_decomp=domain_decomp,
        loading_params=lp,
        weights_params=wp,
        boundary_params=bp,
        domain=domain,
        background=backgr,
        initial_condition=maxw,
    )
    particles.draw_markers()
    particles.initialize_weights()

    # particle to grid coupling
    kernel = Pyccelkernel(charge_density_0form)
    accum = AccumulatorVector(particles, "H1", kernel, mass_ops, domain.args_domain)
    # accum()
    # if do_plot:
    #     accum.show_accumulated_spline_field(mass_ops)

    rho = accum

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

    poisson_solver = Poisson()
    poisson_solver.variables.phi = _phi

    poisson_solver.options = poisson_solver.Options(
        stab_eps=1e-6,
        # sigma_2=0.0,
        # sigma_3=1.0,
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
    e1 = xp.linspace(0.0, 1.0, 50)
    e2 = 0.0
    e3 = 0.0

    num_values = domain.push(_phi.spline, e1, e2, e3, kind="0")
    x, y, z = domain(e1, e2, e3)
    pert_values = pert_exact(x, y, z)
    analytic_values = phi_exact(x, y, z)
    e_values = e_exact(x, y, z)

    _e = FEECVariable(space="Hcurl")
    _e.allocate(derham=derham, domain=domain)
    derham.grad.dot(-_phi.spline.vector, out=_e.spline.vector)
    num_values_e = domain.push(_e.spline, e1, e2, e3, kind="1")

    if do_plot:
        field = derham.create_spline_function("accum_field", "H1")
        field.vector = accum.vectors[0]
        accum_values = field(e1, e2, e3)

        plt.figure(figsize=(18, 12))
        plt.subplot(1, 3, 1)
        plt.plot(x[:, 0, 0], num_values[:, 0, 0], "ob", label="numerical")
        plt.plot(x[:, 0, 0], analytic_values[:, 0, 0], "r--", label="exact")
        plt.xlabel("x")
        plt.title("phi")
        plt.legend()
        plt.subplot(1, 3, 2)
        plt.plot(x[:, 0, 0], accum_values[:, 0, 0], "ob", label="numerical, without L2-proj")
        plt.plot(x[:, 0, 0], pert_values[:, 0, 0], "r--", label="exact")
        plt.xlabel("x")
        plt.title("rhs")
        plt.legend()
        plt.subplot(1, 3, 3)
        plt.plot(x[:, 0, 0], num_values_e[0][:, 0, 0], "ob", label="numerical")
        plt.plot(x[:, 0, 0], e_values[:, 0, 0], "r--", label="exact")
        plt.xlabel("x")
        plt.title("e_field")
        plt.legend()

        plt.show()

    error = xp.max(xp.abs(num_values_e[0][:, 0, 0] - e_values[:, 0, 0])) / xp.max(xp.abs(e_values[:, 0, 0]))
    print(f"{error=}")

    assert error < 0.0086


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("Nel", [[64, 64, 1]])
@pytest.mark.parametrize("p", [[1, 1, 1], [2, 2, 1]])
@pytest.mark.parametrize("bc_type", ["periodic", "dirichlet", "neumann"])
@pytest.mark.parametrize(
    "mapping",
    [
        ["Cuboid", {"l1": 0.0, "r1": 4.0, "l2": 0.0, "r2": 2.0, "l3": 0.0, "r3": 1.0}],
        ["Colella", {"Lx": 4.0, "Ly": 2.0, "alpha": 0.1, "Lz": 1.0}],
    ],
)
@pytest.mark.parametrize("projected_rhs", [False, True])
def test_poisson_2d(Nel, p, bc_type, mapping, projected_rhs, show_plot=False):
    """
    Test the Poisson solver by means of manufactured solutions in 2D .
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
    dirichlet_bc = None

    if bc_type == "periodic":
        spl_kind = [True] * 3

        # manufactured solution in 2D
        def sol2_xyz(x, y, z):
            return xp.sin(2 * xp.pi * x / Lx + 4 * xp.pi / Ly * y)

        def rho2_xyz(x, y, z):
            ddx = xp.sin(2 * xp.pi / Lx * x + 4 * xp.pi / Ly * y) * (2 * xp.pi / Lx) ** 2
            ddy = xp.sin(2 * xp.pi / Lx * x + 4 * xp.pi / Ly * y) * (4 * xp.pi / Ly) ** 2
            return ddx + ddy

    elif bc_type == "dirichlet":
        spl_kind = [False, True, True]
        dirichlet_bc = [(not kd,) * 2 for kd in spl_kind]
        dirichlet_bc = tuple(dirichlet_bc)
        print(f"{dirichlet_bc =}")

        # manufactured solution in 2D
        def sol2_xyz(x, y, z):
            return xp.sin(xp.pi * x / Lx) * xp.sin(4 * xp.pi / Ly * y)

        def rho2_xyz(x, y, z):
            ddx = xp.sin(xp.pi * x / Lx) * xp.sin(4 * xp.pi / Ly * y) * (xp.pi / Lx) ** 2
            ddy = xp.sin(xp.pi * x / Lx) * xp.sin(4 * xp.pi / Ly * y) * (4 * xp.pi / Ly) ** 2
            return ddx + ddy

    elif bc_type == "neumann":
        spl_kind = [False, True, True]

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
    derham = Derham(Nel, p, spl_kind, dirichlet_bc=dirichlet_bc, comm=comm)

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
        rho1.spline.vector = derham.P["0"](rho1_pulled)

        rho2 = FEECVariable(space="H1")
        rho2.allocate(derham=derham, domain=domain)
        rho2.spline.vector = derham.P["0"](rho2_pulled)
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

    poisson_solver1 = Poisson()
    poisson_solver1.variables.phi = _phi1

    poisson_solver1.options = poisson_solver1.Options(
        stab_eps=1e-8,
        # sigma_2=0.0,
        # sigma_3=1.0,
        rho=rho1,
        solver="pcg",
        precond="MassMatrixPreconditioner",
        solver_params=solver_params,
    )

    poisson_solver1.allocate()

    # _phi1 = derham.create_spline_function("test1", "H1")
    # poisson_solver1 = Poisson(
    #     _phi1.vector, sigma_1=1e-8, sigma_2=0.0, sigma_3=1.0, rho=rho_vec1, solver=solver_params
    # )

    _phi2 = FEECVariable(space="H1")
    _phi2.allocate(derham=derham, domain=domain)

    poisson_solver2 = Poisson()
    poisson_solver2.variables.phi = _phi2

    stab_eps = 1e-8
    err_lim = 0.03
    if bc_type == "neumann" and dom_type == "Colella":
        stab_eps = 1e-4
        err_lim = 0.046

    poisson_solver2.options = poisson_solver2.Options(
        stab_eps=stab_eps,
        # sigma_2=0.0,
        # sigma_3=1.0,
        rho=rho2,
        solver="pcg",
        precond="MassMatrixPreconditioner",
        solver_params=solver_params,
    )

    poisson_solver2.allocate()

    # _phi2 = derham.create_spline_function("test2", "H1")
    # poisson_solver2 = Poisson(
    #     _phi2.vector, sigma_1=1e-8, sigma_2=0.0, sigma_3=1.0, rho=rho_vec2, solver=solver_params
    # )

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

    print(f"{p =}, {bc_type =}, {mapping =}")
    print(f"{error1 =}")
    print(f"{error2 =}")
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

    if p[0] == 1 and bc_type == "neumann" and mapping[0] == "Colella":
        pass
    else:
        assert error1 < 0.0053
        assert error2 < err_lim


if __name__ == "__main__":
    # direction = 0
    # bc_type = "dirichlet"
    mapping = ["Cuboid", {"l1": 0.0, "r1": 4.0, "l2": 0.0, "r2": 2.0, "l3": 0.0, "r3": 3.0}]
    # mapping = ['Orthogonal', {'Lx': 4., 'Ly': 2., 'alpha': .1, 'Lz': 3.}]
    # test_poisson_1d(direction, bc_type, mapping, projected_rhs=True, show_plot=True)

    # Nel = [64, 64, 1]
    # p = [2, 2, 1]
    # bc_type = 'neumann'
    # # mapping = ['Cuboid', {'l1': 0., 'r1': 4., 'l2': 0., 'r2': 2., 'l3': 0., 'r3': 3.}]
    # # mapping = ['Orthogonal', {'Lx': 4., 'Ly': 2., 'alpha': .1, 'Lz': 1.}]
    # mapping = ['Colella', {'Lx': 4., 'Ly': 2., 'alpha': .1, 'Lz': 1.}]
    # test_poisson_2d(Nel, p, bc_type, mapping, projected_rhs=True, show_plot=True)

    test_poisson_accum_1d(mapping, do_plot=True)
