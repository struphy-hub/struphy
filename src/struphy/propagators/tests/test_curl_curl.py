import logging

import cunumpy as xp
import matplotlib.pyplot as plt
import pytest
from feectools.ddm.mpi import mpi as MPI

from struphy import (
    BinningPlot,
    BoundaryParameters,
    LoadingParameters,
    WeightsParameters,
    domains,
    perturbations,
)
from struphy.feec.mass import L2Projector, WeightedMassOperators
from struphy.feec.psydac_derham import Derham
from struphy.geometry.base import Domain
from struphy.io.options import DerhamOptions
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.propagators.curl_curl_solve import CurlCurlSolve
from struphy.topology.grids import TensorProductGrid
from struphy.utils.pyccel import Pyccelkernel

logger = logging.getLogger("struphy")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
plt.rcParams.update({"font.size": 22})

# Curl-Curl test: polynomial test field on the logic cube

domain: Domain = domains.Cuboid()

@pytest.mark.parametrize("bc_type", ["dirichlet","periodic","neumann"])
@pytest.mark.parametrize("direction",["1","2","3"])
@pytest.mark.parametrize("pmax",[3,4])
@pytest.mark.parametrize("Nmax",[6,7,8])


def test_convergence_1d(
    bc_type: str,
    direction: str,
    pmax: int,
    Nmax: int,
    Nmin: int = 3,
    sigma: float = 1.5,
    show_plot: bool = False,
):
    """Test of the solver on 1d problems by means of manufactured solution"""

    if direction == "1":

        def E_exact_x(x, y, z) -> float:
            return xp.sin(5 * xp.pi * y)

        def E_exact_y(x, y, z) -> float:
            return 0 * y

        def E_exact_z(x, y, z) -> float:
            return 0 * z

        def j_exact_x(x, y, z) -> float:
            return (25 * (xp.pi**2) - sigma) * E_exact_x(x, y, z)

        def j_exact_y(x, y, z) -> float:
            return 0 * y

        def j_exact_z(x, y, z) -> float:
            return 0 * z

    elif direction == "2":

        def E_exact_x(x, y, z) -> float:
            return 0 * x

        def E_exact_y(x, y, z) -> float:
            return xp.sin(5 * xp.pi * z)

        def E_exact_z(x, y, z) -> float:
            return 0 * z

        def j_exact_x(x, y, z) -> float:
            return 0 * x

        def j_exact_y(x, y, z) -> float:
            return (25 * (xp.pi**2) - sigma) * E_exact_y(x, y, z)

        def j_exact_z(x, y, z) -> float:
            return 0 * z

    elif direction == "3":

        def E_exact_x(x, y, z) -> float:
            return 0 * x

        def E_exact_y(x, y, z) -> float:
            return 0 * y

        def E_exact_z(x, y, z) -> float:
            return xp.sin(5 * xp.pi * x)

        def j_exact_x(x, y, z) -> float:
            return 0 * x

        def j_exact_y(x, y, z) -> float:
            return 0 * y

        def j_exact_z(x, y, z) -> float:
            return (25 * (xp.pi**2) - sigma) * E_exact_z(x, y, z)

    if bc_type == "periodic":

        if Nmin < 4:
            Nmin = 4
        
        if direction == "1":

            def E_exact_x(x, y, z) -> float:
                return xp.sin(8 * xp.pi * y)

            def E_exact_y(x, y, z) -> float:
                return 0 * y

            def E_exact_z(x, y, z) -> float:
                return 0 * z

            def j_exact_x(x, y, z) -> float:
                return (64 * (xp.pi**2) - sigma) * E_exact_x(x, y, z)

            def j_exact_y(x, y, z) -> float:
                return 0 * y

            def j_exact_z(x, y, z) -> float:
                return 0 * z

        elif direction == "2":

            def E_exact_x(x, y, z) -> float:
                return 0 * x

            def E_exact_y(x, y, z) -> float:
                return xp.sin(8 * xp.pi * z)

            def E_exact_z(x, y, z) -> float:
                return 0 * z

            def j_exact_x(x, y, z) -> float:
                return 0 * x

            def j_exact_y(x, y, z) -> float:
                return (64 * (xp.pi**2) - sigma) * E_exact_y(x, y, z)

            def j_exact_z(x, y, z) -> float:
                return 0 * z

        elif direction == "3":

            def E_exact_x(x, y, z) -> float:
                return 0 * x

            def E_exact_y(x, y, z) -> float:
                return 0 * y

            def E_exact_z(x, y, z) -> float:
                return xp.sin(8 * xp.pi * x)

            def j_exact_x(x, y, z) -> float:
                return 0 * x

            def j_exact_y(x, y, z) -> float:
                return 0 * y

            def j_exact_z(x, y, z) -> float:
                return (64 * (xp.pi**2) - sigma) * E_exact_z(x, y, z)

    elif bc_type == "neumann":
        if direction == "1":

            def E_exact_x(x, y, z) -> float:
                return xp.cos(5 * xp.pi * y)

            def E_exact_y(x, y, z) -> float:
                return 0 * y

            def E_exact_z(x, y, z) -> float:
                return 0 * z

            def j_exact_x(x, y, z) -> float:
                return (25 * (xp.pi**2) - sigma) * E_exact_x(x, y, z)

            def j_exact_y(x, y, z) -> float:
                return 0 * y

            def j_exact_z(x, y, z) -> float:
                return 0 * z

        elif direction == "2":

            def E_exact_x(x, y, z) -> float:
                return 0 * x

            def E_exact_y(x, y, z) -> float:
                return xp.cos(5 * xp.pi * z)

            def E_exact_z(x, y, z) -> float:
                return 0 * z

            def j_exact_x(x, y, z) -> float:
                return 0 * x

            def j_exact_y(x, y, z) -> float:
                return (25 * (xp.pi**2) - sigma) * E_exact_y(x, y, z)

            def j_exact_z(x, y, z) -> float:
                return 0 * z

        elif direction == "3":

            def E_exact_x(x, y, z) -> float:
                return 0 * x

            def E_exact_y(x, y, z) -> float:
                return 0 * y

            def E_exact_z(x, y, z) -> float:
                return xp.cos(5 * xp.pi * x)

            def j_exact_x(x, y, z) -> float:
                return 0 * x

            def j_exact_y(x, y, z) -> float:
                return 0 * y

            def j_exact_z(x, y, z) -> float:
                return (25 * (xp.pi**2) - sigma) * E_exact_z(x, y, z)

    assert Nmin < Nmax
    
    # Test over spline degree and grid resolution

    Nels = [2**n for n in range(Nmin, Nmax + 1)]

    e1 = 0.0
    e2 = 0.0
    e3 = 0.0

    for p in range(2, pmax + 1):
        errors = []
        h_vec = []

        for n, Nel in enumerate(Nels):
            if direction == "1":
                degree = [1, p, 1]
                num_elements = [1, Nel, 1]
                bcs = (None, ("dirichlet", "dirichlet"), None)
                e2 = xp.linspace(0.0, 1.0, 64)

            elif direction == "2":
                degree = [1, 1, p]
                num_elements = [1, 1, Nel]
                bcs = (None, None, ("dirichlet", "dirichlet"))
                e3 = xp.linspace(0.0, 1.0, 64)

            elif direction == "3":
                degree = [p, 1, 1]
                num_elements = [Nel, 1, 1]
                bcs = (("dirichlet", "dirichlet"), None, None)
                e1 = xp.linspace(0.0, 1.0, 64)

            if bc_type == "periodic":
                if direction == "1":
                    degree = [1, p, 1]
                    num_elements = [1, Nel, 1]
                    bcs = (None, None, None)
                    e2 = xp.linspace(0.0, 1.0, 64)

                elif direction == "2":
                    degree = [1, 1, p]
                    num_elements = [1, 1, Nel]
                    bcs = (None, None, None)
                    e3 = xp.linspace(0.0, 1.0, 64)

                elif direction == "3":
                    degree = [p, 1, 1]
                    num_elements = [Nel, 1, 1]
                    bcs = (None, None, None)
                    e1 = xp.linspace(0.0, 1.0, 64)

            elif bc_type == "neumann":
                if direction == "1":
                    degree = [1, p, 1]
                    num_elements = [1, Nel, 1]
                    bcs = (None, ("free", "free"), None)
                    e2 = xp.linspace(0.0, 1.0, 64)

                elif direction == "2":
                    degree = [1, 1, p]
                    num_elements = [1, 1, Nel]
                    bcs = (None, None, ("free", "free"))
                    e3 = xp.linspace(0.0, 1.0, 64)

                elif direction == "3":
                    degree = [p, 1, 1]
                    num_elements = [Nel, 1, 1]
                    bcs = (("free", "free"), None, None)
                    e1 = xp.linspace(0.0, 1.0, 64)

            grid = TensorProductGrid(num_elements=num_elements)
            derham_opts = DerhamOptions(degree=degree, bcs=bcs)
            derham = Derham(grid=grid, options=derham_opts, comm=comm)

            mass_ops = WeightedMassOperators(derham=derham, domain=domain)

            Propagator.derham = derham
            Propagator.domain = domain
            Propagator.mass_ops = mass_ops

            def j_pulled_x(e1, e2, e3):
                return domain.pull([j_exact_x, j_exact_y, j_exact_z], e1, e2, e3, kind="1", squeeze_out=False)[0]

            def j_pulled_y(e1, e2, e3):
                return domain.pull([j_exact_x, j_exact_y, j_exact_z], e1, e2, e3, kind="1", squeeze_out=False)[1]

            def j_pulled_z(e1, e2, e3):
                return domain.pull([j_exact_x, j_exact_y, j_exact_z], e1, e2, e3, kind="1", squeeze_out=False)[2]

            j = FEECVariable(space="Hcurl")
            j.allocate(derham=derham, domain=domain)
            j.spline.vector = derham.P1([j_pulled_x, j_pulled_y, j_pulled_z])

            solver_params = SolverParameters(
                tol=1.0e-10,
                maxiter=3000,
                info=True,
                recycle=False,
            )

            _e = FEECVariable(space="Hcurl")
            _e.allocate(derham=derham, domain=domain)

            curlcurl_solver = CurlCurlSolve()
            curlcurl_solver.variables.e = _e

            curlcurl_solver.options = curlcurl_solver.Options(
                sigma=sigma,
                j=j,
                solver="pcg",
                precond="MassMatrixPreconditioner",
                solver_params=solver_params,
            )

            curlcurl_solver.allocate()

            dt = 1.0
            curlcurl_solver(dt)

            E_calculated = domain.push(_e.spline, e1, e2, e3, kind="1")
            x, y, z = domain(e1, e2, e3)
            E_analytical = xp.array([E_exact_x(x, y, z), E_exact_y(x, y, z), E_exact_z(x, y, z)])

            plt.figure(f"degree {p =}, {bc_type =}, {direction =}")
            plt.subplot(2, int((Nmax - Nmin) / 2 + 1), n + 1)

            if direction == "1":
                plt.plot(y[0, :, 0], E_calculated[0][0, :, 0], "o", label=f"{Nel}, numerical")
                plt.plot(y[0, :, 0], E_analytical[0][0, :, 0], "k--", label=f"{Nel}, analytical")

            elif direction == "2":
                plt.plot(z[0, 0, :], E_calculated[1][0, 0, :], "o", label=f"{Nel}, numerical")
                plt.plot(z[0, 0, :], E_analytical[1][0, 0, :], "k--", label=f"{Nel}, analytical")

            elif direction == "3":
                plt.plot(x[:, 0, 0], E_calculated[2][:, 0, 0], "o", label=f"{Nel}, numerical")
                plt.plot(x[:, 0, 0], E_analytical[2][:, 0, 0], "k--", label=f"{Nel}, analytical")

            plt.legend()

            error = xp.max(xp.abs(E_calculated - E_analytical))
            errors.append(error)

            h = 1 / Nel
            h_vec.append(h)

        m, _ = xp.polyfit(xp.log(Nels), xp.log(errors), deg=1)
        logger.info(f"For {p =}, solution converges with rate {-m =} ")
        
        tolerance: float = 0.07
        
        assert -m > (p + 1 - tolerance)

        if show_plot:

            plt.figure(f"Convergence for degree {p =}", figsize=(12, 8))
            plt.title(f"Convergence rate for degree {p =}")
            plt.plot(h_vec, errors, "o", label=f"Calculated error, {m =}")
            plt.plot(
                h_vec,
                [(h ** (p + 1)) / (h_vec[0] ** (p + 1)) * errors[0] for h in h_vec],
                "k--",
                label="Theoretical error, rate = p + 1",
            )
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Grid spacing h")
            plt.ylabel("Error")
            plt.legend()

    if show_plot:

        plt.show()



@pytest.mark.parametrize("bc_type", ["dirichlet","periodic","neumann"])
@pytest.mark.parametrize("direction",["1","2","3"])
@pytest.mark.parametrize("pmax",[3,4])
@pytest.mark.parametrize("Nmax",[6,7,8])

def test_convergence_2d(
    bc_type: str,
    direction: str,
    pmax: int,
    Nmax: int,
    Nmin: int = 3,
    sigma: float = 1.5,
    show_plot: bool = False,
):
    """Test of the solver on 2d problems by means of manufactured solution"""

    if direction == "1":

        def E_exact_x(x, y, z) -> float:
            return xp.sin(3 * xp.pi * y) * xp.sin(5 * xp.pi * z)

        def E_exact_y(x, y, z) -> float:
            return 0 * y

        def E_exact_z(x, y, z) -> float:
            return 0 * z

        def j_exact_x(x, y, z) -> float:
            return (34 * (xp.pi**2) - sigma) * E_exact_x(x, y, z)

        def j_exact_y(x, y, z) -> float:
            return 0 * y

        def j_exact_z(x, y, z) -> float:
            return 0 * z

    elif direction == "2":

        def E_exact_x(x, y, z) -> float:
            return 0 * x

        def E_exact_y(x, y, z) -> float:
            return xp.sin(3 * xp.pi * z) * xp.sin(5 * xp.pi * x)

        def E_exact_z(x, y, z) -> float:
            return 0 * z

        def j_exact_x(x, y, z) -> float:
            return 0 * x

        def j_exact_y(x, y, z) -> float:
            return (34 * (xp.pi**2) - sigma) * E_exact_y(x, y, z)

        def j_exact_z(x, y, z) -> float:
            return 0 * z

    elif direction == "3":

        def E_exact_x(x, y, z) -> float:
            return 0 * x

        def E_exact_y(x, y, z) -> float:
            return 0 * y

        def E_exact_z(x, y, z) -> float:
            return xp.sin(3 * xp.pi * x) * xp.sin(5 * xp.pi * y)

        def j_exact_x(x, y, z) -> float:
            return 0 * x

        def j_exact_y(x, y, z) -> float:
            return 0 * y

        def j_exact_z(x, y, z) -> float:
            return (34 * (xp.pi**2) - sigma) * E_exact_z(x, y, z)

    if bc_type == "periodic":

        if Nmin < 4:
            Nmin = 4
        
        if direction == "1":

            def E_exact_x(x, y, z) -> float:
                return xp.sin(4 * xp.pi * y) * xp.sin(6 * xp.pi * z)

            def E_exact_y(x, y, z) -> float:
                return 0 * y

            def E_exact_z(x, y, z) -> float:
                return 0 * z

            def j_exact_x(x, y, z) -> float:
                return (52 * (xp.pi**2) - sigma) * E_exact_x(x, y, z)

            def j_exact_y(x, y, z) -> float:
                return 0 * y

            def j_exact_z(x, y, z) -> float:
                return 0 * z

        elif direction == "2":

            def E_exact_x(x, y, z) -> float:
                return 0 * x

            def E_exact_y(x, y, z) -> float:
                return xp.sin(4 * xp.pi * z) * xp.sin(6 * xp.pi * x)

            def E_exact_z(x, y, z) -> float:
                return 0 * z

            def j_exact_x(x, y, z) -> float:
                return 0 * x

            def j_exact_y(x, y, z) -> float:
                return (52 * (xp.pi**2) - sigma) * E_exact_y(x, y, z)

            def j_exact_z(x, y, z) -> float:
                return 0 * z

        elif direction == "3":

            def E_exact_x(x, y, z) -> float:
                return 0 * x

            def E_exact_y(x, y, z) -> float:
                return 0 * y

            def E_exact_z(x, y, z) -> float:
                return xp.sin(4 * xp.pi * x) * xp.sin(6 * xp.pi * y)

            def j_exact_x(x, y, z) -> float:
                return 0 * x

            def j_exact_y(x, y, z) -> float:
                return 0 * y

            def j_exact_z(x, y, z) -> float:
                return (52 * (xp.pi**2) - sigma) * E_exact_z(x, y, z)

    elif bc_type == "neumann":
        if direction == "1":

            def E_exact_x(x, y, z) -> float:
                return xp.cos(3 * xp.pi * y) * xp.cos(5 * xp.pi * z)

            def E_exact_y(x, y, z) -> float:
                return 0 * y

            def E_exact_z(x, y, z) -> float:
                return 0 * z

            def j_exact_x(x, y, z) -> float:
                return (34 * (xp.pi**2) - sigma) * E_exact_x(x, y, z)

            def j_exact_y(x, y, z) -> float:
                return 0 * y

            def j_exact_z(x, y, z) -> float:
                return 0 * z

        elif direction == "2":

            def E_exact_x(x, y, z) -> float:
                return 0 * x

            def E_exact_y(x, y, z) -> float:
                return xp.cos(3 * xp.pi * z) * xp.cos(5 * xp.pi * x)

            def E_exact_z(x, y, z) -> float:
                return 0 * z

            def j_exact_x(x, y, z) -> float:
                return 0 * x

            def j_exact_y(x, y, z) -> float:
                return (34 * (xp.pi**2) - sigma) * E_exact_y(x, y, z)

            def j_exact_z(x, y, z) -> float:
                return 0 * z

        elif direction == "3":

            def E_exact_x(x, y, z) -> float:
                return 0 * x

            def E_exact_y(x, y, z) -> float:
                return 0 * y

            def E_exact_z(x, y, z) -> float:
                return xp.cos(3 * xp.pi * x) * xp.cos(5 * xp.pi * y)

            def j_exact_x(x, y, z) -> float:
                return 0 * x

            def j_exact_y(x, y, z) -> float:
                return 0 * y

            def j_exact_z(x, y, z) -> float:
                return (34 * (xp.pi**2) - sigma) * E_exact_z(x, y, z)

    # Test over spline degree and grid resolution

    Nels = [2**n for n in range(Nmin, Nmax + 1)]

    e1 = 0.0
    e2 = 0.0
    e3 = 0.0

    for p in range(2, pmax + 1):
        errors = []
        h_vec = []

        for n, Nel in enumerate(Nels):
            if direction == "1":
                degree = [1, p, p]
                num_elements = [1, Nel, Nel]
                bcs = (None, ("dirichlet", "dirichlet"), ("dirichlet", "dirichlet"))
                e2 = xp.linspace(0.0, 1.0, 64)
                e3 = xp.linspace(0.0, 1.0, 64)

            elif direction == "2":
                degree = [p, 1, p]
                num_elements = [Nel, 1, Nel]
                bcs = (("dirichlet", "dirichlet"), None, ("dirichlet", "dirichlet"))
                e3 = xp.linspace(0.0, 1.0, 64)
                e1 = xp.linspace(0.0, 1.0, 64)

            elif direction == "3":
                degree = [p, p, 1]
                num_elements = [Nel, Nel, 1]
                bcs = (("dirichlet", "dirichlet"), ("dirichlet", "dirichlet"), None)
                e1 = xp.linspace(0.0, 1.0, 64)
                e2 = xp.linspace(0.0, 1.0, 64)

            if bc_type == "periodic":
                if direction == "1":
                    degree = [1, p, p]
                    num_elements = [1, Nel, Nel]
                    bcs = (None, None, None)
                    e2 = xp.linspace(0.0, 1.0, 64)
                    e3 = xp.linspace(0.0, 1.0, 64)

                elif direction == "2":
                    degree = [p, 1, p]
                    num_elements = [Nel, 1, Nel]
                    bcs = (None, None, None)
                    e3 = xp.linspace(0.0, 1.0, 64)
                    e1 = xp.linspace(0.0, 1.0, 64)

                elif direction == "3":
                    degree = [p, p, 1]
                    num_elements = [Nel, Nel, 1]
                    bcs = (None, None, None)
                    e1 = xp.linspace(0.0, 1.0, 64)
                    e2 = xp.linspace(0.0, 1.0, 64)

            elif bc_type == "neumann":
                if direction == "1":
                    degree = [1, p, p]
                    num_elements = [1, Nel, Nel]
                    bcs = (None, ("free", "free"), ("free", "free"))
                    e2 = xp.linspace(0.0, 1.0, 64)
                    e3 = xp.linspace(0.0, 1.0, 64)

                elif direction == "2":
                    degree = [p, 1, p]
                    num_elements = [Nel, 1, Nel]
                    bcs = (("free", "free"), None, ("free", "free"))
                    e3 = xp.linspace(0.0, 1.0, 64)
                    e1 = xp.linspace(0.0, 1.0, 64)

                elif direction == "3":
                    degree = [p, p, 1]
                    num_elements = [Nel, Nel, 1]
                    bcs = (("free", "free"), ("free", "free"), None)
                    e1 = xp.linspace(0.0, 1.0, 64)
                    e2 = xp.linspace(0.0, 1.0, 64)

            grid = TensorProductGrid(num_elements=num_elements)
            derham_opts = DerhamOptions(degree=degree, bcs=bcs)
            derham = Derham(grid=grid, options=derham_opts, comm=comm)

            mass_ops = WeightedMassOperators(derham=derham, domain=domain)

            Propagator.derham = derham
            Propagator.domain = domain
            Propagator.mass_ops = mass_ops

            def j_pulled_x(e1, e2, e3):
                return domain.pull([j_exact_x, j_exact_y, j_exact_z], e1, e2, e3, kind="1", squeeze_out=False)[0]

            def j_pulled_y(e1, e2, e3):
                return domain.pull([j_exact_x, j_exact_y, j_exact_z], e1, e2, e3, kind="1", squeeze_out=False)[1]

            def j_pulled_z(e1, e2, e3):
                return domain.pull([j_exact_x, j_exact_y, j_exact_z], e1, e2, e3, kind="1", squeeze_out=False)[2]

            j = FEECVariable(space="Hcurl")
            j.allocate(derham=derham, domain=domain)
            j.spline.vector = derham.P1([j_pulled_x, j_pulled_y, j_pulled_z])

            solver_params = SolverParameters(
                tol=1.0e-10,
                maxiter=3000,
                info=True,
                recycle=False,
            )

            _e = FEECVariable(space="Hcurl")
            _e.allocate(derham=derham, domain=domain)

            curlcurl_solver = CurlCurlSolve()
            curlcurl_solver.variables.e = _e

            curlcurl_solver.options = curlcurl_solver.Options(
                sigma=sigma,
                j=j,
                solver="pcg",
                precond="MassMatrixPreconditioner",
                solver_params=solver_params,
            )

            curlcurl_solver.allocate()

            dt = 1.0
            curlcurl_solver(dt)

            E_calculated = domain.push(_e.spline, e1, e2, e3, kind="1")
            x, y, z = domain(e1, e2, e3)
            E_analytical = xp.array([E_exact_x(x, y, z), E_exact_y(x, y, z), E_exact_z(x, y, z)])
            E_difference = xp.abs(E_calculated - E_analytical)

            plt.figure(f"degree {p =}, {bc_type =}, {direction =}")
            plt.subplot(2, int((Nmax - Nmin) / 2 + 1), n + 1)

            if direction == "1":
                plt.pcolormesh(
                    y[0, :, :], z[0, :, :], E_difference[0][0, :, :], vmin=0.0, vmax=1.0, label=f"{Nel}x{Nel}"
                )
                plt.colorbar()

            elif direction == "2":
                plt.pcolormesh(
                    z[:, 0, :], x[:, 0, :], E_difference[1][:, 0, :], vmin=0.0, vmax=1.0, label=f"{Nel}x{Nel}"
                )
                plt.colorbar()

            elif direction == "3":
                plt.pcolormesh(
                    x[:, :, 0], y[:, :, 0], E_difference[2][:, :, 0], vmin=0.0, vmax=1.0, label=f"{Nel}x{Nel}"
                )
                plt.colorbar()

            plt.legend()

            error = xp.max(E_difference)
            errors.append(error)

            h = 1 / Nel
            h_vec.append(h)

        m, _ = xp.polyfit(xp.log(Nels), xp.log(errors), deg=1)
        logger.info(f"For {p =}, solution converges with rate {-m =} ")
        
        tolerance: float = 0.07
        
        assert -m > (p + 1 - tolerance)

        if show_plot:
        
            plt.figure(f"Convergence for degree {p =}", figsize=(12, 8))
            plt.title(f"Convergence rate for degree {p =}")
            plt.plot(h_vec, errors, "o", label=f"Calculated error, {m =}")
            plt.plot(
                h_vec,
                [(h ** (p + 1)) / (h_vec[0] ** (p + 1)) * errors[0] for h in h_vec],
                "k--",
                label="Theoretical error, rate = p + 1",
            )
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Grid spacing h")
            plt.ylabel("Error")
            plt.legend()

    if show_plot:
    
        plt.show()


if __name__ == "__main__":

    test_convergence_1d(
        bc_type="periodic",
        direction="1",
        pmax=3,
        sigma=5,
        Nmax=6,
        show_plot=True
    )

    # test_convergence_2d(
    #     bc_type="neumann",
    #     direction = "2",
    #     pmax=4,
    #     Nmax=6,
    #     sigma=5,
     # )
