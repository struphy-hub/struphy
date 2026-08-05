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
    set_logging_level,
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
from cunumpy import Pyccelkernel

logger = logging.getLogger("struphy")
set_logging_level(logging.INFO)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
plt.rcParams.update({"font.size": 22})

# Curl-Curl test: polynomial test field on the logic cube

domain: Domain = domains.Cuboid()


@pytest.mark.parametrize("bc_type", ["dirichlet", "periodic"])
@pytest.mark.parametrize("direction", ["1", "2", "3"])
def test_convergence_1d(
    bc_type: str,
    direction: str,
    show_plot: bool = False,
):
    """Test of the solver on 1d problems by means of manufactured solution"""
    pmax = 4
    Nmin = 4
    Nmax = 8

    sigma = 1.5

    E_exact = lambda e: xp.sin(8 * xp.pi * e)
    j_exact = lambda e: (64 * (xp.pi**2) - sigma) * E_exact(e)

    # Test over spline degree and grid resolution
    Nels = [2**n for n in range(Nmin, Nmax + 1)]

    e1 = 0.0
    e2 = 0.0
    e3 = 0.0
    e = xp.linspace(0.0, 1.0, 64)

    bcs = (None, None, None)
    if direction == "1":
        E_exact_x = lambda x, y, z: 0 * x
        E_exact_y = lambda x, y, z: 0 * y
        E_exact_z = lambda x, y, z: E_exact(x)
        j_exact_x = lambda x, y, z: 0 * x
        j_exact_y = lambda x, y, z: 0 * y
        j_exact_z = lambda x, y, z: j_exact(x)
        e1 = e

    elif direction == "2":
        E_exact_x = lambda x, y, z: E_exact(y)
        E_exact_y = lambda x, y, z: 0 * y
        E_exact_z = lambda x, y, z: 0 * z
        j_exact_x = lambda x, y, z: j_exact(y)
        j_exact_y = lambda x, y, z: 0 * y
        j_exact_z = lambda x, y, z: 0 * z
        e2 = e

    elif direction == "3":
        E_exact_x = lambda x, y, z: 0 * x
        E_exact_y = lambda x, y, z: E_exact(z)
        E_exact_z = lambda x, y, z: 0 * z
        j_exact_x = lambda x, y, z: 0 * x
        j_exact_y = lambda x, y, z: j_exact(z)
        j_exact_z = lambda x, y, z: 0 * z
        e3 = e

    ee1, ee2, ee3 = xp.meshgrid(e1, e2, e3, indexing="ij")

    for p in range(2, pmax + 1):
        errors = []
        h_vec = []

        if show_plot:
            plt.figure(f"degree {p =}, {bc_type =}, {direction =}", figsize=(12, 8))

        for n, Nel in enumerate(Nels):
            if direction == "1":
                degree = (p, 1, 1)
                num_elements = (Nel, 1, 1)
                if bc_type == "dirichlet":
                    bcs = (("dirichlet", "dirichlet"), None, None)
                # elif bc_type == "neumann":
                #     bcs = (("free", "free"), None, None)

            elif direction == "2":
                degree = (1, p, 1)
                num_elements = (1, Nel, 1)
                if bc_type == "dirichlet":
                    bcs = (None, ("dirichlet", "dirichlet"), None)
                # elif bc_type == "neumann":
                #     bcs = (None, ("free", "free"), None)

            elif direction == "3":
                degree = (1, 1, p)
                num_elements = (1, 1, Nel)
                if bc_type == "dirichlet":
                    bcs = (None, None, ("dirichlet", "dirichlet"))
                # elif bc_type == "neumann":
                #     bcs = (None, None, ("free", "free"))

            grid = TensorProductGrid(num_elements=num_elements)
            derham_opts = DerhamOptions(degree=degree, bcs=bcs)
            derham = Derham(grid=grid, options=derham_opts, comm=comm)

            mass_ops = WeightedMassOperators(derham=derham, domain=domain)

            Propagator.derham = derham
            Propagator.domain = domain
            Propagator.mass_ops = mass_ops

            j = FEECVariable(space="Hcurl")
            j.allocate(derham=derham, domain=domain)
            j.spline.vector = derham.P1([j_exact_x, j_exact_y, j_exact_z])

            solver_params = SolverParameters(
                tol=1.0e-10,
                maxiter=3000,
                info=True,
                recycle=False,
            )

            _e = FEECVariable(space="Hcurl")
            _e.allocate(derham=derham, domain=domain)

            curlcurl_solver = CurlCurlSolve(j=j)
            curlcurl_solver.variables.e = _e

            curlcurl_solver.options = curlcurl_solver.Options(
                sigma=sigma,
                solver="pcg",
                precond="MassMatrixPreconditioner",
                solver_params=solver_params,
            )

            curlcurl_solver.allocate()

            dt = 1.0
            curlcurl_solver(dt)

            E_calculated = xp.array(curlcurl_solver.variables.e.spline(ee1, ee2, ee3))
            logger.info(f"{E_calculated.shape = }")
            E_analytical = xp.array([E_exact_x(ee1, ee2, ee3), E_exact_y(ee1, ee2, ee3), E_exact_z(ee1, ee2, ee3)])
            logger.info(f"{E_analytical.shape = }")

            if show_plot:
                plt.subplot(2, int((Nmax - Nmin) / 2 + 1), n + 1)
                if direction == "1":
                    plt.plot(e, E_calculated[2][:, 0, 0], "o", label=f"{Nel}, numerical")
                    plt.plot(e, E_analytical[2][:, 0, 0], "k--", label=f"{Nel}, analytical")

                elif direction == "2":
                    plt.plot(e, E_calculated[1][0, :, 0], "o", label=f"{Nel}, numerical")
                    plt.plot(e, E_analytical[1][0, :, 0], "k--", label=f"{Nel}, analytical")

                elif direction == "3":
                    plt.plot(e, E_calculated[0][0, 0, :], "o", label=f"{Nel}, numerical")
                    plt.plot(e, E_analytical[0][0, 0, :], "k--", label=f"{Nel}, analytical")
                plt.legend()

            error = xp.max(xp.abs(E_calculated - E_analytical))
            errors.append(error)

            h = 1 / Nel
            h_vec.append(h)

        m, _ = xp.polyfit(xp.log(Nels), xp.log(errors), deg=1)
        logger.info(f"For {p =}, solution converges with rate {-m =} ")

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
            plt.show()

        tolerance: float = 0.07
        assert -m > (p + 1 - tolerance)


@pytest.mark.parametrize("bc_type", ["dirichlet", "periodic"])
@pytest.mark.parametrize("direction", ["1", "2", "3"])
def test_convergence_2d(
    bc_type: str,
    direction: str,
    show_plot: bool = False,
):
    """Test of the solver on 2d problems by means of manufactured solution"""
    pmax = 4
    Nmin = 3
    Nmax = 6
    sigma = 1.5

    # Compact direction- and BC-dependent setup in one lookup table.
    periodic_mode = {
        "trig": xp.sin,
        "freq": (4, 6),
        "coef": 52,
    }
    mode_map = {
        "dirichlet": {
            **periodic_mode,
            "bcs": {
                "1": (None, ("dirichlet", "dirichlet"), ("dirichlet", "dirichlet")),
                "2": (("dirichlet", "dirichlet"), None, ("dirichlet", "dirichlet")),
                "3": (("dirichlet", "dirichlet"), ("dirichlet", "dirichlet"), None),
            },
        },
        "periodic": {
            **periodic_mode,
            "bcs": {"1": (None, None, None), "2": (None, None, None), "3": (None, None, None)},
        },
    }

    space_map = {
        "1": {
            "component": 0,
            "coords": (1, 2),
            "degree": lambda p: (1, p, p),
            "elements": lambda Nel: (1, Nel, Nel),
        },
        "2": {
            "component": 1,
            "coords": (2, 0),
            "degree": lambda p: (p, 1, p),
            "elements": lambda Nel: (Nel, 1, Nel),
        },
        "3": {
            "component": 2,
            "coords": (0, 1),
            "degree": lambda p: (p, p, 1),
            "elements": lambda Nel: (Nel, Nel, 1),
        },
    }

    mode = mode_map[bc_type]
    space = space_map[direction]

    if bc_type == "periodic" and Nmin < 4:
        Nmin = 4

    trig = mode["trig"]
    f0, f1 = mode["freq"]
    prefactor = mode["coef"] * (xp.pi**2) - sigma

    def scalar_exact(a, b):
        return trig(f0 * xp.pi * a) * trig(f1 * xp.pi * b)

    def scalar_current(a, b):
        return prefactor * scalar_exact(a, b)

    if direction == "1":
        E_exact_x = lambda x, y, z: scalar_exact(y, z)
        E_exact_y = lambda x, y, z: 0 * y
        E_exact_z = lambda x, y, z: 0 * z
        j_exact_x = lambda x, y, z: scalar_current(y, z)
        j_exact_y = lambda x, y, z: 0 * y
        j_exact_z = lambda x, y, z: 0 * z
    elif direction == "2":
        E_exact_x = lambda x, y, z: 0 * x
        E_exact_y = lambda x, y, z: scalar_exact(z, x)
        E_exact_z = lambda x, y, z: 0 * z
        j_exact_x = lambda x, y, z: 0 * x
        j_exact_y = lambda x, y, z: scalar_current(z, x)
        j_exact_z = lambda x, y, z: 0 * z
    elif direction == "3":
        E_exact_x = lambda x, y, z: 0 * x
        E_exact_y = lambda x, y, z: 0 * y
        E_exact_z = lambda x, y, z: scalar_exact(x, y)
        j_exact_x = lambda x, y, z: 0 * x
        j_exact_y = lambda x, y, z: 0 * y
        j_exact_z = lambda x, y, z: scalar_current(x, y)

    Nels = [2**n for n in range(Nmin, Nmax + 1)]

    e = xp.linspace(0.0, 1.0, 64)
    egrid = [0.0, 0.0, 0.0]
    for idx in space["coords"]:
        egrid[idx] = e
    e1, e2, e3 = egrid
    ee1, ee2, ee3 = xp.meshgrid(e1, e2, e3, indexing="ij")

    for p in range(2, pmax + 1):
        errors = []
        h_vec = []

        if show_plot:
            plt.figure(f"max-error, degree {p =}, {bc_type =}, {direction =}", figsize=(12, 8))

        for n, Nel in enumerate(Nels):
            degree = space["degree"](p)
            num_elements = space["elements"](Nel)
            bcs = mode["bcs"][direction]

            grid = TensorProductGrid(num_elements=num_elements)
            derham_opts = DerhamOptions(degree=degree, bcs=bcs)
            derham = Derham(grid=grid, options=derham_opts, comm=comm)

            mass_ops = WeightedMassOperators(derham=derham, domain=domain)

            Propagator.derham = derham
            Propagator.domain = domain
            Propagator.mass_ops = mass_ops

            j = FEECVariable(space="Hcurl")
            j.allocate(derham=derham, domain=domain)
            j.spline.vector = derham.P1([j_exact_x, j_exact_y, j_exact_z])

            solver_params = SolverParameters(
                tol=1.0e-10,
                maxiter=3000,
                info=True,
                recycle=False,
            )

            _e = FEECVariable(space="Hcurl")
            _e.allocate(derham=derham, domain=domain)

            curlcurl_solver = CurlCurlSolve(j=j)
            curlcurl_solver.variables.e = _e

            curlcurl_solver.options = curlcurl_solver.Options(
                sigma=sigma,
                solver="pcg",
                precond="MassMatrixPreconditioner",
                solver_params=solver_params,
            )

            curlcurl_solver.allocate()

            dt = 1.0
            curlcurl_solver(dt)

            E_calculated = xp.array(curlcurl_solver.variables.e.spline(ee1, ee2, ee3))
            logger.info(f"{E_calculated.shape = }")
            E_analytical = xp.array([E_exact_x(ee1, ee2, ee3), E_exact_y(ee1, ee2, ee3), E_exact_z(ee1, ee2, ee3)])
            logger.info(f"{E_analytical.shape = }")
            E_difference = xp.abs(E_calculated - E_analytical)

            if show_plot:
                plt.subplot(2, int((Nmax - Nmin) / 2 + 1), n + 1)

                if direction == "1":
                    plt.pcolormesh(e2, e3, E_difference[0][0, :, :], vmin=0.0, vmax=1.0, label=f"{Nel}x{Nel}")
                    plt.colorbar()

                elif direction == "2":
                    plt.pcolormesh(e3, e1, E_difference[1][:, 0, :], vmin=0.0, vmax=1.0, label=f"{Nel}x{Nel}")
                    plt.colorbar()

                elif direction == "3":
                    plt.pcolormesh(e1, e2, E_difference[2][:, :, 0], vmin=0.0, vmax=1.0, label=f"{Nel}x{Nel}")
                    plt.colorbar()

            error = xp.max(E_difference)
            errors.append(error)

            h = 1 / Nel
            h_vec.append(h)

        m, _ = xp.polyfit(xp.log(Nels), xp.log(errors), deg=1)
        logger.info(f"For {p =}, solution converges with rate {-m =} ")

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
            plt.show()

        tolerance: float = 0.07
        assert -m > (p + 1 - tolerance)


if __name__ == "__main__":
    # test_convergence_1d(bc_type="dirichlet", direction="1", show_plot=True)
    test_convergence_2d(bc_type="dirichlet", direction="1", show_plot=True)
