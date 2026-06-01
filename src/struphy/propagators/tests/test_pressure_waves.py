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
from struphy.feec.basis_projection_ops import BasisProjectionOperators
from struphy.feec.psydac_derham import Derham
from struphy.geometry.base import Domain
from struphy.io.options import DerhamOptions
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.propagators.pressure_wave import PressureWave
from struphy.topology.grids import TensorProductGrid
from struphy.utils.pyccel import Pyccelkernel

logger = logging.getLogger("struphy")
set_logging_level(logging.INFO)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
plt.rcParams.update({"font.size": 22})

# Pressure Waves test: polynomial test field on the logic cube

domain: Domain = domains.Cuboid()

@pytest.mark.parametrize("bc_type", ["dirichlet","periodic"])
@pytest.mark.parametrize("direction",["1","2","3"])


def test_convergence_1d(
    bc_type: str,
    direction: str,
    show_plot: bool = False,
):
    """Test of the solver on 1d problems by means of manufactured solution.
    
    Tests done considering constant rhobar = 1"""

    pmax: int = 4
    Nmin: int = 3
    Nmax: int = 8
    omega: float = 1.5
    mass: float = 2.0
    Z: float = 1

    u_exact = lambda e: xp.sin(4*xp.pi * e)
    rho_exact_e = lambda e: - 4*xp.pi * xp.cos(4*xp.pi * e) / omega
    theta_exact_e = lambda e: xp.sin(2*xp.pi * e)
    E_exact = lambda e: - 4*xp.pi/(Z*omega) * 2*xp.pi*xp.cos(2*xp.pi * e) * xp.cos(4*xp.pi * e) - (mass*omega/Z - (16*xp.pi**2) * xp.sin(2*xp.pi * e) / (Z*omega))*xp.sin(4*xp.pi * e)

    if bc_type == "dirichlet":
        u_exact = lambda e: xp.cos(5*xp.pi * e)
        rho_exact_e = lambda e: 5*xp.pi * xp.sin(5*xp.pi * e) / omega
        theta_exact_e = lambda e: e * (1 - e)
        E_exact = lambda e: 5*xp.pi/(Z*omega) * (1 - 2*e) * xp.sin(5*xp.pi * e) - (mass*omega/Z - (25*xp.pi**2) * e * (1 - e) / (Z*omega))*xp.cos(5*xp.pi * e)
    
    # Test over spline degree and grid resolution
    Nels = [2**n for n in range(Nmin, Nmax + 1)]

    e1 = 0.0
    e2 = 0.0
    e3 = 0.0
    e = xp.linspace(0.0, 1.0, 64)

    bcs = (None, None, None)

    if direction == "1":
        u_exact_x = lambda x,y,z: u_exact(x)
        u_exact_y = lambda x,y,z: 0*y
        u_exact_z = lambda x,y,z: 0*z

        rho_exact = lambda x,y,z: rho_exact_e(x)

        theta_exact = lambda x,y,z: theta_exact_e(x)

        E_exact_x = lambda x,y,z: E_exact(x)
        E_exact_y = lambda x,y,z: 0*y
        E_exact_z = lambda x,y,z: 0*z

        e1 = e
    
    elif direction == "2":
        u_exact_x = lambda x,y,z: 0*x
        u_exact_y = lambda x,y,z: u_exact(y)
        u_exact_z = lambda x,y,z: 0*z

        rho_exact = lambda x,y,z: rho_exact_e(y)

        theta_exact = lambda x,y,z: theta_exact_e(y)

        E_exact_x = lambda x,y,z: 0*x
        E_exact_y = lambda x,y,z: E_exact(y)
        E_exact_z = lambda x,y,z: 0*z

        e2 = e
    
    elif direction == "3":
        u_exact_x = lambda x,y,z: 0*x
        u_exact_y = lambda x,y,z: 0*y
        u_exact_z = lambda x,y,z: u_exact(z)

        rho_exact = lambda x,y,z: rho_exact_e(z)

        theta_exact = lambda x,y,z: theta_exact_e(z)

        E_exact_x = lambda x,y,z: 0*x
        E_exact_y = lambda x,y,z: 0*y
        E_exact_z = lambda x,y,z: E_exact(z)

        e3 = e

    ee1, ee2, ee3 = xp.meshgrid(e1, e2, e3, indexing="ij")

    for p in range(2, pmax + 1):
        rho_errors = []
        u_errors = []
        h_vec = []

        for n, Nel in enumerate(Nels):
            if direction == "1":
                degree = [p, 1, 1]
                num_elements = [Nel, 1, 1]
                if bc_type == "dirichlet":
                    bcs = (("dirichlet", "dirichlet"), None, None)

            elif direction == "2":
                degree = [1, p, 1]
                num_elements = [1, Nel, 1]
                if bc_type == "dirichlet":
                    bcs = (None, ("dirichlet", "dirichlet"), None)

            elif direction == "3":
                degree = [1, 1, p]
                num_elements = [1, 1, Nel]
                if bc_type == "dirichlet":
                    bcs = (None, None, ("dirichlet", "dirichlet"))

            grid = TensorProductGrid(num_elements=num_elements)
            derham_opts = DerhamOptions(degree=degree, bcs=bcs)
            derham = Derham(grid=grid, options=derham_opts, comm=comm)

            mass_ops = WeightedMassOperators(derham=derham, domain=domain)
            basis_ops = BasisProjectionOperators(derham=derham, domain=domain)

            Propagator.derham = derham
            Propagator.domain = domain
            Propagator.mass_ops = mass_ops
            Propagator.basis_ops = basis_ops

            E = FEECVariable(space="Hcurl")
            E.allocate(derham=derham, domain=domain)
            E.spline.vector = derham.P1([E_exact_x, E_exact_y, E_exact_z])
            
            theta = FEECVariable(space="H1")
            theta.allocate(derham=derham, domain=domain)
            theta.spline.vector = derham.P0(theta_exact)

            solver_params = SolverParameters(
                tol=1.0e-10,
                maxiter=3000,
                info=True,
                recycle=False,
            )

            _rho = FEECVariable(space="H1")
            _rho.allocate(derham=derham, domain=domain)
            
            _u = FEECVariable(space="Hcurl")
            _u.allocate(derham=derham, domain=domain)

            pressure_wave_solver = PressureWave()
            pressure_wave_solver.variables.rho = _rho
            pressure_wave_solver.variables.u = _u

            pressure_wave_solver.options = pressure_wave_solver.Options(
                omega=omega,
                mass=mass,
                Z=Z,
                E=E,
                theta=theta,
                solver="pcg",
                precond="MassMatrixPreconditioner",
                solver_params=solver_params,
            )

            pressure_wave_solver.allocate()

            dt=1.0
            pressure_wave_solver(dt)

            rho_calculated = xp.array(_rho.spline(ee1, ee2, ee3))
            u_calculated = xp.array(_u.spline(ee1, ee2, ee3))
            rho_analytical = rho_exact(ee1, ee2, ee3)
            u_analytical = xp.array([u_exact_x(ee1, ee2, ee3), u_exact_y(ee1, ee2, ee3), u_exact_z(ee1, ee2, ee3)])

            if show_plot:

                plt.figure(f"u, degree {p =}, {bc_type =}, {direction =}")
                plt.subplot(2, int((Nmax - Nmin) / 2 + 1), n + 1)

                if direction == "1":
                    plt.plot(e, u_calculated[0][:, 0, 0], "bo", label=f"{Nel}, numerical")
                    plt.plot(e, u_analytical[0][:, 0, 0], "k--", label=f"{Nel}, analytical")

                elif direction == "2":
                    plt.plot(e, u_calculated[1][0, :, 0], "bo", label=f"{Nel}, numerical")
                    plt.plot(e, u_analytical[1][0, :, 0], "k--", label=f"{Nel}, analytical")

                elif direction == "3":
                    plt.plot(e, u_calculated[2][0, 0, :], "bo", label=f"{Nel}, numerical")
                    plt.plot(e, u_analytical[2][0, 0, :], "k--", label=f"{Nel}, analytical")

                plt.legend()

                plt.figure(f"rho, degree {p =}, {bc_type =}, {direction =}")
                plt.subplot(2, int((Nmax - Nmin) / 2 + 1), n + 1)

                if direction == "1":
                    plt.plot(e, rho_calculated[:, 0, 0], "ro", label=f"{Nel}, numerical")
                    plt.plot(e, rho_analytical[:, 0, 0], "k--", label=f"{Nel}, analytical")

                elif direction == "2":
                    plt.plot(e, rho_calculated[0, :, 0], "ro", label=f"{Nel}, numerical")
                    plt.plot(e, rho_analytical[0, :, 0], "k--", label=f"{Nel}, analytical")

                elif direction == "3":
                    plt.plot(e, rho_calculated[0, 0, :], "ro", label=f"{Nel}, numerical")
                    plt.plot(e, rho_analytical[0, 0, :], "k--", label=f"{Nel}, analytical")

                plt.legend()

            rho_error = xp.max(xp.abs(rho_calculated - rho_analytical))
            rho_errors.append(rho_error)

            u_error = xp.max(xp.abs(u_calculated - u_analytical))
            u_errors.append(u_error)

            h = 1 / Nel
            h_vec.append(h)
        
        m_rho, _ = xp.polyfit(xp.log(Nels), xp.log(rho_errors), deg=1)
        logger.info(f"For {p =}, rho solution converges with rate {-m_rho =} ")

        m_u, _ = xp.polyfit(xp.log(Nels), xp.log(u_errors), deg=1)
        logger.info(f"For {p =}, u solution converges with rate {-m_u =} ")
        
        tolerance: float = 0.07
        
        assert -m_rho > (p + 1 - tolerance)
        assert -m_u > (p - tolerance)

        if show_plot:

            plt.figure(f"u convergence for degree {p =}", figsize=(12, 8))
            plt.title(f"u convergence rate for degree {p =}")
            plt.plot(h_vec, u_errors, "bo", label=f"Calculated u error, {m_u =}")
            plt.plot(
                h_vec,
                [(h ** p) / (h_vec[0] ** p) * u_errors[0] for h in h_vec],
                "k--",
                label="Theoretical u error, rate = p",
            )
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Grid spacing h")
            plt.ylabel("Error")
            plt.legend()

            plt.figure(f"Rho convergence for degree {p =}", figsize=(12, 8))
            plt.title(f"Rho convergence rate for degree {p =}")
            plt.plot(h_vec, rho_errors, "ro", label=f"Calculated rho error, {m_rho =}")
            plt.plot(
                h_vec,
                [(h ** (p + 1)) / (h_vec[0] ** (p + 1)) * rho_errors[0] for h in h_vec],
                "k--",
                label="Theoretical rho error, rate = p + 1",
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
        direction="3",
        show_plot=True,
    )
