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
from struphy.propagators.pressure_wave import PressureWave
from struphy.topology.grids import TensorProductGrid
from struphy.utils.pyccel import Pyccelkernel

logger = logging.getLogger("struphy")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
plt.rcParams.update({"font.size": 22})

# Pressure Waves test: polynomial test field on the logic cube

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
    omega: float = 10,
    mass: float = 2,
    Z: float = 1,
    show_plot: bool = False,
):
    """Test of the solver on 1d problems by means of manufactured solution.
    
    Tests done considering constant rhobar = 1 and theta = x (or y or z depending on direction chosen)"""

    if direction == "1":

        def u_exact_x(x,y,z) -> float:
            return xp.sin(5*xp.pi * x)
        
        def u_exact_y(x,y,z) -> float:
            return 0*y
        
        def u_exact_z(x,y,z) -> float:
            return 0*z
        
        def rho_exact(x,y,z) -> float:
            return 5*xp.pi * xp.cos(5*xp.pi * x) / omega
        
        def theta_exact(x,y,z) -> float:
            return x
        
        def E_exact_x(x,y,z) -> float:
            return 5*xp.pi/(Z*omega) * xp.cos(5*xp.pi * x) - (mass*omega/Z + (25*xp.pi**2) * x / (Z*omega))*xp.sin(5*xp.pi * x)
        
        def E_exact_y(x,y,z) -> float:
            return 0*y
        
        def E_exact_z(x,y,z) -> float:
            return 0*z
    
    elif direction == "2":

        def u_exact_x(x,y,z) -> float:
            return 0*x
        
        def u_exact_y(x,y,z) -> float:
            return xp.sin(5*xp.pi * y)
        
        def u_exact_z(x,y,z) -> float:
            return 0*z
        
        def rho_exact(x,y,z) -> float:
            return 5*xp.pi * xp.cos(5*xp.pi * y) / omega
        
        def theta_exact(x,y,z) -> float:
            return y
        
        def E_exact_x(x,y,z) -> float:
            return 0*x
        
        def E_exact_y(x,y,z) -> float:
            return 5*xp.pi/(Z*omega) * xp.cos(5*xp.pi * y) - (mass*omega/Z + (25*xp.pi**2) * y / (Z*omega))*xp.sin(5*xp.pi * y)
        
        def E_exact_z(x,y,z) -> float:
            return 0*z
    
    elif direction == "3":

        def u_exact_x(x,y,z) -> float:
            return 0*x
        
        def u_exact_y(x,y,z) -> float:
            return 0*y
        
        def u_exact_z(x,y,z) -> float:
            return xp.sin(5*xp.pi * z)
        
        def rho_exact(x,y,z) -> float:
            return 5*xp.pi * xp.cos(5*xp.pi * z) / omega
        
        def theta_exact(x,y,z) -> float:
            return z
        
        def E_exact_x(x,y,z) -> float:
            return 0*x
        
        def E_exact_y(x,y,z) -> float:
            return 0*y
        
        def E_exact_z(x,y,z) -> float:
            return 5*xp.pi/(Z*omega) * xp.cos(5*xp.pi * z) - (mass*omega/Z + (25*xp.pi**2) * z / (Z*omega))*xp.sin(5*xp.pi * z)
    
    if bc_type == "neumann":
        if direction == "1":

            def u_exact_x(x,y,z) -> float:
                return xp.cos(5*xp.pi * x)
            
            def u_exact_y(x,y,z) -> float:
                return 0*y
            
            def u_exact_z(x,y,z) -> float:
                return 0*z
            
            def rho_exact(x,y,z) -> float:
                return - 5*xp.pi * xp.sin(5*xp.pi * x) / omega
            
            def theta_exact(x,y,z) -> float:
                return x
            
            def E_exact_x(x,y,z) -> float:
                return - 5*xp.pi/(Z*omega) * xp.sin(5*xp.pi * x) - (mass*omega/Z + (25*xp.pi**2) * x / (Z*omega))*xp.cos(5*xp.pi * x)
            
            def E_exact_y(x,y,z) -> float:
                return 0*y
            
            def E_exact_z(x,y,z) -> float:
                return 0*z
    
        elif direction == "2":

            def u_exact_x(x,y,z) -> float:
                return 0*x
            
            def u_exact_y(x,y,z) -> float:
                return xp.cos(5*xp.pi * y)
            
            def u_exact_z(x,y,z) -> float:
                return 0*z
            
            def rho_exact(x,y,z) -> float:
                return - 5*xp.pi * xp.sin(5*xp.pi * y) / omega
            
            def theta_exact(x,y,z) -> float:
                return y
            
            def E_exact_x(x,y,z) -> float:
                return 0*x
            
            def E_exact_y(x,y,z) -> float:
                return - 5*xp.pi/(Z*omega) * xp.sin(5*xp.pi * y) - (mass*omega/Z + (25*xp.pi**2) * y / (Z*omega))*xp.cos(5*xp.pi * y)
            
            def E_exact_z(x,y,z) -> float:
                return 0*z
        
        elif direction == "3":

            def u_exact_x(x,y,z) -> float:
                return 0*x
            
            def u_exact_y(x,y,z) -> float:
                return 0*y
            
            def u_exact_z(x,y,z) -> float:
                return xp.cos(5*xp.pi * z)
            
            def rho_exact(x,y,z) -> float:
                return - 5*xp.pi * xp.sin(5*xp.pi * z) / omega
            
            def theta_exact(x,y,z) -> float:
                return z
            
            def E_exact_x(x,y,z) -> float:
                return 0*x
            
            def E_exact_y(x,y,z) -> float:
                return 0*y
            
            def E_exact_z(x,y,z) -> float:
                return - 5*xp.pi/(Z*omega) * xp.sin(5*xp.pi * z) - (mass*omega/Z + (25*xp.pi**2) * z / (Z*omega))*xp.cos(5*xp.pi * z)
    
    if bc_type == "periodic":

        if Nmin < 4:
            Nmin = 4
        
        if direction == "1":

            def u_exact_x(x,y,z) -> float:
                return xp.sin(4*xp.pi * x)
            
            def u_exact_y(x,y,z) -> float:
                return 0*y
            
            def u_exact_z(x,y,z) -> float:
                return 0*z
            
            def rho_exact(x,y,z) -> float:
                return 4*xp.pi * xp.cos(4*xp.pi * x) / omega
            
            def theta_exact(x,y,z) -> float:
                return x
            
            def E_exact_x(x,y,z) -> float:
                return 4*xp.pi/(Z*omega) * xp.cos(4*xp.pi * x) - (mass*omega/Z + (16*xp.pi**2) * x / (Z*omega))*xp.sin(4*xp.pi * x)
            
            def E_exact_y(x,y,z) -> float:
                return 0*y
            
            def E_exact_z(x,y,z) -> float:
                return 0*z
    
        elif direction == "2":

            def u_exact_x(x,y,z) -> float:
                return 0*x
            
            def u_exact_y(x,y,z) -> float:
                return xp.sin(4*xp.pi * y)
            
            def u_exact_z(x,y,z) -> float:
                return 0*z
            
            def rho_exact(x,y,z) -> float:
                return 4*xp.pi * xp.cos(4*xp.pi * y) / omega
            
            def theta_exact(x,y,z) -> float:
                return y
            
            def E_exact_x(x,y,z) -> float:
                return 0*x
            
            def E_exact_y(x,y,z) -> float:
                return 4*xp.pi/(Z*omega) * xp.cos(4*xp.pi * y) - (mass*omega/Z + (16*xp.pi**2) * y / (Z*omega))*xp.sin(4*xp.pi * y)
            
            def E_exact_z(x,y,z) -> float:
                return 0*z
        
        elif direction == "3":

            def u_exact_x(x,y,z) -> float:
                return 0*x
            
            def u_exact_y(x,y,z) -> float:
                return 0*y
            
            def u_exact_z(x,y,z) -> float:
                return xp.sin(4*xp.pi * z)
            
            def rho_exact(x,y,z) -> float:
                return 4*xp.pi * xp.cos(4*xp.pi * z) / omega
            
            def theta_exact(x,y,z) -> float:
                return z
            
            def E_exact_x(x,y,z) -> float:
                return 0*x
            
            def E_exact_y(x,y,z) -> float:
                return 0*y
            
            def E_exact_z(x,y,z) -> float:
                return 4*xp.pi/(Z*omega) * xp.cos(4*xp.pi * z) - (mass*omega/Z + (16*xp.pi**2) * z / (Z*omega))*xp.sin(4*xp.pi * z)
    
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
                degree = [p, 1, 1]
                num_elements = [Nel, 1, 1]
                bcs = (("dirichlet", "dirichlet"), None, None)
                e1 = xp.linspace(0.0, 1.0, 64)

            elif direction == "2":
                degree = [1, p, 1]
                num_elements = [1, Nel, 1]
                bcs = (None, ("dirichlet", "dirichlet"), None)
                e2 = xp.linspace(0.0, 1.0, 64)

            elif direction == "3":
                degree = [1, 1, p]
                num_elements = [1, 1, Nel]
                bcs = (None, None, ("dirichlet", "dirichlet"))
                e3 = xp.linspace(0.0, 1.0, 64)

            if bc_type == "periodic":
                if direction == "1":
                    degree = [p, 1, 1]
                    num_elements = [Nel, 1, 1]
                    bcs = (None, None, None)
                    e1 = xp.linspace(0.0, 1.0, 64)

                elif direction == "2":
                    degree = [1, p, 1]
                    num_elements = [1, Nel, 1]
                    bcs = (None, None, None)
                    e2 = xp.linspace(0.0, 1.0, 64)

                elif direction == "3":
                    degree = [1, 1, p]
                    num_elements = [1, 1, Nel]
                    bcs = (None, None, None)
                    e3 = xp.linspace(0.0, 1.0, 64)

            elif bc_type == "neumann":
                if direction == "1":
                    degree = [p, 1, 1]
                    num_elements = [Nel, 1, 1]
                    bcs = (("free", "free"), None, None)
                    e1 = xp.linspace(0.0, 1.0, 64)

                elif direction == "2":
                    degree = [1, p, 1]
                    num_elements = [1, Nel, 1]
                    bcs = (None, ("free", "free"), None)
                    e2 = xp.linspace(0.0, 1.0, 64)

                elif direction == "3":
                    degree = [1, 1, p]
                    num_elements = [1, 1, Nel]
                    bcs = (None, None, ("free", "free"))
                    e3 = xp.linspace(0.0, 1.0, 64)

            grid = TensorProductGrid(num_elements=num_elements)
            derham_opts = DerhamOptions(degree=degree, bcs=bcs)
            derham = Derham(grid=grid, options=derham_opts, comm=comm)

            mass_ops = WeightedMassOperators(derham=derham, domain=domain)

            Propagator.derham = derham
            Propagator.domain = domain
            Propagator.mass_ops = mass_ops

            def E_pulled_x(e1, e2, e3):
                return domain.pull([E_exact_x, E_exact_y, E_exact_z], e1, e2, e3, kind="1", squeeze_out=False)[0]

            def E_pulled_y(e1, e2, e3):
                return domain.pull([E_exact_x, E_exact_y, E_exact_z], e1, e2, e3, kind="1", squeeze_out=False)[1]

            def E_pulled_z(e1, e2, e3):
                return domain.pull([E_exact_x, E_exact_y, E_exact_z], e1, e2, e3, kind="1", squeeze_out=False)[2]

            E = FEECVariable(space="Hcurl")
            E.allocate(derham=derham, domain=domain)
            E.spline.vector = derham.P1([E_pulled_x, E_pulled_y, E_pulled_z])

            def theta_pulled(e1, e2, e3):
                return domain.pull(theta_exact, e1, e2, e3, kind="0", squeeze_out=False)
            
            theta = FEECVariable(space="H1")
            theta.allocate(derham=derham, domain=domain)
            theta.spline.vector = derham.P0(theta_pulled)

            solver_params = SolverParameters(
                tol=1.0e-10,
                maxiter=3000,
                info=True,
                verbose=False,
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

            rho_calculated = domain.push(_rho.spline, e1, e2, e3, kind="0")
            u_calculated = domain.push(_u.spline, e1, e2, e3, kind="1")
            x, y, z = domain(e1, e2, e3)
            rho_analytical = rho_exact(x, y, z)
            u_analytical = xp.array([u_exact_x(x, y, z), u_exact_y(x, y, z), u_exact_z(x, y, z)])

            error = xp.max(xp.max(xp.abs(rho_calculated - rho_analytical)),xp.max(xp.abs(u_calculated - u_analytical)))
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
        Nmax=6,
        show_plot=True
    )
