import logging

import cunumpy as xp
import matplotlib.pyplot as plt
import pytest
from cunumpy import cos, ones_like, pi, sin, zeros_like
from feectools.ddm.mpi import mpi as MPI

from struphy import domains, equils, grids, set_logging_level
from struphy.feec.basis_projection_ops import BasisProjectionOperators
from struphy.feec.mass import L2Projector, WeightedMassOperators
from struphy.feec.psydac_derham import Derham
from struphy.fields_background.projected_equils import ProjectedMHDequilibrium
from struphy.geometry.base import Domain
from struphy.initial.base import GenericPerturbation, Perturbation
from struphy.io.options import DerhamOptions
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.propagators.two_fluid_quasi_neutral_full import TwoFluidQuasiNeutralFull

logger = logging.getLogger("struphy")
set_logging_level(logging.INFO)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
# plt.rcParams.update({'font.size': 22})


@pytest.mark.parametrize("bc_type", ["periodic", "hom_dirichlet", "inhom_dirichlet"])
@pytest.mark.parametrize(
    "mapping",
    [
        None,
        ["Cuboid", {"l1": 0.0, "r1": 4.0, "l2": 0.0, "r2": 2.0, "l3": 0.0, "r3": 3.0}],
        ["Orthogonal", {"Lx": 4.0, "Ly": 2.0, "alpha": 0.1, "Lz": 3.0}],
    ],
)
def test_one_time_step(bc_type: str, mapping: None | list[str, dict], show_plot=False):
    """Test the propagator TwoFluidQuasiNeutralFull on a single time step against a manufactured solution with different boundary conditions."""

    # domain object
    if mapping is None:
        domain = domains.Cuboid()
    else:
        dom_type = mapping[0]
        dom_params = mapping[1]
        domain_class = getattr(domains, dom_type)
        domain: Domain = domain_class(**dom_params)

    # other options
    B0 = 0
    nu = 10.0
    nu_e = 1.0
    Nel = (32, 1, 1)
    degree = (1, 1, 1)
    epsilon = 1.0
    dt = 1
    Tend = 1
    sigma = 0
    tol = 1e-5

    # derham sequence
    if bc_type == "periodic":
        derham_opts = DerhamOptions(degree=degree, bcs=(None, None, None))

    elif bc_type == "hom_dirichlet":
        derham_opts = DerhamOptions(degree=degree, bcs=(("dirichlet", "dirichlet"), None, None))

    elif bc_type == "inhom_dirichlet":
        derham_opts = DerhamOptions(degree=degree, bcs=(("dirichlet", "dirichlet"), None, None))
        lifting_function_u = GenericPerturbation(lambda x, y, z: x + 1, comp=0, given_in_basis="physical")
        lifting_function_ue = GenericPerturbation(lambda x, y, z: x, comp=0, given_in_basis="physical")

    else:
        raise ValueError(f"Invalid bc_type: {bc_type}")

    grid = grids.TensorProductGrid(num_elements=Nel)
    derham = Derham(grid=grid, options=derham_opts, domain=domain)

    # fluid background
    equil = equils.HomogenSlab(B0x=0, B0y=0, B0z=B0, beta=0, n0=0)
    projected_equil = ProjectedMHDequilibrium(equil=equil, derham=derham)

    # mass operators
    mass_ops = WeightedMassOperators(derham=derham, domain=domain, eq_mhd=equil)

    # basis operators
    basis_ops = BasisProjectionOperators(derham, domain, eq_mhd=equil)

    # ---- manufactured solutions ----
    if bc_type == "periodic":

        def mms_phi(x, y, z):
            return xp.sin(2 * xp.pi * x), xp.zeros_like(x), xp.zeros_like(x)

        def mms_ion_u(x, y, z):
            return xp.sin(2 * xp.pi * x) + 1, xp.zeros_like(x), xp.zeros_like(x)

        def mms_electron_u(x, y, z):
            return xp.sin(2 * xp.pi * x), xp.zeros_like(x), xp.zeros_like(x)

    elif bc_type == "hom_dirichlet":

        def mms_phi(x, y, z):
            return xp.sin(2 * xp.pi * x), xp.zeros_like(x), xp.zeros_like(x)

        def mms_ion_u(x, y, z):
            return xp.sin(2 * xp.pi * x), xp.zeros_like(x), xp.zeros_like(x)

        def mms_electron_u(x, y, z):
            return xp.sin(2 * xp.pi * x), xp.zeros_like(x), xp.zeros_like(x)

    elif bc_type == "inhom_dirichlet":

        def mms_phi(x, y, z):
            return xp.sin(2 * xp.pi * x), xp.zeros_like(x), xp.zeros_like(x)

        def mms_ion_u(x, y, z):
            return xp.sin(2 * xp.pi * x) + x + 1, xp.zeros_like(x), xp.zeros_like(x)

        def mms_electron_u(x, y, z):
            return xp.sin(2 * xp.pi * x) + x, xp.zeros_like(x), xp.zeros_like(x)

    # ---- source terms ----
    if bc_type == "periodic":

        def source_function_u(x, y, z):
            fx = 2.0 * pi * (cos(2 * pi * x) + 2 * nu * pi * sin(2 * pi * x))
            fy = zeros_like(x)
            fz = zeros_like(x)
            return fx, fy, fz

        def source_function_ue(x, y, z):
            fx = -2.0 * pi * cos(2 * pi * x) + nu_e * 4.0 * pi**2 * sin(2 * pi * x) - sigma * sin(2 * pi * x)
            fy = zeros_like(x)
            fz = zeros_like(x)
            return fx, fy, fz

    elif bc_type == "hom_dirichlet":

        def source_function_u(x, y, z):
            fx = 2.0 * pi * (cos(2 * pi * x) + 2 * nu * pi * sin(2 * pi * x))
            fy = zeros_like(x)
            fz = zeros_like(x)
            return fx, fy, fz

        def source_function_ue(x, y, z):
            fx = -2.0 * pi * cos(2 * pi * x) + nu_e * 4.0 * pi**2 * sin(2 * pi * x) - sigma * sin(2 * pi * x)
            fy = zeros_like(x)
            fz = zeros_like(x)
            return fx, fy, fz

    elif bc_type == "inhom_dirichlet":

        def source_function_u(x, y, z):
            fx = 2.0 * pi * (cos(2 * pi * x) + 2 * nu * pi * sin(2 * pi * x))
            fy = zeros_like(x)
            fz = zeros_like(x)
            return fx, fy, fz

        def source_function_ue(x, y, z):
            fx = -2.0 * pi * cos(2 * pi * x) + (4.0 * nu_e * pi**2 - sigma) * sin(2 * pi * x) - sigma * x
            fy = zeros_like(x)
            fz = zeros_like(x)
            return fx, fy, fz

    # ---- perturbation classes for MMS initial conditions ----
    class MMSIonVelocity(Perturbation):
        def __init__(self, comp=0):
            self.comp = comp
            self.given_in_basis = "physical"

        def __call__(self, x, y, z):
            return mms_ion_u(x, y, z)[self.comp]

    class MMSElectronVelocity(Perturbation):
        def __init__(self, comp=0):
            self.comp = comp
            self.given_in_basis = "physical"

        def __call__(self, x, y, z):
            return mms_electron_u(x, y, z)[self.comp]

    class MMSPotential(Perturbation):
        def __init__(self):
            self.given_in_basis = "physical"

        def __call__(self, x, y, z):
            return mms_phi(x, y, z)[0]

    # instance of propagator
    Propagator.derham = derham
    Propagator.domain = domain
    Propagator.mass_ops = mass_ops
    Propagator.basis_ops = basis_ops
    Propagator.projected_equil = projected_equil

    prop = TwoFluidQuasiNeutralFull(allocate_variables=True)

    prop.options = prop.Options(
        nu=nu,
        nu_e=nu_e,
        eps_norm=epsilon,
        stab_sigma=sigma,
        source_u=source_function_u,
        source_ue=source_function_ue,
        solver="gmres",
        solver_params=SolverParameters(info=True, tol=tol),
    )

    prop.allocate()

    if bc_type == "inhom_dirichlet":
        prop.variables.u.lifting_function = lifting_function_u
        prop.variables.ue.lifting_function = lifting_function_ue

    prop(dt)

    x = xp.linspace(0, 1, 100)
    mms_phi_x, _, _ = mms_phi(x, x * 0, x * 0)
    mms_ion_ux, _, _ = mms_ion_u(x, x * 0, x * 0)
    mms_el_ux, _, _ = mms_electron_u(x, x * 0, x * 0)

    e1 = xp.linspace(0, 1, 64)

    if show_plot:
        plt.figure(figsize=(18, 8))

        plt.subplot(1, 3, 1)
        plt.plot(e1, prop.variables.u.spline(e1, 0.5, 0.5, squeeze_out=True)[0], label="numerical")
        plt.plot(x, mms_ion_ux, "--", label="manufactured")
        # plt.plot(e1, prop.variables.phi.spline(e1, 0.5, 0.5, squeeze_out=True), "k.", markersize=4, label="n1 points")
        plt.xlabel("x")
        plt.title(r"$u$")
        # plt.title(f"{title} at t={t:.3f}")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(e1, prop.variables.ue.spline(e1, 0.5, 0.5, squeeze_out=True)[0], label="numerical")
        plt.plot(x, mms_el_ux, "--", label="manufactured")
        # plt.plot(e1, prop.variables.ue.spline(e1, 0.5, 0.5, squeeze_out=True)[0], "k.", markersize=4, label="n1 points")
        plt.xlabel("x")
        plt.title(r"$u_e$")
        # plt.title(f"{title} at t={t:.3f}")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(e1, prop.variables.phi.spline(e1, 0.5, 0.5, squeeze_out=True), label="numerical")
        plt.plot(x, mms_phi_x, "--", label="manufactured")
        # plt.plot(e1, prop.variables.phi.spline(e1, 0.5, 0.5, squeeze_out=True), "k.", markersize=4, label="n1 points")
        plt.xlabel("x")
        plt.title(r"$\phi$")
        # plt.title(f"{title} at t={t:.3f}")
        plt.legend()
        plt.grid(True)

        plt.show()


if __name__ == "__main__":
    test_one_time_step(bc_type="inhom_dirichlet", mapping=None, show_plot=True)
