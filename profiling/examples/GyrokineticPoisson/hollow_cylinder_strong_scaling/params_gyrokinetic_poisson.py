# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "GyrokineticPoisson strong scaling on hollow cylinder, no preconditioner"
description = """
Strong scaling test for the gyrokinetic Poisson equation with adiabatic electrons on a hollow cylinder
(HollowCylinder, a1=1e-2, a2=1, Lz=2*pi) in a screw pinch equilibrium (ScrewPinch, a=1, R0=1).
The manufactured solution is a Gaussian 0-form on the logical domain, centered at eta = (0.5, 0.5, 0.5),
exciting the full spectrum.
Homogeneous Dirichlet boundary conditions are set in direction eta1.
The linear system is solved with unpreconditioned CG.
"""

import logging

from struphy import set_logging_level

set_logging_level(logging.INFO)

import argparse

import numpy as np

# ------------------
# Import Struphy API
# ------------------
from struphy import (
    BaseUnits,
    DerhamOptions,
    EnvironmentOptions,
    ProfilingOptions,
    Simulation,
    Time,
    domains,
    equils,
    grids,
)

# ---------------------
# Instance of the model
# ---------------------
from struphy.models import GyrokineticPoisson

# Units
base_units = BaseUnits()

# Model instance
epsilon = 1.0
charge_number = 1
model = GyrokineticPoisson(base_units=base_units, epsilon=epsilon, Z=charge_number)

# List all variables and decide whether to save their data
model.em_fields.phi.save_data = True
model.em_fields.source.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
# `--id` distinguishes runs that share a rank count but differ in something else; the
# profiling driver passes its launch counter (see `ProfilingJob.build_commands`).
# Unknown flags are ignored so the driver can forward other parameters as well.
parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, default=0, help="Run id, used to name the output folder.")
args, _ = parser.parse_known_args()

env = EnvironmentOptions(
    sim_folder=f"sim_{args.id:02d}",
    restart=False,
)

# Time stepping
time_opts = Time()

# Geometry (the screw pinch requires Lz = 2*pi*R0)
a1 = 1e-2
a2 = 1.0
R0 = 1.0
Lz = 2 * np.pi * R0
domain = domains.HollowCylinder(a1=a1, a2=a2, Lz=Lz)

# MHD equilibrium: a non-flat density and the pressure offset p0 keep n0/T0 = n0**2/p and n0/|B0|**2
# of order one and varying in r (the default p0 = 1e-8 would give n0/T0 ~ 1e7 at the boundary)
B0 = 1.0
q0 = 1.05
q1 = 1.80
n1 = 2.0
n2 = 1.0
na = 0.2
p0 = 0.1
equil = equils.ScrewPinch(a=a2, R0=R0, B0=B0, q0=q0, q1=q1, n1=n1, n2=n2, na=na, p0=p0)

# Grid
grid = grids.TensorProductGrid(num_elements=(64, 64, 64), mpi_dims_mask=(True, True, True))

# Derham options
derham_opts = DerhamOptions(degree=(1, 2, 3), bcs=(("dirichlet", "dirichlet"), None, None))

# Profiling options
profiling_opts = ProfilingOptions(
    use_line_profiler=True,
)

# Simulation object
sim = Simulation(
    model=model,
    name=name,
    description=description,
    params_path=__file__,
    env=env,
    time_opts=time_opts,
    domain=domain,
    equil=equil,
    grid=grid,
    derham_opts=derham_opts,
    profiling_opts=profiling_opts,
)

# ------------------
# Propagator options
# ------------------

from struphy.linear_algebra.solver import SolverParameters

solver_params = SolverParameters(tol=1e-8, maxiter=3000, info=True, recycle=False)
model.propagators.gyrokinetic_poisson.options = model.propagators.gyrokinetic_poisson.Options(
    which_geometry="cylindrical",
    solver="pcg",
    precond=None,
    solver_params=solver_params,
)

# ------------------
# Initial conditions
# ------------------
from scipy.special import erf

from struphy.initial.base import GenericPerturbation

# The exact solution is a Gaussian 0-form on the logical domain, centered at eta = (0.5, 0.5, 0.5).
# Its value at the Dirichlet boundary eta1 = 0, 1 is exp(-0.25 / w**2) ~ 1e-11, and it is periodic
# in eta2 and eta3 up to the same accuracy. Being localized, it excites the whole spectrum of the
# operator, so CG has to work for convergence. Both the solution and the right-hand side are 0-forms
# given as functions of the logical coordinates (given_in_basis="0"); the pushed-forward solution
# at x = F(eta) is phi(eta).
#
# The model solves (see GyrokineticPoisson.doc_pde)
#
#   1/(charge_number*epsilon**2) * n0/T0 * (phi - <phi>) - div(n0/|B0|**2 * (I - b0 b0^T) grad(phi)) = rho / epsilon,
#
# where <phi> is the average over eta3 weighted with n0/T0 * sqrt(g). In logical coordinates the
# diffusion term reads -1/sqrt(g) * d_i(C^ij d_j phi), with C^ij = sqrt(g) * n0/|B0|**2 * (G^ij - b^i b^j),
# the inverse metric G^ij and the contravariant components b^i of b0.
w = 0.1
eta0 = (0.5, 0.5, 0.5)


def gaussian_derivatives(e1, e2, e3):
    """The Gaussian, its first and (pure) second derivatives and the mixed derivative d2 d3."""
    eta = (e1, e2, e3)
    g = np.exp(-sum((e - c) ** 2 for e, c in zip(eta, eta0)) / w**2)
    d = [-2 * (e - c) / w**2 * g for e, c in zip(eta, eta0)]
    dd = [(4 * (e - c) ** 2 / w**4 - 2 / w**2) * g for e, c in zip(eta, eta0)]
    d23 = 4 * (e2 - eta0[1]) * (e3 - eta0[2]) / w**4 * g
    return g, d, dd, d23


def exact_solution(e1, e2, e3):
    return gaussian_derivatives(e1, e2, e3)[0]


def equilibrium_coefficients(e1, e2):
    """sqrt(g), n0/T0 and C^11, C^22, C^23, C^33 (C^12 = C^13 = 0) as functions of (eta1, eta2).

    HollowCylinder: r = a1 + (a2 - a1) * eta1, theta = 2*pi*eta2, z = Lz * eta3, with the scale factors
    h1 = a2 - a1, h2 = 2*pi*r, h3 = Lz. ScrewPinch: B0 = B0 * (e_z + r / (q R0) e_theta),
    q = q0 + (q1 - q0) * r**2 / a**2, n0 = na + (1 - na) * (1 - (r/a)**n1)**n2 and
    p = p0 + B0**2 * (a/R0)**2 * q0 / (2 * (q1 - q0)) * (1/q**2 - 1/q1**2), T0 = p/n0.
    All coefficients depend on eta1 only. Only operations defined for complex numbers are used,
    such that the derivatives can be computed with the complex step.
    """
    da = a2 - a1
    r = a1 + da * e1
    q = q0 + (q1 - q0) * r**2 / a2**2
    n = na + (1 - na) * (1 - (r / a2) ** n1) ** n2
    p = p0 + B0**2 * (a2 / R0) ** 2 * q0 / (2 * (q1 - q0)) * (1 / q**2 - 1 / q1**2)
    b_theta = B0 * r / (R0 * q)
    b_z = B0 + 0 * r
    abs_b_sq = b_theta**2 + b_z**2

    h1, h2, h3 = da, 2 * np.pi * r, Lz
    sqrt_g = h1 * h2 * h3
    # contravariant components of the unit vector b0 = B0 / |B0| (b^1 = 0)
    b2 = b_theta / np.sqrt(abs_b_sq) / h2
    b3 = b_z / np.sqrt(abs_b_sq) / h3

    weight = sqrt_g * n / abs_b_sq
    c11 = weight / h1**2
    c22 = weight * (1 / h2**2 - b2**2)
    c23 = -weight * b2 * b3
    c33 = weight * (1 / h3**2 - b3**2)
    return sqrt_g, n**2 / p, c11, c22, c23, c33


def complex_step(f, x, h=1e-30):
    """Derivative f'(x) with the complex step, exact up to machine precision for analytic f."""
    return f(x + 1j * h).imag / h


def weighted_average(e1, e2, e3):
    """Average of the exact solution over eta3, weighted with n0/T0 * sqrt(g).

    The weight depends on eta1 only, hence it is the plain integral of the Gaussian over eta3.
    """
    g1 = np.exp(-((e1 - eta0[0]) ** 2) / w**2)
    g2 = np.exp(-((e2 - eta0[1]) ** 2) / w**2)
    int3 = w * np.sqrt(np.pi) * erf(0.5 / w)
    return g1 * g2 * int3 + 0 * e3


def gyrokinetic_operator(e1, e2, e3):
    """Left-hand side of the gyrokinetic Poisson equation applied to the exact solution."""
    g, d, dd, d23 = gaussian_derivatives(e1, e2, e3)
    sqrt_g, n_over_t, c11, c22, c23, c33 = equilibrium_coefficients(e1, e2)
    dc11 = complex_step(lambda x: equilibrium_coefficients(x, e2)[2], e1)
    dc22 = complex_step(lambda x: equilibrium_coefficients(e1, x)[3], e2)
    dc23 = complex_step(lambda x: equilibrium_coefficients(e1, x)[4], e2)
    div = dc11 * d[0] + c11 * dd[0] + dc22 * d[1] + c22 * dd[1] + dc23 * d[2] + 2 * c23 * d23 + c33 * dd[2]
    return n_over_t / (charge_number * epsilon**2) * (g - weighted_average(e1, e2, e3)) - div / sqrt_g


def rhs_fun(e1, e2, e3):
    return epsilon * gyrokinetic_operator(e1, e2, e3)


rhs_perturbation = GenericPerturbation(rhs_fun, given_in_basis="0")

model.em_fields.source.add_perturbation(rhs_perturbation)


if __name__ == "__main__":
    # switches for debugging and testing
    estimate_mem = False
    run = True
    pproc = True
    save_figs = True

    if estimate_mem:
        sim.estimate_mem(print_report=True)
        exit()

    if run:
        out = sim.run(profiling_activated=True, one_time_step=True)
    else:
        # use this for working with existing sim data from a previous run
        out = sim.output

    if pproc:
        out.pproc(create_vtk=True, parallel=True)

    from matplotlib import pyplot as plt

    def slice_axes(k, X, Y, Z, E1, E2, E3):
        """Axes (horizontal, vertical, labels) for plotting the slice eta_k = 0.5: the cross-section (x, y)
        at eta3 = 0.5, the half plane (r, z) at eta2 = 0.5 and the logical (eta2, eta3) at eta1 = 0.5."""
        if k == 0:
            return E2, E3, "eta2", "eta3"
        elif k == 1:
            return np.sqrt(X**2 + Y**2), Z, "r", "z"
        else:
            return X, Y, "x", "y"

    def plot_slices(num, exact, name):
        """Numerical solution, exact solution and error on the slices eta_k = 0.5 (k = 1, 2, 3)."""
        fig = plt.figure(figsize=(16, 12))
        for k in range(3):
            idx = tuple(num.shape[k] // 2 if i == k else slice(None) for i in range(3))
            h, v, hlabel, vlabel = slice_axes(k, X[idx], Y[idx], Z[idx], E1[idx], E2[idx], E3[idx])
            eta_k = (E1, E2, E3)[k][idx].flat[0]
            for row, (data, what) in enumerate(
                [(num[idx], "from struphy"), (exact[idx], "exact"), (np.abs(num[idx] - exact[idx]), "error")]
            ):
                plt.subplot(3, 3, 3 * row + k + 1)
                plt.pcolormesh(h, v, data, shading="gouraud")
                plt.colorbar()
                plt.xlabel(hlabel)
                plt.ylabel(vlabel)
                plt.title(f"{name} {what}, slice at eta{k + 1} = {eta_k:.2f}")
        fig.tight_layout()
        return fig

    if sim.rank == 0:
        # Raw FEEC fields are evaluated directly from the saved spline coefficients at the
        # cell centres of the simulation grid (serial Derham, safe on rank 0 only).
        rhs = out.evaluate("em_fields/source", t=0).values

        phi_data = out.evaluate("em_fields/phi", t=-1)
        phi = phi_data.values

        # logical coordinates of the evaluation points and their images x = F(eta)
        E1, E2, E3 = np.meshgrid(
            *(phi_data.coords[c].values for c in ("eta1", "eta2", "eta3")),
            indexing="ij",
        )
        X, Y, Z = (phi_data.coords[c].values for c in ("X", "Y", "Z"))

        # pushed-forward exact solutions: the 0-forms at x = F(eta) are their values at eta
        rhs_exact = rhs_fun(E1, E2, E3)
        phi_exact = exact_solution(E1, E2, E3)

        fig_rhs = plot_slices(rhs, rhs_exact, "RHS")
        fig_phi = plot_slices(phi, phi_exact, "Phi")

        rel_err_rhs = np.max(np.abs(rhs - rhs_exact)) / np.max(np.abs(rhs_exact))
        rel_err_phi = np.max(np.abs(phi - phi_exact)) / np.max(np.abs(phi_exact))

        print(f"Max relative error in RHS: {rel_err_rhs:.2e}")
        print(f"Max relative error in Phi: {rel_err_phi:.2e}")

        assert rel_err_rhs < 2.2e-2, f"The computed RHS does not match the exact RHS, max rel error = {rel_err_rhs}."
        assert rel_err_phi < 1.3e-2, (
            f"The computed solution does not match the exact solution, max rel error = {rel_err_phi}."
        )

        import os

        # `out.path_out` is the out's (absolute) output folder; `sim_folder` alone is a bare
        # name resolved against the CWD. The profiling packaging picks these files up from
        # here and uploads them as `results-out<id>`.
        results_dir = os.path.join(out.path_out, "results")
        os.makedirs(results_dir, exist_ok=True)

        np.save(os.path.join(results_dir, "rel_err_rhs.npy"), rel_err_rhs)
        np.save(os.path.join(results_dir, "rel_err_phi.npy"), rel_err_phi)
        np.save(os.path.join(results_dir, "resolution.npy"), sim.grid.num_elements)
        np.save(os.path.join(results_dir, "spline_degree.npy"), sim.derham_opts.degree)

        if save_figs:
            fig_rhs.savefig(os.path.join(results_dir, "rhs_slices.png"))
            fig_phi.savefig(os.path.join(results_dir, "phi_slices.png"))
        else:
            plt.show()
