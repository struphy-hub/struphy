"""Generate the Poisson profiling cases in this folder from one template.

There are 3 mappings (cube, hollow cylinder, hollow torus) x 4 preconditioners (none, multigrid,
mass matrix, stiffness). For each, this script writes ``<case>/params_poisson.py`` and the submit
script ``submit_poisson_<case>.py`` next to it, and formats them with ruff. Edit the templates here
and rerun instead of editing the 12 generated cases by hand:

    python profiling/examples/Poisson/generate_cases.py

The error limits asserted at the end of each run (``ERROR_LIMITS``) were calibrated at 64^3
elements: about 1.3 times the measured errors. Recalibrate them when changing the exact solution,
the grid or the spline degrees.
"""

import shutil
import subprocess
from pathlib import Path

POISSON_DIR = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------------------------
# mappings
# ----------------------------------------------------------------------------------------------
MAPPINGS = {
    "cube": dict(
        folder="cube_strong_scaling",
        label="poisson_cube_strong_scaling",
        title="3D cube",
        geometry_desc="a 3D cube (Cuboid)",
        domain_code="""# Geometry
Lx = 2.0
Ly = 3.0
Lz = 4.0
domain = domains.Cuboid(r1=Lx, l2=-Ly / 2, r2=Ly / 2, r3=Lz)
""",
        laplacian_code='''def laplacian(e1, e2, e3):
    """Physical Laplacian of the exact solution, in logical coordinates (Cuboid: x_i = l_i + L_i * eta_i)."""
    g, d, dd = gaussian_derivatives(e1, e2, e3)
    return dd[0] / Lx**2 + dd[1] / Ly**2 + dd[2] / Lz**2
''',
        # slices (direction held fixed -> horizontal, vertical axis), see slice_axes in the template
        slice_axes_code='''def slice_axes(k, X, Y, Z, E1, E2, E3):
    """Axes (horizontal, vertical, labels) for plotting the slice eta_k = 0.5 in physical coordinates."""
    if k == 0:
        return Y, Z, "y", "z"
    elif k == 1:
        return X, Z, "x", "z"
    else:
        return X, Y, "x", "y"
''',
    ),
    "hollow_cylinder": dict(
        folder="hollow_cylinder_strong_scaling",
        label="poisson_hollow_cylinder_strong_scaling",
        title="hollow cylinder",
        geometry_desc="a hollow cylinder (HollowCylinder, a1=1e-2, a2=1, Lz=4)",
        domain_code="""# Geometry
a1 = 1e-2
a2 = 1.0
Lz = 4.0
domain = domains.HollowCylinder(a1=a1, a2=a2, Lz=Lz)
""",
        laplacian_code='''def laplacian(e1, e2, e3):
    """Physical Laplacian of the exact solution, in logical coordinates.

    HollowCylinder: r = a1 + (a2 - a1) * eta1, theta = 2*pi*eta2, z = Lz * eta3, and
    Laplace(phi) = phi_rr + phi_r / r + phi_theta,theta / r**2 + phi_zz.
    """
    g, d, dd = gaussian_derivatives(e1, e2, e3)
    da = a2 - a1
    r = a1 + da * e1
    return dd[0] / da**2 + d[0] / (da * r) + dd[1] / (2 * np.pi * r) ** 2 + dd[2] / Lz**2
''',
        slice_axes_code='''def slice_axes(k, X, Y, Z, E1, E2, E3):
    """Axes (horizontal, vertical, labels) for plotting the slice eta_k = 0.5: the cross-section (x, y)
    at eta3 = 0.5, the half plane (r, z) at eta2 = 0.5 and the logical (eta2, eta3) at eta1 = 0.5."""
    if k == 0:
        return E2, E3, "eta2", "eta3"
    elif k == 1:
        return np.sqrt(X**2 + Y**2), Z, "r", "z"
    else:
        return X, Y, "x", "y"
''',
    ),
    "hollow_torus": dict(
        folder="hollow_torus_strong_scaling",
        label="poisson_hollow_torus_strong_scaling",
        title="hollow torus",
        geometry_desc="a full hollow torus (HollowTorus, a1=1e-2, a2=1, R0=4)",
        domain_code="""# Geometry
a1 = 1e-2
a2 = 1.0
R0 = 4.0
tor_period = 1
domain = domains.HollowTorus(a1=a1, a2=a2, R0=R0, sfl=False, pol_period=1, tor_period=tor_period)
""",
        laplacian_code='''def laplacian(e1, e2, e3):
    """Physical Laplacian of the exact solution, in logical coordinates.

    HollowTorus (sfl=False): r = a1 + (a2 - a1) * eta1, theta = 2*pi*eta2, phi = 2*pi*eta3 / tor_period,
    R = R0 + r*cos(theta), metric dr**2 + r**2 dtheta**2 + R**2 dphi**2, and
    Laplace(u) = u_rr + (R + r*cos(theta)) / (r*R) * u_r + u_theta,theta / r**2 - sin(theta) / (r*R) * u_theta
    + u_phi,phi / R**2.
    """
    g, d, dd = gaussian_derivatives(e1, e2, e3)
    da = a2 - a1
    r = a1 + da * e1
    theta = 2 * np.pi * e2
    R = R0 + r * np.cos(theta)
    dphi = 2 * np.pi / tor_period
    return (
        dd[0] / da**2
        + (R + r * np.cos(theta)) / (r * R) * d[0] / da
        + dd[1] / (2 * np.pi * r) ** 2
        - np.sin(theta) / (r * R) * d[1] / (2 * np.pi)
        + dd[2] / (dphi * R) ** 2
    )
''',
        slice_axes_code='''def slice_axes(k, X, Y, Z, E1, E2, E3):
    """Axes (horizontal, vertical, labels) for plotting the slice eta_k = 0.5: the poloidal plane (R, z)
    at eta3 = 0.5 and the logical planes at eta1 = 0.5 and eta2 = 0.5."""
    if k == 0:
        return E2, E3, "eta2", "eta3"
    elif k == 1:
        return E1, E3, "eta1", "eta3"
    else:
        return np.sqrt(X**2 + Y**2), Z, "R", "z"
''',
    ),
}

# ----------------------------------------------------------------------------------------------
# preconditioners
# ----------------------------------------------------------------------------------------------
PRECONDS = {
    "none": dict(
        suffix="",
        title="no preconditioner",
        solve_desc="unpreconditioned CG",
        extra_import="",
        maxiter=3000,
        options="    precond=None,\n",
    ),
    "multigrid": dict(
        suffix="_multigrid",
        title="multigrid preconditioner",
        solve_desc="CG preconditioned by geometric multigrid",
        extra_import="    MultiGridOptions,\n",
        maxiter=3000,
        options='    precond="MultiGrid",\n    multigrid=MultiGridOptions(),\n',
    ),
    "massmatrix": dict(
        suffix="_massmatrix",
        title="mass-matrix preconditioner",
        solve_desc=(
            "CG preconditioned by the Kronecker approximation of the inverse 0-form mass matrix "
            "(MassMatrixPreconditioner)"
        ),
        extra_import="",
        maxiter=3000,
        options='    precond="MassMatrixPreconditioner",\n',
    ),
    "stiffness": dict(
        suffix="_stiffness",
        title="stiffness preconditioner",
        solve_desc="CG preconditioned by the Kronecker approximation of the inverse Laplacian (StiffnessPreconditioner)",
        extra_import="",
        maxiter=3000,
        options='    precond="StiffnessPreconditioner",\n',
    ),
}

# maximal relative errors (rhs, phi) asserted at the end of the run, per mapping; calibrated at 64^3,
# where the measured errors are cube (1.31e-2, 9.03e-3), hollow cylinder (1.65e-2, 9.72e-3) and
# hollow torus (1.71e-2, 9.85e-3)
ERROR_LIMITS = {
    "cube": ("1.7e-2", "1.2e-2"),
    "hollow_cylinder": ("2.2e-2", "1.3e-2"),
    "hollow_torus": ("2.2e-2", "1.3e-2"),
}

PARAMS_TEMPLATE = '''# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "Poisson strong scaling on @TITLE@, @PRECOND_TITLE@"
description = """
Strong scaling test for Poisson equation on @GEOMETRY_DESC@.
The manufactured solution is a Gaussian 0-form on the logical domain, centered at eta = (0.5, 0.5, 0.5),
exciting the full spectrum.
Homogeneous Dirichlet boundary conditions are set in direction eta1.
The linear system is solved with @SOLVE_DESC@.
"""

import logging

from struphy import set_logging_level

set_logging_level(logging.INFO)

import argparse

# ------------------
# Import Struphy API
# ------------------
from struphy import (
    BaseUnits,
    DerhamOptions,
    EnvironmentOptions,
    FieldsBackground,
@EXTRA_IMPORT@    ProfilingOptions,
    Simulation,
    Time,
    domains,
    equils,
    grids,
    perturbations,
)

# ---------------------
# Instance of the model
# ---------------------
from struphy.models import Poisson

# Units
base_units = BaseUnits()

# Model instance
model = Poisson(base_units=base_units)

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

@DOMAIN_CODE@
# Fluid equilibrium (can be used as part of initial conditions)
equil = None

# Grid
grid = grids.TensorProductGrid(num_elements=(64, 64, 64), mpi_dims_mask=(True, True, True))

# Derham options
derham_opts = DerhamOptions(degree=(1, 2, 3), bcs=(("dirichlet", "dirichlet"), None, None))

# Profilinig options
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

solver_params = SolverParameters(tol=1e-8, maxiter=@MAXITER@, info=True, recycle=False)
model.propagators.poisson.options = model.propagators.poisson.Options(
    stab_eps=0.0,
    solver="pcg",
@PRECOND_OPTIONS@    solver_params=solver_params,
)

# ------------------
# Initial conditions
# ------------------
import numpy as np

from struphy.initial.base import GenericPerturbation

# The exact solution is a Gaussian 0-form on the logical domain, centered at eta = (0.5, 0.5, 0.5).
# Its value at the Dirichlet boundary eta1 = 0, 1 is exp(-0.25 / w**2) ~ 1e-11, and it is periodic
# in eta2 and eta3 up to the same accuracy. Being localized, it excites the whole spectrum of the
# Laplacian, so CG has to work for convergence. Both the solution and the right-hand side
# -Laplace(phi) are 0-forms given as functions of the logical coordinates (given_in_basis="0");
# the pushed-forward solution at x = F(eta) is phi(eta).
w = 0.1
eta0 = (0.5, 0.5, 0.5)


def gaussian_derivatives(e1, e2, e3):
    """The Gaussian and its first and second derivatives with respect to eta1, eta2, eta3."""
    eta = (e1, e2, e3)
    g = np.exp(-sum((e - c) ** 2 for e, c in zip(eta, eta0)) / w**2)
    d = [-2 * (e - c) / w**2 * g for e, c in zip(eta, eta0)]
    dd = [(4 * (e - c) ** 2 / w**4 - 2 / w**2) * g for e, c in zip(eta, eta0)]
    return g, d, dd


def exact_solution(e1, e2, e3):
    return gaussian_derivatives(e1, e2, e3)[0]


@LAPLACIAN_CODE@

def rhs_fun(e1, e2, e3):
    return -laplacian(e1, e2, e3)


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

    @SLICE_AXES_CODE@
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

        assert rel_err_rhs < @RHS_LIMIT@, f"The computed RHS does not match the exact RHS, max rel error = {rel_err_rhs}."
        assert rel_err_phi < @PHI_LIMIT@, (
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
'''


def indent(code: str, n: int) -> str:
    return "\n".join((" " * n + line) if line else line for line in code.splitlines()) + "\n"


def render_params(mapping_key, precond_key, limits) -> str:
    m, p = MAPPINGS[mapping_key], PRECONDS[precond_key]
    rhs_limit, phi_limit = limits[mapping_key]
    s = PARAMS_TEMPLATE
    for key, val in {
        "@TITLE@": m["title"],
        "@PRECOND_TITLE@": p["title"],
        "@GEOMETRY_DESC@": m["geometry_desc"],
        "@SOLVE_DESC@": p["solve_desc"],
        "@EXTRA_IMPORT@": p["extra_import"],
        "@DOMAIN_CODE@": m["domain_code"],
        "@MAXITER@": str(p["maxiter"]),
        "@PRECOND_OPTIONS@": p["options"],
        "@LAPLACIAN_CODE@": m["laplacian_code"],
        "@SLICE_AXES_CODE@": indent(m["slice_axes_code"], 4).lstrip(),
        "@RHS_LIMIT@": rhs_limit,
        "@PHI_LIMIT@": phi_limit,
    }.items():
        s = s.replace(key, val)
    assert "@" not in s.replace("@property", ""), [l for l in s.splitlines() if "@" in l]
    return s


SUBMIT_TEMPLATE = '''"""Poisson strong scaling profiling case on @TITLE@, @PRECOND_TITLE@.

This file defines the Poisson strong scaling profiling case (the `ProfilingCase`)
and submits it: for each rank count, `ProfilingCase.launch` builds and submits a
SLURM script (using `clusters.SLURM_PRESETS` by default), or, without a batch
system, runs directly on this machine. `finalize_run` then packages and uploads
each run as soon as its own job finishes.
Each generated script runs the simulation itself by invoking `params_poisson.py`
directly (its `__main__` block is the worker).
"""

import argparse
import sys
from pathlib import Path

# the profiling driver (profiling_job.py, clusters.py, ...) lives in profiling/, two levels up
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from profiling_job import ProfilingCase  # noqa: E402


def main() -> None:

    # Parse arguments, do not remove --upload
    parser = argparse.ArgumentParser(
        description=("Submit profiling jobs to a SLURM cluster and package the results for upload."),
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the packaged profiling results to the profiling-data repo.",
    )
    parser.add_argument(
        "--language",
        default="fortran",
        choices=["fortran", "c"],
        help="Pyccel language to compile the Struphy kernels with.",
    )
    parser.add_argument(
        "--compiler",
        default="GNU",
        help='Pyccel compiler family ("GNU", "intel", "PGI", "nvidia" or "LLVM").',
    )
    args = parser.parse_args()

    # Paths relative to this script's location, so it can be run from anywhere.
    script_dir = Path(__file__).resolve().parent
    params_dir = script_dir / "@FOLDER@"

    profiling_case = ProfilingCase(
        label="@LABEL@",
        name="Poisson on @TITLE@ strong scaling test, @PRECOND_TITLE@",
        description="Strong scaling of the Poisson model with manufactured solution on @GEOMETRY_DESC@, solved with @SOLVE_DESC@.",
        physics_problem="Occurs in many plasma applications.",
        struphy_model_used="Poisson",
        params_source=params_dir / "params_poisson.py",
        language=args.language,
        compiler=args.compiler,
        upload=args.upload,
    )

    # Launch one run per rank count
    for num_tasks in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        profiling_case.launch(num_tasks)

    # Package and push each run as its own job finishes.
    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
'''


def render_submit(mapping_key, precond_key) -> str:
    m, p = MAPPINGS[mapping_key], PRECONDS[precond_key]
    s = SUBMIT_TEMPLATE
    for key, val in {
        "@TITLE@": m["title"],
        "@PRECOND_TITLE@": p["title"],
        "@FOLDER@": m["folder"] + p["suffix"],
        "@LABEL@": m["label"] + p["suffix"],
        "@GEOMETRY_DESC@": m["geometry_desc"],
        "@SOLVE_DESC@": p["solve_desc"],
    }.items():
        s = s.replace(key, val)
    assert "@" not in s, [line for line in s.splitlines() if "@" in line]
    return s


if __name__ == "__main__":
    written = []
    for mk, m in MAPPINGS.items():
        for pk, p in PRECONDS.items():
            folder = POISSON_DIR / (m["folder"] + p["suffix"])
            folder.mkdir(exist_ok=True)
            params = folder / "params_poisson.py"
            params.write_text(render_params(mk, pk, ERROR_LIMITS))
            submit = POISSON_DIR / f"submit_poisson_{folder.name}.py"
            submit.write_text(render_submit(mk, pk))
            written += [params, submit]
            print(f"written {params.relative_to(POISSON_DIR)} and {submit.name}")

    # the templates are not formatted, the generated files are
    ruff = shutil.which("ruff")
    if ruff is None:
        print("ruff not found, the generated files are not formatted")
    else:
        subprocess.run([ruff, "format", "-q", *map(str, written)], check=True)
