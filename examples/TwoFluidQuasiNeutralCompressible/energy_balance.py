from numpy import pi, zeros_like
from struphy.io.options import EnvironmentOptions, Time
from struphy.geometry import domains
from struphy.fields_background import equils
from struphy.topology import grids
from struphy.io.options import DerhamOptions
from struphy.initial.base import GenericPerturbation
from struphy import Simulation
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.two_fluid_quasi_neutral_compressible import TwoFluidQuasiNeutral

import numpy as np
import matplotlib.pyplot as plt
import os
from mpi4py import MPI

# ------------------ parameters ------------------
BC      = "dirichlet_inhom_natural"
name    = "runs/energy_balance_check"
N_STEPS = 5
DT      = 0.001

B0      = 0
nu      = 1.0
nu_e    = 1.0
mu      = 1.0
Nel     = (10, 10, 1)
p       = (2, 2, 1)
epsilon = 1.0
tol     = 1e-8

# ------------------ sim setup ------------------
env         = EnvironmentOptions(sim_folder=name)
time_opts   = Time(dt=DT, Tend=DT * N_STEPS)
domain      = domains.Cuboid()
equil       = equils.HomogenSlab(B0x=0, B0y=0, B0z=B0, beta=0, n0=0)
grid        = grids.TensorProductGrid(num_elements=Nel)
derham_opts = DerhamOptions(
    degree=p,
    bcs=(("dirichlet", "dirichlet"), ("dirichlet", "dirichlet"), None),
)

# ------------------ MMS ------------------
def mms_ion_u(x, y, z):
    return -np.sin(2*pi*x)*np.sin(2*pi*y), -np.sin(2*pi*x)*np.cos(2*pi*y), np.zeros_like(x)

def mms_electron_u(x, y, z):
    return -np.sin(2*pi*x)*np.sin(2*pi*y), -np.sin(2*pi*x)*np.cos(2*pi*y), np.zeros_like(x)

def source_function_u(x, y, z):
    fx = (
        -2*pi*np.sin(2*pi*x)
        - B0/epsilon * np.sin(2*pi*x)*np.cos(2*pi*y)
        - nu*8*pi**2 * np.sin(2*pi*x)*np.sin(2*pi*y)
    )
    fy = (
        2*pi*np.cos(2*pi*y)
        + B0/epsilon * np.sin(2*pi*x)*np.sin(2*pi*y)
        - nu*8*pi**2 * np.sin(2*pi*x)*np.cos(2*pi*y)
    )
    return fx, fy, zeros_like(x)

def source_function_ue(x, y, z):
    fx = (
        2*pi*np.sin(2*pi*x)
        + B0/epsilon * np.sin(2*pi*x)*np.cos(2*pi*y)
        - nu_e*8*pi**2 * np.sin(2*pi*x)*np.sin(2*pi*y)
    )
    fy = (
        -2*pi*np.cos(2*pi*y)
        - B0/epsilon * np.sin(2*pi*x)*np.sin(2*pi*y)
        - nu_e*8*pi**2 * np.sin(2*pi*x)*np.cos(2*pi*y)
    )
    return fx, fy, zeros_like(x)

lifting_function_u = [
    GenericPerturbation(lambda x, y, z: mms_ion_u(x, y, z)[0], comp=0, given_in_basis="physical"),
    GenericPerturbation(lambda x, y, z: mms_ion_u(x, y, z)[1], comp=1, given_in_basis="physical"),
]
lifting_function_ue = [
    GenericPerturbation(lambda x, y, z: mms_electron_u(x, y, z)[0], comp=0, given_in_basis="physical"),
    GenericPerturbation(lambda x, y, z: mms_electron_u(x, y, z)[1], comp=1, given_in_basis="physical"),
]

# ------------------ model ------------------
model = TwoFluidQuasiNeutral()

model.propagators.qn_comp.options = model.propagators.qn_comp.Options(
    nu=nu,
    nu_e=nu_e,
    mu=mu,
    eps_norm=epsilon,
    source_u=source_function_u,
    source_ue=source_function_ue,
    natural_u=lifting_function_u,
    natural_ue=lifting_function_ue,
    solver="gmres",
    solver_params=SolverParameters(info=True, tol=tol),
)

# ------------------ simulation ------------------
sim = Simulation(
    model=model,
    params_path=__file__,
    env=env,
    time_opts=time_opts,
    domain=domain,
    equil=equil,
    grid=grid,
    derham_opts=derham_opts,
)


def arr(v):
    """Extract plain numpy array from a Vector."""
    return v.toarray()


def matvec(A, v):
    """Apply operator A to Vector v, return numpy array."""
    return arr(A.dot(v))


def inner(a, b):
    """Inner product of two numpy arrays."""
    return float(a @ b)


def compute_terms(prop, model, dt, u_i_prev, u_e_prev):
    u_i = arr(model.ions.u.spline.vector)
    u_e = arr(model.electrons.u.spline.vector)

    M1 = prop._M1
    L  = prop._lapl_v0

    # --- energy ---
    W_new = 0.5 * inner(u_i, matvec(M1, model.ions.u.spline.vector))
    W_old = 0.5 * inner(u_i_prev, matvec(M1, prop._u_0.vector))
    dW_dt = (W_new - W_old) / dt

    # --- implicit Euler extra dissipation ---
    diff       = u_i - u_i_prev
    euler_diss = 0.5 / dt * inner(diff, matvec(M1, model.ions.u.spline.vector) - matvec(M1, prop._u_0.vector))

    # --- physical dissipation ---
    diss_i = nu        * inner(u_i, matvec(L, model.ions.u.spline.vector))
    diss_e = mu * nu_e * inner(u_e, matvec(L, model.electrons.u.spline.vector))

    LHS = dW_dt + euler_diss + diss_i + diss_e

    # --- sources ---
    src_i = (inner(arr(prop._src_u.vector),  matvec(prop._M1_u, model.ions.u.spline.vector))
             if prop._src_u is not None else 0.0)
    src_e = (inner(arr(prop._src_ue.vector), matvec(prop._M1_ue, model.electrons.u.spline.vector))
             if prop._src_ue is not None else 0.0)

    # --- boundary terms ---
    bnd_i = nu        * inner(u_i, matvec(M1, prop._grad.dot(prop._M0inv.dot(prop._boundary_normal_u.vector))))
    bnd_e = mu * nu_e * inner(u_e, matvec(M1, prop._grad.dot(prop._M0inv.dot(prop._boundary_normal_ue.vector))))

    RHS = src_i + src_e + bnd_i + bnd_e

    return dict(
        W_new=W_new, W_old=W_old,
        dW_dt=dW_dt, euler_diss=euler_diss,
        diss_i=diss_i, diss_e=diss_e,
        src_i=src_i, src_e=src_e,
        bnd_i=bnd_i, bnd_e=bnd_e,
        LHS=LHS, RHS=RHS,
        residual=LHS - RHS,
    )


def run_check():
    rank = MPI.COMM_WORLD.Get_rank()

    sim.allocate()

    prop    = model.propagators.qn_comp
    history = []

    for step in range(N_STEPS):
        # save u_i^n, u_e^n as numpy arrays before the propagator step
        u_i_prev = arr(model.ions.u.spline.vector)
        u_e_prev = arr(model.electrons.u.spline.vector)

        prop(DT)

        if rank == 0:
            terms = compute_terms(prop, model, DT, u_i_prev, u_e_prev)
            terms["step"] = step + 1
            terms["t"]    = (step + 1) * DT
            history.append(terms)

            print(f"\nStep {step+1}  (t={terms['t']:.4f})")
            print(f"  W_new      = {terms['W_new']:.6e}")
            print(f"  W_old      = {terms['W_old']:.6e}")
            print(f"  dW/dt      = {terms['dW_dt']:.6e}")
            print(f"  euler_diss = {terms['euler_diss']:.6e}")
            print(f"  diss_i     = {terms['diss_i']:.6e}")
            print(f"  diss_e     = {terms['diss_e']:.6e}")
            print(f"  src_i      = {terms['src_i']:.6e}")
            print(f"  src_e      = {terms['src_e']:.6e}")
            print(f"  bnd_i      = {terms['bnd_i']:.6e}")
            print(f"  bnd_e      = {terms['bnd_e']:.6e}")
            print(f"  LHS        = {terms['LHS']:.6e}")
            print(f"  RHS        = {terms['RHS']:.6e}")
            print(f"  residual   = {terms['residual']:.6e}  "
                  f"(rel: {terms['residual'] / max(abs(terms['LHS']), 1e-30):.2e})")

    if rank != 0:
        return

    steps    = [h["step"]     for h in history]
    LHS_vals = [h["LHS"]      for h in history]
    RHS_vals = [h["RHS"]      for h in history]
    res_vals = [h["residual"] for h in history]

    os.makedirs(f"{name}/plots", exist_ok=True)

    # plot 1: LHS vs RHS + residual
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(steps, LHS_vals, "o-", label="LHS")
    ax.plot(steps, RHS_vals, "s--", label="RHS")
    ax.set_xlabel("time step")
    ax.set_ylabel("energy balance")
    ax.set_title("LHS vs RHS")
    ax.legend()
    ax.grid(True)

    ax = axes[1]
    ax.plot(steps, res_vals, "o-", color="tab:red")
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel("time step")
    ax.set_ylabel("LHS - RHS")
    ax.set_title("Residual")
    ax.grid(True)

    fig.suptitle(f"Discrete energy balance  (dt={DT}, N={N_STEPS}, tol={tol})")
    plt.tight_layout()
    out = f"{name}/plots/energy_balance.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out}")

    # plot 2: individual terms
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, label in [
        ("dW_dt",      "dW/dt"),
        ("euler_diss", "Euler diss"),
        ("diss_i",     "nu_i ||u_i||_L"),
        ("diss_e",     "mu nu_e ||u_e||_L"),
        ("src_i",      "src ion"),
        ("src_e",      "src elec"),
        ("bnd_i",      "bnd ion"),
        ("bnd_e",      "bnd elec"),
    ]:
        ax.plot(steps, [h[key] for h in history], "o-", label=label)
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel("time step")
    ax.set_title("Individual terms")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True)
    plt.tight_layout()
    out2 = f"{name}/plots/energy_balance_terms.png"
    plt.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"Saved {out2}")


if __name__ == "__main__":
    run_check()