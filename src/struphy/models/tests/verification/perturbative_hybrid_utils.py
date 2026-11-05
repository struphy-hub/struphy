"""Shared setup of the verification tests of the hybrid perturbative models (PerturbativeMHDWithQVlasovCC/PC):
finite-amplitude perturbations of all MHD variables in the Colella domain (homogeneous slab), energetic ions with a
shifted Maxwellian, tight solver tolerances."""

import cunumpy as xp
from maybempi import MPI

from struphy import (
    DerhamOptions,
    EnvironmentOptions,
    LoadingParameters,
    Simulation,
    Time,
    domains,
    equils,
    grids,
    maxwellians,
    perturbations,
)
from struphy.linear_algebra.solver import NonlinearSolverParameters, SolverParameters


def global_max(value: float) -> float:
    comm = MPI.COMM_WORLD
    if comm.Get_size() > 1:
        value = comm.allreduce(value, op=MPI.MAX)
    return value


def set_tight_options(model):
    """Tight tolerances of all linear and nonlinear solves, weights of the nonlinear terms at t^{n+1/2}
    (second order in time)."""
    tol = SolverParameters(tol=1e-13, maxiter=5000)
    nonlin = NonlinearSolverParameters(tol=1e-12)
    props = model.propagators
    for name in vars(props):
        prop = getattr(props, name)
        if not hasattr(prop, "Options"):
            continue
        fields = prop.Options.__dataclass_fields__
        kwargs = {}
        if "solver_params" in fields:
            kwargs["solver_params"] = tol
        if "nonlin_solver" in fields:
            kwargs["nonlin_solver"] = nonlin
        if "weights" in fields:
            kwargs["weights"] = "midpoint"
        if kwargs:
            prop.options = prop.Options(**kwargs)


def perturb_mhd(model, amp: float = 0.2):
    """Finite-amplitude perturbations of velocity, square root of the pressure, density and magnetic field."""
    m = model.mhd
    m.velocity.add_perturbation(
        perturbations.ModesSin(ls=(1,), ms=(1,), amps=(amp,), comp=0, given_in_basis="physical")
    )
    m.sqrt_p.add_perturbation(perturbations.ModesCos(ls=(1,), ms=(1,), amps=(amp,), given_in_basis="physical"))
    m.density.add_perturbation(perturbations.ModesCos(ms=(1,), amps=(amp,), given_in_basis="physical"))
    model.em_fields.b_field.add_perturbation(
        perturbations.ModesCos(ls=(1,), amps=(amp,), comp=1, given_in_basis="physical")
    )


def add_energetic_ions(model, n_h: float = 0.1):
    """Energetic ions with a Maxwellian of density ``n_h``, shifted in x (non-zero current)."""
    model.energetic_ions.set_markers(loading_params=LoadingParameters(Np=2000, seed=1234))
    model.energetic_ions.var.add_background(maxwellians.Maxwellian3D(n=(n_h, None), u1=(0.3, None), vth1=(0.5, None)))


def run(model, name: str, dt: float, Tend: float):
    """Allocate the simulation and integrate with Strang splitting; returns the FE coefficients of all MHD variables."""
    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders="verification_tests", sim_folder=name),
        time_opts=Time(dt=dt, Tend=Tend, split_algo="Strang"),
        domain=domains.Colella(Lx=1.0, Ly=1.0, alpha=0.1, Lz=1.0),
        equil=equils.HomogenSlab(B0x=0.3, B0z=1.0, beta=1.0, n0=1.0),
        grid=grids.TensorProductGrid(num_elements=(8, 8, 1)),
        derham_opts=DerhamOptions(degree=(2, 2, 1)),
    )
    sim.allocate()
    for _ in range(int(round(Tend / dt))):
        model.integrate(dt, split_algo="Strang")
    fields = (model.mhd.velocity, model.em_fields.b_field, model.mhd.sqrt_p, model.mhd.density)
    return xp.concatenate([f.spline.vector.toarray() for f in fields])


def max_diff(a, b) -> float:
    return global_max(float(xp.max(xp.abs(a - b))))
