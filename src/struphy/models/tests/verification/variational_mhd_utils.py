"""Helpers for the verification tests of the variational (linear, perturbative and full-f) MHD models with the
equilibria of a sheared and a homogeneous slab: allocation of the ideal models, initial perturbations, energy and
coefficient diagnostics, and the frequency of a standing wave."""

import cunumpy as xp
from maybempi import MPI

from struphy import (
    DerhamOptions,
    EnvironmentOptions,
    FieldsBackground,
    Simulation,
    Time,
    domains,
    equils,
    grids,
    perturbations,
)
from struphy.linear_algebra.solver import NonlinearSolverParameters, SolverParameters
from struphy.models import (
    ViscoResistiveLinearPerturbativeMHD,
    ViscoResistiveLinearPerturbativeMHD_with_q,
    ViscoResistiveMHD_with_q,
    ViscoResistivePerturbativeMHD,
    ViscoResistivePerturbativeMHD_with_q,
)
from struphy.propagators.base import Propagator

GAMMA = 5.0 / 3.0

# default solver tolerances (the nonlinear iterations stop at a residual of 1e-8); with much smaller tolerances the
# Jacobian solves break down once the iteration has converged
SOLVER = SolverParameters()
NONLIN = NonlinearSolverParameters()

# variational propagator option `model` for each model class
VARIANTS = {
    ViscoResistivePerturbativeMHD_with_q: "perturbative_q",
    ViscoResistiveMHD_with_q: "full_q",
    ViscoResistiveLinearPerturbativeMHD_with_q: "linear_q",
    ViscoResistivePerturbativeMHD: "perturbative",
    ViscoResistiveLinearPerturbativeMHD: "linear",
}


def sheared_slab(num_elements=16):
    """Non-uniform equilibrium, periodic in x: B = B0 (e_z + a / (q(x) R0) e_y) with q(x) = q0 + q1 sin(2 pi x / a)
    and the pressure in force balance."""
    equil = equils.ShearedSlab(a=1.0, R0=3.0, B0=1.0, q0=1.05, q1=0.5, beta=1.0, q_kind=1)
    domain = domains.Cuboid(r1=1.0, r2=2 * xp.pi, r3=2 * xp.pi * 3.0)
    return equil, domain, (num_elements, 1, 1), (3, 1, 1)


def homogeneous_slab(B0=(0.0, 1.0, 1.0), Lz=10.0, num_elements=16):
    """Homogeneous equilibrium (beta = 1, n0 = 0.7), waves along z."""
    equil = equils.HomogenSlab(B0x=B0[0], B0y=B0[1], B0z=B0[2], beta=1.0, n0=0.7)
    domain = domains.Cuboid(r3=Lz)
    return equil, domain, (1, 1, num_elements), (1, 1, 3)


def allocate_ideal(tmp_path, name, model_class, setup, dt, perturb=None):
    """Allocate an ideal model of `model_class` for the `setup` (equil, domain, num_elements, degree).

    `perturb(model)` adds the perturbations of the perturbative variables. The full-f model gets the equilibrium
    as background in addition. Models must be allocated and run one after the other, since they share the class
    attributes of Propagator (derham, mass_ops, ...)."""
    equil, domain, num_elements, degree = setup
    model = model_class(with_viscosity=False, with_resistivity=False)
    variant = VARIANTS[model_class]

    model.propagators.variat_dens.options = model.propagators.variat_dens.Options(
        model=variant, solver_params=SOLVER, nonlin_solver=NONLIN
    )
    if hasattr(model.propagators, "variat_mom"):
        model.propagators.variat_mom.options = model.propagators.variat_mom.Options(
            solver_params=SOLVER, nonlin_solver=NONLIN
        )
    if hasattr(model.propagators, "variat_pb"):
        model.propagators.variat_pb.options = model.propagators.variat_pb.Options(
            model=variant, solver_params=SOLVER, nonlin_solver=NONLIN
        )
    else:
        model.propagators.variat_qb.options = model.propagators.variat_qb.Options(
            model=variant, solver_params=SOLVER, nonlin_solver=NONLIN
        )

    if variant == "full_q":
        model.mhd.density.add_background(FieldsBackground(type="FluidEquilibrium", variable="n3"))
        model.mhd.sqrt_p.add_background(FieldsBackground(type="FluidEquilibrium", variable="q3"))
        model.em_fields.b_field.add_background(FieldsBackground(type="FluidEquilibrium", variable="b2"))

    if perturb is not None:
        perturb(model)

    # the same output folder on all ranks (pytest gives each rank its own tmp_path)
    comm = MPI.COMM_WORLD
    out_folder = comm.bcast(str(tmp_path) if comm.Get_rank() == 0 else None)

    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=out_folder, sim_folder=name),
        time_opts=Time(dt=dt, Tend=dt),
        domain=domain,
        equil=equil,
        grid=grids.TensorProductGrid(num_elements=num_elements),
        derham_opts=DerhamOptions(degree=degree),
    )
    sim.allocate()
    return model


def state(model):
    """Local coefficient arrays of (rho, u, q or p, B), comparable between different simulations."""
    thermo = model.mhd.sqrt_p if hasattr(model.mhd, "sqrt_p") else model.mhd.pressure
    return [
        var.spline.vector.toarray() for var in (model.mhd.density, model.mhd.velocity, thermo, model.em_fields.b_field)
    ]


def norm(arr):
    """Euclidean norm of a coefficient array distributed over the MPI ranks."""
    return float(xp.sqrt(MPI.COMM_WORLD.allreduce(float(xp.sum(arr**2)), op=MPI.SUM)))


def perturb_velocity(amp, direction="x", L=1.0):
    """Smooth perturbation of all (physical) velocity components, varying along x or z (length L of the domain)."""
    modes = "ls" if direction == "x" else "ns"
    lengths = {"Lx": L} if direction == "x" else {"Lz": L}

    def perturb(model):
        for comp, (mode, phase) in enumerate(((1, 0.0), (1, 1.0), (2, 2.0))):
            model.mhd.velocity.add_perturbation(
                perturbations.ModesSin(
                    **{modes: (mode,)}, amps=(amp,), theta=(phase,), comp=comp, given_in_basis="physical", **lengths
                )
            )

    return perturb


def perturb_all_along_z(amp, Lz):
    """Perturbation of u, q and B along z (B~_x(z), B~_y(z) keep div B = 0)."""

    def perturb(model):
        perturb_velocity(amp, direction="z", L=Lz)(model)
        model.mhd.sqrt_p.add_perturbation(perturbations.ModesCos(ns=(2,), amps=(amp,), given_in_basis="3"))
        model.em_fields.b_field.add_perturbation(
            perturbations.ModesSin(ns=(1,), amps=(amp,), comp=0, given_in_basis="2")
        )
        model.em_fields.b_field.add_perturbation(
            perturbations.ModesCos(ns=(2,), amps=(amp,), comp=1, given_in_basis="2")
        )

    return perturb


def energy_history(model, dt, n_steps):
    """The total energy en_tot before and after each of n_steps time steps."""
    energies = []
    for _ in range(n_steps):
        model.update_scalar_quantities()
        energies.append(float(model.scalars.host_value("en_tot")[0]))
        model.integrate(dt)
    model.update_scalar_quantities()
    energies.append(float(model.scalars.host_value("en_tot")[0]))
    return energies


def standing_wave(tmp_path, model_class, setup, Lz, comp, amp, dt, n_steps):
    """Run a standing wave u_comp = amp sin(2 pi z / Lz) for n_steps steps (should cover more than a quarter period).

    Returns the final velocity coefficients and the frequency from the first zero of the projection of u(t) on u(0),
    which is cos(omega t) for a single wave."""
    model = allocate_ideal(
        tmp_path,
        f"{model_class.__name__}_{amp}",
        model_class,
        setup,
        dt,
        lambda m: m.mhd.velocity.add_perturbation(
            perturbations.ModesSin(ns=(1,), amps=(amp,), comp=comp, Lz=Lz, given_in_basis="physical")
        ),
    )
    Mv = Propagator.mass_ops.Mv
    u0 = model.mhd.velocity.spline.vector.copy()
    norm0 = Mv.dot_inner(u0, u0)
    proj = [1.0]
    for _ in range(n_steps):
        model.integrate(dt)
        proj.append(Mv.dot_inner(model.mhd.velocity.spline.vector, u0) / norm0)
    proj = xp.array(proj)
    i = int(xp.argmax(proj < 0.0))
    assert i > 0, f"no zero crossing of the projection in {n_steps} steps: {proj}"
    t_zero = dt * (i - 1 + proj[i - 1] / (proj[i - 1] - proj[i]))
    return model.mhd.velocity.spline.vector.toarray(), xp.pi / 2 / t_zero
