"""Verification of LinearEuler, the parent model of the LinearEuler family: sound wave dispersion (1D), a standing
sound wave in the curvilinear Colella domain against the exact solution (2D, with mesh refinement), conservation of
energy and mass for a non-uniform equilibrium density, and the convergence order of the time splitting."""

import logging
import os
import shutil

import cunumpy as xp
import numpy as np
import pytest
from maybempi import MPI

from struphy import (
    BaseUnits,
    DerhamOptions,
    EnvironmentOptions,
    Simulation,
    Time,
    domains,
    equils,
    grids,
    perturbations,
    set_logging_level,
)
from struphy.linear_algebra.solver import SolverParameters
from struphy.models import LinearEuler
from struphy.models.tests.verification.spectrum_utils import fit_branches, power_spectrum
from struphy.propagators.base import Propagator

set_logging_level()
logger = logging.getLogger("struphy")

GAMMA = 5 / 3
# equilibrium with sound speed c_s = sqrt(gamma p0 / rho0) = 1 (HomogenSlab: p0 = beta |B0|^2 / 2)
N0 = 1.0
P0 = 0.6
TOL = 1e-12


def _model(tol: float = None):
    """LinearEuler with optional tight tolerances of the sound wave solves."""
    model = LinearEuler(base_units=BaseUnits(kBT=1.0))
    if tol is not None:
        model.propagators.sound.options = model.propagators.sound.Options(
            solver_params=SolverParameters(tol=tol),
            solver_params_M3=SolverParameters(tol=tol),
        )
    return model


def _folder(name: str):
    return os.path.join(os.getcwd(), "verification_tests", "LinearEuler", name)


def _cleanup():
    MPI.COMM_WORLD.Barrier()
    if MPI.COMM_WORLD.Get_rank() == 0:
        shutil.rmtree(os.path.join(os.getcwd(), "verification_tests", "LinearEuler"), ignore_errors=True)


def _global_max(value: float) -> float:
    comm = MPI.COMM_WORLD
    if comm.Get_size() > 1:
        value = comm.allreduce(value, op=MPI.MAX)
    return value


def test_sound_wave_dispersion_1d():
    """Noise in the velocity along z: the space-time spectrum of the pressure has one branch, omega = c_s k."""
    model = _model()
    model.mhd.velocity.add_perturbation(perturbations.Noise(amp=0.1, comp=2, seed=123))

    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=_folder("dispersion_1d"), sim_folder="sim"),
        time_opts=Time(dt=0.15, Tend=90.0),
        domain=domains.Cuboid(r3=60.0),
        equil=equils.HomogenSlab(B0z=1.0, beta=2 * P0, n0=N0),
        grid=grids.TensorProductGrid(num_elements=(1, 1, 64)),
        derham_opts=DerhamOptions(degree=(1, 1, 3)),
    )
    run = sim.run().with_time_units("normalized")
    run.pproc()

    if MPI.COMM_WORLD.Get_rank() == 0:
        omega, k, power = power_spectrum(run.fields.mhd.pressure)
        fits = fit_branches(omega, k, power, n_branches=1, noise_level=0.4)
        c_s = xp.sqrt(GAMMA * P0 / N0)
        logger.info(f"{c_s = }, measured {fits[0][0]}")
        # the fit underestimates the speed by a few percent (numerical dispersion at the larger k of the fit)
        assert xp.abs(fits[0][0] - c_s) < 0.05 * c_s
    _cleanup()


def _standing_wave(num_elements: int, dt: float, Tend: float, name: str):
    """Standing sound wave p = A cos(2 pi x) cos(2 pi y) cos(omega t), u = 0 at t = 0, in the periodic Colella domain
    of unit size (the mapping distorts the grid, the exact solution is given in physical coordinates). Returns the
    output and the relative L2 error of the pressure at t = Tend (on the cell centres of a 32 x 32 grid)."""
    amp = 1e-2
    model = _model(tol=TOL)
    model.mhd.pressure.add_perturbation(
        perturbations.ModesCosCos(ls=(1,), ms=(1,), amps=(amp,), Lx=1.0, Ly=1.0, given_in_basis="physical")
    )

    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=_folder(name), sim_folder="sim"),
        time_opts=Time(dt=dt, Tend=Tend),
        domain=domains.Colella(Lx=1.0, Ly=1.0, alpha=0.1, Lz=1.0),
        equil=equils.HomogenSlab(B0z=1.0, beta=2 * P0, n0=N0),
        grid=grids.TensorProductGrid(num_elements=(num_elements, num_elements, 1)),
        derham_opts=DerhamOptions(degree=(3, 3, 1)),
    )
    run = sim.run().with_time_units("normalized")

    eta = (np.arange(32) + 0.5) / 32
    p = run.evaluate("mhd/pressure", t=-1, eta1=eta, eta2=eta, eta3=[0.5], representation="0").squeeze()
    x, y = p.X.to_numpy(), p.Y.to_numpy()
    t = float(run.time[-1])
    omega = xp.sqrt(GAMMA * P0 / N0) * 2 * np.pi * np.sqrt(2)
    exact = amp * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.cos(omega * t)
    error = np.sqrt(np.mean((p.to_numpy() - exact) ** 2)) / amp
    return run, error, omega


def test_standing_wave_colella():
    """Standing sound wave in the Colella domain: the pressure agrees with the exact solution after a period and its
    error decreases under mesh refinement; the kinetic energy oscillates as en_tot sin^2(omega t) (frequency of the
    eigenmode in the mapped geometry) and en_tot is conserved."""
    period = 1 / np.sqrt(2)
    errors = []
    for num_elements in (8, 16):
        run, error, omega = _standing_wave(num_elements, dt=period / 100, Tend=period, name=f"colella_{num_elements}")
        errors.append(error)
        logger.info(f"{num_elements = }: relative L2 error of the pressure after one period {error:.2e}")

    # energies of the finer run
    t = np.asarray(run.time)
    en_U = run.evaluate("scalars", variables="en_U")["en_U"].to_numpy()
    en_tot = run.evaluate("scalars", variables="en_tot")["en_tot"].to_numpy()
    energy_error = np.max(np.abs(en_U - en_tot[0] * np.sin(omega * t) ** 2)) / en_tot[0]
    conservation = np.max(np.abs(en_tot - en_tot[0])) / en_tot[0]
    logger.info(f"{errors = }, {energy_error = :.2e}, {conservation = :.2e}")

    assert errors[1] < 0.02
    assert errors[0] / errors[1] > 4.0
    assert energy_error < 0.02
    assert conservation < 1e-9
    _cleanup()


def test_conservation_nonuniform_density():
    """Noise in u and p in the Colella domain with a non-uniform equilibrium density rho0(x, y) and constant p0 (an
    equilibrium in force balance): the perturbed energy en_tot and the mass perturbation (sum of the density
    coefficients, periodic domain) are conserved, while energy is exchanged between en_U and en_thermal."""
    model = _model(tol=TOL)
    for comp in range(3):
        model.mhd.velocity.add_perturbation(
            perturbations.Noise(direction="e1e2", amp=1e-2, seed=1234 + comp, comp=comp, given_in_basis="2")
        )
    model.mhd.pressure.add_perturbation(
        perturbations.Noise(direction="e1e2", amp=1e-2, seed=4321, comp=None, given_in_basis="3")
    )

    equil = equils.GenericCartesianFluidEquilibrium(
        n_xyz=lambda x, y, z: 1.0 + 0.5 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y),
        p_xyz=lambda x, y, z: P0 * np.ones_like(x),
    )
    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=_folder("conservation"), sim_folder="sim"),
        time_opts=Time(dt=0.05, Tend=0.05),
        domain=domains.Colella(Lx=1.0, Ly=1.0, alpha=0.1, Lz=1.0),
        equil=equil,
        grid=grids.TensorProductGrid(num_elements=(12, 12, 1)),
        derham_opts=DerhamOptions(degree=(2, 2, 1)),
    )
    sim.allocate()

    mass_ops = Propagator.mass_ops
    u = model.mhd.velocity.spline.vector
    p = model.mhd.pressure.spline.vector
    n = model.mhd.density.spline.vector
    ones = n.space.zeros()
    ones[:] = 1.0

    def energies():
        kinetic = u.inner(mass_ops.M2n.dot(u)) / 2
        return kinetic, kinetic + p.inner(mass_ops.M3p_inv.dot(p)) / (2 * GAMMA)

    kinetic0, total0 = energies()
    totals, kinetics, masses = [], [], []
    for _ in range(20):
        for prop in model.prop_list:
            prop(0.05)
        kinetic, total = energies()
        totals.append(total)
        kinetics.append(kinetic)
        masses.append(n.inner(ones))

    conservation = np.max(np.abs(np.array(totals) - total0)) / total0
    exchange = np.max(np.abs(np.array(kinetics) - kinetic0)) / total0
    mass = np.max(np.abs(masses)) / _global_max(float(np.max(np.abs(n.toarray()))))
    logger.info(f"{conservation = :.2e}, {exchange = :.2e}, {mass = :.2e}")
    assert exchange > 1e-2
    assert conservation < 1e-10
    assert mass < 1e-10
    _cleanup()


def test_time_convergence_order():
    """Self-convergence in dt of the split time stepping, for a non-uniform equilibrium pressure such that the
    background pressure step does not commute with the sound wave step: the differences of the pressure between
    successive halvings of dt decrease with second order, with either splitting algorithm, since the two steps form one
    symmetric (Strang) block (``Propagators.strang_groups``; with Lie-Trotter sequencing of the two steps the order
    would be one, visible at this short end time).
    The time steps are taken with ``model.integrate`` (as in ``Simulation.run``), since the callable equilibrium can
    not be written to the run metadata."""
    Tend = 0.05
    finals = []
    for n_steps in (5, 10, 20, 40):
        dt = Tend / n_steps
        model = _model(tol=TOL)
        model.mhd.pressure.add_perturbation(
            perturbations.ModesCosCos(ls=(1,), ms=(1,), amps=(1e-2,), Lx=1.0, Ly=1.0, given_in_basis="physical")
        )
        equil = equils.GenericCartesianFluidEquilibrium(
            n_xyz=lambda x, y, z: np.ones_like(x),
            p_xyz=lambda x, y, z: P0 * (1.0 + 0.8 * np.sin(2 * np.pi * x)),
        )
        sim = Simulation(
            model=model,
            env=EnvironmentOptions(out_folders=_folder("order"), sim_folder=f"steps_{n_steps}"),
            time_opts=Time(dt=dt, Tend=Tend),
            domain=domains.Colella(Lx=1.0, Ly=1.0, alpha=0.1, Lz=1.0),
            equil=equil,
            grid=grids.TensorProductGrid(num_elements=(8, 8, 1)),
            derham_opts=DerhamOptions(degree=(2, 2, 1)),
        )
        sim.allocate()
        for _ in range(n_steps):
            model.integrate(dt)
        finals.append(model.mhd.pressure.spline.vector.toarray().copy())

    diffs = [_global_max(float(np.max(np.abs(a - b)))) for a, b in zip(finals[:-1], finals[1:])]
    rates = [np.log2(d0 / d1) for d0, d1 in zip(diffs[:-1], diffs[1:])]
    logger.info(f"{diffs = }, {rates = }")
    assert all(abs(rate - 2) < 0.1 for rate in rates)
    _cleanup()


if __name__ == "__main__":
    test_sound_wave_dispersion_1d()
    test_standing_wave_colella()
    test_conservation_nonuniform_density()
    test_time_convergence_order()
