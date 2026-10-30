"""Orszag-Tang vortex with the nonlinear variational ideal MHD model (VariationalMHD without viscosity and
resistivity): conservation of mass, entropy, energy and div B in a nonlinear 2D flow."""

# -----------------------------
# Description of the simulation
# -----------------------------

name = "Orszag-Tang vortex (VariationalMHD)"
description = """
Orszag-Tang vortex, a standard nonlinear test of compressible MHD codes, computed with the variational MHD model
VariationalMHD (density, entropy, velocity and magnetic field), here without viscosity and resistivity,
i.e. ideal MHD.

Physics. On the periodic square [0, 2 pi]^2 the initial state is
    rho = gamma^2,   p = gamma,   u = (-sin y, sin x, 0),   B = (-sin y, sin 2x, 0),
with gamma = 5/3 (sound speed 1, Mach number 1). Both u and B are divergence free. The vortices of u and B
interact nonlinearly; kinetic and magnetic energy are exchanged with the internal energy, and steep gradients
(later shocks and current sheets) form. The variational discretization conserves the total mass and entropy
exactly, the total energy up to the tolerance of the nonlinear solvers, and keeps div B = 0 to machine precision.

Setup. Cuboid [0, 2 pi]^2 x [0, 1], 32 x 32 x 1 elements with degrees (2, 2, 1), periodic. The initial state is
an OrszagTangInitialState (defined below), used as fluid equilibrium: all variables are initialized from it.
Time step dt = 1e-3 up to Tend = 1 (1000 steps), before shocks form, with the default (Newton/Picard) options of
the variational propagators.

Expected result. The pproc plots density, pressure and |B| at the first, middle and last saved time, and the
energies against time. Kinetic energy is converted into internal and magnetic energy (en_U drops from 55 to 40
up to t = 1) and pressure ridges form along the stagnation lines. Total mass and entropy are conserved to
round-off (relative drift 2e-14), the total energy to about the nonlinear solver tolerance (relative drift 3e-9),
and div B stays at round-off (1e-27). The run takes about 10 minutes on one process.

References:
- S. A. Orszag and C.-M. Tang, "Small-scale structure of two-dimensional magnetohydrodynamic turbulence",
  J. Fluid Mech. 90, 129 (1979).
"""

import argparse
from pathlib import Path

# Command line options
# `--out-folder` and `--sim-folder` set the output folder; `--num-elements` the grid in x and y.
# `--test-run` selects a small grid and a short run (three saved time steps) for the tests (test_examples.py).
# Unknown flags are ignored so that a driver can forward other parameters as well.
# no -h here when imported by the pproc script, which has its own help
parser = argparse.ArgumentParser(add_help=__name__ == "__main__")
parser.add_argument(
    "--out-folder",
    default=str(Path(__file__).resolve().parent),
    help="Folder in which the output folder is created (default: beside this file).",
)
parser.add_argument("--sim-folder", default="sim_1", help="Name of the output folder.")
parser.add_argument("--num-elements", type=int, nargs=2, default=[32, 32], help="Number of elements in x and y.")
parser.add_argument(
    "--test-run",
    action="store_true",
    help="Small resolution (8 x 8 elements) and three saved time steps, for the tests (test_examples.py).",
)
args, _ = parser.parse_known_args()
if args.test_run:
    args.num_elements = [8, 8]

import copy
import logging

from struphy import set_logging_level

set_logging_level(logging.INFO)

import cunumpy as xp

from struphy import (
    BaseUnits,
    DerhamOptions,
    EnvironmentOptions,
    FieldsBackground,
    Simulation,
    Time,
    domains,
    grids,
)
from struphy.fields_background.base import CartesianFluidEquilibriumWithB
from struphy.models import VariationalMHD

# ---------------------
# Initial state
# ---------------------


class OrszagTangInitialState(CartesianFluidEquilibriumWithB):
    r"""Orszag-Tang initial state on the periodic square :math:`[0, 2\pi]^2`:

    .. math::

        \rho = \rho_0\,,\quad p = p_0\,,\quad
        \mathbf u = u_0(-\sin y,\, \sin x,\, 0)\,,\quad
        \mathbf B = B_0(-\sin y,\, \sin 2x,\, 0)\,.

    Parameters
    ----------
    rho0, p0 : float
        Constant density and pressure.

    u0, B0 : float
        Amplitudes of velocity and magnetic field.
    """

    def __init__(self, rho0: float = 1.0, p0: float = 1.0, u0: float = 1.0, B0: float = 1.0):
        self.params = copy.deepcopy(locals())

    def n_xyz(self, x, y, z):
        return self.params["rho0"] + 0.0 * x

    def p_xyz(self, x, y, z):
        return self.params["p0"] + 0.0 * x

    def u_xyz(self, x, y, z):
        u0 = self.params["u0"]
        return -u0 * xp.sin(y), u0 * xp.sin(x), 0.0 * z

    def b_xyz(self, x, y, z):
        B0 = self.params["B0"]
        return -B0 * xp.sin(y), B0 * xp.sin(2.0 * x), 0.0 * z

    def gradB_xyz(self, x, y, z):
        """Gradient of |B| (set to zero at the magnetic nulls)."""
        B0 = self.params["B0"]
        absB = xp.sqrt(xp.sin(y) ** 2 + xp.sin(2.0 * x) ** 2)
        nonzero = absB > 1e-14
        safe = xp.where(nonzero, absB, 1.0)
        grad_x = xp.where(nonzero, B0 * 2.0 * xp.sin(2.0 * x) * xp.cos(2.0 * x) / safe, 0.0)
        grad_y = xp.where(nonzero, B0 * xp.sin(y) * xp.cos(y) / safe, 0.0)
        return grad_x, grad_y, 0.0 * z


# ---------------------
# Physical parameters (also used by the pproc)
# ---------------------

gamma = 5.0 / 3.0

# ---------------------
# Instance of the model
# ---------------------

base_units = BaseUnits()
model = VariationalMHD(base_units=base_units, with_viscosity=False, with_resistivity=False)

model.em_fields.b_field.save_data = True
model.mhd.density.save_data = True
model.mhd.velocity.save_data = True
model.mhd.entropy.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

env = EnvironmentOptions(out_folders=args.out_folder, sim_folder=args.sim_folder, restart=False)

# --test-run: the initial, a middle and a final saved time step
dt = 1e-3
time_opts = Time(dt=dt, Tend=2 * dt if args.test_run else 1.0)

domain = domains.Cuboid(r1=2.0 * xp.pi, r2=2.0 * xp.pi, r3=1.0)

equil = OrszagTangInitialState(rho0=gamma**2, p0=gamma, u0=1.0, B0=1.0)

grid = grids.TensorProductGrid(num_elements=(*args.num_elements, 1))

derham_opts = DerhamOptions(degree=(2, 2, 1))

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
)

# ------------------
# Propagator options
# ------------------

model.propagators.variat_dens.options = model.propagators.variat_dens.Options(model="full")
model.propagators.variat_mom.options = model.propagators.variat_mom.Options()
model.propagators.variat_ent.options = model.propagators.variat_ent.Options()
model.propagators.variat_mag.options = model.propagators.variat_mag.Options()

# ------------------
# Initial conditions
# ------------------

# full-f model: all variables are initialized from the initial state
model.mhd.density.add_background(FieldsBackground(type="FluidEquilibrium", variable="n3"))
model.mhd.entropy.add_background(FieldsBackground(type="FluidEquilibrium", variable="s3_monoatomic"))
model.mhd.velocity.add_background(FieldsBackground(type="FluidEquilibrium", variable="uv"))
model.em_fields.b_field.add_background(FieldsBackground(type="FluidEquilibrium", variable="b2"))

if __name__ == "__main__":
    sim.run()
