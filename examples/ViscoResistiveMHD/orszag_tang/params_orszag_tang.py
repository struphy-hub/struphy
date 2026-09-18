# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation. 
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "Default ViscoResistiveMHD"
description = """
This is the default simulation for the model ViscoResistiveMHD. 
It is meant to be a template for users to set up their own simulations with this model. 
It contains all the necessary components of a Struphy simulation, including the model, 
the environment options, the time stepping options, the geometry, the equilibrium, 
the grid, the Derham options, and the initial conditions. 
Users can modify this file to set up their own simulations with different parameters and initial conditions.
"""

import copy
import cunumpy as xp
# from __future__ import annotations
import logging
import numpy as np
from struphy import set_logging_level
set_logging_level(logging.WARNING)



# ------------------
# Import Struphy API
# ------------------

from struphy import (
    BaseUnits,
    DerhamOptions,
    EnvironmentOptions,
    FieldsBackground,
    ProfilingOptions,
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

from struphy.fields_background.base import CartesianFluidEquilibriumWithB
from struphy.models import ViscoResistiveMHD

# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------


class OrszagTangInitialState(CartesianFluidEquilibriumWithB):
    r"""Smooth two-dimensional Orszag--Tang initial state.

    The fields on the periodic square are

    .. math::

        \rho &= \rho_0, \\
        p &= p_0, \\
        \mathbf u &=
        u_0(-\sin y,\,\sin x,\,0), \\
        \mathbf B &=
        B_0(-\sin y,\,\sin(2x),\,0).

    Both vector fields are initially divergence-free.
    """

    def __init__(
        self,
        rho0: float = 1.0,
        p0: float = 1.0,
        velocity_amplitude: float = 1.0,
        magnetic_amplitude: float = 1.0,
    ):
        self.params = copy.deepcopy(locals())

    def n_xyz(self, x, y, z):
        """Constant physical density."""
        return self.params["rho0"] + 0.0 * x

    def p_xyz(self, x, y, z):
        """Constant physical pressure."""
        return self.params["p0"] + 0.0 * x

    def u_xyz(self, x, y, z):
        """Cartesian velocity."""
        amplitude = self.params["velocity_amplitude"]

        return (
            -amplitude * xp.sin(y),
            amplitude * xp.sin(x),
            0.0 * z,
        )

    def b_xyz(self, x, y, z):
        """Cartesian magnetic field."""
        amplitude = self.params["magnetic_amplitude"]

        return (
            -amplitude * xp.sin(y),
            amplitude * xp.sin(2.0 * x),
            0.0 * z,
        )

    def gradB_xyz(self, x, y, z):
        """Cartesian gradient of the magnetic-field magnitude."""
        amplitude = self.params["magnetic_amplitude"]

        sin_y = xp.sin(y)
        sin_2x = xp.sin(2.0 * x)

        square = sin_y**2 + sin_2x**2
        denominator = xp.sqrt(square)

        # Avoid division by zero at magnetic nulls.
        safe_denominator = xp.where(
            denominator > 1.0e-14,
            denominator,
            1.0,
        )

        grad_x = amplitude * 2.0 * sin_2x * xp.cos(2.0 * x) / safe_denominator

        grad_y = amplitude * sin_y * xp.cos(y) / safe_denominator

        grad_x = xp.where(
            denominator > 1.0e-14,
            grad_x,
            0.0,
        )

        grad_y = xp.where(
            denominator > 1.0e-14,
            grad_y,
            0.0,
        )

        return (
            grad_x,
            grad_y,
            0.0 * z,
        )




# Units
base_units = BaseUnits()

# Model instance
model = ViscoResistiveMHD(base_units=base_units, with_viscosity=False, with_resistivity=False)

# List all variables and decide whether to save their data
model.em_fields.b_field.save_data = True
model.mhd.density.save_data = True
model.mhd.velocity.save_data = True
model.mhd.entropy.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
env = EnvironmentOptions()

# Time stepping
time_opts = Time(
    dt = 1.0e-3,
    Tend = 1.0
)

# Geometry
domain = domains.Cuboid(
    r1 = 2.0 * np.pi, 
    r2 = 2.0 * np.pi, 
    r3 = 1.0 , 
)

GAMMA = 5./3.
rho0 = GAMMA**2
p0 = GAMMA
initial_state = OrszagTangInitialState(
    rho0=float(rho0),
    p0=float(p0),
    velocity_amplitude=1.0,
    magnetic_amplitude=1.0,
)



# Grid
grid = grids.TensorProductGrid((32,32,1))

# Derham options
derham_opts = DerhamOptions((2,2,1))

# Profiling options
profiling_opts = ProfilingOptions()

# Simulation object
sim = Simulation(
    model=model,
    name=name,
    description=description,
    params_path=__file__,
    env=env,
    time_opts=time_opts,
    domain=domain,
    equil=initial_state,
    grid=grid,
    derham_opts=derham_opts,
    profiling_opts=profiling_opts,
)

# ------------------
# Propagator options
# ------------------

model.propagators.variat_dens.options = model.propagators.variat_dens.Options(model='full')
model.propagators.variat_mom.options = model.propagators.variat_mom.Options()
model.propagators.variat_ent.options = model.propagators.variat_ent.Options()
model.propagators.variat_mag.options = model.propagators.variat_mag.Options()


# ------------------
# Initial conditions
# ------------------
# Initial conditions are the sum of the background(s) and the perturbation(s).
# If backgrounds or perturbations are not specified, they are assumed to be zero.


# Full-f model: initialize all total fields.
model.mhd.density.add_background(
    FieldsBackground(
        type="FluidEquilibrium",
        variable="n3",
    ),
)
model.mhd.entropy.add_background(
    FieldsBackground(
        type="FluidEquilibrium",
        variable="s3_monoatomic",
    ),
)
model.mhd.velocity.add_background(
    FieldsBackground(
        type="FluidEquilibrium",
        variable="uv",
    ),
)
model.em_fields.b_field.add_background(
    FieldsBackground(
        type="FluidEquilibrium",
        variable="b2",
    ),
)

if __name__ == "__main__":
    sim.run(profiling_activated = True)