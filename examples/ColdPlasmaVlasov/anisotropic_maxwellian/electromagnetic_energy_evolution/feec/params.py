# -----------------------------
# Description of the simulation
# -----------------------------
# Thesis benchmark reproduction.
#
# -----------------------------

import logging
import numpy as np

from struphy import (
    BaseUnits,
    DerhamOptions,
    EnvironmentOptions,
    FieldsBackground,
    Simulation,
    Time,
    domains,
    equils,
    grids,
    perturbations,
    LoadingParameters,
    WeightsParameters,
    BoundaryParameters,
    SortingParameters,
    SavingParameters,
    BinningPlot,
    maxwellians,
    set_logging_level,
)

from struphy.models import ColdPlasmaVlasov

set_logging_level(logging.INFO)

name = "FEEC thesis Run 1 CV-on — Figures 4.1, 4.2 and 4.4"

description = '''
Run 1 anisotropy-driven instability benchmark.
FEEC ColdPlasmaVlasov, Lie-Trotter splitting, control variate enabled.
vth_parallel=0.20c, vth_perpendicular=0.53c, nu_h=0.06,
Omega_pe=2, |Omega_ce|=1, k=2, Lz=pi.
32 elements, degree 1, 100000 markers, dt=0.0125, Tend=200.
Initial perturbation Bx=1e-4 sin(2z).
'''



# Thesis Fig. 4.10: Lie-Trotter splitting with control variate.

# Physical normalisation: B_unit = 1 T, v_unit = c.
# These units derive alpha = -2, epsilon = -1 and kappa = 2.
model = ColdPlasmaVlasov(
    base_units=BaseUnits(
        x=1.7046507633e-3,
        B=1.0,
        n=0.3887618258,
    ),
)

model.em_fields.e_field.save_data = True
model.em_fields.b_field.save_data = True
model.em_fields.phi.save_data = True
model.electrons.current.save_data = True
model.hot_electrons.var.save_data = True

# ----------------
# Simulation
# ----------------

env = EnvironmentOptions(
    sim_folder="thesis_feec_run1_cv_final",
    save_step=1,
    max_runtime=1440,
)

time_opts = Time(
    dt=0.0125,
    Tend=200.0,
    split_algo="LieTrotter",
)

# k = 2, hence Lz = 2*pi/k = pi.
# The other two directions are inactive/minimal.

domain = domains.Cuboid(
    l1=0.0,
    r1=1.0,
    l2=0.0,
    r2=1.0,
    l3=0.0,
    r3=np.pi,
)

equil = equils.HomogenSlab()

grid = grids.TensorProductGrid(
    num_elements=(1, 1, 32),
)

derham_opts = DerhamOptions(
    degree=(1, 1, 1),
    bcs=(None, None, None),
)

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

# ----------------
# Particles
# ----------------

loading_params = LoadingParameters(
    Np=100000,
    loading="pseudo_random",
    seed=1234,
    moments=(0.0, 0.0, 0.0, 0.53, 0.53, 0.20),
)

weights_params = WeightsParameters(
    control_variate=True,
)

boundary_params = BoundaryParameters()
sorting_params = SortingParameters()

# No particle histogram required for Figures 4.1, 4.2 or 4.4.
saving_params = SavingParameters()

model.hot_electrons.set_markers(
    loading_params=loading_params,
    weights_params=weights_params,
    boundary_params=boundary_params,
    sorting_params=sorting_params,
    saving_params=saving_params,
)

# ----------------
# Propagators
# ----------------

model.propagators.maxwell.options = model.propagators.maxwell.Options()
model.propagators.ohm.options = model.propagators.ohm.Options()
model.propagators.jxb.options = model.propagators.jxb.Options()
model.propagators.push_eta.options = model.propagators.push_eta.Options()
model.propagators.push_vxb.options = model.propagators.push_vxb.Options()
model.propagators.coupling_va.options = model.propagators.coupling_va.Options()
model.initial_poisson.options = model.initial_poisson.Options()

# ----------------
# Initial conditions
# ----------------

# Cold current initially zero.
model.electrons.current.add_background(
    FieldsBackground(values=(0.0, 0.0, 0.0))
)

# Hot anisotropic Maxwellian:
# parallel direction is v3 (along B0 / z)
# perpendicular directions are v1 and v2.

hot_background = maxwellians.Maxwellian3D(
    n=(0.06, None),
    vth1=(0.53, None),
    vth2=(0.53, None),
    vth3=(0.20, None),
)

model.hot_electrons.var.add_background(hot_background)

model.hot_electrons.var.add_initial_condition(hot_background)
# Magnetic perturbation:
# B_x = 1e-4 sin(2 z)

magnetic_perturbation = perturbations.ModesSin(
    ls=(0,),
    ms=(0,),
    ns=(1,),
    amps=(1e-4,),
    given_in_basis="v",
    comp=0,
)

model.em_fields.b_field.add_perturbation(magnetic_perturbation)

if __name__ == "__main__":
    sim.run()
