#!/usr/bin/env python3
"""Current-FEEC Figure 4.7 particle-noise run.

Uses the thesis physical parameters and zero initial EM fields. Full-f marker
weights are retained so finite-marker current fluctuations seed the fields.
The local post_allocate override also neutralizes the previously added
unconditional update_weights() debugging call without editing StruPy source.
"""

import logging
from pathlib import Path

from struphy import (
    BaseUnits, DerhamOptions, EnvironmentOptions, FieldsBackground,
    LoadingParameters, WeightsParameters, BoundaryParameters,
    SortingParameters, SavingParameters, Simulation, Time,
    domains, equils, grids, maxwellians, set_logging_level,
)
from struphy.models import ColdPlasmaVlasov

set_logging_level(logging.INFO)

# Thesis Run 3: |Omega_ce|=1, Omega_pe=2, nu_h=0.002.
DT = 0.05
HALF_STEP = -0.5 * DT


class Figure47ColdPlasmaVlasov(ColdPlasmaVlasov):
    """Apply the historical backward half-Boris initialization externally."""

    def post_allocate(self):
        # Perform StruPy's normal initial Poisson solve and delta-f weight
        # restoration first. This is the unmodified model implementation.
        super().post_allocate()

        # Keep the historical leapfrog/Boris half-step staggering.
        self.propagators.push_vxb(HALF_STEP)

        # This production run is full-f (control variate OFF). Restore the
        # initialized marker weights after the inherited allocation hook.
        # This is intentionally local to this Figure-4.7 subclass: no shared
        # StruPhy model source is edited.
        particles = self.hot_electrons.var.particles
        particles.weights = particles.weights0


name = "Figure 4.7 — current FEEC, full-f particle-noise run"
description = """
Current-FEEC Figure 4.7 run using the historical Run-3 physical parameters,
zero imposed electromagnetic perturbation, and finite-marker full-f current
fluctuations. The backward half-step preserves particle time staggering.
Control variate is disabled because the matched isotropic Maxwellian residual
in the historical CV-on setup cancels to zero for uniform B0. The original
full-f marker weights are restored locally after model allocation.
"""

model = Figure47ColdPlasmaVlasov(
    base_units=BaseUnits(x=1.7046507633e-3, B=1.0, n=0.3887618258),
)
model.em_fields.e_field.save_data = True
model.em_fields.b_field.save_data = True
model.em_fields.phi.save_data = True
model.electrons.current.save_data = True
model.hot_electrons.var.save_data = True

# Domain and grid: 1-D variation along z; FEEC transverse spaces remain
# non-degenerate. Cubic order is retained in the physical z direction.
env = EnvironmentOptions(
    sim_folder="thesis_feec_fig4_7_current_feec_fullf_noise",
    save_step=1,
    max_runtime=1440,
)
time_opts = Time(dt=DT, Tend=300.0, split_algo="LieTrotter")
domain = domains.Cuboid(l1=0.0, r1=1.0, l2=0.0, r2=1.0, l3=0.0, r3=80.0)
equil = equils.HomogenSlab()
grid = grids.TensorProductGrid(num_elements=(1, 1, 256))
derham_opts = DerhamOptions(degree=(1, 1, 3), bcs=(None, None, None))

sim = Simulation(
    model=model, name=name, description=description, params_path=__file__,
    env=env, time_opts=time_opts, domain=domain, equil=equil,
    grid=grid, derham_opts=derham_opts,
)

loading_params = LoadingParameters(
    Np=50000, loading="pseudo_random", seed=1234,
    moments=(0.0, 0.0, 0.0, 0.1, 0.1, 0.1),
)
weights_params = WeightsParameters(control_variate=False)
boundary_params = BoundaryParameters()
sorting_params = SortingParameters()
saving_params = SavingParameters(n_markers=3)
model.hot_electrons.set_markers(
    loading_params=loading_params, weights_params=weights_params,
    boundary_params=boundary_params, sorting_params=sorting_params,
    saving_params=saving_params,
)

model.propagators.maxwell.options = model.propagators.maxwell.Options()
model.propagators.ohm.options = model.propagators.ohm.Options()
model.propagators.jxb.options = model.propagators.jxb.Options()
model.propagators.push_eta.options = model.propagators.push_eta.Options()
model.propagators.push_vxb.options = model.propagators.push_vxb.Options(algo="analytic")
model.propagators.coupling_va.options = model.propagators.coupling_va.Options()
model.initial_poisson.options = model.initial_poisson.Options()

model.electrons.current.add_background(FieldsBackground(values=(0.0, 0.0, 0.0)))
hot_maxwellian = maxwellians.Maxwellian3D(
    n=(0.002, None),
    vth1=(0.1, None), vth2=(0.1, None), vth3=(0.1, None),
)
model.hot_electrons.var.add_background(hot_maxwellian)
model.hot_electrons.var.add_initial_condition(hot_maxwellian)

if __name__ == "__main__":
    sim.run()
