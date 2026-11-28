
#!/usr/bin/env python3
"""FEEC counterpart of thesis Figures 4.5 and 4.6 (Run 2).

Run from the StruPhy repository root:
    python examples/ColdPlasmaVlasov/anisotropic_maxwellian/wavenumber_scan/feec/params.py

Runs the seven wavenumbers sequentially, each in a separate simulation folder.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from struphy import (
    BaseUnits,
    BoundaryParameters,
    DerhamOptions,
    EnvironmentOptions,
    FieldsBackground,
    LoadingParameters,
    SavingParameters,
    Simulation,
    SortingParameters,
    Time,
    WeightsParameters,
    domains,
    equils,
    grids,
    maxwellians,
    perturbations,
    set_logging_level,
)
from struphy.models import ColdPlasmaVlasov


set_logging_level(logging.INFO)

# Historical thesis Run 2 parameters
K_VALUES = np.arange(1.50, 3.0001, 0.25)
RUN_PREFIX = "thesis_feec_run3"

OMEGA_CE_ABS = 1.0
OMEGA_PE = 2.0
NU_H = 0.05

VTH_PARALLEL = 0.20
VTH_PERPENDICULAR = 0.60

NUM_ELEMENTS_Z = 200
POLYNOMIAL_DEGREE_Z = 3
NUM_PARTICLES = 50_000

DT = 0.05
T_END = 400.0
CONTROL_VARIATE = True
RANDOM_SEED = 1234
INITIAL_BX_AMPLITUDE = 1.0e-4

# Same BaseUnits as the established FEEC thesis examples.
BASE_UNITS = dict(
    x=1.7046507633e-3,
    B=1.0,
    n=0.3887618258,
)

# With these BaseUnits, alpha_hot = -2.
# FEEC input density 0.05 therefore gives physical density 0.20.
ALPHA_HOT = -2.0
N_HOT_PHYSICAL = ALPHA_HOT**2 * NU_H
N_HOT_THESIS = NU_H * OMEGA_PE**2

if not np.isclose(N_HOT_PHYSICAL, N_HOT_THESIS, atol=1e-14):
    raise RuntimeError(
        f"Hot-density mismatch: FEEC={N_HOT_PHYSICAL}, "
        f"thesis={N_HOT_THESIS}"
    )


def make_and_run(k: float) -> None:
    """Construct and run one independent FEEC wavenumber case."""

    length_z = 2.0 * np.pi / k
    tag = f"{k:.2f}".replace(".", "p")
    run_name = f"{RUN_PREFIX}_k{tag}"

    # Never overwrite an existing simulation directory.
    run_path = Path.cwd() / run_name
    if run_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing run directory:\n{run_path}\n"
            "Archive or rename it before running this case."
        )

    model = ColdPlasmaVlasov(
        base_units=BaseUnits(**BASE_UNITS),
    )

    # Couple the evolved FEEC magnetic perturbation into the hot-electron
    # Lorentz-force pusher. This is experiment-level configuration.
    model.propagators.push_vxb.b2_var = model.em_fields.b_field
    assert model.propagators.push_vxb.b2_var is model.em_fields.b_field

    # Required field coefficients and scalar histories.
    model.em_fields.e_field.save_data = True
    model.em_fields.b_field.save_data = True
    model.em_fields.phi.save_data = True
    model.electrons.current.save_data = True
    model.hot_electrons.var.save_data = True

    name = f"FEEC thesis Run 3 — k={k:.2f}"
    description = f"""
Current-FEEC counterpart of thesis Run 2, Figures 4.5 and 4.6.
k={k:.2f}, Lz=2*pi/k={length_z:.12g}.
vth_parallel={VTH_PARALLEL}c, vth_perpendicular={VTH_PERPENDICULAR}c.
nu_h={NU_H}, Omega_pe={OMEGA_PE}, |Omega_ce|={OMEGA_CE_ABS}.
200 elements in z, degree 3, {NUM_PARTICLES} markers.
dt={DT}, Tend={T_END}, control variate on.
Initial Bx={INITIAL_BX_AMPLITUDE} sin(k z).
FEEC input density={NU_H}; alpha_hot^2*n={N_HOT_PHYSICAL},
matching thesis physical hot density={N_HOT_THESIS}.
"""

    env = EnvironmentOptions(
        sim_folder=run_name,
        save_step=1,
        max_runtime=1440,
    )

    time_opts = Time(
        dt=DT,
        Tend=T_END,
        split_algo="LieTrotter",
    )

    domain = domains.Cuboid(
        l1=0.0, r1=1.0,
        l2=0.0, r2=1.0,
        l3=0.0, r3=length_z,
    )

    equil = equils.HomogenSlab()

    grid = grids.TensorProductGrid(
        num_elements=(1, 1, NUM_ELEMENTS_Z),
    )

    derham_opts = DerhamOptions(
        degree=(1, 1, POLYNOMIAL_DEGREE_Z),
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

    loading_params = LoadingParameters(
        Np=NUM_PARTICLES,
        loading="pseudo_random",
        seed=RANDOM_SEED,
        moments=(
            0.0, 0.0, 0.0,
            VTH_PERPENDICULAR,
            VTH_PERPENDICULAR,
            VTH_PARALLEL,
        ),
    )

    weights_params = WeightsParameters(
        control_variate=CONTROL_VARIATE,
    )

    boundary_params = BoundaryParameters()
    sorting_params = SortingParameters()
    saving_params = SavingParameters()

    model.hot_electrons.set_markers(
        loading_params=loading_params,
        weights_params=weights_params,
        boundary_params=boundary_params,
        sorting_params=sorting_params,
        saving_params=saving_params,
    )

    # Match the established FEEC Run 1 propagator configuration.
    # Do not transfer the Figure 4.7-specific half-step override to Run 2.
    model.propagators.maxwell.options = model.propagators.maxwell.Options()
    model.propagators.ohm.options = model.propagators.ohm.Options()
    model.propagators.jxb.options = model.propagators.jxb.Options()
    model.propagators.push_eta.options = model.propagators.push_eta.Options()
    model.propagators.push_vxb.options = model.propagators.push_vxb.Options()
    model.propagators.coupling_va.options = (
        model.propagators.coupling_va.Options()
    )
    model.initial_poisson.options = model.initial_poisson.Options()

    model.electrons.current.add_background(
        FieldsBackground(values=(0.0, 0.0, 0.0))
    )

    hot_background = maxwellians.Maxwellian3D(
        n=(NU_H, None),
        vth1=(VTH_PERPENDICULAR, None),
        vth2=(VTH_PERPENDICULAR, None),
        vth3=(VTH_PARALLEL, None),
    )

    model.hot_electrons.var.add_background(hot_background)
    model.hot_electrons.var.add_initial_condition(hot_background)

    # Fundamental mode on Lz=2*pi/k gives physical wavenumber k.
    # comp=0 selects Bx, matching the historical magnetic seed.
    magnetic_perturbation = perturbations.ModesSin(
        ls=(0,),
        ms=(0,),
        ns=(1,),
        amps=(INITIAL_BX_AMPLITUDE,),
        given_in_basis="v",
        comp=0,
    )

    model.em_fields.b_field.add_perturbation(magnetic_perturbation)

    print("\n" + "=" * 72)
    print(f"FEEC THESIS RUN 3 — k={k:.2f}")
    print(f"Lz={length_z:.12g}")
    print(f"Elements={NUM_ELEMENTS_Z}, degree={POLYNOMIAL_DEGREE_Z}")
    print(f"Np={NUM_PARTICLES}, dt={DT}, Tend={T_END}")
    print(f"Control variate={CONTROL_VARIATE}")
    print(f"FEEC n={NU_H}; physical n={N_HOT_PHYSICAL}")
    print(f"Output: {run_path}")
    print("=" * 72)

    sim.run()


if __name__ == "__main__":
    # Check every destination before starting any of the seven simulations.
    destinations = [
        Path.cwd() / f"{RUN_PREFIX}_k{float(k):.2f}".replace(".", "p")
        for k in K_VALUES
    ]
    existing = [str(path) for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(
            "Run 3 preflight failed; refusing to start or overwrite anything:\n"
            + "\n".join(existing)
        )

    for k_value in K_VALUES:
        make_and_run(float(k_value))

    print("\nAll seven FEEC Run-3 wavenumber simulations completed.")
