"""Variational MHD with polar splines, end to end: the projected equilibrium in the polar spaces (the FEEC space of the
Derham) and the evaluation of polar spline functions on the quadrature grid (SplineFunction.eval_tp_fixed_loc with the
polar coefficients, issue #921)."""

import numpy as np

from struphy import DerhamOptions, EnvironmentOptions, Simulation, Time, domains, equils, grids
from struphy.initial import perturbations

# relative change of the energy over the steps below: about 1e-8 with the polar coefficients in the evaluation of the
# spline functions, about 2e-7 without (issue #921); measured with the default solver and Newton tolerances of the
# propagators of VariationalMHD_with_q, which bound the energy error and must stay well below this tolerance
ENERGY_TOL = 5e-8


def test_variational_mhd_with_q_polar(mpi_out_folder):
    """The nonlinear variational MHD model (with q = sqrt(p)) on a polar cylinder (C1 polar splines, no hole) with a
    screw pinch: the velocity changes and the energy is conserved."""
    from struphy.models import VariationalMHD_with_q

    model = VariationalMHD_with_q()
    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=mpi_out_folder, sim_folder="variational_polar"),
        time_opts=Time(dt=0.02, Tend=0.02),
        domain=domains.IGAPolarCylinder(num_elements=(4, 6), degree=(2, 2), a=1.0, Lz=4.0),
        equil=equils.ScrewPinch(a=1.0, R0=4.0 / (2 * np.pi), B0=1.0, q0=1.05, q1=1.8, n1=0.0, n2=0.0, na=1.0, beta=0.1),
        grid=grids.TensorProductGrid(num_elements=(4, 6, 2)),
        derham_opts=DerhamOptions(degree=(2, 2, 1), bcs=(("free", "dirichlet"), None, None), polar_splines=True),
    )
    for comp, (ls, ms) in enumerate((((1,), (1,)), ((1,), (2,)), ((2,), (1,)))):
        model.mhd.velocity.add_perturbation(
            perturbations.ModesCos(ls=ls, ms=ms, amps=(1e-3,), comp=comp, given_in_basis="physical")
        )
    sim.allocate()

    def energy():
        model.scalars.update()
        return float(np.squeeze(model.scalars.host_value("en_tot")))

    en0 = energy()
    u0 = model.mhd.velocity.spline.vector.copy()
    for _ in range(3):
        for prop in vars(model.propagators).values():
            prop(0.02)
    en1 = energy()
    du = model.mhd.velocity.spline.vector - u0
    assert du.inner(du) > 0.0
    assert abs(en1 - en0) / en0 < ENERGY_TOL, f"relative energy change {abs(en1 - en0) / en0:.2e}"
