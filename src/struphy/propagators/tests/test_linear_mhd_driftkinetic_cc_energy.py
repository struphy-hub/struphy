import logging

import pytest

from struphy import (
    BoundaryParameters,
    DerhamOptions,
    LoadingParameters,
    Simulation,
    WeightsParameters,
    domains,
    equils,
    grids,
    maxwellians,
    perturbations,
    set_logging_level,
)
from struphy.linear_algebra.solver import DiscreteGradientSolverParameters, SolverParameters
from struphy.models import LinearMHDDriftkineticCC
from struphy.pic.accumulation.filter import FilterParameters

set_logging_level(logging.WARNING)


@pytest.mark.parametrize("use_filter", [None, "fourier_in_tor"])
def test_energy_conservation_full_f(use_filter: str):
    """LinearMHDDriftkineticCC in a tokamak, full-f, sonic steps off, discrete gradients: en_tot changes by the
    energy of the refilled markers only."""
    propagators_class = LinearMHDDriftkineticCC.Propagators
    strang_groups = propagators_class.strang_groups
    model = LinearMHDDriftkineticCC(energy_budget=True)
    for sonic_step in ("bg_current", "bg_pressure", "sound"):
        delattr(model.propagators, sonic_step)
    propagators_class.strang_groups = ()
    try:
        model.energetic_ions.set_markers(
            loading_params=LoadingParameters(ppc=30, seed=1234),
            weights_params=WeightsParameters(control_variate=False),
            boundary_params=BoundaryParameters(bc=("refill", "periodic", "periodic")),
            bufsize=2.0,
        )
        model.energetic_ions.var.add_background(maxwellians.GyroMaxwellian2D(n=(0.01, None)))
        for var in (model.mhd.velocity, model.em_fields.b_field):
            var.add_perturbation(
                perturbations.TorusModesSin(given_in_basis="2", amps=(1e-3,), ms=(2,), ns=(-1,), comp=0),
            )
        model.em_fields.b_field.add_perturbation(
            perturbations.TorusModesCos(given_in_basis="2", amps=(1e-3,), ms=(2,), ns=(-1,), comp=2),
        )

        filter_params = FilterParameters(use_filter=use_filter, modes=(1,))
        tight = SolverParameters(tol=1e-13)
        props = model.propagators
        props.push_bxe.options = props.push_bxe.Options(algo="discrete_gradient_2nd_order")
        props.push_parallel.options = props.push_parallel.Options(algo="discrete_gradient_1st_order_newton")
        props.shear_alf.options = props.shear_alf.Options(filter_params=filter_params, solver_params=tight)
        props.cc5d_density.options = props.cc5d_density.Options(solver_params=tight)
        props.cc5d_gradb.options = props.cc5d_gradb.Options(
            algo="discrete_gradient",
            filter_params=filter_params,
            dg_solver_params=DiscreteGradientSolverParameters(relaxation_factor=1.0),
        )
        props.cc5d_curlb.options = props.cc5d_curlb.Options(filter_params=filter_params, solver_params=tight)

        sim = Simulation(
            model=model,
            domain=domains.HollowTorus(a1=0.1, a2=1.0, R0=10.0, sfl=False, tor_period=6),
            equil=equils.AdhocTorus(a=1.0, R0=10.0, B0=3.0),
            grid=grids.TensorProductGrid(num_elements=(8, 16, 4), mpi_dims_mask=(True, True, False)),
            derham_opts=DerhamOptions(degree=(3, 3, 3), bcs=(("dirichlet", "dirichlet"), None, None)),
        )
        sim.allocate()

        def budget():
            model.update_scalar_quantities()
            return tuple(float(model.scalars.host_value(n)[0]) for n in ("en_tot", "en_refill", "en_weight"))

        en_tot_0, _, _ = budget()
        for _ in range(20):
            model.integrate(0.25)
        en_tot, en_refill, en_weight = budget()
    finally:
        propagators_class.strang_groups = strang_groups

    assert en_weight == 0.0
    assert abs(en_tot - en_tot_0 - en_refill) < 1e-11
    assert model.propagators.cc5d_gradb.n_not_converged == 0
