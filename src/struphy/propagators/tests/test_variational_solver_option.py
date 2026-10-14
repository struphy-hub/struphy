import logging

import pytest

from struphy import set_logging_level

set_logging_level(logging.WARNING)


@pytest.mark.mpi_skip
@pytest.mark.parametrize("solver", ["pcg", "cg"])
def test_variational_barotropic_solver_option(solver):
    """Options.solver selects the mass-matrix solvers of VariationalDensityEvolve and VariationalMomentumAdvection."""
    from feectools.linalg.solvers import ConjugateGradient, PConjugateGradient

    from struphy import DerhamOptions, FieldsBackground, Simulation, domains, grids, perturbations
    from struphy.models import VariationalBarotropicFluid

    model = VariationalBarotropicFluid()
    sim = Simulation(
        model=model,
        domain=domains.Cuboid(),
        grid=grids.TensorProductGrid(num_elements=(8, 1, 1)),
        derham_opts=DerhamOptions(degree=(2, 1, 1)),
    )

    prop_dens = model.propagators.variat_dens
    prop_mom = model.propagators.variat_mom
    prop_dens.options = prop_dens.Options(model="barotropic", solver=solver)
    prop_mom.options = prop_mom.Options(solver=solver)

    model.fluid.density.add_background(FieldsBackground(values=(1.0,)))
    model.fluid.density.add_perturbation(perturbations.ModesCos(ls=(1,), amps=(0.1,)))
    model.fluid.velocity.add_background(FieldsBackground(values=(0.0, 0.0, 0.0)))
    model.fluid.velocity.add_perturbation(perturbations.ModesCos(ls=(1,), amps=(0.1,), comp=0))

    sim.allocate()

    solver_cls = PConjugateGradient if solver == "pcg" else ConjugateGradient
    for prop in (prop_dens, prop_mom):
        assert type(prop._Mrho_inv) is solver_cls
        assert type(prop._inv_Mv) is solver_cls

    prop_dens(0.05)
    prop_mom(0.05)


if __name__ == "__main__":
    test_variational_barotropic_solver_option("cg")
