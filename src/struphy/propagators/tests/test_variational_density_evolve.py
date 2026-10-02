import logging

import pytest

from struphy import set_logging_level

set_logging_level(logging.WARNING)


@pytest.mark.mpi_skip
def test_variational_density_evolve_barotropic_newton():
    """Barotropic Newton iteration of VariationalDensityEvolve converges quadratically (exact Jacobian)."""

    from struphy import DerhamOptions, FieldsBackground, Simulation, domains, grids, perturbations
    from struphy.models import VariationalBarotropicFluid

    model = VariationalBarotropicFluid()
    sim = Simulation(
        model=model,
        domain=domains.Cuboid(),
        grid=grids.TensorProductGrid(num_elements=(16, 1, 1)),
        derham_opts=DerhamOptions(degree=(2, 1, 1)),
    )

    prop = model.propagators.variat_dens
    prop.options = prop.Options(model="barotropic")

    model.fluid.density.add_background(FieldsBackground(values=(1.0,)))
    model.fluid.density.add_perturbation(perturbations.ModesCos(ls=(1,), amps=(0.1,)))
    model.fluid.velocity.add_background(FieldsBackground(values=(0.0, 0.0, 0.0)))

    sim.allocate()

    errors = []
    get_error = prop._get_error_newton

    def _get_error(*args):
        errors.append(get_error(*args))
        return errors[-1]

    prop._get_error_newton = _get_error
    prop(0.05)

    # quadratic convergence: 2 Newton steps (6 with the Jacobian term missing)
    assert len(errors) - 1 <= 3, errors


if __name__ == "__main__":
    test_variational_density_evolve_barotropic_newton()
