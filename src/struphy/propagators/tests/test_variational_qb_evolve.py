import logging

import pytest

from struphy import set_logging_level

set_logging_level(logging.WARNING)


@pytest.mark.mpi_skip
@pytest.mark.parametrize("linearize", [False, True])
@pytest.mark.parametrize("info", [False, True])
def test_variational_qb_evolve_step(linearize: bool, info: bool):
    """One step of VariationalQBEvolve on a tiny grid, with and without linearization and solver info."""

    from struphy import DerhamOptions, FieldsBackground, Simulation, domains, equils, grids, perturbations
    from struphy.linear_algebra.solver import NonlinearSolverParameters
    from struphy.models import ViscoResistiveLinearMHD_with_q

    model = ViscoResistiveLinearMHD_with_q()
    sim = Simulation(
        model=model,
        domain=domains.Cuboid(),
        equil=equils.HomogenSlab(),
        grid=grids.TensorProductGrid(num_elements=(4, 4, 1)),
        derham_opts=DerhamOptions(degree=(1, 1, 1)),
    )

    prop = model.propagators.variat_qb
    prop.options = prop.Options(
        model="linear_q",
        nonlin_solver=NonlinearSolverParameters(linearize=linearize, maxiter=5, info=info),
    )

    model.mhd.density.add_background(FieldsBackground())
    model.mhd.sqrt_p.add_background(FieldsBackground())
    model.mhd.sqrt_p.add_perturbation(perturbations.TorusModesCos())

    sim.allocate()
    prop(0.01)


if __name__ == "__main__":
    test_variational_qb_evolve_step(linearize=True, info=True)
