import logging

import pytest

from struphy import set_logging_level

set_logging_level(logging.WARNING)


@pytest.mark.mpi_skip
@pytest.mark.parametrize("linearize", [False, True])
@pytest.mark.parametrize("info", [False, True])
@pytest.mark.parametrize("name", ["variat_qb", "variat_alfven", "variat_sound"])
def test_variational_qb_evolve_step(linearize: bool, info: bool, name: str):
    """One step of VariationalQBEvolve (unsplit model), VariationalAlfvenEvolve or VariationalSoundEvolve (split model)
    on a tiny grid, with and without linearization and solver info."""

    from struphy import DerhamOptions, FieldsBackground, Simulation, domains, equils, grids, perturbations
    from struphy.linear_algebra.solver import NonlinearSolverParameters
    from struphy.models import VariationalPerturbativeMHD_with_q

    model = VariationalPerturbativeMHD_with_q(linear=True, split_sound=name != "variat_qb")
    sim = Simulation(
        model=model,
        domain=domains.Cuboid(),
        equil=equils.HomogenSlab(),
        grid=grids.TensorProductGrid(num_elements=(4, 4, 1)),
        derham_opts=DerhamOptions(degree=(1, 1, 1)),
    )

    for other in ("variat_dens", "variat_viscous", "variat_resist", "variat_sound"):
        prop = getattr(model.propagators, other, None)
        if prop is not None:
            prop.options = prop.Options(model="linear_q")
    if hasattr(model.propagators, "variat_alfven"):
        model.propagators.variat_alfven.options = model.propagators.variat_alfven.Options(model="linear")
    prop = getattr(model.propagators, name)
    prop.options = prop.Options(
        model="linear" if name == "variat_alfven" else "linear_q",
        nonlin_solver=NonlinearSolverParameters(linearize=linearize, maxiter=5, info=info),
    )

    model.mhd.density.add_background(FieldsBackground())
    model.mhd.sqrt_p.add_background(FieldsBackground())
    model.mhd.sqrt_p.add_perturbation(perturbations.TorusModesCos())

    sim.allocate()
    prop(0.01)


if __name__ == "__main__":
    test_variational_qb_evolve_step(linearize=True, info=True)
