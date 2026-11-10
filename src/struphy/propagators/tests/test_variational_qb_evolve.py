import logging

import numpy as np
import pytest

from struphy import set_logging_level

set_logging_level(logging.WARNING)


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
        # B0 not perpendicular to the wave vector (along x), so that the Alfven step has dynamics
        equil=equils.HomogenSlab(B0x=0.5, B0z=1.0),
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
    # perturbations resolved by the 4 x 4 x 1 grid (TorusModesCos vanishes on it, the state was an equilibrium)
    model.mhd.sqrt_p.add_perturbation(perturbations.ModesCos(ls=(1,), amps=(0.1,)))
    model.mhd.velocity.add_perturbation(perturbations.ModesCos(ls=(1,), amps=(0.1,), comp=1))

    sim.allocate()

    def coeffs():
        return {
            f"{species}.{var}": getattr(getattr(model, species), var).spline.vector.toarray().copy()
            for species, var in (("mhd", "density"), ("mhd", "velocity"), ("mhd", "sqrt_p"), ("em_fields", "b_field"))
            if hasattr(getattr(model, species, None), var)
        }

    before = coeffs()
    prop(0.01)
    after = coeffs()

    for key, value in after.items():
        assert np.all(np.isfinite(value)), f"{name}: {key} not finite after one step"
    changed = [key for key in after if not np.array_equal(after[key], before[key])]
    assert changed, f"{name}: one step changed no variable"


if __name__ == "__main__":
    test_variational_qb_evolve_step(linearize=True, info=True)
