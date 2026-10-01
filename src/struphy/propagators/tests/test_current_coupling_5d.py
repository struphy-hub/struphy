import logging

import pytest

from struphy import set_logging_level

set_logging_level(logging.WARNING)

ALL_PROPAGATORS = (
    "PushGuidingCenterBxEstar",
    "PushGuidingCenterParallel",
    "ShearAlfvenCurrentCoupling5D",
    "Magnetosonic",
    "CurrentCoupling5DDensity",
    "CurrentCoupling5DGradB",
    "CurrentCoupling5DCurlb",
)

PROPAGATORS = {
    "CurrentCoupling5DDensity": "cc5d_density",
    "CurrentCoupling5DCurlb": "cc5d_curlb",
    "CurrentCoupling5DGradB": "cc5d_gradb",
}


@pytest.mark.mpi_skip
@pytest.mark.parametrize("name", list(PROPAGATORS))
def test_current_coupling_5d_without_b_tilde(name: str):
    """One step of the 5D current-coupling propagators with b_tilde=None equals one with b_tilde=0."""
    import numpy as np

    from struphy import DerhamOptions, LoadingParameters, Simulation, domains, equils, grids, maxwellians, perturbations
    from struphy.models import LinearMHDDriftkineticCC

    def step(b_tilde_none: bool):
        turn_off = tuple(p for p in ALL_PROPAGATORS if p != name)
        model = LinearMHDDriftkineticCC(turn_off=turn_off)
        model.energetic_ions.set_markers(loading_params=LoadingParameters(Np=40, seed=1234))
        model.energetic_ions.var.add_background(maxwellians.GyroMaxwellian2D())
        model.mhd.velocity.add_perturbation(perturbations.TorusModesCos(given_in_basis="2", comp=0))
        sim = Simulation(
            model=model,
            domain=domains.Cuboid(),
            equil=equils.HomogenSlab(),
            grid=grids.TensorProductGrid(num_elements=(4, 4, 2)),
            derham_opts=DerhamOptions(degree=(1, 1, 1)),
        )
        prop = getattr(model.propagators, PROPAGATORS[name])
        if b_tilde_none:
            prop.b_tilde = None
        sim.allocate()
        prop(0.01)
        return model.mhd.velocity.spline.vector.toarray()

    # the model's b_field has no initial condition, hence b_tilde=0
    assert np.allclose(step(b_tilde_none=True), step(b_tilde_none=False))


if __name__ == "__main__":
    for name in PROPAGATORS:
        test_current_coupling_5d_without_b_tilde(name)
