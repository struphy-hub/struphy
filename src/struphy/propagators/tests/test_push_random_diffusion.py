import numpy as np
import pytest


@pytest.mark.mpi_skip
@pytest.mark.parametrize("algo", [None, "forward_euler", "heun2", "rk4"])
def test_random_diffusion_msd(algo):
    """One step must add the Wiener increment exactly once: <dx_i^2> = 2 D dt per dimension."""
    from struphy import BoundaryParameters, LoadingParameters, domains
    from struphy.models.variables import PICVariable
    from struphy.ode.utils import ButcherTableau
    from struphy.pic.particles import Particles3D
    from struphy.propagators.push_random_diffusion import PushRandomDiffusion

    diff_coeff, dt = 1e-3, 1e-2

    domain = domains.Cuboid()
    particles = Particles3D(
        loading_params=LoadingParameters(Np=4000, seed=1234),
        boundary_params=BoundaryParameters(),
        domain=domain,
    )
    particles.draw_markers()
    particles.initialize_weights()

    var = PICVariable(space="Particles3D")
    var._particles = particles

    prop = PushRandomDiffusion()
    prop.variables.var = var
    prop.domain = domain
    butcher = None if algo is None else ButcherTableau(algo)
    prop.options = PushRandomDiffusion.Options(butcher=butcher, diff_coeff=diff_coeff)
    prop.allocate()

    np.random.seed(0)
    valid = particles.markers[:, 0] != -1.0
    x0 = particles.markers[valid, :3].copy()
    prop(dt)
    dx = particles.markers[valid, :3] - x0
    dx -= np.round(dx)  # minimal image in the periodic unit cube

    msd = np.mean(dx**2, axis=0) / (2 * diff_coeff * dt)
    # statistical error of the MSD estimate is sqrt(2 / 4000) ~ 0.022
    assert np.all(np.abs(msd - 1.0) < 0.1), msd


if __name__ == "__main__":
    test_random_diffusion_msd(None)
    test_random_diffusion_msd("rk4")
