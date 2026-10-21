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


@pytest.mark.parametrize("nonlinear", [False, True])
def test_shear_alfven_cc5d_stiffness_preconditioner(monkeypatch, nonlinear: bool):
    """The Kronecker preconditioner of ShearAlfvenCurrentCoupling5D (its default) changes the iterations, not the
    solution; it needs fewer iterations than the mass matrix preconditioner for a large time step."""
    import numpy as np

    from struphy import DerhamOptions, LoadingParameters, Simulation, equils, grids, maxwellians, perturbations
    from struphy.linear_algebra.schur_solver import SchurSolver
    from struphy.linear_algebra.solver import SolverParameters
    from struphy.models import LinearMHDDriftkineticCC

    def step(precond: str):
        turn_off = tuple(p for p in ALL_PROPAGATORS if p != "ShearAlfvenCurrentCoupling5D")
        model = LinearMHDDriftkineticCC(turn_off=turn_off)
        model.energetic_ions.set_markers(loading_params=LoadingParameters(Np=200, seed=1234))
        model.energetic_ions.var.add_background(
            maxwellians.GyroMaxwellian2D(n=(1e-3, None))
        )  # dilute energetic ions: |b| << |B0|
        model.mhd.velocity.add_perturbation(
            perturbations.ModesCos(ls=(1,), ms=(1,), given_in_basis="2", comp=0, amps=(1e-2,))
        )
        model.em_fields.b_field.add_perturbation(
            perturbations.ModesSin(ls=(1,), ns=(1,), given_in_basis="2", comp=1, amps=(1e-3,))
        )
        prop = model.propagators.shearalfen_cc5d
        prop.options = prop.Options(
            precond=precond, nonlinear=nonlinear, solver_params=SolverParameters(tol=1e-12, maxiter=1000, info=True)
        )
        sim = Simulation(
            model=model,
            equil=equils.HomogenSlab(B0x=0.2, B0z=1.0),
            grid=grids.TensorProductGrid(num_elements=(8, 8, 4)),
            derham_opts=DerhamOptions(degree=(2, 2, 1)),
        )
        sim.allocate()
        assert prop._precond == precond
        niter = []
        call = SchurSolver.__call__

        def recording_call(self, *args, **kwargs):
            out = call(self, *args, **kwargs)
            niter.append(out[1]["niter"])
            return out

        monkeypatch.setattr(SchurSolver, "__call__", recording_call)
        for _ in range(2):
            prop(1.0)
        monkeypatch.undo()
        return model.mhd.velocity.spline.vector.toarray(), niter

    u_mass, n_mass = step("MassMatrixPreconditioner")
    u_stiff, n_stiff = step("StiffnessPreconditioner")
    assert np.linalg.norm(u_stiff - u_mass) < 1e-8 * np.linalg.norm(u_mass)
    assert all(n < m for n, m in zip(n_stiff, n_mass)), (n_stiff, n_mass)


def test_current_coupling_5d_ep_scale():
    """The energetic-particle contributions are scaled by the mass ratio A_h/A_b by default."""
    from struphy import DerhamOptions, LoadingParameters, Simulation, equils, grids, maxwellians
    from struphy.models import LinearMHDDriftkineticCC

    model = LinearMHDDriftkineticCC(mhd_mass_number=1.0, hot_mass_number=2.0)
    model.energetic_ions.set_markers(loading_params=LoadingParameters(Np=40, seed=1234))
    model.energetic_ions.var.add_background(maxwellians.GyroMaxwellian2D())
    model.propagators.cc5d_curlb.options = model.propagators.cc5d_curlb.Options(ep_scale=0.5)
    sim = Simulation(
        model=model,
        equil=equils.HomogenSlab(),
        grid=grids.TensorProductGrid(num_elements=(4, 4, 2)),
        derham_opts=DerhamOptions(degree=(1, 1, 1)),
    )
    sim.allocate()
    for name in ("shearalfen_cc5d", *PROPAGATORS.values()):
        expected = 0.5 if name == "cc5d_curlb" else 2.0
        assert getattr(model.propagators, name)._ep_scale == expected, name


if __name__ == "__main__":
    for name in PROPAGATORS:
        test_current_coupling_5d_without_b_tilde(name)
