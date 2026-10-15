"""Energy conservation of the propagators of GyrokineticMaxwellZLR, each run in isolation on a small slab."""

import logging

import pytest

from struphy import set_logging_level

set_logging_level(logging.WARNING)

# propagator handle -> relative energy tolerance after a few steps (exact up to the solver tolerance,
# except for the explicit marker push of the drifts, which is consistent to second order)
PROPAGATORS = {
    "maxwell": 1e-10,
    "epar_ions": 1e-10,
    "epar_electrons": 1e-10,
    "push_para_ions": 1e-10,
    "push_para_electrons": 1e-10,
    "drifts": 1e-9,
}


def _make_simulation(seed_e_field: bool):
    from struphy import (
        BaseUnits,
        BoundaryParameters,
        DerhamOptions,
        LoadingParameters,
        Simulation,
        domains,
        equils,
        grids,
        maxwellians,
        perturbations,
    )
    from struphy.models import GyrokineticMaxwellZLR

    # slab with B_0 = e_z, ions and (heavy) electrons in light units
    vth = {"kinetic_ions": 1e-3, "kinetic_electrons": 1e-2}
    model = GyrokineticMaxwellZLR(base_units=BaseUnits(x=1.0, B=1.0, n=1.0), electron_mass_number=1 / 100)
    for name, v in vth.items():
        species = getattr(model, name)
        species.set_markers(
            loading_params=LoadingParameters(Np=200, seed=1234, moments=(0.0, 0.0, v, v), B0=1.0),
            boundary_params=BoundaryParameters(bc=("periodic", "periodic", "periodic")),
        )
        species.var.add_background(
            maxwellians.GyroMaxwellian2D(n=(1.0, None), vth_para=(v, None), vth_perp=(v, None), B0=1.0),
        )

    # shear Alfvén seed B_y ~ cos(2 pi z); with seed_e_field also an electric field and a parallel
    # magnetic perturbation B_z ~ cos(2 pi x), such that grad B_parallel and the E x B drift are non-zero
    model.em_fields.b_field.add_perturbation(perturbations.ModesCos(ns=(1,), amps=(1e-3,), comp=1))
    if seed_e_field:
        model.em_fields.e_field.add_perturbation(perturbations.ModesCos(ns=(1,), amps=(1e-4,), comp=0))
        model.em_fields.b_field.add_perturbation(perturbations.ModesCos(ls=(1,), amps=(1e-3,), comp=2))

    sim = Simulation(
        model=model,
        domain=domains.Cuboid(r1=0.25, r2=0.25, r3=1.0),
        equil=equils.HomogenSlab(B0z=1.0, n0=1.0, beta=0.0),
        grid=grids.TensorProductGrid(num_elements=(4, 1, 8)),
        derham_opts=DerhamOptions(degree=(1, 1, 2)),
    )
    sim.allocate()
    return sim, model


def _total_energy(model):
    model.scalars.update()
    return float(model.scalars.host_value("en_tot")[0])


@pytest.mark.mpi_skip
@pytest.mark.parametrize("name", list(PROPAGATORS))
def test_energy_conservation(name: str):
    """Each propagator alone conserves the model energy H = e^T A e/2 + b^T M2 b/2 + sum alpha^2 w (v^2/2 + mu B_par)."""
    sim, model = _make_simulation(seed_e_field=(name == "drifts"))
    prop = getattr(model.propagators, name)

    en0 = _total_energy(model)
    assert en0 > 0.0

    for _ in range(3):
        prop(5.0)
    en = _total_energy(model)

    assert abs(en - en0) / en0 < PROPAGATORS[name], f"{name}: relative energy error {abs(en - en0) / en0}"


@pytest.mark.mpi_skip
def test_fields_and_markers_change():
    """The propagators actually move the fields and the markers (the energy test must not pass trivially)."""
    import numpy as np

    sim, model = _make_simulation(seed_e_field=True)

    e_old = model.em_fields.e_field.spline.vector.toarray().copy()
    b_old = model.em_fields.b_field.spline.vector.toarray().copy()
    markers_old = model.kinetic_ions.var.particles.markers_wo_holes[:, :4].copy()

    model.integrate(5.0, split_algo="LieTrotter")

    assert not np.allclose(e_old, model.em_fields.e_field.spline.vector.toarray())
    assert not np.allclose(b_old, model.em_fields.b_field.spline.vector.toarray())
    markers = model.kinetic_ions.var.particles.markers_wo_holes[:, :4]
    assert not np.allclose(markers_old[:, :3], markers[:, :3])  # positions (parallel streaming and drifts)
    assert not np.allclose(markers_old[:, 3], markers[:, 3])  # parallel velocity


@pytest.mark.mpi_skip
def test_full_splitting_energy():
    """The Lie-Trotter splitting of all propagators has a small energy error (second-order drift coupling)."""
    sim, model = _make_simulation(seed_e_field=True)

    en0 = _total_energy(model)
    for _ in range(3):
        model.integrate(5.0, split_algo="LieTrotter")
    en = _total_energy(model)

    assert abs(en - en0) / en0 < 1e-5


if __name__ == "__main__":
    for name in PROPAGATORS:
        test_energy_conservation(name)
    test_fields_and_markers_change()
    test_full_splitting_energy()
