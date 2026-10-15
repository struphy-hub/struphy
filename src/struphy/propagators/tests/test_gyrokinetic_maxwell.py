"""Energy conservation and features of the gyrokinetic Maxwell and Darwin models, each propagator run in isolation
on a small slab (zero Larmor radius and finite Larmor radius)."""

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

VTH = {"ions": 1e-3, "electrons": 1e-2}


def _make_simulation(
    seed_e_field: bool,
    n_gyro: int = 0,
    darwin: bool = False,
    anisotropy: float = 1.0,
    flow: float = 0.0,
    num_elements=(4, 1, 8),
    **model_kwargs,
):
    """A slab with B_0 = e_z, ions and (heavy) electrons in light units; the ions' background is anisotropic by the
    factor ``anisotropy`` (vth_perp = anisotropy * vth_para) and flows with ``flow`` (units of c)."""
    import numpy as np

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
    from struphy.models import GyrokineticDarwin, GyrokineticMaxwell

    cls = GyrokineticDarwin if darwin else GyrokineticMaxwell
    model = cls(
        base_units=BaseUnits(x=1.0, B=1.0, n=1.0),
        species=(("ions", 1, 1.0), ("electrons", -1, 1 / 100)),
        n_gyro=n_gyro,
        **model_kwargs,
    )
    for name, v in VTH.items():
        species = getattr(model, f"kinetic_{name}")
        species.set_markers(
            loading_params=LoadingParameters(Np=200, seed=1234, moments=(0.0, 0.0, v, v), B0=1.0),
            boundary_params=BoundaryParameters(bc=("periodic", "periodic", "periodic")),
        )
        if name == "ions":
            background = maxwellians.GyroMaxwellian2D(
                n=(1.0, None),
                u_para=(flow, None),
                vth_para=(v, None),
                vth_perp=(anisotropy * v, None),
                B0=1.0,
            )
        else:
            background = maxwellians.GyroMaxwellian2D(n=(1.0, None), vth_para=(v, None), vth_perp=(v, None), B0=1.0)
        species.var.add_background(background)

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
        grid=grids.TensorProductGrid(num_elements=num_elements),
        derham_opts=DerhamOptions(degree=(1, 1, 2)),
    )
    sim.allocate()
    assert np.isfinite(_total_energy(model))
    return sim, model


def _total_energy(model):
    model.scalars.update()
    return float(model.scalars.host_value("en_tot")[0])


@pytest.mark.mpi_skip
@pytest.mark.parametrize("name", list(PROPAGATORS))
@pytest.mark.parametrize("n_gyro", [0, 4])
def test_energy_conservation(name: str, n_gyro: int):
    """Each propagator alone conserves the model energy H = e^T A e/2 + b^T M2 b/2 + sum alpha^2 w (v^2/2 + mu B_par),
    in the zero-Larmor-radius limit and with finite Larmor radius (gyro- and disc averages)."""
    sim, model = _make_simulation(seed_e_field=(name == "drifts"), n_gyro=n_gyro, solve_initial_poisson=False)
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

    sim, model = _make_simulation(seed_e_field=True, solve_initial_poisson=False)

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
@pytest.mark.parametrize("n_gyro", [0, 4])
def test_full_splitting_energy(n_gyro: int):
    """The Lie-Trotter splitting of all propagators has a small energy error (second-order drift coupling)."""
    sim, model = _make_simulation(seed_e_field=True, n_gyro=n_gyro, solve_initial_poisson=False)

    en0 = _total_energy(model)
    for _ in range(3):
        model.integrate(5.0, split_algo="LieTrotter")
    en = _total_energy(model)

    assert abs(en - en0) / en0 < 1e-5


@pytest.mark.mpi_skip
def test_gauss_law():
    """The initial Poisson solve with the dielectric matrix satisfies the weak Gauss law to solver precision, and the
    Gauss-law residual is tracked as a scalar."""
    sim, model = _make_simulation(seed_e_field=False, solve_initial_poisson=True, measure_gauss_law=True)
    model.scalars.update()
    assert float(model.scalars.host_value("gauss_error")[0]) < 1e-9
    model.integrate(5.0, split_algo="LieTrotter")
    model.scalars.update()
    assert float(model.scalars.host_value("gauss_error")[0]) >= 0.0


@pytest.mark.mpi_skip
def test_anisotropic_background_with_flow():
    """With an anisotropic ion background with parallel flow, the magnetizing matrix and the (anti-symmetric) flow
    operator enter Maxwell's equations, and the Maxwell step still conserves the energy exactly."""
    sim, model = _make_simulation(seed_e_field=True, anisotropy=2.0, flow=5e-4, solve_initial_poisson=False)
    assert model.background.has_anisotropy
    assert model.background.has_flow
    prop = model.propagators.maxwell
    assert prop._K is not None, "the flow operator is missing"
    assert prop._M2H is not sim.mass_ops.M2, "the magnetizing matrix is missing"

    en0 = _total_energy(model)
    for _ in range(3):
        prop(5.0)
    en = _total_energy(model)
    assert abs(en - en0) / en0 < 1e-10


@pytest.mark.mpi_skip
def test_background_moments_match_fluid_equilibrium():
    """The kinetic backgrounds give the same polarization as the fluid equilibrium for a centred Maxwellian with
    the equilibrium density."""
    import numpy as np

    sim, model = _make_simulation(seed_e_field=False, solve_initial_poisson=False)
    e = np.linspace(0.1, 0.9, 3)
    e1, e2, e3 = np.meshgrid(e, e, e, indexing="ij")
    weights = model.background
    polarization = weights.polarization(e1, e2, e3)
    expected = sum(model._alpha2.values()) * 1.0 / 1.0**2
    assert np.allclose(polarization, expected)
    assert np.allclose(weights.flow(e1, e2, e3), 0.0)
    assert np.allclose(weights.anisotropy(e1, e2, e3), 0.0)


@pytest.mark.mpi_skip
def test_parallel_flow_from_equilibrium():
    """The parallel flow of Eq. (5.62) reproduces the parallel equilibrium current of a screw pinch."""
    import numpy as np

    from struphy import domains, equils
    from struphy.propagators.gyrokinetic_maxwell import parallel_flow_from_equilibrium

    equil = equils.ScrewPinch(a=1.0, R0=5.0, B0=1.0, q0=1.05, q1=1.8, n1=2.0, n2=1.0, na=0.2, p0=0.005)
    equil.domain = domains.HollowCylinder(a1=0.2, a2=1.0, Lz=2 * np.pi * 5.0)
    alpha, epsilon = 137.4, 3.13
    density = equil.n0
    u_par = parallel_flow_from_equilibrium(equil, alpha, epsilon, density, fraction=0.5)

    e = np.array([0.3, 0.6])
    e1, e2, e3 = np.meshgrid(e, e, e, indexing="ij")
    j, _ = equil.j_cart(e1, e2, e3)
    b, _ = equil.unit_b_cart(e1, e2, e3)
    j_par = j[0] * b[0] + j[1] * b[1] + j[2] * b[2]
    current = alpha**2 / epsilon * density(e1, e2, e3) * u_par(e1, e2, e3)
    assert np.allclose(current, 0.5 * j_par)
    # also callable with a marker array
    assert np.allclose(u_par(np.stack([e1.ravel(), e2.ravel(), e3.ravel()], axis=1)), u_par(e1, e2, e3).ravel())


@pytest.mark.mpi_skip
def test_three_species():
    """Any number of species: ions, electrons and a hot ion species."""
    from struphy import LoadingParameters, maxwellians
    from struphy.models import GyrokineticMaxwell

    model = GyrokineticMaxwell(species=(("ions", 1, 1.0), ("electrons", -1, 1 / 1836), ("hot", 2, 4.0)))
    assert model.species_names == ["ions", "electrons", "hot"]
    assert model.kinetic_hot.var.species.charge_number == 2
    assert {"epar_hot", "push_para_hot"} <= set(model.propagators.__dict__)
    assert {"en_hot", "n_lost_hot"} <= set(model.scalars.dct)
    for name in model.species_names:
        getattr(model, f"kinetic_{name}").set_markers(loading_params=LoadingParameters(Np=10))
        getattr(model, f"kinetic_{name}").var.add_background(maxwellians.GyroMaxwellian2D())


@pytest.mark.mpi_skip
@pytest.mark.parametrize("n_gyro", [0, 4])
def test_darwin_energy(n_gyro: int):
    """The quasi-neutral Darwin model: the implicit field coupling (without drifts) conserves the Darwin energy."""
    sim, model = _make_simulation(seed_e_field=False, n_gyro=n_gyro, darwin=True, with_drifts=False)
    assert "fields" in model.propagators.__dict__

    en0 = _total_energy(model)
    for _ in range(3):
        model.propagators.fields(5.0)
    en = _total_energy(model)
    assert abs(en - en0) / en0 < 1e-8


@pytest.mark.mpi_skip
def test_darwin_initial_quasi_neutrality():
    """The initial electric field of the Darwin model solves the quasi-neutrality equation."""
    import numpy as np

    sim, model = _make_simulation(seed_e_field=False, darwin=True, with_drifts=False, solve_initial_poisson=True)
    # a parallel flow of the ions makes the marker weights non-uniform (a non-trivial charge density)
    model.kinetic_ions.var.particles.markers[:, 5] *= 1.0 + 0.1 * np.cos(
        2 * np.pi * model.kinetic_ions.var.particles.markers[:, 0]
    )
    model._setup_gauss_law()
    PL = model.propagators.fields.longitudinal_operator
    e = model.em_fields.e_field.spline.vector
    rho = model._charge_density(model._gauss_rhs)
    assert np.linalg.norm(rho.toarray()) > 0.0
    # G^T P e = -rho up to the perpendicular average of rho (the kernel of the perpendicular Laplacian, which the
    # polarization cannot balance and which is removed from the potential)
    residual = model.propagators.fields._P.dot(e)
    residual = PL._grad.T.dot(residual) + rho
    PL.remove_average(residual)
    assert np.linalg.norm(residual.toarray()) < 1e-8 * np.linalg.norm(rho.toarray())


@pytest.mark.mpi_skip
def test_darwin_kperp0_polarization():
    """The modes constant in the perpendicular directions keep the full polarization (P_L e = P e), the others
    only the longitudinal part (P_L e = 0 for a transversal mode)."""
    import numpy as np

    sim, model = _make_simulation(
        seed_e_field=False, darwin=True, with_drifts=False, solve_initial_poisson=False, num_elements=(4, 4, 8)
    )
    fields = model.propagators.fields
    PL, P, P1 = fields.longitudinal_operator, fields._P, fields.derham.P1

    def norm(v):
        return np.linalg.norm(v.toarray())

    zero = lambda x, y, z: 0.0 * x  # noqa: E731
    # k_perp = 0: E_x = cos(2 pi z) is both curl- and divergence-free in the perpendicular plane
    e = P1((lambda x, y, z: np.cos(2 * np.pi * z), zero, zero))
    e.update_ghost_regions()
    assert norm(PL.dot(e) - P.dot(e)) < 1e-10 * norm(P.dot(e))
    # a transversal mode E_x = cos(2 pi y) has no longitudinal part and no perpendicular average
    e = P1((lambda x, y, z: np.cos(2 * np.pi * y), zero, zero))
    e.update_ghost_regions()
    assert norm(PL.dot(e)) < 1e-10 * norm(P.dot(e))
    # a longitudinal mode E_x = cos(2 pi x) is reproduced
    e = P1((lambda x, y, z: np.cos(2 * np.pi * x), zero, zero))
    e.update_ghost_regions()
    assert norm(PL.dot(e) - P.dot(e)) < 1e-8 * norm(P.dot(e))


@pytest.mark.mpi_skip
@pytest.mark.parametrize("n_gyro", [0, 4])
def test_darwin_with_drifts(n_gyro: int):
    """The Darwin model with the E x B and grad-B drifts (pbicgstab on the non-symmetric system): stable, fields and
    markers move, the energy error of the explicit drift coupling stays small."""
    import numpy as np

    sim, model = _make_simulation(seed_e_field=True, n_gyro=n_gyro, darwin=True, with_drifts=True)
    e0 = model.em_fields.e_field.spline.vector.toarray().copy()
    eta0 = model.kinetic_ions.var.particles.markers_wo_holes[:, :3].copy()
    en0 = _total_energy(model)
    for _ in range(3):
        model.integrate(5.0, "LieTrotter")
    en = _total_energy(model)
    assert np.isfinite(en)
    assert abs(en - en0) / en0 < 1e-3
    assert np.linalg.norm(model.em_fields.e_field.spline.vector.toarray() - e0) > 0.0
    assert np.abs(model.kinetic_ions.var.particles.markers_wo_holes[:, :3] - eta0).max() > 0.0


if __name__ == "__main__":
    for name in PROPAGATORS:
        test_energy_conservation(name, 0)
        test_energy_conservation(name, 4)
    test_fields_and_markers_change()
    test_full_splitting_energy(0)
    test_gauss_law()
    test_anisotropic_background_with_flow()
    test_darwin_kperp0_polarization()
    test_darwin_with_drifts(0)
    test_background_moments_match_fluid_equilibrium()
    test_parallel_flow_from_equilibrium()
    test_three_species()
    test_darwin_energy(0)
    test_darwin_initial_quasi_neutrality()
