"""The 1-form versions (``...1form``, magnetic field in H(curl)) of the propagators of the hybrid models reading the
magnetic field against their 2-form versions (H(div)): one step from the same physical state (same physical fields
projected into the two spaces, same markers) gives the same update, up to the projection errors."""

import logging

import cunumpy as xp
import pytest

from struphy import set_logging_level

set_logging_level(logging.WARNING)

# (model, propagator, options): the 1-form kernels of each algorithm / matrix-free variant
CASES = [
    ("LinearMHDVlasovCC", "couple_dens", {"matrix_free": False}),
    ("LinearMHDVlasovCC", "couple_dens", {"matrix_free": True}),
    ("LinearMHDVlasovCC", "couple_curr", {"matrix_free": False}),
    ("LinearMHDVlasovCC", "couple_curr", {"matrix_free": True}),
    ("LinearMHDVlasovCC", "push_vxb", {}),
    ("LinearMHDDriftkineticCC", "cc5d_density", {"matrix_free": False}),
    ("LinearMHDDriftkineticCC", "cc5d_density", {"matrix_free": True}),
    ("LinearMHDDriftkineticCC", "cc5d_curlb", {"matrix_free": False}),
    ("LinearMHDDriftkineticCC", "cc5d_curlb", {"matrix_free": True}),
    ("LinearMHDDriftkineticCC", "cc5d_gradb", {"algo": "explicit"}),
    ("LinearMHDDriftkineticCC", "cc5d_gradb", {"algo": "discrete_gradient"}),
    ("LinearMHDDriftkineticCC", "push_bxe", {"algo": "discrete_gradient_1st_order"}),
    ("LinearMHDDriftkineticCC", "push_bxe", {"algo": "discrete_gradient_2nd_order"}),
    ("LinearMHDDriftkineticCC", "push_parallel", {"algo": "discrete_gradient_1st_order"}),
    ("LinearMHDDriftkineticCC", "push_parallel", {"algo": "discrete_gradient_1st_order_newton"}),
    ("LinearMHDDriftkineticCC", "push_parallel", {"algo": "discrete_gradient_2nd_order"}),
    ("LinearMHDDriftkineticCC", "push_parallel", {"algo": "explicit"}),
]


def _step(model_name: str, prop_name: str, options: dict, b_space: str, dt: float, num_elements=(12, 8, 4)):
    """One step of the propagator ``prop_name`` of the model with the given ``b_space``; returns the velocity and the
    marker phase-space coordinates before and after."""
    import struphy.models.utils as models_utils
    from struphy import DerhamOptions, LoadingParameters, Simulation, domains, equils, grids, maxwellians, perturbations

    model = models_utils.get_model_by_name(model_name)(b_space=b_space)
    prop = getattr(model.propagators, prop_name)
    assert type(prop).__name__.endswith("1form") == (b_space == "Hcurl")
    prop.options = prop.Options(**options)

    six_d = model_name == "LinearMHDVlasovCC"
    # physical perturbations with the box lengths: resolved modes
    lengths = {"Lx": 1.0, "Ly": 2 * xp.pi, "Lz": 6 * xp.pi}
    model.energetic_ions.set_markers(loading_params=LoadingParameters(Np=200, seed=1234))
    model.energetic_ions.var.add_background(
        maxwellians.Maxwellian3D(n=(0.1, None)) if six_d else maxwellians.GyroMaxwellian2D(n=(0.1, None))
    )
    model.mhd.velocity.add_perturbation(
        perturbations.ModesCos(ls=(1,), ms=(1,), given_in_basis="physical", comp=0, amps=(1e-2,), **lengths)
    )
    model.em_fields.b_field.add_perturbation(
        perturbations.ModesSin(ls=(1,), ns=(1,), given_in_basis="physical", comp=1, amps=(1e-2,), **lengths)
    )
    sim = Simulation(
        model=model,
        domain=domains.Cuboid(r1=1.0, r2=2 * xp.pi, r3=6 * xp.pi),
        equil=equils.ShearedSlab(a=1.0, R0=3.0),
        grid=grids.TensorProductGrid(num_elements=num_elements),
        derham_opts=DerhamOptions(degree=(3, 2, 2), bcs=(("free", "free"), None, None)),
    )
    sim.allocate()
    particles = model.energetic_ions.var.particles
    cols = slice(0, 6) if six_d else slice(0, 4)
    u0 = model.mhd.velocity.spline.vector.toarray()
    m0 = _gathered(particles, cols)
    prop(dt)
    u1 = model.mhd.velocity.spline.vector.toarray()
    m1 = _gathered(particles, cols)
    return (u0, u1), (m0, m1)


def _gathered(particles, cols: slice):
    """Marker IDs (first row) and the columns ``cols`` of the markers of all ranks, sorted by ID, without duplicate
    IDs (the pushers in eta move markers across ranks; IDs can overlap at rank boundaries)."""
    data = xp.vstack([particles.marker_ids[None, :], particles.markers[cols, particles.valid_mks]])
    comm = particles.mpi_comm
    if comm is not None and comm.Get_size() > 1:
        data = xp.hstack(comm.allgather(data))
    ids, counts = xp.unique(data[0], return_counts=True)
    data = data[:, xp.isin(data[0], ids[counts == 1])]
    return data[:, xp.argsort(data[0])]


def _differences(model_cls: str, prop_name: str, options: dict, num_elements: tuple):
    """Relative differences of the updates (velocity, markers) of the 1-form and the 2-form propagator."""
    (u0_2, u1_2), (m0_2, m1_2) = _step(model_cls, prop_name, options, "Hdiv", 0.05, num_elements)
    (u0_1, u1_1), (m0_1, m1_1) = _step(model_cls, prop_name, options, "Hcurl", 0.05, num_elements)

    # same initial state (the velocity is in H(div) in both models)
    assert xp.allclose(m0_1, m0_2)
    assert xp.linalg.norm(u0_1 - u0_2) < 1e-10 * xp.linalg.norm(u0_2)

    rels = {}
    for name, d1, d2 in (("u", u1_1 - u0_1, u1_2 - u0_2), ("markers", m1_1 - m0_1, m1_2 - m0_2)):
        norm = xp.linalg.norm(d2)
        if norm == 0.0:
            assert xp.linalg.norm(d1) == 0.0, name
            continue
        rels[name] = xp.linalg.norm(d1 - d2) / norm
    return rels


# the propagators with the gradient of the parallel perturbation or the curvature of b0: larger projection errors
CONVERGENCE = {"push_bxe", "cc5d_gradb", "cc5d_curlb"}


@pytest.mark.parametrize("model_cls, prop_name, options", CASES)
def test_1form_equals_2form(model_cls: str, prop_name: str, options: dict):
    """Same update up to the projection errors of the magnetic fields: below 2 % on the coarse grid, or (for the
    propagators in CONVERGENCE) a difference that decreases under grid refinement (factor 4-5 for 12x8x4 -> 24x16x8)."""
    rels = _differences(model_cls, prop_name, options, (12, 8, 4))
    if prop_name not in CONVERGENCE:
        for name, rel in rels.items():
            assert rel < 2e-2, f"{name}: relative difference {rel:.2e}"
    else:
        rels_fine = _differences(model_cls, prop_name, options, (24, 16, 8))
        for name, rel in rels.items():
            # the marker update of cc5d_gradb is already at the level of the particle (not field) errors
            if rel < 5e-3:
                continue
            assert rels_fine[name] < rel / 3, f"{name}: relative difference {rel:.2e} -> {rels_fine[name]:.2e}"
