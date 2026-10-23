import logging

import cunumpy as xp
import pytest

from struphy import set_logging_level

set_logging_level(logging.WARNING)

LX, LY, LZ = 2.0, 2.0, 4.0
DB = 0.1  # uniform perturbation B~_x on B0 = e_z


def _model(**coupling_options):
    """DriftKineticMaxwell in a box with B0 = e_z and the uniform perturbation B~ = DB e_x, cold guiding centers
    (mu ~ 0) streaming with v_par ~ 1, no electric field."""
    from struphy import (
        DerhamOptions,
        LoadingParameters,
        Simulation,
        domains,
        equils,
        grids,
        maxwellians,
        models,
        perturbations,
    )

    vth = 1e-6
    model = models.DriftKineticMaxwell(epsilon=0.5, lam=2.0, kappa=2.0)
    model.kinetic_ions.set_markers(
        loading_params=LoadingParameters(ppc=5, seed=1234, moments=(1.0, 0.0, vth, vth), B0=1.0)
    )
    model.kinetic_ions.var.add_background(
        maxwellians.GyroMaxwellian2D(
            n=(1.0, None), u_para=(1.0, None), vth_para=(vth, None), vth_perp=(vth, None), B0=1.0
        )
    )
    model.em_fields.b_field.add_perturbation(
        perturbations.ModesCos(ls=(0,), ms=(0,), ns=(0,), amps=(DB,), given_in_basis="physical", comp=0)
    )
    prop = model.propagators.coupling
    prop.options = prop.Options(**coupling_options)
    Simulation(
        model=model,
        domain=domains.Cuboid(r1=LX, r2=LY, r3=LZ),
        equil=equils.HomogenSlab(B0z=1.0),
        grid=grids.TensorProductGrid(num_elements=(4, 4, 4)),
        derham_opts=DerhamOptions(degree=(2, 2, 2)),
    ).allocate()
    return model


def _ids_and_positions(particles):
    """Marker ids and logical positions of all valid markers on all processes, sorted by id."""
    from maybempi import MPI

    local = xp.vstack((particles.marker_ids, particles.markers[:3, particles.valid_mks]))
    comm = MPI.COMM_WORLD
    if comm.Get_size() > 1:
        local = xp.concatenate(comm.allgather(local), axis=1)
    return local[:, xp.argsort(local[0])]


@pytest.mark.parametrize("algo", ["discrete_gradient_1st_order", "explicit"])
def test_push_parallel_along_full_b(algo):
    """PushGuidingCenterParallel with b_tilde streams along B* = B0 + B~ (+ curvature term): with B~ = DB e_x on
    B0 = e_z the guiding centers move in x by DB times their displacement in z (before, B* had B0 only)."""
    model = _model()
    prop = model.propagators.push_para
    prop.options = prop.Options(algo=algo)
    prop.allocate()

    particles = model.kinetic_ions.var.particles
    before = _ids_and_positions(particles)
    assert len(xp.unique(before[0])) == before.shape[1]
    prop(0.05)
    after = _ids_and_positions(particles)
    assert xp.array_equal(before[0], after[0])

    # displacements (periodic, unwrapped) in physical units
    d_eta = (after[1:] - before[1:] + 0.5) % 1.0 - 0.5
    dx, dz = LX * d_eta[0], LZ * d_eta[2]
    assert xp.min(xp.abs(dz)) > 1e-3
    assert xp.max(xp.abs(dx - DB * dz)) < 1e-10 * xp.max(xp.abs(dz))
    assert xp.max(xp.abs(LY * d_eta[1])) < 1e-12


def test_coupling_parallel_direction_full_b():
    """The parallel response of DriftKineticMaxwellCoupling uses the same B* = B0 + B~ (+ curvature term): the
    stored a_p = B*2 / B*3_par (logical components, matrix-free) points along the total field."""
    model = _model(matrix_free=True)
    prop = model.propagators.coupling
    prop._update_b_star()
    prop._accumulate()

    particles = model.kinetic_ions.var.particles
    column = particles.args_markers.first_free_idx
    a = particles.markers[column : column + 3, particles.valid_mks]
    # physical B* = (DB, 0, 1), |b0 . B*| = 1: logical vector components DF^{-1} B* = (DB / LX, 0, 1 / LZ)
    assert xp.allclose(a[0], DB / LX, rtol=1e-10)
    assert xp.allclose(a[1], 0.0, atol=1e-12)
    assert xp.allclose(a[2], 1.0 / LZ, rtol=1e-10)
