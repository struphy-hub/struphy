import logging

import cunumpy as xp
import pytest

from struphy import set_logging_level

set_logging_level(logging.WARNING)


def _allocate(model, perturb_density):
    """Same markers for all models (same loading, seed, domain and grid); a density perturbation makes the initial
    potential of the electrostatic model non-trivial."""
    from struphy import DerhamOptions, LoadingParameters, Simulation, domains, equils, grids, maxwellians, perturbations

    model.kinetic_ions.set_markers(loading_params=LoadingParameters(ppc=10, seed=4321))
    model.kinetic_ions.var.add_background(maxwellians.GyroMaxwellian2D(n=(1.0, None), B0=1.0))
    pert = perturbations.ModesCos(ls=(1,), ms=(1,), amps=(0.3,)) if perturb_density else None
    model.kinetic_ions.var.add_initial_condition(maxwellians.GyroMaxwellian2D(n=(1.0, pert), B0=1.0))
    Simulation(
        model=model,
        domain=domains.Cuboid(r1=4.0, r2=4.0, r3=2.0),
        equil=equils.HomogenSlab(B0x=0.2, B0z=1.0),
        grid=grids.TensorProductGrid(num_elements=(6, 6, 2)),
        derham_opts=DerhamOptions(degree=(2, 2, 1)),
    ).allocate()


def _all_markers(particles):
    """Positions and parallel velocities of all valid markers on all processes, sorted lexicographically (markers
    may change their process during a push)."""
    from maybempi import MPI

    local = particles.markers[:4, particles.valid_mks].copy()
    comm = MPI.COMM_WORLD
    if comm.Get_size() > 1:
        local = xp.concatenate(comm.allgather(local), axis=1)
    return local[:, xp.lexsort(local[::-1])]


@pytest.mark.parametrize("name", ["PushGuidingCenterBxEstar", "PushGuidingCenterParallel"])
def test_e_field_equals_minus_grad_phi(name):
    """A guiding-center pusher with ``e_field = -grad phi`` moves the markers exactly as with the potential ``phi``
    (explicit algorithm, with the electric field evaluated from phi)."""
    from struphy import models, propagators

    dt = 0.1
    attr = {"PushGuidingCenterBxEstar": "push_gc_bxe", "PushGuidingCenterParallel": "push_gc_para"}[name]

    # reference: the phi path in the electrostatic drift-kinetic model (phi from its initial Poisson solve);
    # pushed before the second model is allocated (the Derham objects of propagators are class attributes)
    es = models.DriftKineticElectrostaticAdiabatic(epsilon=0.5)
    prop_phi = getattr(es.propagators, attr)
    prop_phi.options = prop_phi.Options(algo="explicit", evaluate_e_field=True)
    _allocate(es, perturb_density=True)
    phi = es.em_fields.phi.spline.vector
    assert phi.inner(phi) > 0.0
    e_data = [blk._data.copy() for blk in es.em_fields.phi.spline.derham.grad.dot(-phi).blocks]
    p_es = es.kinetic_ions.var.particles
    initial = _all_markers(p_es)
    prop_phi(dt)
    pushed_phi = _all_markers(p_es)
    assert xp.max(xp.abs(pushed_phi - initial)) > 1e-6

    # the e_field path, with e = -grad phi copied into the (identically distributed) Derham of the other model
    em = models.DriftKineticMaxwell(epsilon=0.5)
    _allocate(em, perturb_density=True)
    e = em.em_fields.e_field.spline.vector
    for blk, data in zip(e.blocks, e_data):
        blk._data[:] = data
    e.update_ghost_regions()
    prop_e = getattr(propagators, name)(e_field=em.em_fields.e_field)
    prop_e.variables.ions = em.kinetic_ions.var
    prop_e.options = prop_e.Options(algo="explicit")
    prop_e.allocate()

    p_em = em.kinetic_ions.var.particles
    assert xp.array_equal(initial, _all_markers(p_em))
    prop_e(dt)
    pushed_e = _all_markers(p_em)
    assert pushed_e.shape == pushed_phi.shape
    assert xp.max(xp.abs(pushed_e - pushed_phi)) < 1e-12


@pytest.mark.mpi_skip
def test_phi_and_e_field_exclusive():
    from struphy.models.variables import FEECVariable
    from struphy.propagators.push_guiding_center_bx_estar import PushGuidingCenterBxEstar
    from struphy.propagators.push_guiding_center_parallel import PushGuidingCenterParallel

    for cls in (PushGuidingCenterBxEstar, PushGuidingCenterParallel):
        with pytest.raises(AssertionError):
            cls(phi=FEECVariable(space="H1"), e_field=FEECVariable(space="Hcurl"))
        with pytest.raises(AssertionError):
            cls(e_field=FEECVariable(space="H1"))
