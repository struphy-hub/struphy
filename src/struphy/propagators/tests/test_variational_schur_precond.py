"""The Kronecker preconditioner of the shear-Alfven Schur operator of the linear variational propagators
(``schur_precond="StiffnessPreconditioner"`` of VariationalPBEvolve and VariationalQBEvolve): it changes the
iteration, not the solution, and it falls back to the mass-matrix preconditioner where it does not apply."""

import pytest

from struphy.models import VariationalPerturbativeMHD, VariationalPerturbativeMHD_with_q
from struphy.models.tests.verification.variational_mhd_utils import (
    allocate_ideal,
    homogeneous_slab,
    linear,
    norm,
    perturb_velocity,
    state,
)

LZ = 10.0
DT = 0.2
N_STEPS = 3

LINEAR_P = linear(VariationalPerturbativeMHD)
LINEAR_Q_SPLIT = linear(VariationalPerturbativeMHD_with_q, split_sound=True)
LINEAR_Q = linear(VariationalPerturbativeMHD_with_q)


def _run(tmp_path, model_class, schur_precond):
    model = allocate_ideal(
        tmp_path,
        f"{model_class.__name__}_{schur_precond}",
        model_class,
        homogeneous_slab(Lz=LZ),
        DT,
        perturb_velocity(0.1, direction="z", L=LZ),
        schur_precond=schur_precond,
    )
    for prop in model.prop_list:
        prop.prepare_time_step(DT)
    for _ in range(N_STEPS):
        model.integrate(DT)
    return model


@pytest.mark.parametrize(
    "model_class, name",
    [(LINEAR_P, "variat_pb"), (LINEAR_Q_SPLIT, "variat_qb")],
    ids=["linear_p", "linear_q_split"],
)
def test_same_solution(tmp_path, model_class, name):
    """The stiffness preconditioner is active and gives the same solution as the mass-matrix preconditioner (up to the
    solver tolerance)."""
    stiff = _run(tmp_path, model_class, "StiffnessPreconditioner")
    assert getattr(stiff.propagators, name)._schur_pc is not None
    stiff_state = state(stiff)
    mass_state = state(_run(tmp_path, model_class, "MassMatrixPreconditioner"))
    for i, var in ((1, "u"), (2, "p/q"), (3, "B")):
        rel = norm(stiff_state[i] - mass_state[i]) / norm(mass_state[i])
        assert rel < 1e-6, (var, rel)


def test_fallback(tmp_path):
    """Where the Schur operator is not the shear-Alfven one (the unsplit q model), the mass-matrix preconditioner is
    used."""
    model = _run(tmp_path, LINEAR_Q, "StiffnessPreconditioner")
    assert model.propagators.variat_qb._schur_pc is None
