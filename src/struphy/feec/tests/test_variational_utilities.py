import pytest


@pytest.mark.parametrize(
    "name",
    [
        "evaluate_discrete_de_drho_grid",
        "evaluate_discrete_de_ds_grid",
        "evaluate_discrete_d2e_drho2_grid",
        "evaluate_discrete_d2e_ds2_grid",
    ],
)
def test_internal_energy_evaluator_out(name):
    """The discrete (second) derivatives of InternalEnergyEvaluator return their result, with and without out."""

    import cunumpy as xp
    from feectools.ddm.mpi import mpi as MPI

    from struphy.feec.psydac_derham import Derham
    from struphy.feec.variational_utilities import InternalEnergyEvaluator
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    derham = Derham(
        grid=TensorProductGrid(num_elements=[4, 3, 2]),
        options=DerhamOptions(degree=[2, 2, 1]),
        comm=MPI.COMM_WORLD,
    )
    evaluator = InternalEnergyEvaluator(derham, 5 / 3)

    # the three arguments are 3-forms: (rho, rho1, s) or (rho, s, s1)
    args = []
    for n, val in enumerate([1.0, 1.1, 0.3]):
        f = derham.create_spline_function(f"f{n}", "L2")
        f.vector[:] = val
        f.vector.update_ghost_regions()
        args += [f.vector]

    func = getattr(evaluator, name)

    out = xp.zeros_like(evaluator._tmp_int_grid)
    res = func(*args, out=out)
    assert res is out
    assert xp.all(xp.isfinite(out))

    res_none = func(*args)
    assert res_none is not None
    assert xp.allclose(res_none, out)


if __name__ == "__main__":
    test_internal_energy_evaluator_out("evaluate_discrete_d2e_ds2_grid")
