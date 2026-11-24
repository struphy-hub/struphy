import pytest

from struphy.propagators.jxb_cold import JxBCold


@pytest.mark.mpi_skip
def test_jxb_cold_solver_options():
    """The Crank-Nicolson system of JxBCold is not symmetric: the default is a solver for general matrices, and the
    symmetric solvers of older parameter files are replaced (pcg diverged for dt > epsilon, issue #856)."""
    assert JxBCold.Options().solver == "gmres"
    for solver in ("pcg", "pcg_sr", "cg"):
        assert JxBCold.Options(solver=solver).solver == "gmres"
    for solver in ("bicgstab", "pbicgstab"):
        assert JxBCold.Options(solver=solver).solver == solver
    with pytest.raises(AssertionError):
        JxBCold.Options(solver="minres")


def test_jxb_cold_default_solver_converges(tmp_path):
    """With the default options, JxBCold converges in a few iterations in every step of the default ColdPlasma run
    (noise perturbation, the setup of the model regression test), also after small changes of the current, and
    conserves j^T M1ninv j (Crank-Nicolson with a skew-symmetric matrix). With pbicgstab and the mass matrix
    preconditioner, the number of iterations jumps between 42 and 393 here (and the solver did not converge in
    3000 iterations after a roundoff-level change in an earlier test)."""
    import numpy as np

    from struphy import EnvironmentOptions, Simulation
    from struphy.io.setup import import_parameters_py
    from struphy.models import ColdPlasma
    from struphy.models.tests.regression import utils_regression as ur

    path = str(tmp_path / "params_ColdPlasma.py")
    ColdPlasma().generate_default_parameter_file(path=path, prompt=False)
    params = import_parameters_py(path)
    model = params.model
    ur.set_seeds(model)
    Simulation(
        model=model,
        params_path=path,
        env=EnvironmentOptions(out_folders=str(tmp_path), sim_folder="cold_plasma"),
        time_opts=params.time_opts,
        domain=params.domain,
        equil=params.equil,
        grid=params.grid,
        derham_opts=params.derham_opts,
    ).allocate()

    props = model.propagators
    jxb = props.jxb
    dt = params.time_opts.dt

    def check_step():
        j = model.electrons.current.spline.vector
        energy_0 = j.inner(jxb.mass_ops.M1ninv.dot(j))
        jxb(dt)
        info = jxb._solver._info
        j = model.electrons.current.spline.vector
        assert info["success"], info
        # gmres: 27 to 38 iterations; pbicgstab with the mass matrix preconditioner: 42 to 393 (or no convergence)
        assert info["niter"] <= 60, info
        assert abs(j.inner(jxb.mass_ops.M1ninv.dot(j)) - energy_0) < 1e-6 * energy_0

    # steps of the model (Lie-Trotter: maxwell, ohm, jxb)
    for _ in range(3):
        props.maxwell(dt)
        props.ohm(dt)
        check_step()

    # roundoff-level and small changes of the current
    rng = np.random.default_rng(1)
    for eps in (1e-12, 1e-10, 1e-8, 1e-6):
        for _ in range(3):
            j = model.electrons.current.spline.vector
            for blk in j.blocks:
                blk._data[:] *= 1.0 + eps * rng.standard_normal(blk._data.shape)
            j.update_ghost_regions()
            check_step()
