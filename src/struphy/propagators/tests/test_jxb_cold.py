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
    """With the default options, JxBCold converges in every step of the default ColdPlasma run (noise perturbation,
    the setup of the model regression test), also after roundoff-level changes of the current, and conserves
    j^T M1ninv j (Crank-Nicolson with a skew-symmetric matrix). pbicgstab with the mass matrix preconditioner did not
    converge here after such changes."""
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

    prop = model.propagators.ohm
    jxb = model.propagators.jxb
    j = model.electrons.current.spline.vector
    dt = params.time_opts.dt
    prop(dt)

    rng = np.random.default_rng(1)
    for eps in (0.0, 1e-12, 1e-10):
        # roundoff-level change of the current
        for blk in j.blocks:
            blk._data[:] *= 1.0 + eps * rng.standard_normal(blk._data.shape)
        j.update_ghost_regions()
        energy_0 = j.inner(jxb.mass_ops.M1ninv.dot(j))
        jxb(dt)
        info = jxb._solver._info
        j = model.electrons.current.spline.vector
        assert info["success"], info
        assert info["niter"] < 100, info
        assert abs(j.inner(jxb.mass_ops.M1ninv.dot(j)) - energy_0) < 1e-6 * energy_0
