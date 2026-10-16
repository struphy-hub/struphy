import cunumpy as xp
import pytest

from struphy import EnvironmentOptions, Simulation, Time, equils, perturbations
from struphy.io.options import DerhamOptions
from struphy.linear_algebra.solver import SolverParameters
from struphy.models import ShearAlfvenCurlCurl
from struphy.propagators.base import Propagator
from struphy.propagators.shear_alfven_b1 import ShearAlfvenB1
from struphy.propagators.shear_alfven_curl_curl_propagator import ShearAlfvenCurlCurlPropagator
from struphy.topology import grids


def test_options():
    assert ShearAlfvenCurlCurlPropagator.Options().precond == "HiptmairXu"
    with pytest.raises(AssertionError):
        ShearAlfvenCurlCurlPropagator.Options(algo="explicit")
    with pytest.raises(AssertionError):
        ShearAlfvenCurlCurlPropagator.Options(solver="cg")
    options = ShearAlfvenCurlCurlPropagator.Options(algo="explicit", precond="MassMatrixPreconditioner")
    assert options.butcher is not None


def allocate_model(tmp_path, name, options, num_elements=(8, 8, 4), noise=False):
    """A ShearAlfvenCurlCurl model in a slab with an oblique background field and a velocity perturbation (smooth
    modes, or noise in all directions and components), allocated."""
    model = ShearAlfvenCurlCurl()
    model.propagators.shear_alf.options = options
    if noise:
        for comp in range(3):
            model.mhd.velocity.add_perturbation(
                perturbations.Noise(direction="e1e2e3", amp=1e-2, seed=1234 + comp, comp=comp, given_in_basis="2")
            )
    else:
        model.mhd.velocity.add_perturbation(
            perturbations.ModesCos(ls=(1,), ms=(1,), given_in_basis="2", comp=0, amps=(1e-2,))
        )
        model.mhd.velocity.add_perturbation(
            perturbations.ModesCos(ls=(1,), ns=(1,), given_in_basis="2", comp=1, amps=(1e-2,))
        )
    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(tmp_path), sim_folder=name),
        time_opts=Time(dt=0.1, Tend=0.1),
        equil=equils.HomogenSlab(B0x=0.2, B0z=1.0),
        grid=grids.TensorProductGrid(num_elements=num_elements),
        derham_opts=DerhamOptions(degree=(2, 2, 1)),
    )
    sim.allocate()
    return model


def energy(model):
    u = model.mhd.velocity.spline.vector
    b = model.em_fields.b_field.spline.vector
    mass_ops = Propagator.mass_ops
    return 0.5 * u.inner(mass_ops.M2n.dot(u)) + 0.5 * b.inner(mass_ops.M1.dot(b))


def state(model):
    return model.mhd.velocity.spline.vector.toarray(), model.em_fields.b_field.spline.vector.toarray()


@pytest.mark.parametrize(
    "precond, aux_weight", [("HiptmairXu", 1.0), ("HiptmairXu", 0.0), ("MassMatrixPreconditioner", 1.0)]
)
def test_energy_conservation(tmp_path, precond, aux_weight):
    """The implicit step conserves the perturbed energy up to the solver tolerance (also when dt changes)."""
    options = ShearAlfvenCurlCurlPropagator.Options(
        precond=precond, hx_aux_weight=aux_weight, solver_params=SolverParameters(tol=1e-13, maxiter=1000)
    )
    model = allocate_model(tmp_path, f"{precond}_{aux_weight}", options)
    shear_alf = model.propagators.shear_alf
    en0 = energy(model)
    for dt in (0.5, 0.5, 0.2):
        shear_alf(dt)
        if precond == "HiptmairXu":
            assert shear_alf._hx_dt == dt
    assert xp.linalg.norm(model.em_fields.b_field.spline.vector.toarray()) > 0.0
    assert abs(energy(model) - en0) < 1e-10 * en0


def test_same_step_as_shear_alfven_b1(tmp_path):
    """The b-Schur solve (curl-curl, Hiptmair-Xu) gives the same Crank-Nicolson step as the u-Schur solve of
    ShearAlfvenB1 (the same discrete system)."""
    dt = 0.5
    options = ShearAlfvenCurlCurlPropagator.Options(solver_params=SolverParameters(tol=1e-13, maxiter=1000))
    model = allocate_model(tmp_path, "b1", options)
    u0, b0 = (v.copy() for v in (model.mhd.velocity.spline.vector, model.em_fields.b_field.spline.vector))

    model.propagators.shear_alf(dt)
    u_cc, b_cc = state(model)

    # same step with ShearAlfvenB1 from the same initial state
    u0.copy(out=model.mhd.velocity.spline.vector)
    b0.copy(out=model.em_fields.b_field.spline.vector)
    b1 = ShearAlfvenB1()
    b1.variables.u = model.mhd.velocity
    b1.variables.b = model.em_fields.b_field
    b1.options = ShearAlfvenB1.Options(
        solver_params=SolverParameters(tol=1e-13, maxiter=1000),
        solver_params_M1=SolverParameters(tol=1e-14, maxiter=1000),
    )
    b1.allocate()
    b1(dt)
    u_b1, b_b1 = state(model)

    assert xp.linalg.norm(u_cc - u_b1) < 1e-8 * xp.linalg.norm(u_b1)
    assert xp.linalg.norm(b_cc - b_b1) < 1e-8 * xp.linalg.norm(b_b1)


def test_hiptmair_xu_iterations(tmp_path):
    """Hiptmair-Xu needs fewer PCG iterations than the mass matrix preconditioner for a large time step and rough
    initial data, with the same solution."""
    sols, niter = {}, {}
    for precond in ("HiptmairXu", "MassMatrixPreconditioner"):
        options = ShearAlfvenCurlCurlPropagator.Options(
            precond=precond, solver_params=SolverParameters(tol=1e-10, maxiter=1000)
        )
        model = allocate_model(tmp_path, f"it_{precond}", options, num_elements=(16, 16, 8), noise=True)
        shear_alf = model.propagators.shear_alf
        shear_alf(2.0)
        niter[precond] = shear_alf._schur_solver._solver._info["niter"]
        sols[precond] = state(model)
    print(f"{niter =}")
    assert niter["HiptmairXu"] < niter["MassMatrixPreconditioner"]
    for a, b in zip(sols["HiptmairXu"], sols["MassMatrixPreconditioner"]):
        assert xp.linalg.norm(a - b) < 1e-6 * xp.linalg.norm(b)


def test_explicit(tmp_path):
    """The explicit algorithm agrees with the implicit one for a small time step."""
    dt = 1e-3
    sols = {}
    for algo in ("implicit", "explicit"):
        options = ShearAlfvenCurlCurlPropagator.Options(
            algo=algo,
            precond="HiptmairXu" if algo == "implicit" else "MassMatrixPreconditioner",
            solver_params=SolverParameters(tol=1e-13, maxiter=1000),
        )
        model = allocate_model(tmp_path, algo, options)
        model.propagators.shear_alf(dt)
        sols[algo] = state(model)
    for a, b in zip(sols["implicit"], sols["explicit"]):
        assert xp.linalg.norm(a - b) < 1e-5 * xp.linalg.norm(b)
