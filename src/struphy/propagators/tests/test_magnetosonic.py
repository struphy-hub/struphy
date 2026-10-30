import cunumpy as xp
import pytest
from feectools.linalg.solvers import inverse

from struphy import EnvironmentOptions, FieldsBackground, Simulation, Time, domains, equils, perturbations
from struphy.feec.preconditioner import MassMatrixPreconditioner
from struphy.io.options import DerhamOptions
from struphy.models import LinearMHD
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.propagators.magnetosonic import Magnetosonic
from struphy.topology import grids


def _magnetosonic(model):
    """The magnetosonic step of the variables of an allocated LinearMHD model (which uses the split sound wave and
    background steps instead)."""
    mag_sonic = Magnetosonic(b_field=model.em_fields.b_field)
    mag_sonic.variables.n = model.mhd.density
    mag_sonic.variables.u = model.mhd.velocity
    mag_sonic.variables.p = model.mhd.pressure
    mag_sonic.allocate()
    return mag_sonic


@pytest.mark.parametrize("algo", ["implicit", "explicit"])
@pytest.mark.parametrize("with_equil_background", [False, True])
def test_magnetosonic_uses_shared_b_field(tmp_path, algo, with_equil_background):
    """Magnetosonic must not re-allocate the model's shared b_field (issue #429)."""
    model = LinearMHD()
    model.propagators.shear_alf.options = model.propagators.shear_alf.Options(algo=algo)
    model.mhd.velocity.add_perturbation(perturbations.ModesCos(ls=(1,), given_in_basis="2", comp=0, amps=(1e-2,)))
    model.em_fields.b_field.add_perturbation(perturbations.ModesCos(ls=(1,), given_in_basis="2", comp=1, amps=(1e-2,)))
    if with_equil_background:
        # needs equil at allocation time
        model.em_fields.b_field.add_background(FieldsBackground(type="FluidEquilibrium", variable="b2"))

    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(tmp_path), sim_folder="sim"),
        time_opts=Time(dt=0.05, Tend=0.05),
        equil=equils.HomogenSlab(),
        grid=grids.TensorProductGrid(num_elements=(8, 1, 1)),
        derham_opts=DerhamOptions(degree=(2, 1, 1)),
    )
    sim.allocate()
    mag_sonic = _magnetosonic(model)

    b = model.em_fields.b_field.spline.vector
    shear_alf = model.propagators.shear_alf
    assert mag_sonic._b is b
    assert shear_alf.variables.b.spline.vector is b
    if algo == "explicit":
        assert any(v is b for v in shear_alf._ode_solver.vector_field)

    b0 = b.toarray().copy()
    for prop in (shear_alf, mag_sonic):
        prop(0.05)
    assert xp.max(xp.abs(model.em_fields.b_field.spline.vector.toarray() - b0)) > 0.0


@pytest.mark.mpi_skip
def test_b_field_space():
    """Magnetosonic accepts a magnetic field in Hdiv or Hcurl only."""
    Magnetosonic(b_field=FEECVariable(space="Hcurl"))
    with pytest.raises(AssertionError):
        Magnetosonic(b_field=FEECVariable(space="L2"))


def _lorentz_force_difference(tmp_path, num_elements):
    """L2 norms of the equilibrium force J0 x B0 (projected onto Hdiv) computed with M2J from the 2-form B0 and of its
    difference to M21J applied to the 1-form B0, in a torus with J0 != 0."""
    model = LinearMHD(b_space="Hcurl")
    model.propagators.shear_alf.options = model.propagators.shear_alf.Options(precond="MassMatrixPreconditioner")
    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(tmp_path), sim_folder=f"sim_{num_elements[0]}"),
        time_opts=Time(dt=0.1, Tend=0.1),
        domain=domains.HollowTorus(a1=0.1, a2=1.0, R0=3.0, tor_period=1),
        equil=equils.AdhocTorus(a=1.0, R0=3.0, q0=1.05, q1=1.8, beta=0.1),
        grid=grids.TensorProductGrid(num_elements=num_elements),
        derham_opts=DerhamOptions(degree=(2, 2, 1), bcs=(("free", "free"), None, None)),
    )
    sim.allocate()

    mass_ops = Propagator.mass_ops
    equil = Propagator.projected_equil
    assert _magnetosonic(model)._MJ is mass_ops.M21J

    M2_inv = inverse(mass_ops.M2, "pcg", pc=MassMatrixPreconditioner(mass_ops.M2), tol=1e-12, maxiter=1000)
    f2 = M2_inv.dot(mass_ops.M2J.dot(equil.b2))
    f1 = M2_inv.dot(mass_ops.M21J.dot(equil.b1))
    diff = f2 - f1
    return xp.sqrt(f2.inner(mass_ops.M2.dot(f2))), xp.sqrt(diff.inner(mass_ops.M2.dot(diff)))


def test_mixed_lorentz_force(tmp_path):
    """M21J (Lorentz force of a 1-form) agrees with M2J (of a 2-form) for the same field, up to the projection error,
    which decreases under mesh refinement (relative errors 5e-4 and 3e-5)."""
    errors = []
    for num_elements in ((8, 8, 1), (16, 16, 1)):
        norm, error = _lorentz_force_difference(tmp_path, num_elements)
        assert norm > 1.0
        errors.append(error / norm)
    assert errors[0] < 1e-3
    assert errors[1] < errors[0] / 8
