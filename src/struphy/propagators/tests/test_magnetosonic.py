import cunumpy as xp
import pytest

from struphy import EnvironmentOptions, FieldsBackground, Simulation, Time, equils, perturbations
from struphy.io.options import DerhamOptions
from struphy.models import LinearMHD
from struphy.topology import grids


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

    b = model.em_fields.b_field.spline.vector
    shear_alf = model.propagators.shear_alf
    assert model.propagators.mag_sonic._b is b
    assert shear_alf.variables.b.spline.vector is b
    if algo == "explicit":
        assert any(v is b for v in shear_alf._ode_solver.vector_field)

    b0 = b.toarray().copy()
    for prop in model.prop_list:
        prop(0.05)
    assert xp.max(xp.abs(model.em_fields.b_field.spline.vector.toarray() - b0)) > 0.0
