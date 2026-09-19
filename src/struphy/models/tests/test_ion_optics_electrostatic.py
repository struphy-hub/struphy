from struphy.models import IonOpticsElectrostatic
from struphy.propagators.push_v_in_force_field import PushVinForceField


def test_ion_optics_electrostatic_wires_prescribed_potential_pusher():
    model = IonOpticsElectrostatic(alpha=2.0, epsilon=0.5)

    assert isinstance(model.propagators.push_v, PushVinForceField)
    assert model.propagators.push_v.potential is model.em_fields.phi
    assert model.propagators.push_v.variables.var is model.ions.var
    assert model.propagators.push_eta.variables.var is model.ions.var
    assert model.ions.equation_params.alpha == 2.0
    assert model.ions.equation_params.epsilon == 0.5
