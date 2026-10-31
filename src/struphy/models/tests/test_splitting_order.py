import pytest

from struphy.models.base import apply_splitting_order


class _Prop:
    """Stand-in for a propagator (only the order of the attributes is tested)."""


class _Parent:
    class Propagators:
        def __init__(self):
            self.push_a = _Prop()
            self.push_b = _Prop()


class _Child:
    class Propagators(_Parent.Propagators):
        splitting_order = ("push_c", "push_b", "push_a")

        def __init__(self):
            super().__init__()
            self.push_c = _Prop()


@pytest.mark.mpi_skip
def test_no_splitting_order_keeps_attribute_order():
    props = _Parent.Propagators()
    apply_splitting_order(props)
    assert list(vars(props)) == ["push_a", "push_b"]


@pytest.mark.mpi_skip
def test_splitting_order_as_class_attribute():
    props = _Child.Propagators()
    a, b, c = props.push_a, props.push_b, props.push_c
    apply_splitting_order(props)
    assert list(vars(props)) == ["push_c", "push_b", "push_a"]
    assert list(vars(props).values()) == [c, b, a]


@pytest.mark.mpi_skip
def test_splitting_order_set_in_init():
    class Propagators(_Parent.Propagators):
        def __init__(self, with_c_first: bool):
            super().__init__()
            self.push_c = _Prop()
            if with_c_first:
                self.splitting_order = ("push_c", "push_a", "push_b")

    props = Propagators(with_c_first=True)
    apply_splitting_order(props)
    assert list(vars(props)) == ["push_c", "push_a", "push_b"]
    assert not hasattr(props, "splitting_order")

    props = Propagators(with_c_first=False)
    apply_splitting_order(props)
    assert list(vars(props)) == ["push_a", "push_b", "push_c"]


@pytest.mark.mpi_skip
@pytest.mark.parametrize(
    "order",
    [("push_a", "push_b"), ("push_a", "push_b", "push_c", "push_d"), ("push_a", "push_a", "push_c")],
)
def test_invalid_splitting_order(order):
    props = _Child.Propagators()
    props.splitting_order = order
    with pytest.raises(ValueError):
        apply_splitting_order(props)


@pytest.mark.mpi_skip
def test_model_propagators_in_splitting_order():
    from struphy.models import VlasovAmpereOneSpecies, VlasovMaxwellOneSpecies

    model = VlasovMaxwellOneSpecies()
    assert list(vars(model.propagators)) == ["maxwell", "push_eta", "push_vxb", "coupling_va"]
    assert model.prop_list == list(vars(model.propagators).values())
    assert list(VlasovAmpereOneSpecies.propagators_in_splitting_order()) == ["push_eta", "push_vxb", "coupling_va"]

    # the documentation lists the propagators in splitting order
    links = VlasovMaxwellOneSpecies.propagators_links().splitlines()
    assert [line.split("``")[1] for line in links] == ["maxwell", "push_eta", "push_vxb", "coupling_va"]


@pytest.mark.mpi_skip
def test_removed_model_flags_in_old_metadata():
    from struphy.models import VlasovAmpereOneSpecies
    from struphy.models.base import _drop_removed_model_flags

    params = {"alpha": 1.0, "with_B0": False}
    _drop_removed_model_flags("VlasovAmpereOneSpecies", params)
    assert params == {"alpha": 1.0}
    VlasovAmpereOneSpecies(**params)

    params = {"with_poisson": True}
    _drop_removed_model_flags("VlasovPoissonAdiabaticPauli", params)
    assert params == {}

    with pytest.raises(ValueError, match="VlasovPauli"):
        _drop_removed_model_flags("VlasovPoissonAdiabaticPauli", {"with_poisson": False})


@pytest.mark.mpi_skip
def test_renamed_propagators_in_old_metadata():
    from struphy.models import DriftKineticMaxwell
    from struphy.models.base import StruphyModel

    model = DriftKineticMaxwell()
    model.propagators.push_parallel.options = model.propagators.push_parallel.Options(algo="explicit")
    dct = model.to_dict()

    # metadata written before push_para was renamed to push_parallel
    dct["propagator_options"]["push_para"] = dct["propagator_options"].pop("push_parallel")
    restored = StruphyModel.from_dict(dct)
    assert restored.propagators.push_parallel.options.algo == "explicit"


@pytest.mark.mpi_skip
def test_renamed_propagators_of_dkea_in_old_metadata():
    from struphy.models import DriftKineticElectrostaticAdiabatic
    from struphy.models.base import StruphyModel

    model = DriftKineticElectrostaticAdiabatic()
    model.propagators.push_bxe.options = model.propagators.push_bxe.Options(algo="explicit")
    model.propagators.push_parallel.options = model.propagators.push_parallel.Options(algo="explicit")
    dct = model.to_dict()

    # metadata written before push_gc_bxe and push_gc_para were renamed
    options = dct["propagator_options"]
    options["push_gc_bxe"] = options.pop("push_bxe")
    options["push_gc_para"] = options.pop("push_parallel")
    restored = StruphyModel.from_dict(dct)
    assert restored.propagators.push_bxe.options.algo == "explicit"
    assert restored.propagators.push_parallel.options.algo == "explicit"


@pytest.mark.mpi_skip
def test_unmatched_propagator_options_warn(caplog):
    import logging

    from struphy.models import GuidingCenter
    from struphy.models.base import StruphyModel

    dct = GuidingCenter().to_dict()
    dct["propagator_options"]["push_unknown"] = dct["propagator_options"]["push_bxe"]
    with caplog.at_level(logging.WARNING, logger="struphy"):
        StruphyModel.from_dict(dct)
    assert "push_unknown" in caplog.text


@pytest.mark.mpi_skip
def test_linear_euler_family_splitting_order():
    from struphy.models import LinearEuler, LinearMHD

    assert list(LinearEuler.propagators_in_splitting_order()) == ["bg_pressure", "sound"]
    model = LinearMHD()
    assert isinstance(model, LinearEuler)
    assert list(vars(model.propagators)) == ["shear_alf", "bg_current", "bg_pressure", "sound"]
    assert model.propagators.bg_pressure.u_field is model.mhd.velocity
    assert model.propagators.bg_current.b_field is model.em_fields.b_field

    # the documentation lists both shear-Alfven flavours
    doc = LinearMHD.doc_discretization()
    assert "ShearAlfvenPropagator" in doc and "ShearAlfvenCurlCurlPropagator" in doc


@pytest.mark.mpi_skip
@pytest.mark.parametrize("split_sound", [True, False])
def test_linear_mhd_split_sound_in_old_metadata(split_sound):
    from struphy.models import LinearMHD
    from struphy.models.base import StruphyModel

    model = LinearMHD(b_space="Hcurl")
    model.propagators.sound.options = model.propagators.sound.Options(solver="cg")
    dct = model.to_dict()

    # metadata written when LinearMHD had the flag split_sound (and the unsplit magnetosonic step mag_sonic)
    dct["params"]["split_sound"] = split_sound
    if not split_sound:
        for name in ("bg_current", "sound", "bg_pressure"):
            dct["propagator_options"].pop(name)
        dct["propagator_options"]["mag_sonic"] = {}
    restored = StruphyModel.from_dict(dct)
    assert "split_sound" not in restored.params
    assert list(vars(restored.propagators)) == ["shear_alf", "bg_current", "bg_pressure", "sound"]
    assert restored.em_fields.b_field.space == "Hcurl"
    assert restored.propagators.sound.options.solver == ("cg" if split_sound else "pcg")


@pytest.mark.mpi_skip
def test_linear_mhd_hybrids_inherit_linear_mhd():
    from struphy.models import LinearMHD, LinearMHDDriftkineticCC, LinearMHDVlasovCC, LinearMHDVlasovPC
    from struphy.propagators.shear_alfven_current_coupling_5d import ShearAlfvenCurrentCoupling5D

    for model_class in (LinearMHDVlasovCC, LinearMHDVlasovPC, LinearMHDDriftkineticCC):
        model = model_class(mhd_mass_number=2.0, hot_mass_number=3.0)
        assert isinstance(model, LinearMHD)
        assert model.mhd.mass_number == 2.0 and model.energetic_ions.mass_number == 3.0
        assert model.em_fields.b_field.space == "Hdiv"
        names = list(vars(model.propagators))
        for name in ("shear_alf", "bg_current", "sound", "bg_pressure"):
            assert name in names, (model_class.__name__, name)
        assert names.index("bg_current") + 1 == names.index("bg_pressure") == names.index("sound") - 1
        assert {"en_B", "en_thermal", "en_tot"} <= set(model.scalars._dct)
    assert isinstance(LinearMHDDriftkineticCC().propagators.shear_alf, ShearAlfvenCurrentCoupling5D)


@pytest.mark.mpi_skip
def test_linear_mhd_hybrids_in_old_metadata():
    from struphy.models import LinearMHDDriftkineticCC, LinearMHDVlasovPC
    from struphy.models.base import StruphyModel

    # LinearMHDVlasovPC: turn_off (default) and the propagators shearalfven, magnetosonic of old metadata
    model = LinearMHDVlasovPC()
    model.propagators.shear_alf.options = model.propagators.shear_alf.Options(algo="explicit")
    dct = model.to_dict()
    dct["params"]["turn_off"] = [None]
    dct["propagator_options"]["shearalfven"] = dct["propagator_options"].pop("shear_alf")
    dct["propagator_options"]["magnetosonic"] = {}
    restored = StruphyModel.from_dict(dct)
    assert "turn_off" not in restored.params
    assert restored.propagators.shear_alf.options.algo == "explicit"

    # switched off propagators can not be restored
    dct["params"]["turn_off"] = ["Magnetosonic"]
    with pytest.raises(ValueError, match="turn_off"):
        StruphyModel.from_dict(dct)

    # LinearMHDDriftkineticCC: shearalfen_cc5d -> shear_alf
    model = LinearMHDDriftkineticCC()
    model.propagators.shear_alf.options = model.propagators.shear_alf.Options(nonlinear=False)
    dct = model.to_dict()
    dct["propagator_options"]["shearalfen_cc5d"] = dct["propagator_options"].pop("shear_alf")
    restored = StruphyModel.from_dict(dct)
    assert not restored.propagators.shear_alf.options.nonlinear


@pytest.mark.mpi_skip
def test_maxwell_family():
    from struphy.models import ColdPlasma, ColdPlasmaVlasov, Maxwell

    assert list(Maxwell().propagators.__dict__) == ["maxwell"]
    model = ColdPlasmaVlasov(charge_number=-2, mass_number=0.25, alpha=3.0, epsilon=0.5, hot_epsilon=0.75)
    assert isinstance(model, ColdPlasma) and isinstance(model, Maxwell)
    assert list(vars(model.propagators)) == ["maxwell", "ohm", "jxb", "push_eta", "push_vxb", "coupling_va"]
    assert list(model.species) == ["em_fields", "electrons", "hot_electrons"]
    assert set(model.em_fields.variables) == {"e_field", "b_field", "phi"}
    assert model.bulk_species is model.electrons
    assert model.electrons.equation_params.alpha == 3.0 and model.electrons.equation_params.epsilon == 0.5
    assert model.hot_electrons.equation_params.epsilon == 0.75
    assert list(model.scalars._dct) == ["en_E", "en_B", "en_J", "en_f", "en_tot"]
    assert list(ColdPlasma().scalars._dct) == ["en_E", "en_B", "en_J", "en_tot"]


@pytest.mark.mpi_skip
def test_cold_plasma_vlasov_in_old_metadata():
    from struphy.models import ColdPlasmaVlasov
    from struphy.models.base import StruphyModel

    model = ColdPlasmaVlasov(charge_number=-2, alpha=3.0)
    model.electrons.current.save_data = False
    dct = model.to_dict()

    # metadata written before the Maxwell family: thermal_* arguments, species thermal_elec and hot_elec
    for name in ("charge_number", "mass_number", "alpha", "epsilon"):
        dct["params"][f"thermal_{name}"] = dct["params"].pop(name)
    dct["species"]["thermal_elec"] = dct["species"].pop("electrons")
    dct["species"]["hot_elec"] = dct["species"].pop("hot_electrons")
    restored = StruphyModel.from_dict(dct)
    assert restored.params["charge_number"] == -2 and restored.params["alpha"] == 3.0
    assert restored.electrons.current.save_data is False


@pytest.mark.mpi_skip
def test_sph_family():
    from struphy.models import ColdEulerSPH, EulerSPH, IncompressibleNavierStokesSPH, ViscousEulerSPH

    assert list(ColdEulerSPH().propagators.__dict__) == ["push_eta", "push_vxb", "push_v"]
    model = ViscousEulerSPH()
    assert isinstance(model, EulerSPH) and isinstance(model, ColdEulerSPH)
    assert list(vars(model.propagators)) == ["push_eta", "push_vxb", "push_v", "push_sph_p", "push_viscous"]
    assert list(model.species) == ["fluid"]
    model = IncompressibleNavierStokesSPH()
    assert not isinstance(model, EulerSPH)
    assert list(vars(model.propagators)) == [
        "push_eta",
        "push_vxb",
        "push_v",
        "push_viscous",
        "pressure_poisson",
        "chorin_projection",
    ]
    assert model.fluid.equation_params.epsilon == 1.0
    assert model.propagators.chorin_projection.potential is model.lagrange_multiplier.pressure


@pytest.mark.mpi_skip
def test_sph_models_in_old_metadata():
    from struphy.models import ColdEulerSPH, EulerSPH, ViscousEulerSPH
    from struphy.models.base import StruphyModel

    # PressureLessSPH was renamed to ColdEulerSPH, its species cold_fluid to fluid
    model = ColdEulerSPH(epsilon=0.5)
    model.fluid.var.save_data = False
    dct = model.to_dict()
    dct["model"] = "PressureLessSPH"
    dct["species"]["cold_fluid"] = dct["species"].pop("fluid")
    restored = StruphyModel.from_dict(dct)
    assert type(restored) is ColdEulerSPH
    assert restored.params["epsilon"] == 0.5 and restored.fluid.var.save_data is False

    # ViscousEulerSPH: flags with_B0, with_p and with_viscosity, species euler_fluid
    model = ViscousEulerSPH()
    model.propagators.push_viscous.options = model.propagators.push_viscous.Options(mu=0.5)
    dct = model.to_dict()
    dct["params"].update(with_B0=False, with_p=True, with_viscosity=True)
    dct["species"]["euler_fluid"] = dct["species"].pop("fluid")
    restored = StruphyModel.from_dict(dct)
    assert not {"with_B0", "with_p", "with_viscosity"} & set(restored.params)
    assert restored.propagators.push_viscous.options.mu == 0.5
    assert isinstance(EulerSPH(), ColdEulerSPH)

    # terms that were switched off can not be restored
    for flag in ("with_p", "with_viscosity"):
        dct["params"][flag] = False
        with pytest.raises(ValueError, match=flag):
            StruphyModel.from_dict(dct)
        dct["params"][flag] = True


@pytest.mark.mpi_skip
def test_strang_groups_calls():
    """A group of propagators is integrated symmetrically (last member in the middle) as one block of the splitting."""
    from struphy.models import LinearEuler, LinearMHD
    from struphy.models.base import time_steps_of_blocks

    blocks = [("a",), ("b", "c", "d"), ("e",)]
    lie = [(name, h) for name, h, _ in time_steps_of_blocks(blocks, 1.0, "LieTrotter")]
    assert lie == [("a", 1.0), ("b", 0.5), ("c", 0.5), ("d", 1.0), ("c", 0.5), ("b", 0.5), ("e", 1.0)]
    strang = [(name, h) for name, h, _ in time_steps_of_blocks(blocks, 1.0, "Strang")]
    assert strang == [
        ("a", 0.5),
        ("b", 0.25),
        ("c", 0.25),
        ("d", 0.5),
        ("c", 0.25),
        ("b", 0.25),
        ("e", 1.0),
        ("b", 0.25),
        ("c", 0.25),
        ("d", 0.5),
        ("c", 0.25),
        ("b", 0.25),
        ("a", 0.5),
    ]
    # the time steps of each block sum to the time step
    for algo in ("LieTrotter", "Strang"):
        calls = time_steps_of_blocks(blocks, 1.0, algo)
        assert sum(h for name, h, _ in calls if name == "d") == 1.0

    assert LinearMHD().propagator_calls(1.0) == [
        ("shear_alf", 1.0),
        ("bg_current", 0.5),
        ("bg_pressure", 0.5),
        ("sound", 1.0),
        ("bg_pressure", 0.5),
        ("bg_current", 0.5),
    ]
    # LinearEuler is a single symmetric block: second order with either splitting algorithm
    model = LinearEuler()
    assert model.propagator_calls(1.0, "LieTrotter") == model.propagator_calls(1.0, "Strang")


@pytest.mark.mpi_skip
def test_invalid_strang_groups():
    from struphy.models.base import apply_splitting_order
    from struphy.propagators.push_eta import PushEta
    from struphy.propagators.push_vxb import PushVxB

    class NotConsecutive:
        strang_groups = (("push_a", "push_c"),)

        def __init__(self):
            self.push_a = PushEta()
            self.push_b = PushVxB()
            self.push_c = PushEta()

    class UnknownName:
        strang_groups = (("push_a", "push_x"),)

        def __init__(self):
            self.push_a = PushEta()
            self.push_b = PushVxB()

    class SingleMember:
        strang_groups = (("push_a",),)

        def __init__(self):
            self.push_a = PushEta()
            self.push_b = PushVxB()

    class InInit:
        def __init__(self):
            self.push_a = PushEta()
            self.push_b = PushVxB()
            self.strang_groups = (("push_a", "push_b"),)

    for cls in (NotConsecutive, UnknownName, SingleMember, InInit):
        with pytest.raises(ValueError, match="strang_groups"):
            apply_splitting_order(cls())


@pytest.mark.mpi_skip
def test_unknown_split_algo():
    from struphy.models import LinearEuler

    # LinearEuler is a single block, an unknown splitting algorithm must still raise
    with pytest.raises(NotImplementedError):
        LinearEuler().propagator_calls(1.0, "strang")
