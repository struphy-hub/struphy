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

    assert list(LinearEuler.propagators_in_splitting_order()) == ["sound", "bg_pressure"]
    model = LinearMHD()
    assert isinstance(model, LinearEuler)
    assert list(vars(model.propagators)) == ["shear_alf", "bg_current", "sound", "bg_pressure"]
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
    assert list(vars(restored.propagators)) == ["shear_alf", "bg_current", "sound", "bg_pressure"]
    assert restored.em_fields.b_field.space == "Hcurl"
    assert restored.propagators.sound.options.solver == ("cg" if split_sound else "pcg")
