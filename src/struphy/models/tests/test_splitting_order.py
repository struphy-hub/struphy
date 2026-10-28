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
    with pytest.raises(AssertionError):
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
