import pytest

from struphy.models.tests import utils_testing as ut


# specific tests
@pytest.mark.models
@pytest.mark.toy
@pytest.mark.parametrize("model", ut.toy_models)
def test_toy(
    model: str,
    vrbose: bool,
    nclones: int,
    show_plots: bool,
):
    ut.call_test(model_name=model, module=ut.toy, verbose=vrbose)
    # print("test 1")

@pytest.mark.models
@pytest.mark.fluid
@pytest.mark.parametrize("model", ut.fluid_models)
def test_fluid(
    model: str,
    vrbose: bool,
    nclones: int,
    show_plots: bool,
):
    # ut.call_test(model_name=model, module=ut.fluid, verbose=vrbose)
    print("test 2")

@pytest.mark.models
@pytest.mark.kinetic
@pytest.mark.parametrize("model", ut.kinetic_models)
def test_kinetic(
    model: str,
    vrbose: bool,
    nclones: int,
    show_plots: bool,
):
    # ut.call_test(model_name=model, module=ut.kinetic, verbose=vrbose)
    print("test 3")

@pytest.mark.models
@pytest.mark.hybrid
@pytest.mark.parametrize("model", ut.hybrid_models)
def test_hybrid(
    model: str,
    vrbose: bool,
    nclones: int,
    show_plots: bool,
):
    # ut.call_test(model_name=model, module=ut.hybrid, verbose=vrbose)
    print("test 4")

@pytest.mark.single
def test_single_model(
    model_name: str,
    vrbose: bool,
    nclones: int,
    show_plots: bool,
):
    # ut.call_test(model_name=model_name, module=None, verbose=vrbose)
    print("test 5")

if __name__ == "__main__":
    test_toy("Maxwell")
    test_fluid("LinearMHD")
