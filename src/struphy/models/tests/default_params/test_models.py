import pytest

import struphy.models.utils as models_utils
from struphy.models.tests import utils_testing as ut

# specific tests


@pytest.mark.models
@pytest.mark.toy
def test_toy(
    vrbose: bool = False,
    nclones: int = 1,
    show_plots: bool = False,
):
    for model in models_utils.get_models(model_type="Toy"):
        ut.call_test(model=model(), verbose=vrbose)


@pytest.mark.models
@pytest.mark.fluid
def test_fluid(
    vrbose: bool = False,
    nclones: int = 1,
    show_plots: bool = False,
):
    for model in models_utils.get_models(model_type="Fluid"):
        ut.call_test(model=model(), verbose=vrbose)


@pytest.mark.models
@pytest.mark.kinetic
def test_kinetic(
    vrbose: bool = False,
    nclones: int = 1,
    show_plots: bool = False,
):
    for model in models_utils.get_models(model_type="Kinetic"):
        ut.call_test(model=model(), verbose=vrbose)


@pytest.mark.models
@pytest.mark.hybrid
def test_hybrid(
    vrbose: bool = False,
    nclones: int = 1,
    show_plots: bool = False,
):
    for model in models_utils.get_models(model_type="Kinetic"):
        ut.call_test(model=model(), verbose=vrbose)


@pytest.mark.single
def test_single_model(
    model_name: str,
    vrbose: bool = False,
    nclones: int = 1,
    show_plots: bool = False,
):
    model = models_utils.get_model_by_name(model_name=model_name)
    ut.call_test(model=model(), verbose=vrbose)


if __name__ == "__main__":
    test_toy()
    test_fluid()
    test_single_model("Maxwell")
