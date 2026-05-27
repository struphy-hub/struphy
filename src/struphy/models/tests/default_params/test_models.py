import logging

import pytest

import struphy.models.utils as models_utils
from struphy.models.base import StruphyModel
from struphy.models.tests import utils_testing as ut

logger = logging.getLogger("struphy")


@pytest.mark.models
def test_all_models_expose_new_doc_api():
    doc_methods = (
        "doc_pde",
        "doc_normalization",
        "doc_scalar_quantities",
        "doc_discretization",
        "doc_long_description",
        "doc_examples",
        "doc_use_cases",
        "doc_cannot_be_used_for",
    )

    for model_cls in StruphyModel:
        for method_name in doc_methods:
            method = getattr(model_cls, method_name, None)
            assert method is not None, f"{model_cls.__name__} is missing {method_name}"

            if method_name == "doc_discretization":
                content = method()
            else:
                content = method.__doc__

            assert isinstance(content, str) and content.strip(), (
                f"{model_cls.__name__}.{method_name} did not provide documentation content"
            )


# specific tests


@pytest.mark.models
@pytest.mark.toy
def test_toy(test_profiling: bool = True):
    for model in models_utils.get_models(model_type="Toy"):
        ut.call_test(
            model=model(),
            test_profiling=test_profiling,
        )


@pytest.mark.models
@pytest.mark.fluid
def test_fluid(test_profiling: bool = True):
    for model in models_utils.get_models(model_type="Fluid"):
        ut.call_test(
            model=model(),
            test_profiling=test_profiling,
        )


@pytest.mark.models
@pytest.mark.kinetic
def test_kinetic(
    test_profiling: bool = True):
    for model in models_utils.get_models(model_type="Kinetic"):
        ut.call_test(
            model=model(),
            test_profiling=test_profiling,
        )


@pytest.mark.models
@pytest.mark.hybrid
def test_hybrid(test_profiling: bool = True):
    for model in models_utils.get_models(model_type="Hybrid"):
        ut.call_test(
            model=model(),
            test_profiling=test_profiling,
        )


@pytest.mark.single
def test_single_model(
    model_name: str,
    test_profiling: bool = True,
):
    logger.info(f"{model_name = }")
    model = models_utils.get_model_by_name(model_name=model_name)
    ut.call_test(
        model=model(),
        test_profiling=test_profiling,
    )


if __name__ == "__main__":
    test_toy()
    test_fluid()
    test_single_model("Maxwell")
