import logging
import os

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
def test_kinetic(test_profiling: bool = True):
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


@pytest.mark.models
@pytest.mark.fluid
def test_variational_mhd_with_p_default_options(tmp_path):
    """The pressure-based model must default to the pressure ('full_p') variants of its propagators."""
    from struphy.io.setup import import_parameters_py
    from struphy.models.variational_mhd_with_p import VariationalMHD_with_p

    path = os.path.join(tmp_path, "params_VariationalMHD_with_p.py")
    VariationalMHD_with_p().generate_default_parameter_file(path=path, prompt=False)
    propagators = import_parameters_py(path).model.propagators

    for name in ("variat_dens", "variat_pb", "variat_viscous", "variat_resist"):
        assert getattr(propagators, name).options.model == "full_p", name


@pytest.mark.models
@pytest.mark.fluid
@pytest.mark.parametrize("perturbative_model", ["VariationalPerturbativeMHD", "VariationalPerturbativeMHD_with_q"])
def test_variational_perturbative_linear(perturbative_model: str, tmp_path, test_profiling: bool = True):
    """The linear version (linear=True) of the perturbative models: the default parameter file keeps the flag and
    sets the linear variants of the propagators, and the generic model test passes."""
    from struphy.io.setup import import_parameters_py

    model_class = models_utils.get_model_by_name(model_name=perturbative_model)
    model = model_class(linear=True)

    path = os.path.join(tmp_path, f"params_{perturbative_model}.py")
    model.generate_default_parameter_file(path=path, prompt=False)
    params_model = import_parameters_py(path).model
    assert params_model.params["linear"]
    assert not hasattr(params_model.propagators, "variat_mom")
    for name, variant in model.propagator_variants.items():
        assert getattr(params_model.propagators, name).options.model == variant, name

    # the q model with split_sound: the flag is kept, and the sound substep gets the linear variant
    if perturbative_model == "VariationalPerturbativeMHD_with_q":
        split = model_class(linear=True, split_sound=True)
        path_split = os.path.join(tmp_path, f"params_{perturbative_model}_split.py")
        split.generate_default_parameter_file(path=path_split, prompt=False)
        split_model = import_parameters_py(path_split).model
        assert split_model.params["split_sound"] and split_model.params["linear"]
        assert split_model.propagators.variat_sound.options.model == "linear_q"

    ut.call_test(model=model, test_profiling=test_profiling)


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
