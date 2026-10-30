import logging
import os

import pytest

import struphy.models.utils as models_utils
from struphy.models.base import StruphyModel
from struphy.models.tests import utils_testing as ut
from struphy.models.tests.regression import utils_regression as ur

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


def _call_tests(models: list, update_baselines: bool, test_profiling: bool):
    """Test all models with default arguments; report the regression mismatches of all models at the end."""
    mismatches = []
    for model in models:
        try:
            ut.call_test(model=model(), test_profiling=test_profiling, update_baselines=update_baselines)
        except ur.RegressionMismatch as e:
            mismatches.append(str(e))
    if mismatches:
        raise ur.RegressionMismatch("\n\n".join(mismatches))


# specific tests


@pytest.mark.models
@pytest.mark.toy
def test_toy(update_baselines: bool, test_profiling: bool = True):
    _call_tests(models_utils.get_models(model_type="Toy"), update_baselines, test_profiling)


@pytest.mark.models
@pytest.mark.fluid
def test_fluid(update_baselines: bool, test_profiling: bool = True):
    _call_tests(models_utils.get_models(model_type="Fluid"), update_baselines, test_profiling)


@pytest.mark.models
@pytest.mark.kinetic
def test_kinetic(update_baselines: bool, test_profiling: bool = True):
    _call_tests(models_utils.get_models(model_type="Kinetic"), update_baselines, test_profiling)


@pytest.mark.models
@pytest.mark.hybrid
def test_hybrid(update_baselines: bool, test_profiling: bool = True):
    _call_tests(models_utils.get_models(model_type="Hybrid"), update_baselines, test_profiling)


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
@pytest.mark.parametrize("crank_nicolson", [False, True])
def test_variational_perturbative_linear(
    perturbative_model: str, crank_nicolson: bool, tmp_path, update_baselines: bool, test_profiling: bool = True
):
    """The linear version (linear=True) of the perturbative models, with the Picard propagators or with one linear
    solve per substep (crank_nicolson=True): the default parameter file keeps the flags and sets the linear variants
    of the propagators, and the generic model test passes."""
    from struphy.io.setup import import_parameters_py

    model_class = models_utils.get_model_by_name(model_name=perturbative_model)
    model = model_class(linear=True, crank_nicolson=crank_nicolson)

    path = os.path.join(tmp_path, f"params_{perturbative_model}.py")
    model.generate_default_parameter_file(path=path, prompt=False)
    params_model = import_parameters_py(path).model
    assert params_model.params["linear"]
    assert params_model.params["crank_nicolson"] == crank_nicolson
    assert not hasattr(params_model.propagators, "variat_mom")
    assert hasattr(params_model.propagators, "variat_lin_alfven") == crank_nicolson
    for name, variant in model.propagator_variants.items():
        assert getattr(params_model.propagators, name).options.model == variant, name

    # the q model with the unsplit step (split_sound=False): the flag is kept, and VariationalQBEvolve gets the
    # linear variant
    if perturbative_model == "VariationalPerturbativeMHD_with_q":
        if not crank_nicolson:
            assert params_model.propagators.variat_alfven.options.model == "linear"
        unsplit = model_class(linear=True, split_sound=False)
        path_unsplit = os.path.join(tmp_path, f"params_{perturbative_model}_unsplit.py")
        unsplit.generate_default_parameter_file(path=path_unsplit, prompt=False)
        unsplit_model = import_parameters_py(path_unsplit).model
        assert not unsplit_model.params["split_sound"] and unsplit_model.params["linear"]
        assert unsplit_model.propagators.variat_qb.options.model == "linear_q"

    ut.call_test(
        model=model,
        test_profiling=test_profiling,
        case=f"{perturbative_model}-linear-crank_nicolson={crank_nicolson}",
        update_baselines=update_baselines,
    )


@pytest.mark.models
@pytest.mark.fluid
@pytest.mark.parametrize("model_cls_name", ["ShearAlfven", "LinearMHD"])
@pytest.mark.parametrize("projected", [False, True])
def test_b_space_hcurl(model_cls_name: str, projected: bool, tmp_path, update_baselines: bool):
    """The H(curl) discretizations of the shear-Alfven models: the flags survive the generated parameter file."""
    from struphy.io.setup import import_parameters_py

    model = models_utils.get_model_by_name(model_cls_name)(b_space="Hcurl", projected=projected)
    path = os.path.join(tmp_path, f"params_{model_cls_name}.py")
    model.generate_default_parameter_file(path=path, prompt=False)
    params_model = import_parameters_py(path).model
    assert params_model.params["b_space"] == "Hcurl" and params_model.params["projected"] == projected
    assert params_model.em_fields.b_field.space == "Hcurl"
    ut.call_test(
        model=model,
        case=f"{model_cls_name}-b_space=Hcurl-projected={projected}",
        update_baselines=update_baselines,
    )


@pytest.mark.models
@pytest.mark.fluid
@pytest.mark.parametrize("b_space", ["Hdiv", "Hcurl"])
def test_linear_mhd_split_sound(b_space: str, tmp_path, update_baselines: bool):
    """LinearMHD with the split magnetosonic step: the flag survives the generated parameter file, which has the
    split propagators."""
    from struphy.io.setup import import_parameters_py

    model = models_utils.get_model_by_name("LinearMHD")(b_space=b_space, split_sound=True)
    path = os.path.join(tmp_path, "params_LinearMHD.py")
    model.generate_default_parameter_file(path=path, prompt=False)
    params_model = import_parameters_py(path).model
    assert params_model.params["split_sound"] and params_model.params["b_space"] == b_space
    assert not hasattr(params_model.propagators, "mag_sonic")
    for name in ("bg_current", "sound", "bg_pressure"):
        assert hasattr(params_model.propagators, name)
    ut.call_test(model=model, case=f"LinearMHD-split_sound-b_space={b_space}", update_baselines=update_baselines)


@pytest.mark.single
def test_single_model(
    model_name: str,
    update_baselines: bool,
    test_profiling: bool = True,
):
    logger.info(f"{model_name = }")
    model = models_utils.get_model_by_name(model_name=model_name)
    ut.call_test(
        model=model(),
        test_profiling=test_profiling,
        update_baselines=update_baselines,
    )


if __name__ == "__main__":
    test_toy(update_baselines=False)
    test_fluid(update_baselines=False)
    test_single_model("Maxwell", update_baselines=False)
