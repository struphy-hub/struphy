import inspect
import os
from types import ModuleType

import pytest

from struphy import main
from struphy.io.options import EnvironmentOptions
from struphy.io.setup import import_parameters_py
from struphy.models import fluid, hybrid, kinetic, toy
from struphy.models.base import StruphyModel

# available models
toy_models = []
for name, obj in inspect.getmembers(toy):
    if inspect.isclass(obj) and "models.toy" in obj.__module__:
        toy_models += [name]
print(f"\n{toy_models = }")

fluid_models = []
for name, obj in inspect.getmembers(fluid):
    if inspect.isclass(obj) and "models.fluid" in obj.__module__:
        fluid_models += [name]
print(f"\n{fluid_models = }")

kinetic_models = []
for name, obj in inspect.getmembers(kinetic):
    if inspect.isclass(obj) and "models.kinetic" in obj.__module__:
        kinetic_models += [name]
print(f"\n{kinetic_models = }")

hybrid_models = []
for name, obj in inspect.getmembers(hybrid):
    if inspect.isclass(obj) and "models.hybrid" in obj.__module__:
        hybrid_models += [name]
print(f"\n{hybrid_models = }")


# folder for test simulations
test_folder = os.path.join(os.getcwd(), "struphy_model_tests")


# generic function for calling model tests
def call_test(model_name: str, module: ModuleType, verbose=True):
    print(f"\n*** Testing '{model_name}':")
    model = getattr(module, model_name)()
    assert isinstance(model, StruphyModel)

    # generate paramater file for testing
    path = os.path.join(test_folder, f"params_{model_name}.py")
    model.generate_default_parameter_file(path=path, prompt=False)
    del model
    print("\nDeleting light-weight instance ...")

    # set environment options
    env = EnvironmentOptions(out_folders=test_folder, sim_folder=f"{model_name}_test")

    # read parameters
    params_in = import_parameters_py(path)
    units = params_in.units
    time_opts = params_in.time_opts
    domain = params_in.domain
    equil = params_in.equil
    grid = params_in.grid
    derham_opts = params_in.derham_opts

    # test
    model = params_in.model
    main.run(
        model,
        params_path=path,
        env=env,
        units=units,
        time_opts=time_opts,
        domain=domain,
        equil=equil,
        grid=grid,
        derham_opts=derham_opts,
        verbose=verbose,
    )


# specific tests
@pytest.mark.parametrize("model_name", toy_models)
def test_toy(model_name: str, verbose=True):
    call_test(model_name=model_name, module=toy, verbose=verbose)


@pytest.mark.parametrize("model_name", fluid_models)
def test_fluid(model_name: str, verbose=True):
    call_test(model_name=model_name, module=fluid, verbose=verbose)


@pytest.mark.parametrize("model_name", kinetic_models)
def test_kinetic(model_name: str, verbose=True):
    call_test(model_name=model_name, module=kinetic, verbose=verbose)


@pytest.mark.parametrize("model_name", hybrid_models)
def test_hybrid(model_name: str, verbose=True):
    call_test(model_name=model_name, module=hybrid, verbose=verbose)


if __name__ == "__main__":
    test_toy("Maxwell")
    test_fluid("LinearMHD")
