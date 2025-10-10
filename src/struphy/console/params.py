import sys

import yaml
from mpi4py import MPI

from struphy.models import fluid, hybrid, kinetic, toy
from struphy.models.base import StruphyModel


def struphy_params(model_name: str, params_path: str, yes: bool = False, check_file: bool = False):
    """Create a model's default parameter file and save in current input path.

    Parameters
    ----------
    model_name : str
        The name of the Struphy model.

    params_path : str
        An alternative file name to the default params_<model>.yml.

    yes : bool
        If true, say yes on prompt to overwrite .yml FILE
    """
    objs = [fluid, kinetic, hybrid, toy]
    for obj in objs:
        try:
            model_class = getattr(obj, model_name)
            model: StruphyModel = model_class()
        except AttributeError:
            pass

    # print units
    if check_file:
        print(f"Checking {check_file} with model {model_class}")
        with open(check_file) as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
        # TODO: Enable running struphy without any communicators
        comm = MPI.COMM_WORLD
        try:
            model = model_class(params=params, comm=MPI.COMM_WORLD)
            print("Model initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize model: {e}")
            sys.exit(1)

    else:
        prompt = not yes
        print(f"Generating default parameter file for {model_class}.")
        model_class().generate_default_parameter_file(path=params_path, prompt=prompt)
