import sys

import yaml
from psydac.ddm.mpi import mpi as MPI


def struphy_params(model, file, yes=False, options=False, check_file=None):
    """Create a model's default parameter file and save in current input path.

    Parameters
    ----------
    model : str
        The name of the Struphy model.

    yes : bool
        If true, say yes on prompt to overwrite .yml FILE

    file : str
        An alternative file name to the default params_<model>.yml.

    show_options : bool
        Whether to print to screen all possible options for the model.
    """

    from struphy.models import fluid, hybrid, kinetic, toy

    # load model class
    objs = [fluid, kinetic, hybrid, toy]
    for obj in objs:
        try:
            model_class = getattr(obj, model)
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

    elif options:
        model_class.show_options()
    else:
        prompt = not yes
        params = model_class.generate_default_parameter_file(file=file, prompt=prompt)
