import logging
import sys

import yaml
from feectools.ddm.mpi import mpi as MPI

from struphy import models
from struphy.models.base import StruphyModel
from struphy.models.utils import get_model_by_name

logger = logging.getLogger("struphy")


def struphy_params(model_name: str, yes: bool = False, check_file: bool = False):
    """Create a model's default parameter file and save in current input path.

    Parameters
    ----------
    model_name : str
        The name of the Struphy model.

    yes : bool
        If true, say yes on prompt to overwrite .yml FILE
    """

    model_class = get_model_by_name(model_name=model_name)
    model: StruphyModel = model_class()

    # logger.info(f"{model_name =} {model = }")

    # print units
    if check_file:
        logger.info(f"Checking {check_file} with model {model_class}")
        with open(check_file) as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
        # TODO: Enable running struphy without any communicators
        comm = MPI.COMM_WORLD
        try:
            model = model_class(params=params, comm=MPI.COMM_WORLD)
            logger.info("Model initialized successfully.")
        except Exception as e:
            logger.info(f"Failed to initialize model: {e}")
            sys.exit(1)

    else:
        prompt = not yes
        model.generate_default_parameter_file(path=None, prompt=prompt)
        # logger.info(f"Generating default parameter file for {model_class}.")
        # model_class().generate_default_parameter_file(path=params_path, prompt=prompt)
