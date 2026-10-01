import logging
import os
import sys

from struphy.models.base import StruphyModel
from struphy.models.utils import get_model_by_name

logger = logging.getLogger("struphy")


def struphy_params(model_name: str, yes: bool = False, check_file: str | None = None):
    """Create a model's default parameter file and save in current input path.

    Parameters
    ----------
    model_name : str
        The name of the Struphy model.

    yes : bool
        If true, say yes on prompt to overwrite .py FILE

    check_file : str, optional
        Instead of creating a file, check that the .py parameter file ``check_file``
        runs and defines a Simulation ``sim`` of the model ``model_name``.
    """

    model_class = get_model_by_name(model_name=model_name)

    if check_file:
        check_parameter_file(check_file, model_class)

    else:
        model: StruphyModel = model_class()
        prompt = not yes
        model.generate_default_parameter_file(path=None, prompt=prompt)


def check_parameter_file(path: str, model_class: type):
    """Execute the .py parameter file at ``path`` (without running the simulation) and check that it
    defines a Simulation ``sim`` of the model ``model_class``; exit with status 1 otherwise."""
    from struphy.io.setup import import_parameters_py
    from struphy.simulation.sim import Simulation

    def fail(message):
        print(f"Parameter file {path} is not valid: {message}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(path):
        fail("file does not exist")
    if not path.endswith(".py"):
        fail("parameter files are Python files (.py); create one with 'struphy params MODEL'")

    # like 'python FILE', but the 'if __name__ == "__main__":' block that calls sim.run() is skipped
    dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # no __pycache__ next to the user's file
    try:
        params_in = import_parameters_py(os.path.abspath(path))
    except Exception as e:
        fail(f"executing it raised {type(e).__name__}: {e}")
    finally:
        sys.dont_write_bytecode = dont_write_bytecode

    sim = getattr(params_in, "sim", None)
    if not isinstance(sim, Simulation):
        fail("it does not define a Simulation object named 'sim'")
    if not isinstance(sim.model, model_class):
        fail(f"its model is {type(sim.model).__name__}, not {model_class.__name__}")

    print(f"Parameter file {path} is valid for the model {model_class.__name__}.")
