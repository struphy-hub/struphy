import os
import subprocess
import sys

import yaml

import struphy

# Get the path to the Struphy library
STRUPHY_LIBPATH = struphy.__path__[0]


def read_state(libpath=STRUPHY_LIBPATH):
    """
    Reads the 'state.yml' file located in the Struphy library path.

    Returns:
    --------
    dict
        A dictionary containing the parsed YAML content of the 'state.yml' file.
        If the file is not found or there is an error parsing the YAML file,
        an empty dictionary is returned.
    """

    state_file = os.path.join(libpath, "state.yml")
    try:
        with open(state_file, "r") as f:
            state = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError as e:
        print(f"The state file '{state_file}' was not found. Creating a new one.")
        state = {}
    except yaml.YAMLError as e:
        print(f"Error {e}: parsing the YAML file")
        state = {}

    return state


def get_paths(state, libpath=STRUPHY_LIBPATH):
    """Get input, output, and batch paths from the state or set defaults."""
    i_path = state.get("i_path", os.path.join(libpath, "io/inp"))
    o_path = state.get("o_path", os.path.join(libpath, "io/out"))
    b_path = state.get("b_path", os.path.join(libpath, "io/batch"))

    return i_path, o_path, b_path


def update_state(state):
    """Update i_path, o_path, b_path in state if they are not set."""
    i_path, o_path, b_path = get_paths(state=state)
    state["i_path"] = i_path
    state["o_path"] = o_path
    state["b_path"] = b_path


def save_state(state, libpath=STRUPHY_LIBPATH):
    """Save the state to the state.yml file."""
    state_file = os.path.join(libpath, "state.yml")
    dict_to_yaml(state, state_file)


def print_all_attr(obj):
    """Print all object's attributes that do not start with "_" to screen."""
    import numpy as np

    for k in dir(obj):
        if k[0] != "_":
            v = getattr(obj, k)
            if isinstance(v, np.ndarray):
                v = f"{type(getattr(obj, k))} of shape {v.shape}"
            if "proj_" in k or "quad_grid_" in k:
                v = "(arrays not displayed)"
            print(k.ljust(26), v)


def dict_to_yaml(dictionary: dict, output: str):
    """Write dictionary to file and save in output."""
    with open(output, "w") as file:
        yaml.dump(
            dictionary,
            file,
            Dumper=MyDumper,
            default_flow_style=None,
            sort_keys=False,
            indent=4,
            line_break="\n",
        )


class MyDumper(yaml.SafeDumper):
    # HACK: insert blank lines between top-level objects
    # inspired by https://stackoverflow.com/a/44284819/3786245
    def write_line_break(self, data=None):
        super().write_line_break(data)

        if len(self.indents) == 1:
            super().write_line_break()

    def ignore_aliases(self, data):
        return True


def refresh_models():
    print("Collecting available models ...")

    import inspect
    import pickle

    from struphy.models import fluid, hybrid, kinetic, toy

    list_fluid = []
    fluid_string = ""
    for name, obj in inspect.getmembers(fluid):
        if inspect.isclass(obj) and obj.__module__ == fluid.__name__:
            # if name not in {"StruphyModel", "Propagator"}:
            list_fluid += [name]
            fluid_string += '"' + name + '"\n'

    list_kinetic = []
    kinetic_string = ""
    for name, obj in inspect.getmembers(kinetic):
        if inspect.isclass(obj) and obj.__module__ == kinetic.__name__:
            if name not in {"StruphyModel", "Propagator"}:
                list_kinetic += [name]
                kinetic_string += '"' + name + '"\n'

    list_hybrid = []
    hybrid_string = ""
    for name, obj in inspect.getmembers(hybrid):
        if inspect.isclass(obj) and obj.__module__ == hybrid.__name__:
            if name not in {"StruphyModel", "Propagator"}:
                list_hybrid += [name]
                hybrid_string += '"' + name + '"\n'

    list_toy = []
    toy_string = ""
    for name, obj in inspect.getmembers(toy):
        if inspect.isclass(obj) and obj.__module__ == toy.__name__:
            if name not in {"StruphyModel", "Propagator"}:
                list_toy += [name]
                toy_string += '"' + name + '"\n'

    list_models = list_fluid + list_kinetic + list_hybrid + list_toy

    with open(os.path.join(STRUPHY_LIBPATH, "models", "models_list"), "wb") as fp:
        pickle.dump(list_models, fp)

    # fluid message
    fluid_message = "Fluid models:\n"
    fluid_message += "-------------\n"
    fluid_message += fluid_string

    # kinetic message
    kinetic_message = "Kinetic models:\n"
    kinetic_message += "---------------\n"
    kinetic_message += kinetic_string

    # hybrid message
    hybrid_message = "Hybrid models:\n"
    hybrid_message += "--------------\n"
    hybrid_message += hybrid_string

    # toy message
    toy_message = "Toy models:\n"
    toy_message += "-----------\n"
    toy_message += toy_string

    # model message
    model_message = "run one of the following models:\n"
    model_message += "\n" + fluid_message
    model_message += "\n" + kinetic_message
    model_message += "\n" + hybrid_message
    model_message += "\n" + toy_message

    with open(os.path.join(STRUPHY_LIBPATH, "models", "models_message"), "wb") as fp:
        pickle.dump(
            [
                model_message,
                fluid_message,
                kinetic_message,
                hybrid_message,
                toy_message,
            ],
            fp,
        )

    print("Done.")


def subp_run(cmd, cwd="libpath", check=True):
    """Call subprocess.run and print run command."""

    if cwd == "libpath":
        cwd = struphy.__path__[0]

    print(f"\nRunning the following command as a subprocess:\n{' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=check)


if __name__ == "__main__":
    state = read_state()
    for k, val in state.items():
        print(k, val)
    i_path, o_path, b_path = get_paths(state)
    print(f"{i_path = }")
    print(f"{o_path = }")
    print(f"{b_path = }")
