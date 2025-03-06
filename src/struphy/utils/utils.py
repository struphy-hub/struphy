import os

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

    state_file = os.path.join(libpath, 'state.yml')
    try:
        with open(state_file, 'r') as f:
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
    i_path = state.get('i_path', os.path.join(libpath, 'io/inp'))
    o_path = state.get('o_path', os.path.join(libpath, 'io/out'))
    b_path = state.get('b_path', os.path.join(libpath, 'io/batch'))
    # Update state if defaults were used
    state['i_path'] = i_path
    state['o_path'] = o_path
    state['b_path'] = b_path
    return i_path, o_path, b_path


def save_state(state, libpath=STRUPHY_LIBPATH):
    """Save the state to the state.yml file."""
    state_file = os.path.join(libpath, 'state.yml')
    with open(state_file, 'w') as f:
        yaml.dump(state, f)


def print_all_attr(obj):
    '''Print all object's attributes that do not start with "_" to screen.'''
    import numpy as np

    for k in dir(obj):
        if k[0] != '_':
            v = getattr(obj, k)
            if isinstance(v, np.ndarray):
                v = f'{type(getattr(obj, k))} of shape {v.shape}'
            if 'proj_' in k or 'quad_grid_' in k:
                v = '(arrays not displayed)'
            print(k.ljust(26), v)

def dict_to_yaml(dictionary, output):
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

if __name__ == '__main__':
    state = read_state()
    for k, val in state.items():
        print(k, val)
    i_path, o_path, b_path = get_paths(state)
    print(f'{i_path = }')
    print(f'{o_path = }')
    print(f'{b_path = }')
