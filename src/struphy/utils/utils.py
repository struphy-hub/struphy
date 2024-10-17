import os
import yaml
import struphy

def read_state():
    """
    Reads the 'state.yml' file located in the Struphy library path.

    Returns:
    --------
    dict
        A dictionary containing the parsed YAML content of the 'state.yml' file.
    
    Raises:
    -------
    FileNotFoundError
        If the 'state.yml' file is not found.
    YAMLError
        If there is an error while parsing the YAML file.
    """

    # Get the path to the Struphy library
    LIB_PATH = struphy.__path__[0]
    
    state_file = os.path.join(LIB_PATH, 'state.yml')
    state = {}
    try:
        with open(state_file, 'r') as f:
            state = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError as e:
        print(f"Error: The file '{state_file}' was not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing the YAML file: {e}")

    
    return state


if __name__ == '__main__':
    state = read_state()
    for k,val in state.items():
        print(k,val)