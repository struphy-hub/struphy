import os
import json

"""Helper functions."""



def read_json(filepath: str, filename: str) -> dict:
    """Loads GVEC output from a JSON file.

    GVEC output in a .dat file should be converted into JSON format using `reader/GVEC_Reader`. 
    Then this function loads that JSON file as a `dict`.

    Parameters
    ----------
    filepath : str
        The path to the JSON file.
    filename:
        The name of the JSON file.

    Returns
    -------
    dict
        A `dict` containing GVEC output, where its keys and values are documented in `List_of_data_entries.md`.

    Raises
    ------
        Error: Just a placeholder.
    """

    with open(os.path.join(filepath, filename)) as f:
        return json.load(f)
