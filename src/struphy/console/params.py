from struphy.models import fluid, hybrid, kinetic, toy
from struphy.models.base import StruphyModel


def struphy_params(model_name: str, params_path: str, yes: bool = False):
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

    prompt = not yes
    model.generate_default_parameter_file(path=params_path, prompt=prompt)
