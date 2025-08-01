from struphy.models.base import StruphyModel
from struphy.models import fluid, hybrid, kinetic, toy


def struphy_params(model_name: str, file, yes=False, options=False):
    """Create a model's default parameter file and save in current input path.

    Parameters
    ----------
    model_name : str
        The name of the Struphy model.

    file : str
        An alternative file name to the default params_<model>.yml.
        
    yes : bool
        If true, say yes on prompt to overwrite .yml FILE

    show_options : bool
        Whether to print to screen all possible options for the model.
    """
    objs = [fluid, kinetic, hybrid, toy]
    for obj in objs:
        try:
            model_class = getattr(obj, model_name)
            model: StruphyModel = model_class()
        except AttributeError:
            pass
        
    prompt = not yes
    model.generate_default_parameter_file(file_name=file, prompt=prompt)
