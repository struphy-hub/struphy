def struphy_units(model, input, input_abs=None):
    """
    Show units and some important physics quantities of a Struphy model.

    Parameters
    ----------
    model : str
        The name of the model class.

    inp : str
        The .yml parameter file relative to <struphy_path>/io/inp/.

    inp_abs : str, optional
        The absolute path of the .yml parameter file.
    """

    import os

    import yaml

    import struphy.utils.utils as utils
    from struphy.models import fluid, hybrid, kinetic, toy

    # load model class
    objs = [fluid, kinetic, hybrid, toy]
    for obj in objs:
        try:
            model_class = getattr(obj, model)
        except AttributeError:
            pass

    # Read struphy state file
    state = utils.read_state()

    i_path = state["i_path"]

    # create absolute i/o paths
    if input_abs is None:
        if input is None:
            params = model_class.generate_default_parameter_file(save=False)
        else:
            input_abs = os.path.join(i_path, input)

            with open(input_abs) as file:
                params = yaml.load(file, Loader=yaml.FullLoader)

    # print units
    model_class.model_units(params, verbose=True)
