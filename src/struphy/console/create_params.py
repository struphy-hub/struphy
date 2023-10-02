def struphy_create_params(model, file=None, options=False):
    '''Create a model's default parameter file and save in current input path.

    Parameters
    ----------
    model : str
        The name of the Struphy model.

    file : str
        An alternative file name to the default params_<model>.yml.

    show_options : bool
        Whether to print to screen all possible options for the model.
    '''

    from struphy.models import fluid, kinetic, hybrid, toy

    # load model class
    objs = [fluid, kinetic, hybrid, toy]
    for obj in objs:
        try:
            model_class = getattr(obj, model)
        except AttributeError:
            pass

    # print units
    if options:
        model_class.show_options()
    else:
        params = model_class.generate_default_parameter_file(file=file)
