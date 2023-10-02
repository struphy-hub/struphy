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
        Thea absolute path of the .yml paramter file.
    """

    import os
    import yaml
    import struphy
    from struphy.models import fluid, kinetic, hybrid, toy

    libpath = struphy.__path__[0]

    with open(os.path.join(libpath, 'i_path.txt')) as f:
        i_path = f.readlines()[0]

    # create absolute i/o paths
    if input_abs is None:
        input_abs = os.path.join(i_path, input)

    # load simulation parameters
    with open(input_abs) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    # load model class
    objs = [fluid, kinetic, hybrid, toy]
    for obj in objs:
        try:
            model_class = getattr(obj, model)
        except AttributeError:
            pass

    # print units
    model_class.model_units(params, verbose=True)
