def test_pproc_codes(model: str = None, group: str = None):
    """Tests the post processing of runs in test_codes.py"""

    import inspect
    import os

    from psydac.ddm.mpi import mpi as MPI

    import struphy
    from struphy.models import fluid, hybrid, kinetic, toy
    from struphy.post_processing import pproc_struphy

    comm = MPI.COMM_WORLD

    libpath = struphy.__path__[0]

    list_fluid = []
    for name, obj in inspect.getmembers(fluid):
        if inspect.isclass(obj) and obj.__module__ == fluid.__name__:
            if name not in {"StruphyModel", "Propagator"}:
                list_fluid += [name]

    list_kinetic = []
    for name, obj in inspect.getmembers(kinetic):
        if inspect.isclass(obj) and obj.__module__ == kinetic.__name__:
            if name not in {"StruphyModel", "KineticBackground", "Propagator"}:
                list_kinetic += [name]

    list_hybrid = []
    for name, obj in inspect.getmembers(hybrid):
        if inspect.isclass(obj) and obj.__module__ == hybrid.__name__:
            if name not in {"StruphyModel", "Propagator"}:
                list_hybrid += [name]

    list_toy = []
    for name, obj in inspect.getmembers(toy):
        if inspect.isclass(obj) and obj.__module__ == toy.__name__:
            if name not in {"StruphyModel", "Propagator"}:
                list_toy += [name]

    if group is None:
        list_models = list_fluid + list_kinetic + list_hybrid + list_toy
    elif group == "fluid":
        list_models = list_fluid
    elif group == "kinetic":
        list_models = list_kinetic
    elif group == "hybrid":
        list_models = list_hybrid
    elif group == "toy":
        list_models = list_toy
    else:
        raise ValueError(f"{group = } is not a valid group specification.")

    if comm.Get_rank() == 0:
        if model is None:
            for model in list_models:
                if "Variational" in model or "Visco" in model:
                    print(f"Model {model} is currently excluded from tests.")
                    continue

                path_out = os.path.join(libpath, "io/out/test_" + model)
                pproc_struphy.main(path_out)
        else:
            path_out = os.path.join(libpath, "io/out/test_" + model)
            pproc_struphy.main(path_out)


if __name__ == "__main__":
    test_pproc_codes()
