def get_dependencies(pymod_abs=None):
    """Compute all dependencies that contain the string "kernels" of a Struphy module.

    Parameters
    ----------
    pymod_abs : str
        Absolute path to target (ends with .so). If None, the absolute path must be given as the first command line argument.
    """

    import importlib
    import os
    import shutil
    import sys
    import sysconfig
    import time
    import types

    so_suffix = sysconfig.get_config_var("EXT_SUFFIX")

    if pymod_abs is None:
        # with open('cool.txt', 'w') as f:
        #     print(f'{sys.argv = }', file=f)
        assert len(sys.argv) > 1
        assert sys.argv[1][-3:] == ".so"
        pymod_abs = sys.argv[1]
    else:
        assert "struphy/" in pymod_abs and ".so" in pymod_abs

    # handle psydac modules (TODO: remove)
    if "psydac/" in pymod_abs:
        if "bsplines_kernels" in pymod_abs:
            return pymod_abs.replace("bsplines_kernels", "arrays")
        else:
            return ""

    pymod_abs = pymod_abs.replace(so_suffix, ".py")

    # print(f'{pymod_abs = }')
    pymod_so = pymod_abs.replace(".py", so_suffix)
    # print(f'{pymod_so = }')

    # temporaryily move .py file to _tmp.py for getting correct dependencies
    del_tmp = False
    if os.path.isfile(pymod_so):
        tmp = pymod_abs.replace(".py", "_tmp.py")
        # print(f'{tmp = }')
        shutil.copyfile(pymod_abs, tmp)
        time.sleep(0.01)
        del_tmp = True
    else:
        tmp = pymod_abs

    # struphy modules
    splits = tmp.split("/")

    # print(f'{splits = }')

    booli = [i == "struphy" for i in splits]
    ids = [i for i, x in enumerate(booli) if x]
    stem = "/".join(splits[: ids[-1]]) + "/"

    # print(f'{stem = }')

    splits = splits[::-1]
    file = splits[0]
    assert file[-3:] == ".py"
    name = file[:-3]
    for pkg in splits[1:]:
        name = pkg + "." + name
        if "struphy" in name:
            break

    # print(f'{name = }')

    mod = importlib.import_module(name)

    # print(f'{mod = }')
    # print(f'{dir(mod) = }')
    # print(f'{vars(mod) = }')

    depends = []
    for k, v in vars(mod).items():
        if isinstance(v, types.ModuleType):
            # print(f'{v = }')
            if "kernels" in v.__name__:
                depends += [stem + v.__name__.replace(".", "/") + so_suffix]

    if del_tmp:
        os.remove(tmp)

    return " ".join(depends)


if __name__ == "__main__":
    deps = get_dependencies()
    print(deps)
