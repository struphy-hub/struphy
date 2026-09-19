def get_dependencies(pymod_abs=None):
    """Compute all dependencies that contain the string "kernels" of a Struphy module.

    Parameters
    ----------
    pymod_abs : str
        Absolute path to target (ends with .so). If None, the absolute path must be given as the first command line argument.
    """

    import ast
    import os
    import sys
    import sysconfig

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

    # struphy modules
    splits = pymod_abs.split("/")

    ids = [i for i, x in enumerate(splits) if x == "struphy"]
    stem = "/".join(splits[: ids[-1]]) + "/"

    # dotted name of the module, needed to resolve relative imports
    name = ".".join(splits[ids[-1] :])[: -len(".py")]

    # Parse the source statically instead of importing it (importing struphy takes ~10 s per module).
    # Like the former import-based version, only names bound to a module with "kernels" in its name
    # at module level count; this includes imports nested in if/try blocks, but not those in functions.
    with open(pymod_abs) as f:
        tree = ast.parse(f.read())

    def module_level(nodes):
        for node in nodes:
            yield node
            if isinstance(node, ast.If | ast.Try | ast.With):
                for field in ("body", "orelse", "finalbody"):
                    yield from module_level(getattr(node, field, []))
                for handler in getattr(node, "handlers", []):
                    yield from module_level(handler.body)

    def is_module(dotted):
        path = stem + dotted.replace(".", "/")
        return os.path.isfile(path + ".py") or os.path.isdir(path)

    mods = []
    for node in module_level(tree.body):
        if isinstance(node, ast.Import):
            # "import a.b.c" binds only "a", "import a.b.c as d" binds the module a.b.c
            mods += [alias.name for alias in node.names if alias.asname]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                pkg = name.split(".")[: -node.level]
                base = ".".join(pkg + ([base] if base else []))
            # "from a.b import c" binds a module only if c is a module and not an object in a.b
            mods += [base + "." + alias.name for alias in node.names if is_module(base + "." + alias.name)]

    depends = []
    for mod in mods:
        if "kernels" in mod and mod.startswith("struphy"):
            dep = stem + mod.replace(".", "/") + so_suffix
            if dep not in depends:
                depends += [dep]

    return " ".join(depends)


if __name__ == "__main__":
    deps = get_dependencies()
    print(deps)
