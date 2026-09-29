"""Determine the kernel dependencies of a Struphy kernel module.

This script is called by ``compile_struphy.mk`` for every target ``<kernel>.so`` to obtain its
prerequisites, i.e. the other Struphy kernels it imports. Make uses these prerequisites to
compile kernels in the correct order, which is what allows ``struphy compile -j N`` to build
independent kernels in parallel.

Usage: ``python dependencies.py /abs/path/to/struphy/.../some_kernels<EXT_SUFFIX>``
"""

import logging

logger = logging.getLogger("struphy")


def get_dependencies(pymod_abs=None):
    """Compute all dependencies that contain the string "kernels" of a Struphy module.

    The dependencies are found by statically parsing the module source with :mod:`ast`;
    the module is never imported or executed. A module counts as a dependency if

    * it is bound to a name at module level (also inside ``if``/``try``/``with`` blocks,
      but not inside functions or classes), e.g. ``import struphy.x.a_kernels as a_kernels``,
      ``from struphy.x import a_kernels`` or ``from . import a_kernels``, and
    * its dotted name starts with "struphy" and contains "kernels".

    This reproduces the result of the former import-based implementation, which collected
    all module-type attributes of the imported module.

    Parameters
    ----------
    pymod_abs : str
        Absolute path to target (ends with .so). If None, the absolute path must be given as the first command line argument.

    Returns
    -------
    str
        Space-separated absolute paths of the dependencies (with the .so suffix of the current
        Python), in order of first appearance. Empty string if there are none.
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

    logger.debug(f"\n{pymod_abs = }")
    pymod_abs = pymod_abs.replace(so_suffix, ".py")

    # struphy modules
    splits = pymod_abs.split("/")

    # stem is the directory containing the (innermost) struphy package, e.g. ".../src/"
    ids = [i for i, x in enumerate(splits) if x == "struphy"]
    stem = "/".join(splits[: ids[-1]]) + "/"
    logger.debug(f"{stem = }")

    # dotted name of the module, needed to resolve relative imports
    name = ".".join(splits[ids[-1] :])[: -len(".py")]

    # Parse the source statically instead of importing it (importing struphy takes ~10 s per module).
    # Like the former import-based version, only names bound to a module with "kernels" in its name
    # at module level count; this includes imports nested in if/try blocks, but not those in functions.
    with open(pymod_abs) as f:
        tree = ast.parse(f.read())

    def module_level(nodes):
        """Yield all statements executed at module level.

        Recurses into the blocks of if/try/with statements, but not into function or class bodies.
        """
        for node in nodes:
            yield node
            if isinstance(node, ast.If | ast.Try | ast.With):
                for field in ("body", "orelse", "finalbody"):
                    yield from module_level(getattr(node, field, []))
                # except clauses of try statements
                for handler in getattr(node, "handlers", []):
                    yield from module_level(handler.body)

    def is_module(dotted):
        """Whether a dotted name refers to a module (.py file) or a package (directory) below stem."""
        path = stem + dotted.replace(".", "/")
        return os.path.isfile(path + ".py") or os.path.isdir(path)

    # dotted names of all modules bound to a name at module level
    mods = []
    for node in module_level(tree.body):
        if isinstance(node, ast.Import):
            # "import a.b.c" binds only "a", "import a.b.c as d" binds the module a.b.c
            mods += [alias.name for alias in node.names if alias.asname]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                # relative import: each leading dot strips one component from the module name,
                # e.g. in struphy.a.b "from . import x" gives struphy.a.x, "from ..c import x" gives struphy.c.x
                pkg = name.split(".")[: -node.level]
                base = ".".join(pkg + ([base] if base else []))
            # "from a.b import c" binds a module only if c is a module and not an object in a.b
            mods += [base + "." + alias.name for alias in node.names if is_module(base + "." + alias.name)]
    logger.debug(f"{mods = }")

    # keep only Struphy kernels, convert them to paths of the compiled targets and remove duplicates
    depends = []
    for mod in mods:
        if "kernels" in mod and mod.startswith("struphy"):
            dep = stem + mod.replace(".", "/") + so_suffix
            if dep not in depends:
                depends += [dep]
                logger.debug(f"new dependency: {dep}")

    return " ".join(depends)


if __name__ == "__main__":
    deps = get_dependencies()
    print(deps)
