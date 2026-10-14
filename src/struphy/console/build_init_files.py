"""Regenerate the auto-generated __init__.py files of struphy (struphy build-init-files)."""

import ast
import importlib
import inspect
import logging
import os
import subprocess

import struphy
from struphy.models.base import StruphyModel

logger = logging.getLogger("struphy")

LIBPATH = struphy.__path__[0]


def construct_package_init_file(
    package_dir: str, package_name: str, base_class: type, skip: tuple = ("base.py",)
) -> str:
    """
    Constructs the content of an `__init__.py` file for a package laid out with one class
    per module, by importing each module and collecting the subclasses of `base_class` that
    are defined in it. Preserves the existing module docstring of the `__init__.py`, if any.

    The generated file resolves the classes lazily (PEP 562 module ``__getattr__``), so that
    ``from package import SomeClass`` only imports the module defining ``SomeClass`` instead of
    every module in the package. Importing all of them costs a considerable amount of time and
    is rarely needed. A ``TYPE_CHECKING`` block keeps static analysis and IDEs working.

    Parameters
    ----------
    package_dir : str
        Path to the package directory (e.g. "src/struphy/models").

    package_name : str
        Dotted import path of the package (e.g. "struphy.models").

    base_class : type
        Only classes that are (strict) subclasses of `base_class`, and defined directly in
        the module being scanned, are collected.

    skip : tuple, optional
        Module filenames to skip in addition to `__init__.py` (default=("base.py",)).
    """
    existing_init_path = os.path.join(package_dir, "__init__.py")
    docstring = None
    if os.path.isfile(existing_init_path):
        with open(existing_init_path, "r", encoding="utf-8") as f:
            docstring = ast.get_docstring(ast.parse(f.read()), clean=False)

    init_content = f'"""{docstring}"""\n\n' if docstring else ""
    class_names = []
    class_modules = {}

    for file_name in sorted(os.listdir(package_dir)):
        if file_name in ("__init__.py", *skip):
            continue
        if file_name.endswith(".py"):
            module_name = file_name[:-3]  # strip .py
        elif os.path.isfile(os.path.join(package_dir, file_name, "__init__.py")):
            module_name = file_name  # sub-package laid out as one directory per class
        else:
            continue
        module = importlib.import_module(f"{package_name}.{module_name}")

        # Sub-packages resolve their class lazily via __getattr__; inspect.getmembers would
        # not see it, so go through __all__ (falling back to what getmembers finds).
        candidates = [getattr(module, n) for n in getattr(module, "__all__", ())]
        candidates += [cls for _, cls in inspect.getmembers(module, inspect.isclass)]

        # Loop over all classes in the module
        for cls in candidates:
            # Only subclasses of base_class defined in this module (or its sub-modules)
            if (
                inspect.isclass(cls)
                and issubclass(cls, base_class)
                and cls != base_class
                and (cls.__module__ == module.__name__ or cls.__module__.startswith(module.__name__ + "."))
                and cls.__name__ not in class_modules
            ):
                class_names.append(cls.__name__)
                class_modules[cls.__name__] = f"{package_name}.{module_name}"

    init_content += "import importlib\n"
    init_content += "from typing import TYPE_CHECKING\n\n"

    init_content += "# class name -> module defining it, resolved on first access by __getattr__ below\n"
    init_content += "_LAZY_IMPORTS = {\n"
    for class_name in class_names:
        init_content += f'    "{class_name}": "{class_modules[class_name]}",\n'
    init_content += "}\n\n"

    init_content += "if TYPE_CHECKING:  # static analysis and IDEs see the eager imports\n"
    for class_name in class_names:
        init_content += f"    from {class_modules[class_name]} import {class_name}\n"
    init_content += "\n"

    init_content += f"__all__ = {class_names}\n\n\n"
    init_content += "def __getattr__(name: str):\n"
    init_content += "    module_name = _LAZY_IMPORTS.get(name)\n"
    init_content += "    if module_name is None:\n"
    init_content += '        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")\n'
    init_content += "    value = getattr(importlib.import_module(module_name), name)\n"
    init_content += "    globals()[name] = value  # cache: later lookups bypass __getattr__\n"
    init_content += "    return value\n\n\n"
    init_content += "def __dir__():\n"
    init_content += "    return sorted(set(globals()) | set(_LAZY_IMPORTS))\n"
    return init_content


def construct_models_init_file(models_dir: str = os.path.join(LIBPATH, "models")) -> str:
    """
    Constructs __init__.py for all generated model files by reading actual class names.
    Skips base.py and __init__.py.
    """
    return construct_package_init_file(models_dir, "struphy.models", StruphyModel)


def construct_propagators_init_file(propagators_dir: str = os.path.join(LIBPATH, "propagators")) -> str:
    """
    Constructs __init__.py for all generated propagator files by reading actual class names.
    Skips base.py and __init__.py.
    """
    from struphy.propagators.base import Propagator

    return construct_package_init_file(propagators_dir, "struphy.propagators", Propagator)


def construct_domains_init_file(domains_dir: str = os.path.join(LIBPATH, "geometry/domains")) -> str:
    """
    Constructs __init__.py for all domain files by reading actual class names.
    Skips __init__.py.
    """
    from struphy.geometry.base import Domain

    return construct_package_init_file(domains_dir, "struphy.geometry.domains", Domain)


def struphy_build_init_files(verbose=False):
    """Regenerate the auto-generated __init__.py files and format them with ruff.

    This (re-)writes struphy/models/__init__.py, struphy/propagators/__init__.py and
    struphy/geometry/domains/__init__.py based on the StruphyModel/Propagator/Domain
    subclasses found in those packages, preserving each file's existing module docstring.

    Parameters
    ----------
    verbose : bool
        If True, print the ruff commands that are run.
    """
    init_files = {
        os.path.join(LIBPATH, "models", "__init__.py"): construct_models_init_file,
        os.path.join(LIBPATH, "propagators", "__init__.py"): construct_propagators_init_file,
        os.path.join(LIBPATH, "geometry", "domains", "__init__.py"): construct_domains_init_file,
    }

    for path, construct in init_files.items():
        logger.info(f"Rewriting {path}")
        content = construct()
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    # sort imports, then format
    for ruff_args in (["check", "--fix", "--select", "I"], ["format"]):
        command = ["ruff", *ruff_args, *init_files]
        if verbose:
            logger.info(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True)
