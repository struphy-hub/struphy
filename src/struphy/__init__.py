import importlib
import importlib.metadata
import os
import re
import subprocess
import sys


def _install_psydac_if_needed():
    # Current directory is libpathe root of the project
    libpath = os.path.dirname(__file__)

    # install psydac from wheel if not there
    source_install = False
    for req in importlib.metadata.distribution("struphy").requires:
        if "psydac" in req:
            source_install = True

    # Get struphy version
    struphy_ver = importlib.metadata.version("struphy")

    # Check if psydac is installed
    try:
        import psydac

        psydac_ver = importlib.metadata.version("psydac")
        psydac_installed = True
    except ModuleNotFoundError:
        psydac_ver = None
        psydac_installed = False

    if source_install:
        # If we are installing psydac from Github
        if psydac_installed:
            # only install (from .whl) if psydac not up-to-date
            if psydac_ver < struphy_ver:
                print(
                    f"You have psydac version {psydac_ver}, but version {struphy_ver} is available. Please re-install struphy (e.g. pip install .)\n",
                )
                sys.exit(1)
        else:
            print("Psydac is not installed. To install it, please re-install struphy (e.g. pip install .)\n")
            sys.exit(1)
    else:
        # If we are installing psydac from a wheel
        install_psydac = False
        if psydac_installed:
            # only install (from .whl) if psydac not up-to-date
            if ".".join(psydac_ver.split(".")[:3]) != ".".join(struphy_ver.split(".")[:3]):
                print(f"You have psydac version {psydac_ver}, but version {struphy_ver} is required.\n")
                install_psydac = True
        else:
            install_psydac = True

        if install_psydac:
            psydac_file = None
            for filename in os.listdir(libpath):
                if re.match("psydac-", filename):
                    psydac_file = filename
            if psydac_file is None:
                raise FileNotFoundError("No psydac wheel file found.")

            # Uninstall psydac
            subprocess.run(["pip", "uninstall", "-y", "psydac"], check=True)

            # Install psydac
            print("\nInstalling Psydac ...")
            cmd = [
                "pip",
                "install",
                os.path.join(libpath, psydac_file),
            ]
            subprocess.run(cmd, check=True)


# Run on import
_install_psydac_if_needed()
