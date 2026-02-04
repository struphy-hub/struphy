import sys

from struphy.api import models
from struphy.api.domains import domains
from struphy.api.equils import equils
from struphy.api.grids import grids
from struphy.api.io_options import DerhamOptions, FieldsBackground
from struphy.api.maxwellians import maxwellians
from struphy.api.options import BaseUnits, EnvironmentOptions, Time
from struphy.api.perturbations import perturbations
from struphy.api.pic_utilities import (
    BinningPlot,
    BoundaryParameters,
    KernelDensityPlot,
    LoadingParameters,
    WeightsParameters,
)

# Expose submodules for easier access
sys.modules["struphy.models"] = models

__doc__ = """
Struphy: A Python framework for plasma physics simulations.
"""

# Public API
__all__ = [
    "models",
    "domains",
    "equils",
    "grids",
    "maxwellians",
    "EnvironmentOptions",
    "BaseUnits",
    "Time",
    "perturbations",
    "LoadingParameters",
    "WeightsParameters",
    "BoundaryParameters",
    "BinningPlot",
    "KernelDensityPlot",
    "DerhamOptions",
    "FieldsBackground",
]
