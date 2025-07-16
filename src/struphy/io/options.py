from dataclasses import dataclass
from typing import Literal, get_args

# needed for import in StruphyParameters
from struphy.fields_background import equils
from struphy.geometry import domains
from struphy.initial import perturbations
from struphy.kinetic_background import maxwellians
from struphy.topology import grids

## generic options

SplitAlgos = Literal["LieTrotter", "Strang"]


@dataclass
class Units:
    """...

    Parameters
    ----------
    x : float
        Unit of length in m.
    """

    x: float = 1.0
    B: float = 1.0
    n: float = 1.0
    kBT: float = None


@dataclass
class Time:
    """...

    Parameters
    ----------
    x : float
        Unit of length in m.
    """

    dt: float = 1.0
    Tend: float = 1.0
    split_algo: SplitAlgos = "LieTrotter"

    def __post_init__(self):
        options = get_args(SplitAlgos)
        assert self.split_algo in options, f"'{self.split_algo}' is not in {options}"


## field options

PolarRegularity = Literal[-1, 1]
BackgroundOpts = Literal["LogicalConst", "FluidEquilibrium"]


@dataclass
class DerhamOptions:
    """...

    Parameters
    ----------
    x : float
        Unit of length in m.
    """

    polar_ck: int = -1
    local_projectors: bool = False

    def __post_init__(self):
        options = get_args(PolarRegularity)
        assert self.polar_ck in options, f"'{self.polar_ck}' is not in {options}"


@dataclass
class FieldsBackground:
    """...

    Parameters
    ----------
    x : float
        Unit of length in m.
    """

    kind: str = "LogicalConst"
    values: tuple = (1.5,)
    variable: str = None

    def __post_init__(self):
        options = get_args(BackgroundOpts)
        assert self.kind in options, f"'{self.kind}' is not in {options}"


## kinetic options
