from dataclasses import dataclass
from typing import Literal, get_args

# needed for import in StruphyParameters
from struphy.fields_background import equils
from struphy.geometry import domains
from struphy.topology import grids

SplitAlgos = Literal["LieTrotter", "Strang"]
PolarRegularity = Literal[-1, 1]


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
