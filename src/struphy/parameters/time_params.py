from dataclasses import dataclass
from typing import Literal, get_args

SplitAlgos = Literal["LieTrotter", "Strang"]

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