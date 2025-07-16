from dataclasses import dataclass
from typing import Literal, get_args

polar_options = Literal[-1, 1]

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
        options = get_args(polar_options)
        assert self.polar_ck in options, f"'{self.polar_ck}' is not in {options}"