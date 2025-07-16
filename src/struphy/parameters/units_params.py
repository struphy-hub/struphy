from dataclasses import dataclass

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
    kBT: float = 1.0