from gvec_to_python.base.base import Base

"""Helper functions."""



def make_base(data: dict, axis: str):
    """Recreate B-spline x Fourier basis given parameters and B-spline coefficients.

    Parameters
    ----------
    data : dict
        A `dict` containing GVEC output as given by `GVEC_Reader`.
    axis : str
        One of the "X1", "X2", "LA" axis.

    Returns
    -------
    Base
        A `Base` object which is a combination of B-spline x Fourier basis.
    """

    # Params for s_base.
    Nel  = data["grid"]["nElems"]      # Number of elements.
    el_b = data["grid"]["sGrid"]       # Element boundaries.
    p    = data[axis]["s_base"]["deg"] # Spline degree.
    bc   = False                       # Periodic boundary conditions (use 'False' if clamped).

    # Params for f_base.
    NFP          = data["general"]["nfp"]               # Number of field periods (symmetry).
    modes        = data[axis]["f_base"]["modes"]        # Number of all m-n mode combinations.
    sin_cos      = data[axis]["f_base"]["sin_cos"]      # Whether the data has only sine, only cosine, or both sine and cosine basis.
    excl_mn_zero = data[axis]["f_base"]["excl_mn_zero"] # 
    mn           = data[axis]["f_base"]["mn"]           # mn-mode number, with NFP premultiplied into the n-modes.
    mn_max       = data[axis]["f_base"]["mn_max"]       # Maximum m-mode and n-mode numbers, without NFP.
    modes_sin    = data[axis]["f_base"]["modes_sin"]    # Number of sine modes.
    modes_cos    = data[axis]["f_base"]["modes_cos"]    # Number of cosine modes.
    range_sin    = data[axis]["f_base"]["range_sin"]    # Index range of sine modes in `mn` list.
    range_cos    = data[axis]["f_base"]["range_cos"]    # Index range of cosine modes in `mn` list.

    base = Base(Nel, el_b, p, bc, NFP, modes, sin_cos, excl_mn_zero, mn, mn_max, modes_sin, modes_cos, range_sin, range_cos)

    return base
