"""Array-only mesh sampling and portable, non-pickle geometry archives."""

import io
import json

import numpy as np

from struphy.geometry import domains
from struphy.geometry.base import PoloidalSplineStraight, PoloidalSplineTorus, Spline

_ANALYTIC = (
    "Cuboid",
    "Orthogonal",
    "Colella",
    "HollowCylinder",
    "PoweredEllipticCylinder",
    "HollowTorus",
    "ShafranovShiftCylinder",
    "ShafranovSqrtCylinder",
    "ShafranovDshapedCylinder",
)


def sample_surface(domain, resolution=32, radial_coordinate=1.0):
    """Return x/y/z arrays of shape (resolution, resolution) at fixed eta1.

    This samples a logical coordinate surface, not necessarily the entire domain
    boundary. It is the outer radial surface for cylindrical/toroidal mappings.
    Use the domain directly to sample other faces, slices, or volume grids.
    """
    if not isinstance(resolution, (int, np.integer)) or resolution < 2:
        raise ValueError("resolution must be an integer >= 2")
    if not 0 <= radial_coordinate <= 1:
        raise ValueError("radial_coordinate must lie in [0, 1]")
    eta = np.linspace(0.0, 1.0, resolution)
    xyz = domain(float(radial_coordinate), eta, eta, squeeze_out=True)
    return tuple(np.asarray(component) for component in xyz)


def save_geometry(domain):
    """Return NPZ bytes containing an evaluated mapping's definition.

    Spline archives contain control points, never equilibrium objects. Loading
    GVEC/DESC/Tokamak-derived splines consequently needs no original solver or
    input file. The reconstructed object is a generic spline of the same kind.
    """
    metadata = {"format": "struphy-geometry", "version": 1}
    arrays = {}
    if domain.kind_map in (0, 1, 2):
        metadata.update(
            kind_map=int(domain.kind_map),
            num_elements=[int(x) for x in domain.num_elements],
            degree=[int(x) for x in domain.degree],
            spl_kind=[bool(x) for x in domain.spl_kind],
        )
        arrays = {key: np.asarray(getattr(domain, key)) for key in ("cx", "cy", "cz", "params_numpy")}
    elif type(domain).__name__ in _ANALYTIC:
        metadata.update(type=type(domain).__name__, params=domain.params)
    else:
        raise ValueError(f"Unsupported mapping type: {type(domain).__name__}")
    stream = io.BytesIO()
    np.savez_compressed(stream, metadata=np.array(json.dumps(metadata)), **arrays)
    return stream.getvalue()


def load_geometry(data):
    """Load a mapping from NPZ bytes produced by ``save_geometry`` (no pickle)."""
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata"]))
        if meta.get("format") != "struphy-geometry" or meta.get("version") != 1:
            raise ValueError("Unsupported geometry archive format/version")
        if "kind_map" not in meta:
            if meta.get("type") not in _ANALYTIC:
                raise ValueError("Unknown analytic mapping")
            return getattr(domains, meta["type"])(**meta["params"])
        kind = meta["kind_map"]
        if kind not in (0, 1, 2):
            raise ValueError("Unknown spline mapping kind")
        ndim = 3 if kind == 0 else 2
        kwargs = {key: tuple(meta[key][:ndim]) for key in ("num_elements", "degree", "spl_kind")}
        kwargs.update(cx=archive["cx"].copy(), cy=archive["cy"].copy())
        if kind == 0:
            return Spline(**kwargs, cz=archive["cz"].copy())
        # Domain extends poloidal control points with a singleton third axis.
        kwargs["cx"] = kwargs["cx"].reshape(kwargs["cx"].shape[:2])
        kwargs["cy"] = kwargs["cy"].reshape(kwargs["cy"].shape[:2])
        param = float(archive["params_numpy"][0])
        if kind == 1:
            return PoloidalSplineStraight(**kwargs, Lz=param)
        return PoloidalSplineTorus(**kwargs, tor_period=param)
