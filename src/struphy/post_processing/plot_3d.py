"""3D plots of FEEC fields from the VTK files of a run (written by ``Output.pproc(create_vtk=True)``), with PyVista.

:func:`plot_field_3d` renders one field at one saved time step in the physical domain: a cutaway view, where a wedge of 90
degrees around the vertical axis is removed, one of its faces passing through the largest absolute value, so that
the field is visible inside (for a torus, the poloidal cross-section through the maximum).
The resolution of the plot is that of the VTK files, set by ``celldivide`` of ``Output.pproc``. Vector fields are
shown by their magnitude; they need the Cartesian components of ``pproc(physical=True)``.

Run as a script to plot a saved run::

    python -m struphy.post_processing.plot_3d <sim_folder> <variable> [--step -1] [-o file.png] [--colorbar]
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.image import imsave


def vtk_files(path_out, species="em_fields", physical=False) -> list[Path]:
    """The VTK files of ``species`` in the run ``path_out``, ordered by time step. ``physical`` selects the Cartesian
    components of vector fields (``pproc(physical=True)``) instead of their logical components."""
    folder = Path(path_out) / "post_processing" / "fields_data" / species / ("vtk_phy" if physical else "vtk")
    files = sorted(folder.glob("step_*.vts"), key=lambda f: int(re.search(r"step_(\d+)", f.name).group(1)))
    if not files:
        raise FileNotFoundError(f"No VTK files in {folder}; post-process the run with Output.pproc(create_vtk=True).")
    return files


def close_periodic(grid: pv.StructuredGrid) -> pv.StructuredGrid:
    """``grid`` with its periodic directions closed: the VTK files do not repeat the first point of a periodic
    direction (e.g. the angles of a cylinder or torus) at its end, which leaves a slit of one cell in the surface.
    A direction is periodic when its last slice of points is about one cell away from its first one; the first
    slice is then appended at its end."""
    dims = grid.dimensions
    coords = np.stack([grid.points[:, k].reshape(dims, order="F") for k in range(3)], axis=-1)
    data = {name: grid.point_data[name].reshape(dims, order="F") for name in grid.point_data}
    for axis in range(3):
        if dims[axis] < 3:
            continue
        first, second, last = (np.take(coords, i, axis=axis) for i in (0, 1, -1))
        gap = np.max(np.linalg.norm(last - first, axis=-1))
        cell = np.max(np.linalg.norm(second - first, axis=-1))
        if 0 < gap < 1.5 * cell:
            coords = np.concatenate([coords, np.take(coords, [0], axis=axis)], axis=axis)
            data = {n: np.concatenate([v, np.take(v, [0], axis=axis)], axis=axis) for n, v in data.items()}
    closed = pv.StructuredGrid(*(coords[..., k] for k in range(3)))
    for name, values in data.items():
        closed.point_data[name] = values.reshape(-1, order="F")
    return closed


def load_field(path_out, variable, species="em_fields", step=-1) -> tuple[pv.StructuredGrid, str]:
    """The grid of the saved time step ``step`` (an index into the saved steps, -1 for the last one) with the field
    as active scalars, and the name of these scalars: ``variable`` for a scalar field, ``|variable|`` for the
    magnitude of a vector field."""
    grid = close_periodic(pv.read(vtk_files(path_out, species)[step]))
    if variable in grid.point_data:
        grid.set_active_scalars(variable)
        return grid, variable
    if f"{variable}_1" not in grid.point_data:
        raise KeyError(f"No field {variable!r} in {species}; available: {sorted(grid.point_data.keys())}.")
    # vector field: the Cartesian components are written by pproc(physical=True)
    try:
        phy = close_periodic(pv.read(vtk_files(path_out, species, physical=True)[step]))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"{variable!r} is a vector field; its magnitude needs the Cartesian components of "
            "Output.pproc(create_vtk=True, physical=True)."
        ) from error
    name = f"|{variable}|"
    values = np.linalg.norm(np.stack([phy.point_data[f"{variable}_{k}"] for k in (1, 2, 3)]), axis=0)
    # the push-forward is singular on the axis of polar domains (eta1 = 0): take the magnitude of the next ring
    values = values.reshape(grid.dimensions, order="F")
    values[0] = np.where(np.isfinite(values[0]), values[0], values[1])
    grid.point_data[name] = values.reshape(-1, order="F")
    grid.set_active_scalars(name)
    return grid, name


def wedge_distance(points, center, azimuth):
    """``min(start, left)`` of the ``points``, with ``start`` and ``left`` their distances to the two faces of the
    wedge of 90 degrees around the vertical axis through ``center`` that starts at ``azimuth`` (the vertical half
    planes at ``azimuth`` and ``azimuth + pi/2``): positive inside the wedge."""
    rel = np.asarray(points) - np.asarray(center)
    start = np.cos(azimuth) * rel[:, 0] + np.sin(azimuth) * rel[:, 1]  # along the first face
    left = -np.sin(azimuth) * rel[:, 0] + np.cos(azimuth) * rel[:, 1]  # normal to the first face, into the wedge
    return np.minimum(start, left)


def cut_wedge(grid, center, azimuth):
    """``grid`` without the wedge of 90 degrees around the vertical axis through ``center`` that starts at
    ``azimuth`` (see :func:`wedge_distance`), clipped in one go so that no other seams arise."""
    marked = grid.copy()
    marked.point_data["wedge"] = wedge_distance(grid.points, center, azimuth)
    active = grid.active_scalars_name
    shown = marked.clip_scalar(scalars="wedge", value=0.0, invert=True)
    shown.set_active_scalars(active)
    return shown


# colormaps of the 3D plots: zero is light gray, positive values red, negative values blue
ZERO_COLOR = "#d3d3d3"
POSITIVE = LinearSegmentedColormap.from_list(
    "struphy_positive", [ZERO_COLOR, "#fcae91", "#fb6a4a", "#cb181d", "#67000d"]
)
NEGATIVE = LinearSegmentedColormap.from_list(
    "struphy_negative", ["#08306b", "#2171b5", "#6baed6", "#c6dbef", ZERO_COLOR]
)
SIGNED = LinearSegmentedColormap.from_list(
    "struphy_signed",
    ["#053061", "#2166ac", "#4393c3", "#92c5de", ZERO_COLOR, "#f4a582", "#d6604d", "#b2182b", "#67001f"],
)


def default_colormap(vmin, vmax, noise=1e-2):
    """Colormap and color limits in which zero is light gray: a diverging map (blue - gray - red) symmetric about
    zero for a field that changes sign, a sequential map from zero (gray - red, or blue - gray for negative values)
    for a field of one sign. Values of the other sign below ``noise`` times the largest absolute value are
    ignored."""
    scale = max(abs(vmin), abs(vmax)) or 1.0
    if vmin < -noise * scale and vmax > noise * scale:
        return SIGNED, (-scale, scale)
    if vmax > noise * scale:
        return POSITIVE, (0.0, vmax)
    return NEGATIVE, (vmin, 0.0)


def crop_image(image, margin=10, background=240):
    """``image`` (an RGB array) cropped to the pixels darker than ``background`` (nearly white pixels, e.g. at the
    border of an anti-aliased render, count as background), plus ``margin``."""
    rows, cols = np.nonzero(np.any(image[..., :3] < background, axis=-1))
    if rows.size == 0:
        return image
    top, bottom = max(rows.min() - margin, 0), min(rows.max() + margin + 1, image.shape[0])
    left, right = max(cols.min() - margin, 0), min(cols.max() + margin + 1, image.shape[1])
    return image[top:bottom, left:right]


# domain classes of tori around the vertical axis x = y = 0 (with their subclasses)
TORUS_DOMAINS = ("HollowTorus", "PoloidalSplineTorus", "IGAPolarTorus", "Tokamak", "GVECunit", "DESCunit")


def is_torus(path_out, grid=None) -> bool:
    """Whether the domain of the run is a torus around the vertical axis x = y = 0 (one of :data:`TORUS_DOMAINS` or
    a subclass). Without the simulation data of the run, guessed from ``grid``: no point close to that axis."""
    try:
        from struphy.post_processing.output import Output

        domain = Output(path_out).domain
    except Exception:
        if grid is None:
            return False
        horizontal = np.hypot(grid.points[:, 0], grid.points[:, 1])
        return bool(np.min(horizontal) > 0.1 * np.max(horizontal))
    return any(cls.__name__ in TORUS_DOMAINS for cls in type(domain).__mro__)


def plot_field_3d(
    path_out,
    variable,
    *,
    species="em_fields",
    step=-1,
    filename=None,
    show=False,
    cutaway=True,
    title=None,
    cmap=None,
    window_size=(1600, 1600),
    crop=True,
    margin=10,
    show_axes=False,
    show_scalar_bar=False,
    context=1.2,
):
    """Render ``variable`` of ``species`` at the saved time step ``step`` in 3D.

    Parameters
    ----------
    path_out : str | Path
        Output folder of the run.
    variable : str
        Name of the field, e.g. ``"phi"`` or ``"e_field"`` (vector fields are shown by their magnitude).
    species : str
        Species of the field (``"em_fields"`` for the electromagnetic fields).
    step : int
        Index of the saved time step (-1 for the last one).
    filename : str | Path | None
        Save the image there (e.g. ``"phi_3d.png"``).
    show : bool
        Open an interactive window (otherwise render off screen).
    cutaway : bool
        Remove a wedge of 90 degrees around the vertical axis through the center of the domain, starting at the
        azimuth of the largest absolute value, to show the field inside (one face of the cut passes through it).
    title : str | None
        Title of the image (default: none).
    cmap : str | None
        Matplotlib colormap (default: chosen by :func:`default_colormap`, with zero light gray).
    window_size : tuple[int, int]
        Size of the render window in pixels.
    crop : bool
        Crop the saved image to the rendered object (plus ``margin`` pixels).
    margin : int
        Margin of the cropped image in pixels.
    show_axes : bool
        Show the orientation of the x-y-z axes.
    show_scalar_bar : bool
        Show the color bar.
    context : float
        The camera frames the removed wedge enlarged by this factor, i.e. the cut and some of the domain around it
        for the 3D context (the whole domain is framed without ``cutaway``). For a domain around the vertical axis
        (a torus), the face of the cut through the largest absolute value is framed instead, enlarged by twice
        this factor.

    Returns
    -------
    pyvista.Plotter
        The plotter (closed unless ``show``).
    """
    grid, scalars = load_field(path_out, variable, species, step)
    default_cmap, clim = default_colormap(*grid.get_data_range(scalars))
    cmap = cmap or default_cmap

    # the vertical axis of the cut: through the center of the domain, for a torus its axis x = y = 0
    x0, x1, y0, y1, z0, z1 = grid.bounds
    center = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1)])
    diagonal = np.linalg.norm([x1 - x0, y1 - y0, z1 - z0])
    toroidal = is_torus(path_out, grid)
    if toroidal:
        center[:2] = 0.0
    # the azimuth of the largest |value| around that axis
    offset = grid.points[np.nanargmax(np.abs(grid.point_data[scalars]))] - center
    azimuth = np.arctan2(offset[1], offset[0]) if np.hypot(offset[0], offset[1]) > 1e-6 * diagonal else 0.0

    # merge coincident points (periodic seams, the axis of polar domains), so that the seams are not drawn as faces
    grid = grid.cast_to_unstructured_grid().clean(tolerance=1e-9 * diagonal)

    plotter = pv.Plotter(off_screen=not show, window_size=list(window_size))
    plotter.set_background("white")
    plotter.enable_anti_aliasing("ssaa")
    shown = cut_wedge(grid, center, azimuth) if cutaway else grid
    plotter.add_mesh(
        shown,
        scalars=scalars,
        cmap=cmap,
        clim=clim,
        smooth_shading=True,
        # lighting that keeps the colors of the colormap on surfaces facing the light (zero stays light gray)
        ambient=0.2,
        diffuse=0.8,
        show_scalar_bar=show_scalar_bar,
        scalar_bar_args={"title": scalars, "vertical": True, "color": "black"},
    )
    if show_axes:
        plotter.add_axes()
    if title is not None:
        plotter.add_text(title, font_size=12, color="black")
    # look into the wedge from above, framing the removed wedge (between the two faces of the cut) and some of the
    # domain around it, so that the cut fills the view and the rest of the domain gives the 3D context
    # elevation of the camera above the horizontal plane: about 39 degrees, half of it for a torus
    elevation = np.arctan(0.8) / (2 if toroidal else 1)
    # azimuth of the camera: along the bisector of the wedge, for a torus more in front of the cut face at `azimuth`
    # (whose normal points to azimuth + pi/2)
    turn = 3 * np.pi / 8 if toroidal else np.pi / 4
    view = np.array([np.cos(azimuth + turn), np.sin(azimuth + turn), np.tan(elevation)])
    plotter.camera.focal_point = center
    plotter.camera.position = center + 2.0 * diagonal * view / np.linalg.norm(view)
    plotter.camera.up = (0.0, 0.0, 1.0)
    plotter.reset_camera()
    if cutaway:
        framed = grid.points[wedge_distance(grid.points, center, azimuth) > 0]
        # a domain around the vertical axis (a torus): frame only the face of the cut through the largest |value|
        # (a poloidal cross-section), enlarged by twice the context
        rel = shown.points - center
        if toroidal:
            start = np.cos(azimuth) * rel[:, 0] + np.sin(azimuth) * rel[:, 1]
            left = -np.sin(azimuth) * rel[:, 0] + np.cos(azimuth) * rel[:, 1]
            face = shown.points[(np.abs(left) < 1e-6 * diagonal) & (start > 0)]
            if len(face):
                framed, context = face, 2 * context
        if len(framed):
            lo, hi = framed.min(axis=0), framed.max(axis=0)
            mid, half = 0.5 * (lo + hi), 0.5 * context * (hi - lo)
            plotter.reset_camera(
                bounds=[
                    mid[0] - half[0],
                    mid[0] + half[0],
                    mid[1] - half[1],
                    mid[1] + half[1],
                    mid[2] - half[2],
                    mid[2] + half[2],
                ]
            )

    if filename is not None:
        image = plotter.screenshot(return_img=True)
        if crop:
            image = crop_image(image, margin)
        imsave(filename, image)
    if show:
        plotter.show()
    else:
        plotter.close()
    return plotter


def main():
    parser = argparse.ArgumentParser(description="3D plot of a field of a saved run.")
    parser.add_argument("path_out", help="Simulation output folder.")
    parser.add_argument("variable", help='Name of the field, e.g. "phi" or "e_field".')
    parser.add_argument("--species", default="em_fields", help="Species of the field (default: em_fields).")
    parser.add_argument("--step", type=int, default=-1, help="Index of the saved time step (default: -1, the last).")
    parser.add_argument("-o", "--output", default=None, help="Save the image to this file instead of showing it.")
    parser.add_argument("--no-cutaway", action="store_true", help="Show the whole domain without cutaway.")
    parser.add_argument("--no-crop", action="store_true", help="Save the whole render window, without cropping.")
    parser.add_argument("--axes", action="store_true", help="Show the orientation of the x-y-z axes.")
    parser.add_argument("--colorbar", action="store_true", help="Show the color bar.")
    parser.add_argument("--title", default=None, help="Title of the image.")
    parser.add_argument(
        "--context", type=float, default=1.2, help="Framed region: the removed wedge enlarged by this factor."
    )
    args = parser.parse_args()
    plot_field_3d(
        args.path_out,
        args.variable,
        species=args.species,
        step=args.step,
        filename=args.output,
        show=args.output is None,
        cutaway=not args.no_cutaway,
        title=args.title,
        crop=not args.no_crop,
        show_axes=args.axes,
        show_scalar_bar=args.colorbar,
        context=args.context,
    )


if __name__ == "__main__":
    main()
