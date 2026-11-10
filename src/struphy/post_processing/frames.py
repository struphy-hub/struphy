"""Frames of videos of FEEC fields: one PNG image per saved time step, for a video made e.g. with ffmpeg (see
:func:`frames_to_video`).

* :func:`write_slice_frames`: the field in a physical plane (e.g. the poloidal plane ``rz`` of a torus, see
  :func:`struphy.post_processing.slices.physical_planes`), with matplotlib;
* :func:`write_3d_frames`: the field in the 3D domain with a cutaway (see
  :func:`struphy.post_processing.plot_3d.plot_field_3d`), with PyVista.

What is shown is the same for both: a scalar field, one ``component`` of a vector field in a ``basis`` (``"norm"``
by default: the components along the normalized logical basis vectors, e.g. the radial and poloidal components in a
torus with orthogonal coordinates; ``"1"``, ``"2"``, ``"v"`` or ``"physical"`` for the Cartesian components), or the
magnitude of a vector field. The fields are evaluated from the saved spline coefficients (no post-processing of the
run is needed). All frames of a video share the colors (one color scale, fixed over the video by default), the view
and the image size, and are written as ``frame_0000.png``, ``frame_0001.png``, ... into one folder, at most
``max_frames`` of them (evenly spaced saved time steps, see :func:`frame_steps`), together with ``frames.txt``, which
lists the time step and the time of each frame.

Example (a pproc script)::

    from struphy import Output
    from struphy.post_processing.frames import frames_to_video, write_3d_frames, write_slice_frames

    run = Output("sim_data")
    write_slice_frames(run, "mhd/velocity", "frames/u_r_rz", component=0, label="u_r")
    write_3d_frames(run, "mhd/velocity", "frames/u_r_3d", component=0, label="u_r")
    frames_to_video("frames/u_r_rz")  # frames/u_r_rz.mp4, needs ffmpeg

The frames are written by the process of rank 0 only (the other ranks of an MPI run return an empty list).
"""

import shutil
import subprocess
from pathlib import Path

import cunumpy as xp
import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.image import imsave
from maybempi import MPI
from scipy.spatial import cKDTree

from struphy.post_processing.output import Output
from struphy.post_processing.plot_3d import (
    TORUS_DOMAINS,
    add_field_mesh,
    crop_box,
    cut_center,
    cut_wedge,
    default_colormap,
    merge_seams,
    new_plotter,
    off_grid_azimuth,
    peak_azimuth,
    repeat_around_axis,
    set_camera,
)
from struphy.post_processing.slices import BASES, evaluate_plane, physical_planes, rotate

VECTOR_FORMS = ("1", "2", "v")


def saved_times(out: Output) -> np.ndarray:
    """The times of all saved time steps of the run (from the raw output, no post-processing needed)."""
    with h5py.File(out.path_out / "data" / "data_proc0.hdf5", "r") as file:
        return np.asarray(file["time/value"][:], dtype=float).reshape(-1)


def frame_steps(out: Output, max_frames=500) -> list[int]:
    """Positions of at most ``max_frames`` saved time steps, evenly spaced over the run, with the first and the last
    one: all saved time steps if there are not more than ``max_frames``."""
    count = len(saved_times(out))
    if count <= max_frames:
        return list(range(count))
    return sorted({int(n) for n in np.round(np.linspace(0, count - 1, max_frames))})


def field_label(out: Output, name, component=None, basis=None) -> str:
    """Default label of what is shown of the field ``name``: e.g. ``velocity_1 (normalized basis)`` or
    ``|velocity|``."""
    variable = name.split("/")[-1]
    if out.field_form(name) not in VECTOR_FORMS:
        return variable
    if component is None:
        return f"|{variable}|"
    basis = basis or "norm"
    return f"{variable}_{component + 1} ({BASES.get(basis, basis)})"


def _kind(out: Output, name, component, basis):
    """``(kind, components)`` of :func:`struphy.post_processing.slices.evaluate_plane` for what is shown."""
    form = out.field_form(name)
    if form not in VECTOR_FORMS:
        return None, False
    if component is None:
        return form, False
    return basis or "norm", True


def _limits(clim, values, vmin, vmax, cmap):
    """Colormap and color limits of a frame with ``values``: the fixed ``clim`` ``(vmin, vmax)``, else from the range
    of ``values`` (``clim="frame"``) or the range ``vmin``, ``vmax`` of all frames (``clim`` None), see
    :func:`struphy.post_processing.plot_3d.default_colormap`."""
    if clim is not None and clim != "frame":
        return cmap or default_colormap(*clim)[0], tuple(clim)
    if clim == "frame":
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    default_cmap, limits = default_colormap(vmin, vmax)
    return cmap or default_cmap, limits


def _check_clim(clim):
    if not (clim is None or clim == "frame" or (len(clim) == 2 and clim[0] < clim[1])):
        raise ValueError(f'clim must be None, "frame" or (vmin, vmax) with vmin < vmax, got {clim!r}.')


def _even(n):
    """``n`` rounded down to an even number (video encoders need even image sizes)."""
    return n - n % 2


def _figure(width, height, dpi, **kwargs):
    """A figure of ``width`` x ``height`` pixels (half a pixel more, against the truncation of the size of the
    canvas; see :func:`_save_figure`)."""
    return plt.figure(figsize=((width + 0.5) / dpi, (height + 0.5) / dpi), dpi=dpi, facecolor="white", **kwargs)


def _save_figure(fig, path, width, height):
    """Save ``fig`` (of :func:`_figure`) as an image of exactly ``width`` x ``height`` pixels and close it."""
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:height, :width, :3]
    plt.close(fig)
    imsave(path, image)


def _prepare_folder(folder) -> Path:
    """``folder``, created, without the frames of a previous call."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("frame_*.png"):
        old.unlink()
    return folder


def _write_index(folder, steps, times):
    with open(folder / "frames.txt", "w") as file:
        file.write("# frame  saved_step  time\n")
        for k, (n, t) in enumerate(zip(steps, times)):
            file.write(f"{k:5d}  {n:10d}  {t:.10g}\n")


def _frame_name(k, count):
    return f"frame_{k:0{max(4, len(str(count - 1)))}d}.png"


def write_slice_frames(
    out: Output,
    name,
    folder,
    *,
    component=None,
    basis=None,
    plane="rz",
    phi=np.pi,
    steps=None,
    max_frames=500,
    clim=None,
    cmap=None,
    label=None,
    points_per_cell=4,
    height=6.0,
    dpi=150,
) -> list[Path]:
    """Write the frames of a video of the FEEC field ``name`` in a physical plane.

    Parameters
    ----------
    out : Output
        Output of the run.
    name : str
        Raw FEEC field, ``"species/variable"`` (e.g. ``"mhd/velocity"``).
    folder : str | Path
        Folder of the frames (created; frames of a previous call are removed).
    component : int | None
        Component (0, 1 or 2) of a vector field in ``basis``; None for the magnitude of a vector field (and for
        scalar fields).
    basis : str | None
        Basis of ``component``: ``"norm"`` (default), ``"1"``, ``"2"``, ``"v"`` or ``"physical"`` (Cartesian).
    plane : str
        The plane, one of :func:`struphy.post_processing.slices.physical_planes` (``"xy"``, ``"xz"``, ``"yz"``, or
        ``"rz"`` for cylinders and tori).
    phi : float
        Angle of the poloidal plane ``rz``.
    steps : list[int] | None
        Positions of the saved time steps; by default :func:`frame_steps` with ``max_frames``.
    max_frames : int
        Largest number of frames.
    clim : None | "frame" | tuple[float, float]
        Color limits: None for one scale over all frames (symmetric about zero when the field changes sign),
        ``"frame"`` for a scale per frame, or fixed ``(vmin, vmax)``.
    cmap : str | None
        Matplotlib colormap (default: zero light gray, see
        :func:`struphy.post_processing.plot_3d.default_colormap`).
    label : str | None
        Name of what is shown, in the titles (default: :func:`field_label`).
    points_per_cell : int
        Evaluation points per cell of the simulation grid.
    height : float
        Height of the frames in inches (the width follows from the shape of the plane).
    dpi : int
        Resolution of the frames.

    Returns
    -------
    list[Path]
        The frames, in order.
    """
    _check_clim(clim)
    if MPI.COMM_WORLD.Get_rank() != 0:
        return []
    planes = physical_planes(out.domain, phi, all_planes=True)
    if plane not in planes:
        raise ValueError(f"No plane {plane!r} for {type(out.domain).__name__}; available: {list(planes)}.")
    plane_def = planes[plane]
    steps = frame_steps(out, max_frames) if steps is None else list(steps)
    times = saved_times(out)[steps]
    kind, components = _kind(out, name, component, basis)
    label = label or field_label(out, name, component, basis)
    n_pts = points_per_cell * max(out.grid.num_elements) + 1

    values = []
    for n in steps:
        num, _, coords = evaluate_plane(out, out.domain, name, n, None, plane_def, n_pts, kind, components)
        values.append(num[component] if components else num)
    vmin = min(float(np.nanmin(v)) for v in values)
    vmax = max(float(np.nanmax(v)) for v in values)

    axis = ("eta1", "eta2", "eta3").index(plane_def.cut)
    h, v = coords[plane_def.hlabel], coords[plane_def.vlabel]
    aspect = (np.ptp(v) or 1.0) / (np.ptp(h) or 1.0)
    # the plane, plus room for the color bar and the labels; even numbers of pixels
    width_px = _even(int(dpi * (height / max(aspect, 0.2) + 1.6)))
    height_px = _even(int(dpi * height))
    folder = _prepare_folder(folder)
    files = []
    for k, (data, t) in enumerate(zip(values, times)):
        frame_cmap, (lo, hi) = _limits(clim, data, vmin, vmax, cmap)
        fig = _figure(width_px, height_px, dpi, layout="constrained")
        ax = fig.add_subplot()
        for i in range(data.shape[axis]):
            pcm = ax.pcolormesh(
                np.take(h, i, axis=axis),
                np.take(v, i, axis=axis),
                np.take(data, i, axis=axis),
                shading="gouraud",
                cmap=frame_cmap,
                vmin=lo,
                vmax=hi,
            )
        fig.colorbar(pcm, ax=ax)
        ax.set_aspect("equal")
        ax.set_xlabel(plane_def.hlabel)
        ax.set_ylabel(plane_def.vlabel)
        ax.set_title(f"{label}, {plane_def.hlabel}-{plane_def.vlabel} plane ({plane_def.where}), t = {t:.4g}")
        files.append(folder / _frame_name(k, len(steps)))
        _save_figure(fig, files[-1], width_px, height_px)
    _write_index(folder, steps, times)
    return files


def _grid_values(out: Output, name, n, etas, component, basis):
    """The values shown of the field ``name`` at the saved time step ``n`` on the tensor-product grid ``etas``:
    a dict of point arrays (shape of the grid; the three Cartesian components ``c1``, ``c2``, ``c3`` for the
    ``"physical"`` basis, so that copies of a torus sector can rotate them, else ``values``) and the Cartesian
    coordinates X, Y, Z of the grid."""
    form = out.field_form(name)
    vector = form in VECTOR_FORMS
    physical = vector and (component is None or basis == "physical")
    representation = form if physical else (basis or "norm") if vector else None
    data = out.evaluate(name, t=n, representation=representation, **etas)
    xyz = tuple(data.coords[c].values for c in ("X", "Y", "Z"))
    if not vector:
        return {"values": data.values}, xyz
    if not physical:
        return {"values": data.isel(component=component).values}, xyz
    grids = [data.coords[e].values for e in ("eta1", "eta2", "eta3")]
    cartesian = out.domain.push([data.isel(component=k).values for k in range(3)], *grids, kind=form)
    cartesian = [np.asarray(xp.to_numpy(c)) for c in cartesian]
    if component is None:
        return {"values": np.linalg.norm(np.stack(cartesian), axis=0)}, xyz
    return {f"c{k + 1}": c for k, c in enumerate(cartesian)}, xyz


def _shown(arrays, component):
    """The shown values from the arrays of :func:`_grid_values`."""
    return arrays["values"] if "values" in arrays else arrays[f"c{component + 1}"]


def _render_3d(grid, values, center, azimuth, toroidal, cmap, clim, options):
    """Image (RGB array) of ``grid`` (with merged seams) colored by ``values``, cut at ``azimuth``, without title and
    color bar; ``options`` are ``view``, ``window_size`` and ``context`` of :func:`write_3d_frames`."""
    scalars = "values"
    grid.point_data[scalars] = values
    grid.set_active_scalars(scalars)
    shown = cut_wedge(grid, center, azimuth)
    plotter = new_plotter(False, options["window_size"])
    add_field_mesh(plotter, shown, scalars, cmap, clim)
    set_camera(
        plotter, grid, shown, center, azimuth, toroidal, cutaway=options["view"] == "cut", context=options["context"]
    )
    image = plotter.screenshot(return_img=True)
    plotter.close()
    plotter.deep_clean()
    return image


def _compose(image, title, cmap, clim, label, show_colorbar, dpi=100):
    """Figure of ``image`` (pixel for pixel) with ``title`` above it and, with ``show_colorbar``, the color bar of
    ``cmap`` and ``clim`` on its right, and its width and height in pixels (even numbers)."""
    height, width = image.shape[:2]
    title_px = int(0.5 * dpi)
    bar_px = int(1.4 * dpi) if show_colorbar else 0
    fig_w, fig_h = width + bar_px, height + title_px
    fig_w, fig_h = fig_w + fig_w % 2, fig_h + fig_h % 2
    fig = _figure(fig_w, fig_h, dpi)
    ax = fig.add_axes((0.0, 0.0, width / fig_w, height / fig_h))
    ax.imshow(image, interpolation="none")
    ax.set_axis_off()
    fig.text(0.5 * width / fig_w, 1.0 - 0.5 * title_px / fig_h, title, ha="center", va="center", fontsize=13)
    if show_colorbar:
        cax = fig.add_axes(
            (width / fig_w + 0.1 * dpi / fig_w, 0.1 * height / fig_h, 0.2 * dpi / fig_w, 0.8 * height / fig_h)
        )
        fig.colorbar(ScalarMappable(norm=Normalize(*clim), cmap=cmap), cax=cax, label=label)
    return fig, fig_w, fig_h


def write_3d_frames(
    out: Output,
    name,
    folder,
    *,
    component=None,
    basis=None,
    steps=None,
    max_frames=500,
    clim=None,
    cmap=None,
    label=None,
    azimuth=None,
    points_per_cell=4,
    view="cut",
    window_size=(1200, 1200),
    margin=10,
    show_scalar_bar=True,
    context=1.2,
) -> list[Path]:
    """Write the frames of a video of the FEEC field ``name`` in 3D, with a cutaway: a wedge of 90 degrees around the
    vertical axis is removed (for a torus, one face of the cut is a poloidal cross-section), as in
    :func:`struphy.post_processing.plot_3d.plot_field_3d`. The view is the same in all frames.

    Parameters
    ----------
    out : Output
        Output of the run.
    name : str
        Raw FEEC field, ``"species/variable"`` (e.g. ``"mhd/velocity"``).
    folder : str | Path
        Folder of the frames (created; frames of a previous call are removed).
    component : int | None
        Component (0, 1 or 2) of a vector field in ``basis``; None for the magnitude of a vector field (and for
        scalar fields).
    basis : str | None
        Basis of ``component``: ``"norm"`` (default), ``"1"``, ``"2"``, ``"v"`` or ``"physical"`` (Cartesian).
    steps : list[int] | None
        Positions of the saved time steps; by default :func:`frame_steps` with ``max_frames``.
    max_frames : int
        Largest number of frames.
    clim : None | "frame" | tuple[float, float]
        Color limits: None for one scale over all frames (symmetric about zero when the field changes sign),
        ``"frame"`` for a scale per frame, or fixed ``(vmin, vmax)``.
    cmap : str | None
        Matplotlib colormap (default: zero light gray, see
        :func:`struphy.post_processing.plot_3d.default_colormap`).
    label : str | None
        Name of what is shown, in the titles (default: :func:`field_label`).
    azimuth : float | None
        Azimuth (angle about the vertical axis, in radians) of the face of the cut that the camera looks at; by
        default that of the largest absolute value over all frames. It is moved to the middle between two planes of
        the evaluation grid (see :func:`struphy.post_processing.plot_3d.off_grid_azimuth`).
    points_per_cell : int
        Evaluation points per cell of the simulation grid in each direction.
    view : str
        ``"cut"``: the camera frames the cut (for a torus its face at ``azimuth``, a poloidal cross-section) and some
        of the domain around it, see ``context``; ``"domain"``: the whole domain.
    window_size : tuple[int, int]
        Size of the render window in pixels; the render is cropped to the domain (the same box for all frames), the
        title and the color bar are added around it.
    margin : int
        Margin of the cropped frames in pixels.
    show_scalar_bar : bool
        Show the color bar (on the right of the image).
    context : float
        With ``view="cut"``, how much of the domain around the cut is framed, see
        :func:`struphy.post_processing.plot_3d.set_camera`.

    Returns
    -------
    list[Path]
        The frames, in order.
    """
    _check_clim(clim)
    if view not in ("cut", "domain"):
        raise ValueError(f'view must be "cut" or "domain", got {view!r}.')
    if MPI.COMM_WORLD.Get_rank() != 0:
        return []
    steps = frame_steps(out, max_frames) if steps is None else list(steps)
    times = saved_times(out)[steps]
    label = label or field_label(out, name, component, basis)
    etas = {
        e: np.linspace(0.0, 1.0, points_per_cell * n + 1)
        for e, n in zip(("eta1", "eta2", "eta3"), out.grid.num_elements)
    }
    domain = out.domain
    toroidal = any(cls.__name__ in TORUS_DOMAINS for cls in type(domain).__mro__)
    copies = int(domain.params.get("tor_period", 1)) if toroidal else 1

    def frame_grid(n):
        """The structured grid of the domain (the full torus) with the arrays of :func:`_grid_values`."""
        arrays, xyz = _grid_values(out, name, n, etas, component, basis)
        import pyvista as pv

        grid = pv.StructuredGrid(*xyz)
        for key, value in arrays.items():
            grid.point_data[key] = np.asarray(value, dtype=float).reshape(-1, order="F")
        grid.set_active_scalars(next(iter(arrays)))
        cartesian = ("c1", "c2") if "c1" in arrays else None
        grid = repeat_around_axis(grid, copies, cartesian)
        return grid, np.asarray(_shown(grid.point_data, component))

    # first pass: the range of the values and the point of the largest absolute value over all frames
    vmin, vmax, peak, peak_point = np.inf, -np.inf, -1.0, None
    for n in steps:
        grid, values = frame_grid(n)
        vmin, vmax = min(vmin, float(np.nanmin(values))), max(vmax, float(np.nanmax(values)))
        i = int(np.nanargmax(np.abs(values)))
        if abs(values[i]) > peak:
            peak, peak_point = abs(values[i]), grid.points[i]

    center, diagonal = cut_center(grid, toroidal)
    if azimuth is None:
        azimuth = peak_azimuth(peak_point[None], [1.0], center, diagonal)
    # not through a plane of grid points, which would leave dark specks of degenerate cells
    azimuth = off_grid_azimuth(grid.points, center, azimuth)
    # coincident points merged once; the values of each frame are taken from the nearest point of the grid
    merged = merge_seams(grid, diagonal)
    _, index = cKDTree(grid.points).query(merged.points)
    options = dict(view=view, window_size=window_size, context=context)

    def render(values):
        frame_cmap, limits = _limits(clim, values, vmin, vmax, cmap)
        image = _render_3d(merged, values[index], center, azimuth, toroidal, frame_cmap, limits, options)
        return image, frame_cmap, limits

    # the crop box, the same for all frames (the rendered domain does not change): that of the first frame
    _, values = frame_grid(steps[0])
    top, bottom, left, right = crop_box(render(values)[0], margin)

    folder = _prepare_folder(folder)
    files = []
    for k, (n, t) in enumerate(zip(steps, times)):
        _, values = frame_grid(n)
        image, frame_cmap, limits = render(values)
        title = f"{label}, t = {t:.4g}"
        fig, width, height = _compose(image[top:bottom, left:right], title, frame_cmap, limits, label, show_scalar_bar)
        files.append(folder / _frame_name(k, len(steps)))
        _save_figure(fig, files[-1], width, height)
    _write_index(folder, steps, times)
    return files


def frames_to_video(folder, output=None, fps=25, crf=18) -> Path:
    """Make an MP4 video (H.264) of the frames in ``folder`` with ffmpeg, which must be installed.

    Parameters
    ----------
    folder : str | Path
        Folder of the frames (``frame_0000.png``, ...).
    output : str | Path | None
        The video file (default: ``<folder>.mp4``).
    fps : int
        Frames per second.
    crf : int
        Quality (lower is better, 18 is visually lossless).

    Returns
    -------
    Path
        The video file.
    """
    folder = Path(folder)
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg not found; install it or make the video from the frames yourself.")
    frames = sorted(folder.glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError(f"No frames in {folder}.")
    digits = len(frames[0].stem.split("_")[1])
    output = Path(output) if output is not None else folder.with_suffix(".mp4")
    # even image sizes for yuv420p (the frames of this module have them already)
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(folder / f"frame_%0{digits}d.png"),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", str(crf), str(output),
    ]  # fmt: skip
    subprocess.run(command, check=True)
    return output
