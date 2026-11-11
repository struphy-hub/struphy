"""Frames of videos of FEEC fields: one PNG image per saved time step, for a video made e.g. with ffmpeg (see
:func:`frames_to_video`).

* :func:`write_slice_frames`: the field in a physical plane (e.g. the poloidal plane ``rz`` of a torus, see
  :func:`struphy.post_processing.slices.physical_planes`), with matplotlib;
* :func:`write_3d_frames`: the field in the 3D domain with a cutaway (see
  :func:`struphy.post_processing.plot_3d.plot_field_3d`), with PyVista;
* :func:`write_videos`: the standard videos of a field (its planes and both 3D views).

The planes are those of :func:`struphy.post_processing.slices.physical_planes`: ``rz`` for tori and cylinders,
``xy``, ``xz``, ``yz`` for a slab (a Cuboid), Colella or Orthogonal, the logical mid-surfaces ``eta3``, ``eta2`` for
other domains; domains covering only part of the poloidal angle have none (:func:`write_videos` then writes the 3D
views only). The 3D cutaway works for any domain: it removes a wedge around the vertical axis through the center of
the domain (the axis of a torus).

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
    # or all standard videos at once: frames/u_r_rz, frames/u_r_3d_cut, frames/u_r_3d
    write_videos(run, "mhd/velocity", "frames", component=0, prefix="u_r", label="u_r")

The frames are written by the process of rank 0 only (the other ranks of an MPI run return an empty list). The
fields are evaluated in that process; the frames are rendered and saved by ``workers`` processes (a pool of spawned
processes, see :func:`default_workers`), with the same result as in one process. Spawned processes import the main
script again, so a script that writes frames must run its code under ``if __name__ == "__main__":`` (otherwise, and in
MPI runs, the frames are rendered in one process, with a warning).
"""

import multiprocessing
import os
import shutil
import subprocess
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cunumpy as xp
import h5py
import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.image import imsave
from maybempi import MPI, launched_under_mpi
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
    wedge_distance,
)
from struphy.post_processing.slices import BASES, evaluate_plane, physical_planes, rotate

VECTOR_FORMS = ("1", "2", "v")
# default resolution of write_3d_frames: at most this many points of the 3D grid (with the copies of a torus sector)
MAX_POINTS_3D = 1e6
# the values of the first pass of write_3d_frames are kept for the rendering (of all views) up to this size in bytes
FIRST_PASS_CACHE = 1e9
# largest default number of rendering processes (see default_workers)
MAX_WORKERS = 4
# frames evaluated at once when the values are not kept (per rendering process)
_BATCH_PER_WORKER = 4


def saved_times(out: Output) -> np.ndarray:
    """The times of all saved time steps of the run (from the raw output, no post-processing needed), in the time
    units of ``out`` (see :meth:`Output.with_time_units`)."""
    with h5py.File(out.path_out / "data" / "data_proc0.hdf5", "r") as file:
        return np.asarray(file["time/value"][:], dtype=float).reshape(-1) * out.time_scale


def frame_steps(out: Output, max_frames=500) -> list[int]:
    """Positions of at most ``max_frames`` saved time steps, evenly spaced over the run, with the first and (for
    ``max_frames`` > 1) the last one: all saved time steps if there are not more than ``max_frames``."""
    if int(max_frames) < 1:
        raise ValueError(f"max_frames must be at least 1, got {max_frames}.")
    count = len(saved_times(out))
    if count <= max_frames:
        return list(range(count))
    return sorted({int(n) for n in np.round(np.linspace(0, count - 1, max_frames))})


def _steps_and_times(out: Output, steps, max_frames):
    """The positions of the saved time steps of the frames (``steps``, else :func:`frame_steps`) and their times."""
    steps = frame_steps(out, max_frames) if steps is None else [int(n) for n in steps]
    if not steps:
        raise ValueError("No time steps for the frames.")
    return steps, saved_times(out)[steps]


def _time_title(out: Output, t) -> str:
    """``t = <t>/<end time of the run> [unit]``, for the orientation in a video."""
    unit = out.time_unit
    return f"t = {t:.4g}/{saved_times(out)[-1]:.4g}" + (f" {unit}" if unit else "")


def _finite(values):
    """``values`` (eta1 along the first axis) as floats, with the non-finite values at eta1 = 0 (the singular axis
    of polar domains, e.g. of a Tokamak or a HollowTorus with a1 = 0) replaced by those of the next ring of points
    and the remaining non-finite values by NaN (not shown, ignored by the color scale)."""
    values = np.array(values, dtype=float)
    if values.ndim > 0 and values.shape[0] > 1:
        values[0] = np.where(np.isfinite(values[0]), values[0], values[1])
    values[~np.isfinite(values)] = np.nan
    return values


def _evaluate_uncached(out: Output, n, evaluate):
    """``evaluate()`` without keeping the spline functions of the saved time step ``n`` in the cache of ``out``
    (unless they were there before): the cache would hold all fields of up to ``max_frames`` time steps."""
    cache = getattr(out, "_spline_snapshots", None)
    key = n % len(saved_times(out))
    cached = cache is not None and key in cache
    try:
        return evaluate()
    finally:
        if cache is not None and not cached:
            cache.pop(key, None)


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
    if clim is not None and not isinstance(clim, str):
        return cmap or default_colormap(*clim)[0], (float(clim[0]), float(clim[1]))
    if clim == "frame":
        vmin, vmax = _range(values)
    default_cmap, (lo, hi) = default_colormap(vmin, vmax)
    if lo == hi:
        # a field that is zero (or constant) everywhere
        delta = abs(lo) or 1.0
        lo, hi = lo - delta, hi + delta
    return cmap or default_cmap, (lo, hi)


def _range(values):
    """Smallest and largest value, ignoring NaN; (0, 0) if all values are NaN."""
    if np.all(np.isnan(values)):
        return 0.0, 0.0
    return float(np.nanmin(values)), float(np.nanmax(values))


def _check_clim(clim):
    if clim is None or (isinstance(clim, str) and clim == "frame"):
        return
    if isinstance(clim, str) or len(clim) != 2 or not clim[0] < clim[1]:
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


def default_workers() -> int:
    """Default number of processes that render the frames: 1 in an MPI run (the cores belong to the ranks, and
    processes started from an MPI rank cannot use MPI), else the number of available cores, at most
    :data:`MAX_WORKERS`."""
    if launched_under_mpi() or MPI.COMM_WORLD.Get_size() > 1:
        return 1
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    return max(1, min(MAX_WORKERS, cores))


# the context of the frame jobs in a rendering process (set by _init_jobs)
_JOB_CONTEXT = {}


def _main_is_guarded() -> bool:
    """Whether processes can be spawned: the main script (if any) runs its code under ``if __name__ == "__main__":``.
    Spawned processes import the main script again; without the guard they would run it (and hang). Interactive
    sessions and notebooks have no main script."""
    path = getattr(sys.modules.get("__main__"), "__file__", None)
    if path is None:
        return True
    try:
        return "__main__" in Path(path).read_text()
    except (OSError, UnicodeDecodeError):
        return True


def _init_jobs(context):
    global _JOB_CONTEXT
    _JOB_CONTEXT = context


def _call_job(job, item):
    return job(_JOB_CONTEXT, *item)


def _run_jobs(job, context, make_item, count, workers, eager=True):
    """``job(context, *make_item(k))`` for ``k = 0, ..., count - 1``, in order, with ``workers`` processes (a pool of
    spawned processes, which get ``context`` once; serial for one worker). The items are made in this process, all at
    once (``eager``) or in batches of :data:`_BATCH_PER_WORKER` per process (when they are large)."""
    workers = max(1, min(int(workers), count))
    if workers > 1 and (launched_under_mpi() or MPI.COMM_WORLD.Get_size() > 1):
        # processes started from an MPI rank import MPI and hang
        warnings.warn(
            "Rendering frames in one process in an MPI run (workers > 1 needs a run without MPI).", stacklevel=3
        )
        workers = 1
    if workers > 1 and not _main_is_guarded():
        warnings.warn(
            'Rendering frames in one process: the main script has no `if __name__ == "__main__":` guard, which '
            "the rendering processes (spawned) need.",
            stacklevel=3,
        )
        workers = 1
    if workers == 1:
        return [job(context, *make_item(k)) for k in range(count)]
    results = []
    batch = count if eager else _BATCH_PER_WORKER * workers
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_jobs,
        initargs=(context,),
    ) as pool:
        for start in range(0, count, batch):
            items = [make_item(k) for k in range(start, min(start + batch, count))]
            results.extend(pool.map(_call_job, [job] * len(items), items))
    return results


def _slice_job(ctx, k, data, title):
    """Render and save frame ``k`` of a video in a plane (see :func:`write_slice_frames`)."""
    frame_cmap, (lo, hi) = _limits(ctx["clim"], data, ctx["vmin"], ctx["vmax"], ctx["cmap"])
    fig = _figure(ctx["width_px"], ctx["height_px"], ctx["dpi"], layout="constrained")
    ax = fig.add_subplot()
    axis, h, v = ctx["axis"], ctx["h"], ctx["v"]
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
    ax.set_xlabel(ctx["hlabel"])
    ax.set_ylabel(ctx["vlabel"])
    ax.set_title(title)
    path = ctx["folder"] / _frame_name(k, ctx["count"])
    _save_figure(fig, path, ctx["width_px"], ctx["height_px"])
    return path


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
    workers=None,
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
        The plane, one of :func:`struphy.post_processing.slices.physical_planes` (``"eta1"``, ``"eta2"``, ``"eta3"``
        for the domains without physical planes; else ``"xy"``, ``"xz"``, ``"yz"``, or
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
    workers : int | None
        Number of processes that render the frames (default: :func:`default_workers`).

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
    steps, times = _steps_and_times(out, steps, max_frames)
    kind, components = _kind(out, name, component, basis)
    label = label or field_label(out, name, component, basis)
    n_pts = points_per_cell * max(out.grid.num_elements) + 1

    values = []
    for n in steps:
        num, _, coords = _evaluate_uncached(
            out, n, lambda: evaluate_plane(out, out.domain, name, n, None, plane_def, n_pts, kind, components)
        )
        values.append(_finite(num[component] if components else num))
    vmin = min(_range(v)[0] for v in values)
    vmax = max(_range(v)[1] for v in values)

    axis = ("eta1", "eta2", "eta3").index(plane_def.cut)
    h, v = coords[plane_def.hlabel], coords[plane_def.vlabel]
    aspect = (np.ptp(v) or 1.0) / (np.ptp(h) or 1.0)
    # the plane, plus room for the color bar and the labels; even numbers of pixels
    width_px = _even(int(dpi * (height / max(aspect, 0.2) + 1.6)))
    height_px = _even(int(dpi * height))
    folder = _prepare_folder(folder)
    context = dict(
        clim=clim,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        width_px=width_px,
        height_px=height_px,
        dpi=dpi,
        axis=axis,
        h=h,
        v=v,
        hlabel=plane_def.hlabel,
        vlabel=plane_def.vlabel,
        folder=folder,
        count=len(steps),
    )
    where = f"{plane_def.hlabel}-{plane_def.vlabel} plane ({plane_def.where})"
    titles = [f"{label}, {where}, {_time_title(out, t)}" for t in times]
    files = _run_jobs(
        _slice_job,
        context,
        lambda k: (k, values[k], titles[k]),
        len(steps),
        default_workers() if workers is None else workers,
    )
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
    data = _evaluate_uncached(out, n, lambda: out.evaluate(name, t=n, representation=representation, **etas))
    xyz = tuple(data.coords[c].values for c in ("X", "Y", "Z"))
    if not vector:
        return {"values": _finite(data.values)}, xyz
    if not physical:
        return {"values": _finite(data.isel(component=component).values)}, xyz
    grids = [data.coords[e].values for e in ("eta1", "eta2", "eta3")]
    cartesian = out.domain.push([data.isel(component=k).values for k in range(3)], *grids, kind=form)
    cartesian = [_finite(xp.to_numpy(c)) for c in cartesian]
    if component is None:
        return {"values": np.linalg.norm(np.stack(cartesian), axis=0)}, xyz
    return {f"c{k + 1}": c for k, c in enumerate(cartesian)}, xyz


def _shown(arrays, component):
    """The shown values from the arrays of :func:`_grid_values`."""
    return arrays["values"] if "values" in arrays else arrays[f"c{component + 1}"]


def _render_3d(grid, values, center, azimuth, toroidal, cmap, clim, options, surface=None, surface_values=None):
    """Image (RGB array) of ``grid`` (with merged seams) colored by ``values``, cut at ``azimuth``, without title and
    color bar; ``options`` are ``view``, ``window_size``, ``context`` and ``opacity`` of :func:`write_3d_frames`. The
    faces of the cut are opaque, the rest of the boundary of the domain has the ``opacity``; ``surface`` (a surface
    inside the domain, with merged seams) colored by ``surface_values`` is cut the same way and shown opaque."""
    scalars = "values"
    grid.point_data[scalars] = values
    grid.set_active_scalars(scalars)
    shown = cut_wedge(grid, center, azimuth)
    plotter = new_plotter(False, options["window_size"])
    opacity = options["opacity"]
    if opacity < 1.0:
        # translucent boundary: the faces of the cut (wedge distance zero) opaque, the rest with the opacity
        plotter.enable_depth_peeling(number_of_peels=8, occlusion_ratio=0.0)
        boundary = shown.extract_surface(algorithm="dataset_surface")
        diagonal = np.linalg.norm(np.subtract(grid.bounds[1::2], grid.bounds[::2]))
        # the faces of the cut: all their points on the faces of the wedge (the clip carries the wedge distance)
        on_plane = np.abs(boundary.point_data["wedge"]) < 1e-6 * diagonal
        on_cut = np.zeros(boundary.n_cells, dtype=bool)
        if np.any(on_plane):
            faces = boundary.extract_points(on_plane, adjacent_cells=False)
            on_cut[np.asarray(faces.cell_data["vtkOriginalCellIds"])] = True
        if np.any(on_cut):
            add_field_mesh(plotter, boundary.extract_cells(on_cut), scalars, cmap, clim)
        if np.any(~on_cut):
            add_field_mesh(
                plotter, boundary.extract_cells(~on_cut), scalars, cmap, clim, opacity=opacity, culling="back"
            )
    else:
        add_field_mesh(plotter, shown, scalars, cmap, clim)
    if surface is not None:
        surface.point_data[scalars] = surface_values
        surface.set_active_scalars(scalars)
        add_field_mesh(plotter, cut_wedge(surface, center, azimuth), scalars, cmap, clim)
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
    points_per_cell=None,
    view="cut",
    opacity=0.35,
    surface="peak",
    window_size=(1200, 1200),
    margin=10,
    show_scalar_bar=True,
    context=1.2,
    workers=None,
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
    points_per_cell : int | None
        Evaluation points per cell of the simulation grid in each direction; by default as many as possible (at
        most 4) with at most :data:`MAX_POINTS_3D` points in the 3D grid (of the full torus).
    view : str
        ``"cut"``: the camera frames the cut (for a torus its face at ``azimuth``, a poloidal cross-section) and some
        of the domain around it, see ``context``; ``"domain"``: the whole domain.
    opacity : float
        Opacity of the boundary of the domain (without the faces of the cut, which are opaque), so that ``surface``
        is visible through it; 1 for an opaque boundary.
    surface : str | float | None
        A surface eta1 = const inside the domain, colored by the field and opaque, cut like the domain: ``"peak"``
        through the largest absolute value over all frames (e.g. the flux surface of the mode in a torus, showing its
        toroidal structure), a value of eta1 in [0, 1], or None for none. The nearest eta1 of the evaluation grid is
        used, at least one point away from the boundary of the domain (and from the axis of a polar domain).
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
    workers : int | None
        Number of processes that render the frames (default: :func:`default_workers`).

    Returns
    -------
    list[Path]
        The frames, in order.
    """
    if view not in ("cut", "domain"):
        raise ValueError(f'view must be "cut" or "domain", got {view!r}.')
    return _write_3d_views(
        out,
        name,
        {view: folder},
        component=component,
        basis=basis,
        steps=steps,
        max_frames=max_frames,
        clim=clim,
        cmap=cmap,
        label=label,
        azimuth=azimuth,
        points_per_cell=points_per_cell,
        opacity=opacity,
        surface=surface,
        window_size=window_size,
        margin=margin,
        show_scalar_bar=show_scalar_bar,
        context=context,
        workers=workers,
    )[view]


def _render_view(ctx, values, opaque=False):
    """Render ``values`` (on the points of the grid of the domain) in the view of ``ctx``: the image, the colormap
    and the color limits."""
    frame_cmap, limits = _limits(ctx["clim"], values, ctx["vmin"], ctx["vmax"], ctx["cmap"])
    options = {**ctx["options"], "opacity": 1.0} if opaque else ctx["options"]
    surface = None if opaque else ctx["surface_grid"]
    image = _render_3d(
        ctx["merged"],
        values[ctx["index"]],
        ctx["center"],
        ctx["azimuth"],
        ctx["toroidal"],
        frame_cmap,
        limits,
        options,
        surface,
        None if surface is None else values[ctx["surface_index"]],
    )
    return image, frame_cmap, limits


def _frame_3d_job(ctx, k, values, title):
    """Render and save frame ``k`` of a 3D video (see :func:`write_3d_frames`)."""
    image, frame_cmap, limits = _render_view(ctx, values)
    top, bottom, left, right = ctx["crop"]
    fig, width, height = _compose(
        image[top:bottom, left:right], title, frame_cmap, limits, ctx["label"], ctx["show_scalar_bar"]
    )
    path = ctx["folder"] / _frame_name(k, ctx["count"])
    _save_figure(fig, path, width, height)
    return path


def _write_3d_views(
    out: Output,
    name,
    folders: dict,
    *,
    component=None,
    basis=None,
    steps=None,
    max_frames=500,
    clim=None,
    cmap=None,
    label=None,
    azimuth=None,
    points_per_cell=None,
    opacity=0.35,
    surface="peak",
    window_size=(1200, 1200),
    margin=10,
    show_scalar_bar=True,
    context=1.2,
    workers=None,
) -> dict:
    """The frames of the 3D views ``folders`` (``{view: folder}``, views ``"cut"`` and/or ``"domain"``) of
    :func:`write_3d_frames`, from one evaluation of the field per frame (shared by the views); returns
    ``{view: frames}``."""
    if any(v not in ("cut", "domain") for v in folders):
        raise ValueError(f'The views must be "cut" or "domain", got {list(folders)}.')
    _check_clim(clim)
    if not (surface is None or surface == "peak" or (isinstance(surface, (int, float)) and 0 <= surface <= 1)):
        raise ValueError(f'surface must be None, "peak" or a value of eta1 in [0, 1], got {surface!r}.')
    if not 0 < opacity <= 1:
        raise ValueError(f"opacity must be in (0, 1], got {opacity!r}.")
    if MPI.COMM_WORLD.Get_rank() != 0:
        return {v: [] for v in folders}
    steps, times = _steps_and_times(out, steps, max_frames)
    label = label or field_label(out, name, component, basis)
    domain = out.domain
    toroidal = any(cls.__name__ in TORUS_DOMAINS for cls in type(domain).__mro__)
    copies = int(domain.params.get("tor_period", 1)) if toroidal else 1
    cells = out.grid.num_elements
    if points_per_cell is None:
        points_per_cell = int(np.clip(np.floor(np.cbrt(MAX_POINTS_3D / (copies * np.prod(cells))) + 1e-9), 1, 4))
    etas = {e: np.linspace(0.0, 1.0, points_per_cell * n + 1) for e, n in zip(("eta1", "eta2", "eta3"), cells)}

    first_xyz = []

    def frame_grid(n):
        """The structured grid of the domain (the full torus) with the arrays of :func:`_grid_values`."""
        arrays, xyz = _grid_values(out, name, n, etas, component, basis)
        if not first_xyz:
            first_xyz.extend(xyz)
        grid = pv.StructuredGrid(*xyz)
        for key, value in arrays.items():
            grid.point_data[key] = np.asarray(value, dtype=float).reshape(-1, order="F")
        grid.set_active_scalars(next(iter(arrays)))
        cartesian = ("c1", "c2") if "c1" in arrays else None
        grid = repeat_around_axis(grid, copies, cartesian)
        return grid, np.asarray(_shown(grid.point_data, component))

    # first pass: the range of the values and the point of the largest absolute value over all frames (the values
    # are kept for the rendering of all views if they are not too large, see FIRST_PASS_CACHE)
    vmin, vmax, peak, peak_point = np.inf, -np.inf, -1.0, None
    first_pass = {}
    for n in steps:
        grid, values = frame_grid(n)
        if len(steps) * values.nbytes <= FIRST_PASS_CACHE:
            first_pass[n] = values
        low, high = _range(values)
        vmin, vmax = min(vmin, low), max(vmax, high)
        if not np.all(np.isnan(values)):
            i = int(np.nanargmax(np.abs(values)))
            if abs(values[i]) > peak:
                peak, peak_point = abs(values[i]), grid.points[i]

    def frame_values(n):
        return first_pass[n] if n in first_pass else frame_grid(n)[1]

    center, diagonal = cut_center(grid, toroidal)
    if azimuth is None:
        azimuth = 0.0 if peak_point is None else peak_azimuth(peak_point[None], [1.0], center, diagonal)
    # not through a plane of grid points, which would leave dark specks of degenerate cells
    azimuth = off_grid_azimuth(grid.points, center, azimuth)
    # coincident points merged once; the values of each frame are taken from the nearest point of the grid
    merged = merge_seams(grid, diagonal)
    tree = cKDTree(grid.points)
    _, index = tree.query(merged.points)

    # the surface eta1 = const inside the domain (e.g. a flux surface of a torus), with the copies of a torus sector
    n1 = len(etas["eta1"])
    i1 = None
    if surface == "peak" and peak_point is not None:
        # eta1 of the peak: the nearest point of the sector to the peak rotated back into the sector (copies)
        sector_tree = cKDTree(np.stack([c.reshape(-1, order="F") for c in first_xyz], axis=1))
        candidates = []
        for k in range(copies):
            angle = -2 * np.pi * k / copies
            x, y = (
                peak_point[0] * np.cos(angle) - peak_point[1] * np.sin(angle),
                peak_point[0] * np.sin(angle) + peak_point[1] * np.cos(angle),
            )
            candidates.append(sector_tree.query([x, y, peak_point[2]]))
        i1 = int(min(candidates)[1]) % n1
    elif surface is not None and surface != "peak":
        i1 = int(np.argmin(np.abs(etas["eta1"] - float(surface))))
    if i1 is not None and n1 > 2:
        # not on the boundary of the domain (or on the axis of a polar domain)
        i1 = min(max(i1, 1), n1 - 2)
    surface_grid = surface_index = None
    if i1 is not None and n1 > 2:
        sector = pv.StructuredGrid(*(c[i1 : i1 + 1] for c in first_xyz))
        surface_grid = merge_seams(repeat_around_axis(sector, copies), diagonal)
        # its values: those of the nearest points of the grid of the domain (the same points)
        _, surface_index = tree.query(surface_grid.points)

    workers = default_workers() if workers is None else workers
    titles = [f"{label}, {_time_title(out, t)}" for t in times]
    # values of all frames in memory: all frames at once to the rendering processes, else in batches
    eager = len(first_pass) == len(steps)
    results = {}
    for view, folder in folders.items():
        ctx = dict(
            merged=merged,
            index=index,
            center=center,
            azimuth=azimuth,
            toroidal=toroidal,
            options=dict(view=view, window_size=window_size, context=context, opacity=opacity),
            surface_grid=surface_grid,
            surface_index=surface_index,
            clim=clim,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            label=label,
            show_scalar_bar=show_scalar_bar,
            count=len(steps),
        )
        # the crop box, the same for all frames (the rendered domain does not change): that of the first frame
        # (from an opaque render: the translucent boundary may be nearly white)
        ctx["crop"] = crop_box(_render_view(ctx, frame_values(steps[0]), opaque=True)[0], margin)
        ctx["folder"] = _prepare_folder(folder)
        results[view] = _run_jobs(
            _frame_3d_job,
            ctx,
            lambda k: (k, frame_values(steps[k]), titles[k]),
            len(steps),
            workers,
            eager=eager,
        )
        _write_index(ctx["folder"], steps, times)
    return results


def write_videos(
    out: Output,
    name,
    folder,
    *,
    component=None,
    basis=None,
    label=None,
    prefix=None,
    planes=None,
    views=("cut", "domain"),
    max_frames=500,
    steps=None,
    clim=None,
    cmap=None,
    slice_options=None,
    options_3d=None,
    workers=None,
) -> dict[str, list[Path]]:
    """Write the frames of the standard videos of the FEEC field ``name``: one per physical plane
    (:func:`write_slice_frames`) and one per 3D view (:func:`write_3d_frames`), into the subfolders
    ``<prefix>_<plane>``, ``<prefix>_3d_cut`` (view ``"cut"``) and ``<prefix>_3d`` (view ``"domain"``) of ``folder``.

    Parameters
    ----------
    out : Output
        Output of the run.
    name : str
        Raw FEEC field, ``"species/variable"`` (e.g. ``"mhd/velocity"``).
    folder : str | Path
        Folder of the videos.
    component, basis, label, max_frames, steps, clim, cmap :
        As in :func:`write_slice_frames` and :func:`write_3d_frames`.
    prefix : str | None
        Start of the names of the subfolders (default: the name of the variable).
    planes : list[str] | None
        Physical planes (see :func:`struphy.post_processing.slices.physical_planes`); by default the poloidal plane
        ``rz`` of tori and cylinders, else all planes of the domain.
    views : tuple[str, ...]
        3D views, ``"cut"`` and/or ``"domain"``.
    slice_options : dict | None
        Further keyword arguments of :func:`write_slice_frames` (e.g. ``points_per_cell``, ``dpi``).
    options_3d : dict | None
        Further keyword arguments of :func:`write_3d_frames` (e.g. ``window_size``, ``azimuth``). The 3D views share
        one evaluation of the field per frame.
    workers : int | None
        Number of processes that render the frames (default: :func:`default_workers`).

    Returns
    -------
    dict[str, list[Path]]
        The frames of each video, by the name of its subfolder (empty lists on the ranks other than 0).
    """
    prefix = prefix or name.split("/")[-1]
    if planes is None:
        try:
            available = physical_planes(out.domain, all_planes=True)
            planes = ["rz"] if "rz" in available else list(physical_planes(out.domain))
        except ValueError as error:
            # e.g. a domain covering only part of the poloidal angle: the 3D views only
            warnings.warn(f"No videos in planes: {error}", stacklevel=2)
            planes = []
    common = dict(
        component=component,
        basis=basis,
        label=label,
        max_frames=max_frames,
        steps=steps,
        clim=clim,
        cmap=cmap,
        workers=workers,
    )
    videos = {}
    for plane in planes:
        sub = f"{prefix}_{plane}"
        videos[sub] = write_slice_frames(out, name, Path(folder) / sub, plane=plane, **common, **(slice_options or {}))
    # both 3D views from one evaluation of the field per frame
    subs = {view: f"{prefix}_3d_cut" if view == "cut" else f"{prefix}_3d" for view in views}
    if subs:
        frames = _write_3d_views(
            out, name, {view: Path(folder) / sub for view, sub in subs.items()}, **common, **(options_3d or {})
        )
        for view, sub in subs.items():
            videos[sub] = frames[view]
    return videos


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
