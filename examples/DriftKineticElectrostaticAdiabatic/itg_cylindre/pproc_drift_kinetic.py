import importlib.util
import os
import sys

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import glob

import h5py
import pyvista as pv
import cunumpy as xp
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from struphy import PlottingData, PostProcessor


# ============================================================
# User options
# ============================================================
FIT_T0 = 0.0
FIT_T1 = None              # None -> last available time
FIT_QUANTITY = "phi_integral"  # or "en_phi" if this scalar exists in data_proc0.hdf5

SHOW_EQUIL_PROFILE = False
SHOW_DENSITY_SLIDER = True
SHOW_FIELD_SLIDER = True

DENSITY_PLOTS = [
    {
        "bin": "e1_e2_density",
        "quantity": "f_binned",
        "physical": True,
        "axes": "XY",
        "vmin": None,
        "vmax": None,
        "title": "delta_f (R,Z)",
    },
    {
        "bin": "e1_e2_density",
        "quantity": "delta_f_binned",
        "physical": True,
        "axes": "XY",
        "vmin": None,
        "vmax": None,
        "title": "delta_f (X,Y)",
    },
]

FIELD_PLOTS = [
    {
        "species": "em_fields",
        "field": "phi_phy",
        "component": 0,
        "axes": "XYZ",
        "fixed_index": 0,
        "vmin": None,
        "vmax": None,
        "title": "Electric potential phi",
    },
    {
        "species": "diagnostics",
        "field": "rho_phy",
        "component": 0,
        "axes": "XYZ",
        "fixed_index": 0,
        "vmin": None,
        "vmax": None,
        "title": "Right-hand side of Poisson equation",
    },
]



# ============================================================
# Small utilities
# ============================================================
def load_params(sim_path):
    spec = importlib.util.spec_from_file_location("params", os.path.join(sim_path, "parameters.py"))
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)
    return params


def ensure_post_processing(params, sim_path):
    if not os.path.isdir(os.path.join(sim_path, "post_processing")):
        pp = PostProcessor(sim=params.sim)
        pp.process(physical=True)


def load_scalar(data_path, quantity):
    with h5py.File(os.path.join(data_path, "data_proc0.hdf5"), "r") as f:
        time = xp.asarray(f["time"]["value"][()])
        if quantity in f["scalar"]:
            y = xp.asarray(f["scalar"][quantity][()])
        elif quantity == "en_phi" and "phi_integral" in f["scalar"]:
            y = xp.asarray(f["scalar"]["phi_integral"][()])
        else:
            available = list(f["scalar"].keys())
            raise KeyError(f"Scalar {quantity!r} not found. Available scalars: {available}")
    return time, y


def fit_window(time, y, t0, t1):
    """Return safe indices for a log-fit of sqrt(y)."""
    if t1 is None:
        t1 = float(time[-1])

    lo, hi = sorted((float(t0), float(t1)))
    mask = (time >= lo) & (time <= hi) & xp.isfinite(y) & (y > 0.0)

    # If the user window is unusable, use all positive finite samples.
    if xp.count_nonzero(mask) < 2:
        mask = xp.isfinite(y) & (y > 0.0)

    # If there are still too few points, disable fit cleanly.
    if xp.count_nonzero(mask) < 2:
        return None

    idx = xp.nonzero(mask)[0]
    return int(idx[0]), int(idx[-1]) + 1


def plot_energy_fit(time, en_phi, t0=FIT_T0, t1=FIT_T1):
    window = fit_window(time, en_phi, t0, t1)

    fig, ax = plt.subplots()
    ax.plot(time, en_phi, label=FIT_QUANTITY)
    ax.set_xlabel("time")
    ax.set_ylabel(FIT_QUANTITY)
    ax.set_title(f"Evolution of {FIT_QUANTITY}")

    if window is not None:
        i0, i1 = window
        fit_time = time[i0:i1]
        fit_signal = xp.log(xp.sqrt(en_phi[i0:i1]))
        gamma, b = xp.polyfit(fit_time, fit_signal, 1)
        en_fit = xp.exp(2.0 * (gamma * fit_time + b))
        ax.plot(fit_time, en_fit, "--", label=f"fit: gamma={gamma:.4e}")
        ax.axvspan(fit_time[0], fit_time[-1], alpha=0.12)
        print(f"{FIT_QUANTITY} fit window: [{fit_time[0]:.6e}, {fit_time[-1]:.6e}]")
        print(f"growth rate from log(sqrt({FIT_QUANTITY})): gamma = {gamma:.8e}")
    else:
        print(f"No valid positive data found for exponential fit of {FIT_QUANTITY}.")

    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_equilibrium_profile(sim_path):
    equil_data = pv.read(os.path.join(sim_path, "geometry.vts"))
    dims = equil_data.dimensions
    grid = xp.reshape(equil_data.points, dims + (3,))
    r = xp.sqrt(grid[:, :, :, 0] ** 2 + grid[:, :, :, 1] ** 2)
    p0 = xp.reshape(equil_data.point_data["p0"], dims)

    fig, ax = plt.subplots()
    ax.set_title("Radial equilibrium profiles")
    ax.set_xlabel("R")
    ax.plot(r[0, 0, :], p0[0, 0, :], label="p0")

    if "n0" in equil_data.point_data:
        n0 = xp.reshape(equil_data.point_data["n0"], dims)
        ax.plot(r[0, 0, :], n0[0, 0, :], label="n0")
        ax.plot(r[0, 0, :], p0[0, 0, :] / n0[0, 0, :], label="T0")

    ax.legend()
    fig.tight_layout()
    plt.show()


def match_field_to_grid(field, xgrid):
    """Return field with orientation compatible with xgrid when possible."""
    if field.shape == xgrid.shape:
        return field
    if field.T.shape == xgrid.shape:
        return field.T
    raise ValueError(f"Cannot match field shape {field.shape} with grid shape {xgrid.shape}.")


def get_binned_data(pdata, bin_name, quantity):
    return xp.asarray(getattr(getattr(pdata.f.kinetic_ions, bin_name), quantity))


def get_binned_grids(
    params,
    pdata,
    bin_name,
    in_physical=True,
    plot_axes="RZ",
    fixed_eta=(0.5, 0.0, 0.0),
):
    bin_data = getattr(pdata.f.kinetic_ions, bin_name)

    bin_axes = [int(part[1]) for part in bin_name.split("_") if part.startswith("e")]
    if len(bin_axes) != 2:
        raise ValueError(f"Cannot infer two binned axes from bin_name={bin_name!r}")

    g0 = getattr(bin_data, f"grid_e{bin_axes[0]}")
    g1 = getattr(bin_data, f"grid_e{bin_axes[1]}")

    if not in_physical:
        xgrid, ygrid = xp.meshgrid(g0, g1, indexing="ij")
        return xgrid, ygrid, f"eta{bin_axes[0]}", f"eta{bin_axes[1]}"

    etas = []
    for ax in (1, 2, 3):
        if ax == bin_axes[0]:
            etas.append(g0)
        elif ax == bin_axes[1]:
            etas.append(g1)
        else:
            etas.append(fixed_eta[ax - 1])

    x, y, z = params.domain(*etas, squeeze_out=True)

    if plot_axes == "RZ":
        return xp.sqrt(x**2 + y**2), z, "R", "Z"
    elif plot_axes == "XY":
        return x, y, "X", "Y"
    elif plot_axes == "XZ":
        return x, z, "X", "Z"
    elif plot_axes == "YZ":
        return y, z, "Y", "Z"
    else:
        raise ValueError(f"Unknown plot_axes={plot_axes!r}")


def make_slider_plot(time_grid, xgrid, ygrid, data, *, title, xlabel, ylabel, vmin=None, vmax=None):
    data0 = match_field_to_grid(xp.asarray(data[0]), xgrid)

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.20)
    pcm = ax.pcolormesh(xgrid, ygrid, data0, shading="auto", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(pcm, ax=ax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} at t = {time_grid[0]:.4e}")

    slider_ax = fig.add_axes([0.20, 0.07, 0.60, 0.03])
    slider = Slider(
        slider_ax,
        "time index",
        0,
        len(time_grid) - 1,
        valinit=0,
        valstep=1,
    )

    def update(_):
        idx = int(slider.val)
        field = match_field_to_grid(xp.asarray(data[idx]), xgrid)
        pcm.set_array(field.ravel())
        if vmin is None and vmax is None:
            pcm.set_clim(float(xp.nanmin(field)), float(xp.nanmax(field)))
            cbar.update_normal(pcm)
        ax.set_title(f"{title} at t = {time_grid[idx]:.4e}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def plot_binned_quantity_slider(params, pdata, *, bin_name, quantity, in_physical=True, axes="RZ", vmin=None, vmax=None, title="density binned"):
    data = get_binned_data(pdata, bin_name, quantity)
    xgrid, ygrid, xlabel, ylabel = get_binned_grids(params, pdata, bin_name, in_physical=in_physical, plot_axes=axes)
    make_slider_plot(
        pdata.t_grid,
        xgrid,
        ygrid,
        data,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        vmin=vmin,
        vmax=vmax,
    )


def get_field_3d(pdata, species, field, component=0):
    field_data = getattr(getattr(pdata.spline_values, species), field)
    times = list(field_data.data.keys())
    values = [field_data.data[t][component] for t in times]
    return xp.array(times), values


def get_slice_from_field(pdata, arr3d, axes="RZT", fixed_index=0):
    X = pdata.grids_phy[0]
    Y = pdata.grids_phy[1]
    Z = pdata.grids_phy[2]
    R = xp.sqrt(X**2 + Y**2)

    if axes == "RZT":
        data = arr3d[:, :, fixed_index]
        return R[:, :, fixed_index], Z[:, :, fixed_index], data, "R", "Z"

    elif axes == "XYZ":
        data = arr3d[:, :, fixed_index]
        return X[:, :, fixed_index], Y[:, :, fixed_index], data, "X", "Y"

    elif axes == "RTP":
        data = arr3d[:, fixed_index, :]
        return R[:, fixed_index, :], Z[:, fixed_index, :], data, "R", "Z"

    else:
        raise ValueError(f"Unknown field slice axes={axes!r}")


def plot_field_slider(
    pdata,
    species,
    field,
    component=0,
    axes="RZT",
    vmin=None,
    vmax=None,
    title=None,
):
    times, values = get_field_3d(pdata, species, field, component=component)

    nt = len(values)
    shape = values[0].shape

    if axes == "RZT":
        nslice = shape[2]
        slice_label = "toroidal index"
    elif axes == "XYZ":
        nslice = shape[2]
        slice_label = "poloidal/radial slice index"
    elif axes == "RTP":
        nslice = shape[1]
        slice_label = "eta2 index"
    else:
        raise ValueError(f"Unknown field slice axes={axes!r}")

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.22)

    time_idx = 0
    slice_idx = min(nslice - 1, nslice // 2)

    xg, yg, data, xlabel, ylabel = get_slice_from_field(
        pdata,
        values[time_idx],
        axes=axes,
        fixed_index=slice_idx,
    )

    pcm = ax.pcolormesh(xg, yg, data, shading="auto", vmin=vmin, vmax=vmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{species}.{field}")

    cbar = fig.colorbar(pcm, ax=ax)

    ax_time = plt.axes([0.15, 0.10, 0.70, 0.03])
    ax_slice = plt.axes([0.15, 0.05, 0.70, 0.03])

    s_time = Slider(ax_time, "time", 0, nt - 1, valinit=time_idx, valstep=1)
    s_slice = Slider(ax_slice, slice_label, 0, nslice - 1, valinit=slice_idx, valstep=1)

    def update(_):
        ti = int(s_time.val)
        si = int(s_slice.val)

        xg, yg, data, xlabel, ylabel = get_slice_from_field(
            pdata,
            values[ti],
            axes=axes,
            fixed_index=si,
        )

        pcm.set_array(data.ravel())
        pcm.set_clim(
            vmin if vmin is not None else xp.nanmin(data),
            vmax if vmax is not None else xp.nanmax(data),
        )
        ax.set_title(f"{title or field} | t = {times[ti]:.4e}, slice = {si}")
        fig.canvas.draw_idle()

    s_time.on_changed(update)
    s_slice.on_changed(update)

    plt.show()



def load_marker_data(pdata, species="kinetic_ions", max_markers=200):
    orbs = getattr(pdata.orbits, species)
    nb_markers = min(orbs.shape[1], max_markers)
    return orbs[:, :nb_markers, 0], orbs[:, :nb_markers, 1], orbs[:, :nb_markers, 2], orbs[:, :nb_markers, 6]


def plot_marker_trajectories_slider(
    pdata,
    species="kinetic_ions",
    max_markers=200,
    show_paths=None,
    title="Marker trajectories",
):
    if show_paths is None:
        show_paths = max_markers <= 200

    x, y, z, weights = load_marker_data(
        pdata=pdata,
        species=species,
        max_markers=max_markers,
    )

    nt, nmarkers = x.shape
    print(f"loaded markers: {x.shape}")
    print(f"plotted trajectories: {nmarkers}")
    print("x min/max:", xp.nanmin(x), xp.nanmax(x))
    print("y min/max:", xp.nanmin(y), xp.nanmax(y))
    print("z min/max:", xp.nanmin(z), xp.nanmax(z))

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(bottom=0.18)

    it0 = 0
    sc = ax.scatter(
        x[it0],
        y[it0],
        z[it0],
        c=weights[it0],
        s=8,
        cmap="viridis",
    )

    lines = []
    if show_paths:
        for j in range(nmarkers):
            line, = ax.plot(
                x[: it0 + 1, j],
                y[: it0 + 1, j],
                z[: it0 + 1, j],
                lw=0.8,
                alpha=0.5,
            )
            lines.append(line)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{title} | step {it0}/{nt - 1}")

    fig.colorbar(sc, ax=ax, label="marker weights")

    ax_slider = plt.axes([0.18, 0.06, 0.65, 0.03])
    slider = Slider(ax_slider, "time index", 0, nt - 1, valinit=it0, valstep=1)

    def update(_):
        it = int(slider.val)

        sc._offsets3d = (x[it], y[it], z[it])
        sc.set_array(weights[it])

        if show_paths:
            for j, line in enumerate(lines):
                line.set_data(x[: it + 1, j], y[: it + 1, j])
                line.set_3d_properties(z[: it + 1, j])

        ax.set_title(f"{title} | step {it}/{nt - 1}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

# ============================================================
# Main
# ============================================================
def main():
    if len(sys.argv) > 1 and __name__ == "__main__":
        sim_name = sys.argv[1]
    else:
        sim_name = "sim_1"

    sim_path = os.path.join(os.getcwd(), sim_name)
    params = load_params(sim_path)
    ensure_post_processing(params, sim_path)

    pdata = PlottingData(sim=params.sim)
    pdata.load()

    params.domain.show()

    data_path = os.path.join(sim_path, "data")
    
    time, en_phi = load_scalar(data_path, FIT_QUANTITY)
    plot_energy_fit(time, en_phi)

    if SHOW_EQUIL_PROFILE:
        plot_equilibrium_profile(sim_path)

    if SHOW_DENSITY_SLIDER:
        for cfg in DENSITY_PLOTS:
            plot_binned_quantity_slider(
                params,
                pdata,
                bin_name=cfg["bin"],
                quantity=cfg["quantity"],
                in_physical=cfg.get("physical", True),
                axes=cfg.get("axes", "RZ"),
                vmin=cfg.get("vmin"),
                vmax=cfg.get("vmax"),
                title=cfg.get("title"),
            )

    if SHOW_FIELD_SLIDER:
        for cfg in FIELD_PLOTS:
            plot_field_slider(
                pdata,
                species=cfg["species"],
                field=cfg["field"],
                component=cfg.get("component", 0),
                axes=cfg.get("axes", "RZT"),
                vmin=cfg.get("vmin"),
                vmax=cfg.get("vmax"),
                title=cfg.get("title"),
            )
    
    plot_marker_trajectories_slider(
        pdata=pdata,
        species="kinetic_ions",
        max_markers=1000,
        show_paths=True,
    )


if __name__ == "__main__":
    main()
