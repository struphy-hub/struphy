"""Post-processing shared by the particle profiling cases (Vlasov, GuidingCenter).

The cases test whether an equilibrium distribution stays in equilibrium for long times. From the binned
distributions (``BinningPlot``) of a run, :func:`save_equilibrium_diagnostics` writes to ``<path_out>/results``:

* ``binning_<slice>.png``: the binned distribution at the first and last time (and the exact profile if given);
  2d slices as maps of the first and last time and of their difference,
* ``deviation.png`` and ``deviation_<slice>.npy``: the relative L2 deviation of the binned distribution from
  the first time (and from the exact profile) over time,
* ``scalars.png`` and ``<scalar>.npy``: the relative drift of the given scalars (e.g. the energy) over time,
* ``time.npy``.

The profiling packaging picks these files up from ``results`` and uploads them with the run.
"""

import os

import numpy as np


def _rel_l2(f, ref):
    """Relative L2 deviation of each time slice of f (time first) from ref."""
    axes = tuple(range(1, f.ndim))
    return np.sqrt(np.sum((f - ref) ** 2, axis=axes)) / np.sqrt(np.sum(ref**2))


def save_equilibrium_diagnostics(out, species, results_dir=None, exact=None, scalars=(), comm=None):
    """Plot and save the binned distributions and scalars of a run, see the module docstring.

    Parameters
    ----------
    out : struphy.post_processing.output.Output
        Output of the run (``sim.run()`` or ``sim.output``).

    species : str
        Name of the kinetic species, e.g. ``"kinetic_ions"``.

    results_dir : str, optional
        Folder for the results (default ``<out.path_out>/results``).

    exact : dict[str, callable], optional
        Exact (unnormalized) equilibrium profile per slice name (e.g. ``"e1_density"``), called with the
        meshgrid of the bin centers (``indexing="ij"``). It is scaled to the integral of the first time.

    scalars : tuple[str]
        Names of scalars whose relative drift is saved.

    comm : MPI communicator, optional
        Post-processing needs all ranks, plotting is done on rank 0.
    """
    from matplotlib import pyplot as plt

    rank = 0 if comm is None else comm.Get_rank()

    # explicit post-processing on all ranks (lazy evaluation does not work under MPI)
    out.pproc()
    if rank != 0:
        return

    if results_dir is None:
        results_dir = os.path.join(out.path_out, "results")
    os.makedirs(results_dir, exist_ok=True)
    exact = {} if exact is None else exact

    time = None
    fig_dev, ax_dev = plt.subplots(figsize=(7, 4.5))
    for key in out.distribution_catalog:
        spec, slice_name, name = key.split("/")
        if spec != species or name != "f":
            continue

        f_arr = out.distribution_catalog[key]
        dims = [dim for dim in f_arr.dims if dim != "t"]
        time = f_arr.coords["t"].values
        f = f_arr.values
        centers = [f_arr.coords[dim].values for dim in dims]

        dev_initial = _rel_l2(f, f[0])
        np.save(os.path.join(results_dir, f"deviation_{slice_name}.npy"), dev_initial)
        ax_dev.semilogy(time[1:], dev_initial[1:], label=f"{slice_name} (from t=0)")

        f_exact = None
        if slice_name in exact:
            f_exact = exact[slice_name](*np.meshgrid(*centers, indexing="ij"))
            f_exact = f_exact * np.sum(f[0]) / np.sum(f_exact)
            dev_exact = _rel_l2(f, f_exact)
            np.save(os.path.join(results_dir, f"deviation_exact_{slice_name}.npy"), dev_exact)
            ax_dev.semilogy(time, dev_exact, "--", label=f"{slice_name} (from exact)")

        if len(dims) == 1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(centers[0], f[0], label=f"t={time[0]:.3g}")
            ax.plot(centers[0], f[-1], label=f"t={time[-1]:.3g}")
            if f_exact is not None:
                ax.plot(centers[0], f_exact, "k--", label="exact")
            ax.set_xlabel(dims[0])
            ax.set_ylabel("f")
            ax.legend()
        elif len(dims) == 2:
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            vmin, vmax = np.min(f[[0, -1]]), np.max(f[[0, -1]])
            for ax, values, title in zip(
                axs,
                (f[0], f[-1], f[-1] - f[0]),
                (f"t={time[0]:.3g}", f"t={time[-1]:.3g}", "difference"),
            ):
                kwargs = {} if title == "difference" else {"vmin": vmin, "vmax": vmax}
                im = ax.pcolormesh(centers[0], centers[1], values.T, shading="nearest", **kwargs)
                fig.colorbar(im, ax=ax)
                ax.set_xlabel(dims[0])
                ax.set_ylabel(dims[1])
                ax.set_title(title)
        else:
            continue
        fig.suptitle(f"{species}: {slice_name}")
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, f"binning_{slice_name}.png"))
        plt.close(fig)

    ax_dev.set_xlabel("t")
    ax_dev.set_ylabel("relative L2 deviation")
    ax_dev.legend(fontsize=7)
    fig_dev.tight_layout()
    fig_dev.savefig(os.path.join(results_dir, "deviation.png"))
    plt.close(fig_dev)

    if scalars:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax_abs = None
        for name in scalars:
            values = out.scalars[name].values
            t = out.scalars[name].coords["t"].values
            np.save(os.path.join(results_dir, f"{name}.npy"), values)
            if values[0] != 0.0:
                ax.plot(t, (values - values[0]) / np.abs(values[0]), label=name)
            else:
                # e.g. the number of lost markers: absolute values on a second axis
                if ax_abs is None:
                    ax_abs = ax.twinx()
                ax_abs.plot(t, values, "--", label=f"{name} (absolute, right axis)")
            if time is None:
                time = t
        ax.set_xlabel("t")
        ax.set_ylabel("relative drift")
        handles, labels = ax.get_legend_handles_labels()
        if ax_abs is not None:
            handles_abs, labels_abs = ax_abs.get_legend_handles_labels()
            handles, labels = handles + handles_abs, labels + labels_abs
        ax.legend(handles, labels)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "scalars.png"))
        plt.close(fig)

    if time is not None:
        np.save(os.path.join(results_dir, "time.npy"), time)
