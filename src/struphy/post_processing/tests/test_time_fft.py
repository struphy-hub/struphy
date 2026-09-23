import os
import pickle
import shutil

import numpy as np
import pytest
from feectools.ddm.mpi import mpi as MPI

from struphy.post_processing.post_processing_tools import FieldData
from struphy.post_processing.time_fft import clean_field_pipeline, filter_val_dict, fwhm_window

# spatial profile and time grid shared by the synthetic tests
X = np.linspace(0.0, 1.0, 16)
PROF = np.exp(-(((X - 0.5) / 0.15) ** 2))
N_T, DT = 400, 0.5
T = np.arange(N_T) * DT
DOMEGA = 2 * np.pi / (N_T * DT)  # frequency bin width


def _signal(omega_main, omega_other=12 * DOMEGA, amp_other=0.4, offset=0.7):
    """Dominant mode + weaker second mode + constant offset; returns (signal, pure dominant mode)."""
    main = np.sin(omega_main * T)[:, None] * PROF
    other = amp_other * np.cos(omega_other * T)[:, None] * PROF
    return main + other + offset, main


def _rel_err(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def test_fwhm_window():
    power = np.array([0.0, 1.0, 3.0, 8.0, 10.0, 6.0, 4.0, 1.0, 0.0])
    # half max = 5: bins 3 (8) and 5 (6) are in, bins 2 (3) and 6 (4) are out
    assert fwhm_window(power, 4) == (3, 5)
    assert fwhm_window(power, 4, pad_bins=1) == (2, 6)
    assert fwhm_window(power, 4, pad_bins=2) == (1, 7)
    # padding is clamped to [idx_min, n - 1]
    assert fwhm_window(power, 4, idx_min=1, pad_bins=10) == (1, 8)

    # the FWHM search itself never goes below idx_min (keeps the DC bin out)
    power = np.array([9.0, 10.0, 2.0])
    assert fwhm_window(power, 1) == (0, 1)
    assert fwhm_window(power, 1, idx_min=1) == (1, 1)


def test_on_bin_mode_is_recovered_exactly():
    """A dominant mode exactly on a frequency bin is recovered to machine precision."""
    omega = 5 * DOMEGA
    sig, main = _signal(omega)

    cleaned, sp = clean_field_pipeline(sig, T)

    assert sp["idx_dom"] == 5
    assert np.isclose(sp["omega_dom"], omega)
    assert (sp["idx_lo"], sp["idx_hi"]) == (5, 5)
    assert _rel_err(cleaned, main) < 1e-10


@pytest.mark.parametrize("omega_main", [5.3 * DOMEGA, 5.5 * DOMEGA])
def test_pad_bins(omega_main):
    """Off-bin mode: padding widens the band symmetrically and reduces leakage error, but still excludes the second mode."""
    sig, main = _signal(omega_main)

    errs = []
    lo0, hi0 = None, None
    for pad in range(6):
        cleaned, sp = clean_field_pipeline(sig, T, pad_bins=pad)
        lo, hi = sp["idx_lo"], sp["idx_hi"]
        if pad == 0:
            lo0, hi0 = lo, hi
            # FWHM band contains the peak and lies between the two neighbouring bins of the true frequency
            assert lo <= sp["idx_dom"] <= hi
            assert 5 <= lo <= hi <= 6
        # padding never reaches the DC bin (index 0)
        assert (lo, hi) == (max(lo0 - pad, 1), hi0 + pad)
        assert np.isclose(sp["omega_lo"], sp["omega"][lo]) and np.isclose(sp["omega_hi"], sp["omega"][hi])
        assert hi < 12  # second mode is never kept
        errs += [_rel_err(cleaned, main)]

    print(f"omega = {omega_main / DOMEGA:.1f} bins, rel. errors for pad_bins 0..5: {np.round(errs, 3)}")
    assert all(e1 < e0 for e0, e1 in zip(errs[:-1], errs[1:]))
    assert errs[-1] < 0.2
    # unfiltered data is far off (offset + second mode)
    assert _rel_err(sig, main) > 1.0


def test_filter_val_dict_layout():
    sig, _ = _signal(5 * DOMEGA)
    val_dict = {t: [sig[i], 2 * sig[i], -sig[i]] for i, t in enumerate(T)}

    filtered, fft_data = filter_val_dict(val_dict, pad_bins=1)

    assert list(filtered.keys()) == list(val_dict.keys())
    assert all(len(comps) == 3 and comps[0].shape == X.shape for comps in filtered.values())
    for key in ("power", "dominant_frequency", "idx_dominant", "idx_lo", "idx_hi", "omega_lo", "omega_hi"):
        assert fft_data[key].shape[0] == 3
    assert fft_data["pad_bins"] == 1
    assert fft_data["power"].shape == (3, fft_data["omega"].size)
    # filtering is linear, so the components keep their ratios
    assert np.allclose(filtered[T[7]][1], 2 * filtered[T[7]][0])
    assert np.allclose(filtered[T[7]][2], -filtered[T[7]][0])


def test_field_data(tmp_path):
    sig, _ = _signal(5.4 * DOMEGA)
    val_dict = {t: [sig[i], 3 * sig[i]] for i, t in enumerate(T)}

    # what pproc(perform_time_fft=True, fft_pad_bins=2) writes to fields_data_filtered/
    saved, saved_fft = filter_val_dict(val_dict, pad_bins=2)
    path_filtered = str(tmp_path / "velocity_log.bin")
    with open(path_filtered, "wb") as f:
        pickle.dump(saved, f)
    with open(str(tmp_path / "velocity_log_fft.pkl"), "wb") as f:
        pickle.dump(saved_fft, f)

    v = FieldData(val_dict, path_filtered)

    # behaves like the plain dict
    assert isinstance(v, dict)
    assert sorted(v.keys()) == list(T)
    assert np.array_equal(v[T[3]][1], val_dict[T[3]][1])

    # default: the saved data, read lazily and cached
    v_f = v.filtered()
    assert v.spectrum()["pad_bins"] == 2
    assert v.filtered() is v_f
    assert v.filtered(pad_bins=2) is v_f
    for t in T:
        assert np.array_equal(v_f[t][0], saved[t][0]) and np.array_equal(v_f[t][1], saved[t][1])

    # other pad_bins: computed in memory, equal to filtering the unfiltered data directly
    for pad in (0, 4):
        ref, ref_fft = filter_val_dict(val_dict, pad_bins=pad)
        v_p = v.filtered(pad_bins=pad)
        assert v.filtered(pad_bins=pad) is v_p
        assert v.spectrum(pad_bins=pad)["pad_bins"] == pad
        assert np.array_equal(v.spectrum(pad_bins=pad)["idx_lo"], ref_fft["idx_lo"])
        assert all(np.allclose(v_p[t][c], ref[t][c]) for t in T for c in range(2))
        assert not np.allclose(v_p[T[10]][0], v_f[T[10]][0])

    # no filtered data on disk (pproc without perform_time_fft): computed with pad_bins=0
    w = FieldData(val_dict, str(tmp_path / "missing.bin"))
    assert w.spectrum()["pad_bins"] == 0
    ref, _ = filter_val_dict(val_dict, pad_bins=0)
    assert np.allclose(w.filtered()[T[10]][0], ref[T[10]][0])
    assert FieldData(val_dict).spectrum()["pad_bins"] == 0

    # filtered data from an older pproc without the spectrum file: computed in memory
    old = tmp_path / "old"
    old.mkdir()
    with open(str(old / "velocity_log.bin"), "wb") as f:
        pickle.dump(saved, f)
    u = FieldData(val_dict, str(old / "velocity_log.bin"))
    assert u.spectrum()["pad_bins"] == 0
    assert np.allclose(u.filtered()[T[10]][0], ref[T[10]][0])


def test_pproc_time_fft_linear_mhd(do_plot=False):
    """End-to-end: 1D LinearMHD slab with two shear Alfvén waves (n=1 dominant, n=3 weaker).

    pproc(perform_time_fft=True, fft_pad_bins=...) must find the n=1 frequency, the filtered
    data must be accessible via .filtered() without reloading and must no longer contain the n=3 wave.
    """
    from struphy import DerhamOptions, EnvironmentOptions, Simulation, Time, domains, equils, grids, perturbations
    from struphy.models import LinearMHD

    Lz = 20.0
    omega_1 = 2 * np.pi / Lz  # shear Alfvén frequency k*v_A of n=1 (B0 along z, n0 = 1 -> v_A = 1)

    model = LinearMHD()
    model.mhd.velocity.add_perturbation(
        perturbations.ModesSin(ns=(1, 3), amps=(1e-3, 3e-4), Lz=Lz, comp=0, given_in_basis="physical")
    )
    test_folder = os.path.join(os.getcwd(), "time_fft_test")
    env = EnvironmentOptions(out_folders=test_folder, sim_folder="two_alfven_waves")
    sim = Simulation(
        model=model,
        env=env,
        time_opts=Time(dt=0.1, Tend=100.0),
        domain=domains.Cuboid(r3=Lz),
        grid=grids.TensorProductGrid(num_elements=(1, 1, 32)),
        derham_opts=DerhamOptions(degree=(1, 1, 3)),
        equil=equils.HomogenSlab(B0x=0.0, B0y=0.0, B0z=1.0, beta=1.0, n0=1.0),
    )
    sim.run()

    if MPI.COMM_WORLD.Get_rank() == 0:
        pad = 2
        sim.pproc(perform_time_fft=True, fft_pad_bins=pad, physical=True)
        sim.load_plotting_data()

        v = sim.spline_values.mhd.velocity_log.data
        v_f = v.filtered()
        t_grid = sim.t_grid
        z = sim.grids_phy[2][0, 0, :]

        # filtered data sits next to the unfiltered one, same layout
        assert isinstance(v, FieldData)
        assert list(v_f.keys()) == list(v.keys())
        assert v_f[t_grid[0]][0].shape == v[t_grid[0]][0].shape

        # dominant frequency is the n=1 shear Alfvén frequency (within one bin)
        sp = v.spectrum()
        domega = sp["omega"][1]
        print(f"omega_1 = {omega_1:.4f}, found {sp['dominant_frequency'][0]:.4f}, bin width {domega:.4f}")
        print(f"kept band [{sp['omega_lo'][0]:.4f}, {sp['omega_hi'][0]:.4f}]")
        assert sp["pad_bins"] == pad
        assert abs(sp["dominant_frequency"][0] - omega_1) < domega
        assert sp["omega_hi"][0] < 3 * omega_1

        # spatial n=3 content: present in unfiltered, (almost) gone in filtered; n=1 kept
        def mode_amp(data, n):
            u_x = np.array([data[t][0][0, 0, :] for t in t_grid])
            return np.abs(np.sum(u_x * np.sin(2 * np.pi * n * z / Lz), axis=1)).max()

        ratio_raw = mode_amp(v, 3) / mode_amp(v, 1)
        ratio_filt = mode_amp(v_f, 3) / mode_amp(v_f, 1)
        print(f"n=3/n=1 amplitude ratio: unfiltered {ratio_raw:.3e}, filtered {ratio_filt:.3e}")
        assert ratio_raw > 0.2
        assert ratio_filt < 0.05 * ratio_raw

        # a different padding, without re-running pproc: wider band, recomputed from v
        sp5 = v.spectrum(pad_bins=5)
        assert sp5["idx_lo"][0] == max(sp["idx_lo"][0] - 3, 1) and sp5["idx_hi"][0] == sp["idx_hi"][0] + 3
        assert v.filtered(pad_bins=5) is not v_f

        # energies from quadrature of the fields reproduce the scalars saved during the run
        import h5py

        from struphy.post_processing.field_energies import linear_mhd_energies

        E = linear_mhd_energies(sim)
        E_f = linear_mhd_energies(sim, filtered=True)
        with h5py.File(os.path.join(sim.env.path_out, "data", "data_proc0.hdf5"), "r") as f:
            t_h5 = f["time/value"][:]
            # errors relative to the total energy (en_thermal is ~0 here: shear Alfvén waves are incompressible)
            scale = np.abs(f["scalar/en_tot"][:]).max()
            for key in ("en_U", "en_B", "en_thermal", "en_tot"):
                ref = np.interp(E["t"], t_h5, f["scalar/" + key][:])
                err = np.abs(E[key] - ref).max() / scale
                print(f"{key}: quadrature vs. hdf5 rel. error {err:.2e}")
                assert err < 0.02
        assert list(E_f["t"]) == list(E["t"])
        # the n=3 wave carries energy, so the filtered total energy is smaller
        assert E_f["en_tot"].mean() < E["en_tot"].mean()

        # push-forwarded (physical) components are filtered too
        v_phy = sim.spline_values.mhd.velocity_phy.data
        assert v_phy.spectrum()["pad_bins"] == pad
        assert abs(v_phy.spectrum()["dominant_frequency"][0] - omega_1) < domega
        assert mode_amp(v_phy.filtered(), 3) / mode_amp(v_phy.filtered(), 1) < 0.05 * ratio_raw
        # spectra are not loaded as extra fields
        assert not hasattr(sim.spline_values.mhd, "velocity_log_fft")

        if do_plot:
            import matplotlib.pyplot as plt

            iz = len(z) // 4
            plt.plot(t_grid, [v[t][0][0, 0, iz] for t in t_grid], label="unfiltered")
            plt.plot(t_grid, [v_f[t][0][0, 0, iz] for t in t_grid], label=f"filtered, pad_bins={pad}")
            plt.legend()
            plt.show()

        shutil.rmtree(test_folder)


if __name__ == "__main__":
    test_pproc_time_fft_linear_mhd(do_plot=True)
