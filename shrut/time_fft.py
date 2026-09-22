import numpy as np
import matplotlib.pyplot as plt

def time_fft(field_series, t_grid):
    """
    FFT along the time axis (axis 0) of a real-valued field time series.

    field_series : ndarray, shape (n_t, ...) -- real values at each saved time.
    t_grid       : 1D array, length n_t -- the (assumed uniformly spaced) saved times.

    Returns
    -------
    omega    : 1D array of angular frequencies, only non-negative (see note below)
    fft_vals : ndarray, shape (len(omega), ...), complex
    """
    dt = t_grid[1] - t_grid[0]
    fft_vals = np.fft.rfft(field_series, axis=0)
    freqs = np.fft.rfftfreq(len(t_grid), d=dt)
    omega = 2 * np.pi * freqs
    return omega, fft_vals


def plot_spectrum(omega, fft_vals, omega_min=1e-8):
    """
    Summed power spectrum vs omega, for picking out the dominant frequency by eye.
    omega_min excludes the DC (omega=0) bin, which otherwise tends to dominate
    the plot due to any nonzero time-averaged offset, drowning out the real oscillation.
    """
    power = np.sum(np.abs(fft_vals) ** 2, axis=tuple(range(1, fft_vals.ndim)))
    mask = omega >= omega_min

    plt.figure(figsize=(7, 4))
    plt.plot(omega[mask], power[mask])
    plt.xlabel(r"$\omega$")
    plt.ylabel("summed power")
    plt.title("Time-FFT power spectrum")
    plt.tight_layout()
    plt.show()
    return power


def find_dominant_frequency(omega, power, omega_min=1e-8):
    """Index and value of the largest peak, excluding the DC bin by default."""
    mask = omega >= omega_min
    idx_local = np.argmax(power[mask])
    idx_global = np.nonzero(mask)[0][idx_local]
    return omega[idx_global], idx_global


def filter_to_dominant_mode(fft_vals, idx_dominant, bandwidth_bins=0):
    """
    Zero out every frequency bin except the dominant one (and optionally a small
    window of neighbors, via bandwidth_bins, in case the true peak straddles two bins).
    """
    filtered = np.zeros_like(fft_vals)
    lo = max(0, idx_dominant - bandwidth_bins)
    hi = min(fft_vals.shape[0] - 1, idx_dominant + bandwidth_bins)
    filtered[lo : hi + 1] = fft_vals[lo : hi + 1]
    return filtered


def inverse_time_fft(filtered_fft_vals, n_t):
    """Back to the time domain -- filtered, cleaned field time series."""
    return np.fft.irfft(filtered_fft_vals, n=n_t, axis=0)


def clean_field_pipeline(field_series, t_grid, bandwidth_bins=0):
    """Full pipeline: FFT -> plot -> find dominant peak -> filter -> inverse FFT."""
    omega, fft_vals = time_fft(field_series, t_grid)
    power = plot_spectrum(omega, fft_vals)
    omega_dom, idx_dom = find_dominant_frequency(omega, power)
    print(f"Dominant frequency: omega = {omega_dom:.4f}")

    filtered_fft = filter_to_dominant_mode(fft_vals, idx_dom, bandwidth_bins=bandwidth_bins)
    cleaned_field_series = inverse_time_fft(filtered_fft, n_t=len(t_grid))
    return cleaned_field_series, omega_dom