"""
Ripple Counter — Adaptive Envelope Thresholding Method
======================================================
Instead of computing one global MAD threshold for the entire signal,
this method uses the Hilbert-envelope of the bandpass-filtered signal
and computes a LOCAL, sliding-window threshold that adapts to the
changing ripple amplitude throughout the recording.

This catches ripples that shrink or grow mid-run (e.g., motor speed
ramps, load changes) where a single global threshold would miss the
quiet sections or over-count the loud sections.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, hilbert
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt


# ── Data Loading ──────────────────────────────────────────────────────
def load_data(filepath, sheet_name=0, current_col='I', time_col=None, fs=4000):
    print(f"Loading data from '{filepath}' (Sheet: {sheet_name})...")
    df = pd.read_excel(filepath, sheet_name=sheet_name, skiprows=[1, 2, 3, 4, 5])
    if current_col not in df.columns:
        raise ValueError(f"Column '{current_col}' not found in sheet.")
    current = pd.to_numeric(df[current_col], errors='coerce')
    valid_mask = current.notna()
    current = current[valid_mask].values.astype(float)
    if time_col and time_col in df.columns:
        time = pd.to_numeric(df[time_col], errors='coerce')[valid_mask].values.astype(float)
    else:
        time = np.arange(len(current)) / fs
    return time, current


# ── FFT Frequency Analysis ───────────────────────────────────────────
def analyze_frequency(signal, fs=4000, min_ripple_freq=20.0):
    nyq = 0.5 * fs
    hp_cutoff = min_ripple_freq / nyq
    b_hp, a_hp = butter(4, hp_cutoff, btype='high')
    signal_hp = filtfilt(b_hp, a_hp, signal)

    N = len(signal_hp)
    yf = np.abs(rfft(signal_hp))
    xf = rfftfreq(N, 1.0 / fs)

    min_freq_idx = np.searchsorted(xf, min_ripple_freq)
    max_freq_idx = np.searchsorted(xf, fs / 2 * 0.9)

    search_mag = yf[min_freq_idx:max_freq_idx]

    if len(search_mag) == 0:
        return 0, xf, yf

    dominant_idx = np.argmax(search_mag) + min_freq_idx
    dominant_freq = xf[dominant_idx]
    return dominant_freq, xf, yf


# ── Bandpass Filter ──────────────────────────────────────────────────
def preprocess_signal(signal, fs=4000, lowcut=None, highcut=None, dominant_freq=None, order=4):
    if dominant_freq and dominant_freq > 0:
        if lowcut is None:
            lowcut = max(1.0, dominant_freq * 0.3)
        if highcut is None:
            highcut = min(dominant_freq * 3.0, fs * 0.45)
    else:
        lowcut = lowcut or 10.0
        highcut = highcut or 500.0

    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)

    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


# ── Adaptive Envelope Threshold ──────────────────────────────────────
def compute_adaptive_threshold(filtered_signal, fs=4000, dominant_freq=None,
                                window_cycles=10, threshold_multiplier=1.5):
    """Compute a local, sliding-window threshold from the Hilbert envelope.

    Parameters
    ----------
    filtered_signal : array
        The bandpass-filtered motor current.
    fs : int
        Sampling frequency.
    dominant_freq : float
        The dominant ripple frequency (Hz). Used to size the window.
    window_cycles : int
        How many ripple cycles to include in each sliding window.
        Larger = smoother threshold, smaller = more responsive.
    threshold_multiplier : float
        The local RMS is multiplied by this to set the threshold.
        Lower = more sensitive (catches weak ripples), higher = stricter.

    Returns
    -------
    envelope : array
        The Hilbert envelope of the signal.
    adaptive_thresh : array
        The local, sliding threshold at every sample.
    """
    # 1. Compute the Hilbert envelope (instantaneous amplitude)
    analytic_signal = hilbert(filtered_signal)
    envelope = np.abs(analytic_signal)

    # 2. Determine window size in samples
    if dominant_freq and dominant_freq > 0:
        period_samples = int(fs / dominant_freq)
        window_size = period_samples * window_cycles
    else:
        window_size = int(fs * 0.1)  # fallback: 100ms window

    window_size = max(window_size, 10)

    # 3. Compute local RMS using a sliding window (via convolution)
    squared = envelope ** 2
    kernel = np.ones(window_size) / window_size
    local_mean_sq = np.convolve(squared, kernel, mode='same')
    local_rms = np.sqrt(local_mean_sq)

    # 4. Set threshold as a multiple of the local RMS
    adaptive_thresh = local_rms * threshold_multiplier

    # 5. Enforce a small minimum to avoid zero-threshold in dead zones
    global_noise_floor = np.percentile(envelope, 10)
    adaptive_thresh = np.maximum(adaptive_thresh, global_noise_floor * 0.5)

    return envelope, adaptive_thresh


# ── Ripple Counting with Adaptive Threshold ──────────────────────────
def count_ripples_adaptive(filtered_signal, fs=4000, dominant_freq=None,
                           window_cycles=10, threshold_multiplier=1.5):
    """Count ripples using adaptive envelope thresholding.

    Instead of one global prominence threshold, peaks and troughs are
    only counted if their amplitude exceeds the LOCAL adaptive threshold
    at that specific point in time.
    """
    # 1. Compute the adaptive threshold
    envelope, adaptive_thresh = compute_adaptive_threshold(
        filtered_signal, fs=fs, dominant_freq=dominant_freq,
        window_cycles=window_cycles, threshold_multiplier=threshold_multiplier
    )

    # 2. Enforce minimum distance between detections
    if dominant_freq and dominant_freq > 0:
        min_distance = int(fs / dominant_freq * 0.6)
    else:
        min_distance = 4
    min_distance = max(min_distance, 2)

    # 3. Find ALL candidate peaks and troughs (with a very low global prominence)
    global_floor = np.percentile(np.abs(filtered_signal), 10) * 0.1
    global_floor = max(global_floor, 0.001)

    all_peaks, peak_props = find_peaks(filtered_signal, prominence=global_floor,
                                        distance=min_distance)
    all_troughs, trough_props = find_peaks(-filtered_signal, prominence=global_floor,
                                            distance=min_distance)

    # 4. Filter: keep only peaks/troughs whose amplitude exceeds the local threshold
    valid_peaks = []
    for p in all_peaks:
        if abs(filtered_signal[p]) > adaptive_thresh[p]:
            valid_peaks.append(p)
    valid_peaks = np.array(valid_peaks, dtype=int)

    valid_troughs = []
    for t in all_troughs:
        if abs(filtered_signal[t]) > adaptive_thresh[t]:
            valid_troughs.append(t)
    valid_troughs = np.array(valid_troughs, dtype=int)

    # 5. Merge and apply smart pairing logic (same as hybrid method)
    extrema = [(idx, 'P') for idx in valid_peaks] + [(idx, 'T') for idx in valid_troughs]
    extrema.sort(key=lambda x: x[0])

    ripple_count = 0
    i = 0
    while i < len(extrema):
        if i + 1 < len(extrema) and extrema[i][1] != extrema[i + 1][1]:
            ripple_count += 1
            i += 2
        else:
            ripple_count += 1
            i += 1

    return ripple_count, valid_peaks, valid_troughs, extrema, envelope, adaptive_thresh


# ── Plotting ─────────────────────────────────────────────────────────
def plot_results(time, raw_signal, filtered_signal, fft_freqs, fft_mag,
                 peaks=None, troughs=None, ripple_count=0, dominant_freq=None,
                 envelope=None, adaptive_thresh=None,
                 title="Motor Current Ripple Analysis (Adaptive Envelope Method)"):
    """5-panel plot: Raw, FFT, Envelope+Threshold, Filtered (Full), Filtered (Zoomed)."""
    n_panels = 5 if envelope is not None else 4
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3 * n_panels))

    ax_idx = 0

    # Panel 1: Raw Signal
    axes[ax_idx].plot(time, raw_signal, color='gray', linewidth=0.5)
    axes[ax_idx].set_title("Raw Signal")
    axes[ax_idx].set_ylabel("Current")
    axes[ax_idx].grid(True, alpha=0.3)
    ax_idx += 1

    # Panel 2: FFT
    axes[ax_idx].plot(fft_freqs, fft_mag, color='purple', linewidth=0.8)
    if dominant_freq:
        axes[ax_idx].axvline(dominant_freq, color='red', linestyle='--',
                              label=f'{dominant_freq:.1f} Hz')
    axes[ax_idx].set_title("Frequency Spectrum (FFT)")
    axes[ax_idx].set_xlim(0, 1500)
    axes[ax_idx].set_ylabel("Magnitude")
    axes[ax_idx].grid(True, alpha=0.3)
    axes[ax_idx].legend()
    ax_idx += 1

    # Panel 3: Envelope + Adaptive Threshold
    if envelope is not None:
        axes[ax_idx].plot(time, envelope, color='orange', linewidth=0.5,
                          label='Hilbert Envelope', alpha=0.8)
        if adaptive_thresh is not None:
            axes[ax_idx].plot(time, adaptive_thresh, color='red', linewidth=1.2,
                              linestyle='--', label='Adaptive Threshold', alpha=0.9)
        axes[ax_idx].set_title("Hilbert Envelope vs. Adaptive Threshold")
        axes[ax_idx].set_ylabel("Amplitude")
        axes[ax_idx].legend(loc='upper right', fontsize=8)
        axes[ax_idx].grid(True, alpha=0.3)
        ax_idx += 1

    # Panel 4: Filtered Signal (Full) with peak and trough markers
    num_peaks = len(peaks) if peaks is not None else 0
    num_troughs = len(troughs) if troughs is not None else 0
    axes[ax_idx].plot(time, filtered_signal, color='blue', linewidth=0.5)
    if peaks is not None and len(peaks) > 0:
        axes[ax_idx].plot(time[peaks], filtered_signal[peaks], "v",
                          color='red', markersize=4, alpha=0.7,
                          label=f'Peaks ({num_peaks})')
    if troughs is not None and len(troughs) > 0:
        axes[ax_idx].plot(time[troughs], filtered_signal[troughs], "^",
                          color='green', markersize=4, alpha=0.7,
                          label=f'Troughs ({num_troughs})')
    axes[ax_idx].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[ax_idx].set_title(
        f"Filtered Signal — Peaks: {num_peaks}  |  Troughs: {num_troughs}  |  Ripples: {ripple_count}")
    axes[ax_idx].set_ylabel("Filtered Current")
    axes[ax_idx].legend(loc='upper right', fontsize=8)
    axes[ax_idx].grid(True, alpha=0.3)
    ax_idx += 1

    # Panel 5: Zoomed-in View
    mid_idx = len(time) // 2
    samples_in_half_sec = int((0.5 / (time[-1] - time[0])) * len(time))
    start_idx = max(0, mid_idx - samples_in_half_sec // 2)
    end_idx = min(len(time), start_idx + samples_in_half_sec)

    axes[ax_idx].plot(time[start_idx:end_idx], filtered_signal[start_idx:end_idx],
                      color='blue', linewidth=1)
    if peaks is not None and len(peaks) > 0:
        zoom_peaks = peaks[(peaks >= start_idx) & (peaks < end_idx)]
        axes[ax_idx].plot(time[zoom_peaks], filtered_signal[zoom_peaks], "v",
                          color='red', markersize=8, label='Peaks')
    if troughs is not None and len(troughs) > 0:
        zoom_troughs = troughs[(troughs >= start_idx) & (troughs < end_idx)]
        axes[ax_idx].plot(time[zoom_troughs], filtered_signal[zoom_troughs], "^",
                          color='green', markersize=8, label='Troughs')
    if adaptive_thresh is not None:
        axes[ax_idx].plot(time[start_idx:end_idx], adaptive_thresh[start_idx:end_idx],
                          color='red', linewidth=1, linestyle='--', alpha=0.5,
                          label='Local Threshold')
        axes[ax_idx].plot(time[start_idx:end_idx], -adaptive_thresh[start_idx:end_idx],
                          color='red', linewidth=1, linestyle='--', alpha=0.5)
    axes[ax_idx].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[ax_idx].set_title("Zoomed View (0.5s window) — Red ▼ = peaks, Green ▲ = troughs")
    axes[ax_idx].set_xlabel("Time (s)")
    axes[ax_idx].set_ylabel("Filtered Current")
    axes[ax_idx].legend(loc='upper right', fontsize=8)
    axes[ax_idx].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig("ripple_analysis_adaptive.png", dpi=200, bbox_inches='tight')
    plt.show()
