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


# ── Segment Steady State ─────────────────────────────────────────────
def segment_steady_state(signal, fs=4000):
    """Finds the longest steady-state motor run, trimming start/stop transients."""
    window = int(fs * 0.1)  # 100ms
    kernel = np.ones(window) / window
    energy = np.convolve(np.abs(signal), kernel, mode='same')
    
    threshold = np.max(energy) * 0.1
    is_on = energy > threshold
    
    diff = np.diff(is_on.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    if len(is_on) > 0 and is_on[0]:
        starts = np.insert(starts, 0, 0)
    if len(is_on) > 0 and is_on[-1]:
        ends = np.append(ends, len(signal) - 1)
        
    if len(starts) == 0:
        return signal  # Fallback
        
    lengths = ends - starts
    longest_idx = np.argmax(lengths)
    best_start = starts[longest_idx]
    best_end = ends[longest_idx]
    
    trim_samples = int(fs * 0.5)  # Trim 0.5s from both ends
    if best_end - best_start > 2 * trim_samples:
        return signal[best_start + trim_samples : best_end - trim_samples]
    else:
        # If too short to trim 0.5s, just take the middle half
        mid = (best_start + best_end) // 2
        half_len = (best_end - best_start) // 4
        return signal[mid - half_len : mid + half_len]

# ── FFT Frequency Analysis ───────────────────────────────────────────
def analyze_frequency(signal, fs=4000, min_ripple_freq=20.0):
    # Segment out the start-up and wind-down transients first!
    signal_core = segment_steady_state(signal, fs)
    
    if len(signal_core) < 50: # safety check
        signal_core = signal
        
    nyq = 0.5 * fs
    hp_cutoff = min_ripple_freq / nyq
    b_hp, a_hp = butter(4, hp_cutoff, btype='high')
    signal_hp = filtfilt(b_hp, a_hp, signal_core)

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
        min_distance = int(fs / dominant_freq * 0.4)
    else:
        min_distance = 4
    min_distance = max(min_distance, 2)
    print(f"  [Debug] min_distance set to: {min_distance} samples")

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
    """4-panel plot zooming in on 4 separate time chunks."""
    total_time = time[-1] - time[0]
    quarter = total_time / 4.0
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 14))
    
    for i in range(4):
        ax = axes[i]
        start_t = time[0] + i * quarter
        end_t = time[0] + (i + 1) * quarter
        
        mask = (time >= start_t) & (time <= end_t)
        
        ax.plot(time[mask], filtered_signal[mask], color='blue', linewidth=1.0, label="Filtered Signal")
        
        if envelope is not None:
            ax.plot(time[mask], envelope[mask], color='orange', linewidth=0.8,
                    label='Hilbert Envelope', alpha=0.5)
        
        if adaptive_thresh is not None:
            ax.plot(time[mask], adaptive_thresh[mask], color='red', linewidth=1.5,
                    linestyle='--', label='Local Threshold (+)', alpha=0.8)
            ax.plot(time[mask], -adaptive_thresh[mask], color='red', linewidth=1.5,
                    linestyle='--', alpha=0.8)

        # Calculate a dynamic offset (10% of the max amplitude in this window) so arrows hover
        max_amp = np.max(np.abs(filtered_signal[mask])) if np.any(mask) else 1.0
        offset = max_amp * 0.15

        if peaks is not None and len(peaks) > 0:
            zoom_peaks = peaks[(time[peaks] >= start_t) & (time[peaks] <= end_t)]
            # Plot hovering downward triangle above the peak
            ax.plot(time[zoom_peaks], filtered_signal[zoom_peaks] + offset, "v",
                    color='red', markersize=6, alpha=0.9, label='Peaks' if i==0 else "")
            
        if troughs is not None and len(troughs) > 0:
            zoom_troughs = troughs[(time[troughs] >= start_t) & (time[troughs] <= end_t)]
            # Plot hovering upward triangle below the trough
            ax.plot(time[zoom_troughs], filtered_signal[zoom_troughs] - offset, "^",
                    color='green', markersize=6, alpha=0.9, label='Troughs' if i==0 else "")
            
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlim(start_t, end_t)
        ax.set_ylabel("Current")
        ax.grid(True, alpha=0.4)
        
        if i == 0:
            ax.set_title(f"{title} | Total Ripples: {ripple_count}")
            ax.legend(loc='upper right', fontsize=10)
        if i == 3:
            ax.set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig("ripple_analysis_adaptive.png", dpi=200, bbox_inches='tight')
