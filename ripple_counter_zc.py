import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt

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

def count_ripples_hybrid(filtered_signal, fs=4000, dominant_freq=None, prominence=None):
    """Counts ripples using both peaks and troughs with smart pairing.

    Counting logic:
    - If a peak is followed by a trough (or vice versa), they form a pair
      and count as 1 ripple (classic full cycle).
    - If a peak is followed by another peak (or trough by trough), each one
      is a standalone ripple (the signal crossed zero between them without
      producing a detectable extremum on the other side).

    This catches ripples that stay entirely above or below the zero line,
    which the pure zero-crossing method misses.
    """
    # Enforce strict distance to avoid counting high-frequency noise
    if dominant_freq and dominant_freq > 0:
        min_distance = int(fs / dominant_freq * 0.4)
    else:
        min_distance = 4
    min_distance = max(min_distance, 2)
    print(f"  [Debug] min_distance set to: {min_distance} samples")

    # Use MAD for robust prominence (immune to start/stop spikes)
    if prominence is None:
        median = np.median(filtered_signal)
        mad = np.median(np.abs(filtered_signal - median))
        # Lowered multiplier from 2.5 to 1.2 to avoid missing small ripples
        prominence = mad * 0.8
        prominence = max(prominence, 0.01)

    # Find peaks (local maxima)
    peaks, _ = find_peaks(filtered_signal, prominence=prominence, distance=min_distance)

    # Find troughs (local minima) by finding peaks in the negated signal
    troughs, _ = find_peaks(-filtered_signal, prominence=prominence, distance=min_distance)

    # Merge into a single list sorted by time, labeled as 'P' (peak) or 'T' (trough)
    extrema = [(idx, 'P') for idx in peaks] + [(idx, 'T') for idx in troughs]
    extrema.sort(key=lambda x: x[0])

    # Apply the smart counting rule:
    #   alternating (P-T or T-P) → pair = 1 ripple
    #   same type   (P-P or T-T) → each is a standalone ripple
    ripple_count = 0
    i = 0
    while i < len(extrema):
        if i + 1 < len(extrema) and extrema[i][1] != extrema[i + 1][1]:
            # Alternating pair: count as 1 ripple, skip both
            ripple_count += 1
            i += 2
        else:
            # Same type next, or last element: standalone ripple
            ripple_count += 1
            i += 1

    return ripple_count, peaks, troughs, extrema

def plot_results(time, raw_signal, filtered_signal, fft_freqs, fft_mag,
                 peaks=None, troughs=None, ripple_count=0, dominant_freq=None,
                 title="Motor Current Ripple Analysis (Hybrid Peak+Trough Method)"):
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
        
        if peaks is not None and len(peaks) > 0:
            zoom_peaks = peaks[(time[peaks] >= start_t) & (time[peaks] <= end_t)]
            ax.plot(time[zoom_peaks], filtered_signal[zoom_peaks], "v",
                    color='red', markersize=7, alpha=0.9, label='Peaks' if i==0 else "")
            
        if troughs is not None and len(troughs) > 0:
            zoom_troughs = troughs[(time[troughs] >= start_t) & (time[troughs] <= end_t)]
            ax.plot(time[zoom_troughs], filtered_signal[zoom_troughs], "^",
                    color='green', markersize=7, alpha=0.9, label='Troughs' if i==0 else "")
            
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
    plt.savefig("ripple_analysis_zc.png", dpi=200, bbox_inches='tight')
