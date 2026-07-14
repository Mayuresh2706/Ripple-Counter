import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
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
    search_freq = xf[min_freq_idx:max_freq_idx]

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

def count_ripples_zero_crossing(filtered_signal):
    """Counts ripples by counting zero crossings in the bandpass-filtered signal.

    After bandpass filtering, the signal oscillates symmetrically around zero.
    Each full ripple cycle crosses zero exactly twice (once going up, once going
    down). So: ripple_count = zero_crossings / 2.

    This method has no amplitude threshold, so it cannot miss weak ripples that
    a prominence-based detector would skip.
    """
    # Detect sign changes between consecutive samples
    # np.sign returns -1, 0, or 1. A sign change means the signal crossed zero.
    signs = np.sign(filtered_signal)

    # Treat exact zeros as positive to avoid ambiguity (e.g. signal lands exactly
    # on zero between two negative dips — we still want to count that crossing)
    signs[signs == 0] = 1

    # A zero crossing occurs wherever consecutive samples have different signs
    crossings = np.where(np.diff(signs) != 0)[0]

    num_crossings = len(crossings)
    ripple_count = num_crossings // 2

    return ripple_count, crossings

def plot_results(time, raw_signal, filtered_signal, fft_freqs, fft_mag,
                 crossings=None, ripple_count=0, dominant_freq=None,
                 title="Motor Current Ripple Analysis (Zero-Crossing Method)"):
    """4-panel plot: Raw, FFT, Filtered (Full) with crossings, Filtered (Zoomed)."""
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12))

    # Panel 1: Raw Signal
    ax1.plot(time, raw_signal, color='gray', linewidth=0.5)
    ax1.set_title("Raw Signal")
    ax1.set_ylabel("Current")
    ax1.grid(True, alpha=0.3)

    # Panel 2: FFT
    ax2.plot(fft_freqs, fft_mag, color='purple', linewidth=0.8)
    if dominant_freq:
        ax2.axvline(dominant_freq, color='red', linestyle='--', label=f'{dominant_freq:.1f} Hz')
    ax2.set_title("Frequency Spectrum (FFT)")
    ax2.set_xlim(0, 1500)
    ax2.set_ylabel("Magnitude")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Panel 3: Filtered Signal (Full) with zero-crossing markers
    ax3.plot(time, filtered_signal, color='blue', linewidth=0.5)
    if crossings is not None and len(crossings) > 0:
        ax3.plot(time[crossings], filtered_signal[crossings], "|",
                 color='red', markersize=6, alpha=0.5)
    ax3.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax3.set_title(f"Filtered Signal (Full) — Zero Crossings: {len(crossings) if crossings is not None else 0}  |  Ripples: {ripple_count}")
    ax3.set_ylabel("Filtered Current")
    ax3.grid(True, alpha=0.3)

    # Panel 4: Zoomed-in View (0.5 seconds from the middle of the signal)
    mid_idx = len(time) // 2
    samples_in_half_sec = int((0.5 / (time[-1] - time[0])) * len(time))
    start_idx = max(0, mid_idx - samples_in_half_sec // 2)
    end_idx = min(len(time), start_idx + samples_in_half_sec)

    ax4.plot(time[start_idx:end_idx], filtered_signal[start_idx:end_idx],
             color='blue', linewidth=1)
    if crossings is not None and len(crossings) > 0:
        zoom_crossings = crossings[(crossings >= start_idx) & (crossings < end_idx)]
        ax4.plot(time[zoom_crossings], filtered_signal[zoom_crossings], "|",
                 color='red', markersize=10, markeredgewidth=1.5)
    ax4.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax4.set_title("Zoomed View (0.5s window) — Red lines show zero crossings")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Filtered Current")
    ax4.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig("ripple_analysis_zc.png", dpi=200, bbox_inches='tight')
    plt.show()
